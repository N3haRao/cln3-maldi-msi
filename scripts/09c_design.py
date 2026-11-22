"""Phase 09c: the design questions from the 11/2026 meeting, from 09's per-animal means.

    python scripts/09c_design.py --config config.yaml [--mode lip_neg]

09 is the main analysis: paired over the 6 slides, sexes pooled. This phase asks how much
that answer depends on the design choices, with the same per-animal means (log2), region by
region, for every single peak and for the class totals and ratios:

  paired_female     females only (slides E, F, G): moderated paired t over 3 pairs
  paired_male       males only (slides I, J, K): moderated paired t over 3 pairs
  paired_sex_adj    all 6 pairs, pair difference ~ 1 + sex: the genotype effect averaged
                    over the sexes (intercept) and the female-minus-male difference in it
                    (sex x genotype interaction). Sex itself cancels within a slide.
  unpaired          Welch t-test, 6 KO vs 6 Control animals, ignoring the slide pairing
  unpaired_sex_adj  linear model log2 ~ genotype + sex over the 12 animals (no pairing)

plus 09's paired result (all 6 pairs) side by side. FDR (BH) per analysis and region, over
the peaks of 09's untargeted family plus the hypothesis-tagged peaks; class tests over all
class/ratio features of the mode.

PCA of the animals' lipid profiles (09 feature-QC peaks, log2, z-scored), per region:
all 12 animals, after removing each slide's mean (what the paired test sees: slide, run and
sex removed), and each sex alone. For every PC, the share of its variance explained by
genotype, sex and slide (R2) says what drives the main variation.

Caveats (written into the summary too): 3 pairs per sex is very little power, so "not
significant in males" is not evidence of no effect; sex is confounded with run and with slide
orientation (females Control on Top, males KO on Top); the unpaired tests ignore the slide
pairing and so keep the slide (batch) noise that the paired test removes.

Writes results/09c_design/<mode>/{peak_tests,class_tests,counts,pca_variance,pca_scores}.csv,
figures and summaries/09c_design_<mode>.md.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sst

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cln3msi import cli, figstyle, samples, stats  # noqa: E402
from cln3msi.report import Summary, plt  # noqa: E402

ANALYSES = ["paired_all", "paired_female", "paired_male", "paired_sex_adj", "unpaired", "unpaired_sex_adj"]
NICE = {"paired_all": "Paired, both sexes (09)", "paired_female": "Paired, females", "paired_male": "Paired, males",
        "paired_sex_adj": "Paired, sex-adjusted", "unpaired": "Unpaired (Welch)",
        "unpaired_sex_adj": "Unpaired, sex-adjusted"}


def ols(y: np.ndarray, X: np.ndarray, j: int) -> tuple[float, float]:
    """Coefficient j and its two-sided p from ordinary least squares (rows with NaN y dropped)."""
    ok = np.isfinite(y)
    y, X = y[ok], X[ok]
    n, k = X.shape
    if n <= k or np.linalg.matrix_rank(X) < k:
        return np.nan, np.nan
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ beta
    s2 = res @ res / (n - k)
    cov = s2 * np.linalg.inv(X.T @ X)
    se = np.sqrt(cov[j, j])
    if not np.isfinite(se) or se == 0:
        return float(beta[j]), np.nan
    return float(beta[j]), float(2 * sst.t.sf(abs(beta[j] / se), n - k))


def design_tests(L: pd.DataFrame, info: pd.DataFrame, ctl: str, mut: str) -> pd.DataFrame:
    """L: animals x features (log2). info: animal_id -> slide, genotype, sex.
    Returns one row per feature with log2fc and p for every analysis (no FDR yet), or None
    if fewer than 4 slides have both animals."""
    info = info.loc[L.index]
    slides = sorted(info["slide"].unique())
    def one(slide, g):
        a = info.index[(info.slide == slide) & (info.genotype == g)]
        return a[0] if len(a) else None
    pairs = [(s, one(s, mut), one(s, ctl)) for s in slides]
    pairs = [(s, m, c) for s, m, c in pairs if m is not None and c is not None]
    if len(pairs) < 4:
        return None
    D = pd.DataFrame({s: L.loc[m] - L.loc[c] for s, m, c in pairs}).T          # slides x features
    sex_of = info.drop_duplicates("slide").set_index("slide")["sex"]
    psex = sex_of.loc[D.index].to_numpy()
    out = pd.DataFrame(index=L.columns)
    out["n_pairs"] = D.notna().sum().to_numpy()
    for tag, sel in (("female", psex == "Female"), ("male", psex == "Male")):
        Ds = D[sel]
        out[f"log2fc_paired_{tag}"] = Ds.mean().to_numpy()
        out[f"n_pairs_{tag}"] = Ds.notna().sum().to_numpy()
        out[f"p_paired_{tag}"] = stats.moderated_t(Ds.to_numpy())["p"] if len(Ds) >= 2 else np.nan
    # sex-adjusted paired: d ~ 1 + sex (coded +-1/2, so the intercept is the mean of the two sex means)
    sc = np.where(psex == "Female", 0.5, -0.5)
    Xp = np.column_stack([np.ones(len(sc)), sc])
    est = [ols(D[f].to_numpy(float), Xp, 0) for f in L.columns]
    inter = [ols(D[f].to_numpy(float), Xp, 1) for f in L.columns]
    out["log2fc_paired_sex_adj"] = [e[0] for e in est]
    out["p_paired_sex_adj"] = [e[1] for e in est]
    out["sex_x_genotype_log2"] = [e[0] for e in inter]          # female minus male genotype effect
    out["p_sex_x_genotype"] = [e[1] for e in inter]
    # unpaired
    g = info["genotype"].to_numpy()
    A, B = L[g == mut], L[g == ctl]
    out["log2fc_unpaired"] = (A.mean() - B.mean()).to_numpy()
    with np.errstate(all="ignore"):
        out["p_unpaired"] = sst.ttest_ind(A.to_numpy(float), B.to_numpy(float), equal_var=False,
                                          nan_policy="omit").pvalue
    Xu = np.column_stack([np.ones(len(L)), (g == mut).astype(float), np.where(info["sex"] == "Female", 0.5, -0.5)])
    u = [ols(L[f].to_numpy(float), Xu, 1) for f in L.columns]
    out["log2fc_unpaired_sex_adj"] = [e[0] for e in u]
    out["p_unpaired_sex_adj"] = [e[1] for e in u]
    return out


def add_fdr(R: pd.DataFrame, by: list[str], family_col: str | None = None) -> pd.DataFrame:
    for a in ANALYSES[1:] + ["sex_x_genotype"]:
        pc, qc = f"p_{a}", f"q_{a}"
        R[qc] = np.nan
        for _, idx in R.groupby(by).groups.items():
            idx = list(idx)
            if family_col is not None:
                idx = [i for i in idx if bool(R.at[i, family_col])]
            if idx:
                R.loc[idx, qc] = stats.bh(R.loc[idx, pc])
    return R


def pca(Z: np.ndarray, k: int = 4):
    Z = Z - Z.mean(axis=0)
    U, s, Vt = np.linalg.svd(Z, full_matrices=False)
    var = s ** 2 / max((s ** 2).sum(), 1e-12)
    k = min(k, len(s))
    return U[:, :k] * s[:k], var[:k]


def r2(y: np.ndarray, groups: np.ndarray) -> float:
    """Share of y's variance explained by a grouping (one-way ANOVA R2)."""
    tot = ((y - y.mean()) ** 2).sum()
    if tot <= 0:
        return np.nan
    between = sum(((y[groups == g].mean() - y.mean()) ** 2) * (groups == g).sum() for g in np.unique(groups))
    return float(between / tot)


def main():
    t0 = time.time()
    ctx, log, modes, _ = cli.start("09c_design", __doc__)
    C = ctx.cfg.get("design_checks") or {}
    if not C.get("enabled", True):
        log.info("design_checks.enabled is false, nothing to do")
        return
    G = ctx.cfg["project"]["genotypes"]
    ctl, mut = G["control"], G["mutant"]
    alpha = float(ctx.cfg["differential"]["fdr_alpha"])
    sheet = samples.load_sheet(ctx)
    info = sheet.set_index("animal_id")[["slide", "genotype", "sex"]]
    pca_units = C.get("pca_units", ["whole_section", "white_matter", "cortex"])
    gcol = figstyle.genotype_colors(ctx)
    mk = {"Female": "o", "Male": "s"}

    for mode in modes:
        src = ctx.output_root / "results" / "09_differential" / mode
        if not (src / "per_animal_means.csv").exists():
            log.warning(f"{mode}: no 09 outputs ({src}); run 09 first")
            continue
        S = Summary(ctx, "09c_design", mode, f"Design checks (sex, pairing, PCA), {mode}")
        out = lambda name: ctx.out("results", "09c_design", mode, name)  # noqa: E731
        PM = pd.read_csv(src / "per_animal_means.csv")
        DF = pd.read_csv(src / "differential.csv")
        if "unit_kind" in DF:          # atlas and lipid-defined regions; the 08 territories stay in 09
            PM = PM[PM.unit.isin(set(DF.loc[DF.unit_kind == "region", "unit"]))]
        keep_cols = [c for c in ("unit", "peak", "label", "lipid_class", "targeted", "in_untargeted_family",
                                 "log2fc", "p_primary", "q_all", "q_targeted", "significant") if c in DF.columns]
        DF = DF[keep_cols]
        fam = DF.assign(family=DF.get("targeted", False).fillna(False).astype(bool)
                        | DF.get("in_untargeted_family", False).fillna(False).astype(bool))

        S.h("Design")
        S.p("Same per-animal means as 09 (TIC-normalized, log2), tested five more ways per region. "
            "Females: slides E, F, G (Control on Top). Males: slides I, J, K (CLN3 KO on Top). "
            "09's paired test over all 6 pairs is the main analysis; these are sensitivity analyses.")
        S.warn("Read with care: 3 pairs per sex gives very little power, so a change that is not significant in "
               "males (or females) is not evidence that it is absent. Sex is confounded with run (females runs 1-3, "
               "males run 4) and with slide orientation, so female-versus-male differences may be technical. The "
               "unpaired tests ignore the slide pairing and therefore keep the slide-to-slide noise the paired test "
               "removes; they are shown because they were asked for, not as an alternative main analysis.", log)

        # ---- single peaks ------------------------------------------------------------------
        rows = []
        for unit, g in PM.groupby("unit"):
            W = g.pivot_table(index="animal_id", columns="peak", values="mean_norm_intensity")
            W = W[[c for c in W.columns if W[c].notna().sum() >= 4]]
            if W.empty:
                continue
            Lg = np.log2(W + stats.pseudocount(W))
            T = design_tests(Lg, info, ctl, mut)
            if T is None:
                continue
            T.insert(0, "unit", unit)
            rows.append(T.rename_axis("peak").reset_index())
        R = pd.concat(rows, ignore_index=True)
        R = R.merge(fam.rename(columns={"log2fc": "log2fc_paired_all", "p_primary": "p_paired_all",
                                        "significant": "significant_paired_all"}), on=["unit", "peak"], how="left")
        R["family"] = R["family"].fillna(False).astype(bool)
        R["q_paired_all"] = np.where(R.get("targeted", False).fillna(False).astype(bool),
                                     R.get("q_targeted"), R.get("q_all"))
        R = add_fdr(R, ["unit"], "family")
        front = ["unit", "peak", "label", "lipid_class", "family"]
        R = R[front + [c for c in R.columns if c not in front]]
        R.to_csv(out("peak_tests.csv"), index=False)

        # ---- classes and ratios ------------------------------------------------------------
        K = None
        cp = src / "class_values_per_animal.csv"
        if cp.exists():
            CV = pd.read_csv(cp)
            rows = []
            for (unit, kind), g in CV.groupby(["unit", "kind"]):
                W = g.pivot_table(index="animal_id", columns="feature", values="value")
                W = W.where(W > 0)
                W = W[[c for c in W.columns if W[c].notna().sum() >= 4]]
                if W.empty:
                    continue
                T = design_tests(np.log2(W), info, ctl, mut)
                if T is None:
                    continue
                T.insert(0, "kind", kind)
                T.insert(0, "unit", unit)
                rows.append(T.rename_axis("feature").reset_index())
            if rows:
                K = pd.concat(rows, ignore_index=True)
                ct = src / "class_tests.csv"
                if ct.exists():
                    CT = pd.read_csv(ct)
                    cols = [c for c in ("unit", "kind", "feature", "log2fc", "p_ttest", "q_bh") if c in CT.columns]
                    K = K.merge(CT[cols].rename(columns={"log2fc": "log2fc_paired_all", "p_ttest": "p_paired_all",
                                                         "q_bh": "q_paired_all"}),
                                on=[c for c in ("unit", "kind", "feature") if c in cols], how="left")
                K["all"] = True
                K = add_fdr(K, ["all"]).drop(columns="all")
                K.to_csv(out("class_tests.csv"), index=False)

        # ---- counts ------------------------------------------------------------------------
        cnt = []
        for unit, g in R[R["family"]].groupby("unit"):
            rec = {"unit": unit, "peaks_tested": len(g)}
            for a in ANALYSES:
                q = g.get(f"q_{a}")
                rec[f"{a}_q<{alpha:g}"] = int((q < alpha).sum()) if q is not None else np.nan
            rec[f"sex_x_genotype_q<{alpha:g}"] = int((g["q_sex_x_genotype"] < alpha).sum())
            cnt.append(rec)
        CNT = pd.DataFrame(cnt)
        CNT.to_csv(out("counts.csv"), index=False)
        S.h("How many peaks reach FDR < " + f"{alpha:g}, per analysis and region")
        S.table(CNT)

        # ---- agreement figures ------------------------------------------------------------
        main_units = [u for u in pca_units if u in set(R.unit)] or sorted(R.unit.unique())[:3]
        fig, axes = plt.subplots(2, len(main_units), figsize=(3.0 * len(main_units), 5.9), squeeze=False,
                                 gridspec_kw={"hspace": 0.55, "wspace": 0.35})
        agree = []
        for j, u in enumerate(main_units):
            g = R[(R.unit == u) & R["family"]]
            ax = axes[0, j]
            x, y = g["log2fc_paired_female"], g["log2fc_paired_male"]
            sig = (g["q_paired_all"] < alpha).to_numpy()
            ax.scatter(x[~sig], y[~sig], s=6, color="0.7", edgecolors="none", rasterized=True, label="Other peaks")
            ax.scatter(x[sig], y[sig], s=9, color=figstyle.MUTANT, edgecolors="none",
                       label=f"FDR < {alpha:g} in 09 (paired, both sexes)")
            lim = np.nanmax(np.abs(np.r_[x, y])) if len(g) else 1
            ax.plot([-lim, lim], [-lim, lim], color="k", lw=0.5, ls="--")
            ax.axhline(0, color="0.5", lw=0.4); ax.axvline(0, color="0.5", lw=0.4)  # noqa: E702
            rho = sst.spearmanr(x, y, nan_policy="omit")[0] if len(g) > 3 else np.nan
            ax.set_title(f"{figstyle.region_name(u)}\nSpearman r = {rho:.2f}")
            ax.set_xlabel("log2 FC, females"); ax.set_ylabel("log2 FC, males")  # noqa: E702
            ax = axes[1, j]
            pa = -np.log10(np.clip(g["p_paired_all"].to_numpy(float), 1e-300, 1))
            pu = -np.log10(np.clip(g["p_unpaired"].to_numpy(float), 1e-300, 1))
            ax.scatter(pa, pu, s=6, color=figstyle.CONTROL, edgecolors="none", rasterized=True)
            m = np.nanmax(np.r_[pa, pu, 1]) if len(g) else 1
            ax.plot([0, m], [0, m], color="k", lw=0.5, ls="--")
            ax.set_xlabel("-log10 p, paired (09)"); ax.set_ylabel("-log10 p, unpaired")  # noqa: E702
            agree.append({"unit": u, "spearman_female_vs_male_log2fc": rho,
                          "spearman_paired_vs_unpaired_log10p": sst.spearmanr(pa, pu, nan_policy="omit")[0]
                          if len(g) > 3 else np.nan})
        axes[0, 0].legend(loc="lower right", fontsize=6)
        figstyle.panel_labels(axes.ravel(), x=-0.25)
        figstyle.title(fig, f"{figstyle.mode_name(mode)}: females vs males, paired vs unpaired")
        S.fig(fig, "sex_and_pairing.png", caption=(
            "Top row: genotype effect (log2 fold change, KO / Control, mean of within-slide differences) in females "
            "(slides E, F, G) against males (slides I, J, K), one point per peak (09 untargeted family plus "
            "hypothesis peaks); vermillion, peaks significant in the main paired analysis. Points on the dashed line "
            "change equally in both sexes. Bottom row: evidence (-log10 p) from the main paired test against the "
            "unpaired Welch test on the same per-animal means; points below the line gain from pairing by slide. "
            "Sex is confounded with run and slide orientation; 3 pairs per sex."))
        S.h("Agreement")
        S.table(pd.DataFrame(agree).round(3))

        # ---- PCA ---------------------------------------------------------------------------
        S.h("PCA of the animals' lipid profiles")
        qc = set(DF.loc[DF.get("in_untargeted_family", False).fillna(False).astype(bool), "peak"]) \
            if "in_untargeted_family" in DF else set()
        var_rows, score_rows = [], []
        units = [u for u in pca_units if u in set(PM.unit)]
        if units:
            fig, axes = plt.subplots(len(units), 4, figsize=(11.5, 2.75 * len(units) + 0.5), squeeze=False,
                                     gridspec_kw={"hspace": 0.6, "wspace": 0.38})
            for i, u in enumerate(units):
                W = PM[PM.unit == u].pivot_table(index="animal_id", columns="peak", values="mean_norm_intensity")
                if qc:
                    W = W[[c for c in W.columns if c in qc]]
                W = W.dropna(axis=1)
                if W.shape[1] < 3 or W.shape[0] < 4:
                    continue
                Lg = np.log2(W + stats.pseudocount(W))
                inf = info.loc[Lg.index]
                cen = Lg - Lg.groupby(inf["slide"]).transform("mean")
                views = [("All animals", Lg, inf), ("Slide mean removed", cen, inf),
                         ("Females only", Lg[inf.sex == "Female"], inf[inf.sex == "Female"]),
                         ("Males only", Lg[inf.sex == "Male"], inf[inf.sex == "Male"])]
                for j, (vname, M, vi) in enumerate(views):
                    ax = axes[i, j]
                    if len(M) < 3:
                        ax.set_visible(False)
                        continue
                    sd = M.std(ddof=1).replace(0, np.nan)
                    Z = ((M - M.mean()) / sd).dropna(axis=1).to_numpy()
                    sc, ve = pca(Z)
                    for k in range(sc.shape[1]):
                        rec = {"unit": u, "view": vname, "pc": k + 1, "variance_explained": ve[k],
                               "r2_genotype": r2(sc[:, k], vi["genotype"].to_numpy())}
                        if vi["sex"].nunique() > 1:
                            rec["r2_sex"] = r2(sc[:, k], vi["sex"].to_numpy())
                        if vname != "Slide mean removed":
                            rec["r2_slide"] = r2(sc[:, k], vi["slide"].to_numpy())
                        var_rows.append(rec)
                    for a, (aid, r) in enumerate(vi.iterrows()):
                        score_rows.append({"unit": u, "view": vname, "animal_id": aid, **r.to_dict(),
                                           **{f"PC{k + 1}": sc[a, k] for k in range(min(3, sc.shape[1]))}})
                        ax.scatter(sc[a, 0], sc[a, 1], color=gcol.get(r["genotype"], figstyle.GREY),
                                   marker=mk.get(r["sex"], "o"), s=30, edgecolors="0.2", linewidths=0.4, zorder=3)
                        ax.annotate(r["slide"], (sc[a, 0], sc[a, 1]), fontsize=5.5, xytext=(3, 2),
                                    textcoords="offset points")
                    ax.set_xlabel(f"PC1 ({100 * ve[0]:.0f}%)")
                    ax.set_ylabel(f"PC2 ({100 * ve[1]:.0f}%)")
                    ax.set_title(f"{figstyle.region_name(u)}: {vname.lower()}" if j == 0 else vname.capitalize())
            from matplotlib.lines import Line2D
            handles = [Line2D([], [], marker="o", ls="", color=gcol[ctl], label="Control"),
                       Line2D([], [], marker="o", ls="", color=gcol[mut], label=ctx.cfg["project"]["mutant_label"]),
                       Line2D([], [], marker="o", ls="", color="0.6", markeredgecolor="0.2", label="Female"),
                       Line2D([], [], marker="s", ls="", color="0.6", markeredgecolor="0.2", label="Male")]
            fig.legend(handles=handles, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.02))
            figstyle.title(fig, f"{figstyle.mode_name(mode)}: PCA of per-animal lipid profiles")
            S.fig(fig, "pca.png", caption=(
                "Principal component analysis of the animals' profiles (per-animal mean of each feature-QC peak, "
                "log2, z-scored) per region. Columns: all 12 animals; after subtracting each slide's mean (removes "
                "slide, run and sex, leaving what the paired test compares); females only; males only. Colour, "
                "genotype; shape, sex; label, slide. The table pca_variance.csv gives, for each PC, the share of its "
                "variance explained by genotype, sex and slide."))
            V = pd.DataFrame(var_rows)
            V.to_csv(out("pca_variance.csv"), index=False)
            pd.DataFrame(score_rows).to_csv(out("pca_scores.csv"), index=False)
            if len(V):
                S.p("Share of each principal component's variance explained by genotype, sex and slide (R2, "
                    "one-way). Sex and slide are nested (each slide is one sex), so R2 for slide is always at least "
                    "R2 for sex.")
                S.table(V[V.pc <= 2].round(3))
                top = V[(V.view == "All animals") & (V.pc == 1)]
                for _, r in top.iterrows():
                    drv = max((("genotype", r.get("r2_genotype", 0)), ("sex", r.get("r2_sex", 0)),
                               ("slide", r.get("r2_slide", 0))), key=lambda t: (t[1] if np.isfinite(t[1]) else -1))
                    S.p(f"{figstyle.region_name(r.unit)}, PC1 ({100 * r.variance_explained:.0f}% of variance): "
                        f"best explained by {drv[0]} (R2 {drv[1]:.2f}; genotype {r.get('r2_genotype', np.nan):.2f}, "
                        f"sex {r.get('r2_sex', np.nan):.2f}, slide {r.get('r2_slide', np.nan):.2f}).")

        # ---- class summary -----------------------------------------------------------------
        if K is not None:
            S.h("Class totals and ratios, every analysis (whole section and white matter)")
            show = K[K.unit.isin(["whole_section", "white_matter"])].copy()
            if "p_paired_all" in show:
                show = show.sort_values("p_paired_all")
            cols = ["unit", "kind", "feature", "log2fc_paired_all", "q_paired_all", "log2fc_paired_female",
                    "p_paired_female", "log2fc_paired_male", "p_paired_male", "p_sex_x_genotype",
                    "log2fc_unpaired", "p_unpaired", "p_unpaired_sex_adj"]
            S.table(show[[c for c in cols if c in show]].head(25).round(4))
        S.h("Files")
        S.p("peak_tests.csv and class_tests.csv: every feature and region with log2fc_*, p_* and q_* for each "
            "analysis (paired_all = 09's result), plus sex_x_genotype_log2 (female minus male genotype effect) and "
            "its p/q. counts.csv, pca_variance.csv, pca_scores.csv.")
        log.info(f"summary: {S.write()}")
    cli.finish(log, t0)


if __name__ == "__main__":
    main()
