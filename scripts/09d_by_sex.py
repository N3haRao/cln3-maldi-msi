"""Phase 09d: females and males analysed separately, paired by slide (decided 11/23/2026).

    python scripts/09d_by_sex.py --config config.yaml

Why: the three female pairs agree closely, while the three male pairs do not, and the male
labels from the sectioning records have been changed once already. So the females get a
complete analysis of their own, and the males are analysed under each candidate labelling,
with a molecular-signature check of which section on each male slide looks like a knockout.
The pairing by slide is kept throughout (no unpaired tests).

What it does, from 09's per-animal means (nothing is recomputed from pixels):

  lipid set    the unique quality-controlled lipids of the two lipid modes: 09 feature-QC
               ions (qc_pass) named in shorthand (class carbons:double bonds), one ion per
               lipid (the brighter ion when a lipid passes in both modes or with two adducts).
               These are the "161 lipids" of the main analysis.
  females      paired test over slides E, F, G per region (by_sex.regions): moderated t
               (empirical Bayes over the lipid set), BH within each region. Volcano plots,
               significant-lipid list, class totals and ratios (all three modes).
  males        the same, for each labelling in by_sex.male_labelings (slides listed there
               have their two genotypes swapped relative to the sample sheet), e.g.
               corrected = sample sheet as it is (I, J, K swapped by the records) and
               flipJ = only J swapped (I and K back as originally recorded).
  quadrants    female log2FC (x) against male log2FC (y) per lipid and region, coloured by
               where it is significant: both same direction, both opposite, female only,
               male only, neither. Counted at q < alpha and, as a looser view, at p < alpha.
  male labels  PCA of the female Controls with all six male sections (the 161 lipids,
               log2, z-scored), per region, every point labelled slide_animal (K_418), and
               the same with the female knockouts added. Plus a signature score: for each
               male slide, the Spearman correlation between its Top-minus-Bottom profile
               and the female knockout-minus-Control profile (positive: Top looks like the
               knockout), which is free of run and slide offsets.
  areas        females: each region's share of the section's pixels, knockout vs Control on
               each slide, and the whole-section test redone with each knockout section's
               regions put back at the Control's proportions (by_sex.composition_parts, which
               must not overlap). Tells whether a difference in how the sections were cut
               could explain the whole-section result.
  annotation   for the lipid set: counts per confidence tier (Lipid Brain Atlas scoring),
               per annotation source (class rules checked against LIPID MAPS, LIPID MAPS or
               SwissLipids second pass, curated list) and per Lipid Brain Atlas evidence.

Writes results/09d_by_sex/ (CSV tables and by_sex_results.xlsx), figures/09d_by_sex/,
summaries/09d_by_sex.md.
"""
from __future__ import annotations

import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sst

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cln3msi import cli, figstyle, samples, stats  # noqa: E402
from cln3msi.report import Summary, plt  # noqa: E402

SHORTHAND = re.compile(r"^[A-Za-z0-9/\-]+ (O-|P-)?\d+:\d+")
MAG = "#B0127A"
CAT_COL = {"both, same direction": MAG, "both, opposite": "#000000", "female only": figstyle.MUTANT,
           "male only": figstyle.CONTROL, "neither": "#C8C8C8"}
TIER_TEXT = {
    "high": "score above 4: in practice seen by LC-MS/MS in mouse brain (8 points), or by both LC-MS "
            "methods plus a published brain data set",
    "medium": "score 2 to 4: seen by LC-MS without MS/MS in mouse brain, possibly with published support",
    "low": "score 1: only in one published brain data set (Fitzner LC-MS or Vandenbosch MALDI) or in "
           "the atlas's own MALDI list",
    "database only": "score 0: the formula exists in LIPID MAPS, and the atlas tables cover this lipid "
                     "class, but none of them reports this species in brain",
    "not covered": "no atlas table covers this lipid class (for example gangliosides), so the atlas "
                   "cannot say either way",
}
SOURCE_TEXT = {
    "class_rules": "Composition generated from class rules (head group plus total carbons, double bonds, "
                   "oxygens), kept only because that exact formula exists in LIPID MAPS",
    "fixed": "Curated list of specific brain lipids in the configuration (exact formula)",
    "LIPID MAPS": "Second pass: matched directly to a LIPID MAPS structure (formula and name from LIPID MAPS)",
    "SwissLipids": "Second pass: matched directly to a SwissLipids species",
}
EVID_TEXT = {"msms": "LC-MS/MS in mouse brain (atlas tandem MS table or quantitative HILIC-MS/MS)",
             "lcms": "LC-MS without MS/MS in mouse brain (atlas HILIC and reverse-phase lists)",
             "fitzner": "Published brain LC-MS (Fitzner et al.)",
             "vandenbosch": "Published brain MALDI imaging (Vandenbosch et al.)",
             "maldi": "The atlas's own MALDI imaging peak list (positive mode)"}


def short_id(aid: str, slide: str) -> str:
    """cKO-418 on slide K -> K_418."""
    m = re.search(r"(\d+)$", str(aid))
    return f"{slide}_{m.group(1) if m else aid}"


def lipid_set(ctx, modes, log):
    """One row per unique lipid: name, mode, peak, label, class, tier, evidence, source."""
    rows = []
    for m in modes:
        src = ctx.output_root / "results" / "09_differential" / m
        fq = pd.read_csv(src / "feature_qc.csv", index_col=0)
        fq = fq[fq["qc_pass"].astype(bool)]
        ap = ctx.out("processed", m, "annotations.csv", mkdir=False)
        A = pd.read_csv(ap).set_index("peak_id") if ap.exists() else pd.DataFrame()
        pm = pd.read_csv(src / "per_animal_means.csv")
        bright = pm[pm.unit == "whole_section"].groupby("peak")["mean_norm_intensity"].mean()
        for pk, r in fq.iterrows():
            a = A.loc[pk] if pk in A.index else pd.Series(dtype=object)
            name = str(a.get("annotation", "") or "") or re.sub(r"\s*[\[\*].*$", "", str(r["label"])).strip()
            if not SHORTHAND.match(name):
                continue
            rows.append({"lipid": name, "mode": m, "peak": pk, "label": r["label"],
                         "lipid_class": a.get("lipid_class", name.split()[0]),
                         "confidence_tier": a.get("confidence_tier", r.get("confidence_tier", "")),
                         "evidence_score": a.get("evidence_score", np.nan),
                         "evidence_sources": a.get("evidence_sources", ""),
                         "annotation_source": a.get("annotation_source", ""),
                         "ambiguous": a.get("ambiguous", False), "brightness": float(bright.get(pk, np.nan))})
    L = pd.DataFrame(rows)
    if L.empty:
        return L
    L = L.sort_values("brightness", ascending=False).drop_duplicates("lipid").sort_values(["lipid_class", "lipid"])
    log.info(f"lipid set: {len(L)} unique lipids ({(L['mode'] == 'lip_neg').sum()} from negative, "
             f"{(L['mode'] == 'lip_pos').sum()} from positive mode)")
    return L.reset_index(drop=True)


def wide_log2(ctx, L, unit):
    """animals x lipids, log2 per-animal mean (pseudocount only where a mean is 0)."""
    parts = []
    for m, g in L.groupby("mode"):
        pm = pd.read_csv(ctx.output_root / "results" / "09_differential" / m / "per_animal_means.csv")
        pm = pm[(pm.unit == unit) & pm.peak.isin(g.peak)]
        W = pm.pivot_table(index="animal_id", columns="peak", values="mean_norm_intensity")
        W = W.rename(columns=dict(zip(g.peak, g.lipid)))
        parts.append(W)
    W = pd.concat(parts, axis=1)
    return np.log2(W + stats.pseudocount(W))


def pair_diffs(Lg, info, slides, ctl, mut, swap):
    """slides x lipids of log2(KO) - log2(Control); slides in `swap` have genotypes exchanged."""
    out = {}
    for s in slides:
        sub = info[info.slide == s]
        g = sub["genotype"].copy()
        if s in swap:
            g = g.map({ctl: mut, mut: ctl})
        k, c = sub.index[g == mut], sub.index[g == ctl]
        if len(k) and len(c) and k[0] in Lg.index and c[0] in Lg.index:
            out[s] = Lg.loc[k[0]] - Lg.loc[c[0]]
    return pd.DataFrame(out).T


def test_pairs(D: pd.DataFrame, alpha: float) -> pd.DataFrame:
    """Per lipid: log2fc, % change, 95% CI (paired t), moderated-t p, BH q, pairs up."""
    n = D.notna().sum()
    mean = D.mean()
    sd = D.std(ddof=1)
    tcrit = sst.t.ppf(0.975, np.maximum(n - 1, 1))
    mod = stats.moderated_t(D.to_numpy(float))
    R = pd.DataFrame({"n_pairs": n, "log2fc": mean, "ci_low": mean - tcrit * sd / np.sqrt(n), "ci_high": mean + tcrit * sd / np.sqrt(n),
                      "p": mod["p"], "n_pairs_up": (D > 0).sum()})
    with np.errstate(all="ignore"):
        R["p_ttest"] = sst.ttest_1samp(D.to_numpy(float), 0, nan_policy="omit").pvalue
    R["q"] = stats.bh(R["p"])
    R["pct_change"] = 100 * (2 ** R["log2fc"] - 1)
    R["significant"] = R["q"] < alpha
    for s in D.index:
        R[f"pair_{s}"] = D.loc[s]
    return R.rename_axis("lipid").reset_index()


def composition_check(ctx, L, info, slides, ctl, mut, parts, min_px, alpha):
    """Region areas and a region-adjusted whole-section test for the given (female) pairs.

    The regions in `parts` must not overlap (atlas labels: one per pixel); the rest of the
    section is "other". The whole-section mean of a lipid is then exactly the area-weighted
    mean of the region means, so a knockout-vs-Control difference splits into two parts:
      adjusted          log2 sum(s_Ctl * m_KO) - log2 sum(s_Ctl * m_Ctl)
                        (the knockout's region means at the Control's region proportions)
      composition only  log2 sum(s_KO * m_Ctl) - log2 sum(s_Ctl * m_Ctl)
                        (what the area difference alone would do)
    s = each region's share of the section's pixels, m = per-animal region means (09).
    Returns (area table, per-pair area log2 ratios, lipid table, adjusted pair differences)."""
    units = ["whole_section"] + list(parts)
    means, npx = {}, {}
    for m, g in L.groupby("mode"):
        pm = pd.read_csv(ctx.output_root / "results" / "09_differential" / m / "per_animal_means.csv")
        pm = pm[pm.unit.isin(units)]
        npx[m] = pm.drop_duplicates(["unit", "animal_id"]).pivot(index="animal_id", columns="unit", values="n_pixels")
        sub = pm[pm.peak.isin(g.peak)]
        for u in units:
            W = sub[sub.unit == u].pivot_table(index="animal_id", columns="peak", values="mean_norm_intensity")
            means.setdefault(u, []).append(W.rename(columns=dict(zip(g.peak, g.lipid))))
    means = {u: pd.concat(v, axis=1) for u, v in means.items()}
    ref_mode = "lip_neg" if "lip_neg" in npx else sorted(npx)[0]
    N = npx[ref_mode].reindex(columns=units)

    area_rows, ratio_rows, obs_d, adj_d, comp_d = [], [], {}, {}, {}
    for s in slides:
        sub = info[info.slide == s]
        k = sub.index[sub.genotype == mut]
        c = sub.index[sub.genotype == ctl]
        if not (len(k) and len(c)) or k[0] not in N.index or c[0] not in N.index:
            continue
        k, c = k[0], c[0]
        rr = {"slide": s}
        for a, gname in ((c, ctl), (k, mut)):
            n = N.loc[a].fillna(0)
            whole = n["whole_section"]
            other = whole - n[list(parts)].sum()
            row = {"slide": s, "section": info.loc[a, "short"], "genotype": gname,
                   "position": info.loc[a, "position_on_slide"], "whole_section_pixels": int(whole)}
            for u in parts:
                row[f"{u}_pct"] = 100 * n[u] / whole if whole else np.nan
            row["other_pct"] = 100 * other / whole if whole else np.nan
            area_rows.append(row)
        nk, nc = N.loc[k].fillna(0), N.loc[c].fillna(0)
        rr["whole_section"] = np.log2(nk["whole_section"] / nc["whole_section"])
        for u in parts:
            sk, sc = nk[u] / nk["whole_section"], nc[u] / nc["whole_section"]
            rr[u] = np.log2(sk / sc) if sk > 0 and sc > 0 else np.nan
        ratio_rows.append(rr)

        # region shares and means per animal, per lipid (each lipid with its own mode's pixel counts)
        def parts_of(a):
            S_, M_ = {}, {}
            for m, g in L.groupby("mode"):
                n = npx[m].reindex(columns=units).loc[a] if a in npx[m].index else pd.Series(np.nan, index=units)
                whole = n["whole_section"]
                mw = means["whole_section"].loc[a, g.lipid] if a in means["whole_section"].index else None
                if mw is None or not whole > 0:
                    continue
                tot = mw * whole
                used = 0.0
                for u in parts:
                    if a in means[u].index and n.get(u, 0) > 0 and means[u].loc[a, g.lipid].notna().all():
                        S_.setdefault(u, {}).update(dict.fromkeys(g.lipid, n[u] / whole))
                        M_.setdefault(u, {}).update(means[u].loc[a, g.lipid].to_dict())
                        tot = tot - means[u].loc[a, g.lipid] * n[u]
                        used += n[u]
                rest = whole - used
                if rest < -0.5:
                    raise ValueError(f"regions {parts} overlap: they cover more pixels than the section of {a}")
                if rest >= min_px:
                    S_.setdefault("other", {}).update(dict.fromkeys(g.lipid, rest / whole))
                    M_.setdefault("other", {}).update((tot / rest).clip(lower=0).to_dict())
                else:   # nothing left outside the regions: fold the few remaining pixels into the shares
                    for u in list(S_):
                        S_[u].update({l: S_[u][l] * whole / max(used, 1) for l in g.lipid if l in S_[u]})
            return (pd.DataFrame(S_).reindex(columns=list(parts) + ["other"]).fillna(0),
                    pd.DataFrame(M_).reindex(columns=list(parts) + ["other"]).fillna(0))
        sC, mC = parts_of(c)
        sK, mK = parts_of(k)
        lip = sC.index.intersection(sK.index)
        if not len(lip):
            continue
        sC, mC, sK, mK = (x.loc[lip] for x in (sC, mC, sK, mK))
        eps = 1e-12
        base = (sC * mC).sum(axis=1).clip(lower=eps)
        obs_d[s] = np.log2((sK * mK).sum(axis=1).clip(lower=eps)) - np.log2(base)
        adj_d[s] = np.log2((sC * mK).sum(axis=1).clip(lower=eps)) - np.log2(base)
        comp_d[s] = np.log2((sK * mC).sum(axis=1).clip(lower=eps)) - np.log2(base)
    A = pd.DataFrame(area_rows)
    RT = pd.DataFrame(ratio_rows)
    if not adj_d:
        return A, RT, pd.DataFrame(), pd.DataFrame()
    Dobs, Dadj, Dcomp = (pd.DataFrame(d).T for d in (obs_d, adj_d, comp_d))
    Ro, Ra = test_pairs(Dobs, alpha), test_pairs(Dadj, alpha)
    T = Ro[["lipid", "log2fc", "p", "q", "significant"]].rename(columns={
        "log2fc": "log2fc_observed", "p": "p_observed", "q": "q_observed", "significant": "significant_observed"})
    T = T.merge(Ra[["lipid", "log2fc", "p", "q", "significant", "n_pairs_up"]].rename(columns={
        "log2fc": "log2fc_adjusted", "p": "p_adjusted", "q": "q_adjusted", "significant": "significant_adjusted",
        "n_pairs_up": "n_pairs_up_adjusted"}), on="lipid")
    T["log2fc_composition_only"] = T["lipid"].map(Dcomp.mean())
    for s in Dadj.index:
        T[f"adjusted_pair_{s}"] = T["lipid"].map(Dadj.loc[s])
    return A, RT, T, Dadj


def category(fq, mq, fx, mx, alpha):
    fs, ms = fq < alpha, mq < alpha
    return np.where(fs & ms & (np.sign(fx) == np.sign(mx)), "both, same direction",
                    np.where(fs & ms, "both, opposite", np.where(fs, "female only", np.where(ms, "male only", "neither"))))


def volcano_grid(results, regions, title, fname, S, caption, alpha, label_n=8):
    n = len(regions)
    ncol = 3 if n > 4 else n
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.3 * ncol, 3.9 * nrow), squeeze=False,
                             gridspec_kw={"hspace": 0.45, "wspace": 0.3})
    for ax, u in zip(axes.ravel(), regions):
        R = results[results.unit == u]
        y = -np.log10(R["p"].clip(lower=1e-300))
        sig = R["significant"].to_numpy()
        col = np.where(~sig, "#C8C8C8", np.where(R["log2fc"] > 0, figstyle.MUTANT, figstyle.CONTROL))
        ax.scatter(R["log2fc"], y, s=14, c=col, edgecolors="none")
        ax.axvline(0, color="#999999", lw=0.5)
        top = R[sig].nsmallest(label_n, "p")
        for _, r in top.iterrows():
            ax.annotate(r["lipid"], (r["log2fc"], -np.log10(max(r["p"], 1e-300))), fontsize=6, xytext=(3, 2),
                        textcoords="offset points")
        ax.set_title(f"{figstyle.region_name(u)}  ({int(sig.sum())} at q < {alpha:g})")
        ax.set_xlabel("log2 fold change (KO / Control)")
        ax.set_ylabel("−log10 p")
    for ax in axes.ravel()[n:]:
        ax.set_visible(False)
    figstyle.title(fig, title)
    S.fig(fig, fname, caption=caption)


def main():
    t0 = time.time()
    ctx, log, _, _ = cli.start("09d_by_sex", __doc__, mode=False)
    C = ctx.cfg.get("by_sex") or {}
    if not C.get("enabled", True):
        log.info("by_sex.enabled is false, nothing to do")
        return
    G = ctx.cfg["project"]["genotypes"]
    ctl, mut = G["control"], G["mutant"]
    alpha = float(C.get("alpha", ctx.cfg["differential"]["fdr_alpha"]))
    regions = C.get("regions", ["whole_section", "white_matter", "corpus_callosum", "cortex", "hippocampus", "thalamus"])
    lip_modes = [m for m in C.get("lipid_modes", ["lip_neg", "lip_pos"])
                 if (ctx.output_root / "results" / "09_differential" / m / "per_animal_means.csv").exists()]
    class_modes = [m for m in C.get("class_modes", ["lip_neg", "lip_pos", "met_neg"])
                   if (ctx.output_root / "results" / "09_differential" / m / "class_values_per_animal.csv").exists()]
    labelings = C.get("male_labelings") or {"corrected": []}
    out = lambda name: ctx.out("results", "09d_by_sex", name)  # noqa: E731
    S = Summary(ctx, "09d_by_sex", None, "Females and males separately (paired), male labels, annotation")

    sheet = samples.load_sheet(ctx)
    info = sheet.set_index("animal_id")[["slide", "genotype", "sex", "position_on_slide"]]
    info["short"] = [short_id(a, s) for a, s in zip(info.index, info.slide)]
    fem = sorted(info.loc[info.sex == "Female", "slide"].unique())
    mal = sorted(info.loc[info.sex == "Male", "slide"].unique())

    L = lipid_set(ctx, lip_modes, log)
    if L.empty:
        S.warn("No quality-controlled lipids found (run 09 for the lipid modes first).", log)
        S.write()
        return
    L.to_csv(out("lipid_set.csv"), index=False)
    regions = [u for u in regions if u in set(pd.read_csv(ctx.output_root / "results" / "09_differential" / lip_modes[0] / "per_animal_means.csv", usecols=["unit"]).unit)]

    S.h("Design")
    S.p(f"Paired by slide throughout. Females: slides {', '.join(fem)} (3 pairs). Males: slides {', '.join(mal)}, "
        "under each labelling: " + "; ".join(f"**{k}** = sample sheet with {', '.join(v) if v else 'no'} slide(s) swapped"
                                              for k, v in labelings.items()) + ". "
        f"Lipid set: {len(L)} unique quality-controlled lipids ({(L['mode'] == 'lip_neg').sum()} measured in negative, "
        f"{(L['mode'] == 'lip_pos').sum()} in positive mode; one ion per lipid). Regions: "
        + ", ".join(figstyle.region_name(u) for u in regions) + f". Single lipids: moderated t, BH within each region; "
        f"significant = q < {alpha:g}.")
    S.warn("Three pairs per sex: the confidence intervals are wide, and a lipid that is not significant is not shown "
           "to be unchanged. Sex is confounded with run and slide orientation.", log)

    # ---- annotation tables ----------------------------------------------------------------
    S.h("The lipid set: confidence and annotation source")
    tiers = ["high", "medium", "low", "database only", "not covered"]
    T1 = (L["confidence_tier"].fillna("").replace("", "unknown").value_counts().reindex(tiers + ["unknown"], fill_value=0)
          .rename("lipids").to_frame())
    T1 = T1[T1.lipids > 0]
    T1["share"] = (100 * T1.lipids / len(L)).round(0).astype(int).astype(str) + "%"
    T1["meaning"] = [TIER_TEXT.get(t, "") for t in T1.index]
    T1.rename_axis("confidence tier").to_csv(out("confidence_tiers.csv"))
    S.p("Each name is scored the way the Lipid Brain Atlas (Nature 2026) scored its own: LC-MS/MS in mouse brain "
        "8 points, LC-MS without MS/MS 2, each published brain data set 1, the atlas's own MALDI list 1; LIPID MAPS "
        "alone 0. The tier follows the total.")
    S.table(T1.rename_axis("confidence tier").reset_index())
    T2 = L["annotation_source"].fillna("").replace("", "unknown").value_counts().rename("lipids").to_frame()
    T2["share"] = (100 * T2.lipids / len(L)).round(0).astype(int).astype(str) + "%"
    T2["meaning"] = [SOURCE_TEXT.get(s, "") for s in T2.index]
    T2.rename_axis("annotation source").to_csv(out("annotation_sources.csv"))
    S.table(T2.rename_axis("annotation source").reset_index())
    ev = L["evidence_sources"].fillna("").astype(str)
    T3 = pd.DataFrame([{"Lipid Brain Atlas evidence": k, "lipids": int(ev.str.contains(rf"\b{k}\b").sum()),
                        "meaning": v} for k, v in EVID_TEXT.items()]
                      + [{"Lipid Brain Atlas evidence": "any of these", "lipids": int((ev != "").sum()), "meaning": ""},
                         {"Lipid Brain Atlas evidence": "none", "lipids": int((ev == "").sum()),
                          "meaning": "named from LIPID MAPS / SwissLipids only"}])
    T3["share"] = (100 * T3.lipids / len(L)).round(0).astype(int).astype(str) + "%"
    T3.to_csv(out("atlas_evidence.csv"), index=False)
    S.p("A lipid can have several kinds of evidence, so the first rows overlap.")
    S.table(T3)

    # ---- per-region tests ----------------------------------------------------------------
    allres, Lg_by_unit = [], {}
    for u in regions:
        Lg = wide_log2(ctx, L, u)
        Lg_by_unit[u] = Lg
        Df = pair_diffs(Lg, info, fem, ctl, mut, [])
        R = test_pairs(Df, alpha)
        R.insert(0, "analysis", "female")
        R.insert(0, "unit", u)
        allres.append(R)
        for lab, swap in labelings.items():
            Dm = pair_diffs(Lg, info, mal, ctl, mut, list(swap))
            Rm = test_pairs(Dm, alpha)
            Rm.insert(0, "analysis", f"male_{lab}")
            Rm.insert(0, "unit", u)
            allres.append(Rm)
    RES = pd.concat(allres, ignore_index=True).merge(L[["lipid", "lipid_class", "mode", "label", "confidence_tier"]], on="lipid", how="left")
    RES.to_csv(out("lipid_tests.csv"), index=False)
    F = RES[RES.analysis == "female"]
    sigF = F[F.significant].sort_values(["unit", "p"])
    cols = ["unit", "lipid", "lipid_class", "mode", "pct_change", "ci_low", "ci_high", "p", "q", "n_pairs_up", "n_pairs", "confidence_tier"]
    sigF[cols].to_csv(out("female_significant.csv"), index=False)

    S.h("Females: significant lipids per region")
    cnt = F.groupby("unit").agg(tested=("lipid", "size"), significant=("significant", "sum"),
                                up=("log2fc", lambda x: int(((x > 0) & F.loc[x.index, "significant"]).sum())))
    cnt["down"] = cnt["significant"] - cnt["up"]
    cnt = cnt.reindex(regions)
    S.table(cnt.reset_index())
    show = sigF.copy()
    show["change"] = show["pct_change"].map(lambda v: f"{v:+.0f}%")
    S.table(show[["unit", "lipid", "lipid_class", "change", "p", "q", "n_pairs_up", "confidence_tier"]].round(4), max_rows=80)
    # ---- region areas: does a difference in section make-up explain the whole-section result? ----
    parts = [u for u in C.get("composition_parts", ["white_matter", "cortex", "hippocampus", "thalamus"]) if u in regions]
    comp, A, RT = None, pd.DataFrame(), pd.DataFrame()
    if "whole_section" in regions and parts:
        S.h("Females: region areas and the region-adjusted whole section")
        try:
            A, RT, comp, _ = composition_check(ctx, L, info, fem, ctl, mut, parts,
                                               int(ctx.cfg["differential"]["min_pixels_per_unit"]), alpha)
        except ValueError as e:
            S.warn(f"Region-area check skipped: {e}", log)
            A = RT = pd.DataFrame()
        if not A.empty:
            A.round(2).to_csv(out("female_region_areas.csv"), index=False)
            RT.to_csv(out("female_region_area_ratios.csv"), index=False)
            S.p("The whole-section mean of a lipid is the area-weighted mean of its region means. If the knockout and "
                "Control sections on a slide were cut at slightly different levels, or one lost tissue, the regions take up "
                "different shares of the two sections and the whole-section means differ even when no region changes. "
                f"Regions used (they do not overlap; the rest of the section is 'other'): "
                + ", ".join(figstyle.region_name(u) for u in parts) + ". Share of each section's pixels:")
            S.table(A.round(1), max_rows=20)
            show_rt = RT.set_index("slide")
            show_rt = (100 * (2 ** show_rt - 1)).round(0)
            show_rt.loc["mean"] = (100 * (2 ** RT.set_index("slide").mean() - 1)).round(0)
            S.p("Knockout share relative to the Control share on the same slide, in percent (whole_section: total "
                "section size):")
            S.table(show_rt.reset_index())
        if comp is not None and not comp.empty:
            comp.to_csv(out("female_whole_section_adjusted.csv"), index=False)
            no, na = int(comp.significant_observed.sum()), int(comp.significant_adjusted.sum())
            both = int((comp.significant_observed & comp.significant_adjusted).sum())
            r = float(np.corrcoef(comp.log2fc_observed, comp.log2fc_adjusted)[0, 1])
            share = float(np.nanmedian(np.abs(comp.log2fc_composition_only) / np.abs(comp.log2fc_observed).clip(lower=1e-6)))
            S.p(f"Whole section, recomputed with each knockout section's regions put back at the Control's proportions: "
                f"**{na}** lipids at q < {alpha:g} (observed: {no}; {both} in both). Correlation of the observed and "
                f"adjusted fold changes across lipids: r = {r:.2f}. The area difference alone accounts for a median of "
                f"{100 * share:.0f}% of the observed change per lipid.")
            S.p("Reading it: if most hits survive and the area-only part is small, the whole-section result is not a "
                "product of how the sections were cut. If many hits disappear, report the region results instead.")
            fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.3), gridspec_kw={"wspace": 0.3})
            ax = axes[0]
            cat = np.where(comp.significant_observed & comp.significant_adjusted, "both",
                           np.where(comp.significant_observed, "observed only",
                                    np.where(comp.significant_adjusted, "adjusted only", "neither")))
            for c_, col in (("neither", "#C8C8C8"), ("observed only", "#000000"), ("adjusted only", figstyle.CONTROL),
                            ("both", MAG)):
                m_ = cat == c_
                ax.scatter(comp.log2fc_observed[m_], comp.log2fc_adjusted[m_], s=13, c=col, edgecolors="none",
                           label=f"{c_} ({int(m_.sum())})")
            lim = np.nanmax(np.abs(np.r_[comp.log2fc_observed, comp.log2fc_adjusted, 0.1])) * 1.08
            ax.plot([-lim, lim], [-lim, lim], color="#999999", lw=0.5, ls=":")
            ax.axhline(0, color="#999999", lw=0.5); ax.axvline(0, color="#999999", lw=0.5)  # noqa: E702
            ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)  # noqa: E702
            ax.set_xlabel("observed log2 FC, whole section"); ax.set_ylabel("region-adjusted log2 FC")  # noqa: E702
            ax.set_title(f"Lipids significant at q < {alpha:g}")
            ax.legend(fontsize=7, loc="best", frameon=True, framealpha=0.85, edgecolor="none")
            ax = axes[1]
            if not RT.empty:
                cols_ = ["whole_section"] + parts
                x = np.arange(len(cols_))
                w = 0.8 / max(len(RT), 1)
                greys = ["#000000", "#777777", "#BBBBBB", MAG, "#555555"]
                for i, (_, rw) in enumerate(RT.iterrows()):
                    ax.bar(x + i * w - 0.4 + w / 2, [100 * (2 ** rw[c_] - 1) for c_ in cols_], w,
                           color=greys[i % len(greys)], label=f"slide {rw['slide']}")
                ax.axhline(0, color="#999999", lw=0.5)
                ax.set_xticks(x)
                ax.set_xticklabels([figstyle.region_name(c_) if c_ != "whole_section" else "section size" for c_ in cols_],
                                   rotation=30, ha="right", fontsize=8)
                ax.set_ylabel("knockout vs Control share (%)")
                ax.set_title("Region share of the section")
                ax.legend(fontsize=7, frameon=False)
            figstyle.title(fig, "Females: does section make-up explain the whole-section result?")
            S.fig(fig, "female_region_areas.png", caption=(
                "Left: each lipid's whole-section fold change as measured (x) and after putting each knockout section's "
                "regions back at the Control's proportions (y); dotted line, no change. Colour: significant in both "
                "(magenta), as measured only (black), adjusted only (blue), neither (grey). Right: knockout share of each "
                "region relative to the Control on the same slide (0 = same share); 'section size' compares total pixels."))

    volcano_grid(F, regions, "Females: knockout vs Control by region", "volcano_female.png", S, alpha=alpha, caption=(
        f"Each point is one of the {len(L)} quality-controlled lipids in one region; x, mean of the three female "
        "within-slide log2 differences (knockout / Control); y, −log10 p of the moderated paired t. Vermillion, higher "
        f"in the knockout; blue, lower; grey, q ≥ {alpha:g} (Benjamini-Hochberg within the region). The most significant "
        "lipids are named."))
    for lab in labelings:
        M = RES[RES.analysis == f"male_{lab}"]
        volcano_grid(M, regions, f"Males ({lab} labels): knockout vs Control by region", f"volcano_male_{lab}.png", S,
                     alpha=alpha, caption=(f"As the female volcano plot, for the three male pairs with the {lab} labelling."))

    # ---- quadrants -------------------------------------------------------------------------
    S.h("Females vs males: four-quadrant comparison")
    qrows = []
    for lab in labelings:
        M = RES[RES.analysis == f"male_{lab}"]
        J = F.merge(M, on=["unit", "lipid"], suffixes=("_f", "_m"))
        for stat in ("q", "p"):
            J[f"cat_{stat}"] = category(J[f"{stat}_f"], J[f"{stat}_m"], J["log2fc_f"], J["log2fc_m"], alpha)
        J["labeling"] = lab
        qrows.append(J)
        ncol = 3
        nrow = int(np.ceil(len(regions) / ncol))
        for stat, nice in (("q", f"q < {alpha:g}"), ("p", f"nominal p < {alpha:g}")):
            fig, axes = plt.subplots(nrow, ncol, figsize=(4.3 * ncol, 4.1 * nrow), squeeze=False,
                                     gridspec_kw={"hspace": 0.45, "wspace": 0.3})
            for ax, u in zip(axes.ravel(), regions):
                g = J[J.unit == u]
                for c, col in CAT_COL.items():
                    gg = g[g[f"cat_{stat}"] == c]
                    ax.scatter(gg["log2fc_f"], gg["log2fc_m"], s=16 if c != "neither" else 9, c=col, edgecolors="none",
                               label=f"{c} ({len(gg)})", zorder=3 if c != "neither" else 2)
                lim = np.nanmax(np.abs(np.r_[g["log2fc_f"], g["log2fc_m"], 0.1])) * 1.08
                ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)  # noqa: E702
                ax.axhline(0, color="#999999", lw=0.5); ax.axvline(0, color="#999999", lw=0.5)  # noqa: E702
                ax.plot([-lim, lim], [-lim, lim], color="#999999", lw=0.4, ls=":")
                for _, r in g[g[f"cat_{stat}"].isin(["both, same direction", "both, opposite"])].nsmallest(6, f"p_f").iterrows():
                    ax.annotate(r["lipid"], (r["log2fc_f"], r["log2fc_m"]), fontsize=6, xytext=(3, 2), textcoords="offset points")
                ax.set_title(figstyle.region_name(u))
                ax.set_xlabel("log2 FC, females")
                ax.set_ylabel(f"log2 FC, males ({lab})")
                ax.legend(fontsize=5.5, loc="best", handletextpad=0.2, borderaxespad=0.2, frameon=True, framealpha=0.85, edgecolor="none")
            for ax in axes.ravel()[len(regions):]:
                ax.set_visible(False)
            figstyle.title(fig, f"Females vs males ({lab} labels), significance at {nice}")
            S.fig(fig, f"quadrants_{lab}_{stat}.png", caption=(
                f"Each point is one lipid: genotype effect in females (x) against males with the {lab} labelling (y). "
                f"Colour, where the change is significant ({nice} in each sex's own paired test): magenta, both sexes in "
                "the same direction; black, both but opposite; vermillion, females only; blue, males only; grey, neither. "
                "Dotted line, equal effect in both sexes."))
    Q = pd.concat(qrows, ignore_index=True)
    Q.to_csv(out("female_vs_male.csv"), index=False)
    QC = (Q.groupby(["labeling", "unit", "cat_q"]).size().unstack(fill_value=0)
          .reindex(columns=list(CAT_COL), fill_value=0))
    QP = (Q.groupby(["labeling", "unit", "cat_p"]).size().unstack(fill_value=0)
          .reindex(columns=list(CAT_COL), fill_value=0))
    QC.to_csv(out("quadrant_counts_q.csv"))
    QP.to_csv(out("quadrant_counts_p.csv"))
    S.p(f"Counts per category at q < {alpha:g}:")
    S.table(QC.reset_index())
    S.p(f"The same at nominal p < {alpha:g} (looser, for trends only):")
    S.table(QP.reset_index())

    # ---- male labels from the molecular signature ----------------------------------------
    S.h("Male labels from the molecular signature")
    sig_rows, pca_rows = [], []
    fcol = {ctl: figstyle.CONTROL, mut: figstyle.MUTANT}
    for with_fko in (False, True):
        ncol = 3
        nrow = int(np.ceil(len(regions) / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(4.6 * ncol, 4.3 * nrow), squeeze=False,
                                 gridspec_kw={"hspace": 0.45, "wspace": 0.32})
        for ax, u in zip(axes.ravel(), regions):
            Lg = Lg_by_unit[u]
            keep = [a for a in Lg.index if a in info.index and (info.at[a, "sex"] == "Male" or info.at[a, "genotype"] == ctl or with_fko)]
            M = Lg.loc[keep].dropna(axis=1)
            sd = M.std(ddof=1).replace(0, np.nan)
            Z = ((M - M.mean()) / sd).dropna(axis=1).to_numpy()
            Z = Z - Z.mean(axis=0)
            U, sv, _ = np.linalg.svd(Z, full_matrices=False)
            sc = U[:, :2] * sv[:2]
            ve = sv ** 2 / (sv ** 2).sum()
            for i, a in enumerate(M.index):
                r = info.loc[a]
                female = r["sex"] == "Female"
                ax.scatter(sc[i, 0], sc[i, 1], marker="o" if female else "s", s=46,
                           facecolors=fcol[r["genotype"]] if female else "white",
                           edgecolors=fcol[r["genotype"]], linewidths=1.6, zorder=3)
                ax.annotate(r["short"], (sc[i, 0], sc[i, 1]), fontsize=7, xytext=(4, 3), textcoords="offset points")
                pca_rows.append({"view": "with female KO" if with_fko else "female Controls + males", "unit": u,
                                 "section": r["short"], "sex": r["sex"], "genotype_in_sheet": r["genotype"],
                                 "position": r["position_on_slide"], "PC1": sc[i, 0], "PC2": sc[i, 1],
                                 "PC1_var": ve[0], "PC2_var": ve[1]})
            ax.set_xlabel(f"PC1 ({100 * ve[0]:.0f}%)")
            ax.set_ylabel(f"PC2 ({100 * ve[1]:.0f}%)")
            ax.set_title(figstyle.region_name(u))
        for ax in axes.ravel()[len(regions):]:
            ax.set_visible(False)
        from matplotlib.lines import Line2D
        h = [Line2D([], [], marker="o", ls="", mfc=figstyle.CONTROL, mec=figstyle.CONTROL, label="Female Control")]
        if with_fko:
            h.append(Line2D([], [], marker="o", ls="", mfc=figstyle.MUTANT, mec=figstyle.MUTANT, label="Female KO"))
        h += [Line2D([], [], marker="s", ls="", mfc="white", mec=figstyle.CONTROL, mew=1.6, label="Male, Control in sheet"),
              Line2D([], [], marker="s", ls="", mfc="white", mec=figstyle.MUTANT, mew=1.6, label="Male, KO in sheet")]
        fig.legend(handles=h, loc="lower center", ncol=len(h), bbox_to_anchor=(0.5, -0.01))
        name = "pca_male_labels_with_female_ko.png" if with_fko else "pca_male_labels.png"
        figstyle.title(fig, "PCA of " + ("all female sections and the male sections" if with_fko else
                                         "the female Controls and the male sections"),
                       f"{len(L)} quality-controlled lipids, log2, z-scored; labels slide_animal")
        S.fig(fig, name, caption=(
            "Principal component analysis of per-section lipid profiles in each region. Circles, females (filled); squares, "
            "males (open), coloured by the genotype in the current sample sheet (blue Control, vermillion knockout). "
            "Labels, slide and animal number. A male section that sits with the female Controls on PC1 and PC2 behaves "
            "like a Control. All males were imaged in one run, so a shared offset of all male sections is a run effect, "
            "not genotype; the signature score table is free of that."))
    pd.DataFrame(pca_rows).to_csv(out("pca_male_labels.csv"), index=False)

    # signature score: Top - Bottom of each male slide vs the female KO - Control profile
    for u in regions:
        Lg = Lg_by_unit[u]
        fsig = pair_diffs(Lg, info, fem, ctl, mut, []).mean()
        for s in mal:
            sub = info[info.slide == s]
            top = sub.index[sub.position_on_slide == "Top"]
            bot = sub.index[sub.position_on_slide == "Bottom"]
            if not (len(top) and len(bot)) or top[0] not in Lg.index or bot[0] not in Lg.index:
                continue
            d = Lg.loc[top[0]] - Lg.loc[bot[0]]
            ok = d.notna() & fsig.notna()
            rho = sst.spearmanr(d[ok], fsig[ok])[0] if ok.sum() > 5 else np.nan
            sig_rows.append({"unit": u, "slide": s, "top": info.at[top[0], "short"], "bottom": info.at[bot[0], "short"],
                             "r_top_minus_bottom_vs_female_signature": rho,
                             "looks_like_KO": (info.at[top[0], "short"] if rho > 0 else info.at[bot[0], "short"]) if np.isfinite(rho) else ""})
    SIG = pd.DataFrame(sig_rows)
    # glycerophosphodiesters, the clearest CLN3 marker, if the metabolite mode ran
    mp = ctx.output_root / "results" / "09_differential" / "met_neg" / "class_values_per_animal.csv"
    if mp.exists() and len(SIG):
        CV = pd.read_csv(mp)
        for f in C.get("marker_features", ["GPI", "GPG"]):
            vals = []
            for _, r in SIG.iterrows():
                g = CV[(CV.unit == r.unit) & (CV.feature == f)].set_index("animal_id")["value"]
                sub = info[info.slide == r.slide]
                t = sub.index[sub.position_on_slide == "Top"][0]
                b = sub.index[sub.position_on_slide == "Bottom"][0]
                vals.append(np.log2(g.get(t, np.nan) / g.get(b, np.nan)) if g.get(b, 0) else np.nan)
            SIG[f"log2_{f}_top_vs_bottom"] = vals
    for lab, swap in labelings.items():
        # under this labelling, is the Top section of each male slide the KO?
        def top_is_ko(s):
            sub = info[info.slide == s]
            g = sub.set_index("position_on_slide")["genotype"].get("Top")
            return (g == mut) != (s in swap)
        SIG[f"agrees_with_{lab}"] = [((r > 0) == top_is_ko(s)) if np.isfinite(r) else np.nan
                                     for r, s in zip(SIG["r_top_minus_bottom_vs_female_signature"], SIG["slide"])]
    SIG.to_csv(out("male_signature_scores.csv"), index=False)
    S.p("Signature score: Spearman correlation, over the lipid set, between each male slide's Top-minus-Bottom profile and "
        "the mean female knockout-minus-Control profile. Positive: the Top section looks like the knockout; negative: the "
        "Bottom one does; near 0: no knockout signature either way. Being a within-slide difference, it is free of run "
        "and slide offsets. GPI and GPG columns: log2 Top/Bottom of the two glycerophosphodiesters that define CLN3 loss "
        "(positive: higher on Top). The agrees_with columns say whether each labelling matches the score.")
    S.table(SIG.round(3), max_rows=60)

    # ---- classes and ratios ----------------------------------------------------------------
    S.h("Class totals and ratios")
    crows = []
    for m in class_modes:
        CV = pd.read_csv(ctx.output_root / "results" / "09_differential" / m / "class_values_per_animal.csv")
        CV = CV[CV.unit.isin(regions)]
        for (u, kind), g in CV.groupby(["unit", "kind"]):
            W = g.pivot_table(index="animal_id", columns="feature", values="value")
            Wl = np.log2(W.where(W > 0))
            for an, slides, swap in [("female", fem, [])] + [(f"male_{k}", mal, list(v)) for k, v in labelings.items()]:
                D = pair_diffs(Wl, info, slides, ctl, mut, swap)
                if D.empty:
                    continue
                n = D.notna().sum()
                with np.errstate(all="ignore"):
                    p = sst.ttest_1samp(D.to_numpy(float), 0, nan_policy="omit").pvalue
                R = pd.DataFrame({"mode": m, "unit": u, "kind": kind, "feature": D.columns, "analysis": an, "n_pairs": n.to_numpy(),
                                  "log2fc": D.mean().to_numpy(), "p": p, "n_pairs_up": (D > 0).sum().to_numpy()})
                for s in D.index:
                    R[f"pair_{s}"] = D.loc[s].to_numpy()
                crows.append(R)
    if crows:
        K = pd.concat(crows, ignore_index=True)
        K["pct_change"] = 100 * (2 ** K["log2fc"] - 1)
        K["q"] = np.nan
        for _, idx in K.groupby(["mode", "analysis"]).groups.items():
            K.loc[idx, "q"] = stats.bh(K.loc[idx, "p"])
        K.to_csv(out("class_tests.csv"), index=False)
        keyf = C.get("key_features", ["HexCer", "Sulfatide / PE+PI+PS", "GM3", "Hex2Cer / HexCer", "GPI", "GPG"])
        KK = K[K.feature.isin(keyf) & K.unit.isin(["whole_section", "white_matter"])]
        piv = KK.pivot_table(index=["mode", "unit", "feature"], columns="analysis", values="pct_change").round(0)
        pq = KK.pivot_table(index=["mode", "unit", "feature"], columns="analysis", values="q").round(3)
        tab = piv.astype(str).add("% (q ").add(pq.astype(str)).add(")")
        S.p(f"Key classes and ratios, percent change knockout vs Control (q = BH within each mode and analysis):")
        S.table(tab.reset_index())

    # ---- workbook ---------------------------------------------------------------------------
    try:
        with pd.ExcelWriter(out("by_sex_results.xlsx")) as xw:
            for nm, df in [("female_significant", sigF[cols]), ("lipid_tests", RES), ("female_vs_male", Q),
                           ("quadrant_counts_q", QC.reset_index()), ("quadrant_counts_p", QP.reset_index()),
                           ("male_signature", SIG), ("class_tests", K if crows else pd.DataFrame()),
                           ("female_region_areas", A),
                           ("female_whole_adjusted", comp if comp is not None else pd.DataFrame()),
                           ("confidence_tiers", T1.rename_axis("tier").reset_index()),
                           ("annotation_sources", T2.rename_axis("source").reset_index()), ("atlas_evidence", T3),
                           ("lipid_set", L)]:
                df.to_excel(xw, sheet_name=nm[:31], index=False)
        log.info(f"workbook: {out('by_sex_results.xlsx')}")
    except Exception as e:     # a missing Excel writer must not stop the phase; the CSVs are there
        log.warning(f"could not write the Excel workbook: {e}")
    log.info(f"summary: {S.write()}")
    cli.finish(log, t0)


if __name__ == "__main__":
    main()
