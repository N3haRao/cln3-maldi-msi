"""Phase 09b: lipizones (pixels grouped by lipid profile) and per-animal tests inside them.

    python scripts/09b_lipizones.py --config config.yaml [--mode lip_neg]

Runs after 09 (it uses 09's feature QC to pick the peaks). See src/cln3msi/lipizones.py for
the method. In short: the pixels of all 12 sections are clustered on their lipid profile at
several levels (lipizones.levels, coarse to fine, nested), without genotype labels, with the
same number of training pixels from every section. Then, per animal:

  area tests     each animal's share of its section in each lipizone, log2, paired over the
                 slides (does a lipizone grow or shrink in the KO?). One BH family per level.
  lipid tests    each animal's mean of each QC-passing peak inside the lipizone, paired
                 moderated t as in 09. One BH family per lipizone, and q_level over the level.
  class tests    the class sums and ratios of class_tests (HexCer, sulfatide / PE+PI+PS, ...)
                 inside each lipizone. One BH family per level.

Pixels are never the replicates: every test has n = number of slides with both animals.

Writes processed/<mode>/lipizones.parquet and results/09b_lipizones/<mode>/:
  lipizones.csv              one row per finest lipizone: parents, size, slide/genotype
                             balance, best atlas region, top marker lipids
  lipizone_markers.csv       mean z-score of every peak in every finest lipizone
  lipizone_composition.csv   share of each animal's section per lipizone (every level)
  lipizone_atlas_overlap.csv share of each finest lipizone in each atlas region
  area_tests.csv, lipid_tests.csv, class_tests.csv
  lipizone_pixels.csv.gz     every pixel: position, animal, genotype, sex, lipizone per level
                             (same layout as a lipizone metadata table, for sharing)
  embedding.csv              2D embedding (UMAP if installed, else PCA) of a pixel subsample
and figures under figures/09b_lipizones/<mode>/.
"""
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cln3msi import classes, cli, figstyle, lipizones, stats  # noqa: E402
from cln3msi.images import section_image  # noqa: E402
from cln3msi.report import Summary, plt  # noqa: E402
from cln3msi.util import read_h5ad  # noqa: E402


def zname(k: int, i: int) -> str:
    return f"L{k}_{int(i):02d}"


def pick_features(ctx, mode, var, iso, S, log):
    """Peaks for clustering: 09's feature QC, else 08's Moran selection, else all non-isotope."""
    src = (ctx.cfg.get("lipizones") or {}).get("features", "feature_qc")
    fq = ctx.output_root / "results" / "09_differential" / mode / "feature_qc.csv"
    mp = ctx.output_root / "results" / "08_explore" / mode / "morans_i.csv"
    if src == "feature_qc" and fq.exists():
        F = pd.read_csv(fq, index_col=0)
        ids = [p for p in F.index[F["qc_pass"].astype(bool)] if p in var.index]
        if len(ids) >= 10:
            return ids, "09 feature QC (qc_pass)"
        S.warn(f"only {len(ids)} peaks pass the feature QC; using the Moran selection instead", log)
    if src in ("feature_qc", "structured") and mp.exists():
        M = pd.read_csv(mp)
        ids = [p for p in M.loc[M["selected"].astype(bool), "peak"] if p in var.index]
        if len(ids) >= 10:
            return ids, "08 Moran's I selection"
    ids = list(var.index[~iso])
    return ids, "all non-isotope peaks (no QC results found)"


def main():
    t0 = time.time()
    ctx, log, modes, _ = cli.start("09b_lipizones", __doc__)
    LZ = ctx.cfg.get("lipizones") or {}
    if not LZ.get("enabled", True):
        log.info("lipizones.enabled is false, nothing to do")
        return
    if LZ.get("modes"):
        modes = [m for m in modes if m in LZ["modes"]]
    D = ctx.cfg["differential"]
    G = ctx.cfg["project"]["genotypes"]
    ctl, mut = G["control"], G["mutant"]
    seed = int(ctx.get("runtime.seed", 0))
    rng = np.random.default_rng(seed)
    levels = sorted({int(k) for k in LZ.get("levels", [4, 12, 40])})
    test_levels = [k for k in LZ.get("test_levels", levels[:-1] or levels) if k in levels]
    min_px = int(D["min_pixels_per_unit"])
    min_pairs = int(D.get("min_pairs", 5))
    alpha = float(D["fdr_alpha"])

    for mode in modes:
        S = Summary(ctx, "09b_lipizones", mode, f"Lipizones, {mode}")
        out = lambda name: ctx.out("results", "09b_lipizones", mode, name)  # noqa: E731
        p = ctx.out("processed", mode, "matrix_norm.h5ad", mkdir=False)
        if not p.exists():
            S.warn("no matrix_norm.h5ad, run 06 first", log)
            S.write()
            continue
        a = read_h5ad(p)
        obs, var = a.obs, a.var
        iso = var.get("is_isotope", pd.Series(False, index=var.index)).fillna(False).astype(bool).to_numpy()
        feat_ids, feat_src = pick_features(ctx, mode, var, iso, S, log)
        cols = var.index.get_indexer(feat_ids)
        labels = (var["label"].astype(str) if "label" in var else pd.Series(var.index, index=var.index)).loc[feat_ids].tolist()
        X = np.asarray(a.X[:, cols], dtype=np.float64)
        slides = obs["slide"].astype(str).to_numpy()
        sec_key = (obs["slide"].astype(str) + ":" + obs["section"].astype(str)).to_numpy()
        core = obs["core"].to_numpy().astype(bool) if "core" in obs else np.ones(len(obs), bool)
        animals = obs.drop_duplicates("animal_id").set_index("animal_id")[["slide", "genotype", "sex", "section"]]
        log.info(f"{mode}: {len(feat_ids)} peaks ({feat_src}), {len(obs)} pixels, levels {levels}")

        # ---- cluster ------------------------------------------------------------------------
        Z = lipizones.scale_features(X, slides, bool(LZ.get("per_slide_scaling", True)))
        geno = obs["genotype"].astype(str).to_numpy()
        train_on = LZ.get("train_on", "control")
        pool = (geno == ctl) if train_on == "control" else np.ones(len(obs), bool)
        tr_local = lipizones.balanced_sample(sec_key[pool], int(LZ.get("train_pixels_per_section", 5000)), rng)
        train = np.flatnonzero(pool)[tr_local]
        if levels[-1] >= len(train):
            S.warn(f"too few training pixels ({len(train)}) for {levels[-1]} lipizones", log)
            S.write()
            continue
        fit = lipizones.fit(Z, train, levels, int(LZ.get("n_pcs", 30)), seed)
        lab = fit["labels"]
        k_fine = levels[-1]
        fine = lab[k_fine]
        expl = float(fit["pca"].explained_variance_ratio_.sum())
        log.info(f"{mode}: PCA {fit['pcs'].shape[1]} components explain {expl:.0%} of the variance")

        px = pd.DataFrame({f"lipizone_k{k}": [zname(k, i) for i in lab[k]] for k in levels}, index=obs.index)
        px.to_parquet(ctx.out("processed", mode, "lipizones.parquet"))
        share = obs[["x", "y", "slide", "section", "animal_id", "genotype", "sex"]].copy()
        share.insert(0, "spotId", obs.index)
        share = pd.concat([share.reset_index(drop=True), px.reset_index(drop=True)], axis=1)
        share["core_pixel"] = core
        share.to_csv(out("lipizone_pixels.csv.gz"), index=False, compression="gzip")

        # ---- describe each lipizone ------------------------------------------------------------
        mk = lipizones.markers(Z, fine, labels)
        mk.index = [zname(k_fine, i) for i in mk.index]
        mk.to_csv(out("lipizone_markers.csv"))
        top_n = int(LZ.get("marker_top_n", 5))
        rp = ctx.out("processed", mode, "regions.parquet", mkdir=False)
        reg = None
        if rp.exists():
            reg = pd.read_parquet(rp).loc[obs.index]
            rcols = [c for c in reg.columns if c.startswith("region_")
                     and c not in ("region_whole_section", "region_wm_lipid", "region_gm_lipid")]
            if not any(reg[c].any() for c in rcols):
                rcols = [c for c in ("region_wm_lipid", "region_gm_lipid") if c in reg]
            reg = reg[rcols] if rcols else None
        rows = []
        for i in range(1, k_fine + 1):
            sel = fine == i
            n = int(sel.sum())
            slide_share = pd.Series(slides[sel]).value_counts(normalize=True)
            r = {"lipizone": zname(k_fine, i)}
            for k in levels[:-1]:
                r[f"parent_k{k}"] = zname(k, fit["parent"][k][i - 1])
            r.update({"n_pixels": n, "pct_of_tissue": 100 * n / len(obs),
                      "n_sections": int(pd.Series(sec_key[sel]).nunique()),
                      "share_mutant_pixels": float((geno[sel] == mut).mean()) if n else np.nan,
                      "top_slide": slide_share.index[0] if n else "",
                      "top_slide_share": float(slide_share.iloc[0]) if n else np.nan})
            if reg is not None and n:
                fr = reg.loc[sel].mean()
                r["best_region"] = fr.idxmax().replace("region_", "")
                r["best_region_share"] = float(fr.max())
            m = mk.loc[zname(k_fine, i)].sort_values(ascending=False)
            r["top_markers"] = "; ".join(f"{lbl} ({v:+.1f})" for lbl, v in m.head(top_n).items())
            r["lowest_markers"] = "; ".join(f"{lbl} ({v:+.1f})" for lbl, v in m.tail(3)[::-1].items())
            rows.append(r)
        info = pd.DataFrame(rows)
        best = {}                     # any level's lipizone -> its best atlas region
        if reg is not None:
            for k in levels:
                for i in np.unique(lab[k]):
                    fr = reg.loc[lab[k] == i].mean()
                    best[zname(k, i)] = f"{fr.idxmax().replace('region_', '')} ({fr.max():.0%})"
        n_sl = len(np.unique(slides))
        info["slide_specific"] = info["top_slide_share"] > float(LZ.get("slide_specific_share", 0.5))
        info["genotype_specific"] = (info["share_mutant_pixels"] - 0.5).abs() > 0.4
        info.to_csv(out("lipizones.csv"), index=False)
        if reg is not None:
            ov = pd.DataFrame({zname(k_fine, i): reg.loc[fine == i].mean() for i in range(1, k_fine + 1)}).T
            ov.columns = [c.replace("region_", "") for c in ov.columns]
            ov.to_csv(out("lipizone_atlas_overlap.csv"))
        comp = []
        for k in levels:
            c = lipizones.composition(obs, lab[k])
            c.columns = [zname(k, i) for i in c.columns]
            comp.append(c)
        comp = animals[["slide", "genotype", "sex"]].join(pd.concat(comp, axis=1))
        comp.to_csv(out("lipizone_composition.csv"))

        # ---- per-animal tests ------------------------------------------------------------------
        # 1. area: share of the section in each lipizone
        vals = []
        for k in levels:
            c = lipizones.composition(obs, lab[k])
            for i in c.columns:
                vals.append(pd.DataFrame({"unit": f"k{k}", "kind": "area", "feature": zname(k, i), "n_species": np.nan,
                                          "animal_id": c.index, "value": c[i].to_numpy()}))
        V = pd.concat(vals, ignore_index=True)
        A = classes.test_features(V, animals, ctl, mut, min_pairs)
        if len(A):
            A = A.rename(columns={"unit": "level"})
            A["q_level"] = np.nan
            for _, idx in A.groupby("level").groups.items():
                A.loc[idx, "q_level"] = stats.bh(A.loc[idx, "p_ttest"])
            A["best_region"] = A["feature"].map(best)
            A["top_markers"] = A["feature"].map(dict(zip(info["lipizone"], info["top_markers"])))
            A = A.drop(columns=["kind", "n_species"], errors="ignore")
            A.to_csv(out("area_tests.csv"), index=False)

        # 2. lipids inside lipizones, and 3. classes inside lipizones
        LT, ML = [], []
        for k in test_levels:
            for i in np.unique(lab[k]):
                unit = zname(k, i)
                mask = (lab[k] == i) & core
                means, npx = stats.per_animal_means(X, obs, mask, min_px)
                means.columns = feat_ids
                if means.notna().any(axis=1).sum() < 4:
                    continue
                log2 = np.log2(means + stats.pseudocount(means))
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)   # lipizones missing in some animals
                    res = stats.paired_tests(log2, animals, ctl, mut, min_pairs=min_pairs, moderated=True)
                res.insert(0, "lipizone", unit)
                res.insert(0, "level", f"k{k}")
                res["median_pixels_per_animal"] = float(npx.median())
                LT.append(res)
                ml = means.stack().rename("mean_norm_intensity").reset_index()
                ml.columns = ["animal_id", "peak", "mean_norm_intensity"]
                ml.insert(0, "unit", unit)
                ML.append(ml)
        L = pd.DataFrame()
        if LT:
            L = pd.concat(LT, ignore_index=True)
            L["label"] = L["peak"].map(dict(zip(feat_ids, labels)))
            for c in ("annotation", "lipid_class", "confidence_tier"):
                if c in var:
                    L[c] = L["peak"].map(var[c])
            L["p_primary"] = L["p_moderated"].fillna(L["p_ttest"]) if "p_moderated" in L else L["p_ttest"]
            L["q_lipizone"] = np.nan
            L["q_level"] = np.nan
            for _, idx in L.groupby("lipizone").groups.items():
                L.loc[idx, "q_lipizone"] = stats.bh(L.loc[idx, "p_primary"])
            for _, idx in L.groupby("level").groups.items():
                L.loc[idx, "q_level"] = stats.bh(L.loc[idx, "p_primary"])
            L["best_region"] = L["lipizone"].map(best)
            L["pct_change"] = 100 * (2 ** L["log2fc"] - 1)
            L["large_effect"] = L["log2fc"].abs() >= float(D.get("large_effect_log2fc", 0.2))
            L["_k"] = L["level"].str[1:].astype(int)
            L = L.sort_values(["_k", "p_primary"]).drop(columns="_k")
            L.to_csv(out("lipid_tests.csv"), index=False)

        CT = pd.DataFrame()
        C = ctx.cfg.get("class_tests") or {}
        ap = ctx.out("processed", mode, "annotations.csv", mkdir=False)
        class_cfg = (C.get("classes") or {}).get(mode) or {}
        if ML and class_cfg and ap.exists():
            MLd = pd.concat(ML, ignore_index=True)
            ann = classes.usable_peaks(pd.read_csv(ap), int(C.get("min_sections", 0)), C.get("ambiguity", "tags"))
            pid = ann["peak_id"] if "peak_id" in ann else ann.index
            ann = ann[pd.Series(pid, index=ann.index).isin(set(feat_ids))]
            vals, _ = classes.totals(MLd, ann, class_cfg, (C.get("ratios") or {}).get(mode) or [],
                                     list(MLd["unit"].unique()))
            if not vals.empty:
                CT = classes.test_features(vals, animals, ctl, mut, min_pairs)
                CT = CT.rename(columns={"unit": "lipizone"})
                CT.insert(0, "level", CT["lipizone"].str.extract(r"^L(\d+)_")[0].map(lambda s: f"k{s}"))
                CT["q_level"] = np.nan
                for _, idx in CT.groupby("level").groups.items():
                    CT.loc[idx, "q_level"] = stats.bh(CT.loc[idx, "p_ttest"])
                CT["best_region"] = CT["lipizone"].map(best)
                CT.to_csv(out("class_tests.csv"), index=False)

        # ---- embedding --------------------------------------------------------------------------
        emb_n = int(LZ.get("embedding_pixels", 20000))
        sub = lipizones.balanced_sample(sec_key, max(1, emb_n // max(1, len(np.unique(sec_key)))), rng)
        emb_kind = "PCA"
        E2 = fit["pcs"][sub, :2]
        if LZ.get("umap", True):
            try:
                import umap  # type: ignore
                E2 = umap.UMAP(n_neighbors=30, min_dist=0.3, random_state=seed).fit_transform(fit["pcs"][sub])
                emb_kind = "UMAP"
            except Exception as e:     # umap-learn is optional
                log.info("UMAP (optional umap-learn package) not installed; embedding = first two PCs")
        emb = pd.DataFrame({"spotId": obs.index[sub], "dim1": E2[:, 0], "dim2": E2[:, 1],
                            "slide": slides[sub], "genotype": geno[sub], f"lipizone_k{k_fine}": px.iloc[sub, -1].to_numpy()})
        emb.to_csv(out("embedding.csv"), index=False)

        # ---- figures ----------------------------------------------------------------------------
        from matplotlib.colors import ListedColormap
        from matplotlib.patches import Patch
        secs = list(obs.groupby(["slide", "section"], observed=True).groups)
        um = float(ctx.get("acquisition.pixel_size_um", 50))
        mname = figstyle.mode_name(mode)
        train_txt = ("learned on the Control sections only (5,000 pixels per section at most) and every KO pixel "
                     "assigned to its nearest Control lipizone" if train_on == "control" else
                     "learned on an equal number of pixels from every section")
        asp = figstyle.section_aspect(obs)
        for k in levels:
            colors = lipizones.palette(k)
            cm = ListedColormap(colors)
            ncol = 6
            nrow = int(np.ceil(len(secs) / ncol))
            fig, axes = figstyle.image_grid(nrow, ncol, 1.75, asp, title_h=0.25)
            for b_, (slide, section) in enumerate(secs):
                ax = axes.ravel()[b_]
                img, *_ = section_image(obs, lab[k].astype(float), slide, section)
                ax.imshow(img, cmap=cm, vmin=0.5, vmax=k + 0.5, interpolation="nearest")
                g = obs.loc[(obs.slide == slide) & (obs.section == section), "genotype"].iloc[0]
                ax.set_title(f"Slide {slide}, {'KO' if g == mut else 'Control'}", fontsize=7, fontweight="normal",
                             color=figstyle.MUTANT if g == mut else figstyle.CONTROL)
            for ax in axes.ravel():
                figstyle.image_axes(ax)
                if not ax.images:
                    ax.axis("off")
            figstyle.scalebar(axes.ravel()[len(secs) - 1], um)
            if k <= 12:
                hd = [Patch(color=colors[i - 1], label=f"{zname(k, i)}" + (f"  {best.get(zname(k, i), '').split(' (')[0].replace('_', ' ')}"
                                                                          if best else "")) for i in range(1, k + 1)]
                fig.legend(handles=hd, loc="upper center", ncol=min(k, 6), bbox_to_anchor=(0.5, -0.01), fontsize=6.5)
            figstyle.title(fig, f"{mname}: {k} lipizones", "Pixels clustered by lipid profile without genotype labels")
            S.fig(fig, f"lipizone_maps_k{k}.png", caption=(
                f"Lipizones at the {k}-cluster level mapped onto every section. Pixels were clustered on their profile "
                f"of {len(feat_ids)} quality-controlled ions (per-slide median scaling, log transform, z-score, "
                f"{fit['pcs'].shape[1]} principal components, k-means), {train_txt}. Coarser levels merge finer "
                "lipizones by Ward linkage, so the levels nest. Colours are arbitrary"
                + (" (legend: lipizone and the atlas region it overlaps most)" if k <= 12 else
                   "; lipizone identities and marker lipids are listed in lipizones.csv") + ". Scale bar, 1 mm."))

        fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.5), gridspec_kw={"wspace": 0.08})
        cfine = np.array(lipizones.palette(k_fine))
        ids = np.array([int(s.split("_")[1]) for s in emb[f"lipizone_k{k_fine}"]])
        order_ = np.random.default_rng(seed).permutation(len(emb))     # no colour drawn last on top
        axes[0].scatter(emb.dim1.to_numpy()[order_], emb.dim2.to_numpy()[order_], s=0.6, c=cfine[ids - 1][order_],
                        rasterized=True, linewidths=0)
        axes[0].set_title(f"Lipizone (k = {k_fine})")
        gcol = figstyle.genotype_colors(ctx)
        for g in pd.unique(emb.genotype.to_numpy()[order_]):
            s_ = (emb.genotype == g).to_numpy()
            axes[1].scatter(emb.dim1[s_], emb.dim2[s_], s=0.6, color=gcol.get(g, figstyle.GREY), alpha=0.35,
                            label="CLN3 KO" if g == mut else "Control", rasterized=True, linewidths=0)
        axes[1].legend(markerscale=10, loc="upper right")
        axes[1].set_title("Genotype")
        sl_u = sorted(emb.slide.unique())
        for j, s in enumerate(sl_u):
            s_ = (emb.slide == s).to_numpy()
            axes[2].scatter(emb.dim1[s_], emb.dim2[s_], s=0.6, color=figstyle.OKABE_ITO[j % 8], alpha=0.35,
                            label=f"Slide {s}", rasterized=True, linewidths=0)
        axes[2].legend(markerscale=10, loc="upper right", ncol=2)
        axes[2].set_title("Slide")
        for ax in axes:
            ax.set_xticks([]); ax.set_yticks([])  # noqa: E702
            ax.set_xlabel(f"{emb_kind} 1"); ax.set_ylabel(f"{emb_kind} 2")  # noqa: E702
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
        figstyle.panel_labels(axes, x=0.0)
        figstyle.title(fig, f"{mname}: lipid-profile embedding of pixels")
        S.fig(fig, "lipizone_embedding.png", caption=(
            f"Two-dimensional {emb_kind} embedding of {len(emb):,} pixels sampled equally from all sections, coloured by "
            f"(A) lipizone, (B) genotype and (C) slide. Lipizones should form distinct groups, while genotypes and "
            "slides should overlap; groups made of one slide would indicate a batch effect rather than anatomy."))

        show = mk.copy()
        tops = []
        for z in show.index:
            for c_ in show.loc[z].sort_values(ascending=False).head(2).index:
                if c_ not in tops:
                    tops.append(c_)
        H = show[tops]
        fig, ax = plt.subplots(figsize=(min(18, 0.17 * len(tops) + 3), 0.16 * len(H) + 1.8))
        im = ax.imshow(H.to_numpy(), aspect="auto", cmap="RdBu_r", vmin=-2, vmax=2)
        ax.set_yticks(range(len(H)), H.index, fontsize=5.5)
        ax.set_xticks(range(len(tops)), tops, rotation=90, fontsize=5)
        for sp in ax.spines.values():
            sp.set_visible(False)
        figstyle.colorbar(fig, im, ax, "Mean z-score in lipizone", fraction=0.02)
        figstyle.title(fig, f"{mname}: marker lipids of each lipizone")
        S.fig(fig, "lipizone_markers.png", caption=(
            f"Mean z-scored intensity of the marker ions (the two most enriched per lipizone) in each of the {k_fine} "
            "finest lipizones. Rows follow the Ward dendrogram of lipizone centroids, so similar lipizones are "
            "adjacent. Red, above the tissue average; blue, below. Values beyond \u00b12 are clipped."))

        if reg is not None:
            ov = pd.read_csv(out("lipizone_atlas_overlap.csv"), index_col=0)
            ov.columns = [figstyle.region_name(c) for c in ov.columns]
            fig, ax = plt.subplots(figsize=(0.5 * ov.shape[1] + 2.2, 0.16 * len(ov) + 1.6))
            im = ax.imshow(ov.to_numpy(), aspect="auto", cmap="viridis", vmin=0, vmax=1)
            ax.set_yticks(range(len(ov)), ov.index, fontsize=5.5)
            ax.set_xticks(range(ov.shape[1]), ov.columns, rotation=45, ha="right")
            for sp in ax.spines.values():
                sp.set_visible(False)
            figstyle.colorbar(fig, im, ax, "Share of lipizone pixels in region", fraction=0.05)
            figstyle.title(fig, f"{mname}: lipizones and atlas regions")
            S.fig(fig, "lipizone_atlas_overlap.png", caption=(
                f"Share of each finest lipizone's pixels that fall inside each registered Allen atlas region. "
                "Regions overlap (white matter contains the corpus callosum), so rows need not sum to 1; a lipizone "
                "concentrated in one column follows that anatomical region."))

        if len(A):
            kk = test_levels[-1] if test_levels else levels[0]
            Aa = A[A.level == f"k{kk}"].copy()
            pcols = [c for c in Aa.columns if c.startswith("pair_")]
            fig, ax = plt.subplots(figsize=(4.8, 0.24 * len(Aa) + 1.3))
            y = np.arange(len(Aa))
            for j, c_ in enumerate(pcols):
                ax.scatter(Aa[c_], y, s=9, color=figstyle.OKABE_ITO[j % 8], alpha=0.8, zorder=2, linewidths=0,
                           label=f"Slide {c_.replace('pair_', '')}")
            sigA = (Aa["q_level"] < alpha).to_numpy()
            fc_, lo_, hi_ = (Aa[c].to_numpy(float) for c in ("log2fc", "ci_low", "ci_high"))
            ax.errorbar(fc_, y, xerr=[fc_ - lo_, hi_ - fc_],
                        fmt="none", ecolor="k", elinewidth=1, zorder=3)
            ax.scatter(fc_, y, s=22, marker="D", color=np.where(sigA, "k", "white"), edgecolors="k",
                       linewidths=0.8, zorder=4)
            ax.axvline(0, color="k", lw=0.5)
            ax.set_yticks(y, Aa["feature"] + "  " + Aa["best_region"].fillna("").astype(str).str.replace("_", " "))
            ax.invert_yaxis()
            ax.set_xlabel("log2(KO / Control) of the lipizone's share of the section")
            ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0))
            figstyle.title(fig, f"{mname}: lipizone size, KO vs Control ({kk} lipizones)")
            S.fig(fig, f"area_tests_k{kk}.png", caption=(
                f"Change in the share of each section occupied by each of the {kk} lipizones. Coloured dots, single "
                "slides (log2 KO share / Control share); diamonds and bars, mean and 95% confidence interval of the "
                f"paired t-test over slides (filled, Benjamini-Hochberg q < {alpha:g} within this level). Labels give "
                "the atlas region the lipizone overlaps most. A shift means the region changed in size or its lipid "
                "make-up moved it toward another lipizone."))

        # ---- summary ----------------------------------------------------------------------------
        S.p("Pixels were grouped by lipid profile without genotype labels; every test below is per "
            "animal (n = slides with both animals), never per pixel. The lipids that define a lipizone "
            "also decide which pixels fall in it, so a lipid test inside a lipizone partly conditions on "
            "that lipid: treat lipid and class tests here as exploratory, and the 09 atlas-region tests "
            "(regions drawn without lipid data) as the main analysis.")
        S.h("Setup")
        S.kv({"features": f"{len(feat_ids)} peaks ({feat_src})", "pixels": len(obs),
              "training pixels": f"{len(train)} (up to {int(LZ.get('train_pixels_per_section', 5000))} per section, "
                                 f"{'Control sections only; KO pixels assigned to the nearest Control lipizone' if train_on == 'control' else 'all sections'})",
              "PCA": f"{fit['pcs'].shape[1]} components, {expl:.0%} of the variance",
              "levels": ", ".join(map(str, levels)), "tested levels (lipids, classes)": ", ".join(map(str, test_levels))})
        S.h(f"Finest lipizones (k = {k_fine})")
        cols_show = [c for c in ("lipizone", "pct_of_tissue", "best_region", "best_region_share", "share_mutant_pixels",
                                 "top_slide_share", "top_markers") if c in info]
        S.table(info[cols_show].round(3), max_rows=60)
        nss, ngs = int(info["slide_specific"].sum()), int(info["genotype_specific"].sum())
        if nss:
            S.warn(f"{nss} lipizone(s) have more than {float(LZ.get('slide_specific_share', 0.5)):.0%} of their pixels "
                   f"on one slide (expected about {1 / n_sl:.0%}): batch, not anatomy. See lipizones.csv.", log)
        if ngs:
            S.p(f"{ngs} lipizone(s) are more than 90% from one genotype (share_mutant_pixels). Check whether "
                "they are on one slide too; if not, they may be KO- or Control-specific tissue states.")
        if len(A):
            S.h("Area tests (share of the section in each lipizone)")
            S.p("A change in share means KO tissue matches that Control lipizone more or less often: either "
                "the region itself is bigger or smaller (atrophy), or its lipid make-up has shifted so its "
                "pixels now look like a neighbouring lipizone. Check the maps and the class tests before "
                "reading it as a change in size.")
            S.table(A.sort_values("p_ttest")[[c for c in ("level", "feature", "best_region", "log2fc", "p_ttest", "q_level",
                                                          "n_pairs", "n_pairs_up") if c in A]].head(15))
        if len(CT):
            S.h("Class tests inside lipizones")
            S.table(CT.sort_values("p_ttest")[[c for c in ("level", "lipizone", "best_region", "feature", "log2fc",
                                                           "pct_change", "p_ttest", "q_level", "n_pairs_up") if c in CT]].head(20))
        if len(L):
            S.h("Single lipids inside lipizones")
            n_sig = int((L["q_lipizone"] < alpha).sum())
            S.p(f"{len(L)} tests; {n_sig} with q < {alpha} within their lipizone, "
                f"{int((L['q_level'] < alpha).sum())} with q < {alpha} over the whole level.")
            S.table(L.sort_values("p_primary").head(20)[[c for c in ("level", "lipizone", "best_region", "label", "log2fc", "p_primary", "q_lipizone", "q_level",
                                            "n_pairs_up") if c in L]])
        log.info(f"summary: {S.write()}")
    cli.finish(log, t0)


if __name__ == "__main__":
    main()
