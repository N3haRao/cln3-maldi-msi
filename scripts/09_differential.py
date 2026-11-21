"""Phase 09: which lipids change in CLN3 KO, and where. Per-animal summaries, paired by slide.

    python scripts/09_differential.py --config config.yaml

Units we test in:
  regions      from 07: whole_section, wm_lipid / gm_lipid (always), and the atlas regions
               (white_matter, corpus_callosum, cortex, hippocampus, thalamus) for sections
               whose registration passed QC
  territories  from 08 (differential.run_territories), each territory as a region

For each unit and lipid: mean normalized intensity per animal (edge pixels left out,
at least differential.min_pixels_per_unit pixels), log2, then Mutant minus Control within
each slide. See src/cln3msi/stats.py for the tests. Isotope peaks aren't tested (they just
mirror their monoisotopic peak).

FDR (Benjamini-Hochberg) families, within each mode and unit:
  q_targeted   only the peaks tagged with a hypothesis class (Cer, HexCer, SHexCer, ...)
  q_all        the untargeted view: with differential.untargeted_peaks = lamanno, only peaks
               passing the Lipid Brain Atlas feature QC (src/cln3msi/featureqc.py); the rest
               are tested and reported but get no q_all
Every result also gets large_effect = |log2FC| >= differential.large_effect_log2fc.

Position flag: within a slide genotype and position coincide (female slides: Control on
Top; male slides: CLN3 KO on Top, from the sample sheet). From 03 we know, per slide, how
much brighter the bottom of the slide was than the top in the off-tissue DAN signal. That
is turned into a predicted log2(Mutant/Control) per slide (sign from where the Mutant sits)
and averaged over slides, like the log2FC itself. If a hit's log2FC has the same sign and is
less than qc.gradient.flag_ratio times that, it's flagged "could_be_position". It's a
warning, not a correction.

Then, from the same per-animal means (see src/cln3msi/classes.py):
  class tests        totals per lipid class and within-animal ratios (config class_tests),
                     one BH family per mode
  reaction ratios    product / precursor per ceramide backbone (config class_tests.reactions),
                     their own BH family
  slide consistency  do the slides' KO/Control difference profiles agree? A slide that is
                     the mirror image of the others is flagged as a possible label swap
                     (config pair_check). A warning, never an automatic fix.

Writes results/09_differential/<mode>/differential.csv, per_animal_means.csv,
class_tests.csv, class_members.csv, reaction_tests.csv, feature_qc.csv, pair_consistency.csv,
pair_correlation_<unit>.csv,
volcanoes and class/consistency figures, summary (hypothesis classes first, then untargeted).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cln3msi import classes, cli, featureqc, figstyle, stats  # noqa: E402
from cln3msi.util import read_h5ad  # noqa: E402
from cln3msi.report import Summary, plt  # noqa: E402


def gradient_prediction(ctx, mode: str, obs: pd.DataFrame, mutant: str) -> tuple[float, pd.Series]:
    """Mean over slides of the predicted log2(Mutant/Control) from the off-tissue trend (the mean,
    because log2fc is the mean of the per-slide differences). The sign of each slide's prediction
    follows the sample sheet: + if the Mutant is the Bottom section, - if it is the Top one."""
    p = ctx.output_root / "results" / "03_qc_all" / mode / "gradient.csv"
    if not p.exists():
        return np.nan, pd.Series(dtype=float)
    g = pd.read_csv(p).set_index("slide")
    col = next((c for c in ("matrix_off_pred_log2_bottom_vs_top", "tic_off_pred_log2_bottom_vs_top") if c in g), None)
    if col is None:
        return np.nan, pd.Series(dtype=float)
    mut_pos = obs[obs.genotype == mutant].drop_duplicates("slide").set_index("slide")["section"]
    sign = pd.Series({s: (1.0 if mut_pos.get(s) == "Bottom" else -1.0) for s in g.index})
    pred = g[col] * sign
    return float(pred.mean()), pred


SLIDE_COLOURS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#000000", "#F0E442"]


def class_level(ctx, mode, ML, animals, ctl, mut, S, log, offsets=None, ML_alt=None, QC=None):
    """Class totals, ratios and the slide consistency check, from the per-animal means."""
    C = ctx.cfg.get("class_tests", {}) or {}
    PC = ctx.cfg.get("pair_check", {}) or {}
    out = lambda name: ctx.out("results", "09_differential", mode, name)  # noqa: E731
    min_pairs = int(ctx.cfg["differential"].get("min_pairs", 5))
    alpha = float(ctx.cfg["differential"]["fdr_alpha"])

    # ---- slide consistency first: the class summary refers to it --------------------------
    flagged = []
    if PC.get("enabled", True):
        units = [u for u in PC.get("units", ["whole_section"]) if u in set(ML["unit"])]
        mats, T, flagged = classes.consistency(ML, animals, ctl, mut, units, float(PC.get("flag_below", -0.3)),
                                                  int(PC.get("min_peaks", 20)))
        S.h("Slide consistency (do the KO/Control differences agree between slides?)")
        if T.empty:
            S.p(f"Not enough slides or peaks to check (needs 3 slides and pair_check.min_peaks = {int(PC.get('min_peaks', 20))} peaks).")
        else:
            T.to_csv(out("pair_consistency.csv"))
            for u, M in mats.items():
                M.to_csv(out(f"pair_correlation_{u}.csv"))
            S.p("Spearman correlation, over all tested peaks, between the slides' log2(Mutant/Control) "
                "profiles. With correct labels the slides share the genotype effect, so the correlations "
                "should be positive or near 0. The table gives each slide's median correlation with the "
                f"other slides; a slide is flagged when that is below {float(PC.get('flag_below', -0.3)):g} "
                "in at least half of the regions checked.")
            S.table(T.reset_index().round(3))
            if flagged:
                S.warn(f"slide(s) {', '.join(flagged)}: the KO/Control difference profile is the mirror image "
                       "of the other slides. This looks like the two sections are swapped (sample sheet, or "
                       "slide loaded the other way round). Check the sectioning/mounting records; only if they "
                       "confirm it, add the slide to qc.sections.flip_slides and rerun from 03. Do not relabel "
                       "because of this result alone.", log)
            else:
                S.p("No slide flagged.")
            fig, axes = plt.subplots(1, len(mats), figsize=(2.9 * len(mats) + 0.8, 2.9), squeeze=False)
            for ax, (u, M) in zip(axes[0], mats.items()):
                im = ax.imshow(M.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1)
                ax.set_xticks(range(len(M)), M.columns)
                ax.set_yticks(range(len(M)), M.index)
                for i in range(len(M)):
                    for j in range(len(M)):
                        v = M.iat[i, j]
                        ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=5.5,
                                color="white" if abs(v) > 0.6 else "black")
                ax.set_title(figstyle.region_name(u))
                ax.set_xlabel("Slide"); ax.set_ylabel("Slide")  # noqa: E702
                for sp in ax.spines.values():
                    sp.set_visible(False)
            cb = fig.colorbar(im, ax=axes[0].tolist(), shrink=0.8, pad=0.02)
            cb.set_label("Spearman correlation"); cb.outline.set_linewidth(0.5)  # noqa: E702
            figstyle.panel_labels(axes[0], x=-0.2)
            figstyle.title(fig, f"{figstyle.mode_name(mode)}: agreement between slides")
            S.fig(fig, "pair_consistency.png", caption=(
                "Spearman correlation between slides of their KO-minus-Control difference profiles (per-animal log2 "
                "mean intensity, over all tested peaks) in each region. With correct genotype labels the slides share "
                "the genotype effect and correlate positively or near zero; a slide anti-correlated with the others "
                f"(median below {float(PC.get('flag_below', -0.3)):g}) suggests swapped section labels."))

    # ---- class and ratio tests --------------------------------------------------------------
    if not C.get("enabled", True):
        return
    class_cfg = (C.get("classes") or {}).get(mode) or {}
    ratio_cfg = (C.get("ratios") or {}).get(mode) or []
    if not class_cfg:
        log.info(f"{mode}: no class_tests.classes for this mode, class tests skipped")
        return
    ap = ctx.out("processed", mode, "annotations.csv", mkdir=False)
    if not ap.exists():
        S.warn("no annotations.csv (run 05): class tests skipped", log)
        return
    ann = classes.usable_peaks(pd.read_csv(ap), int(C.get("min_sections", 0)), C.get("ambiguity", "tags"))
    regions = [u for u in C.get("regions", ["whole_section"]) if u in set(ML["unit"])]
    vals, mem = classes.totals(ML, ann, class_cfg, ratio_cfg, regions)
    if vals.empty:
        S.warn("none of the configured classes were found among the tested peaks", log)
        return
    use_off = offsets if ctx.cfg["differential"].get("gradient_corrected", True) else None
    R = classes.test_features(vals, animals, ctl, mut, min_pairs, offsets=use_off)
    # same class tests on another normalization (decided in advance: a check, not a choice)
    for meth, MLa in (ML_alt or {}).items():
        va, _ = classes.totals(MLa, ann, class_cfg, ratio_cfg, regions)
        if va.empty:
            continue
        Ra = classes.test_features(va, animals, ctl, mut, min_pairs)[["unit", "feature", "log2fc", "p_ttest", "n_pairs_up"]]
        Ra.columns = ["unit", "feature", f"log2fc_{meth}_norm", f"p_{meth}_norm", f"n_pairs_up_{meth}_norm"]
        R = R.merge(Ra, on=["unit", "feature"], how="left")
    # sensitivity: the same tests with only the species that pass the feature QC (decided in
    # advance as a check; the primary analysis keeps the pre-specified class definitions)
    if QC is not None and "qc_pass" in QC:
        ok = set(QC.index[QC["qc_pass"]])
        pid = ann["peak_id"] if "peak_id" in ann else ann.index
        ann_qc = ann[pd.Series(pid, index=ann.index).isin(ok)]
        vq, mq = classes.totals(ML, ann_qc, class_cfg, ratio_cfg, regions)
        if not vq.empty:
            Rq = classes.test_features(vq, animals, ctl, mut, min_pairs)[["unit", "feature", "log2fc", "p_ttest", "n_pairs_up"]]
            Rq.columns = ["unit", "feature", "log2fc_featureqc", "p_featureqc", "n_pairs_up_featureqc"]
            R = R.merge(Rq, on=["unit", "feature"], how="left")
            if len(mq):
                R = R.merge(mq[["feature", "n_species"]].rename(columns={"n_species": "n_species_featureqc"}),
                            on="feature", how="left")
    if "confidence_tier" in ann and len(mem):
        pid = ann["peak_id"] if "peak_id" in ann else ann.index
        tier = dict(zip(pid, ann["confidence_tier"].fillna("")))
        mem["n_species_high_evidence"] = [sum(tier.get(p, "") == "high" for p in ids) for ids in mem["peak_ids"]] \
            if "peak_ids" in mem else np.nan
        R = R.merge(mem[["feature", "n_species_high_evidence"]], on="feature", how="left")
    big = float(ctx.cfg["differential"].get("large_effect_log2fc", 0.2))
    R["large_effect"] = R["log2fc"].abs() >= big
    # pre-specified primary endpoints: their own, small BH family
    prim = {tuple(x.strip() for x in str(e).split("|", 1)) for e in ((C.get("primary") or {}).get(mode) or [])}
    R["primary"] = [(u, f) in prim for u, f in zip(R["unit"], R["feature"])]
    R["q_primary"] = np.nan
    if R["primary"].any():
        R.loc[R["primary"], "q_primary"] = stats.bh(R.loc[R["primary"], "p_ttest"])
    missing = prim - set(zip(R["unit"], R["feature"]))
    if missing:
        S.warn(f"primary endpoints not found among the class tests: {sorted(missing)}", log)
    if PC.get("sensitivity", True) and flagged:
        R = R.merge(classes.sensitivity(vals, animals, ctl, mut, flagged, min_pairs), on=["unit", "feature"], how="left")
    R.to_csv(out("class_tests.csv"), index=False)
    mem.to_csv(out("class_members.csv"), index=False)
    vals.to_csv(out("class_values_per_animal.csv"), index=False)

    if R["primary"].any():
        S.h("Primary endpoints (pre-specified, their own BH family)")
        S.p("A short list of class-level outcomes named in advance in config.yaml (class_tests.primary). "
            "They are corrected only among themselves, so this is the family with real power at n = 6. "
            "Effect and its interval first; q_primary is the bar for calling a result.")
        pc = ["unit", "feature", "log2fc", "pct_change", "ci_low", "ci_high", "n_pairs_up", "n_pairs", "p_ttest",
              "q_primary", "log2fc_gradient_corrected", "p_gradient_corrected"] + \
             [c for c in R.columns if c.endswith("_norm") and c.startswith(("log2fc_", "p_"))] + \
             ["log2fc_Female", "log2fc_Male", "I2_sex"]
        S.table(R[R["primary"]][[c for c in pc if c in R]].round(4))
        S.kv({f"primary endpoints with q < {alpha}": int((R["q_primary"] < alpha).sum()),
              "primary endpoints tested": int(R["primary"].sum())})
    S.h("Class-level and ratio tests (pre-specified, one BH family for this mode)")
    S.p(f"{len(R)} tests ({R['feature'].nunique()} classes/ratios x {R['unit'].nunique()} regions). "
        f"Species per class are in class_members.csv. log2fc is the mean over slides of "
        "log2(Mutant/Control); pair_<slide> columns show each slide on its own.")
    pair_cols = [c for c in R.columns if c.startswith("pair_")]
    show = ["unit", "feature", "kind", "n_species", "log2fc", "pct_change", "ci_low", "ci_high", "n_pairs_up",
            "n_pairs", "p_ttest", "q_bh", "p_gradient_corrected", "I2_sex"] + pair_cols
    S.table(R.sort_values("p_ttest")[[c for c in show if c in R]].round(4), max_rows=30)
    S.kv({f"q < {alpha}": int((R["q_bh"] < alpha).sum()),
          f"q < {alpha} and |log2FC| >= {big:g}": int(((R["q_bh"] < alpha) & R["large_effect"]).sum()),
          "lowest q": float(R["q_bh"].min())})
    if "p_featureqc" in R:
        S.h("Class tests with feature-QC species only (sensitivity)")
        S.p("Same classes and ratios, built only from species that pass the Lipid Brain Atlas feature QC "
            "(spatially structured, tissue over matrix, one adduct per lipid). A check of the class "
            "results above, not a replacement: the class definitions were fixed in advance.")
        cq = ["unit", "feature", "n_species", "n_species_featureqc", "log2fc", "p_ttest", "log2fc_featureqc",
              "p_featureqc", "n_pairs_up_featureqc", "primary"]
        S.table(R.sort_values("p_ttest")[[c for c in cq if c in R]].round(4), max_rows=25)
    reaction_tests(ctx, mode, ML, ann, animals, ctl, mut, S, log, offsets=use_off)
    if flagged:
        tag = "+".join(flagged)
        S.p(f"Sensitivity (exploratory, slide(s) {tag} flagged above): the CSV has extra columns with "
            f"{tag} left out (`without_{tag}_*`) and flipped (`flipped_{tag}_*`). Use them to see how much "
            "the flagged slide matters, not as the result: choosing the analysis after seeing the data "
            "is circular. Only the records can decide whether the slide is swapped.")
        fc = f"flipped_{tag}_p_ttest"
        if fc in R:
            S.table(R.sort_values(fc)[["unit", "feature", "log2fc", "p_ttest", f"flipped_{tag}_log2fc", fc,
                                       f"flipped_{tag}_n_pairs_up", f"flipped_{tag}_q_bh", f"without_{tag}_p_ttest"]]
                    .round(4), max_rows=15)

    # dot plot: each slide's log2FC for a few chosen features, per region
    feats = [f for f in ((C.get("plot_features") or {}).get(mode) or []) if f in set(R["feature"])]
    if feats and pair_cols:
        regs = [u for u in regions if u in set(R["unit"])]
        fig, axes = plt.subplots(len(regs), 1, figsize=(1.2 * len(feats) + 2.2, 1.7 * len(regs) + 0.9), squeeze=False, sharex=True)
        slides = [c.replace("pair_", "") for c in pair_cols]
        for ax, u in zip(axes[:, 0], regs):
            sub = R[R["unit"] == u].set_index("feature")
            for i, f in enumerate(feats):
                if f not in sub.index:
                    continue
                for j, s in enumerate(slides):
                    v = sub.at[f, f"pair_{s}"]
                    mk = "D" if s in flagged else "o"
                    ax.scatter(i + (j - (len(slides) - 1) / 2) * 0.09, v, s=20, marker=mk,
                               color=SLIDE_COLOURS[j % len(SLIDE_COLOURS)], edgecolor="k" if s in flagged else "none",
                               label=f"Slide {s}" if (i == 0 and u == regs[0]) else None, zorder=3)
                if "log2fc" in sub and np.isfinite(sub.at[f, "log2fc"]):
                    ax.plot([i - 0.32, i + 0.32], [sub.at[f, "log2fc"]] * 2, color="k", lw=1.2, zorder=4)
            ax.axhline(0, color="0.5", lw=0.6)
            ax.set_ylabel(f"{figstyle.region_name(u)}\nlog2(KO / Control)", fontsize=7)
        axes[-1, 0].set_xticks(range(len(feats)), feats, rotation=20, ha="right")
        figstyle.title(fig, f"{figstyle.mode_name(mode)}: lipid classes, slide by slide",
                       f"Each dot is one slide's KO-minus-Control difference; bar, mean"
                       + (f"; diamonds, flagged slide {', '.join(flagged)}" if flagged else ""))
        fig.tight_layout(rect=(0, 0, 1, 1 - 0.3 / fig.get_size_inches()[1]))
        fig.legend(*axes[0, 0].get_legend_handles_labels(), ncol=len(slides), loc="lower center",
                   bbox_to_anchor=(0.5, 1 - 0.3 / fig.get_size_inches()[1]), handletextpad=0.2, columnspacing=1.0)
        S.fig(fig, "class_tests_per_slide.png", caption=(
            "Summed class intensities and class ratios, per slide pair. Each dot is log2(KO / Control) of the "
            "per-animal values on one slide (Control and KO sections from the same slide); the black bar is the mean "
            "over slides, the estimate tested by the paired t-test. Consistent sign across slides, rather than one "
            "extreme pair, is what supports a genotype effect."))

    # forest plot for whole section + white matter
    F = R[R["unit"].isin(["whole_section", "white_matter"]) & R["p_ttest"].notna()].sort_values(["unit", "kind", "feature"])
    if len(F):
        fig, ax = plt.subplots(figsize=(5.8, 0.2 * len(F) + 1.3))
        yy = np.arange(len(F))
        sig = (F["q_bh"] < float(ctx.cfg["differential"]["fdr_alpha"])).to_numpy() if "q_bh" in F else np.zeros(len(F), bool)
        for ok_, col in ((False, "0.55"), (True, figstyle.MUTANT)):
            m = sig == ok_
            fc_, lo_, hi_ = (F[c].to_numpy(float) for c in ("log2fc", "ci_low", "ci_high"))
            ax.errorbar(fc_[m], yy[m], xerr=[(fc_ - lo_)[m], (hi_ - fc_)[m]],
                        fmt="o", ms=3.5, color=col, ecolor=col, elinewidth=0.9, capsize=0,
                        label=("q < 0.05" if ok_ else "q \u2265 0.05"))
        ax.set_yticks(yy, F["feature"] + "  (" + F["unit"].map(figstyle.region_name) + ")", fontsize=6.5)
        ax.axvline(0, color="k", lw=0.5)
        ax.invert_yaxis()
        ax.margins(y=0.01)
        ax.set_xlabel("log2(KO / Control), mean and 95% CI over slide pairs")
        ax.legend(loc="lower right")
        figstyle.title(fig, f"{figstyle.mode_name(mode)}: lipid classes and ratios")
        S.fig(fig, "class_tests_forest.png", caption=(
            "Genotype effect on summed lipid classes and on class ratios in the whole section and white matter. "
            "Points, mean of the per-slide log2(KO / Control) differences of per-animal values; bars, 95% confidence "
            "interval from the paired t-test over slides. Vermillion, Benjamini-Hochberg q < 0.05 within the "
            "class-test family of this mode."))


def reaction_tests(ctx, mode, ML, ann, animals, ctl, mut, S, log, offsets=None):
    """Per-backbone precursor -> product ratios (class_tests.reactions), their own BH family."""
    RC = (ctx.cfg.get("class_tests") or {}).get("reactions") or {}
    steps = RC.get("steps") or []
    if not steps:
        return
    regions = [u for u in RC.get("regions", ["whole_section"]) if u in set(ML["unit"])]
    vals, pairs = classes.reaction_ratios(ML, ann, steps, regions)
    S.h("Precursor -> product ratios per backbone (exploratory, own BH family)")
    if vals.empty:
        S.p("No backbone has both the precursor and the product among the usable species in this mode.")
        return
    min_pairs = int(ctx.cfg["differential"].get("min_pairs", 5))
    R = classes.test_features(vals, animals, ctl, mut, min_pairs, offsets=offsets)
    R = R.merge(pairs, on="feature", how="left")
    big = float(ctx.cfg["differential"].get("large_effect_log2fc", 0.2))
    R["large_effect"] = R["log2fc"].abs() >= big
    R.to_csv(ctx.out("results", "09_differential", mode, "reaction_tests.csv"), index=False)
    S.p("Each ratio is product / precursor with the same ceramide backbone (same chains), within the same "
        "animal and region, so it indexes one enzyme step; comparing log ratios between genotypes cancels "
        "the two lipids' different ionization. Steps and regions were set in config before running. "
        "q_bh is over this table only.")
    show = ["unit", "feature", "enzyme", "log2fc", "pct_change", "ci_low", "ci_high", "n_pairs_up", "n_pairs",
            "p_ttest", "q_bh", "large_effect", "p_gradient_corrected"]
    S.table(R.sort_values("p_ttest")[[c for c in show if c in R]].round(4), max_rows=40)
    alpha = float(ctx.cfg["differential"]["fdr_alpha"])
    S.kv({"ratios tested": int(R["p_ttest"].notna().sum()), f"q < {alpha}": int((R["q_bh"] < alpha).sum())})


def region_qc(ctx, obs, units, core, animals, S, log):
    """Per section: share of its pixels in each atlas region vs the median section. A region share
    far from the others usually means the registration is off for that section. Warn, and leave
    sections listed in differential.exclude_sections_from_regions (e.g. ["K:Top"]) out of the
    atlas-region tests (whole section and territories keep them)."""
    D = ctx.cfg["differential"]
    lo, hi = D.get("region_fraction_flag", [0.6, 1.67])
    rows = []
    for (kind, name), m in units.items():
        if kind != "region" or name == "whole_section":
            continue
        share = pd.Series(m & core).groupby(obs["animal_id"].to_numpy()).mean() / \
            pd.Series(core).groupby(obs["animal_id"].to_numpy()).mean()
        rel = share / share.median()
        for aid, r in rel.items():
            if np.isfinite(r) and (r < lo or r > hi):
                rows.append({"animal_id": aid, "slide": animals.loc[aid, "slide"], "section": animals.loc[aid, "section"],
                             "region": name, "share_of_section": round(float(share[aid]), 3),
                             "vs_median_section": round(float(r), 2)})
    if rows:
        T = pd.DataFrame(rows)
        S.warn(f"{len(T)} region/section combinations have a region share outside {lo}-{hi}x the median section "
               "(possible registration problem; check the overlays). Listed under 'Region size check'.", log)
        S.h("Region size check (atlas region share of each section vs the median section)")
        S.table(T)
    excl = [str(e) for e in (D.get("exclude_sections_from_regions") or [])]
    ex_animals = [aid for aid in animals.index if f"{animals.loc[aid, 'slide']}:{animals.loc[aid, 'section']}" in excl]
    if ex_animals:
        S.p(f"Excluded from atlas-region tests (differential.exclude_sections_from_regions): {', '.join(excl)} "
            f"= {', '.join(ex_animals)}. Their slide pairs have no value in those regions.")
    return ex_animals, obs["animal_id"].isin(ex_animals).to_numpy()


def main():
    t0 = time.time()
    ctx, log, modes, _ = cli.start("09_differential", __doc__)
    D = ctx.cfg["differential"]
    G = ctx.cfg["project"]["genotypes"]
    ctl, mut = G["control"], G["mutant"]
    alpha = float(D["fdr_alpha"])
    flag_ratio = float(ctx.get("qc.gradient.flag_ratio", 2.0))
    for mode in modes:
        S = Summary(ctx, "09_differential", mode, f"Differential analysis, {mode}: {ctx.cfg['project']['mutant_label']} vs Control")
        p = ctx.out("processed", mode, "matrix_norm.h5ad", mkdir=False)
        if not p.exists():
            S.warn("no matrix_norm.h5ad, run 06 first", log)
            S.write()
            continue
        a = read_h5ad(p)
        obs, var = a.obs, a.var
        iso = var.get("is_isotope", pd.Series(False, index=var.index)).fillna(False).astype(bool).to_numpy()
        test_cols = np.flatnonzero(~iso)
        family = D.get("untargeted_peaks", "lamanno")
        annotated = var.get("annotation", pd.Series("", index=var.index)).fillna("").astype(str) != ""
        mp = ctx.output_root / "results" / "08_explore" / mode / "morans_i.csv"
        structured = np.zeros(len(var), bool)
        if mp.exists():
            mor = pd.read_csv(mp)
            structured = var.index.isin(mor.loc[mor["selected"], "peak"])
        elif family != "all":
            S.warn("no 08 Moran's I results: no peak counts as spatially structured", log)
        # feature QC (Lipid Brain Atlas rules); decides the untargeted family when family = lamanno
        FQ = D.get("feature_qc") or {}
        add = featureqc.adduct_choice(var, a.X, np.arange(len(var)), float(FQ.get("min_adduct_r", 0.5)),
                                      int(FQ.get("max_pixels_for_r", 50000)))
        QC = featureqc.table(var, pd.Series(structured, index=var.index), add, float(FQ.get("max_off_on_ratio", 1.0)))
        QC.to_csv(ctx.out("results", "09_differential", mode, "feature_qc.csv"))
        if family != "all":
            keep = (~iso) & (annotated.to_numpy() | np.asarray(structured))
            log.info(f"{mode}: untargeted family = {int(keep.sum())} peaks "
                     f"({int((annotated & ~pd.Series(iso, index=var.index)).sum())} annotated, "
                     f"{int(np.asarray(structured).sum())} spatially structured) of {int((~iso).sum())} non-isotope peaks")
            test_cols = np.flatnonzero(keep)
        X = np.asarray(a.X[:, test_cols], dtype=np.float64)
        peaks = var.index[test_cols]
        core = obs["core"].to_numpy().astype(bool)

        units = {}
        rp = ctx.out("processed", mode, "regions.parquet", mkdir=False)
        if rp.exists():
            reg = pd.read_parquet(rp).loc[obs.index]
            atlas_cols = [c for c in reg.columns if c.startswith("region_")
                          and c not in ("region_whole_section", "region_wm_lipid", "region_gm_lipid")]
            have_atlas = any(reg[c].any() for c in atlas_cols)
            for c in [c for c in reg.columns if c.startswith("region_")]:
                if c in ("region_wm_lipid", "region_gm_lipid") and have_atlas \
                        and D.get("lipid_wm_gm", "fallback_only") == "fallback_only":
                    continue
                units[("region", c.replace("region_", ""))] = reg[c].to_numpy().astype(bool)
        else:
            S.warn("no regions.parquet (run 07): only whole sections are tested", log)
            units[("region", "whole_section")] = np.ones(len(obs), bool)
        tp = ctx.out("processed", mode, "territories.parquet", mkdir=False)
        if D.get("run_territories", True) and tp.exists():
            ter = pd.read_parquet(tp).loc[obs.index, "territory"].to_numpy()
            for t in np.unique(ter):
                units[("territory", f"T{t}")] = ter == t

        animals = obs.drop_duplicates("animal_id").set_index("animal_id")[["slide", "genotype", "sex", "section"]]
        g_med, g_slide = gradient_prediction(ctx, mode, obs, mut)
        excl_animals, excl_px = region_qc(ctx, obs, units, core, animals, S, log)
        use_moderated = D.get("single_peak_test", "moderated") == "moderated"
        offsets = g_slide if (D.get("gradient_corrected", True) and len(g_slide)) else None
        alt_methods = [m for m in (D.get("normalization_sensitivity") or []) if f"size_factor_{m}" in obs
                       and "size_factor_tic" in obs]
        class_units = set((ctx.cfg.get("class_tests") or {}).get("regions") or [])
        log.info(f"{mode}: {len(peaks)} peaks x {len(units)} units; gradient-predicted log2(Mut/Ctl) = {g_med:.3f}")

        results, means_long = [], []
        alt_long = {m: [] for m in alt_methods}
        for (kind, name), m in units.items():
            mask = m & core
            if kind == "region" and name != "whole_section" and excl_px.any():
                mask = mask & ~excl_px          # sections excluded from atlas regions (bad registration)
            if mask.sum() == 0:
                continue
            means, npx = stats.per_animal_means(X, obs, mask, int(D["min_pixels_per_unit"]))
            means.columns = peaks
            if means.notna().any(axis=1).sum() < 4:
                log.info(f"  {kind} {name}: fewer than 4 animals with enough pixels, skipped")
                continue
            c = stats.pseudocount(means)
            log2 = np.log2(means + c)
            res = stats.paired_tests(log2, animals, ctl, mut, run_mixed=bool(D.get("run_mixedlm", False)),
                                     min_pairs=int(D.get("min_pairs", 5)), moderated=use_moderated,
                                     offsets=offsets)
            res.insert(0, "unit", name)
            res.insert(0, "unit_kind", kind)
            res["median_pixels_per_animal"] = float(npx.median())
            results.append(res)
            ml = means.stack().rename("mean_norm_intensity").reset_index()
            ml.columns = ["animal_id", "peak", "mean_norm_intensity"]
            ml.insert(0, "unit", name)
            ml["n_pixels"] = ml["animal_id"].map(npx)
            means_long.append(ml)
            # per-animal means under another normalization, for the class-test sensitivity check
            if kind == "region" and name in class_units:
                for meth in alt_methods:
                    f = (obs["size_factor_tic"].to_numpy(float) / obs[f"size_factor_{meth}"].to_numpy(float))
                    ma, _ = stats.per_animal_means(X * f[:, None], obs, mask, int(D["min_pixels_per_unit"]))
                    ma.columns = peaks
                    mla = ma.stack().rename("mean_norm_intensity").reset_index()
                    mla.columns = ["animal_id", "peak", "mean_norm_intensity"]
                    mla.insert(0, "unit", name)
                    alt_long[meth].append(mla)
        if not results:
            S.warn("nothing could be tested", log)
            S.write()
            continue
        R = pd.concat(results, ignore_index=True)
        ann_cols = [c for c in ("mz", "label", "annotation", "adduct", "lipid_class", "tag", "hypothesis",
                                "ambiguous", "typeII_flag", "confidence_tier", "evidence_score", "candidates")
                    if c in var]
        R = R.join(var[ann_cols], on="peak")
        R["targeted"] = R.get("hypothesis", pd.Series("", index=R.index)).fillna("").astype(str) != ""
        # the p-value that decides significance: the moderated t (default) or the plain paired t
        R["p_primary"] = R["p_moderated"] if (use_moderated and "p_moderated" in R) else R["p_ttest"]
        R["primary_test"] = "moderated t" if use_moderated else "paired t"
        R["q_all"] = np.nan
        R["q_targeted"] = np.nan
        R["q_all_ttest"] = np.nan
        R["in_untargeted_family"] = R["peak"].map(QC["qc_pass"]).fillna(False).astype(bool) \
            if family == "lamanno" else True
        for _, idx in R.groupby("unit").groups.items():
            f_idx = [i for i in idx if R.loc[i, "in_untargeted_family"]]
            if f_idx:
                R.loc[f_idx, "q_all"] = stats.bh(R.loc[f_idx, "p_primary"])
                R.loc[f_idx, "q_all_ttest"] = stats.bh(R.loc[f_idx, "p_ttest"])
            t_idx = [i for i in idx if R.loc[i, "targeted"]]
            if t_idx:
                R.loc[t_idx, "q_targeted"] = stats.bh(R.loc[t_idx, "p_primary"])
        exp_dir = R.get("hypothesis", pd.Series("", index=R.index)).fillna("")
        R["matches_hypothesis"] = np.where(exp_dir == "accumulated", R["log2fc"] > 0,
                                           np.where(exp_dir == "depleted", R["log2fc"] < 0,
                                                    np.where(exp_dir == "both", True, np.nan)))
        R["gradient_pred_log2fc"] = g_med
        if np.isfinite(g_med) and g_med != 0:
            same = np.sign(R["log2fc"]) == np.sign(g_med)
            R["could_be_position"] = same & (R["log2fc"].abs() < flag_ratio * abs(g_med))
        else:
            R["could_be_position"] = np.nan
        R["significant"] = (np.where(R["targeted"], R["q_targeted"], R["q_all"]) < alpha)
        big = float(D.get("large_effect_log2fc", 0.2))
        R["large_effect"] = R["log2fc"].abs() >= big
        R = R.sort_values(["unit_kind", "unit", "p_primary"])
        R.to_csv(ctx.out("results", "09_differential", mode, "differential.csv"), index=False)
        ML = pd.concat(means_long, ignore_index=True)
        ML.to_csv(ctx.out("results", "09_differential", mode, "per_animal_means.csv"), index=False)

        # ---- volcanoes ------------------------------------------------------------------
        reg_units = [u for k, u in units if k == "region" and u in set(R.unit)]
        reg_units = sorted(reg_units, key=lambda u: (u != "whole_section", u != "white_matter", u))
        ncol = min(4, len(reg_units))
        nrow = int(np.ceil(len(reg_units) / max(ncol, 1)))
        fig, axes = plt.subplots(nrow, ncol, figsize=(3.1 * ncol, 2.9 * nrow + 0.5), squeeze=False,
                                 gridspec_kw={"wspace": 0.4, "hspace": 0.5})
        for ax, u in zip(axes.ravel(), reg_units):
            sub = R[R.unit == u]
            if np.isfinite(g_med):
                ax.axvspan(min(0, flag_ratio * g_med), max(0, flag_ratio * g_med), color=figstyle.OKABE_ITO[4],
                           alpha=0.18, lw=0)
            col = np.where(sub["significant"], np.where(sub["log2fc"] > 0, figstyle.MUTANT, figstyle.CONTROL), "0.75")
            ax.scatter(sub["log2fc"], -np.log10(sub["p_primary"].clip(lower=1e-300)), c=col, s=9,
                       edgecolor=np.where(sub["targeted"], "k", "none"), linewidth=0.4, rasterized=True)
            xs = sub["log2fc"].abs().max()
            ax.set_xlim(-1.15 * xs if np.isfinite(xs) and xs > 0 else -1, 1.15 * xs if np.isfinite(xs) and xs > 0 else 1)
            yr = -np.log10(sub["p_primary"].clip(lower=1e-300)).max() or 1.0
            placed = []
            for _, r in sub[sub["significant"]].nsmallest(5, "p_primary").iterrows():
                right = r["log2fc"] > 0           # label toward the centre, so it stays inside the panel
                yv = -np.log10(max(r["p_primary"], 1e-300))
                if any(sd == right and abs(yv - yp) < 0.045 * yr for sd, yp in placed):
                    continue                      # would sit on top of a label already drawn
                placed.append((right, yv))
                ax.annotate(str(r.get("label", r["peak"])).replace(" *", "")[:22],
                            (r["log2fc"], -np.log10(max(r["p_primary"], 1e-300))), fontsize=5,
                            xytext=(-3 if right else 3, 2), textcoords="offset points",
                            ha="right" if right else "left", annotation_clip=False)
            ax.axvline(0, color="k", lw=0.5)
            ax.set_title(f"{figstyle.region_name(u)} ({int(sub['significant'].sum())} at q < {alpha:g})")
            ax.set_xlabel("log2(KO / Control)")
            ax.set_ylabel(f"\u2212log10 p ({R['primary_test'].iloc[0]})")
        for ax in axes.ravel()[len(reg_units):]:
            ax.axis("off")
        figstyle.title(fig, f"{figstyle.mode_name(mode)}: single-ion differences by region")
        S.fig(fig, "volcano_regions.png", caption=(
            f"Volcano plots of every tested ion in each atlas region. x, mean of per-slide log2(KO / Control) of "
            f"per-animal mean intensity (n = 6 slide pairs); y, \u2212log10 p of the {R['primary_test'].iloc[0]} test. "
            f"Coloured points, Benjamini-Hochberg q < {alpha:g} within the region (hypothesis-class ions in their "
            "own targeted family, others in the untargeted family): vermillion higher, blue lower in KO. Black "
            f"outline, hypothesis-class ions. Shaded band, effects smaller than {flag_ratio:g} times the technical "
            "Top-to-Bottom trend, which position alone could produce."))

        # hypothesis classes: forest plot in whole section + white matter
        hyp = R[R["targeted"] & R.unit.isin(["whole_section", "white_matter", "corpus_callosum"])
                & R["p_ttest"].notna()].sort_values(["unit", "lipid_class", "mz"] if "lipid_class" in R else ["unit"])
        if len(hyp):
            fig, ax = plt.subplots(figsize=(6, 0.2 * len(hyp) + 1.4))
            yl = hyp["label"].astype(str) + "  (" + hyp["unit"].map(figstyle.region_name) + ")"
            yy = np.arange(len(hyp))
            if np.isfinite(g_med):
                ax.axvspan(min(0, flag_ratio * g_med), max(0, flag_ratio * g_med), color=figstyle.OKABE_ITO[4],
                           alpha=0.18, lw=0)
            sg = hyp["significant"].to_numpy()
            for ok_, col in ((False, "0.55"), (True, figstyle.MUTANT)):
                m = sg == ok_
                fc_, lo_, hi_ = (hyp[c].to_numpy(float) for c in ("log2fc", "ci_low", "ci_high"))
                ax.errorbar(fc_[m], yy[m], xerr=[(fc_ - lo_)[m], (hi_ - fc_)[m]],
                            fmt="o", ms=3.5, color=col, ecolor=col, elinewidth=0.9, label=f"q < {alpha:g}" if ok_ else f"q \u2265 {alpha:g}")
            ax.set_yticks(yy, yl, fontsize=6)
            ax.invert_yaxis()
            ax.margins(y=0.01)
            ax.axvline(0, color="k", lw=0.5)
            ax.set_xlabel("log2(KO / Control), mean and 95% CI over slide pairs")
            ax.legend(loc="lower right")
            figstyle.title(fig, f"{figstyle.mode_name(mode)}: hypothesis-class ions")
            S.fig(fig, "forest_hypothesis.png", caption=(
                "Genotype effect on each ion of the pre-specified hypothesis classes (sphingolipids, gangliosides, "
                "BMP/PG and others) in the whole section, white matter and corpus callosum. Points, mean per-slide "
                "log2(KO / Control); bars, 95% confidence interval (paired t over slides). Vermillion, q < "
                f"{alpha:g} in the targeted family. Shaded band, within {flag_ratio:g} times the technical position trend."))

        # ---- summary --------------------------------------------------------------------
        S.h("Design")
        pairs = R["pairs"].iloc[0]
        S.p(f"Pairs (slides): {pairs}. Unit = animal; mean of TIC-normalized intensity over the region's "
            "non-edge pixels, log2, Mutant minus Control within slide.")
        S.p(f"Position gradient: mean over slides of the predicted log2(Mutant/Control) from the off-tissue "
            "DAN trend (sign per slide from where the Mutant sits) = "
            f"{g_med:.3f}" + ("" if np.isfinite(g_med) else " (not available)") +
            f". Hits with the same sign and |log2FC| < {flag_ratio:g}x that are flagged `could_be_position`.")
        # effect and interval first; Wilcoxon/sign stay in the CSV (their 0.031 floor makes them
        # uninformative in a table with n = 6)
        show = ["unit", "label", "hypothesis", "log2fc", "ci_low", "ci_high", "n_pairs_up", "n_pairs",
                "p_primary", "q_targeted", "large_effect", "p_ttest", "p_gradient_corrected", "matches_hypothesis",
                "could_be_position", "ambiguous", "confidence_tier"]
        show = [c for c in show if c in R]
        S.h("1. Hypothesis classes (targeted, BH within region over tagged peaks)")
        T = R[R["targeted"]].sort_values(["p_primary"])
        S.table(T[show], max_rows=40)
        S.h("2. Untargeted: top hits per region (BH over all peaks in the region)")
        U = R[(R.unit_kind == "region") & R["in_untargeted_family"]].sort_values("p_primary").groupby("unit").head(5)
        S.table(U[[c if c != "q_targeted" else "q_all" for c in show]], max_rows=60)
        if (R.unit_kind == "territory").any():
            S.h("3. Territories: top hits")
            Tt = R[R.unit_kind == "territory"].sort_values("p_primary").head(15)
            S.table(Tt[[c if c != "q_targeted" else "q_all" for c in show]])
        S.h("Feature QC (Lipid Brain Atlas rules) and the untargeted family")
        S.p("Rules: monoisotopic, annotated, spatially structured (Moran's I >= "
            f"{float(ctx.get('explore.morans_i.min_I', 0.4)):g} in at least half the sections), more signal on tissue "
            f"than on bare glass (off/on < {float(FQ.get('max_off_on_ratio', 1.0)):g}), one adduct per lipid. "
            + ("Only peaks passing all of them get a q_all; hypothesis-class peaks keep their own family."
               if family == "lamanno" else f"untargeted_peaks = {family}: the QC is reported but not used."))
        mono = QC[QC["monoisotopic"]]
        S.kv({"monoisotopic peaks": len(mono), "annotated": int(mono["annotated"].sum()),
              "spatially structured": int(mono["spatially_structured"].sum()),
              "tissue over matrix": int(mono["tissue_over_matrix"].sum()),
              "redundant adducts dropped": int((mono["adduct_status"] == "redundant").sum()),
              "adducts kept although not correlated with their partner": int((mono["adduct_status"] == "kept, not correlated").sum()),
              "pass all (untargeted family)": int(mono["qc_pass"].sum())})
        S.h("Counts")
        sig = R[R["significant"]]
        S.kv({"tests": len(R), f"significant (q < {alpha})": len(sig),
              f"significant and |log2FC| >= {big:g} (large_effect)": int((sig["large_effect"]).sum()),
              "of which flagged could_be_position": int(sig["could_be_position"].fillna(False).astype(bool).sum()),
              "units": R.unit.nunique()})
        if "p_sex_difference_exploratory" in R:
            S.h("Sex (exploratory only)")
            S.p("Females were run in runs 1-3, all males in run 4, so sex and run are completely confounded. "
                "Sex is also confounded with slide orientation: on the female slides the Control is the Top "
                "section, on the male slides the CLN3 KO is, so a technical top-to-bottom trend would add to the "
                "female-versus-male difference. `p_sex_difference_exploratory` compares the per-slide log2FC of "
                "3 female vs 3 male pairs; treat it as a hint for follow-up, not a result.")
            Sx = R[R.unit == "whole_section"].nsmallest(10, "p_sex_difference_exploratory")
            S.table(Sx[[c for c in ("label", "log2fc", "log2fc_Female", "log2fc_Male",
                                     "p_sex_difference_exploratory") if c in Sx]].round(4))
        ML_alt = {m: pd.concat(v, ignore_index=True) for m, v in alt_long.items() if v}
        class_level(ctx, mode, ML, animals, ctl, mut, S, log, offsets=offsets, ML_alt=ML_alt, QC=QC)
        log.info(f"summary: {S.write()}")
    cli.finish(log, t0)


if __name__ == "__main__":
    main()
