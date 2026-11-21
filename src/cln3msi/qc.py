"""One-slide QC, shared by 02_qc_one_slide.py (one slide) and 03_qc_all.py (every slide).

Two streaming passes over the slide:
  pass 1  per-pixel TIC and DAN matrix-ion intensity  -> tissue mask, Top/Bottom sections
  pass 2  mean spectrum of the Top section, the Bottom section and the off-tissue area

Then, from those mean spectra: mass accuracy on calibrant lipids, which DAN ions are there,
resolving power from single profile spectra, and the top-to-bottom gradient check.

Saved for later phases (04 reuses them, so it doesn't have to re-read everything):
  processed/<mode>/qc/sections_<slide>.npz   section labels per pixel and per spectrum
  processed/<mode>/qc/meanspec_<slide>.npz   axis + mean spectra (Top, Bottom, off)
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import yaml
from scipy import stats

from . import chem, figstyle, samples, spectra
from .config import REPO_ROOT
from .imzml_io import SlideReader, detect_profile
from .report import Summary, decimate, plt
from .tissue import BOTTOM, POSITION_NAME, TOP, segment_slide

CAL = yaml.safe_load((REPO_ROOT / "reference" / "calibrants.yaml").read_text())


def matrix_ions(ctx, polarity: str, mz_lo: float, mz_hi: float) -> pd.DataFrame:
    f = ctx.get("acquisition.matrix_formula", "C10H10N2")
    rows = [{"name": f"DAN {a}", "formula": f, "adduct": a, "mz": chem.ion_mz(f, a)}
            for a in CAL["matrix"][polarity]]
    df = pd.DataFrame(rows)
    return df[(df.mz > mz_lo) & (df.mz < mz_hi)].reset_index(drop=True)


def calibrants(analyte: str, polarity: str, mz_lo: float, mz_hi: float) -> pd.DataFrame:
    rows = CAL["mass_accuracy"].get(analyte, {}).get(polarity, [])
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["mz"] = [chem.ion_mz(f, a) for f, a in zip(df.formula, df.adduct)]
    return df[(df.mz > mz_lo) & (df.mz < mz_hi)].reset_index(drop=True)


def spectrum_type_for(ctx, mode: str, slide: str, reader: SlideReader, log) -> str:
    """Profile or centroid, from 01's files.csv if available, otherwise detected now."""
    f = ctx.output_root / "metadata" / "files.csv"
    if f.exists():
        df = pd.read_csv(f)
        hit = df[(df["mode"] == mode) & (df["slide"] == slide)]
        if len(hit) and isinstance(hit["spectrum_type_data"].iloc[0], str):
            return hit["spectrum_type_data"].iloc[0]
    rng = np.random.default_rng(0)
    g = [detect_profile(*reader.spectrum(int(i))) for i in rng.choice(reader.n, min(20, reader.n), replace=False)]
    t = "profile" if np.nanmedian([x["shoulder_fraction"] for x in g]) > 0.5 else "centroid"
    log.info(f"spectrum type detected from data: {t}")
    return t


def mz_range(reader: SlideReader) -> tuple[float, float]:
    if reader.shares_mz_axis:
        a = reader.mz_axis()
        return float(a.min()), float(a.max())
    lo, hi = np.inf, -np.inf
    for i in np.linspace(0, reader.n - 1, min(300, reader.n)).astype(int):
        m = reader.read_mz(int(i))
        if len(m):
            lo, hi = min(lo, m.min()), max(hi, m.max())
    return float(lo), float(hi)


def theil_slope(y: np.ndarray, x: np.ndarray) -> tuple[float, float, float]:
    ok = np.isfinite(y) & np.isfinite(x)
    if ok.sum() < 10 or np.ptp(x[ok]) == 0:
        return np.nan, np.nan, np.nan
    if ok.sum() > 3000:   # Theil-Sen is O(n^2); a random subset is plenty
        idx = np.random.default_rng(0).choice(np.flatnonzero(ok), 3000, replace=False)
    else:
        idx = np.flatnonzero(ok)
    res = stats.theilslopes(y[idx], x[idx])
    rho = stats.spearmanr(x[idx], y[idx]).statistic
    return float(res.slope), float(res.intercept), float(rho)


def run_slide_qc(ctx, log, mode: str, slide: str, phase: str, sheet: pd.DataFrame,
                 S: Summary | None = None, make_figures: bool = True) -> dict:
    d = ctx.mode_def(mode)
    path = samples.files_for_mode(ctx, mode, sheet)[slide]
    log.info(f"=== QC {mode} slide {slide}: {path.name}")
    r = SlideReader(path, log)
    stype = spectrum_type_for(ctx, mode, slide, r, log)
    lo, hi = mz_range(r)
    log.info(f"m/z range {lo:.2f} to {hi:.2f}, spectrum type {stype}, {r.n} spectra")
    search = float(ctx.get("qc.calibrant_search_ppm", 50))
    chunk = int(ctx.get("runtime.spectra_chunk", 2000))
    every = int(ctx.get("runtime.log_memory_every", 10000))

    # ---- pass 1: TIC + matrix images ------------------------------------------------------
    mx = matrix_ions(ctx, d["polarity"], lo, hi)
    mlo, mhi = spectra.ppm_windows(mx["mz"].to_numpy(), search)
    tic, win = spectra.stream_pixel_sums(r, mlo, mhi, chunk, log, every)
    matrix_sum = win.sum(axis=1) if win.shape[1] else np.zeros(r.n)
    nonmatrix = np.clip(tic - matrix_sum, 0, None)
    log.mem(f"pass 1 done ({slide})")

    signal_name = ctx.get("qc.tissue.signal", "nonmatrix_tic")
    sig_img = r.to_image(nonmatrix if signal_name == "nonmatrix_tic" else tic)
    seg = segment_slide(sig_img, slide, ctx.cfg["qc"])
    pos_img, core_img = seg["positions"], seg["core"]
    yx = (r.coords[:, 1] - 1, r.coords[:, 0] - 1)
    pos_spec = pos_img[yx]          # 0 off-tissue, 1 Top, 2 Bottom, for each spectrum
    core_spec = core_img[yx] > 0
    if seg["n_sections"] != ctx.get("qc.sections.n_expected", 2) and S is not None:
        S.warn(f"slide {slide}: found {seg['n_sections']} sections instead of 2 ({seg['how']}). "
               "Set qc.sections.manual_split for this slide if needed.", log)

    sec_tab = sheet[sheet.slide == slide].set_index("position_on_slide")
    animal_of = {TOP: sec_tab.loc["Top", "animal_id"], BOTTOM: sec_tab.loc["Bottom", "animal_id"]}
    geno_of = {TOP: sec_tab.loc["Top", "genotype"], BOTTOM: sec_tab.loc["Bottom", "genotype"]}

    np.savez_compressed(ctx.out("processed", mode, "qc", f"sections_{slide}.npz"),
                        positions=pos_img, core=core_img, pos_spec=pos_spec, core_spec=core_spec,
                        coords=r.coords, tic=tic, matrix=matrix_sum, mask=seg["mask"], split_axis=seg["axis"])

    # ---- pass 2: mean spectra -------------------------------------------------------------
    labels = np.where(pos_spec == TOP, "Top", np.where(pos_spec == BOTTOM, "Bottom", "off")).astype(object)
    groups = [g for g in ("Top", "Bottom", "off") if (labels == g).any()]
    grid = None
    if not r.shares_mz_axis:
        bin_ppm = float(ctx.get("qc.bin_ppm", 2.0))
        if stype == "profile":
            # never coarser than the data's own point spacing, or peak shapes get lost
            sp = [np.median(np.diff(m) / m[:-1]) * 1e6 for m in
                  (r.read_mz(int(i)) for i in np.linspace(0, r.n - 1, 20).astype(int)) if len(m) > 2]
            bin_ppm = min(bin_ppm, float(np.median(sp))) if sp else bin_ppm
        grid = spectra.PpmGrid(lo, hi, bin_ppm)
        log.info(f"processed data: mean spectra on a {bin_ppm:.2f} ppm grid ({grid.n} bins, "
                 f"{'interpolated' if stype == 'profile' else 'summed'})")
    axis, means = spectra.stream_mean_spectra(r, labels, groups, grid, chunk, log, every,
                                              profile=stype == "profile")
    np.savez_compressed(ctx.out("processed", mode, "qc", f"meanspec_{slide}.npz"), axis=axis,
                        spectrum_type=stype, binned=grid is not None, **means)
    log.mem(f"pass 2 done ({slide})")

    nw = int(ctx.get("peaks.noise_window", 401))

    # ---- mass accuracy ------------------------------------------------------------------
    cal = calibrants(d["analyte"], d["polarity"], lo, hi)
    cal_rows = []
    for _, c in cal.iterrows():
        for g in ("Top", "Bottom"):
            if g not in means:
                continue
            hit = spectra.find_ion(axis, means[g], c.mz, search, nw)
            cal_rows.append({"slide": slide, "section": g, "name": c["name"], "adduct": c.adduct,
                             "theoretical_mz": c.mz, **({"mz": np.nan, "ppm_error": np.nan, "snr": 0,
                                                         "fwhm_ppm": np.nan} if hit is None else hit)})
    cal_df = pd.DataFrame(cal_rows)
    min_snr = float(ctx.get("qc.calibrant_min_snr", 10))
    good = cal_df[cal_df.snr >= min_snr] if len(cal_df) else cal_df
    med = float(good.ppm_error.median()) if len(good) else np.nan
    sd = float(good.ppm_error.std()) if len(good) > 2 else np.nan
    mad = float(stats.median_abs_deviation(good.ppm_error, scale="normal")) if len(good) > 2 else np.nan

    # ---- matrix ions -------------------------------------------------------------------
    mrows = []
    for _, m in mx.iterrows():
        rec = {"name": m["name"], "theoretical_mz": m.mz}
        for g in groups:
            h = spectra.find_ion(axis, means[g], m.mz, search, nw)
            rec[f"snr_{g}"] = h["snr"] if h else 0.0
            rec[f"height_{g}"] = h["height"] if h else 0.0
            if h and g == "off":
                rec["ppm_error_off"] = h["ppm_error"]
        mrows.append(rec)
    mx_df = pd.DataFrame(mrows)

    # ---- resolving power (profile only) -------------------------------------------------
    res = {"note": "centroid data: peak widths are not available"}
    widths = pd.DataFrame()
    if stype == "profile":
        tis = np.flatnonzero(core_spec)
        k = int(ctx.get("qc.n_resolution_spectra", 20))
        pick = tis[np.argsort(tic[tis])[::-1][:k]] if len(tis) else []
        rows = []
        for i in pick:
            mz_i, it_i = r.spectrum(int(i))
            pk = spectra.pick_peaks(mz_i, it_i.astype(float), snr=20, noise_window=nw)
            spacing = np.median(np.diff(mz_i))
            ok = pk["fwhm"] > 2 * spacing          # need a few points across the peak
            rows += [{"mz": a, "fwhm": b} for a, b in zip(pk["mz"][ok], pk["fwhm"][ok])]
        widths = pd.DataFrame(rows)
        res = spectra.fit_resolution(widths["mz"].to_numpy(), widths["fwhm"].to_numpy()) if len(widths) else res

    # ---- gradient between the two section positions (the position confound check) -------
    # Measured along the direction that separates the sections: image rows if they sit one
    # above the other, image columns if they sit side by side.
    rows_spec = (r.coords[:, 1] if seg["axis"] == "rows" else r.coords[:, 0]).astype(float)
    c_top = float(np.mean(rows_spec[pos_spec == TOP])) if (pos_spec == TOP).any() else np.nan
    c_bot = float(np.mean(rows_spec[pos_spec == BOTTOM])) if (pos_spec == BOTTOM).any() else np.nan
    grad = {"split_axis": seg["axis"], "top_center_row": c_top, "bottom_center_row": c_bot,
            "n_off_pixels": int((pos_spec == 0).sum())}
    off = pos_spec == 0
    for name, vals in (("matrix_off", matrix_sum), ("tic_off", tic)):
        if off.sum() >= 20 and np.nanmax(vals[off]) > 0:
            v = np.log2(vals[off] + 1)
            slope, icpt, rho = theil_slope(v, rows_spec[off])
            grad[f"{name}_slope_log2_per_row"] = slope
            grad[f"{name}_spearman"] = rho
            # Bottom minus Top, in log2 units, if the off-tissue trend applied to the sections
            grad[f"{name}_pred_log2_bottom_vs_top"] = slope * (c_bot - c_top)
    on = pos_spec > 0
    for g, code in (("tic_tissue", None),):
        if on.sum() >= 20:
            v = np.log2(tic[on] + 1)
            slope, _, rho = theil_slope(v, rows_spec[on])
            grad["tic_tissue_slope_log2_per_row"], grad["tic_tissue_spearman"] = slope, rho
    if matrix_sum.any() and on.sum() >= 20:
        v = np.log2(matrix_sum[on] + 1)
        slope, _, rho = theil_slope(v, rows_spec[on])
        grad["matrix_tissue_slope_log2_per_row"], grad["matrix_tissue_spearman"] = slope, rho

    sec_stats = []
    px_area_mm2 = (float(ctx.get("acquisition.pixel_size_um", 50)) / 1000) ** 2
    for code in (TOP, BOTTOM):
        sel = pos_spec == code
        sec_stats.append({"slide": slide, "position": POSITION_NAME[code], "animal_id": animal_of[code],
                          "genotype": geno_of[code], "n_pixels": int(sel.sum()),
                          "area_mm2": float(sel.sum() * px_area_mm2),
                          "median_tic": float(np.median(tic[sel])) if sel.any() else np.nan,
                          "median_matrix": float(np.median(matrix_sum[sel])) if sel.any() else np.nan})
    result = {"mode": mode, "slide": slide, "file": path.name, "n_spectra": r.n, "spectrum_type": stype,
              "storage": "continuous" if r.shares_mz_axis else "processed", "mz_min": lo, "mz_max": hi,
              "section_split": seg["how"], "n_sections": seg["n_sections"], "split_axis": seg["axis"],
              "tissue_threshold_log": seg["threshold_log"], "tissue_signal": signal_name,
              "tissue_threshold_method": seg["threshold_method"],
              "n_off_tissue": int((pos_spec == 0).sum()),
              "mass_accuracy": {"n_calibrant_hits": int(len(good)), "median_ppm": med, "sd_ppm": sd,
                                "mad_ppm": mad},
              "resolution": res, "gradient": grad, "sections": sec_stats}
    out_dir = ("results", phase, mode)
    ctx.out(*out_dir, f"qc_{slide}.json").write_text(json.dumps(result, indent=2, default=float))
    cal_df.to_csv(ctx.out(*out_dir, f"calibrants_{slide}.csv"), index=False)
    mx_df.to_csv(ctx.out(*out_dir, f"matrix_ions_{slide}.csv"), index=False)

    if make_figures and S is not None:
        _figures(S, r, slide, tic, matrix_sum, nonmatrix, seg, animal_of, geno_of, axis, means, cal_df, mx_df,
                 widths, res, rows_spec, pos_spec, ctx)
    r.close()
    result["_cal"], result["_mx"] = cal_df, mx_df
    return result


def _figures(S, r, slide, tic, matrix_sum, nonmatrix, seg, animal_of, geno_of, axis, means, cal_df, mx_df,
             widths, res, rows_spec, pos_spec, ctx):
    mut, ctl = ctx.cfg["project"]["mutant_label"], ctx.cfg["project"]["control_label"]
    label = {ctx.cfg["project"]["genotypes"]["mutant"]: mut, ctx.cfg["project"]["genotypes"]["control"]: ctl}
    gcol = figstyle.genotype_colors(ctx)
    # colour of each position on THIS slide = colour of the genotype the sample sheet puts there
    pos_col = {POSITION_NAME[c]: gcol.get(geno_of[c], figstyle.GREY) for c in (TOP, BOTTOM)}
    pos_lab = {POSITION_NAME[c]: f"{POSITION_NAME[c]} section ({label.get(geno_of[c], geno_of[c])}, {animal_of[c]})"
               for c in (TOP, BOTTOM)}

    um = float(ctx.get("acquisition.pixel_size_um", 50))

    # images
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.4))
    for ax, v, t in zip(axes, (tic, nonmatrix, matrix_sum), ("Total ion current", "TIC without matrix (DAN) ions",
                                                              "Matrix (DAN) ions")):
        im = ax.imshow(np.log1p(r.to_image(v)), cmap="magma", interpolation="nearest")
        ax.set_title(t)
        figstyle.colorbar(fig, im, ax, "log(1 + intensity)")
        figstyle.image_axes(ax)
        figstyle.scalebar(ax, um, color="white")
    figstyle.panel_labels(axes, x=0.0)
    figstyle.title(fig, f"Slide {slide}: signal overview")
    S.fig(fig, f"images_{slide}.png", caption=(
        f"Slide {slide}, whole acquired raster. (A) Total ion current per pixel. (B) Total ion current after removing "
        "the known 1,5-diaminonaphthalene (DAN) matrix ions, the signal used to find tissue. (C) Summed DAN matrix "
        "ions; bare glass is bright, tissue is darker. All panels log(1 + intensity). Scale bar, 1 mm."))

    # labelled sections: THE figure to check that each section carries the right animal
    h, w = seg["positions"].shape
    fig, ax = plt.subplots(figsize=(min(9, 5.5 * w / h), 5.2) if w > h else (4.5, 5.5))
    show = np.ma.masked_equal(seg["positions"].astype(float), 0)
    ax.imshow(np.log1p(r.to_image(nonmatrix)), cmap="Greys_r", interpolation="nearest")
    from matplotlib.colors import ListedColormap
    ax.imshow(show, cmap=ListedColormap([pos_col["Top"], pos_col["Bottom"]]), alpha=0.4, vmin=1, vmax=2,
              interpolation="nearest")
    for code in (TOP, BOTTOM):
        yy, xx = np.nonzero(seg["positions"] == code)
        if len(yy):
            ax.text(xx.mean(), yy.mean(), f"{POSITION_NAME[code]}\n{animal_of[code]}\n{label.get(geno_of[code], geno_of[code])}",
                    ha="center", va="center", fontsize=9, weight="bold",
                    bbox=dict(facecolor="white", alpha=0.85, lw=0, boxstyle="round,pad=0.3"))
    figstyle.image_axes(ax)
    figstyle.scalebar(ax, um, color="white")
    figstyle.title(fig, f"Slide {slide}: section assignment", f"Split found by: {seg['how']}")
    S.fig(fig, f"sections_{slide}.png", caption=(
        f"Assignment of the two tissue sections on slide {slide} to animals. Greyscale: non-matrix ion current. "
        "Overlay colour gives the genotype the sample sheet assigns to each section (blue, Control; vermillion, "
        "CLN3 KO). The image is shown as stored in the imzML file (row 0 at the top, column 0 on the left). Labels "
        "give the sample sheet position (Top or Bottom), animal ID and genotype; check them against the slide "
        "layout before trusting any statistics. Scale bar, 1 mm."))

    # mean spectra
    fig, ax = plt.subplots(figsize=(10, 3.2))
    colors = {**pos_col, "off": figstyle.GREY}
    names = {**pos_lab, "off": "Off tissue"}
    for g, sp in means.items():
        ax.plot(*decimate(axis, sp), lw=0.5, color=colors[g], label=names.get(g, g), alpha=0.85)
    for i, (_, c) in enumerate(cal_df.drop_duplicates("name").iterrows()):
        ax.axvline(c.theoretical_mz, color="k", lw=0.5, ls=":", label="Calibrant ions" if i == 0 else None)
    for i, (_, m) in enumerate(mx_df.iterrows()):
        ax.axvline(m.theoretical_mz, color=figstyle.OKABE_ITO[3], lw=0.5, ls="--", label="DAN matrix ions" if i == 0 else None)
    ax.set_xlabel("m/z"); ax.set_ylabel("Mean intensity (a.u.)")  # noqa: E702
    ax.legend(loc="upper right", ncol=2)
    figstyle.title(fig, f"Slide {slide}: mean spectra")
    S.fig(fig, f"mean_spectrum_{slide}.png", caption=(
        f"Mean spectrum of each tissue section (blue, Control; vermillion, CLN3 KO; the legend gives its position "
        f"and animal) and of the off-tissue pixels of slide {slide}. Dotted black lines mark the calibrant ions used "
        "to measure mass accuracy, dashed purple lines the DAN matrix ions. Spectra are max-pooled for display only."))

    if len(cal_df):
        fig, ax = plt.subplots(figsize=(6.5, 3.2))
        for g, mk in (("Top", "o"), ("Bottom", "s")):
            sub = cal_df[cal_df.section == g]
            sc = ax.scatter(sub.theoretical_mz, sub.ppm_error, marker=mk, label=f"{g} section", s=18,
                            c=np.log10(sub.snr.clip(lower=1)), cmap="viridis", edgecolors="0.2", linewidths=0.3)
        ax.axhline(0, color="k", lw=0.5)
        ax.set_xlabel("m/z"); ax.set_ylabel("Mass error (ppm)")  # noqa: E702
        figstyle.colorbar(fig, sc, ax, "log10 signal-to-noise")
        ax.legend(loc="upper right")
        figstyle.title(fig, f"Slide {slide}: calibrant mass accuracy")
        S.fig(fig, f"mass_error_{slide}.png", caption=(
            f"Mass error (observed minus theoretical m/z, in ppm) of each calibrant ion found on slide {slide}, per "
            "section (circles Top, squares Bottom), coloured by signal-to-noise. The median error sets the per-slide "
            "recalibration; the spread sets the annotation tolerance."))

    if len(widths):
        fig, ax = plt.subplots(figsize=(5.5, 3.2))
        ax.scatter(widths["mz"], widths["mz"] / widths["fwhm"], s=5, alpha=0.4, color=figstyle.CONTROL,
                   edgecolors="none", rasterized=True)
        ax.set_xlabel("m/z"); ax.set_ylabel("Resolving power (m / FWHM)")  # noqa: E702
        figstyle.title(fig, f"Slide {slide}: mass resolving power", res.get("analyzer_hint", "") or None)
        S.fig(fig, f"resolution_{slide}.png", caption=(
            f"Resolving power (m/z divided by the full width at half maximum) of peaks in high-signal single spectra "
            f"of slide {slide}. The trend with m/z indicates the analyzer type: resolving power falls with m/z on "
            "Orbitrap and FT-ICR instruments and stays roughly flat or rises on a time-of-flight (QTOF)."))

    # profiles between the two section positions: is there a trend in the off-tissue matrix signal?
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    nb = int(ctx.get("qc.gradient.n_bins_y", 20))
    bins = np.linspace(rows_spec.min(), rows_spec.max() + 1, nb + 1)
    cen = (bins[:-1] + bins[1:]) / 2
    for (sel, v, name), col in zip(((pos_spec == 0, matrix_sum, "DAN ions, off tissue"),
                                    (pos_spec == 0, tic, "TIC, off tissue"), (pos_spec > 0, tic, "TIC, tissue")),
                                   (figstyle.OKABE_ITO[3], figstyle.GREY, figstyle.OKABE_ITO[2])):
        if sel.sum() < 10:
            continue
        b = np.digitize(rows_spec[sel], bins) - 1
        med = [np.median(v[sel][b == k]) if (b == k).sum() >= 3 else np.nan for k in range(nb)]
        ax.plot(cen, np.log2(np.asarray(med) + 1), marker="o", ms=3, label=name, color=col)
    which = "Image row" if seg["axis"] == "rows" else "Image column"
    ax.set_xlabel(f"{which} (Top section side to Bottom section side)")
    ax.set_ylabel("log2 median intensity")
    ax.legend(loc="best")
    figstyle.title(fig, f"Slide {slide}: signal along the Top-to-Bottom axis")
    S.fig(fig, f"y_profile_{slide}.png", caption=(
        f"Median signal in bins along the axis that separates the two sections of slide {slide}. A slope in the "
        "off-tissue DAN signal means a technical trend (matrix coating or instrument drift) that would look like a "
        "genotype difference on this slide, because the two genotypes sit at the two ends of this axis."))


def suggest_tolerance(ppm_errors: np.ndarray, floor: float) -> dict:
    """Tolerances from calibrant errors: raw (offset + spread) and after removing the offset."""
    e = np.asarray(ppm_errors, float)
    e = e[np.isfinite(e)]
    if len(e) < 3:
        return {"n": int(len(e)), "median_ppm": float(np.median(e)) if len(e) else np.nan,
                "robust_sd_ppm": np.nan, "tol_raw_ppm": np.nan, "tol_recalibrated_ppm": np.nan}
    med = float(np.median(e))
    rsd = float(stats.median_abs_deviation(e, scale="normal"))
    return {"n": int(len(e)), "median_ppm": med, "robust_sd_ppm": rsd,
            "tol_raw_ppm": float(max(floor, abs(med) + 3 * rsd)),
            "tol_recalibrated_ppm": float(max(floor, 3 * rsd))}
