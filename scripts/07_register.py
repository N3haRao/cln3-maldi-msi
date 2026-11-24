"""Phase 07: register every section to the Allen CCF and make region masks.

    python scripts/07_register.py --config config.yaml

Per mode and per section (12 per mode):
  * guide image: the optical TIFF if registration.optical_tiffs gives one for the slide,
    otherwise a lipid image with clear anatomy. auto: strongest sulfatide, then HexCer, then
    the first principal component. neutral: a lipid that shows white matter like the
    sulfatide (pixel correlation) but differs little between the two sections of any slide
    (no genotype labels used), so knockout and Control sections are registered on equal
    terms; candidates in results/07_register/<mode>/guide_candidates.csv. Or a lipid name
    (e.g. "PE 36:1") or an m/z.
  * search AP levels around Bregma -1.58 to -2.18 (+ a margin), affine, then B-spline
  * warp atlas labels onto the section -> per-pixel Allen structure
  * region masks from registration.regions (descendants included)
  * fallback (always computed): white vs grey matter from the guide image itself

Bregma to CCF is approximate (registration.bregma_ccf_ap_mm). The AP level that fits best is
reported per section; if all 12 sections land at very different levels, check the overlays.

Writes processed/<mode>/regions.parquet (one row per pixel of matrix_norm.h5ad, boolean
region columns), results/07_register/<mode>/registration.csv, overlays, summary.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cln3msi import cli, figstyle, register  # noqa: E402
from cln3msi.util import read_h5ad  # noqa: E402
from cln3msi.images import scaled01, section_image  # noqa: E402
from cln3msi.report import Summary, plt  # noqa: E402


def _sulfatide_guide(a: ad.AnnData):
    """Index of the strongest annotated sulfatide (else HexCer), or None."""
    var = a.var
    if "tag" not in var:
        return None, None
    ok = ~var.get("is_isotope", pd.Series(False, index=var.index)).fillna(False).astype(bool)
    for tag in ("SHexCer", "HexCer"):
        cand = var[ok & (var["tag"] == tag)]
        if len(cand):
            pid = cand["median_height"].idxmax()
            return var.index.get_loc(pid), tag
    return None, None


def neutral_guide(a: ad.AnnData, opts: dict, log) -> tuple[int | None, pd.DataFrame]:
    """A guide lipid that shows white matter like the sulfatide but does not differ between the
    two sections of a slide, so knockout and Control sections are registered on equal terms.

    Uses no genotype labels: for every bright, annotated, non-isotope peak,
      anatomy   = Pearson r with the sulfatide image over all tissue pixels (log intensities,
                  centred and scaled within each section, so slide offsets do not count)
      top/bottom = per slide, |log2(mean Top / mean Bottom)| of the peak; the largest over slides
    Kept: r >= min_r and largest top/bottom <= max_slide_change_log2; the one with the smallest
    top/bottom difference wins (ties: higher r). Returns (column index or None, candidate table)."""
    var, obs = a.var, a.obs
    js, _ = _sulfatide_guide(a)
    if js is None:
        return None, pd.DataFrame()
    iso = var.get("is_isotope", pd.Series(False, index=var.index)).fillna(False).astype(bool).to_numpy()
    lab = var.get("annotation", var.get("label", pd.Series("", index=var.index))).fillna("").astype(str).to_numpy()
    h = var["median_height"].to_numpy(float) if "median_height" in var else np.asarray(a.X.mean(0)).ravel()
    q = float(opts.get("min_brightness_quantile", 0.75))
    cand = np.flatnonzero(~iso & (lab != "") & (h >= np.nanquantile(h[~iso], q)))
    cand = np.union1d(cand, [js])
    X = np.asarray(a.X[:, cand], dtype=np.float64)
    key = (obs["slide"].astype(str) + "|" + obs["section"].astype(str)).to_numpy()
    rng = np.random.default_rng(0)
    take = np.concatenate([rng.permutation(np.flatnonzero(key == k))[:4000] for k in np.unique(key)])
    Z = np.log2(X[take] + 1e-3)
    kk = key[take]
    for k in np.unique(kk):
        m = kk == k
        Z[m] = (Z[m] - Z[m].mean(0)) / (Z[m].std(0) + 1e-9)
    zs = Z[:, list(cand).index(js)]
    r = (Z * zs[:, None]).mean(0) / (Z.std(0) * zs.std() + 1e-12)
    tb = []
    for sl in sorted(obs["slide"].astype(str).unique()):
        t = (obs["slide"].astype(str) == sl).to_numpy() & (obs["section"].astype(str) == "Top").to_numpy()
        b = (obs["slide"].astype(str) == sl).to_numpy() & (obs["section"].astype(str) == "Bottom").to_numpy()
        if t.any() and b.any():
            tb.append(np.abs(np.log2((X[t].mean(0) + 1e-9) / (X[b].mean(0) + 1e-9))))
    tb = np.vstack(tb) if tb else np.zeros((1, len(cand)))
    T = pd.DataFrame({"peak": var.index[cand], "label": var["label"].to_numpy()[cand] if "label" in var else lab[cand],
                      "mz": var["mz"].to_numpy()[cand], "median_height": h[cand], "r_with_sulfatide": r,
                      "max_top_bottom_log2": tb.max(0), "mean_top_bottom_log2": tb.mean(0)})
    T["is_sulfatide_guide"] = T["peak"] == var.index[js]
    min_r, max_d = float(opts.get("min_r", 0.6)), float(opts.get("max_slide_change_log2", 0.2))
    T["eligible"] = (T.r_with_sulfatide >= min_r) & (T.max_top_bottom_log2 <= max_d) & ~T.is_sulfatide_guide
    T = T.sort_values(["eligible", "max_top_bottom_log2", "r_with_sulfatide"], ascending=[False, True, False])
    if not T["eligible"].any():
        log.warning(f"no peak has r >= {min_r} with the sulfatide and a Top/Bottom difference <= {max_d} log2 on every slide")
        return None, T.reset_index(drop=True)
    return int(var.index.get_loc(T.iloc[0]["peak"])), T.reset_index(drop=True)


def choose_guide(a: ad.AnnData, how, log, neutral_opts: dict | None = None) -> tuple[np.ndarray, str, pd.DataFrame]:
    """Per-pixel guide values for the whole mode, a description and (for 'neutral') the
    candidate table. how: auto (strongest sulfatide, then HexCer, then PC1), neutral (see
    neutral_guide; falls back to auto), a lipid name or label (e.g. "PE 36:1"), or an m/z."""
    var = a.var
    none = pd.DataFrame()
    if how == "neutral":
        j, T = neutral_guide(a, neutral_opts or {}, log)
        if j is not None:
            r = T.iloc[0]
            return (np.asarray(a.X[:, j]).ravel(),
                    f"{var['label'].iloc[j] if 'label' in var else var.index[j]} (genotype-neutral: r = "
                    f"{r.r_with_sulfatide:.2f} with the sulfatide, Top/Bottom within {100 * (2 ** r.max_top_bottom_log2 - 1):.0f}% on every slide)",
                    T)
        log.warning("no genotype-neutral guide found: using the sulfatide (auto)")
        how, none = "auto", T
    if how not in (None, "auto"):
        try:
            mz = float(how)
        except (TypeError, ValueError):
            mz = None
        if mz is not None:
            j = int(np.argmin(np.abs(var["mz"].to_numpy() - mz)))
            return np.asarray(a.X[:, j]).ravel(), f"m/z {var['mz'].iloc[j]:.4f} (from config)", none
        iso = var.get("is_isotope", pd.Series(False, index=var.index)).fillna(False).astype(bool)
        names = [var.get(c, pd.Series("", index=var.index)).fillna("").astype(str) for c in ("annotation", "label")]
        lab_clean = names[1].str.replace(r"\s*\*$", "", regex=True)
        hit = iso.eq(False) & (names[0].eq(str(how)) | lab_clean.eq(str(how)) | names[1].str.startswith(str(how) + " "))
        if not hit.any():
            raise ValueError(f"registration.guide_ion: no non-isotope peak is named {how!r}")
        pid = var.loc[hit, "median_height"].idxmax() if "median_height" in var else var.index[hit][0]
        j = var.index.get_loc(pid)
        return np.asarray(a.X[:, j]).ravel(), f"{var.loc[pid, 'label'] if 'label' in var else pid} (from config)", none
    js, tag = _sulfatide_guide(a)
    if js is not None:
        return np.asarray(a.X[:, js]).ravel(), f"{var['label'].iloc[js]} (strongest {tag})", none
    Z = np.log2(np.asarray(a.X) + 1e-3)
    Z = (Z - Z.mean(0)) / (Z.std(0) + 1e-9)
    sub = np.random.default_rng(0).choice(len(Z), min(len(Z), 20000), replace=False)
    v = np.linalg.svd(Z[sub], full_matrices=False)[2][0]
    log.info("no sulfatide/HexCer annotated: guide = PC1 of all peaks")
    pc = Z @ v
    return pc - pc.min(), "PC1 of all peaks (no sulfatide/HexCer annotated)", none


def tiff_guide(ctx, slide: str, shape, log) -> np.ndarray | None:
    tiffs = ctx.get("registration.optical_tiffs") or {}
    t = tiffs.get(slide)
    if not t:
        return None
    p = Path(t)
    if not p.is_absolute():
        p = ctx.data_root / p
    if not p.exists():
        log.warning(f"TIFF for slide {slide} not found: {p}")
        return None
    import tifffile
    from skimage import transform
    img = tifffile.imread(p).astype(np.float32)
    if img.ndim == 3:
        img = img[..., :3].mean(axis=-1) if img.shape[-1] in (3, 4) else img[0]
    um = ctx.get("registration.tiff_um_per_px")
    pix = float(ctx.get("acquisition.pixel_size_um", 50))
    if um:
        img = transform.rescale(img, float(um) / pix, anti_aliasing=True, preserve_range=True)
        img = img[: shape[0], : shape[1]]
        pad = [(0, max(0, shape[0] - img.shape[0])), (0, max(0, shape[1] - img.shape[1]))]
        img = np.pad(img, pad)
    else:
        img = transform.resize(img, shape, anti_aliasing=True, preserve_range=True)
    log.info(f"slide {slide}: using TIFF {p.name} as guide (assumed to cover the MSI field of view)")
    return img.astype(np.float32)


def reuse_registration(ctx, log, S, mode: str, src: str, obs: pd.DataFrame, gvals: np.ndarray,
                       gdesc: str, R: dict) -> bool:
    """Copy another mode's per-pixel regions when both modes share the same pixel grid.

    On these slides the three modes were exported on identical rasters (same pixel count and
    coordinates per slide), so one good registration, from the mode with the clearest
    white-matter marker, serves all three. Pixels are matched by (animal, x, y). If fewer
    than 90% of this mode's pixels have a partner, we register this mode on its own instead.
    Returns True if the regions were reused.
    """
    rp = ctx.output_root / "processed" / src / "regions.parquet"
    if not rp.exists():
        S.warn(f"registration.reuse_from = {src}, but {src} has no regions.parquet yet; "
               "registering this mode on its own", log)
        return False
    src_reg = pd.read_parquet(rp)
    matched = obs.index.isin(src_reg.index)
    frac = float(matched.mean())
    if frac < 0.9:
        S.warn(f"only {frac:.0%} of the {mode} pixels exist in {src}: the pixel grids differ, "
               "so this mode is registered on its own", log)
        return False

    regions = src_reg.reindex(obs.index)
    for c in regions.columns:
        if c.startswith("region_") or c == "registration_ok":
            regions[c] = regions[c].astype("boolean").fillna(False).astype(bool)
    regions["atlas_id"] = regions["atlas_id"].fillna(0).astype(np.int64)
    regions["atlas_acronym"] = regions["atlas_acronym"].fillna("").astype(str)
    regions["region_whole_section"] = True
    regions.to_parquet(ctx.out("processed", mode, "regions.parquet"))

    sp = ctx.output_root / "results" / "07_register" / src / "registration.csv"
    reg = pd.read_csv(sp) if sp.exists() else pd.DataFrame()
    if len(reg):
        reg.insert(0, "reused_from", src)
    reg.to_csv(ctx.out("results", "07_register", mode, "registration.csv"), index=False)

    # overlays: this mode's guide image + the reused region outlines
    names = [c for c in regions.columns if c.startswith("region_")
             and c not in ("region_whole_section", "region_wm_lipid", "region_gm_lipid")]
    keys = list(obs.groupby(["slide", "section"]).groups)
    ncol = 4
    nrow = int(np.ceil(len(keys) / ncol))
    asp = figstyle.section_aspect(obs, pad=4)
    fig, axes = figstyle.image_grid(nrow, ncol, 2.6, asp, title_h=0.3)
    colors = lambda k: figstyle.REGION_COLORS[k % len(figstyle.REGION_COLORS)]  # noqa: E731
    um = float(ctx.get("acquisition.pixel_size_um", 50))
    for ax, (slide, section) in zip(axes.ravel(), keys):
        img, mask, box, _, _ = section_image(obs, gvals, slide, section, pad=4, fill=0)
        g = np.zeros_like(img)
        g[mask] = scaled01(img[mask])
        ax.imshow(np.where(mask, g, np.nan), cmap="Greys_r", interpolation="nearest")
        for k, name in enumerate(names):
            m, *_ = section_image(obs, regions[name].to_numpy().astype(float), slide, section, pad=4, fill=0, box=box)
            if np.nanmax(m) > 0:
                ax.contour(m, levels=[0.5], colors=[colors(k)], linewidths=0.9)
        gt = obs.loc[(obs.slide == slide) & (obs.section == section), "genotype"].iloc[0]
        ax.set_title(f"Slide {slide}, {section} ({figstyle.short_genotype(ctx, gt)})", fontsize=8)
        figstyle.image_axes(ax)
        figstyle.scalebar(ax, um, color="white")
    for ax in axes.ravel()[len(keys):]:
        ax.axis("off")
    handles = [plt.Line2D([], [], color=colors(k), lw=1.5, label=figstyle.region_name(n.replace("region_", "")))
               for k, n in enumerate(names)]
    fig.legend(handles=handles, loc="upper center", ncol=min(len(handles), 7), bbox_to_anchor=(0.5, -0.01))
    figstyle.title(fig, f"{figstyle.mode_name(mode)}: atlas regions", f"Regions registered on {src}, drawn on this mode's guide image ({gdesc})")
    S.fig(fig, "overlays.png", caption=(
        f"Allen Mouse Brain Atlas regions used for the statistics, outlined on each section. The regions were "
        f"registered once on the {figstyle.mode_name(src).lower()} data and transferred pixel by pixel to this mode "
        f"(same acquisition grid). Greyscale, this mode's guide image ({gdesc}), scaled per section. Scale bar, 1 mm."))

    S.h("Settings")
    S.kv({"regions": f"reused from {src} (registration.reuse_from)",
          f"{mode} pixels with a partner in {src}": f"{frac:.1%}"})
    S.h("Per section (copied from the source mode)")
    S.table(reg.round(3))
    cnt = regions.filter(like="region_").groupby(obs["animal_id"]).sum()
    S.h("Pixels per region and animal")
    S.table(cnt.reset_index())
    log.info(f"{mode}: regions reused from {src} ({frac:.1%} of pixels matched)")
    return True


def main():
    t0 = time.time()
    ctx, log, modes, _ = cli.start("07_register", __doc__)
    R = ctx.cfg["registration"]
    atlas = register.load_atlas(ctx, log)
    pix = float(ctx.get("acquisition.pixel_size_um", 50))
    if atlas is not None:
        b0, b1 = R["ap_search_bregma_mm"]
        steps = np.arange(min(b0, b1), max(b0, b1) + 1e-9, float(R["ap_step_mm"]))
        ap_idx = sorted({int(round((float(R["bregma_ccf_ap_mm"]) - b) * 1000 / atlas.res)) for b in steps})
        ap_idx = [i for i in ap_idx if 0 <= i < atlas.annotation.shape[0]]
        log.info(f"AP indices searched: {ap_idx}")
        region_ids = {name: atlas.ids_under(acrs) for name, acrs in R["regions"].items()}
        for name, ids in region_ids.items():
            if not ids:
                log.warning(f"region {name}: none of {R['regions'][name]} found in the atlas")

    # the mode whose registration the others reuse must run first, or they would copy its regions
    # from an earlier run
    src0 = R.get("reuse_from")
    if src0 in modes:
        modes = [src0] + [m for m in modes if m != src0]
    for mode in modes:
        S = Summary(ctx, "07_register", mode, f"Registration and regions, {mode}")
        p = ctx.out("processed", mode, "matrix_norm.h5ad", mkdir=False)
        if not p.exists():
            S.warn("no matrix_norm.h5ad, run 06 first", log)
            S.write()
            continue
        a = read_h5ad(p)
        obs = a.obs
        guide_cfg = R.get("guide_ion", "auto")
        how = guide_cfg.get(mode, "auto") if isinstance(guide_cfg, dict) else guide_cfg
        gvals, gdesc, gcand = choose_guide(a, how, log, R.get("guide_neutral") or {})
        log.info(f"{mode}: guide = {gdesc}")
        if len(gcand):
            gcand.round(4).to_csv(ctx.out("results", "07_register", mode, "guide_candidates.csv"), index=False)
        src = R.get("reuse_from")
        if src and src != mode and reuse_registration(ctx, log, S, mode, src, obs, gvals, gdesc, R):
            log.info(f"summary: {S.write()}")
            continue

        regions = pd.DataFrame(index=obs.index)
        regions["atlas_id"] = 0
        regions["atlas_acronym"] = ""
        regions["registration_ok"] = False
        for name in R["regions"]:
            regions[f"region_{name}"] = False
        regions["region_wm_lipid"] = False
        regions["region_gm_lipid"] = False
        regions["region_whole_section"] = True

        rows, panels = [], []
        for slide in sorted(obs["slide"].unique()):
            z = np.load(ctx.out("processed", mode, "qc", f"sections_{slide}.npz", mkdir=False))
            tif = tiff_guide(ctx, slide, z["positions"].shape, log)
            for section in ("Top", "Bottom"):
                if not ((obs.slide == slide) & (obs.section == section)).any():
                    continue
                img, mask, box, rows_idx, (yy, xx) = section_image(obs, gvals, slide, section, pad=4, fill=0)
                if tif is not None:
                    r0, r1, c0, c1 = box
                    g = np.zeros((r1 - r0, c1 - c0), np.float32)      # box can stick out of the image
                    part = tif[r0:min(r1, tif.shape[0]), c0:min(c1, tif.shape[1])]
                    g[:part.shape[0], :part.shape[1]] = part
                    g = scaled01(g)
                    g[~mask] = 0
                    src = "TIFF"
                else:
                    g = np.zeros_like(img)
                    g[mask] = scaled01(img[mask])
                    src = gdesc
                wm, gm = register.wm_gm_from_guide(np.where(mask, img, 0), mask)
                pix_ids = regions.index[rows_idx]
                regions.loc[pix_ids, "region_wm_lipid"] = wm[yy, xx]
                regions.loc[pix_ids, "region_gm_lipid"] = gm[yy, xx]
                rec = {"slide": slide, "section": section, "animal_id": obs.loc[pix_ids[0], "animal_id"],
                       "genotype": obs.loc[pix_ids[0], "genotype"], "guide": src,
                       "wm_lipid_fraction": float(wm[mask].mean())}
                res = None
                if atlas is not None:
                    log.info(f"registering {mode} {slide} {section} ...")
                    res = register.register_section(g, mask, atlas, ap_idx, [float(r) for r in R["try_rotations_deg"]],
                                                    pix, float(R["bspline_grid_mm"]), int(R["bspline_iterations"]), log)
                    if res["ok"]:
                        ok = res["dice"] >= float(R["min_dice"])
                        lab = res["labels"]
                        ids = lab[yy, xx]
                        regions.loc[pix_ids, "atlas_id"] = ids
                        regions.loc[pix_ids, "atlas_acronym"] = [atlas.id_to_acronym.get(int(i), "") for i in ids]
                        regions.loc[pix_ids, "registration_ok"] = ok
                        if ok:
                            for name, rid in region_ids.items():
                                regions.loc[pix_ids, f"region_{name}"] = np.isin(ids, list(rid))
                        ap_mm = res["ap_index"] * atlas.res / 1000
                        rec.update({"ap_index": res["ap_index"],
                                    "bregma_estimate_mm": round(float(R["bregma_ccf_ap_mm"]) - ap_mm, 3),
                                    "rotation_start": res["rotation"], "ncc_affine": res["ncc_affine"],
                                    "ncc_final": res["ncc_final"], "dice": res["dice"], "bspline_used": res["bspline_used"],
                                    "scale_section_vs_atlas": res["scale_section_vs_atlas"], "used": ok})
                    else:
                        rec.update({"used": False, "note": res.get("reason")})
                rows.append(rec)
                panels.append((slide, section, g, mask, wm, res, rec))
            log.mem(f"after slide {slide}")

        reg = pd.DataFrame(rows)
        reg.to_csv(ctx.out("results", "07_register", mode, "registration.csv"), index=False)
        regions.to_parquet(ctx.out("processed", mode, "regions.parquet"))

        # overlays: guide image + contours of the region groups + lipid WM outline
        ncol = 4
        nrow = int(np.ceil(len(panels) / ncol))
        asp = np.median([p_[3].shape[0] / max(p_[3].shape[1], 1) for p_ in panels]) if panels else 1.0
        fig, axes = figstyle.image_grid(nrow, ncol, 2.6, asp, title_h=0.42)
        colors = lambda k: figstyle.REGION_COLORS[k % len(figstyle.REGION_COLORS)]  # noqa: E731
        um = float(ctx.get("acquisition.pixel_size_um", 50))
        for ax, (slide, section, g, mask, wm, res, rec) in zip(axes.ravel(), panels):
            ax.imshow(np.where(mask, g, np.nan), cmap="Greys_r", interpolation="nearest")
            if res is not None and res.get("ok"):
                lab = res["labels"]
                for k, (name, rid) in enumerate(region_ids.items()):
                    m = np.isin(lab, list(rid)).astype(float)
                    if m.any():
                        ax.contour(m, levels=[0.5], colors=[colors(k)], linewidths=0.9)
                ax.contour((lab > 0).astype(float), levels=[0.5], colors="white", linewidths=0.6, linestyles=":")
                br = rec.get("bregma_estimate_mm")
                t = f"Dice {res['dice']:.2f}" + (f", bregma \u2248 {float(br):.2f} mm" if br is not None and str(br) != "nan" else "")
            else:
                ax.contour(wm.astype(float), levels=[0.5], colors="#56B4E9", linewidths=0.8)
                t = "No atlas fit: lipid-defined white matter"
            ax.set_title(f"Slide {slide}, {section}, {rec['animal_id']}\n{t}", fontsize=7, fontweight="normal")
            figstyle.image_axes(ax)
            figstyle.scalebar(ax, um, color="white")
        for ax in axes.ravel()[len(panels):]:
            ax.axis("off")
        if atlas is not None:
            handles = [plt.Line2D([], [], color=colors(k), lw=1.5, label=figstyle.region_name(n))
                       for k, n in enumerate(region_ids)]
            handles.append(plt.Line2D([], [], color="0.4", ls=":", label="Atlas brain outline"))
            fig.legend(handles=handles, loc="upper center", ncol=min(len(handles), 7), bbox_to_anchor=(0.5, -0.01))
        figstyle.title(fig, f"{figstyle.mode_name(mode)}: registration to the Allen atlas", f"Guide image: {gdesc}")
        S.fig(fig, "overlays.png", caption=(
            "Allen Mouse Brain Atlas (25 \u00b5m) regions warped onto each section by similarity, affine and B-spline "
            f"registration of the guide image ({gdesc}, greyscale) to the atlas, searching coronal positions in the "
            "configured bregma range. Coloured outlines, the regions tested in phase 09; dotted, the atlas brain "
            "outline. Dice, overlap between the warped atlas brain and the tissue mask (sections below the "
            "threshold fall back to lipid-defined white and grey matter). Scale bar, 1 mm."))

        fig, axes = figstyle.image_grid(nrow, ncol, 2.2, asp, title_h=0.28)
        for ax, (slide, section, g, mask, wm, res, rec) in zip(axes.ravel(), panels):
            show = np.full(mask.shape, np.nan)
            show[mask] = 0.35
            show[wm] = 1.0
            ax.imshow(show, cmap="Greys", vmin=0, vmax=1, interpolation="nearest")
            ax.set_title(f"Slide {slide}, {section}: {rec['wm_lipid_fraction']:.0%} white matter", fontsize=7.5)
            figstyle.image_axes(ax)
            figstyle.scalebar(ax, um)
        for ax in axes.ravel()[len(panels):]:
            ax.axis("off")
        figstyle.title(fig, f"{figstyle.mode_name(mode)}: lipid-defined white and grey matter")
        S.fig(fig, "wm_gm_fallback.png", caption=(
            f"White matter (black) and grey matter (grey) defined from the guide image alone ({gdesc}, Otsu threshold "
            "within the tissue). Used only for sections whose atlas registration failed; the percentage is the "
            "white-matter share of the section. Scale bar, 1 mm."))

        S.h("Settings")
        S.kv({"atlas": atlas.name if atlas is not None else "none (fallback WM/GM only)", "guide": gdesc,
              "min Dice to use atlas regions": R["min_dice"]})
        if len(gcand):
            S.h("Guide lipid: genotype-neutral choice")
            S.p("The guide should show anatomy (white matter) the same way in knockout and Control sections, so "
                "both are registered on equal terms. Candidates: bright annotated peaks. r, pixel correlation with the "
                "strongest sulfatide (anatomy, within sections); Top/Bottom, the largest |log2 Top/Bottom| over the "
                "slides (no genotype labels used). Kept if r >= "
                f"{(R.get('guide_neutral') or {}).get('min_r', 0.6)} and Top/Bottom <= "
                f"{(R.get('guide_neutral') or {}).get('max_slide_change_log2', 0.2)}; the smallest difference wins.")
            S.table(gcand.head(15)[["label", "r_with_sulfatide", "max_top_bottom_log2", "mean_top_bottom_log2",
                                    "eligible", "is_sulfatide_guide"]].round(3))
            sg = gcand[gcand.is_sulfatide_guide]
            if len(sg):
                S.p(f"For comparison, the sulfatide guide: largest Top/Bottom difference "
                    f"{sg.max_top_bottom_log2.iloc[0]:.2f} log2.")
        S.h("Per section")
        S.table(reg.round(3))
        if atlas is not None and "used" in reg:
            n_ok = int(reg["used"].fillna(False).sum())
            S.p(f"{n_ok}/{len(reg)} sections passed the Dice threshold and get atlas regions.")
            if n_ok < len(reg):
                S.warn(f"{len(reg) - n_ok} sections failed registration QC; they only get the lipid WM/GM "
                       "regions (and whole_section).", log)
            if "scale_section_vs_atlas" in reg:
                sc = reg.groupby("genotype")["scale_section_vs_atlas"].median()
                S.p("Median scale section/atlas by genotype (descriptive, <1 = smaller than atlas): "
                    + ", ".join(f"{k} {v:.2f}" for k, v in sc.items()))
        cnt = regions.filter(like="region_").groupby(obs["animal_id"]).sum()
        S.h("Pixels per region and animal")
        S.table(cnt.reset_index())
        S.p("Look at overlays.png: contours should follow the corpus callosum and the hippocampus. If the "
            "AP levels are off, adjust registration.ap_search_bregma_mm or bregma_ccf_ap_mm.")
        log.info(f"summary: {S.write()}")
    cli.finish(log, t0)


if __name__ == "__main__":
    main()
