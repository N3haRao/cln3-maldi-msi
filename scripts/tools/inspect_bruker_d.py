"""Is there MS/MS for our lipids in the raw Bruker .d folders? (read-only)

    python scripts/tools/inspect_bruker_d.py --config config.yaml [--d-root /path/to/folder]

Looks in every *.d folder under paths.bruker_d_root (or --d-root) and reports:

  1. what each folder holds: instrument, software, date, MALDI or not, polarity, how many
     spectra of each kind (MS1, MS/MS, PRM, DIA, PASEF), the slide spot grid
  2. every MS/MS isolation window (precursor m/z, width, collision energy, number of spots)
  3. which of OUR annotated lipids (processed/<mode>/annotations.csv, classes in
     msms_check.target_classes) fall inside one of those windows, in the right polarity
  4. if the Bruker library is available (msms_check.sdk_library): the summed fragment spectrum
     of each covered lipid, and whether the class's diagnostic fragments are there
     (reference/msms_diagnostic_fragments.csv: e.g. m/z 96.96 and 241.00 for sulfatides,
     loss of 162 for HexCer, 184.07 for SM/PC)

Writes results/msms_check/ (CSV tables, spectra, figures) and summaries/msms_check.md. Nothing is
written into the .d folders: the SQLite files are opened read-only and immutable, and the
spectra are read through Bruker's library in read-only mode.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from cln3msi import bruker, cli, figstyle  # noqa: E402
from cln3msi.annotate import REF  # noqa: E402
from cln3msi.config import _is_inside, _resolve  # noqa: E402
from cln3msi.report import Summary, plt  # noqa: E402


def targets_from_annotations(ctx, M, log) -> pd.DataFrame:
    rows = []
    want = set(M.get("target_classes") or [])
    for mode in M.get("modes") or ctx.modes_to_run:
        ap = ctx.out("processed", mode, "annotations.csv", mkdir=False)
        if not ap.exists():
            log.warning(f"{mode}: no annotations.csv, its lipids are not checked")
            continue
        A = pd.read_csv(ap)
        iso = A["is_isotope"].fillna(False).astype(bool) if "is_isotope" in A else pd.Series(False, index=A.index)
        A = A[(A["annotation"].fillna("") != "") & ~iso]
        if want:
            A = A[A["lipid_class"].isin(want)]
        pol = ctx.mode_def(mode)["polarity"]
        for _, r in A.iterrows():
            rows.append({"mode": mode, "peak_id": r.get("peak_id", ""), "name": f"{r['annotation']} {r.get('adduct', '')}".strip(),
                         "lipid_class": r["lipid_class"], "mz": float(r["mz"]), "polarity": pol,
                         "confidence_tier": r.get("confidence_tier", "")})
    for e in M.get("extra_targets") or []:
        rows.append({"mode": "extra", "peak_id": "", "name": e["name"], "lipid_class": e.get("lipid_class", ""),
                     "mz": float(e["mz"]), "polarity": e["polarity"], "confidence_tier": ""})
    return pd.DataFrame(rows)


def main():
    t0 = time.time()

    def extra(p):
        p.add_argument("--d-root", default=None, help="folder with the .d folders (overrides paths.bruker_d_root)")
        p.add_argument("--no-spectra", action="store_true", help="metadata only, don't read spectra")

    ctx, log, _, args = cli.start("msms_check", __doc__, mode=False, extra_args=extra, check_data=False)
    M = ctx.cfg.get("msms_check") or {}
    raw = args.d_root or (ctx.cfg.get("paths") or {}).get("bruker_d_root")
    S = Summary(ctx, "msms_check", None, "MS/MS check of the raw Bruker .d folders")
    if not raw:
        S.warn("no folder given: set paths.bruker_d_root in the config or pass --d-root", log)
        S.write()
        sys.exit(2)
    root = _resolve(raw, ctx.config_path.parent)
    if _is_inside(ctx.output_root, root):
        S.warn(f"output_root is inside {root}; refusing to run (raw data stay untouched)", log)
        S.write()
        sys.exit(2)
    folders = bruker.find_d_folders(root)
    log.info(f"{len(folders)} .d folder(s) under {root}")
    out = lambda *p: ctx.out("results", "msms_check", *p)  # noqa: E731

    inv, all_windows, cov_all = [], [], []
    T = targets_from_annotations(ctx, M, log)
    tol = float(M.get("isolation_tolerance_da", 0.05))
    per_d = {}
    rel = {d: "/".join(d.relative_to(root).parts) if d != root else d.name for d in folders}
    tagof = {d: rel[d].replace("/", "__").removesuffix(".d") for d in folders}
    for d in folders:
        db = bruker.db_file(d)
        rec = {"folder": rel[d], "path": str(d), "size_gb": round(sum(f.stat().st_size for f in d.rglob("*") if f.is_file()) / 1e9, 2),
               "format": db.suffix[1:] if db else "", "methods": "; ".join(sorted(p.name for p in d.glob("*.m")))}
        if db is None:
            rec["note"] = "no analysis.tdf / analysis.tsf"
            inv.append(rec)
            continue
        con = bruker.connect_ro(db)
        try:
            Tb = bruker.tables(con)
            G = bruker.global_metadata(con, Tb)
            pd.Series(G, name="value").rename_axis("key").to_csv(out(tagof[d], "global_metadata.csv"))
            pd.DataFrame([{"table": t, "columns": ", ".join(c), "rows": con.execute(f'SELECT COUNT(*) FROM "{t}"').fetchone()[0]}
                          for t, c in Tb.items()]).to_csv(out(tagof[d], "tables.csv"), index=False)
            for k in ("InstrumentName", "InstrumentVendor", "AcquisitionSoftware", "AcquisitionSoftwareVersion",
                      "AcquisitionDateTime", "MaldiApplicationType", "ImagingAreaName", "OperatorName",
                      "MzAcqRangeLower", "MzAcqRangeUpper", "MethodName", "SampleName"):
                if k in G:
                    rec[k] = G[k]
            FS = bruker.frames_summary(con, Tb)
            FS.to_csv(out(tagof[d], "frames_by_type.csv"), index=False)
            F = bruker.frames(con, Tb)
            rec["n_frames"] = len(F)
            for t in ("MS1", "MS/MS", "PRM", "DIA", "PASEF"):
                rec[f"frames_{t}"] = int(FS.loc[FS.msms_type == t, "n_frames"].sum()) if "msms_type" in FS else 0
            rec["polarity"] = ",".join(sorted({bruker.polarity_sign(p) for p in F.get("Polarity", pd.Series(dtype=str)).unique()} - {""}))
            rec["maldi_spots"] = int(F["XIndexPos"].notna().sum()) if "XIndexPos" in F else 0
            P = bruker.precursors(con, Tb)
            rec["msms_windows"] = int(P.drop_duplicates(["source", "isolation_mz", "isolation_width"]).shape[0]) if len(P) else 0
        finally:
            con.close()
        pol = F.set_index("Frame")["Polarity"] if "Polarity" in F else pd.Series(dtype=str)
        if len(P):
            W = P.assign(polarity=P["Frame"].map(pol).map(bruker.polarity_sign)).groupby(
                ["source", "isolation_mz", "isolation_width", "collision_energy", "polarity", "target"], dropna=False
            ).agg(n_frames=("Frame", "nunique")).reset_index()
            W.insert(0, "folder", rel[d])
            all_windows.append(W)
        C = bruker.coverage(T, P, pol, tol)
        if len(C):
            C.insert(0, "folder", rel[d])
            cov_all.append(C)
        per_d[d] = (F, P)
        inv.append(rec)
        # spot map: where on the slide each kind of spectrum was taken
        if "XIndexPos" in F and F["XIndexPos"].notna().any():
            fig, ax = plt.subplots(figsize=(5.5, 4.6))
            kinds = F["MsMsType"].map(bruker.MSMS_TYPES).fillna("other") if "MsMsType" in F else pd.Series("all", index=F.index)
            for k_, (t, g) in enumerate(F.groupby(kinds)):
                ax.scatter(g["XIndexPos"], g["YIndexPos"], s=2, label=f"{t} (n = {len(g):,})", rasterized=True,
                           color=figstyle.OKABE_ITO[k_ % 8], linewidths=0)
            ax.invert_yaxis()
            ax.set_aspect("equal")
            ax.set_xlabel("Raster x position (spot index)"); ax.set_ylabel("Raster y position (spot index)")  # noqa: E702
            ax.legend(markerscale=5, loc="upper left", bbox_to_anchor=(1.01, 1.0))
            figstyle.title(fig, f"{rel[d]}: acquired spectra on the slide")
            S.fig(fig, f"spots_{tagof[d]}.png", caption=(
                f"Laser spot positions stored in {rel[d]}, coloured by acquisition type (MS1, full-scan imaging; "
                "MS/MS, PRM and DIA, fragmentation spectra). Shows whether fragmentation was done on tissue and over "
                "which area. Positions are raster indices from the MaldiFrameInfo table."))

    INV = pd.DataFrame(inv)
    INV.to_csv(out("d_inventory.csv"), index=False)
    WIN = pd.concat(all_windows, ignore_index=True) if all_windows else pd.DataFrame()
    WIN.to_csv(out("msms_windows.csv"), index=False)
    COV = pd.concat(cov_all, ignore_index=True) if cov_all else pd.DataFrame()
    if len(COV):
        COV.drop(columns=["frames"]).to_csv(out("msms_coverage.csv"), index=False)

    # ---- spectra -------------------------------------------------------------------------------
    FR = pd.DataFrame()
    lib = None if args.no_spectra or not M.get("spectra", True) else bruker.find_sdk(M.get("sdk_library", "auto"))
    spec_note = ""
    if len(COV) and lib is None:
        spec_note = ("Bruker library not found, so fragment spectra were not read. On a Linux cluster: "
                     "pip install opentims_bruker_bridge (in the cln3-msi env), or set msms_check.sdk_library.")
    elif len(COV):
        sdk = bruker.SDK(lib)
        frag_ref = bruker.load_fragments(REF / "msms_diagnostic_fragments.csv")
        ppm = float(M.get("fragment_ppm", 20))
        max_frames = int(M.get("max_frames_per_window", 400))
        rows = []
        # one spectrum per (folder, isolation window), shared by every target inside the window
        groups = COV.groupby(["folder", "source", "isolation_mz", "isolation_width", "collision_energy"], dropna=False)
        for key, g in groups:
            d = next(p for p in per_d if rel[p] == key[0])
            F, P = per_d[d]
            frames = sorted(set().union(*g["frames"]))
            if "SummedIntensities" in F and len(frames) > max_frames:
                top = F.set_index("Frame").loc[frames, "SummedIntensities"].sort_values(ascending=False)
                frames = list(top.index[:max_frames])
            sc = P[P.Frame.isin(frames) & (P.isolation_mz == key[2])].drop_duplicates("Frame").set_index("Frame")
            nsc = F.set_index("Frame")["NumScans"] if "NumScans" in F else pd.Series(dtype=float)
            if not any((d / b).exists() for b in ("analysis.tdf_bin", "analysis.tsf_bin")):
                log.warning(f"{rel[d]}: no analysis.tdf_bin / .tsf_bin (metadata-only copy), spectra skipped")
                spec_note = spec_note or ("Some .d folders were copied without their spectra file (METADATA_ONLY=1), "
                                          "so their fragment spectra were not read. Upload the full folder for those.")
                continue
            try:
                kind, h = sdk.open(d)
                try:
                    specs = [sdk.spectrum(kind, h, int(fr), sc["scan_begin"].get(fr), sc["scan_end"].get(fr), nsc.get(fr))
                             for fr in frames]
                finally:
                    sdk.close(kind, h)
            except Exception as e:
                log.warning(f"{rel[d]} window {key[2]:.4f}: could not read spectra ({e})")
                continue
            mz, it = bruker.sum_spectra(specs, float(M.get("sum_bin_ppm", 10)))
            if not len(mz):
                continue
            tag = f"{tagof[d]}_{key[1]}_{key[2]:.3f}".replace("/", "")
            top = np.argsort(it)[::-1][:300]
            pd.DataFrame({"mz": mz[top], "intensity": it[top], "rel_intensity": it[top] / it.max()}).sort_values("mz") \
                .to_csv(out("spectra", f"{tag}.csv"), index=False)
            for _, t in g.iterrows():
                exp = bruker.expected_fragments(frag_ref, t["lipid_class"], t["polarity"], t["mz"])
                mf = bruker.match_fragments(mz, it, exp, ppm)
                minrel = float(M.get("min_fragment_rel_intensity", 0.01))
                strong = mf[mf.found & (mf.rel_intensity >= minrel)]
                diag = strong[strong.specificity.isin(["high", "medium"])]
                rows.append({"folder": key[0], "name": t["name"], "lipid_class": t["lipid_class"], "mz": t["mz"],
                             "polarity": t["polarity"], "source": key[1], "isolation_mz": key[2],
                             "isolation_width": key[3], "collision_energy": key[4], "n_spectra": len(frames),
                             "diagnostic_found": "; ".join(f"{r.description} ({r.rel_intensity:.0%})" for r in diag.itertuples()),
                             "chain_ions_found": "; ".join(f"{r.description} ({r.rel_intensity:.0%})"
                                                           for r in strong[strong.specificity == "chain"].itertuples()),
                             "diagnostic_missing": "; ".join(mf.loc[~mf.found & mf.specificity.isin(["high"]), "description"]),
                             "verdict": ("class confirmed" if (diag.specificity == "high").any() else
                                         "supported" if len(diag) else "no diagnostic fragment"),
                             "spectrum_file": f"spectra/{tag}.csv"})
                mf.insert(0, "name", t["name"])
                mf.to_csv(out("spectra", f"{tag}_fragments_{t['lipid_class'].replace('/', '-')}.csv"), index=False)
            # figure: the spectrum with the expected fragments marked
            fig, ax = plt.subplots(figsize=(7.5, 3.0))
            ax.vlines(mz, 0, 100 * it / it.max(), lw=0.6, color="0.25")
            from matplotlib.lines import Line2D
            hd = []
            for c_i, (cls, gc) in enumerate(g.groupby("lipid_class")):
                col = figstyle.OKABE_ITO[(c_i + 1) % 8]
                ex = bruker.expected_fragments(frag_ref, cls, gc["polarity"].iloc[0], float(gc["mz"].iloc[0]))
                ex = ex[ex.specificity != "chain"]
                for _, e in ex.iterrows():
                    ax.axvline(e["fragment_mz"], color=col, lw=0.6, ls="--", zorder=0)
                    ax.text(e["fragment_mz"], 101, f"{e['fragment_mz']:.2f}", rotation=90, fontsize=5, color=col,
                            va="bottom", ha="center")
                if len(ex):
                    hd.append(Line2D([], [], color=col, ls="--", lw=0.8, label=f"Expected {cls} fragments"))
            ax.set_ylim(0, 100)
            ax.set_xlabel("m/z")
            ax.set_ylabel("Relative intensity (%)")
            if hd:
                ax.legend(handles=hd, loc="upper left", bbox_to_anchor=(1.01, 1.0))
            win = f"{key[1]} window {key[2]:.3f} \u00b1 {float(key[3]) / 2:g}" if pd.notna(key[3]) else f"{key[1]} precursor {key[2]:.3f}"
            figstyle.title(fig, f"MS/MS, {win}", f"{key[0]}; collision energy {key[4]}; {len(frames)} spectra summed; "
                           + ", ".join(g["name"].head(3)), y=1.06)
            S.fig(fig, f"msms_{tag}.png", caption=(
                f"Summed on-tissue fragment spectrum of the {key[1]} isolation window at m/z {key[2]:.3f} in {key[0]} "
                f"({len(frames)} strongest spectra, binned at {float(M.get('sum_bin_ppm', 10)):g} ppm), scaled to the "
                "base peak. Dashed lines mark the diagnostic fragments expected for the lipid classes of our annotated "
                "ions inside the window (reference/msms_diagnostic_fragments.csv); a peak on a line within "
                f"{float(M.get('fragment_ppm', 20)):g} ppm supports that class."))
        FR = pd.DataFrame(rows)
        FR.to_csv(out("msms_fragments.csv"), index=False)

    # ---- summary ----------------------------------------------------------------------------
    S.h("What is in the folders")
    S.kv({"root": str(root), ".d folders": len(folders), "our lipids checked": len(T)})
    show = [c for c in ("folder", "format", "size_gb", "InstrumentName", "AcquisitionDateTime", "MaldiApplicationType",
                        "polarity", "n_frames", "frames_MS1", "frames_MS/MS", "frames_PRM", "frames_DIA", "frames_PASEF",
                        "maldi_spots", "msms_windows", "note") if c in INV]
    S.table(INV[show] if len(INV) else INV)
    S.h("MS/MS isolation windows")
    if len(WIN):
        S.table(WIN.sort_values(["folder", "isolation_mz"]), max_rows=60)
    else:
        S.p("No MS/MS of any kind in these folders: they are MS1 images only.")
    S.h("Which of our lipids have MS/MS")
    if len(COV):
        by = COV.groupby("lipid_class").agg(lipids_with_msms=("name", "nunique")).reset_index()
        tot = T.groupby("lipid_class").agg(lipids_checked=("name", "nunique")).reset_index()
        S.table(tot.merge(by, on="lipid_class", how="left").fillna(0))
        S.table(COV.drop(columns=["frames"]).sort_values(["lipid_class", "mz"])[
            [c for c in ("folder", "name", "mz", "polarity", "source", "isolation_mz", "isolation_width",
                         "collision_energy", "n_frames", "offset_da") if c in COV]], max_rows=80)
        wide = COV["isolation_width"].fillna(1).astype(float) > 3
        if wide.any():
            S.p(f"{int(wide.sum())} matches come from wide windows (> 3 Da, typical of DIA): several lipids are "
                "fragmented together there, so a fragment can belong to a neighbour.")
    elif len(T):
        S.p("None of our annotated lipids falls in an MS/MS isolation window.")
    if spec_note:
        S.warn(spec_note, log)
    if len(FR):
        S.h("Diagnostic fragments")
        S.table(FR[["folder", "name", "isolation_mz", "isolation_width", "n_spectra", "verdict", "diagnostic_found",
                    "diagnostic_missing"]], max_rows=80)
        S.p("MALDI MS/MS on tissue sums every ion inside the window, so with a wide window or an isobaric "
            "neighbour the fragments can come from another lipid. 'class confirmed' means a fragment that "
            "only that class gives (e.g. m/z 241.00 / 96.96 for sulfatides) is there.")
    log.info(f"summary: {S.write()}")
    cli.finish(log, t0)


if __name__ == "__main__":
    main()
