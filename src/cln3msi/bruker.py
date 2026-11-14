"""Read-only look inside Bruker timsTOF .d folders: what was acquired, and is there MS/MS for our lipids?

A .d folder holds a SQLite file with the acquisition metadata (analysis.tdf for TIMS data,
analysis.tsf for spectra without ion mobility) and a binary file with the spectra
(analysis.tdf_bin / analysis.tsf_bin). The metadata alone tell us a lot:

  GlobalMetadata          instrument, software, acquisition date, MALDI application, polarity
  Frames                  one row per spectrum (per laser spot for MALDI) with MsMsType:
                          0 = MS1, 2 = MS/MS (fixed precursor), 8 = PASEF, 9 = DIA, 10 = PRM
  MaldiFrameInfo          spot positions on the slide (XIndexPos, YIndexPos), region number
  PrmTargets / PrmFrameMsMsInfo        PRM: the precursor list and which frames isolated which
  DiaFrameMsMsWindows / DiaFrameMsMsInfo   DIA: isolation windows, and which frames used them
  FrameMsMsInfo / PasefFrameMsMsInfo   MS/MS with a trigger (precursor) mass

The SQLite file is opened read-only and "immutable", so SQLite never writes a journal or lock
file next to it (raw data stay untouched).

Reading the fragment spectra needs Bruker's own library (libtimsdata.so on Linux, TDF-SDK; it
also reads .tsf). It is not part of this repo: msms_check.sdk_library in the config points to
it, or "auto" uses the copy that `pip install opentims_bruker_bridge` puts in the environment
(Linux and Windows only, so on a Linux cluster, not on a Mac). Without it, only the metadata are reported.
"""
from __future__ import annotations

import ctypes
import sqlite3
from pathlib import Path
from urllib.parse import quote

import numpy as np
import pandas as pd

from .chem import ELECTRON, mono_mass

MSMS_TYPES = {0: "MS1", 2: "MS/MS", 8: "PASEF", 9: "DIA", 10: "PRM"}


# ---- metadata (SQLite) ----------------------------------------------------------------------

def find_d_folders(root: Path) -> list[Path]:
    root = Path(root)
    if root.suffix.lower() == ".d" and root.is_dir():
        return [root]
    return sorted(p for p in root.rglob("*") if p.is_dir() and p.suffix.lower() == ".d"
                  and not any(q.suffix.lower() == ".d" for q in p.parents))


def db_file(d: Path) -> Path | None:
    for name in ("analysis.tdf", "analysis.tsf"):
        if (d / name).exists():
            return d / name
    return None


def connect_ro(path: Path) -> sqlite3.Connection:
    """Read-only, immutable: SQLite won't create -journal/-wal/-shm files in the data folder."""
    uri = f"file:{quote(str(Path(path).resolve()))}?mode=ro&immutable=1"
    return sqlite3.connect(uri, uri=True)


def tables(con) -> dict[str, list[str]]:
    names = [r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")]
    return {t: [r[1] for r in con.execute(f'PRAGMA table_info("{t}")')] for t in names}


def read(con, table: str) -> pd.DataFrame:
    return pd.read_sql_query(f'SELECT * FROM "{table}"', con)


def global_metadata(con, T: dict) -> dict:
    if "GlobalMetadata" not in T:
        return {}
    g = read(con, "GlobalMetadata")
    return dict(zip(g.iloc[:, 0].astype(str), g.iloc[:, 1].astype(str)))


def frames_summary(con, T: dict) -> pd.DataFrame:
    if "Frames" not in T:
        return pd.DataFrame()
    cols = [c for c in ("Id", "Polarity", "ScanMode", "MsMsType", "NumScans", "NumPeaks", "SummedIntensities")
            if c in T["Frames"]]
    F = pd.read_sql_query(f'SELECT {", ".join(cols)} FROM Frames', con)
    keys = [c for c in ("MsMsType", "Polarity", "ScanMode") if c in F]
    agg = F.groupby(keys, dropna=False).agg(n_frames=("Id", "size"),
                                            median_peaks=("NumPeaks", "median") if "NumPeaks" in F else ("Id", "size"))
    agg = agg.reset_index()
    if "MsMsType" in agg:
        agg.insert(1, "msms_type", agg["MsMsType"].map(MSMS_TYPES).fillna("other"))
    return agg


def frames(con, T: dict) -> pd.DataFrame:
    cols = [c for c in ("Id", "Polarity", "MsMsType", "SummedIntensities", "NumScans") if c in T.get("Frames", [])]
    F = pd.read_sql_query(f'SELECT {", ".join(cols)} FROM Frames', con).rename(columns={"Id": "Frame"})
    if "MaldiFrameInfo" in T:
        mc = [c for c in ("Frame", "SpotName", "RegionNumber", "XIndexPos", "YIndexPos") if c in T["MaldiFrameInfo"]]
        M = pd.read_sql_query(f'SELECT {", ".join(mc)} FROM MaldiFrameInfo', con)
        F = F.merge(M, on="Frame", how="left")
    return F


def precursors(con, T: dict) -> pd.DataFrame:
    """Every (frame, isolation window) with MS/MS, whatever the acquisition type.

    Columns: source, Frame, isolation_mz, isolation_width, collision_energy, scan_begin,
    scan_end, target (PRM target description or DIA window group)."""
    out = []

    def pick(df, **ren):
        keep = {k: v for k, v in ren.items() if k in df}
        return df[list(keep)].rename(columns=keep)

    if "PrmFrameMsMsInfo" in T:
        P = read(con, "PrmFrameMsMsInfo")
        if "PrmTargets" in T and "Target" in P:
            tg = read(con, "PrmTargets")
            desc = tg.get("Description", tg.get("ExternalId", pd.Series("", index=tg.index))).astype(str)
            P["target"] = P["Target"].map(dict(zip(tg["Id"], desc)))
        P = pick(P, Frame="Frame", IsolationMz="isolation_mz", IsolationWidth="isolation_width",
                 CollisionEnergy="collision_energy", ScanNumBegin="scan_begin", ScanNumEnd="scan_end", target="target")
        P["source"] = "PRM"
        out.append(P)
    if "DiaFrameMsMsInfo" in T and "DiaFrameMsMsWindows" in T:
        I, W = read(con, "DiaFrameMsMsInfo"), read(con, "DiaFrameMsMsWindows")
        P = I.merge(W, on="WindowGroup")
        P["target"] = "window group " + P["WindowGroup"].astype(str)
        P = pick(P, Frame="Frame", IsolationMz="isolation_mz", IsolationWidth="isolation_width",
                 CollisionEnergy="collision_energy", ScanNumBegin="scan_begin", ScanNumEnd="scan_end", target="target")
        P["source"] = "DIA"
        out.append(P)
    if "PasefFrameMsMsInfo" in T:
        P = pick(read(con, "PasefFrameMsMsInfo"), Frame="Frame", IsolationMz="isolation_mz",
                 IsolationWidth="isolation_width", CollisionEnergy="collision_energy", ScanNumBegin="scan_begin",
                 ScanNumEnd="scan_end")
        P["source"] = "PASEF"
        out.append(P)
    if "FrameMsMsInfo" in T:
        P = pick(read(con, "FrameMsMsInfo"), Frame="Frame", TriggerMass="isolation_mz",
                 IsolationWidth="isolation_width", CollisionEnergy="collision_energy")
        P["source"] = "MS/MS"
        out.append(P)
    if not out:
        return pd.DataFrame(columns=["source", "Frame", "isolation_mz", "isolation_width", "collision_energy",
                                     "scan_begin", "scan_end", "target"])
    P = pd.concat(out, ignore_index=True)
    for c in ("isolation_width", "collision_energy", "scan_begin", "scan_end", "target"):
        if c not in P:
            P[c] = np.nan
    return P


def polarity_sign(p) -> str:
    s = str(p).strip().lower()
    return "positive" if s in ("+", "positive", "1", "pos") else "negative" if s in ("-", "negative", "neg", "-1") else ""


def coverage(targets: pd.DataFrame, P: pd.DataFrame, frame_pol: pd.Series, tol_da: float,
             default_width: float = 1.0) -> pd.DataFrame:
    """Which targets fall inside an isolation window of the right polarity.

    targets: columns name, mz, polarity (+ anything else, carried along).
    Returns one row per (target, source, isolation window) with n_frames."""
    if P.empty or targets.empty:
        return pd.DataFrame()
    P = P.copy()
    P["polarity"] = P["Frame"].map(frame_pol).map(polarity_sign)
    w = P["isolation_width"].fillna(default_width).astype(float)
    P["lo"], P["hi"] = P["isolation_mz"] - w / 2 - tol_da, P["isolation_mz"] + w / 2 + tol_da
    keys = ["source", "isolation_mz", "isolation_width", "collision_energy", "polarity", "lo", "hi", "target"]
    win = P.groupby(keys, dropna=False).agg(n_frames=("Frame", "nunique"),
                                            frames=("Frame", lambda f: list(np.unique(f)))).reset_index()
    rows = []
    for _, t in targets.iterrows():
        hit = win[(win.lo <= t["mz"]) & (win.hi >= t["mz"]) & ((win.polarity == t["polarity"]) | (win.polarity == ""))]
        for _, h in hit.iterrows():
            r = t.to_dict()
            r.update({k: h[k] for k in ("source", "isolation_mz", "isolation_width", "collision_energy", "target",
                                        "n_frames", "frames")})
            r["offset_da"] = float(t["mz"] - h["isolation_mz"])
            rows.append(r)
    return pd.DataFrame(rows)


# ---- diagnostic fragments --------------------------------------------------------------------

def load_fragments(path: Path) -> pd.DataFrame:
    F = pd.read_csv(path)
    m = F["formula"].map(mono_mass)
    # an anion carries one extra electron, a cation one fewer
    F["mass"] = np.where(F["ion_charge"] < 0, m + ELECTRON, np.where(F["ion_charge"] > 0, m - ELECTRON, m))
    return F


def expected_fragments(F: pd.DataFrame, lipid_class: str, polarity: str, precursor_mz: float) -> pd.DataFrame:
    sel = F[((F.lipid_class == lipid_class) | (F.lipid_class == "*")) & (F.polarity == polarity)].copy()
    sel["fragment_mz"] = np.where(sel["kind"] == "neutral_loss", precursor_mz - sel["mass"], sel["mass"])
    return sel[sel["fragment_mz"] > 0]


def match_fragments(mz: np.ndarray, inten: np.ndarray, expected: pd.DataFrame, ppm: float) -> pd.DataFrame:
    """For each expected fragment: the strongest peak within ppm, and its share of the base peak."""
    base = float(inten.max()) if len(inten) else 0.0
    rows = []
    for _, e in expected.iterrows():
        d = np.abs(mz - e["fragment_mz"]) / e["fragment_mz"] * 1e6
        ok = d <= ppm
        r = e[["lipid_class", "kind", "description", "specificity", "fragment_mz"]].to_dict()
        if ok.any():
            j = np.flatnonzero(ok)[np.argmax(inten[ok])]
            r.update(found=True, observed_mz=float(mz[j]), ppm_error=float((mz[j] - e["fragment_mz"]) / e["fragment_mz"] * 1e6),
                     rel_intensity=float(inten[j] / base) if base else np.nan)
        else:
            r.update(found=False, observed_mz=np.nan, ppm_error=np.nan, rel_intensity=0.0)
        rows.append(r)
    return pd.DataFrame(rows)


# ---- spectra (Bruker library) -------------------------------------------------------------------

def find_sdk(setting) -> Path | None:
    """The Bruker library: a path from the config, or 'auto' = the copy that the
    opentims_bruker_bridge package installs (pip install opentims_bruker_bridge)."""
    if setting and str(setting) != "auto":
        p = Path(str(setting)).expanduser()
        return p if p.exists() else None
    import importlib.util
    import platform
    spec = importlib.util.find_spec("opentims_bruker_bridge")      # located, not imported
    if spec is None or not spec.origin:
        return None
    base = Path(spec.origin).parent
    cands = ([base / "libtimsdata.so"] if platform.system() == "Linux"
             else [base / "win64" / "timsdata.dll"] if platform.system() == "Windows" else [])
    return next((c for c in cands if c.exists()), None)


class SDK:
    """Minimal ctypes wrapper around Bruker's libtimsdata (TDF-SDK), read-only use."""

    def __init__(self, lib: Path):
        self.lib = ctypes.cdll.LoadLibrary(str(lib))
        L = self.lib
        for fn, res, args in [
            ("tims_open_v2", ctypes.c_uint64, [ctypes.c_char_p, ctypes.c_uint32, ctypes.c_uint32]),
            ("tims_close", None, [ctypes.c_uint64]),
            ("tims_read_scans_v2", ctypes.c_uint32, [ctypes.c_uint64, ctypes.c_int64, ctypes.c_uint32,
                                                     ctypes.c_uint32, ctypes.c_void_p, ctypes.c_uint32]),
            ("tims_index_to_mz", ctypes.c_uint32, [ctypes.c_uint64, ctypes.c_int64, ctypes.POINTER(ctypes.c_double),
                                                   ctypes.POINTER(ctypes.c_double), ctypes.c_uint32]),
            ("tsf_open", ctypes.c_uint64, [ctypes.c_char_p, ctypes.c_uint32]),
            ("tsf_close", None, [ctypes.c_uint64]),
            ("tsf_read_line_spectrum_v2", ctypes.c_int32, [ctypes.c_uint64, ctypes.c_int64,
                                                           ctypes.POINTER(ctypes.c_double),
                                                           ctypes.POINTER(ctypes.c_float), ctypes.c_int32]),
            ("tsf_index_to_mz", ctypes.c_uint32, [ctypes.c_uint64, ctypes.c_int64, ctypes.POINTER(ctypes.c_double),
                                                  ctypes.POINTER(ctypes.c_double), ctypes.c_uint32]),
        ]:
            if hasattr(L, fn):
                f = getattr(L, fn)
                f.restype, f.argtypes = res, args
        self.has_tsf = hasattr(L, "tsf_open")

    def open(self, d: Path):
        kind = "tdf" if (d / "analysis.tdf").exists() else "tsf"
        if kind == "tsf" and not self.has_tsf:
            raise RuntimeError("this Bruker library has no .tsf functions")
        h = (self.lib.tims_open_v2(str(d).encode(), 0, 0) if kind == "tdf"
             else self.lib.tsf_open(str(d).encode(), 0))
        if h == 0:
            raise RuntimeError(f"Bruker library could not open {d}")
        return kind, h

    def close(self, kind, h):
        (self.lib.tims_close if kind == "tdf" else self.lib.tsf_close)(h)

    def _to_mz(self, kind, h, frame, idx: np.ndarray) -> np.ndarray:
        idx = np.ascontiguousarray(idx, dtype=np.float64)
        out = np.empty_like(idx)
        f = self.lib.tims_index_to_mz if kind == "tdf" else self.lib.tsf_index_to_mz
        f(h, frame, idx.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
          out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), len(idx))
        return out

    def spectrum(self, kind, h, frame: int, scan_begin=None, scan_end=None, n_scans=None):
        """(mz, intensity) of one frame; for TIMS frames summed over scan_begin..scan_end."""
        if kind == "tsf":
            n = 4096
            while True:
                ia = np.empty(n, np.float64)
                fa = np.empty(n, np.float32)
                got = self.lib.tsf_read_line_spectrum_v2(h, frame, ia.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                                                         fa.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), n)
                if got < 0:
                    raise RuntimeError(f"tsf read failed for frame {frame}")
                if got <= n:
                    return self._to_mz(kind, h, frame, ia[:got]), fa[:got].astype(np.float64)
                n = got
        b = int(scan_begin) if scan_begin is not None and np.isfinite(scan_begin) else 0
        e = int(scan_end) if scan_end is not None and np.isfinite(scan_end) else int(n_scans or 1)
        e = max(e, b + 1)
        size = 1 << 16
        while True:
            buf = np.empty(size // 4, np.uint32)
            need = self.lib.tims_read_scans_v2(h, frame, b, e, buf.ctypes.data, size)
            if need == 0:
                raise RuntimeError(f"tdf read failed for frame {frame}")
            if need <= size:
                break
            size = int(need)
        counts = buf[: e - b]
        pos, idx, inten = e - b, [], []
        for c in counts:
            c = int(c)
            idx.append(buf[pos:pos + c]); pos += c  # noqa: E702
            inten.append(buf[pos:pos + c]); pos += c  # noqa: E702
        if not idx or sum(len(x) for x in idx) == 0:
            return np.array([]), np.array([])
        I = np.concatenate(idx).astype(np.float64)
        V = np.concatenate(inten).astype(np.float64)
        u, inv = np.unique(I, return_inverse=True)
        s = np.bincount(inv, weights=V)
        return self._to_mz(kind, h, frame, u), s


def sum_spectra(spectra: list[tuple[np.ndarray, np.ndarray]], bin_ppm: float = 10.0) -> tuple[np.ndarray, np.ndarray]:
    """Add spectra on a log-spaced m/z grid of bin_ppm; returns the intensity-weighted m/z per bin."""
    mz = np.concatenate([s[0] for s in spectra if len(s[0])]) if spectra else np.array([])
    it = np.concatenate([s[1] for s in spectra if len(s[0])]) if spectra else np.array([])
    if mz.size == 0:
        return mz, it
    k = np.floor(np.log(mz) / np.log1p(bin_ppm * 1e-6)).astype(np.int64)
    u, inv = np.unique(k, return_inverse=True)
    w = np.bincount(inv, weights=it)
    m = np.bincount(inv, weights=it * mz) / np.where(w > 0, w, 1)
    return m, w
