"""One figure style for the whole pipeline: publication-ready by default.

Applied once when cln3msi.report is imported (every phase imports it), so all figures share:

  - sans-serif type at journal sizes (7 to 9 pt at the printed figure size), no top/right spines,
    ticks out, no legend frames
  - PNG at 300 dpi, plus an editable vector PDF (text kept as text: pdf.fonttype 42)
  - colour-blind-safe colours (Okabe and Ito): Control blue, knockout vermillion
  - helpers for titles with a subtitle, panel letters, image axes and a 1 mm scale bar

Captions are written by Summary.fig (report.py) next to each figure.
"""
from __future__ import annotations

import string

import matplotlib as mpl
import numpy as np

# Okabe and Ito, colour-blind safe
OKABE_ITO = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#F0E442", "#000000"]
CONTROL = "#0072B2"
MUTANT = "#D55E00"
GREY = "#7F7F7F"
# region outlines on a greyscale image (bright, colour-blind safe, no black)
REGION_COLORS = ["#E69F00", "#56B4E9", "#009E73", "#CC79A7", "#D55E00", "#F0E442", "#0072B2"]
# slide positions. Deliberately NOT the genotype colours: the genotype on each position differs
# between slides (females: Control on Top; males: CLN3 KO on Top, see the sample sheet).
TOP, BOTTOM = "#E69F00", "#56B4E9"     # orange = Top section, sky blue = Bottom section

MODE_NAME = {"lip_neg": "Lipids, negative ion mode", "lip_pos": "Lipids, positive ion mode",
             "met_neg": "Metabolites, negative ion mode"}
REGION_NAME = {"whole_section": "Whole section", "white_matter": "White matter", "corpus_callosum": "Corpus callosum",
               "cortex": "Cortex", "hippocampus": "Hippocampus", "thalamus": "Thalamus",
               "wm_lipid": "White matter (lipid-defined)", "gm_lipid": "Grey matter (lipid-defined)"}


def apply() -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.titleweight": "semibold",
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "legend.title_fontsize": 7,
        "legend.frameon": False,
        "figure.titlesize": 10,
        "figure.titleweight": "bold",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "lines.linewidth": 1.0,
        "lines.markersize": 4,
        "axes.prop_cycle": mpl.cycler(color=OKABE_ITO),
        "image.interpolation": "nearest",
        "figure.dpi": 100,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "savefig.facecolor": "white",
        "pdf.fonttype": 42,          # editable text in Illustrator / Inkscape
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "axes.titlepad": 4,
    })


def mode_name(mode: str | None) -> str:
    return MODE_NAME.get(mode or "", mode or "")


def region_name(unit: str) -> str:
    u = str(unit)
    return REGION_NAME.get(u, u.replace("_", " ").capitalize())


def genotype_colors(ctx) -> dict:
    g = ctx.cfg["project"]["genotypes"]
    return {g["control"]: CONTROL, g["mutant"]: MUTANT}


def short_genotype(ctx, value) -> str:
    g = ctx.cfg["project"]["genotypes"]
    return "KO" if value == g["mutant"] else "Control" if value == g["control"] else str(value)


def section_genotype_colors(ctx, sheet, slide: str) -> dict:
    """{"Top": colour, "Bottom": colour} for one slide, from the genotype the sample sheet puts on each
    position (Control blue, KO vermillion). Use this, not TOP/BOTTOM, when a figure shows genotype."""
    gc = genotype_colors(ctx)
    sub = sheet[sheet.slide == slide].set_index("position_on_slide")["genotype"]
    return {pos: gc.get(sub.get(pos), GREY) for pos in ("Top", "Bottom")}


def title(fig, text: str, subtitle: str | None = None, y: float | None = None) -> None:
    """Bold figure title with an optional lighter subtitle. Only recorded here: finalize() (called by
    Summary.fig just before saving) puts it right above the plotted content, so tall figures don't
    get a gap between title and plot. y is accepted for backwards compatibility and ignored."""
    fig._cln3_title = (text, subtitle)


def finalize(fig) -> None:
    """Place the recorded title and subtitle just above the axes, their labels and any figure legend."""
    t = getattr(fig, "_cln3_title", None)
    if not t:
        return
    try:
        r = fig.canvas.get_renderer()
    except Exception:
        fig.canvas.draw()
        r = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    boxes = []
    for a in fig.axes:
        if not a.get_visible():
            continue
        try:
            bb = a.get_tightbbox(r)
        except Exception:
            bb = None
        if bb is not None and bb.width > 0:
            boxes.append(bb.transformed(inv))
    for lg in fig.legends:
        boxes.append(lg.get_window_extent(r).transformed(inv))
    top = max((b.y1 for b in boxes), default=0.95)
    x0 = min((b.x0 for b in boxes), default=0.0)
    x1 = max((b.x1 for b in boxes), default=1.0)
    xc = (x0 + x1) / 2
    h_pt = fig.get_size_inches()[1] * 72.0
    y = top + 5.0 / h_pt
    text, sub = t
    if sub:
        fig.text(xc, y, sub, ha="center", va="bottom", fontsize=8, color="0.3")
        y += 12.0 / h_pt
    fig.text(xc, y, text, ha="center", va="bottom", fontsize=10, fontweight="bold")


def panel_labels(axes, start: int = 0, x: float = -0.12, y: float = 1.06) -> None:
    """A, B, C ... in the top-left corner of each axis."""
    for i, ax in enumerate(np.ravel(axes)):
        ax.text(x, y, string.ascii_uppercase[start + i], transform=ax.transAxes, fontsize=10,
                fontweight="bold", va="bottom", ha="left")


def image_axes(ax) -> None:
    """Ion images: no ticks, no frame."""
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def section_aspect(obs, pad: int = 3) -> float:
    """Median height / width of the sections' bounding boxes, to size image grids without gaps."""
    import pandas as pd
    asp = []
    for (s, sec), g in obs.groupby(["slide", "section"], observed=True):
        h = g["y"].max() - g["y"].min() + 1 + 2 * pad
        w = g["x"].max() - g["x"].min() + 1 + 2 * pad
        asp.append(h / max(w, 1))
    return float(pd.Series(asp).median()) if asp else 1.0


def image_grid(nrow: int, ncol: int, panel_w: float, aspect: float, title_h: float = 0.28,
               wspace_in: float = 0.06, right_in: float = 0.0, bottom_in: float = 0.0):
    """A grid of image panels that fill their cells exactly (no default subplot margins), so there
    are no empty bands between rows. panel_w in inches, aspect = image height / width,
    title_h = room above each panel for its title, right_in / bottom_in = extra room for
    colour bars or a legend. Returns (fig, axes[nrow, ncol])."""
    import matplotlib.pyplot as plt
    ph = panel_w * aspect
    W = ncol * panel_w + (ncol - 1) * wspace_in + right_in
    H = nrow * (ph + title_h) + bottom_in
    fig = plt.figure(figsize=(W, H))
    gs = fig.add_gridspec(nrow, ncol, left=0.0, right=1 - right_in / W, bottom=bottom_in / H, top=1 - title_h / H,
                          wspace=wspace_in / panel_w, hspace=title_h / ph)
    axes = np.array([[fig.add_subplot(gs[i, j]) for j in range(ncol)] for i in range(nrow)])
    return fig, axes


def side_colorbar(fig, im, ax, label: str | None = None, width_in: float = 0.07, gap_in: float = 0.06):
    """A slim colour bar to the right of ax that doesn't shrink ax."""
    W, H = fig.get_size_inches()
    pos = ax.get_position()
    cax = fig.add_axes([pos.x1 + gap_in / W, pos.y0 + 0.1 * pos.height, width_in / W, 0.8 * pos.height])
    cb = fig.colorbar(im, cax=cax)
    cb.ax.tick_params(labelsize=5, width=0.5, length=2)
    cb.outline.set_linewidth(0.4)
    if label:
        cb.set_label(label, fontsize=6)
    return cb


def scalebar(ax, um_per_px: float, length_um: float | None = None, color: str = "black") -> None:
    """A horizontal scale bar in the lower right corner of an image axis. length_um None picks the
    longest round length (5, 2, 1, 0.5, 0.2, 0.1 mm) that takes at most a quarter of the width."""
    if not um_per_px or um_per_px <= 0:
        return
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()           # images: y0 is the bottom edge (inverted axis)
    w, h = abs(x1 - x0), abs(y1 - y0)
    if length_um is None:
        fits = [L for L in (5000, 2000, 1000, 500, 200, 100) if L / um_per_px <= 0.25 * w]
        if not fits:
            return
        length_um = fits[0]
    n_px = length_um / um_per_px
    if n_px > 0.6 * w:
        return
    xr = max(x0, x1) - 0.04 * w
    yb = max(y0, y1) - 0.05 * h
    ax.plot([xr - n_px, xr], [yb, yb], color=color, lw=2, solid_capstyle="butt")
    label = f"{length_um / 1000:g} mm" if length_um >= 1000 else f"{length_um:g} µm"
    ax.text(xr - n_px / 2, yb - 0.025 * h, label, ha="center", va="bottom", fontsize=6, color=color)


def colorbar(fig, im, ax, label: str, **kw):
    cb = fig.colorbar(im, ax=ax, fraction=kw.pop("fraction", 0.04), pad=kw.pop("pad", 0.02), **kw)
    cb.set_label(label, fontsize=7)
    cb.ax.tick_params(labelsize=6, width=0.6)
    cb.outline.set_linewidth(0.5)
    return cb
