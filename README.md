# CLN3 KO brain MALDI-MSI lipidomics pipeline

Analysis code for MALDI mass spectrometry imaging (MALDI-MSI) data of the CLN3 knockout mouse
brain: which lipids accumulate or are depleted in the knockout, and where, with a focus on
myelin. Twelve animals (6 knockout, 6 Control), one coronal section each, two sections per
slide, three ion modes.

Every step is a script with a command-line interface, so the same code runs on a laptop and
as SLURM jobs on a cluster. Every path and parameter lives in one file, `config.yaml`.

> **Data policy.** The imaging data are **not** in this repository. Nothing
> in this code uploads data anywhere. The only network access is the one-time download of
> public reference databases in `00_setup.py --download` (LIPID MAPS, HMDB, SwissLipids, the
> Allen Mouse Brain Atlas). 

## Documentation

| document | for |
|---|---|
| this README | quick start on a Mac, repository layout, configuration keys, phase overview, troubleshooting |
| [`docs/METHODS.md`](docs/METHODS.md) | study design, every processing step, and which statistical tests are used and why |
| [`docs/REPLICATION.md`](docs/REPLICATION.md) | rerunning the analysis from the raw files, checking that a rerun matches, which phase to rerun after a config change, and a guide to the result tables |
| [`docs/HPC_RUN_GUIDE.md`](docs/HPC_RUN_GUIDE.md) | running everything on a SLURM cluster from the Mac terminal, every command |

## Quick start (Mac terminal)

All commands are typed in the Terminal app (Applications > Utilities > Terminal). Lines
starting with `#` are comments.

```bash
# 1. Install Miniforge once (gives you conda/mamba): https://github.com/conda-forge/miniforge
#    e.g. with Homebrew:  brew install --cask miniforge   then close and reopen the Terminal

# 2. Get the code (or download the ZIP from GitHub: Code > Download ZIP, and unzip it)
git clone https://github.com/N3haRao/cln3-maldi-msi.git
cd cln3-maldi-msi

# 3. Create the Python environment (once, 5 to 15 minutes)
bash bin/setup_env.sh
conda activate cln3-msi                   # if it asks you to run "conda init": conda init zsh, reopen the Terminal
                                          # (every new Terminal window: cd into the folder and activate again)

# 4. Check the installation on synthetic data (no real data or internet needed, ~5 minutes)
bash bin/run_synthetic_test.sh            # the last lines should say "37/37 checks passed"

# 5. Point the config at your data: open config.yaml in any text editor and set
#    paths.data_root   = the folder that CONTAINS "imzML files for biostats"
#    paths.output_root = where results should go (anywhere outside data_root)
open -e config.yaml

# 6. Download the reference databases once, then run every phase
python scripts/00_setup.py --config config.yaml --download
bash bin/run_all.sh config.yaml           # phases 01 to 10, stops at the first error

# 7. Look at the results
open ~/cln3_msi_outputs/report/report.html   # or <output_root>/report/report.html
```

Every run records the code fingerprint, config hash and package versions in
`<output_root>/metadata/provenance/`, and the report shows them, so a result can always be
traced to the exact code and settings that produced it.

---

## Repository layout

```
config.yaml              every path and parameter; the only file you normally edit
config.test.yaml         the synthetic test config (inherits config.yaml, overrides a few keys)
config.hpc*.yaml         the same for a SLURM cluster (paths only, set from bin/hpc/settings.sh)
environment.yml          conda env, Python 3.11 (pins in requirements.txt)
requirements-optional.txt   harmonypy, jupyterlab (uMAIA: separate env, see below)
metadata/sample_sheet.csv   the study design: 12 de-identified animals (genotype, sex, slide, Top/Bottom, run)
                         + imzML file names; see metadata/README.md to use your own IDs
reference/               lipid class rules, adducts, calibrants, curated metabolites (all as formulas);
                         lamanno2026/ = mouse-brain lipid evidence from the Lipid Brain Atlas (Nature 2026);
                         msms_diagnostic_fragments.csv = class fragments for the optional MS/MS check
src/cln3msi/             the shared code the scripts import
scripts/00_setup.py ... 11_compare_runs.py, 04b_scils_crosscheck.py, 09b_lipizones.py, 09c_design.py, 09d_by_sex.py, make_synthetic_data.py
scripts/tools/           build_lamanno_reference.py (rebuilds reference/lamanno2026 from the supplements),
                         inspect_bruker_d.py (optional, read-only MS/MS check of raw Bruker .d folders)
notebooks/               review notebooks (they only read outputs, they never recompute)
bin/                     setup_env.sh, run_phase.sh, run_all.sh, run_synthetic_test.sh
bin/hpc/                 everything for a SLURM cluster, driven from the Mac (see docs/HPC_RUN_GUIDE.md)
tests/                   unit tests + the synthetic end-to-end check
```

Outputs all go under `paths.output_root` (never inside the data folder; the config loader refuses):

```
<output_root>/
  metadata/    files.csv (01), header_<file>.txt (every header cvParam)
  processed/<mode>/  qc/*.npz (03), matrix_raw.h5ad + peaks.csv (04), annotations.csv (05),
                     matrix_norm.h5ad (06), regions.parquet (07), territories.parquet (08)
  results/<phase>/<mode>/   tables (csv/json)
  figures/<phase>/<mode>/   png
  summaries/<phase>_<mode>.md   <- short text summary of each phase, with warnings
  metadata/provenance/          code fingerprint, config and package versions of every phase run
  logs/<phase>_<mode>_<time>.log  (includes memory use)
  cache/       LIPID MAPS, HMDB, Allen atlas
  report/report.html
```

---

## 1. Setup (Mac, once)

Needs [Miniforge](https://github.com/conda-forge/miniforge) (gives you `mamba`/`conda`).

```bash
cd ~/path/to/cln3-maldi-msi
bash bin/setup_env.sh
conda activate cln3-msi
```

That creates the `cln3-msi` env (Python 3.11, pinned packages), installs this repo with
`pip install -e .`, tries the optional extras (a failure there is fine), and registers a
Jupyter kernel called `cln3-msi`.

**Optional: uMAIA.** It needs JAX and numpy < 2, which clash with this env (same as in the
CAJAL tutorial). If you want it, make a separate env following the uMAIA instructions
(https://github.com/lamanno-epfl/uMAIA), run `pip install -e .` from this repo inside it, set
`normalization.umaia.enabled: true`, and run phase 06 in that env. If uMAIA can't be imported,
06 says so in the log and carries on with TIC/RMS/median only. (The uMAIA route has not been
tested; the analysis as run does not use it.)

## 2. Test on fake data first (no real data or internet needed)

```bash
conda activate cln3-msi
bash bin/run_synthetic_test.sh
```

This writes ~600 MB of fake imzML/ibd files to `synthetic_data/` (same folder names with spaces,
same file names, 6 slides x 3 modes; each mode uses a different storage/spectrum type so every
code branch runs), runs phases 00 to 10 with `config.test.yaml`, runs the unit tests, and checks
that the planted effects were found (37 checks, e.g. "Cer up in KO", "sulfatide down in white
matter", "slide G has the strongest gradient"). Takes ~5 minutes. `FRESH=1 bash bin/run_synthetic_test.sh`
regenerates the fake data.

## 3. Configure for the real data

Open `config.yaml`. The things to check:

| key | what |
|---|---|
| `paths.data_root` | the folder that contains `imzML files for biostats`, e.g. `"/Users/your_name/Desktop/MSI_data"` (keep the quotes; spaces are fine; upper/lower case must match on Linux) |
| `paths.output_root` | where outputs go. Must not be inside `data_root` |
| `modes.run` | which modes to process: `[lip_neg, lip_pos, met_neg]` |
| `qc.slide`, `qc.mode` | the slide/mode phase 02 looks at first (default E, lip_neg) |
| `qc.sections.*` | split direction (`split_axis`: auto, rows or cols), which side is the sheet's Top (`top_is_low_index`, `flip_slides`), or a forced split line per slide (`manual_split`) |
| `annotation.ppm_tolerance` | leave `null` to use the value measured in 02/03 |
| `normalization.for_stats` | `tic` (decided up front; don't change it after seeing 09) |
| `registration.optical_tiffs` | per-slide optical scan (TIFF) path, relative to `data_root`, if available |
| `scils_crosscheck.enabled` | `true` to compare with the SCiLS exports |
| `class_tests.*` | which lipid classes are summed and which ratios are tested, per mode (set before looking at results) |
| `pair_check.*` | slide consistency check: regions used, flag threshold, and whether to add exploratory left-out / flipped columns for a flagged slide |

### What the pipeline reads from the data folder

Only `imzML files for biostats/` is read: 36 files, 6 slides x 3 modes, one `.imzML` and one
`.ibd` each, in this layout (folder names with spaces are fine):

```
<data_root>/
  imzML files for biostats/
    Lipid Negative/       slide_e_lip_neg.imzML, slide_e_lip_neg.ibd, ... slide_k_...
    Lipid Positive/       slide_e_lip_pos.imzML, ...
    Metabolite Negative/  slide_e_met_neg.imzML, ...
```

Anything else in `data_root` (vendor project files, SCiLS exports, other runs) is ignored,
except SCiLS intensity exports if the optional cross-check `scils_crosscheck.enabled` is on.

## 4. Run

Every phase is `python scripts/<phase>.py --config config.yaml [--mode lip_neg]`. Without
`--mode`, it loops over `modes.run`. The same thing with the helper: `bash bin/run_phase.sh 04 config.yaml --mode lip_neg`.

```bash
conda activate cln3-msi
cd ~/path/to/cln3-maldi-msi

python scripts/00_setup.py --config config.yaml --download     # once: validate + download/caches
python scripts/01_inventory.py --config config.yaml            # headers only, a few minutes
python scripts/02_qc_one_slide.py --config config.yaml         # slide E, lip_neg; look at the figures!
python scripts/03_qc_all.py --config config.yaml               # all slides: the slow one
python scripts/04_peaks_matrix.py --config config.yaml
python scripts/04b_scils_crosscheck.py --config config.yaml    # optional
python scripts/05_annotate.py --config config.yaml
python scripts/06_normalize.py --config config.yaml
python scripts/07_register.py --config config.yaml
python scripts/08_explore.py --config config.yaml
python scripts/09_differential.py --config config.yaml
python scripts/10_report.py --config config.yaml
```

Or everything in order, stopping at the first error:

```bash
bash bin/run_all.sh config.yaml
FROM=05 bash bin/run_all.sh config.yaml     # restart from a phase
```

If a download in 00 is blocked (LIPID MAPS sometimes wants a click-through), download the file
in the browser and point to it:

| **09c_design** | sensitivity analyses from 09's per-animal means, per atlas region, single peaks and class totals/ratios: **females only and males only** (paired, 3 pairs each), all pairs with **sex as a covariate** (averaged genotype effect + sex x genotype difference), **unpaired** (Welch) and unpaired with sex; **PCA** of the animals' profiles (all, slide mean removed, each sex) with the variance share of genotype, sex and slide per PC | `results/09c_design/<mode>/{peak_tests,class_tests,counts,pca_variance,pca_scores}.csv`, `sex_and_pairing.png`, `pca.png` |
| **09d_by_sex** | decided 11/23/2026: **females alone** (paired, 3 pairs) per region on the unique quality-controlled lipids (one ion per lipid), volcano plots and significant-lipid list; **males under each candidate labelling** (`by_sex.male_labelings`: corrected and flipJ); **four-quadrant** female vs male fold-change plots; **male labels from the molecular signature** (PCA of female Controls with the male sections, labelled slide_animal, plus a within-slide signature score); confidence-tier, annotation-source and Lipid Brain Atlas evidence tables for the lipid set | `results/09d_by_sex/{lipid_tests,female_significant,female_vs_male,male_signature_scores,class_tests,confidence_tiers,annotation_sources,atlas_evidence}.csv`, `by_sex_results.xlsx`, volcano, quadrant and PCA figures |
```bash
python scripts/00_setup.py --config config.yaml \
  --lipidmaps-file "$HOME/Downloads/LMSD.sdf.zip" --hmdb-file "$HOME/Downloads/structures.zip"
```

**Suggested first pass on new data:** run 00, 01 and 02, and check `summaries/01_inventory_*.md`,
`summaries/02_qc_lip_neg.md` and the `figures/02_qc/lip_neg/*.png` (especially
`sections_E.png` and `mean_spectrum_E.png`), and adjust `qc.*` settings if needed before the long phases.

---

## What each phase does and writes

| phase | does | main outputs |
|---|---|---|
| **00_setup** | validates config + sample sheet (one Control and one Mutant per slide, Top/Bottom, file names), checks the data folder, caches LIPID MAPS / HMDB as parquet, downloads the Allen CCF via brainglobe, lists optional packages | `cache/`, `summaries/00_setup.md` |
| **01_inventory** | reads every imzML header with lxml (no intensities): spectrum type, storage, polarity, instrument/software cvParams, pixel grid, #spectra, m/z range, data types. Checks profile vs centroid **from the data** (30 spectra) and flags mismatches | `metadata/files.csv`, `metadata/header_*.txt`, acquired-pixel maps |
| **02_qc_one_slide** | 2 streaming passes over one slide: TIC / non-matrix TIC / DAN images, tissue mask, Top/Bottom split + labelled figure, mean spectra (Top, Bottom, off-tissue), calibrant mass error (m/z from formulas), DAN ions, resolving power from single profile spectra (fit says TOF-, Orbitrap- or FT-ICR-like), gradient check, suggested ppm tolerance | `results/02_qc/<mode>/`, `figures/02_qc/<mode>/` |
| **03_qc_all** | 02 on every slide + cross-slide mean spectra, per-slide m/z offsets, pooled tolerance, section areas, the top-vs-bottom gradient table | `results/03_qc_all/<mode>/{mass_accuracy.json,gradient.csv,sections.csv}`, `sections_all.png` |
| **04_peaks_matrix** | peak picking on each section's mean spectrum (= centroiding if profile), per-slide recalibration, alignment across the 12 sections, keep peaks in >= 50% of sections, integrate each peak in every tissue pixel, presence filter, DAN peaks moved out of X | `processed/<mode>/matrix_raw.h5ad`, `peaks.csv` |
| **04b_scils_crosscheck** | inspects a SCiLS CSV/XLSX export (prints what it finds), compares its m/z list to ours | `results/04b_scils_crosscheck/<mode>/` |
| **05_annotate** | sum-composition library generated from class rules (exact formulas), filtered by LIPID MAPS; HMDB + curated list for metabolites; mode adducts; flags ambiguous, isotope (M+1/M+2) and type II overlaps; hypothesis tags | `processed/<mode>/annotations.csv`, `hypothesis_peaks.csv` |
| **06_normalize** | TIC, RMS, median (and uMAIA if available); slide effect before/after (partial R2 of slide on section means, PCA); X = TIC for stats; any batch-corrected embedding only in obsm for plots | `processed/<mode>/matrix_norm.h5ad`, `slide_effect_pca.png` |
| **07_register** | per section: guide image (TIFF, else sulfatide/HexCer image, else PC1), AP search around Bregma -1.58 to -2.18, similarity -> affine -> B-spline (SimpleITK, mutual information), warped Allen labels, region masks, Dice QC, lipid WM/GM fallback | `processed/<mode>/regions.parquet`, `overlays.png`, `registration.csv` |
| **08_explore** | Moran's I per section with permutations, seeded NMF on Control, Leiden territories on Control transferred to all pixels by kNN | `territories.parquet`, `morans_i.csv`, `nmf_programs.csv` |
| **09_differential** | per-animal means per region / territory, log2, paired over slides: paired t (primary, with 95% CI), Wilcoxon; BH within region (targeted and all); position flag; exploratory sex. Then, from the same means: **class totals and ratios** (`class_tests`, one BH family per mode) and the **slide consistency check** (`pair_check`: a slide whose KO/Control difference is the mirror image of the others is flagged as a possible label swap) | `results/09_differential/<mode>/{differential,class_tests,pair_consistency}.csv`, volcanoes, forest plots, per-slide dot plot, correlation heatmap |
| **09b_lipizones** | pixels clustered by lipid profile (09's QC-passing peaks; per-slide median scaling, log, PCA; k-means at the finest level, Ward merges for coarser, nested levels), **learned on Control sections only** and KO pixels assigned to the nearest lipizone; then per animal: share of the section in each lipizone, lipids and class sums/ratios inside lipizones, paired over slides | `processed/<mode>/lipizones.parquet`, `results/09b_lipizones/<mode>/{lipizones,area_tests,class_tests,lipid_tests,lipizone_composition}.csv`, `lipizone_pixels.csv.gz` (pixel table for sharing), maps per level, embedding, marker heatmap |
| **11_compare_runs** | compares two output folders (e.g. the main run and the slide J sensitivity run): class tests, single peaks, slide consistency, counts | `results/11_compare/<mode>/`, scatter plots |
| **10_report** | self-contained HTML with QC, overlays, tables, top-hit ion images (Control vs KO per slide, shared scale), volcanoes, caveats, all summaries | `report/report.html` |

### How the "don't assume" rules are handled

* **Profile vs centroid, continuous vs processed:** detected per file in 01 (header *and* data),
  used everywhere after. Continuous uncompressed files are read through a read-only numpy
  memmap; processed files spectrum by spectrum. Processed data are binned on a ppm grid
  (`qc.bin_ppm`, never coarser than the data's own spacing for profile data).
* **Instrument / resolution:** whatever the header says is logged; resolving power is measured.
* **Mass accuracy:** measured on calibrants (formulas in `reference/calibrants.yaml`); 04
  recalibrates each slide by its median offset; 05 takes its tolerance from that.
* **Memory:** nothing loads a whole slide. Peak RAM is logged in every log file. The biggest
  object is the pixels x peaks matrix in 04/06 (e.g. 250k pixels x 1000 peaks = 1 GB).

### Design limits the code respects

1. **Paired by slide**, n = 6 pairs; slide is the pairing factor.
2. **Genotype = position within a slide.** Female slides (E, F, G) have the Control on Top,
   male slides (I, J, K) the knockout on Top (see "Sample layout" below). 03 measures the
   top-to-bottom trend in the off-tissue DAN signal and TIC; 09 turns it into a predicted
   log2(Mutant/Control) per slide, signed by where the Mutant sits, and flags hits within
   `qc.gradient.flag_ratio` x the mean of that (`could_be_position`). Not corrected, only reported.
3. **Sex = run = orientation** (females runs 1-3, males run 4; the genotype order on the slide
   differs by sex). Sex results are labelled exploratory.
4. **Animal = unit.** All tests are on per-animal means, never on pixels.
5. **Labels are checked, never changed by the code.** 09 compares the slides' KO/Control
   difference profiles; a slide that runs opposite to the others is flagged as a possible swap.
   Only the sectioning/mounting records can confirm it; then correct the sample sheet.

---

## Running on a SLURM cluster

**Step-by-step guide with every command, including the ssh login: [`docs/HPC_RUN_GUIDE.md`](docs/HPC_RUN_GUIDE.md).**

Everything is driven from the Mac terminal by the scripts in `bin/hpc/`:

| script | runs on | does |
|---|---|---|
| `settings.sh` | (edit once) | your cluster login and host, work folder, partition/account, local data path |
| `upload_to_hpc.sh` | Mac | rsync the code + `imzML files for biostats` (~40 GB, resumable) |
| `start_on_hpc.sh` | Mac | ssh in, run `hpc_setup.sh`, then `submit_all.sh` |
| `hpc_setup.sh` | cluster login node | conda (module, existing install, or Miniforge), env, unit tests, phase 00 + downloads |
| `submit_all.sh` | cluster login node | submits 01 → 02, and per mode 03 → 09b → 09c as dependent jobs, then 09d (once, all modes) and 10 |
| `job_phase.sbatch` | compute node | one phase (4 cores, 32 GB by default) |
| `status.sh` / `fetch_results.sh` | Mac | queue + progress / copy summaries, figures, report back |
| `archive_outputs.sh` | Mac | move a finished run's outputs aside on the cluster (cache stays) before a full rerun |

```bash
cd ~/path/to/cln3-maldi-msi
open -e bin/hpc/settings.sh              # set HPC_USER, HPC_HOST, LOCAL_DATA_ROOT (and HPC_WORK, partition if needed)
bash bin/hpc/upload_to_hpc.sh            # code + data
bash bin/hpc/start_on_hpc.sh             # setup + submit everything
bash bin/hpc/status.sh                   # any time
bash bin/hpc/fetch_results.sh            # when jobs are done -> ~/cln3_msi_outputs_from_hpc
```

On the cluster the layout is `HPC_WORK/{repo,data,outputs,envs,slurm_logs}`, and `config.hpc.yaml`
points at it through `${CLN3_WORK}` (outputs) and `${CLN3_DATA}` (data, `HPC_DATA` in `settings.sh`,
default `HPC_WORK/data`). A second user can read the data from another account's folder and
write to their own: see "Running as another user" in `docs/HPC_RUN_GUIDE.md`. Edit code and
config on the Mac and re-run `upload_to_hpc.sh --code` (the upload mirrors the repo, so edits
made directly on the cluster get overwritten). To rerun from a phase: `FROM=05 bash bin/hpc/start_on_hpc.sh`.

## Sample layout: the male slides have the knockout on Top

On the female slides (E, F, G) the Control is the Top section; on the male slides (I, J, K) the
knockout is, because the male sections were mounted in animal-number order. The sample sheet
records this actual layout, and every figure colours sections by genotype (Control blue, knockout
vermillion), so the male slides show vermillion on Top.

How it was found: with an earlier sheet that had the Control on Top everywhere, the slide
consistency check (phase 09) flagged slide J in all three ion modes as the mirror image of the
female slides, and slides I and K showed almost no genotype signature. The sectioning records
then confirmed the reversed layout for all three male slides, and the sheet was corrected.

To compare two labelings (for example while records are being checked), run one with
`qc.sections.flip_slides` set in its own config, `config.hpc.<name>.yaml` with its own
`output_root`, and let phase 11 line the runs up:

```bash
RUN=<name> bash bin/hpc/start_on_hpc.sh            # config.hpc.<name>.yaml -> HPC_WORK/outputs_<name>, then phase 11 vs main
COMPARE_WITH=outputs_<name> bash bin/hpc/start_on_hpc.sh   # or: main run, ending with phase 11 vs that folder
bash bin/hpc/archive_outputs.sh <label>            # move a finished main run aside (cache stays) before a full rerun
```

On the Mac directly: `python scripts/11_compare_runs.py --config config.yaml --baseline <other output folder> --labels other,main`.

## Figures

Every figure is publication-ready as written: 300 dpi PNG plus an editable vector PDF (text
stays text), one style for the whole pipeline (`src/cln3msi/figstyle.py`: sans-serif at journal
sizes, colour-blind-safe Okabe-Ito colours with Control in blue and CLN3 KO in vermillion, scale
bars on ion images), and a title plus a full caption. Captions are written next to each figure
as `<figure>.caption.txt`, listed under "Figures" in the phase summary, and shown under the
figure in `report/report.html`. Resolution and formats: `report.figure_dpi` and
`report.figure_formats` in `config.yaml`.

## Lipizones (phase 09b)

Pixels grouped by their whole lipid profile, the Lipid Brain Atlas idea, at three nested levels
(`lipizones.levels`, default 4, 12 and 40). They are learned on
the Control sections only, with the same number of pixels from each, and every KO pixel is
assigned to its nearest Control lipizone. Learned on all sections, a genotype effect splits
lipizones into KO-only and Control-only ones, and comparing their areas becomes circular (the
synthetic data show it). Every test is per animal, paired over slides, never per pixel. The
lipids that define a lipizone also decide which pixels fall in it, so lipid and class tests
inside lipizones are exploratory; the 09 atlas-region tests stay the main analysis.

On a cluster, for an existing run (needs its 09 outputs):

```bash
bash bin/hpc/upload_to_hpc.sh --code
FROM=09b bash bin/hpc/start_on_hpc.sh     # 09b and 09c for each mode, then 10
bash bin/hpc/fetch_results.sh
```

UMAP is used for the embedding plot if `umap-learn` is installed, otherwise the first two
principal components.

## MS/MS check of the raw .d folders

`scripts/tools/inspect_bruker_d.py` opens Bruker `.d` folders read-only and reports what was
acquired (MS1, MS/MS, PRM, DIA), every MS/MS isolation window, and which of our annotated
lipids fall inside one. With Bruker's reader library (the `opentims_bruker_bridge` pip package
on Linux) it also sums the fragment spectra per window and looks for each class's diagnostic
fragments (`reference/msms_diagnostic_fragments.csv`, e.g. m/z 96.96 and 241.00 for sulfatides).

```bash
# settings.sh: LOCAL_D_ROOT = the Mac folder that holds the .d folders, then
bash bin/hpc/upload_d_to_hpc.sh                 # or: bash bin/hpc/upload_d_to_hpc.sh "/path/x.d"
METADATA_ONLY=1 bash bin/hpc/upload_d_to_hpc.sh  # only the small metadata files (no spectra): enough to
                                                 # see whether any MS/MS was acquired at all
bash bin/hpc/run_msms_check.sh                  # one SLURM job; installs the reader library once
bash bin/hpc/fetch_results.sh                   # summaries/msms_check.md, results/msms_check/
```

On a Mac without a cluster, the metadata part runs directly (the Bruker reader library for
the fragment spectra is Linux/Windows only):
`python scripts/tools/inspect_bruker_d.py --config config.yaml --d-root "/path/to/raw_d" --no-spectra`.

## Troubleshooting

* `CONFIG / DATA ERROR` with an expected layout printed: `paths.data_root` is wrong or a mode
  folder is missing. Paths with spaces must be quoted in YAML.
* Wrong Top/Bottom: look at `figures/03_qc_all/<mode>/sections_all.png`, then use
  `qc.sections.top_is_low_index: false` (all slides) or `flip_slides: [J]`.
* Only one section found on a slide: set `qc.sections.manual_split: {E: <row or column>}`.
* Registration bad on some sections: check `overlays.png`; sections below `registration.min_dice`
  automatically get only the lipid WM/GM masks. A TIFF usually fixes it.
* Any other error: the newest log in `<output_root>/logs/` has the full traceback and the
  settings the phase used.

## Notebooks

`notebooks/02_qc_review.ipynb`, `07_registration_review.ipynb`, `08_territories_review.ipynb`,
`09_hits_review.ipynb`. Open with the `cln3-msi` kernel; set `CONFIG` in the first cell.
