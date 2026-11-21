# Sample sheet

`sample_sheet.csv` describes the study design: one row per animal (= one tissue section).
The animal IDs in this repository are de-identified (`animal_01` ... `animal_12`); the
design columns are the real ones, so the analysis can be reproduced as run. The pipeline only
uses the IDs as labels in tables and figures.

## Columns

| column | meaning | allowed values |
|---|---|---|
| `animal_id` | a unique label per animal | any text without commas, unique |
| `genotype` | group | `Control` or `Mutant` (the names set in `config.yaml`, `project.genotypes`) |
| `sex` | sex | `Female` or `Male` |
| `slide` | the slide the section is on | one uppercase letter, matching the imzML file names (`slide_e_...` = `E`) |
| `position_on_slide` | where the section sits on the slide | `Top` or `Bottom` |
| `run_order` | acquisition run (batch) | integer |
| `imzml_lip_neg`, `imzml_lip_pos`, `imzml_met_neg` | the imzML file of that slide for each ion mode | file name inside the mode folder |

Rules checked by phase 00 (it stops with a clear message if one is broken): every slide has
exactly one Control and one Mutant, one on Top and one on the Bottom; both animals on a slide
share sex and run; IDs are unique; the file names exist in the data folder.

## Using your own (identifiable) IDs

Keep the identified sheet outside the repository, so it is never committed, and point the
config at it:

```yaml
# config.yaml
paths:
  sample_sheet: "/path/to/private/sample_sheet.csv"
```

Relative paths are taken from the repository folder; absolute paths work from anywhere. The
provenance record of every run stores a hash of the sheet that was used
(`sample_sheet_sha256`), so a run can be matched to its sheet without storing the IDs.

## A new study

Copy `sample_sheet.csv`, keep the header row, and fill in one row per section.
`position_on_slide` must give where each animal actually is on the glass. It does not have to
be the same genotype on Top for every slide: in this study the female slides have the Control on
Top and the male slides the knockout (the male sections were mounted in animal-number order).

If the sections on a slide may be the other way round, do not edit the sheet because of the
results: check the sectioning and mounting records first. If they confirm it, correct
`position_on_slide` and rerun from phase 02. To see how much a suspected swap matters before the
records are checked, set `qc.sections.flip_slides` in a separate config and compare the two
runs with phase 11.
