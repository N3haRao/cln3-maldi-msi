# Running the pipeline on a SLURM cluster, from the Mac terminal

This guide runs the whole analysis on a SLURM cluster while you stay in the Mac Terminal. The
scripts in `bin/hpc/` log in for you, copy the code and data, set up the software, submit every
phase as a chain of jobs, and copy the results back.

Commands are typed in the **Mac Terminal** unless a step says **on the cluster**. On the
cluster your prompt usually shows your user name and the login node, for example
`your_username@login01:~$`. Lines starting with `#` are comments; you don't type them.
Everything in `<angle brackets>` or written `your_username` / `cluster.example.edu` is a
placeholder for your own value.

Contents

1. [What you need](#1-what-you-need)
2. [One-time setup on the Mac](#2-one-time-setup-on-the-mac)
3. [Edit the settings](#3-edit-the-settings)
4. [Upload the code and the data](#4-upload-the-code-and-the-data)
5. [Set up the cluster and start the analysis](#5-set-up-the-cluster-and-start-the-analysis)
6. [Follow progress](#6-follow-progress)
7. [Get the results](#7-get-the-results)
8. [Rerun part of the analysis, or a sensitivity run](#8-rerun-part-of-the-analysis-or-a-sensitivity-run)
9. [Optional: test on fake data on the cluster](#9-optional-test-on-fake-data-on-the-cluster)
10. [Optional: the MS/MS check of raw Bruker .d folders](#10-optional-the-msms-check-of-raw-bruker-d-folders)
11. [Running as another user](#11-running-as-another-user)
12. [If something goes wrong](#12-if-something-goes-wrong)

---

## 1. What you need

| | |
|---|---|
| A cluster account | SLURM scheduler (`sbatch`, `squeue`), ssh access, about 60 GB of space in your work folder. Internet on the login node is needed once, for the software and reference downloads. |
| On the Mac | this repository (cloned or unzipped), and the data folder that contains `imzML files for biostats/` (36 files, about 40 GB) |
| Network | if the cluster is only reachable on your institution's network, connect to its VPN first |

Nothing needs to be installed on the Mac for the cluster route except `ssh` and `rsync`, which
macOS already has.

## 2. One-time setup on the Mac

Open the Terminal and go to the repository folder (drag the folder onto the Terminal window
after typing `cd ` to paste its path):

```bash
cd ~/path/to/cln3-maldi-msi
```

Optional but recommended, so you don't type your cluster password for every step: create an
ssh key and copy it to the cluster.

```bash
ls ~/.ssh/id_ed25519.pub 2>/dev/null || ssh-keygen -t ed25519    # press Enter 3 times
ssh-copy-id your_username@cluster.example.edu                     # your password, one last time
ssh your_username@cluster.example.edu 'echo connected'            # should print "connected" without a password
```

The first time you connect, ssh asks whether to trust the host: type `yes`.

## 3. Edit the settings

All cluster settings are in one file, `bin/hpc/settings.sh`. Open it in TextEdit:

```bash
open -e bin/hpc/settings.sh
```

| setting | what to put |
|---|---|
| `HPC_USER` | your login name on the cluster |
| `HPC_HOST` | the cluster's login address, for example `login.cluster.example.edu` |
| `HPC_WORK` | leave empty to use `~/cln3_msi` in your cluster home. If your home is small (check with `df -h ~` on the cluster), give a scratch or project folder, no spaces |
| `HPC_DATA` | leave empty: the data are uploaded to `HPC_WORK/data`. Set it only to read someone else's copy (Section 11) |
| `SLURM_PARTITION`, `SLURM_ACCOUNT`, `SLURM_QOS` | leave empty unless your cluster requires them (its documentation says so) |
| `CONDA_MODULE` | leave empty unless conda is provided as a module (`module avail conda` on the cluster lists it) |
| `LOCAL_DATA_ROOT` | the folder **on the Mac** that contains `imzML files for biostats`, in quotes, e.g. `"/Users/your_name/Desktop/MSI_data"` |
| `LOCAL_D_ROOT` | only for the optional MS/MS check (Section 10); leave empty otherwise |
| `MODES` | which ion modes to run; the default runs all three |

Save the file (Cmd+S) and close TextEdit. Every other setting of the analysis is in
`config.yaml`; you don't need to change anything there for the cluster, because
`config.hpc.yaml` inherits it and only redirects the folders.

## 4. Upload the code and the data

```bash
bash bin/hpc/upload_to_hpc.sh
```

* It first checks the local data (`Local data: 36 imzML/ibd files (expected 36)`) and prints
  the free space in your cluster work folder.
* Then it copies the code (seconds) and the data, with a progress line per file.
* At the end it compares the two copies and prints `OK: all files arrived`.
* If it stops (Ctrl+C, network drop, Mac asleep), run the same command again: it continues
  where it stopped. While it runs, the Mac is kept awake; leave it plugged in.

Only `imzML files for biostats/` is uploaded. Anything else in the data folder stays on the Mac.

After a code or config change, upload only the code:

```bash
bash bin/hpc/upload_to_hpc.sh --code
```

## 5. Set up the cluster and start the analysis

```bash
bash bin/hpc/start_on_hpc.sh
```

This logs in and, on the login node:

1. finds conda (a module, an existing installation, or installs Miniforge into `HPC_WORK`),
2. creates the `cln3-msi` environment in `HPC_WORK/envs` (first time only, 5 to 15 minutes),
3. runs the unit tests (should end in `passed`),
4. runs phase 00: validates the config and sample sheet and downloads LIPID MAPS, HMDB,
   SwissLipids and the Allen Mouse Brain Atlas (first time only),
5. submits every phase as SLURM jobs, each starting only when the one before it succeeded:

```
01 inventory ─┬─ 02 QC of one slide
              ├─ lip_neg: 03 → 04 → 05 → 06 → 07 → 08 → 09 → 09b → 09c ─┐
              ├─ lip_pos: 03 → 04 → 05 → 06 → 07 → 08 → 09 → 09b → 09c ─┼─ 10 report
              └─ met_neg: 03 → 04 → 05 → 06 → 07 → 08 → 09 → 09b → 09c ─┘
```

The jobs keep running after you close the Terminal. A full run takes a few hours, depending
on how many jobs the cluster runs at once (each asks for 4 cores and 32 GB).

**HMDB download refused (HTTP 403)?** Some clusters are blocked by the HMDB website. Without
HMDB the metabolite mode uses the small curated list in the repository; the lipid modes are
not affected. To add HMDB anyway, download the metabolite structures file (`structures.zip`,
SDF) from https://hmdb.ca/downloads in the Mac browser, then:

```bash
scp ~/Downloads/structures.zip your_username@cluster.example.edu:~/cln3_msi/
ssh your_username@cluster.example.edu "source ~/cln3_msi/cln3_env.sh && cd ~/cln3_msi/repo && python scripts/00_setup.py --config config.hpc.yaml --hmdb-file ~/cln3_msi/structures.zip | tail -n 5"
```

(Use your `HPC_WORK` folder instead of `~/cln3_msi` if you set one.)

## 6. Follow progress

```bash
bash bin/hpc/status.sh
```

It lists your queued and running jobs (`PENDING (Dependency)` = waiting for the previous phase,
`RUNNING`), the phase summaries written so far, and any job log that contains an error. When
the job list is empty, everything has finished.

More detail, **on the cluster** (`ssh your_username@cluster.example.edu` first):

```bash
squeue -u your_username                                                         # queued / running
sacct -u your_username -S today --format=JobID,JobName%22,State,Elapsed,MaxRSS  # finished, how it went
tail -n 30 $(ls -t ~/cln3_msi/slurm_logs/*.out | head -1)                       # newest job log
scancel -u your_username                                                        # stop everything (only if needed)
```

## 7. Get the results

```bash
bash bin/hpc/fetch_results.sh                     # into ~/cln3_msi_outputs_from_hpc
open ~/cln3_msi_outputs_from_hpc/report/report.html
```

This copies the summaries, tables, figures, logs and the report, but not the large processed
matrices, which stay on the cluster. Start with `report/report.html` and the short text
summaries in `summaries/`. Every figure is in `figures/<phase>/<mode>/` as a 300 dpi PNG and an
editable PDF, with its caption in `<figure>.caption.txt`.

Check `figures/03_qc_all/lip_neg/sections_all.png` first: on every slide the Control animal
should sit on the section the sample sheet calls Top.

## 8. Rerun part of the analysis, or a sensitivity run

After changing `config.yaml` (on the Mac), upload the code and resubmit from the first phase the
change affects (see the table in `docs/REPLICATION.md`, Section 9):

```bash
bash bin/hpc/upload_to_hpc.sh --code
FROM=05 bash bin/hpc/start_on_hpc.sh      # rerun from phase 05, reusing the outputs of 01-04
```

`FROM=03` reruns almost everything, `FROM=09` only the statistics and the report, `FROM=09b`
only the lipizones, the design checks and the report, and `FROM=09c` only the design checks
(each sex alone, sex as a covariate, unpaired, PCA) and the report.

A change to the sample sheet (which animal is at Top/Bottom) needs a full rerun. To keep the
old results, move them aside first (the reference cache stays where it is):

```bash
bash bin/hpc/archive_outputs.sh old_labels   # HPC_WORK/outputs -> HPC_WORK/outputs_old_labels
bash bin/hpc/start_on_hpc.sh
```

**Sensitivity run**: add `config.hpc.<name>.yaml` (inherit `config.hpc.yaml`, set its own
`output_root` and the setting to test, e.g. `qc.sections.flip_slides`). It runs into
`HPC_WORK/outputs_<name>` and ends with phase 11, which compares it with the main run:

```bash
RUN=<name> bash bin/hpc/start_on_hpc.sh
RUN=<name> bash bin/hpc/status.sh
RUN=<name> bash bin/hpc/fetch_results.sh   # into ~/cln3_msi_outputs_from_hpc_<name>
```

`COMPARE_WITH=outputs_<name> bash bin/hpc/start_on_hpc.sh` does it the other way round: the main
run ends with phase 11 against that folder.

## 9. Optional: test on fake data on the cluster

To check the cluster setup on a small synthetic dataset before the long upload (about 10
minutes), **on the cluster**:

```bash
cd ~/cln3_msi/repo
sbatch --output=$HOME/cln3_msi/slurm_logs/test_%j.out bin/hpc/job_test.sbatch
squeue -u your_username          # when the job is gone from the list, it has finished
tail -n 25 $(ls -t ~/cln3_msi/slurm_logs/test_*.out | head -1)   # last line: "37/37 checks passed"
```

The code must be on the cluster and the environment set up first: run
`bash bin/hpc/upload_to_hpc.sh --code`, then on the cluster `bash bin/hpc/hpc_setup.sh`.

## 10. Optional: the MS/MS check of raw Bruker .d folders

`scripts/tools/inspect_bruker_d.py` reports, read-only, what each raw Bruker `.d` folder holds
(MS1 imaging, MS/MS, PRM, DIA), which annotated lipids fall inside an MS/MS isolation window,
and, with Bruker's reader library, whether the class-specific fragments are present.

1. Set `LOCAL_D_ROOT` in `bin/hpc/settings.sh` to the Mac folder that holds the `.d` folders.
2. Upload. Raw `.d` folders can be several GB each; `METADATA_ONLY=1` copies only the small
   metadata files, which is enough to see whether any MS/MS was acquired at all:

   ```bash
   METADATA_ONLY=1 bash bin/hpc/upload_d_to_hpc.sh     # or without METADATA_ONLY for the spectra too
   ```

3. Run the check (one SLURM job; installs the reader library into the environment once) and
   fetch the results (`summaries/msms_check.md`, `results/msms_check/`):

   ```bash
   bash bin/hpc/run_msms_check.sh
   bash bin/hpc/fetch_results.sh
   ```

The quickest answer needs no upload at all: macOS includes `sqlite3`, and the frame types are
in each folder's `analysis.tdf` (0 = MS1 imaging only; 2, 8, 9 or 10 = some kind of MS/MS):

```bash
for f in "/path/to/raw_d"/*/*.d/analysis.tdf; do
  echo "== $f"; sqlite3 -readonly "$f" "SELECT MsMsType, Polarity, COUNT(*) FROM Frames GROUP BY MsMsType, Polarity;"
done
```

## 11. Running as another user

A second person can run the pipeline under their own account, reading the data from the first
person's folder (no second 40 GB copy) and writing everything to their own folder. The
pipeline only ever reads the data folder.

**1. The data owner grants read access (once, on the cluster).** Replace `<colleague>` with the
colleague's user name and `<owner_home>` with the owner's home folder (`echo $HOME`). If the
cluster supports access-control lists, this gives access to that one person only:

```bash
setfacl -m u:<colleague>:x <owner_home> <owner_home>/cln3_msi
setfacl -R -m u:<colleague>:rX <owner_home>/cln3_msi/data <owner_home>/cln3_msi/outputs
```

Without `setfacl`, ask the cluster administrators for a shared group, then `chgrp -R <group>`
and `chmod -R g+rX` those folders (plus `chmod g+x` on the two parent folders). Avoid
`chmod o+r`: that would let every user of the cluster read unpublished data.

**2. The colleague sets up their own copy (on the cluster, in their own account).**

```bash
mkdir -p ~/cln3_msi && cd ~/cln3_msi
git clone <URL of this repository> repo
cd repo
```

Without git: download the repository ZIP on a Mac, unzip it, and upload it with
`bash bin/hpc/upload_to_hpc.sh --code` after editing `bin/hpc/settings.sh`.

**3. Point the settings at the shared data.** In `bin/hpc/settings.sh`:

```bash
HPC_USER="<colleague>"
HPC_WORK=""                                  # = ~/cln3_msi in the colleague's home
HPC_DATA="<owner_home>/cln3_msi/data"        # read-only use of the owner's data
```

**4. Set up and run (on the cluster).**

```bash
cd ~/cln3_msi/repo
bash bin/hpc/hpc_setup.sh       # own environment + reference downloads; ends with "== setup OK"
bash bin/hpc/submit_all.sh      # submits every phase as SLURM jobs
squeue -u <colleague>           # watch; empty = finished
```

The setup stops with a clear message if the data folder is not readable yet. To check that two
runs match, compare the Reproducibility section of the two reports: the same code fingerprint
and config hash should give the same numbers.

## 12. If something goes wrong

| what you see | what to do |
|---|---|
| `ssh: ... timed out` | connect to your institution's VPN, check `HPC_HOST`, try again |
| `Permission denied` on ssh | wrong user name or password, or the account isn't active yet: ask your HPC team |
| `Set LOCAL_DATA_ROOT ...` or `Not found: .../imzML files for biostats` | `LOCAL_DATA_ROOT` in `bin/hpc/settings.sh` is not the folder that contains `imzML files for biostats` |
| `hpc_setup.sh` stops at `conda create` or `pip install` | look at the last 30 lines; often no internet on that node, or a full home folder (`df -h ~`) |
| a job ends in `TIMEOUT` | raise that phase's value on the `declare -A TIME=(...)` line of `bin/hpc/submit_all.sh`, upload `--code`, rerun with `FROM=` that phase |
| a job ends in `OUT_OF_MEMORY` | change `--mem=32G` to `64G` in `bin/hpc/job_phase.sbatch`, upload `--code`, rerun with `FROM=` that phase |
| `sbatch: error: invalid partition` or `account` | set `SLURM_PARTITION` / `SLURM_ACCOUNT` in `bin/hpc/settings.sh` |
| anything else | the job's `.out` log in `HPC_WORK/slurm_logs/` and the newest file in `<output_root>/logs/` have the full error and the settings used |
