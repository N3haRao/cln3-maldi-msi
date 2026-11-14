# ---------------------------------------------------------------------------------------
# HPC settings. EDIT THIS FILE (once), everything in bin/hpc/ reads it.
# It's used on your computer (to know what to upload) and on the cluster (to know where
# things are). Lines starting with # are comments. Keep the quotes around every value.
# ---------------------------------------------------------------------------------------

# How you log in: ssh HPC_USER@HPC_HOST
HPC_USER="your_username"          # your login name on the cluster
HPC_HOST="cluster.example.edu"     # the cluster's login address (ask your HPC team)

# Work folder on the cluster. Needs ~60 GB. No spaces in this path, please.
# Empty = "cln3_msi" in your home folder on the cluster. If your home folder is small (check
# with `df -h ~` on the cluster, see docs/HPC_RUN_GUIDE.md), put a scratch/project path here
# instead, e.g. HPC_WORK="/scratch/your_username/cln3_msi"
HPC_WORK=""

# Where the imzML data sit on the cluster (the folder that CONTAINS "imzML files for biostats").
# Empty = HPC_WORK/data, which is where upload_to_hpc.sh puts it. A second user who reads
# the data from someone else's folder sets it here, e.g. HPC_DATA="/home/other_user/cln3_msi/data"
# (the data are only read, never written; see docs/HPC_RUN_GUIDE.md, "Running as another user").
HPC_DATA=""

# SLURM: leave empty to use the cluster defaults. Ask the HPC team / docs if unsure.
SLURM_PARTITION=""          # e.g. "defq", "compute"
SLURM_ACCOUNT=""            # only if the cluster makes you charge jobs to an account / project
SLURM_QOS=""                # only if the cluster uses QOS

# Only needed if the cluster has conda as a module (otherwise setup finds conda on your PATH
# or installs Miniforge in HPC_WORK).
CONDA_MODULE=""             # e.g. "miniforge3" or "anaconda3"; check with: module avail conda

# On your computer: the folder that CONTAINS "imzML files for biostats" (used by
# upload_to_hpc.sh). Spaces are fine, keep the quotes. Example:
#   LOCAL_DATA_ROOT="/Users/your_name/Desktop/MSI_data"
LOCAL_DATA_ROOT="/path/to/data_root"

# On your computer: the folder that holds raw Bruker .d folders, only for the optional MS/MS
# check (upload_d_to_hpc.sh copies every *.d folder under it, keeping subfolders).
# Empty = not set. Example: LOCAL_D_ROOT="/Users/your_name/Desktop/MSI_data/raw_d"
LOCAL_D_ROOT=""

# Which modes to run (one chain of jobs per mode, they run side by side)
MODES="lip_neg lip_pos met_neg"
