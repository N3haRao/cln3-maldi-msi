#!/usr/bin/env bash
# RUN ON THE MAC. Runs the MS/MS check of the raw .d folders on the cluster as one SLURM job.
#   bash bin/hpc/run_msms_check.sh                # uses the main run's annotations (outputs/)
#   RUN=test2 bash bin/hpc/run_msms_check.sh       # uses outputs_test2 (same annotations, own copy)
#   INSTALL_BRUKER_LIB=0 bash bin/hpc/run_msms_check.sh   # metadata only, no fragment spectra
# Steps: uploads the code, installs Bruker's reader library (opentims_bruker_bridge, a pip
# package with Bruker's libtimsdata.so; first time only, on the login node because it needs
# internet), then submits scripts/tools/inspect_bruker_d.py. Results come back with
# fetch_results.sh, under results/msms_check/ and summaries/msms_check.md.
set -euo pipefail
source "$(dirname "$0")/_common.sh"
bash "$(dirname "$0")/upload_to_hpc.sh" --code
CFG="config.hpc${RUN:+.$RUN}.yaml"
INSTALL="${INSTALL_BRUKER_LIB:-1}"
ssh "${SSH_OPTS[@]}" "$REMOTE" bash -s <<REMOTE_EOF
set -euo pipefail
source "$HPC_WORK/cln3_env.sh" >/dev/null 2>&1 || { echo "run start_on_hpc.sh once first (sets up the env)"; exit 2; }
cd "$REMOTE_REPO"
source bin/hpc/settings.sh
if [[ "$INSTALL" == "1" ]]; then
  python -c "import importlib.util as u; exit(0 if u.find_spec('opentims_bruker_bridge') else 1)" \
    || pip install -q opentims_bruker_bridge || echo "could not install opentims_bruker_bridge: metadata only"
fi
mkdir -p "$HPC_WORK/slurm_logs"
EXTRA=()
[[ -n "\$SLURM_PARTITION" ]] && EXTRA+=(--partition="\$SLURM_PARTITION")
[[ -n "\$SLURM_ACCOUNT" ]] && EXTRA+=(--account="\$SLURM_ACCOUNT")
[[ -n "\$SLURM_QOS" ]] && EXTRA+=(--qos="\$SLURM_QOS")
j=\$(PHASE=msms MODE="" CLN3_CONFIG="$CFG" CLN3_ENV_FILE="$HPC_WORK/cln3_env.sh" sbatch --parsable --export=ALL \
  "\${EXTRA[@]+"\${EXTRA[@]}"}" --job-name=cln3_msms --time=02:00:00 --mem=16G \
  --output="$HPC_WORK/slurm_logs/%x_%j.out" bin/hpc/job_phase.sbatch)
echo "submitted MS/MS check: job \$j   (config $CFG)"
echo "log: $HPC_WORK/slurm_logs/cln3_msms_\$j.out"
REMOTE_EOF
echo "When it's done (squeue empty):  ${RUN:+RUN=$RUN }bash bin/hpc/fetch_results.sh"
