#!/usr/bin/env bash
# RUN ON THE MAC. Logs in to the cluster, sets up the software (first time only), downloads the
# reference data, and submits the whole pipeline as SLURM jobs.
#   bash bin/hpc/start_on_hpc.sh
#   FROM=05 bash bin/hpc/start_on_hpc.sh     # resubmit from phase 05 on (earlier outputs reused)
#   RUN=test2 bash bin/hpc/start_on_hpc.sh   # a sensitivity run (config.hpc.test2.yaml), own outputs
#   COMPARE_WITH=outputs_test2 bash bin/hpc/start_on_hpc.sh   # end with phase 11 against that run
# The jobs keep running on the cluster after you close the terminal.
set -euo pipefail
source "$(dirname "$0")/_common.sh"
ssh -t "${SSH_OPTS[@]}" "$REMOTE" "cd '$REMOTE_REPO' && RUN='$RUN' bash bin/hpc/hpc_setup.sh && RUN='$RUN' FROM='${FROM:-01}' COMPARE_WITH='${COMPARE_WITH:-}' bash bin/hpc/submit_all.sh"
