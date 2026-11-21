#!/usr/bin/env bash
# RUN ON THE MAC. Moves the results of a finished run on the cluster into their own folder, so the next
# run starts clean. Nothing is deleted, and the reference cache stays where the configs expect it.
#
#   bash bin/hpc/archive_outputs.sh original_labels
#       HPC_WORK/outputs/{results,figures,...}  ->  HPC_WORK/outputs_original_labels/
#       HPC_WORK/slurm_logs                     ->  HPC_WORK/slurm_logs_original_labels
#       HPC_WORK/outputs/cache                  stays (LIPID MAPS, HMDB, atlas: no new download)
#   RUN=test2 bash bin/hpc/archive_outputs.sh old  # the same for outputs_test2
#
# Only folders under the cluster work folder are moved; the raw data are never touched.
set -euo pipefail
NAME="${1:-}"
[[ "$NAME" =~ ^[A-Za-z0-9_]+$ ]] || { echo "usage: bash bin/hpc/archive_outputs.sh <name>  (letters, digits, _)"; exit 2; }
source "$(dirname "$0")/_common.sh"
DEST="$HPC_WORK/outputs_$NAME"
ssh "${SSH_OPTS[@]}" "$REMOTE" bash -s -- "$REMOTE_OUT" "$DEST" "$HPC_WORK" "$NAME" <<'REMOTE_SCRIPT'
set -euo pipefail
SRC=$1 DEST=$2 WORK=$3 NAME=$4
[[ -d "$SRC" ]] || { echo "nothing to archive: $SRC does not exist"; exit 0; }
[[ -e "$DEST" ]] && { echo "$DEST already exists; pick another name"; exit 2; }
if squeue -h -u "$USER" -o '%j' 2>/dev/null | grep -q '^cln3'; then
  echo "cln3 jobs are still queued or running (squeue -u $USER); archive after they finish"; exit 2
fi
mkdir -p "$DEST"
for d in "$SRC"/*; do
  [[ -e "$d" ]] || continue
  [[ "$(basename "$d")" == "cache" ]] && continue
  mv "$d" "$DEST"/
done
echo "moved $SRC/* (except cache) -> $DEST"
if [[ -d "$WORK/slurm_logs" && ! -e "$WORK/slurm_logs_$NAME" ]]; then
  mv "$WORK/slurm_logs" "$WORK/slurm_logs_$NAME"
  echo "moved $WORK/slurm_logs -> $WORK/slurm_logs_$NAME"
fi
ls "$SRC"
REMOTE_SCRIPT
