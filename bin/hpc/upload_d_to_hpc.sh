#!/usr/bin/env bash
# RUN ON THE MAC. Copies raw Bruker .d folders to the cluster for the MS/MS check (HPC_WORK/raw_d).
#   bash bin/hpc/upload_d_to_hpc.sh                          # every *.d under LOCAL_D_ROOT (settings.sh)
#   bash bin/hpc/upload_d_to_hpc.sh "/path/one.d" "/path/two.d"   # just these folders
#   METADATA_ONLY=1 bash bin/hpc/upload_d_to_hpc.sh          # only analysis.tdf/.tsf and the method
#       folders (a few hundred MB instead of tens of GB): enough for the inventory, the MS/MS
#       windows and which of our lipids are covered; run_msms_check.sh then skips the spectra.
# rsync: safe to stop (Ctrl+C) and re-run, it continues where it stopped. Only reads the Mac
# copy; on the cluster the folders are only read too (the check opens them read-only).
set -euo pipefail
source "$(dirname "$0")/_common.sh"
DEST="$HPC_WORK/raw_d"
RSYNC_SSH="ssh ${SSH_OPTS[*]}"
# METADATA_ONLY=1: leave out the big spectra files (*.tdf_bin, *.tsf_bin)
FILTER=()
[[ "${METADATA_ONLY:-0}" == "1" ]] && FILTER=(--exclude "*.tdf_bin" --exclude "*.tsf_bin" --exclude ".DS_Store") \
  && echo "metadata only: *.tdf_bin / *.tsf_bin are not copied"
KEEP_AWAKE=()
command -v caffeinate >/dev/null 2>&1 && KEEP_AWAKE=(caffeinate -i)
ssh "${SSH_OPTS[@]}" "$REMOTE" "mkdir -p '$DEST' && echo \"free space on the cluster: \$(df -h '$HPC_WORK' | awk 'NR==2{print \$4}')\""

if [[ $# -gt 0 ]]; then
  for d in "$@"; do
    [[ -d "$d" && "$d" == *.d ]] || { echo "not a .d folder: $d"; exit 2; }
    echo "== $d ($(du -sh "$d" | cut -f1))"
    ${KEEP_AWAKE[@]+"${KEEP_AWAKE[@]}"} rsync -a --partial --progress -e "$RSYNC_SSH" ${FILTER[@]+"${FILTER[@]}"} "${d%/}" "$REMOTE:$DEST/"
  done
else
  [[ -n "$LOCAL_D_ROOT" && -d "$LOCAL_D_ROOT" ]] || { echo "Set LOCAL_D_ROOT in bin/hpc/settings.sh (or pass .d folders)"; exit 2; }
  echo "== .d folders under $LOCAL_D_ROOT:"
  find "$LOCAL_D_ROOT" -type d -name "*.d" -prune -print | sed 's/^/   /'
  # only the .d folders (and the subfolders leading to them), nothing else
  ${KEEP_AWAKE[@]+"${KEEP_AWAKE[@]}"} rsync -a --partial --progress --prune-empty-dirs -e "$RSYNC_SSH" \
    ${FILTER[@]+"${FILTER[@]}"} --include "*.d/***" --include "*/" --exclude "*" "${LOCAL_D_ROOT%/}/" "$REMOTE:$DEST/"
fi
echo
echo "on the cluster:"
ssh "${SSH_OPTS[@]}" "$REMOTE" "cd '$DEST' && find . -type d -name '*.d' -prune -exec du -sh {} \;"
echo "Next: bash bin/hpc/run_msms_check.sh"
