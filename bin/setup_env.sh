#!/usr/bin/env bash
# One-time setup on the Mac (or any Linux box): create the conda env and install this repo.
#   bash bin/setup_env.sh
# Needs Miniforge (mamba/conda): https://github.com/conda-forge/miniforge
set -euo pipefail
cd "$(dirname "$0")/.."

if command -v mamba >/dev/null 2>&1; then CONDA=mamba
elif command -v conda >/dev/null 2>&1; then CONDA=conda
else
  echo "conda/mamba not found. Install Miniforge first: https://github.com/conda-forge/miniforge"
  echo "  (Mac with Homebrew: brew install --cask miniforge, then run: conda init zsh, and reopen the Terminal)"
  exit 2
fi
if $CONDA env list | grep -qE '^cln3-msi[[:space:]]'; then
  echo "env cln3-msi exists, updating it"
  $CONDA env update -n cln3-msi -f environment.yml
else
  $CONDA env create -f environment.yml
fi

# `conda run` so we don't depend on `conda activate` working inside a script
$CONDA run -n cln3-msi pip install -e .
# optional extras: a failure here is fine, the pipeline has fallbacks
$CONDA run -n cln3-msi pip install -r requirements-optional.txt || echo "optional packages failed to install (that's OK)"
$CONDA run -n cln3-msi python -m ipykernel install --user --name cln3-msi --display-name "cln3-msi" || true

echo
echo "Done. Next:"
echo "  conda activate cln3-msi              # if it says 'run conda init': conda init zsh, then reopen the Terminal"
echo "  bash bin/run_synthetic_test.sh       # check the installation on fake data (~5 min)"
echo "  then set paths.data_root in config.yaml and run:"
echo "  python scripts/00_setup.py --config config.yaml --download"
