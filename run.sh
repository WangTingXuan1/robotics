#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if [[ "${CONDA_DEFAULT_ENV:-}" != "robotics-exp1" ]]; then
  if ! command -v conda >/dev/null 2>&1; then
    for base in "$HOME/anaconda3" "$HOME/miniconda3" "$HOME/miniforge3"; do
      if [[ -f "$base/etc/profile.d/conda.sh" ]]; then
        source "$base/etc/profile.d/conda.sh"
        break
      fi
    done
  fi
  if ! command -v conda >/dev/null 2>&1; then
    echo "Conda not found. Install Conda and run ./setup.sh first." >&2
    exit 1
  fi
  eval "$(conda shell.bash hook)"
  conda activate robotics-exp1
fi
export PYOPENGL_PLATFORM=egl
export MPLBACKEND=Agg

case "${1:-all}" in
  generate) python experiment.py generate --samples-per-class "${2:-50}" ;;
  train) python experiment.py train --epochs "${2:-35}" ;;
  evaluate) python experiment.py evaluate ;;
  compare)
    python model_sweep.py --seeds 42 43 44 --epochs 45 --patience 10
    python ensemble_eval.py
    ;;
  all)
    python experiment.py generate --samples-per-class "${2:-50}"
    python experiment.py train --epochs "${3:-35}"
    python experiment.py evaluate
    ;;
  *) echo "Usage: ./run.sh [generate N|train EPOCHS|evaluate|compare|all N EPOCHS]" >&2; exit 2 ;;
esac
