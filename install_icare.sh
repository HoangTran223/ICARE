#!/usr/bin/env bash
# Create conda env "icare" and install ICARE dependencies (portable across machines).
# Usage (from repo root):
#   bash install_icare.sh
#   conda activate icare
#   python -m spacy download en_core_web_sm
#   bash scripts/gpt2_120m/ResidualKD_full.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="${ICARE_CONDA_ENV:-icare}"
PYTHON_VERSION="${ICARE_PYTHON_VERSION:-3.13}"

# Symlink bundled assets from ICARE_final when missing (portable handoff).
link_if_missing() {
  local name="$1"
  local src="${REPO_ROOT}/ICARE_final/${name}"
  local dst="${REPO_ROOT}/${name}"
  if [[ ! -e "${dst}" && -e "${src}" ]]; then
    ln -s "${src}" "${dst}"
    echo "Linked ${name} -> ICARE_final/${name}"
  fi
}
link_if_missing "ALM"
link_if_missing "ResidualKD_MTA"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda not found. Install Miniconda/Anaconda first." >&2
  exit 1
fi

# shellcheck source=/dev/null
source "$(conda info --base)/etc/profile.d/conda.sh"

if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  echo "Conda env '${ENV_NAME}' already exists; activating and updating packages."
  conda activate "${ENV_NAME}"
else
  echo "Creating conda env '${ENV_NAME}' (python=${PYTHON_VERSION})..."
  conda create -n "${ENV_NAME}" "python=${PYTHON_VERSION}" -y
  conda activate "${ENV_NAME}"
fi

echo "Installing CUDA nvcc (required by DeepSpeed)..."
conda install -y -c nvidia cuda-nvcc=12.4

echo "Installing Python packages from requirements.txt..."
pip install -r "${REPO_ROOT}/requirements.txt"

# ALM: vendored tokenkit at ALM/tokenkit-main (not PyPI `pip install tokenkit`)
if [[ ! -d "${REPO_ROOT}/ALM/tokenkit-main" ]]; then
  for _alm_src in "${REPO_ROOT}/ICARE_final/ALM" "/mnt/hungpv/projects/ALM"; do
    if [[ -d "${_alm_src}/tokenkit-main" ]]; then
      ln -sfn "${_alm_src}" "${REPO_ROOT}/ALM"
      echo "Linked ALM -> ${_alm_src}"
      break
    fi
  done
fi
if [[ -d "${REPO_ROOT}/ALM/tokenkit-main" ]]; then
  if pip show tokenkit >/dev/null 2>&1; then
    _tk="$(python -c "import tokenkit; print(getattr(tokenkit,'__file__',''))" 2>/dev/null || true)"
    if [[ -n "${_tk}" && "${_tk}" != *"${REPO_ROOT}/ALM/tokenkit-main"* ]]; then
      pip uninstall -y tokenkit
    fi
  fi
  pip install -e "${REPO_ROOT}/ALM/tokenkit-main" --no-deps
fi

echo ""
echo "Done. Next steps:"
echo "  conda activate ${ENV_NAME}"
echo "  cd ${REPO_ROOT}   # copy this folder anywhere (e.g. ~/ICARE)"
echo "  python -m spacy download en_core_web_sm   # ResidualKD span matching"
echo "  # Place weights in model_hub/ and data in data/dolly/"
echo "  bash scripts/gpt2_120m/ALM_IMPACT.sh"
