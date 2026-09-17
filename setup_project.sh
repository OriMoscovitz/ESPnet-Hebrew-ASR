from pathlib import Path

script = r'''#!/usr/bin/env bash
set -euo pipefail

# ESPnet-Hebrew-ASR bootstrap
# Run from the root of a cloned ESPnet-Hebrew-ASR repository.
# Target: Ubuntu/Debian Linux or WSL2.
#
# NVIDIA note:
# Install a working NVIDIA driver on the host before running GPU experiments.
# This script does not install/replace the host GPU driver.

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ESPNET_DIR="$PROJECT_ROOT/espnet"
PYTHON_BIN="${PYTHON_BIN:-python3}"
ESPNET_TAG="${ESPNET_TAG:-v.202604}"

echo "==> Project root: $PROJECT_ROOT"

if [[ ! -d "$PROJECT_ROOT/.git" ]]; then
    echo "ERROR: Run this script from the cloned ESPnet-Hebrew-ASR repository."
    exit 1
fi

if ! command -v sudo >/dev/null 2>&1; then
    echo "ERROR: sudo is required for system packages."
    exit 1
fi

echo "==> Installing system dependencies..."
sudo apt-get update
sudo apt-get install -y git git-lfs build-essential cmake ffmpeg sox flac libsndfile1-dev python3 python3-dev python3-venv python3-pip curl wget

echo "==> Initializing Git LFS and downloading repository LFS files..."
git lfs install
git lfs pull

echo "==> Cloning ESPnet ($ESPNET_TAG)..."
if [[ ! -d "$ESPNET_DIR/.git" ]]; then
    git clone --branch "$ESPNET_TAG" --depth 1 https://github.com/espnet/espnet.git "$ESPNET_DIR"
else
    echo "    ESPnet already exists; leaving it in place."
fi

echo "==> Creating ESPnet Python environment..."
cd "$ESPNET_DIR/tools"
if [[ ! -f activate_python.sh ]]; then
    ./setup_venv.sh "$(command -v "$PYTHON_BIN")"
fi

# Activate the environment created by ESPnet.
source "$ESPNET_DIR/tools/activate_python.sh"

echo "==> Updating packaging tools..."
python -m pip install --upgrade pip setuptools wheel

echo "==> Installing ESPnet ASR dependencies..."
cd "$ESPNET_DIR"
python -m pip install -e ".[asr]"

echo "==> Installing project-specific Python packages..."
python -m pip install datasets huggingface_hub soundfile tensorboard pandas matplotlib jiwer

echo "==> Rebuilding the Hebrew Campus subset..."
cd "$PROJECT_ROOT"
python prepare_campus_subset.py

echo "==> Rebuilding token list..."
python create_hebrew_tokens.py

echo "==> Rewriting project-local wav.scp paths..."
python - "$PROJECT_ROOT" <<'PY'
from pathlib import Path
import sys

root = Path(sys.argv[1]).resolve()

# The campus preparation script already writes correct absolute paths.
# Fix the custom my_test wav.scp so the repository can live anywhere.
my_test = root / "data" / "my_test" / "wav.scp"
recordings = root / "data" / "heb_recordings"

if my_test.exists():
    output = []
    for line in my_test.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        utt, old_path = line.split(maxsplit=1)
        filename = Path(old_path).name
        output.append(f"{utt} {recordings / filename}")
    my_test.write_text("\n".join(output) + "\n", encoding="utf-8")

# Fix any old absolute paths that may remain in campus wav.scp files.
for split in ("train", "dev"):
    scp = root / "hebrew_campus_subset" / "data" / split / "wav.scp"
    if scp.exists():
        output = []
        for line in scp.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            utt, old_path = line.split(maxsplit=1)
            filename = Path(old_path).name
            output.append(f"{utt} {root / 'hebrew_campus_subset' / 'audio' / filename}")
        scp.write_text("\n".join(output) + "\n", encoding="utf-8")
PY

echo "==> Checking ESPnet installation..."
cd "$ESPNET_DIR/tools"
bash -c ". ./activate_python.sh; python3 check_install.py" || {
    echo
    echo "WARNING: ESPnet check_install.py reported optional/missing components."
    echo "Review the messages above. Core ASR may still be usable."
}

cd "$PROJECT_ROOT"

echo
echo "============================================================"
echo "Setup complete."
echo "============================================================"
echo
echo "Project: $PROJECT_ROOT"
echo "ESPnet:  $ESPNET_DIR ($ESPNET_TAG)"
echo
echo "For a new terminal, activate ESPnet with:"
echo "  source \"$ESPNET_DIR/tools/activate_python.sh\""
echo
echo "Check GPU access with:"
echo "  python -c \"import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')\""
echo
echo "TensorBoard:"
echo "  tensorboard --logdir \"$PROJECT_ROOT/exp\""
echo
echo "NOTE: trained .pth checkpoints are not stored in this repository."
echo "      Existing TensorBoard logs and selected decoding results are restored by Git LFS."
'''

path = Path("/mnt/data/setup_project.sh")
path.write_text(script, encoding="utf-8", newline="\n")
path.chmod(0o755)
print(f"Created {path}")
