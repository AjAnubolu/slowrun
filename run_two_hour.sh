#!/usr/bin/env bash
# One-shot runner for the NanoGPT Slowrun two-hour track attempt
# (baseline + document-level shuffling + learnable XSA).
#
# Usage on the cluster (zero setup):
#   curl -sL https://raw.githubusercontent.com/AjAnubolu/slowrun/two-hour-doc-shuffle-xsa/run_two_hour.sh | bash
# or, if you already cloned the repo:
#   ./run_two_hour.sh [extra args passed through to two_hour/train.py]
#
# Optional environment variables:
#   HF_TOKEN         HuggingFace token (avoids FineWeb download rate-limits). Recommended.
#   WANDB_API_KEY    Weights & Biases key for the live run link (needed for the record PR).
#                    If unset, the run logs to wandb OFFLINE so it never blocks.
#   RUN_ID           Run name (default: two_hour_<timestamp>).
#   NPROC            GPUs to use (default: auto-detect; the valid record config is 8xH100).
set -euo pipefail

REPO_URL="https://github.com/AjAnubolu/slowrun.git"
BRANCH="baseline-offline"
RUN_ID="${RUN_ID:-two_hour_$(date +%Y%m%d_%H%M%S)}"

# --- 0. Get into the repo (clone if we're being piped in via curl) -----------
if [ ! -f two_hour/train.py ]; then
  if [ ! -d slowrun ]; then
    echo ">>> Cloning $REPO_URL ($BRANCH)"
    git clone --branch "$BRANCH" --single-branch "$REPO_URL"
  fi
  cd slowrun
fi
# Sync an existing clone to the latest branch tip (untracked data/kernel dirs are preserved).
echo ">>> Syncing repo to latest $BRANCH"
git fetch --depth 1 origin "$BRANCH" 2>/dev/null && git reset --hard FETCH_HEAD 2>/dev/null \
  || echo "!!! could not sync (offline?); using existing checkout"
echo ">>> Working dir: $(pwd)  |  commit: $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"

# --- 1. Dependencies ---------------------------------------------------------
echo ">>> Installing requirements"
pip install -q -r requirements.txt

# --- 2. Secrets / logging mode ----------------------------------------------
if [ -z "${HF_TOKEN:-}" ]; then
  echo "!!! HF_TOKEN not set — FineWeb is public so this may still work, but could hit rate limits."
else
  export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"   # datasets reads either name
fi
if [ -z "${WANDB_API_KEY:-}" ]; then
  echo "!!! WANDB_API_KEY not set — logging to wandb OFFLINE (no live link; fine for a first run)."
  export WANDB_MODE=offline
fi

# --- 3. Data: pull pre-built CANONICAL files from the GitHub release ----------
# The cluster nodes can't reach huggingface.co reliably (and FineWeb needs auth),
# so the canonical 100M/10M-token files are hosted as release assets and verified
# by SHA256 (same hashes prepare_data.py asserts). Falls back to HF if both unset.
DATA_BASE="https://github.com/AjAnubolu/slowrun/releases/download/data-v1"
TRAIN_SHA="36e7c95c1e7f6ed952fb002d76a03044e8617fea7e696a68d7dc1ce78465dcaf"
VAL_SHA="6868ed375b289a89c72c2f9df1ecbdcff700c4b9478ca806435d2dbfad8573b1"
mkdir -p fineweb_data
if [ -f fineweb_data/fineweb_train.pt ] && [ -f fineweb_data/fineweb_val.pt ]; then
  echo ">>> FineWeb data already present, skipping download"
else
  echo ">>> Downloading pre-built canonical FineWeb data from GitHub release"
  curl -fL --retry 5 --retry-delay 3 -o fineweb_data/fineweb_val.pt   "$DATA_BASE/fineweb_val.pt"
  curl -fL --retry 5 --retry-delay 3 -o fineweb_data/fineweb_train.pt "$DATA_BASE/fineweb_train.pt"
fi
echo ">>> Verifying data integrity (SHA256)"
echo "${VAL_SHA}  fineweb_data/fineweb_val.pt"     | sha256sum -c - || { echo "ERROR: val data hash mismatch"; exit 1; }
echo "${TRAIN_SHA}  fineweb_data/fineweb_train.pt" | sha256sum -c - || { echo "ERROR: train data hash mismatch"; exit 1; }

# --- 3b. FA3 kernel (node can't reach huggingface.co; load a staged copy) -----
# kernels.get_local_kernel() runs the same variant-resolution logic offline, so
# this is the identical FA3 binary the hub would serve, just relayed via GitHub.
KERNEL_DIR="$PWD/fa3_kernel"
if [ ! -d "$KERNEL_DIR/build" ]; then
  echo ">>> Downloading staged FA3 kernel (cu128) from GitHub release"
  mkdir -p "$KERNEL_DIR"
  curl -fL --retry 5 --retry-delay 3 -o fa3_cu128.tgz "$DATA_BASE/fa3_cu128.tgz"
  tar -xzf fa3_cu128.tgz -C "$KERNEL_DIR"
  rm -f fa3_cu128.tgz
fi
export SLOWRUN_FA3_REPO="$KERNEL_DIR"
echo ">>> SLOWRUN_FA3_REPO=$SLOWRUN_FA3_REPO  builds: $(ls "$KERNEL_DIR/build" 2>/dev/null | tr '\n' ' ')"

# --- 3c. Python dev headers (Triton JITs cuda_utils.c, needs Python.h) --------
# The node lacks the python3.12-dev system package, so stage the exact 3.12.3
# headers and add them to the compiler search path (no sudo needed).
if ! python -c "import sysconfig,os,sys; sys.exit(0 if os.path.exists(os.path.join(sysconfig.get_path('include'),'Python.h')) else 1)" 2>/dev/null; then
  HDR_DIR="$PWD/py312_headers"
  if [ ! -f "$HDR_DIR/python3.12/Python.h" ]; then
    echo ">>> Downloading staged Python 3.12 dev headers from GitHub release"
    mkdir -p "$HDR_DIR"
    curl -fL --retry 5 --retry-delay 3 -o py312_headers.tgz "$DATA_BASE/py312_headers.tgz"
    tar -xzf py312_headers.tgz -C "$HDR_DIR"
    rm -f py312_headers.tgz
  fi
  export CPATH="$HDR_DIR/python3.12:$HDR_DIR:$HDR_DIR/x86_64-linux-gnu/python3.12:${CPATH:-}"
  echo ">>> Staged Python.h on CPATH (no system python3.12-dev): $HDR_DIR/python3.12"
else
  echo ">>> System Python.h present; Triton JIT will use it"
fi

# --- 4. GPU sanity -----------------------------------------------------------
DETECTED=$(nvidia-smi -L 2>/dev/null | wc -l || echo 0)
NPROC="${NPROC:-$DETECTED}"
if [ "$NPROC" -eq 0 ]; then echo "ERROR: no GPUs detected"; exit 1; fi
echo ">>> Detected $DETECTED GPU(s); using nproc_per_node=$NPROC"
nvidia-smi -L | head -1 || true
if [ "$NPROC" -ne 8 ]; then
  echo "!!! WARNING: the valid two-hour record config is a single 8xH100 node."
  echo "!!! Running with $NPROC GPUs is fine for testing but is NOT a submittable record."
fi

# --- 5. Train ----------------------------------------------------------------
mkdir -p runs
echo ">>> Launching two-hour run '$RUN_ID' (defaults: doc-shuffle ON, xsa-mode=first6, 22 epochs)"
echo ">>> WATCH the early 'eta:' line — the cap is 120 min and the record runs ~118 min."
echo ">>>   If eta projects > ~119 min, re-run with: NPROC=$NPROC ./run_two_hour.sh --max-train-steps <N>"
# NOTE: do not pass --run here. This torchrun version greedily matches the
# script's --run flag to its own --run-path (argparse prefix matching), so the
# script auto-names the run by timestamp instead. Our log file uses RUN_ID below.
set -x
torchrun --standalone --nproc_per_node="$NPROC" two_hour/train.py "$@" 2>&1 | tee "runs/${RUN_ID}.log"
set +x
echo ">>> Done. Log saved to runs/${RUN_ID}.log"
echo ">>> Final val loss (target: beat 3.144):"
grep -iE "val/loss|best_val|final" "runs/${RUN_ID}.log" | tail -5 || true
