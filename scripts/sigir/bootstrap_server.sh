#!/usr/bin/env bash
# Bootstrap a fresh SeetaCloud/AutoDL GPU instance for the Lumen-Rec SIGIR project.
# Idempotent: safe to re-run. Usage (on server):  bash scripts/sigir/bootstrap_server.sh [--models] [--data]
# Large artifacts go to the data disk ($LUMEN_DATA, default /root/autodl-tmp/lumen), never into git.
set -euo pipefail

LUMEN_DATA="${LUMEN_DATA:-/root/autodl-tmp/lumen}"
ENV_NAME="${ENV_NAME:-lumen}"
# Space-separated ModelScope ids; default Qwen only (data disk is 50 GB — add Llama when needed).
MODELS="${MODELS:-Qwen/Qwen3-8B}"
export PATH="/root/miniconda3/bin:$PATH"   # AutoDL: conda is not on the non-interactive PATH
DO_MODELS=0; DO_DATA=0
for a in "$@"; do case "$a" in --models) DO_MODELS=1;; --data) DO_DATA=1;; esac; done

echo "== hardware"; nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
nvcc --version 2>/dev/null | tail -1 || true
df -h / "$(dirname "$LUMEN_DATA")" 2>/dev/null || true; free -g | head -2; nproc

# NOTE: do NOT source /etc/network_turbo here — it proxies traffic and slows the domestic pip/conda mirrors
# (and ModelScope / hf-mirror are domestic anyway). Use it only for GitHub operations.
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export PIP_INDEX_URL="${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"

mkdir -p "$LUMEN_DATA"/{models,data,runs,cache}
export HF_HOME="$LUMEN_DATA/cache/hf"

# ---- python env (conda if present, else venv) ----
if command -v conda >/dev/null 2>&1; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  # repo.anaconda.com ("defaults") is unreachable from AutoDL: pin the Tsinghua mirror explicitly.
  conda env list | grep -q "^$ENV_NAME " || conda create -y -n "$ENV_NAME" python=3.11 --override-channels \
    -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
  conda activate "$ENV_NAME"
else
  python3 -m venv "$LUMEN_DATA/venv" && source "$LUMEN_DATA/venv/bin/activate"
fi
python -m pip install -U pip wheel
# vLLM pins a compatible torch; install it first, then the rest.
python -m pip install "vllm>=0.8.5" 2>&1 | tail -2
python -m pip install transformers peft accelerate datasets trl bitsandbytes \
  pandas pyarrow scipy scikit-learn statsmodels matplotlib seaborn tqdm pyyaml \
  modelscope sentencepiece protobuf ujson 2>&1 | tail -2
python - <<'PY'
import torch, transformers
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), torch.cuda.device_count())
print("transformers", transformers.__version__)
try:
    import vllm; print("vllm", vllm.__version__)
except Exception as e: print("vllm import failed:", e)
PY

# ---- models (ModelScope is fast inside mainland China) ----
if [ "$DO_MODELS" = 1 ]; then
  for m in $MODELS; do
    tgt="$LUMEN_DATA/models/$(basename "$m")"
    [ -f "$tgt/config.json" ] || modelscope download --model "$m" --local_dir "$tgt"
  done
fi

# ---- raw data ----
if [ "$DO_DATA" = 1 ]; then
  cd "$LUMEN_DATA/data"
  [ -d ml-1m ] || { wget -q https://files.grouplens.org/datasets/movielens/ml-1m.zip && unzip -q ml-1m.zip; }
  # Amazon Reviews 2023 is NOT downloaded raw (tens of GB). Run, from the repo root:
  #   python scripts/sigir/slim_amazon2023.py --domains sports,toys,home,tools,beauty --root data/raw
  # which streams each category from the HF mirror and keeps only the fields Lumen reads.
fi
echo "== bootstrap done: $LUMEN_DATA"
