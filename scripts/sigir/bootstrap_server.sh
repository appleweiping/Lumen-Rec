#!/usr/bin/env bash
# Bootstrap a fresh SeetaCloud/AutoDL GPU instance for the Lumen-Rec SIGIR project.
# Idempotent: safe to re-run. Usage (on server):  bash scripts/sigir/bootstrap_server.sh [--models]
# Large artifacts go to the data disk: models to $LUMEN_DATA (default /root/autodl-tmp/lumen); data/ and outputs/
# are repo-relative, so the repo itself must live on the data disk (e.g. /root/autodl-tmp/lumen-rec).
# Raw data is fetched by the run scripts, not here: rebuild_panels.sh (Amazon, slimmed), run_pilot1_mirror.sh
# (ML-1M, Amazon Toys), run_pilot2_3.sh (KuaiRec, Amazon Toys + Video_Games).
set -euo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd -P)"
LUMEN_DATA="${LUMEN_DATA:-/root/autodl-tmp/lumen}"
ENV_NAME="${ENV_NAME:-lumen}"
# Space-separated ModelScope ids; default Qwen only (data disk is 50 GB — add Llama when needed).
MODELS="${MODELS:-Qwen/Qwen3-8B}"
export PATH="/root/miniconda3/bin:$PATH"   # AutoDL: conda is not on the non-interactive PATH
DO_MODELS=0
for a in "$@"; do
  case "$a" in
    --models) DO_MODELS=1 ;;
    --data) echo "NOTE: --data is gone; the run scripts download and slim what they need" ;;
  esac
done
case "$REPO" in
  /root/autodl-tmp/*) ;;
  *) echo "WARNING: repo $REPO is not under /root/autodl-tmp (data disk); data/ and outputs/ may fill the system disk" >&2 ;;
esac

echo "== hardware"; nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
nvcc --version 2>/dev/null | tail -1 || true
df -h / "$(dirname "$LUMEN_DATA")" 2>/dev/null || true; free -g | head -2; nproc

# NOTE: do NOT source /etc/network_turbo here — it proxies traffic and slows the domestic pip/conda mirrors
# (and ModelScope / hf-mirror are domestic anyway). Use it only for GitHub operations.
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export PIP_INDEX_URL="${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"

mkdir -p "$LUMEN_DATA"/{models,runs,cache}
export HF_HOME="$LUMEN_DATA/cache/hf"

# ---- python env (conda if present, else venv) ----
if command -v conda >/dev/null 2>&1; then
  CONDA_BASE="$(conda info --base)"
  set +u
  source "$CONDA_BASE/etc/profile.d/conda.sh"
  # repo.anaconda.com ("defaults") is unreachable from AutoDL: pin the Tsinghua mirror explicitly.
  if [ ! -d "$CONDA_BASE/envs/$ENV_NAME" ]; then
    conda create -y -n "$ENV_NAME" python=3.11 --override-channels \
      -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
  fi
  conda activate "$ENV_NAME"
  set -u
else
  [ -d "$LUMEN_DATA/venv" ] || python3 -m venv "$LUMEN_DATA/venv"
  source "$LUMEN_DATA/venv/bin/activate"
fi
python -m pip install -U pip wheel
# vLLM pins a compatible torch; install it first, then the rest. Versions are pinned: the C0 numerics were probed
# on vllm 0.30.0, and pandas 2 parses the Amazon ms timestamps as ns (no rebuilt panel could match the refs).
python -m pip install "vllm==0.30.0" 2>&1 | tail -2
python -m pip install "pandas>=3,<4" transformers peft accelerate datasets trl bitsandbytes \
  pyarrow scipy scikit-learn statsmodels matplotlib seaborn tqdm pyyaml requests \
  modelscope sentencepiece protobuf ujson 2>&1 | tail -2
python - <<'PY'
import sys
import numpy, pandas, torch, transformers
print("python", sys.executable, sys.version.split()[0])
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), torch.cuda.device_count())
print("transformers", transformers.__version__, "| numpy", numpy.__version__, "| pandas", pandas.__version__)
try:
    import vllm
    print("vllm", vllm.__version__)
except Exception as e:
    sys.exit(f"vllm import failed: {e}")
bad = [f"pandas {pandas.__version__} (need >=3,<4)"] if not pandas.__version__.startswith("3.") else []
bad += [f"vllm {vllm.__version__} (need 0.30.0)"] if vllm.__version__ != "0.30.0" else []
sys.exit("version pin violated: " + "; ".join(bad) if bad else 0)
PY
echo "== interpreter: $(command -v python)   (non-interactive runs: export PYTHON=$(command -v python))"

# ---- models (ModelScope is fast inside mainland China) ----
# Always invoke the download: it skips complete files and resumes partial ones (a lone config.json from an
# interrupted first run must not count as a finished model).
if [ "$DO_MODELS" = 1 ]; then
  for m in $MODELS; do
    tgt="$LUMEN_DATA/models/$(basename "$m")"
    modelscope download --model "$m" --local_dir "$tgt"
  done
fi
echo "== bootstrap done: repo $REPO, data disk $LUMEN_DATA"
