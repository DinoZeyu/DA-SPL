#!/usr/bin/env bash
# One run: frozen ResNet152 features -> GRAPE XGBoost training -> A/B reports.
# Uses the existing da-spl-repro Conda environment; never creates a venv.
set -Eeuo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
cd -- "$project_root"

# Prespecified primary analysis. Paths are relative to the project root.
run_name="grape_internal_$(date -u +%Y%m%dT%H%M%SZ)_$$"
device="cuda"
image_weights=""
allow_download=1
image_batch_size=""
xgb_threads=""
conda_environment="da-spl-repro"

usage() {
    cat <<'HELP'
Usage: bash run_grape.sh [options]

  --run-name NAME                    New run name (default: UTC timestamp + process ID)
  --device DEVICE                    cuda (default: all visible GPUs), cuda:N, or cpu
  --image-weights FILE               Optional local official ImageNet V1 ResNet152 checkpoint
  --image-batch-size N               Global image batch (default: 64 per GPU; two A100: 128)
  --xgb-threads N                    XGBoost host helper threads (default: 1)
  --offline                         Require cached/local encoder weights; do not download
  --allow-download                  Allow encoder downloads (default; useful after --offline)
  -h, --help                         Show this help without running anything

Runs in the existing da-spl-repro Conda environment. Trains separate PLR2, PLR3,
and MD-slope progression models using the paper's ResNet152 + XGBoost method.
Primary settings: >=3 CFP visits; outer 3-fold / inner 2-fold patient splitting;
seed 42; 2,000 patient bootstrap draws; persistence threshold 0.5; logistic C=1.
ImageNet encoder weights download to .cache/glaboost if missing. No raw data is
downloaded or modified. No environment installation or venv creation is performed.

Models:  artifacts/<run-name>/models/ (+ frozen features and provenance)
Report:  result/<run-name>/report.html (+ tables, figures, and audit files)
Reports stay in the project result directory. Model/package caches use .cache.
The default uses all CUDA GPUs visible to this process; it never falls back to CPU.
tqdm displays image, endpoint/fold training and bootstrap progress.
GPU: ResNet, XGBoost gpu_hist/prediction, temporal features, logistic heads, metrics,
and bootstrap. Independent endpoints run across the visible GPUs. Host code handles
file I/O, patient-split bookkeeping, task dispatch and report rendering.
This is retrospective INTERNAL validation on GRAPE, not a diagnosis reproduction.
Existing runs are never overwritten. Relative paths are resolved from this script.
HELP
}

fail() {
    printf '错误：%s\n' "$*" >&2
    exit 2
}

while (($#)); do
    case "$1" in
        -h|--help) usage; exit 0 ;;
        --allow-download) allow_download=1; shift ;;
        --offline) allow_download=0; shift ;;
        --run-name|--device|--image-weights|--image-batch-size|--xgb-threads)
            (($# >= 2)) || fail "$1 缺少参数。"
            [[ -n "$2" && "$2" != --* ]] || fail "$1 缺少参数。"
            case "$1" in
                --run-name) run_name="$2" ;;
                --device) device="$2" ;;
                --image-weights) image_weights="$2" ;;
                --image-batch-size) image_batch_size="$2" ;;
                --xgb-threads) xgb_threads="$2" ;;
            esac
            shift 2
            ;;
        *) fail "未知参数：$1。使用 --help 查看用法。" ;;
    esac
done

[[ "$run_name" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]{0,79}$ ]] || fail "运行名格式不正确。"
[[ "$device" == cpu || "$device" =~ ^cuda(:[0-9]+)?$ ]] || fail "--device 必须为 cpu、cuda 或 cuda:N。"
[[ -z "$image_batch_size" || "$image_batch_size" =~ ^[1-9][0-9]*$ ]] || fail "--image-batch-size 必须为正整数。"
[[ -z "$xgb_threads" || "$xgb_threads" =~ ^[1-9][0-9]*$ ]] || fail "--xgb-threads 必须为正整数。"

grape_root="$project_root/data/raw/grape"
artifact_run="$project_root/artifacts/$run_name"
report_dir="$project_root/result/$run_name"

[[ -f "$grape_root/files/VF and clinical information.xlsx" && -d "$grape_root/extracted/CFPs" ]] || fail \
    "未找到保留的 GRAPE 原表或 CFP 目录；脚本不会重新下载数据。"
[[ -z "$image_weights" || -f "$image_weights" ]] || fail "未找到 --image-weights 指定的文件。"
[[ ! -e "$artifact_run" && ! -L "$artifact_run" ]] || fail "模型目录已存在：$artifact_run。请换一个 --run-name。"
[[ ! -e "$report_dir" && ! -L "$report_dir" ]] || fail "报告目录已存在：$report_dir。请换一个 --run-name。"
if ((allow_download == 0)) && [[ -z "$image_weights" ]]; then
    [[ -f "$project_root/.cache/glaboost/torch/resnet152-394f9c45.pth" ]] || fail \
        "离线模式缺少 ImageNet 编码器权重。请提供 --image-weights，或去掉 --offline 以首次下载。"
fi

# Avoid accidentally using /usr/bin/python or a different activated environment.
if [[ "${CONDA_DEFAULT_ENV:-}" == "$conda_environment" && -n "${CONDA_PREFIX:-}" && -x "$CONDA_PREFIX/bin/python" ]]; then
    python_command=("$CONDA_PREFIX/bin/python")
elif command -v conda >/dev/null 2>&1; then
    python_command=(conda run --no-capture-output -n "$conda_environment" python)
elif [[ -n "${CONDA_EXE:-}" && -x "$CONDA_EXE" ]]; then
    python_command=("$CONDA_EXE" run --no-capture-output -n "$conda_environment" python)
elif [[ -x "$HOME/.conda/envs/$conda_environment/bin/python" ]]; then
    python_command=("$HOME/.conda/envs/$conda_environment/bin/python")
else
    fail "找不到 Conda 环境 $conda_environment；请先载入 Conda 或激活该环境。"
fi

export PYTHONPATH="$project_root/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
# Keep small tabular fits from spawning competing BLAS/OpenMP thread pools.
export OMP_NUM_THREADS="${xgb_threads:-1}"
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export MPLBACKEND=Agg
export MPLCONFIGDIR="$project_root/.cache/matplotlib"
export UV_CACHE_DIR="$project_root/.cache/uv"
export PIP_CACHE_DIR="$project_root/.cache/pip"
export TORCH_HOME="$project_root/.cache/torch"
export HF_HOME="$project_root/.cache/huggingface"
trap 'printf "运行失败，已停止后续步骤。运行名：%s；已有文件保留供检查。\n" "$run_name" >&2' ERR

# Read-only preflight, including symlink-aware protection of original data.
"${python_command[@]}" - "$grape_root" "$artifact_run" "$report_dir" "$MPLCONFIGDIR" "$device" <<'PY'
import sys
from pathlib import Path
from glaboost.encoders import resolve_image_devices
from glaboost.study import ensure_outside_raw
from glaboost.longitudinal import EvaluationConfig

root, artifacts, report, plotting_cache, device = sys.argv[1:]
cache_root = Path(plotting_cache).parent
for output in (artifacts, report, plotting_cache,
               *(cache_root / name for name in ("glaboost", "uv", "pip", "torch", "huggingface"))):
    ensure_outside_raw(output, root)
if (Path(report).parent / "INDEX.md").is_symlink():
    raise ValueError("result/INDEX.md must not be a symbolic link.")
resolve_image_devices(device)
EvaluationConfig(n_splits=3, seed=42, bootstrap_replicates=2000)
print(f"Conda Python: {sys.executable}")
PY

train_command=("${python_command[@]}" -m glaboost train-grape
    --root "$grape_root" --run-name "$run_name"
    --result-dir "$project_root/result" --artifact-dir "$project_root/artifacts"
    --min-visits 3 --device "$device" --cache-dir "$project_root/.cache/glaboost"
    --folds 3 --inner-folds 2 --seed 42 --bootstrap 2000
    --persistence-threshold 0.5 --logistic-c 1.0)
[[ -z "$image_weights" ]] || train_command+=(--image-weights "$image_weights")
[[ -z "$image_batch_size" ]] || train_command+=(--image-batch-size "$image_batch_size")
[[ -z "$xgb_threads" ]] || train_command+=(--xgb-threads "$xgb_threads")
if ((allow_download)); then
    train_command+=(--allow-download)
fi

printf '运行名：%s\n[1/2] 检查 GRAPE 队列\n' "$run_name"
"${python_command[@]}" -m glaboost inspect-grape --root "$grape_root" --min-visits 3

printf '[2/2] ResNet-152 特征、XGBoost 训练、患者级 A/B 评估与报告\n'
"${train_command[@]}"

printf '\n完成。报告：%s/report.html\n全部报告：%s/result/INDEX.md\n' "$report_dir" "$project_root"
