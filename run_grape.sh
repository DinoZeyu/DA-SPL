#!/usr/bin/env bash
# External HF training -> fixed GRAPE visit scores -> longitudinal A/B reports.
# Uses the existing Conda environment and GPU allocation; never creates a venv.
set -Eeuo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
cd -- "$project_root"
run_name=""
plan=""
training_plan="configs/hf_training.json"
hf_root="/scratch/users/zeyuhan/DA-SPL/archive/glaucoma_diagnosis_json_analysis"
source_options=0
device="cuda"
image_weights=""
image_batch_size=""
allow_download=0
conda_environment="da-spl-repro"

usage() {
    cat <<'HELP'
Usage: bash run_grape.sh [options]

Run inside your existing GPU-node shell; no scheduler resources are requested.
Default: train GlaBoost diagnosis detectors on retained HF data, freeze them,
then compare latest-visit versus longitudinal evidence on GRAPE. GRAPE labels
are used only for patient-grouped A/B logistic fitting and held-out evaluation.

  --training-plan FILE     External training configurations and analysis settings
                           (default: configs/hf_training.json)
  --hf-root DIRECTORY      Existing HF archive (default: project archive on scratch)
  --plan FILE              Alternative: use already trained external model bundles;
                           skips source training, incompatible with source options
  --run-name NAME          New name (default: grape_external + UTC timestamp + PID)
  --device DEVICE          cuda (default: all visible GPUs), cuda:N, or explicit cpu
  --image-weights FILE     Relocated image checkpoint matching the saved detector
  --image-batch-size N     Global image batch (default: 64 per selected GPU)
  --allow-download         Explicitly allow missing pretrained encoder downloads
  --offline                Require existing encoder weights (default)
  -h, --help               Show help without executing an analysis

Tree configurations are specified in the training plan and trained only on the
HF source data. Each detector is then fixed before GRAPE scoring. With --plan,
all model bundles must already exist and no diagnostic detector is trained.

Uses the existing da-spl-repro Conda environment. No installation or venv creation.
Both the raw GRAPE data and the retained HF archive are protected and never
downloaded or modified.
Weights/cache and analysis artifacts use project scratch links (.cache, artifacts).
Source-training reports and model artifacts use <run-name>_source/ under result/
and artifacts/. GRAPE reports use result/<run-name>/ and evaluation artifacts
use artifacts/<run-name>/. Existing runs are never overwritten. All configurations
are reported; no winner is selected from GRAPE performance.
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
        --plan|--training-plan|--hf-root|--run-name|--device|--image-weights|--image-batch-size)
            (($# >= 2)) || fail "$1 缺少参数。"
            [[ -n "$2" && "$2" != --* ]] || fail "$1 缺少参数。"
            case "$1" in
                --plan) plan="$2" ;;
                --training-plan) training_plan="$2"; source_options=1 ;;
                --hf-root) hf_root="$2"; source_options=1 ;;
                --run-name) run_name="$2" ;;
                --device) device="$2" ;;
                --image-weights) image_weights="$2" ;;
                --image-batch-size) image_batch_size="$2" ;;
            esac
            shift 2
            ;;
        *) fail "未知参数：$1。使用 --help 查看外部训练与 GRAPE 验证的运行方式。" ;;
    esac
done

if [[ -z "$run_name" ]]; then
    run_name="grape_external_$(date -u +%Y%m%dT%H%M%SZ)_$$"
fi
[[ "$run_name" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]{0,79}$ ]] || fail "运行名格式不正确。"
[[ "$device" == cpu || "$device" =~ ^cuda(:[0-9]+)?$ ]] || fail "--device 必须为 cpu、cuda 或 cuda:N。"
[[ -z "$image_batch_size" || "$image_batch_size" =~ ^[1-9][0-9]*$ ]] || fail "--image-batch-size 必须为正整数。"
workflow="hf"
if [[ -n "$plan" ]]; then
    ((source_options == 0)) || fail "--plan 使用已有模型，不能同时指定 --training-plan 或 --hf-root。"
    [[ -f "$plan" ]] || fail "未找到外部模型计划：$plan。"
    workflow="fixed"
else
    ((${#run_name} <= 73)) || fail "HF 训练流程的运行名最多 73 个字符，以保留 _source 后缀。"
    [[ -f "$training_plan" ]] || fail "未找到外部训练计划：$training_plan。"
    [[ -d "$hf_root" ]] || fail "未找到保留的 HF 原始数据：$hf_root。脚本不会重新下载。"
fi
[[ -z "$image_weights" || -f "$image_weights" ]] || fail "未找到 --image-weights 指定的文件。"

# Do not silently recreate large artifact/cache directories in home.
for storage_entry in artifacts .cache; do
    [[ -L "$project_root/$storage_entry" && -d "$project_root/$storage_entry" ]] || fail \
        "$storage_entry 必须是指向项目 scratch 目录的有效软链接；已停止，避免写满 home。"
done
grape_root="$project_root/data/raw/grape"
artifact_run="$project_root/artifacts/$run_name"
report_dir="$project_root/result/$run_name"
[[ -f "$grape_root/files/VF and clinical information.xlsx" && -d "$grape_root/extracted/CFPs" ]] || fail \
    "未找到保留的 GRAPE 原表或 CFP 目录；脚本不会重新下载数据。"
[[ ! -e "$artifact_run" && ! -L "$artifact_run" ]] || fail "分析目录已存在：$artifact_run。请换一个 --run-name。"
[[ ! -e "$report_dir" && ! -L "$report_dir" ]] || fail "报告目录已存在：$report_dir。请换一个 --run-name。"
if [[ "$workflow" == hf ]]; then
    for source_output in "${artifact_run}_source" "${report_dir}_source"; do
        [[ ! -e "$source_output" && ! -L "$source_output" ]] || fail "源数据训练目录已存在：$source_output。请换一个 --run-name。"
    done
fi

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
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export MPLBACKEND=Agg
export MPLCONFIGDIR="$project_root/.cache/matplotlib"
export UV_CACHE_DIR="$project_root/.cache/uv"
export PIP_CACHE_DIR="$project_root/.cache/pip"
export TORCH_HOME="$project_root/.cache/torch"
export HF_HOME="$project_root/.cache/huggingface"
trap 'printf "运行失败，已停止后续步骤。运行名：%s；已有文件保留供检查。\n" "$run_name" >&2' ERR

# Read-only storage preflight protects both retained data sources. Plan and model
# validation happen in the selected workflow before CUDA resolution or fitting.
"${python_command[@]}" - "$grape_root" "$hf_root" "$artifact_run" "$report_dir" "$MPLCONFIGDIR" "$workflow" <<'PY'
import sys
from pathlib import Path
from glaboost.study import ensure_outside_raw

root, hf_root, artifacts, report, plotting_cache, workflow = sys.argv[1:]
cache_root = Path(plotting_cache).parent
outputs = [artifacts, report, plotting_cache,
           *(cache_root / name for name in ("glaboost", "uv", "pip", "torch", "huggingface"))]
if workflow == "hf":
    outputs.extend((artifacts + "_source", report + "_source"))
for output in outputs:
    ensure_outside_raw(output, root)
    ensure_outside_raw(output, hf_root)
home_root = Path.home().resolve()
for output in (Path(artifacts), cache_root):
    resolved = output.resolve()
    if resolved == home_root or home_root in resolved.parents:
        raise ValueError(f"Model/cache storage must be outside home on project scratch: {resolved}")
if (Path(report).parent / "INDEX.md").is_symlink():
    raise ValueError("result/INDEX.md must not be a symbolic link")
print(f"Conda Python: {sys.executable}")
print(f"Analysis artifacts: {Path(artifacts).resolve()}")
print(f"Project cache: {cache_root.resolve()}")
print(f"Report directory: {Path(report).resolve()}")
if workflow == "hf":
    print(f"HF source model artifacts: {Path(artifacts + '_source').resolve()}")
    print(f"HF source reports: {Path(report + '_source').resolve()}")
PY

if [[ "$workflow" == hf ]]; then
    validation_command=("${python_command[@]}" -m glaboost run-hf-grape
        --training-plan "$training_plan" --hf-root "$hf_root")
    printf '外部训练计划：%s\nHF 数据：%s\n' "$training_plan" "$hf_root"
else
    validation_command=("${python_command[@]}" -m glaboost validate-grape --plan "$plan")
    printf '固定外部模型计划：%s\n' "$plan"
fi
validation_command+=(--root "$grape_root" --run-name "$run_name"
    --result-dir "$project_root/result" --artifact-dir "$project_root/artifacts"
    --device "$device" --cache-dir "$project_root/.cache/glaboost")
[[ -z "$image_weights" ]] || validation_command+=(--image-weights "$image_weights")
[[ -z "$image_batch_size" ]] || validation_command+=(--image-batch-size "$image_batch_size")
if ((allow_download)); then validation_command+=(--allow-download); fi

printf '运行名：%s\n' "$run_name"
"${validation_command[@]}"
printf '\n完成。报告：%s/report.html\n全部报告：%s/result/INDEX.md\n' "$report_dir" "$project_root"
