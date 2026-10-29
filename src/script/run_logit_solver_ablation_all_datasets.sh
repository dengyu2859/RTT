#!/usr/bin/env bash
# 依次运行所选数据集的求解器对照；只写临时 YAML，不改源配置。
set -eo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
OUTPUT_ROOT="$PROJECT_ROOT/Outputs/logit_solver_ablation"
# 可改为 (adam) 或 (closed_form)，只运行一个求解器。
LOGIT_SOLVERS=(adam closed_form)
# 选择要循环的数据集：删除或注释掉不需要的 YAML 即可。
CONFIGS=(
    "$PROJECT_ROOT/src/configs/AGNews.yaml"
    "$PROJECT_ROOT/src/configs/CIFAR10.yaml"
    "$PROJECT_ROOT/src/configs/DBPedia.yaml"
    "$PROJECT_ROOT/src/configs/FashionMNIST.yaml"
    "$PROJECT_ROOT/src/configs/GTSRB.yaml"
    "$PROJECT_ROOT/src/configs/ImageNet.yaml"
    "$PROJECT_ROOT/src/configs/LOAN.yaml"
    "$PROJECT_ROOT/src/configs/SpeechCommands.yaml"
)

# 仅覆盖块内写出的参数；GPU、margin、模型等默认沿用各自 YAML。
# 恶意客户端选择沿用现有 run_poisoned_experiments.sh。
# 可在任意块添加 gpu、adaptive_margin、desired_margin、up_margin_step 等。
declare -A DATASET_OVERRIDES=(
    [AGNews]='
attack: 1
attack_type: Following_Nature
malicious: [2, 7]
'
    [CIFAR10]='
attack: 1
attack_type: Following_Nature
malicious: [0]
'
    [DBPedia]='
attack: 1
attack_type: Following_Nature
malicious: [2, 7]
'
    [FashionMNIST]='
attack: 1
attack_type: Following_Nature
malicious: [0]
'
    [GTSRB]='
attack: 1
attack_type: Following_Nature
malicious: [0]
'
    [ImageNet]='
attack: 1
attack_type: Following_Nature
malicious: [0]
'
    [LOAN]='
attack: 1
attack_type: Following_Nature
malicious: [2]
'
    [SpeechCommands]='
attack: 1
attack_type: Following_Nature
malicious: [2, 7]
'
)
DRY_RUN=0
GPU_OVERRIDE=""

usage() {
    cat <<'EOF'
Usage: bash src/script/run_logit_solver_ablation_all_datasets.sh [options]

Edit CONFIGS to choose datasets and LOGIT_SOLVERS to choose adam/closed_form.
Defaults: eight datasets, two solvers each, sixteen experiments sequentially.
Edit DATASET_OVERRIDES for temporary per-dataset parameters. Source YAMLs
are never modified. Within each dataset only logit_solver differs between
the two runs; all other merged settings are identical.
Margin mode, step sizes and GPU retain each YAML's values unless overridden.
The default overrides enable attack and reuse the malicious client IDs from
the existing poisoned-experiment script. They do not enable adaptive_margin.

Options:
  --gpu N             Override GPU for all runs; -1 selects CPU.
  --output-root PATH  Default: <project>/Outputs/logit_solver_ablation.
  --dry-run           Validate all runs and print effective YAMLs/commands.
  -h, --help          Show this help.

Outputs: <output-root>/<dataset>/<solver>/train.log, config.yaml, and plots.
Final-round logs contain mean L2/cosine and cumulative solver time.
Model/trigger saving remains disabled in main.py.
Existing run directories cause preflight rejection before any training.
Failed runs retain logs and later dataset/solver combinations continue.
EOF
}

while (($#)); do
    case "$1" in
        --gpu|--output-root)
            if (($# < 2)) || [[ -z "$2" ]]; then
                echo "Missing value for $1" >&2
                exit 2
            fi
            if [[ "$1" == --gpu ]]; then
                GPU_OVERRIDE="$2"
            else
                OUTPUT_ROOT="$2"
            fi
            shift 2
            ;;
        --dry-run) DRY_RUN=1; shift ;;
        -h|--help) usage; exit 0 ;;
        --*) echo "Unknown option: $1" >&2; exit 2 ;;
        *) echo "Select YAML files by editing CONFIGS at the top of this script." >&2; exit 2 ;;
    esac
done

source /home/a100/anaconda3/etc/profile.d/conda.sh
conda activate WPAD
set -u
export MPLBACKEND=Agg
TEMP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/nba-all-dataset-solvers.XXXXXX")"
trap 'rm -rf -- "$TEMP_DIR"' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

OVERRIDE_ARGS=()
for dataset in "${!DATASET_OVERRIDES[@]}"; do
    OVERRIDE_ARGS+=("$dataset" "${DATASET_OVERRIDES[$dataset]}")
done
python - "$OUTPUT_ROOT" "$TEMP_DIR" "$GPU_OVERRIDE" \
    "${#LOGIT_SOLVERS[@]}" "${LOGIT_SOLVERS[@]}" \
    "${#DATASET_OVERRIDES[@]}" "${OVERRIDE_ARGS[@]}" "${CONFIGS[@]}" <<'PY'
import math
from pathlib import Path
import re
import sys
import yaml

arguments = iter(sys.argv[1:])
output_root, temporary = (Path(next(arguments)).resolve() for _ in range(2))
gpu_text = next(arguments)
solvers = [next(arguments) for _ in range(int(next(arguments)))]
overrides = {next(arguments): next(arguments) for _ in range(int(next(arguments)))}
sources = [Path(value).resolve() for value in arguments]
if not sources:
    raise SystemExit('CONFIGS must contain at least one YAML file')
if not solvers or len(solvers) != len(set(solvers)) or not set(solvers) <= {'adam', 'closed_form'}:
    raise SystemExit('LOGIT_SOLVERS must contain unique adam/closed_form values')
gpu = None
if gpu_text:
    try:
        gpu = int(gpu_text)
    except ValueError:
        raise SystemExit('--gpu must be an integer >= -1')
    if gpu < -1:
        raise SystemExit('--gpu must be an integer >= -1')

runs = []
datasets = set()
for source in sources:
    with source.open(encoding='utf-8') as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise SystemExit(f'{source}: YAML must contain a parameter mapping')
    dataset = config.get('dataset')
    if not isinstance(dataset, str) or not re.fullmatch(r'[A-Za-z0-9_]+', dataset):
        raise SystemExit(f'{source}: invalid dataset name')
    if dataset in datasets:
        raise SystemExit(f'Duplicate dataset in CONFIGS: {dataset}')
    datasets.add(dataset)
    changes = yaml.safe_load(overrides.get(dataset, '')) or {}
    if not isinstance(changes, dict):
        raise SystemExit(f'{dataset}: DATASET_OVERRIDES must be a YAML mapping')
    if changes.get('dataset', dataset) != dataset:
        raise SystemExit(f'{dataset}: DATASET_OVERRIDES cannot change dataset')
    config.update(changes)
    if gpu is not None:
        config['gpu'] = gpu
    if config.get('attack') != 1 or config.get('attack_type') != 'Following_Nature':
        raise SystemExit(f'{dataset}: enable attack: 1 and attack_type: Following_Nature')
    for field in ('epochs', 'clients', 'local_ep', 'distill_ep'):
        if type(config.get(field)) is not int or config[field] < 1:
            raise SystemExit(f'{dataset}: {field} must be a positive integer')
    malicious = config.get('malicious')
    if (not isinstance(malicious, list) or not malicious
            or any(type(cid) is not int or not 0 <= cid < config['clients'] for cid in malicious)
            or len(malicious) != len(set(malicious))):
        raise SystemExit(f'{dataset}: malicious must contain distinct valid client IDs')
    for field in ('adaptive_margin', 'use_global_feedback'):
        if type(config.get(field, True)) is not bool:
            raise SystemExit(f'{dataset}: {field} must be a YAML boolean')
    if config.get('logit_update_mode', 'full') not in ('full', 'target_only'):
        raise SystemExit(f'{dataset}: invalid logit_update_mode')
    for field, default in [('desired_margin', 50), ('up_margin_step', 20), ('down_margin_step', 25)]:
        value = config.get(field, default)
        if type(value) not in (int, float) or not math.isfinite(value):
            raise SystemExit(f'{dataset}: {field} must be a finite number')
    if config.get('use_global_feedback', True):
        value = config.get('target_prob')
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
            raise SystemExit(f'{dataset}: target_prob must be between 0 and 1')
    if type(config.get('seed')) is not int:
        raise SystemExit(f'{dataset}: seed must be an integer')
    if type(config.get('gpu')) is not int or config['gpu'] < -1:
        raise SystemExit(f'{dataset}: gpu must be an integer >= -1')
    for solver in solvers:
        destination = output_root / dataset / solver
        if destination.exists() or destination.is_symlink():
            raise SystemExit(f'Output already exists: {destination}. Use a different --output-root.')
        runs.append((dataset, solver, dict(config, logit_solver=solver)))

# Validate every run before writing the manifest or creating output directories.
with (temporary / 'runs.tsv').open('w', encoding='utf-8') as manifest:
    for dataset, solver, config in runs:
        runtime = temporary / f'{dataset}_{solver}.yaml'
        with runtime.open('w', encoding='utf-8') as handle:
            yaml.safe_dump(config, handle, allow_unicode=True, sort_keys=False)
        manifest.write(f'{dataset}\t{solver}\t{runtime}\n')
(temporary / 'output_root.txt').write_text(str(output_root))
print(f'Datasets: {len(datasets)} | solvers: {", ".join(solvers)} | experiments: {len(runs)}')
PY

OUTPUT_ROOT="$(cat -- "$TEMP_DIR/output_root.txt")"
cd -- "$PROJECT_ROOT/src"
FAILED=()
while IFS=$'\t' read -r dataset solver config; do
    run_dir="$OUTPUT_ROOT/$dataset/$solver"
    label="$dataset | logit_solver=$solver"
    command=(python -u main.py --params "$config" --output-dir "$run_dir")
    if ((DRY_RUN)); then
        echo "=== $label: temporary YAML ==="
        cat -- "$config"
        printf '%q ' "${command[@]}"
        printf '\n'
        continue
    fi
    mkdir -p -- "$(dirname -- "$run_dir")"
    mkdir -- "$run_dir"
    cp -- "$config" "$run_dir/config.yaml"
    run_started=$SECONDS
    if (
        echo "Experiment: $label"
        cat -- "$config"
        echo "Started: $(date '+%Y-%m-%d %H:%M:%S %z')"
        "${command[@]}"
    ) 2>&1 | tee "$run_dir/train.log"; then
        echo "Completed: $label" | tee -a "$run_dir/train.log"
    else
        status=$?
        echo "Failed: $label (exit $status); see $run_dir/train.log" | tee -a "$run_dir/train.log"
        FAILED+=("$label")
    fi
    echo "Experiment wall seconds (initialization, training, evaluation and saving): $((SECONDS - run_started))" | tee -a "$run_dir/train.log"
done < "$TEMP_DIR/runs.tsv"

if ((${#FAILED[@]})); then
    printf 'Failed experiment: %s\n' "${FAILED[@]}" >&2
    exit 1
fi
if ((DRY_RUN)); then
    echo "Dry run complete; no training started or outputs created."
else
    echo "All dataset/solver experiments completed. Outputs: $OUTPUT_ROOT"
fi
