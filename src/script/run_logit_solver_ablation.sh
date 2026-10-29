#!/usr/bin/env bash
# 读取一份 YAML；仅切换 logits 求解器，原 YAML 不修改。
set -eo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
OUTPUT_ROOT="$PROJECT_ROOT/Outputs/logit_solver_ablation"
LOGIT_SOLVERS=(adam closed_form)
# 在这里选择要使用的 YAML；其余设置（包括是否自适应）全部沿用该文件。
CONFIG="$PROJECT_ROOT/src/configs/CIFAR10.yaml"
DRY_RUN=0

usage() {
    cat <<'EOF'
Usage: bash src/script/run_logit_solver_ablation.sh [options]

Set CONFIG near the top of this script to choose one dataset YAML.
Run adam and closed_form with identical settings except logit_solver.
GPU, seed, model, trigger settings, margin mode and logit update mode all
come from that YAML. This script does not enable adaptive_margin for you.
The YAML must enable attack: 1 and attack_type: Following_Nature.

Options:
  --output-root PATH  Default: <project>/Outputs/logit_solver_ablation.
  --dry-run           Validate and print both effective YAMLs and commands.
  -h, --help          Show this help.

Outputs: <output-root>/<dataset>/{adam,closed_form}/
         train.log, config.yaml, models/client_*.pth, triggers/global_trigger.*,
         and training plots. Existing run directories are never overwritten.
Final-round logs include mean L2/cosine and cumulative logit-solver time
(CPU wall time for the selected-sample solving loop, excluding metric logging).
train.log also records experiment wall time, including initialization,
training, evaluation and saving. Failed runs keep logs; the next solver runs.
EOF
}

while (($#)); do
    case "$1" in
        --output-root)
            if (($# < 2)) || [[ -z "$2" ]]; then
                echo "Missing value for --output-root" >&2
                exit 2
            fi
            OUTPUT_ROOT="$2"
            shift 2
            ;;
        --dry-run) DRY_RUN=1; shift ;;
        -h|--help) usage; exit 0 ;;
        --*) echo "Unknown option: $1" >&2; exit 2 ;;
        *) echo "Select the YAML by editing CONFIG at the top of this script." >&2; exit 2 ;;
    esac
done

source /home/a100/anaconda3/etc/profile.d/conda.sh
conda activate WPAD
set -u
export MPLBACKEND=Agg
TEMP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/nba-logit-solver-ablation.XXXXXX")"
trap 'rm -rf -- "$TEMP_DIR"' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

dataset="$(python - "$CONFIG" "$OUTPUT_ROOT" "$TEMP_DIR" "${LOGIT_SOLVERS[@]}" <<'PY'
import math
from pathlib import Path
import re
import sys
import yaml

source, output_root, temporary = (Path(value).resolve() for value in sys.argv[1:4])
with source.open(encoding='utf-8') as handle:
    config = yaml.safe_load(handle)
if not isinstance(config, dict):
    raise SystemExit(f'{source}: YAML must contain a parameter mapping')
dataset = config.get('dataset')
if not isinstance(dataset, str) or not re.fullmatch(r'[A-Za-z0-9_]+', dataset):
    raise SystemExit(f'{source}: invalid dataset name')
if config.get('attack') != 1 or config.get('attack_type') != 'Following_Nature':
    raise SystemExit(f'{source}: set attack: 1 and attack_type: Following_Nature')
for field in ('epochs', 'clients', 'local_ep', 'distill_ep'):
    if type(config.get(field)) is not int or config[field] < 1:
        raise SystemExit(f'{source}: {field} must be a positive integer')
malicious = config.get('malicious')
if (not isinstance(malicious, list) or not malicious
        or any(type(cid) is not int or not 0 <= cid < config['clients'] for cid in malicious)
        or len(malicious) != len(set(malicious))):
    raise SystemExit(f'{source}: malicious must contain distinct valid client IDs')
for field in ('adaptive_margin', 'use_global_feedback'):
    if type(config.get(field, True)) is not bool:
        raise SystemExit(f'{source}: {field} must be a YAML boolean')
if config.get('logit_update_mode', 'full') not in ('full', 'target_only'):
    raise SystemExit(f'{source}: logit_update_mode must be full or target_only')
for field, default in [('desired_margin', 50), ('up_margin_step', 20), ('down_margin_step', 25)]:
    value = config.get(field, default)
    if type(value) not in (int, float) or not math.isfinite(value):
        raise SystemExit(f'{source}: {field} must be a finite number')
if config.get('use_global_feedback', True):
    probability = config.get('target_prob')
    if type(probability) not in (int, float) or not math.isfinite(probability) or not 0 <= probability <= 1:
        raise SystemExit(f'{source}: target_prob must be between 0 and 1')
if type(config.get('seed')) is not int:
    raise SystemExit(f'{source}: seed must be an integer')
solvers = sys.argv[4:]
if len(solvers) != 2 or set(solvers) != {'adam', 'closed_form'}:
    raise SystemExit('LOGIT_SOLVERS must contain adam and closed_form exactly once')
for solver in solvers:
    run = output_root / dataset / solver
    if run.exists() or run.is_symlink():
        raise SystemExit(f'Output already exists: {run}. Use a different --output-root.')
    with (temporary / f'{solver}.yaml').open('w', encoding='utf-8') as handle:
        yaml.safe_dump(dict(config, logit_solver=solver), handle, allow_unicode=True, sort_keys=False)
(temporary / 'output_root.txt').write_text(str(output_root))
print(dataset)
PY
)"
OUTPUT_ROOT="$(cat -- "$TEMP_DIR/output_root.txt")"
echo "Dataset: $dataset | logit solvers: ${LOGIT_SOLVERS[*]} | experiments: ${#LOGIT_SOLVERS[@]}"
cd -- "$PROJECT_ROOT/src"
FAILED=()
for solver in "${LOGIT_SOLVERS[@]}"; do
    config="$TEMP_DIR/$solver.yaml"
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
done
if ((${#FAILED[@]})); then
    printf 'Failed experiment: %s\n' "${FAILED[@]}" >&2
    exit 1
fi
if ((DRY_RUN)); then
    echo "Dry run complete; no training started or outputs created."
else
    echo "Logit solver ablation completed. Outputs: $OUTPUT_ROOT"
fi
