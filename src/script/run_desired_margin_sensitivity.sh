#!/usr/bin/env bash
# 读取一份 YAML；运行一组自适应和四组固定值实验，原 YAML 不修改。
set -eo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
OUTPUT_ROOT="$PROJECT_ROOT/Outputs/desired_margin_sensitivity"
DESIRED_MARGINS=(50 100 150 200)
# 在这里选择要使用的 YAML 文件，只需修改这一行。
CONFIG="$PROJECT_ROOT/src/configs/CIFAR10.yaml"
DRY_RUN=0

usage() {
    cat <<'EOF'
Usage: bash src/script/run_desired_margin_sensitivity.sh [options]

Set CONFIG near the top of this script to choose the YAML file.
Read that YAML and run one adaptive baseline plus fixed desired_margin =
50, 100, 150, 200. The adaptive run sets adaptive_margin: true and keeps
the YAML initial margin, steps, and target probability. Fixed runs set
adaptive_margin: false; margin steps are inactive.
All remaining parameters, including GPU, malicious clients, and seed, come
unchanged from that YAML. Source YAML files are never modified.
The YAML must enable attack: 1 and attack_type: Following_Nature.
This fixed/adaptive comparison requires use_global_feedback: true (default).
Run main.py with your YAML for the no-feedback increase-only ablation.

Options:
  --output-root PATH  Default: <project>/Outputs/desired_margin_sensitivity.
  --dry-run           Validate and print all five YAMLs and commands.
  -h, --help          Show this help.

Outputs: <output-root>/<dataset>/adaptive/
         <output-root>/<dataset>/desired_margin_<value>/
         train.log, config.yaml, models/client_*.pth, triggers/global_trigger.*,
         and training plots. Existing run directories are never overwritten.
         Failed runs retain their logs; later parameter values still run.

Examples (run from the project root):
  bash src/script/run_desired_margin_sensitivity.sh --dry-run
  bash src/script/run_desired_margin_sensitivity.sh
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
        *) echo "Select the YAML file by editing CONFIG at the top of this script." >&2; exit 2 ;;
    esac
done
if [[ -z "$CONFIG" ]]; then
    echo "Set CONFIG at the top of this script to a YAML file path." >&2
    exit 2
fi

# Conda activation scripts may access unset variables.
source /home/a100/anaconda3/etc/profile.d/conda.sh
conda activate WPAD
set -u
export MPLBACKEND=Agg
TEMP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/nba-fixed-margin-sensitivity.XXXXXX")"
trap 'rm -rf -- "$TEMP_DIR"' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

# Validate the selected YAML and all five output paths before training.
# Paths are resolved before changing the working directory to src.
dataset="$(python - "$CONFIG" "$OUTPUT_ROOT" "$TEMP_DIR" "${DESIRED_MARGINS[@]}" <<'PY'
import math
from pathlib import Path
import re
import sys
import yaml

source, output_root, temporary = map(lambda value: Path(value).resolve(), sys.argv[1:4])
with source.open(encoding='utf-8') as handle:
    config = yaml.safe_load(handle)
if not isinstance(config, dict):
    raise SystemExit(f'{source}: YAML must contain a parameter mapping')
dataset = config.get('dataset')
if not isinstance(dataset, str) or not re.fullmatch(r'[A-Za-z0-9_]+', dataset):
    raise SystemExit(f'{source}: invalid dataset name')
if config.get('attack') != 1 or config.get('attack_type') != 'Following_Nature':
    raise SystemExit(f'{source}: set attack: 1 and attack_type: Following_Nature in your YAML')
feedback = config.get('use_global_feedback', True)
if feedback is not True:
    raise SystemExit(f'{source}: this fixed/adaptive script requires use_global_feedback: true; '
                     'run main.py directly for the no-feedback increase-only ablation')
for field in ('epochs', 'clients', 'local_ep', 'distill_ep'):
    if type(config.get(field)) is not int or config[field] < 1:
        raise SystemExit(f'{source}: {field} must be a positive integer')
malicious = config.get('malicious')
if not isinstance(malicious, list) or not malicious:
    raise SystemExit(f'{source}: malicious must be a nonempty list of client IDs')
if any(type(client) is not int or not 0 <= client < config['clients'] for client in malicious):
    raise SystemExit(f'{source}: invalid malicious client ID')
if type(config.get('seed')) is not int:
    raise SystemExit(f'{source}: seed must be an integer')
for field, default in [('desired_margin', 50), ('up_margin_step', 20), ('down_margin_step', 25)]:
    value = config.get(field, default)
    if type(value) not in (int, float) or not math.isfinite(value):
        raise SystemExit(f'{source}: {field} must be a finite number')
target_prob = config.get('target_prob')
if type(target_prob) not in (int, float) or not math.isfinite(target_prob) or not 0 <= target_prob <= 1:
    raise SystemExit(f'{source}: target_prob must be a number between 0 and 1')

margins = []
for text in sys.argv[4:]:
    try:
        value = float(text)
    except ValueError:
        raise SystemExit('DESIRED_MARGINS must contain finite nonnegative numbers')
    if not math.isfinite(value) or value < 0:
        raise SystemExit('DESIRED_MARGINS must contain finite nonnegative numbers')
    margins.append(int(value) if value.is_integer() else value)
if len(margins) != 4 or len(set(margins)) != 4:
    raise SystemExit('DESIRED_MARGINS must contain exactly four distinct values')
runs = [('adaptive', dict(config, adaptive_margin=True))]
runs.extend((f'desired_margin_{margin}', dict(config, adaptive_margin=False, desired_margin=margin))
            for margin in margins)
for name, effective in runs:
    run = output_root / dataset / name
    if run.exists() or run.is_symlink():
        raise SystemExit(f'Output already exists: {run}. Use a different --output-root.')
    with (temporary / f'{name}.yaml').open('w', encoding='utf-8') as handle:
        yaml.safe_dump(effective, handle, allow_unicode=True, sort_keys=False)
(temporary / 'runs.txt').write_text(''.join(f'{name}\n' for name, _ in runs))
(temporary / 'margins.txt').write_text(''.join(f'{margin}\n' for margin in margins))
(temporary / 'output_root.txt').write_text(str(output_root))
print(dataset)
PY
)"
mapfile -t DESIRED_MARGINS < "$TEMP_DIR/margins.txt"
mapfile -t RUN_NAMES < "$TEMP_DIR/runs.txt"
OUTPUT_ROOT="$(cat -- "$TEMP_DIR/output_root.txt")"
echo "Dataset: $dataset | adaptive + fixed desired_margin values: ${DESIRED_MARGINS[*]} | experiments: ${#RUN_NAMES[@]}"
cd -- "$PROJECT_ROOT/src"
FAILED=()
for name in "${RUN_NAMES[@]}"; do
    config="$TEMP_DIR/$name.yaml"
    run_dir="$OUTPUT_ROOT/$dataset/$name"
    label="$dataset | $name"
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
done

if ((${#FAILED[@]})); then
    printf 'Failed experiment: %s\n' "${FAILED[@]}" >&2
    exit 1
fi
if ((DRY_RUN)); then
    echo "Dry run complete; no training started or outputs created."
else
    echo "All desired_margin sensitivity experiments completed. Outputs: $OUTPUT_ROOT"
fi
