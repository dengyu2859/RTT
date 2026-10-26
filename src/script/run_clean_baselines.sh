#!/usr/bin/env bash
# Run one clean federated-training experiment per dataset, using WPAD.
set -eo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
OUTPUT_ROOT="$PROJECT_ROOT/Outputs"
DRY_RUN=0
GPU_ARGS=()
DATASETS=()

usage() {
    cat <<'EOF'
Usage: bash src/script/run_clean_baselines.sh [options] [DATASET ...]

With no dataset arguments, run every YAML in src/configs once, sequentially.
Each run forces attack=0 and malicious=[], preserving other YAML settings.

Options:
  --gpu N             Override the GPU for all runs (-1 for CPU).
  --output-root PATH  Output directory (default: <project>/Outputs).
  --dry-run           Validate configs and print commands without training.
  -h, --help          Show this help.

Outputs: <output-root>/<dataset>/train.log, config.yaml, models/client_*.pth,
         and the existing training plots. Existing dataset directories are
         never overwritten. A failed run retains its log and the batch
         continues with the next dataset, finally returning a nonzero status.

Examples:
  bash src/script/run_clean_baselines.sh
  bash src/script/run_clean_baselines.sh --gpu 0 CIFAR10 FashionMNIST
  bash src/script/run_clean_baselines.sh --dry-run
EOF
}

while (($#)); do
    case "$1" in
        --gpu|--output-root)
            if (($# < 2)); then
                echo "Missing value for $1" >&2
                exit 2
            fi
            if [[ "$1" == --gpu ]]; then
                if [[ ! "$2" =~ ^(-1|[0-9]+)$ ]]; then
                    echo "GPU must be -1 or a nonnegative integer." >&2
                    exit 2
                fi
                GPU_ARGS=(--gpu "$2")
            else
                OUTPUT_ROOT="$2"
            fi
            shift 2
            ;;
        --dry-run) DRY_RUN=1; shift ;;
        -h|--help) usage; exit 0 ;;
        --*) echo "Unknown option: $1" >&2; exit 2 ;;
        *) DATASETS+=("$1"); shift ;;
    esac
done

if ((${#DATASETS[@]} == 0)); then
    for config in "$PROJECT_ROOT"/src/configs/*.yaml; do
        DATASETS+=("$(basename -- "$config" .yaml)")
    done
fi

# Activate before enabling nounset: Conda activation scripts may use unset vars.
source /home/a100/anaconda3/etc/profile.d/conda.sh
conda activate WPAD
set -u
export MPLBACKEND=Agg
# Resolve a relative output root before changing into src.
OUTPUT_ROOT="$(python -c 'import os, sys; print(os.path.abspath(sys.argv[1]))' "$OUTPUT_ROOT")"

# Validate the entire batch before starting the first long-running experiment.
declare -A SEEN=()
for dataset in "${DATASETS[@]}"; do
    if [[ ! "$dataset" =~ ^[A-Za-z0-9_]+$ ]]; then
        echo "Invalid dataset name: $dataset" >&2
        exit 2
    fi
    if [[ -n "${SEEN[$dataset]:-}" ]]; then
        echo "Duplicate dataset: $dataset" >&2
        exit 2
    fi
    SEEN[$dataset]=1
    config="$PROJECT_ROOT/src/configs/$dataset.yaml"
    python - "$config" "$dataset" <<'PY'
import sys
import yaml

with open(sys.argv[1], encoding='utf-8') as handle:
    config = yaml.safe_load(handle)
if config.get('dataset') != sys.argv[2]:
    raise SystemExit(f'Dataset name does not match configuration: {sys.argv[1]}')
for field in ('epochs', 'clients', 'local_ep', 'distill_ep'):
    if not isinstance(config.get(field), int) or config[field] < 1:
        raise SystemExit(f'{sys.argv[1]}: {field} must be a positive integer')
PY
    if [[ -e "$OUTPUT_ROOT/$dataset" || -L "$OUTPUT_ROOT/$dataset" ]]; then
        echo "Output already exists: $OUTPUT_ROOT/$dataset. Use a different --output-root." >&2
        exit 2
    fi
done

cd -- "$PROJECT_ROOT/src"
FAILED=()
for dataset in "${DATASETS[@]}"; do
    run_dir="$OUTPUT_ROOT/$dataset"
    command=(python -u main.py --params "$PROJECT_ROOT/src/configs/$dataset.yaml"
             --clean-baseline --output-dir "$run_dir" "${GPU_ARGS[@]}")
    if ((DRY_RUN)); then
        printf '%q ' "${command[@]}"
        printf '\n'
        continue
    fi

    mkdir -p -- "$OUTPUT_ROOT"
    # An exclusive mkdir also protects against another batch starting this run.
    mkdir -- "$run_dir"
    if (
        echo "Dataset: $dataset | clean baseline: attack=0, malicious=[]"
        echo "Started: $(date '+%Y-%m-%d %H:%M:%S %z')"
        "${command[@]}"
    ) 2>&1 | tee "$run_dir/train.log"; then
        echo "Completed: $dataset" | tee -a "$run_dir/train.log"
    else
        status=$?
        echo "Failed: $dataset (exit $status); see $run_dir/train.log" | tee -a "$run_dir/train.log"
        FAILED+=("$dataset")
    fi
done

if ((${#FAILED[@]})); then
    echo "Failed datasets: ${FAILED[*]}" >&2
    exit 1
fi
if ((DRY_RUN)); then
    echo "Dry run complete; no training started or outputs created."
else
    echo "All clean baselines completed. Outputs: $OUTPUT_ROOT"
fi
