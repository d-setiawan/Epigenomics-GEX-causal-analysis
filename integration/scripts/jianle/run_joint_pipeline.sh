#!/usr/bin/env bash
set -euo pipefail

# Full joint Jianle pipeline:
# preprocess -> train -> validation
#
# Usage:
#   bash integration/scripts/jianle/run_joint_pipeline.sh <RUN_ID> [options]

usage() {
  cat <<'EOF'
Usage:
  bash integration/scripts/jianle/run_joint_pipeline.sh <RUN_ID> [options]

Options:
  --skip-preprocess
  --skip-train
  --skip-validate
  --preprocess-arg ARG   (repeatable)
  --train-arg ARG        (repeatable)
  --validate-arg ARG     (repeatable)
EOF
}

if [[ $# -ge 1 && ( "$1" == "-h" || "$1" == "--help" ) ]]; then
  usage
  exit 0
fi

if [[ $# -lt 1 ]]; then
  usage >&2
  exit 1
fi

RUN_ID="$1"
shift

SKIP_PREPROCESS=0
SKIP_TRAIN=0
SKIP_VALIDATE=0
PREPROCESS_ARGS=()
TRAIN_ARGS=()
VALIDATE_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-preprocess)
      SKIP_PREPROCESS=1
      shift
      ;;
    --skip-train)
      SKIP_TRAIN=1
      shift
      ;;
    --skip-validate)
      SKIP_VALIDATE=1
      shift
      ;;
    --preprocess-arg)
      PREPROCESS_ARGS+=("$2")
      shift 2
      ;;
    --train-arg)
      TRAIN_ARGS+=("$2")
      shift 2
      ;;
    --validate-arg)
      VALIDATE_ARGS+=("$2")
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

if [[ $SKIP_PREPROCESS -eq 0 ]]; then
  bash integration/scripts/jianle/run_preprocess_joint.sh "$RUN_ID" "${PREPROCESS_ARGS[@]}"
fi

if [[ $SKIP_TRAIN -eq 0 ]]; then
  bash integration/scripts/jianle/run_train_joint.sh "$RUN_ID" "${TRAIN_ARGS[@]}"
fi

if [[ $SKIP_VALIDATE -eq 0 ]]; then
  bash integration/scripts/jianle/run_validate_joint.sh "$RUN_ID" "${VALIDATE_ARGS[@]}"
fi

echo "Jianle joint pipeline complete: integration/outputs/jianle/joint/${RUN_ID}"
