#!/bin/bash
{
    set -euxo pipefail
    cd "$(dirname "$0")/../.."
    export CUDA_VISIBLE_DEVICES="0"
    export TRANSFORMERS_NO_ADVISORY_WARNINGS=1

    python eval.py --config_path examples/eval/qwen.yaml

    exit
}