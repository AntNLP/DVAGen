#!/bin/bash
{
    set -euxo pipefail
    cd "$(dirname "$0")/../.."
    export CUDA_VISIBLE_DEVICES="7"
    export TRANSFORMERS_NO_ADVISORY_WARNINGS=1

    python chat.py --config_path examples/chat/qwen.yaml

    exit
}