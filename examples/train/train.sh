#!/bin/bash
{
    set -euxo pipefail
    cd "$(dirname "$0")/../.."
    export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
    export TOKENIZERS_PARALLELISM=false
    export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
    export WANDB_ENTITY="duwei98-east-china-normal-university"
    export WANDB_PROJECT="DVAGen"

    deepspeed --include localhost:0,1,2,3 \
              --master_port 12610 train.py \
              --config_path examples/train/qwen.yaml

    exit
}