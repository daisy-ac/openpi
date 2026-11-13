
#!/bin/bash
set -euo pipefail

cd /root/autodl-tmp/openpi-workspace/openpi
source /root/venvs/openpi/bin/activate

export HF_LEROBOT_HOME=/root/autodl-tmp/openpi-workspace/datasets
export HF_HOME=/root/autodl-tmp/hf-download-cache
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export WANDB_MODE=offline
export WANDB_DIR=/root/autodl-tmp/openpi-workspace/logs/wandb
unset WANDB_API_KEY

mkdir -p /root/autodl-tmp/openpi-workspace/logs
mkdir -p /root/autodl-tmp/openpi-workspace/logs/wandb

python -u scripts/train.py pi05_libero_lora \
    --exp-name pi05_libero_dual_lora_bs4_30k_3 \
    --batch-size 4 \
    --num-train-steps 30000 \
    --log-interval 100 \
    --save-interval 20000 \
    --keep-period 20000 \
    --wandb-enabled \
    2>&1 | tee -a /root/autodl-tmp/openpi-workspace/logs/pi05_bs4_30k.log
