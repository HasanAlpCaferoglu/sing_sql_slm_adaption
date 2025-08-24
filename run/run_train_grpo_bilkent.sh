#!/bin/bash
#
# ========================================
# ===== RESOURCES =====
# ========================================
#
#SBATCH --job-name=grpo_train                  #Setting a job name
#SBATCH --partition=GPU_nvrtx4090              # GPU_nvgtx1080(gpu1, gpu2) | GPU_nvrtx4090 (gpu3, gpu4,)
#SBATCH --nodelist=gpu3                        # gpu1, gpu2, gpu3, gpu4
#SBATCH --cpus-per-task=16                   # Number of CPU cores per task     
#SBATCH --mem=60G                           # Memory limit
#SBATCH --output=outputs/train_%j.out    # Standard output log file

# ========================================
# ===== LOADING LIBRARY AND MODULES =====
# ========================================

# Create the log directory if it doesn't exist
mkdir -p joblogs

# Set stack size to unlimited
echo "Setting stack size to unlimited..."
ulimit -s unlimited
ulimit -l unlimited
ulimit -a


export NCCL_P2P_LEVEL=NVL
export PYTHONUNBUFFERED=TRUE

source .env
dataset_root_path=$DATASET_ROOT_PATH # like "../dataset"
export WANDB_API_KEY=$WANDB_KEY    

config="./run/configs/config_grpo.yaml"

python3 -u ./src/train.py \
    --dataset_root_path "$dataset_root_path" \
    --config "$config" 

