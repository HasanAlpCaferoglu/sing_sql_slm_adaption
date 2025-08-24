#!/bin/bash
#
# ========================================
# ===== RESOURCES =====
# ========================================
#
#SBATCH --job-name=prep_schemaless             #Setting a job name
#SBATCH --partition=CPU_himem              
#SBATCH --nodelist=tm3
#SBATCH --cpus-per-task=16                   # Number of CPU cores per task
#SBATCH --mem=120G                           # Memory limit
#SBATCH --output=outputs/prep_%j.out    # Standard output log file

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

config="./run/configs/config_preprocess.yaml"



python3 -u ./src/preprocess.py \
    --dataset_root_path "$dataset_root_path" \
    --config "$config" 

