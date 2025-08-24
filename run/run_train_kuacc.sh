#!/bin/bash
#
# ========================================
# ===== RESOURCES =====
# ========================================
#
#SBATCH --job-name=train_schemaless         #Setting a job name
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=long
#SBATCH --account=users
#SBATCH --time=7-0:0:0
#SBATCH --gres=gpu:tesla_v100:8          # tesla_v100 OR rtx_a6000 (use with partition=mid and time:1-0:0:0)
#SBATCH --output=outputs/train_%j.out    # Standard output log file
#SBATCH --mail-type=ALL
#SBATCH --mail-user=alp.caferoglu@bilkent.edu.tr
#SBATCH --mem=128G

# ========================================
# ===== LOADING LIBRARY AND MODULES =====
# ========================================

module load anaconda/2024.02
# module load gcc/9.3.0
module load glibc/2.27
module load gcc/11.2.0
module load cuda/11.8.0
module load cudnn/8.9.5/cuda-12.x 
module load nccl/2.9.6-1/cuda-11.3

source activate schemaless

echo "which python:"
which python
echo "-------"

echo "which -a python:"
which -a python
echo "-------"

echo "gcc version: "
which gcc
gcc --version
echo "-------"

echo "glibc version: "
ldd --version
echo "-------"

# Create the log directory if it doesn't exist
mkdir -p joblogs

# Set stack size to unlimited
echo "Setting stack size to unlimited..."
ulimit -s unlimited
ulimit -l unlimited
ulimit -a

# Print Cuda
echo "CUDA Version: $(nvcc --version | grep release | sed 's/.*release //' | sed 's/,.*//')"

export NCCL_P2P_LEVEL=NVL
export PYTHONUNBUFFERED=TRUE
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

source .env
dataset_root_path=$DATASET_ROOT_PATH # like "../dataset"
export WANDB_API_KEY=$WANDB_KEY    

config="./run/configs/config_kuacc.yaml"

accelerate launch ./src/train.py \
    --dataset_root_path "$dataset_root_path" \
    --config "$config" 

