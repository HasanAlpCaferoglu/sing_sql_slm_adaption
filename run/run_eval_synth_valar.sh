#!/bin/bash
#
# ========================================
# ===== RESOURCES =====
# ========================================
#
#SBATCH --job-name=eval_schemaless         #Setting a job name
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=ai                     # avg for lovelace_l40s | ai for ampere_a40
#SBATCH --account=ai
#SBATCH --gres=gpu:ampere_a40:1          # lovelace_l40s OR ampere_a40 (use with partition=ai and time:1-0:0:0)
#SBATCH --output=outputs/eval_%j.out    # Standard output log file
#SBATCH --mail-type=ALL
#SBATCH --mail-user=alp.caferoglu@bilkent.edu.tr
#SBATCH --mem=32G

# ========================================
# ===== LOADING LIBRARY AND MODULES =====
# ========================================
echo "-------"
# Run nvidia-smi and save output to a variable
gpu_info=$(nvidia-smi)

# Print the GPU info
echo "==== NVIDIA-SMI Output ===="
echo "$gpu_info"

source activate t2s_unsloth_sft

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
echo "====================================================================================="
echo "====================================================================================="

export NCCL_P2P_LEVEL=NVL
export PYTHONUNBUFFERED=TRUE
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export TORCHDYNAMO_DISABLE=1 # for phi models

source .env
dataset_root_path=$DATASET_ROOT_PATH # like "../dataset"

config="./run/configs/config.yaml"

python3 -u ./src/eval_synth.py \
    --dataset_root_path "$dataset_root_path" \
    --config "$config" 

