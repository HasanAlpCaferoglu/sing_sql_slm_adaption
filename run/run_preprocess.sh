#!/bin/bash
#
# ========================================
# ===== RESOURCES =====
# ========================================
#
#SBATCH --job-name=P3S_preprocess                                   #Setting a job name
#SBATCH --nodes=1                                          #Asking for only one node
#SBATCH --ntasks=1                       
#SBATCH --cpus-per-task=8                       
#SBATCH --partition=long                                     # Running on ai queue(maks 7 days)       
#SBATCH --account=users                                       # Running on ai partitions(group of nodes)
#SBATCH --time=7-0:0:0                                     # Reserving for 7 days time limit.
#SBATCH --gres=gpu:tesla_v100:1                              # Asking a tesla_v100 GPU
#SBATCH --output=outputs/prep_%j.out                            # Setting a output file name.
#SBATCH --mail-type=ALL                                    # All types all emails (BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=alp.caferoglu@bilkent.edu.tr           # Where to send emails                     
#SBATCH --mem=48G                       

# ========================================
# ===== LOADING LIBRARY AND MODULES =====
# ========================================

module load anaconda/2024.02
source activate t2s2t
module load cuda/12.3
module load gcc/9.3.0 
module load cudnn/8.9.5/cuda-12.x 
module load nccl/2.9.6-1/cuda-11.3

# Set stack size to unlimited
echo "Setting stack size to unlimited..."
ulimit -s unlimited
ulimit -l unlimited
ulimit -a

export NCCL_P2P_LEVEL=NVL
export PYTHONUNBUFFERED=TRUE

source .env
dataset_root_path=$DATASET_ROOT_PATH # like "../dataset"

config="./run/configs/config.yaml"



python3 -u ./src/preprocess.py \
    --dataset_root_path "$dataset_root_path" \
    --config "$config" 

