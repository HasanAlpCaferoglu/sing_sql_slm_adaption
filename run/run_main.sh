#!/bin/bash
#
# ========================================
# ===== RESOURCES =====
# ========================================
#
#SBATCH --job-name=t2s2t                                   #Setting a job name
#SBATCH --nodes=1                                          #Asking for only one node
#SBATCH --ntasks=1                       
#SBATCH --cpus-per-task=8                       
#SBATCH --partition=long                                     # Running on ai queue(maks 7 days)       
#SBATCH --account=users                                       # Running on ai partitions(group of nodes)
#SBATCH --time=7-0:0:0                                     # Reserving for 7 days time limit.
#SBATCH --gres=gpu:tesla_t4:1                              # Asking a tesla_v100 GPU
#SBATCH --output=outputs/%j.out                            # Setting a output file name.
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
data_mode=$DATA_MODE # Options: 'dev', 'train' 
data_path=$DATA_PATH # UPDATE THIS WITH THE PATH TO THE TARGET DATASET

config="./run/configs/CHESS_IR_CG_UT.yaml"



python3 -u ./src/main.py --data_mode ${data_mode} --data_path ${data_path} --config "$config" \

