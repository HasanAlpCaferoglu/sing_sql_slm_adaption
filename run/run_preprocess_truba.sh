#!/bin/bash
#SBATCH -p barbun            # Kuyruk adi: GPU kullanilacaksa uzerinde GPU olan kuyruk olmasina dikkat edin.
#SBATCH -A hcaferoglu       # Kullanici adi
#SBATCH -J schemaless_preprocess        # Gonderilen isin ismi
#SBATCH -o outputs/prep_%j.out    # Ciktinin yazilacagi dosya adi
#SBATCH -N 1                # Gorev kac node'da calisacak?
#SBATCH -n 1                # Ayni gorevden kac adet calistirilacak?
#SBATCH --cpus-per-task 40  # Her bir gorev kac cekirdek kullanacak? Kumeleri kontrol edin.
#SBATCH --time=1:00:00      # Sure siniri koyun. ==> time=7-0:0:0 ==> 7 days


export NCCL_P2P_LEVEL=NVL
export PYTHONUNBUFFERED=TRUE

##### BINDINGS FOR APPTAINER #####
export SINGULARITY_BINDPATH="/etc/pki:/etc/pki,/arf/scratch/hcaferoglu" 

##### VARIABLES #####
source .env
dataset_root_path=$DATASET_ROOT_PATH # like "../dataset"
config="./run/configs/config_preprocess.yaml"

##### RUN #####
apptainer exec /arf/scratch/hcaferoglu/container-user/miniconda3-user.sif python3 -u ./src/preprocess.py \
    --dataset_root_path "$dataset_root_path" \
    --config "$config" 

