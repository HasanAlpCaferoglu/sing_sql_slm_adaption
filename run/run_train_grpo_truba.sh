#!/bin/bash
#SBATCH -p barbun-cuda        # Kuyruk adi: Uzerinde GPU olan kuyruk olmasina dikkat edin.
#SBATCH -C barbun-cuda
#SBATCH -A hcaferoglu       # Kullanici adi
#SBATCH -J grpo_train        # Gonderilen isin ismi
#SBATCH -o outputs/train_%j.out    # Ciktinin yazilacagi dosya adi
#SBATCH --gres=gpu:1        # Her bir sunucuda kac GPU istiyorsunuz? Kumeleri kontrol edin.
#SBATCH -N 1                # Gorev kac node'da calisacak?
#SBATCH -n 1                # Ayni gorevden kac adet calistirilacak?
#SBATCH --cpus-per-task 20  # Her bir gorev kac cekirdek kullanacak? Kumeleri kontrol edin. (barbun-cuda => 20 | akya-cuda => 10)
#SBATCH --time=3-00:00:00     # Sure siniri koyun. ==> time=7-0:0:0 ==> 7 days ||| 1:00:00 => 1 hour

export NCCL_P2P_LEVEL=NVL
export PYTHONUNBUFFERED=TRUE

##### BINDINGS FOR APPTAINER #####
export APPTAINER_BINDPATH="/etc/pki:/etc/pki,/arf/scratch/hcaferoglu" 

source .env
# Dataset path
dataset_root_path=$DATASET_ROOT_PATH # like "../dataset"
export WANDB_API_KEY=$WANDB_KEY    

config="./run/configs/config_grpo.yaml"

##### RUN #####
apptainer exec --nv /arf/scratch/hcaferoglu/container-user/schemaless.sif python3 -u ./src/train.py \
    --dataset_root_path "$dataset_root_path" \
    --config "$config" 

