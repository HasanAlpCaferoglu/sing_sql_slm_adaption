export NCCL_P2P_LEVEL=NVL
export PYTHONUNBUFFERED=TRUE

source .env
dataset_root_path=$DATASET_ROOT_PATH # like "../dataset"

config="./run/configs/config.yaml"


python3 -u ./src/prep_few_shots.py \
    --dataset_root_path "$dataset_root_path" \
    --config "$config" 