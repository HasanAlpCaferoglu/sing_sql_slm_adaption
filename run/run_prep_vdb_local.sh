export NCCL_P2P_LEVEL=NVL
export PYTHONUNBUFFERED=TRUE

source .env
dataset_root_path=$DATASET_ROOT_PATH # like "../dataset"

config="./run/configs/config_prep_vdb.yaml"


python3 -u ./src/prep_vector_db.py \
    --dataset_root_path "$dataset_root_path" \
    --config "$config" 