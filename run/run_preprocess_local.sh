export NCCL_P2P_LEVEL=NVL
export PYTHONUNBUFFERED=TRUE

source .env
dataset_root_path=$DATASET_ROOT_PATH # like "../dataset"

config="./run/configs/config_preprocess.yaml"



# python3 -u ./src/preprocess.py \
#     --dataset_root_path "$dataset_root_path" \
#     --config "$config" 



max_retries=20
retry_count=0

while [ $retry_count -lt $max_retries ]; do
    echo "Run attempt: $((retry_count + 1))"
    
    python3 -u ./src/preprocess.py \
        --dataset_root_path "$dataset_root_path" \
        --config "$config"
    
    exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        echo "Process completed successfully."
        break
    elif [ $exit_code -eq 137 ]; then  # 128 + 9 = 137 (SIGKILL)
        echo "Process was killed (likely OOM). Retrying..."
        retry_count=$((retry_count + 1))
        sleep 5  # Optional cooldown before retry
    else
        echo "Process failed with exit code $exit_code. Not retrying."
        break
    fi
done

if [ $retry_count -eq $max_retries ]; then
    echo "Maximum retry limit reached. Exiting with failure."
    exit 1
fi