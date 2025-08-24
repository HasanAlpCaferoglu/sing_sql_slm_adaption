import os
import argparse
import yaml
import json
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from typing import List, Dict, Any

from runner.eval_runner import EvalRunner



def parse_arguments() -> argparse.Namespace:
    
    parser = argparse.ArgumentParser(description="Run the pipeline with the specified configuration.")

    parser.add_argument('--dataset_root_path', type=str, required=True, help="Path to the benchmark file.")
    parser.add_argument('--config', type=str, required=True, help="Path to the configuration file.")
    args = parser.parse_args()

    args.run_start_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    with open(args.config, 'r') as file:
        args.config=yaml.safe_load(file)

    db_sub_schema_configs_path =  args.config['preprocess']['sub_schema_generator']['db_ss_configs']
    with open(db_sub_schema_configs_path, 'r') as file:
        args.config['preprocess']['sub_schema_generator']['db_ss_configs'] = yaml.safe_load(file)

    db_completion_configs_path = args.config['preprocess']['db_completion_dataset']['db_completion_configs']
    with open(db_completion_configs_path, 'r') as file:
        args.config['preprocess']['db_completion_dataset']['db_completion_configs'] = yaml.safe_load(file)
    
    return args

def update_args(args):
    args.data_mode = args.config['data_mode']
    args.dataset = args.config['dataset']
    args.db_ids = args.config['db_ids']
    
    data_mode = args.data_mode
    dataset = args.dataset
    dataset_root_path = Path(args.dataset_root_path)
    dataset_root_path = dataset_root_path.resolve()
    print(f"dataset_root_path: {dataset_root_path}")
    print(f"[DEBUG] Current working directory: {os.getcwd()}")

    if dataset == 'bird' or dataset == "bird-sql":
        args.dataset_path = dataset_root_path / "bird-sql"
        args.dataset_mode_root_path = args.dataset_path / data_mode # dataset_mode_root_path = DB_ROOT_PATH
        # Convert to string and set environment variable
        os.environ["DB_ROOT_PATH"] = str(args.dataset_mode_root_path)
        os.environ["DATASET_MODE_ROOT_PATH"] = str(args.dataset_mode_root_path)
        
        args.data_json_path = args.dataset_mode_root_path / f"{data_mode}.json"
        args.tables_json_path = args.dataset_mode_root_path / f"{data_mode}_tables.json"
        args.dbs_root_dir = args.dataset_mode_root_path / f"{data_mode}_databases"
        args.column_meaning_path = args.dataset_mode_root_path / f"column_meaning.json"
        args.processed_column_meaning_path = args.dataset_mode_root_path / f"processed_column_meaning.json"

    else:
        raise ValueError(f"Your dataset is set to {dataset}. Currently this code is not support all datasets.")


def evaluate(args):
    """
    Model evaluation
    """

    print("=== EVALUATION ===")
    print("--- CONFIG ---")
    print(args.config)
    eval_runner = EvalRunner(args)
    eval_runner.evaluate()

if __name__ == "__main__":
    load_dotenv()
    args = parse_arguments()
    update_args(args)
    evaluate(args)
