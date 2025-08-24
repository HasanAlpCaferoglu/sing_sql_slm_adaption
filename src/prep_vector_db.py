import os
import argparse
import yaml
import json
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from typing import List, Dict, Any

from runner.prep_vector_db_runner import PrepVDBRunner



def parse_arguments() -> argparse.Namespace:
    
    parser = argparse.ArgumentParser(description="Run the pipeline with the specified configuration.")

    parser.add_argument('--dataset_root_path', type=str, required=True, help="Path to the benchmark file.")
    parser.add_argument('--config', type=str, required=True, help="Path to the configuration file.")
    args = parser.parse_args()

    args.run_start_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    with open(args.config, 'r') as file:
        args.config=yaml.safe_load(file)

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
        
        args.dbs_root_dir = args.dataset_mode_root_path / f"{data_mode}_databases"

    else:
        raise ValueError(f"Your dataset is set to {dataset}. Currently this code is not support all datasets.")
    

def prep_vector_db():
    print("=== VECTOR DB CREATION ===")
    prep_vector_db_runner = PrepVDBRunner(args)
    prep_vector_db_runner.construct_vdbs()
    
    return


if __name__ == "__main__":
    """
    Creating Vector Database for synthetic text-to-sql data
    """
    load_dotenv()
    args = parse_arguments()
    update_args(args)
    prep_vector_db()