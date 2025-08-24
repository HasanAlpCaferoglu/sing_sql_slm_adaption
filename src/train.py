import os
import subprocess
import argparse
import random
import yaml
import json
import logging
import numpy as np
import _codecs
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from typing import List, Dict, Any

import torch
import datasets


import wandb

import requests
from transformers.utils import hub

hub.constants.DEFAULT_TIMEOUT = 60  # Increase timeout to 60 seconds
requests.adapters.DEFAULT_RETRIES = 5  # Retry downloading up to 5 times


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
    args.seed = args.config['seed']
    
    data_mode = args.data_mode
    dataset = args.dataset
    dataset_root_path = Path(args.dataset_root_path)

    if dataset == 'bird':
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


def create_wandb_run_id(args):
    wandb_run_id = ''

    ## Train configurations
    train_configs = args.config['train']
    prompt_temp_name = train_configs.get("prompt_temp_name", "")
    ptn = "ST" if prompt_temp_name == "slm_t2s" else "T"
    use_few_shot = bool(train_configs.get("use_few_shot", False))
    few_shot_cnt = int(train_configs.get("few_shot_cnt", 0)) if use_few_shot else 0
    use_reasoning_in_few_shots = bool(train_configs.get("use_reasoning_in_few_shots", False)) if use_few_shot else False
    urifs = "t" if use_reasoning_in_few_shots else "f"
    use_lora = bool(train_configs.get("use_lora", False))
    base_model_name = train_configs['base_model_name']
    lora_params = train_configs.get('lora_params', {})
    training_params = train_configs['training_params']

    ## Set Huggingface repo name
    data_mode = args.config['data_mode']
    t2s_dataset_name = args.config['dataset']
    base_model_id_without_user = base_model_name.split("/")[-1]
    db_ids_initials = []
    for db_name in args.db_ids:
        db_inits = ""
        db_name_words = db_name.split("_")
        for db_name_word in db_name_words:
            db_name_word_initial = db_name_word[0]
            db_inits += db_name_word_initial
        db_ids_initials.append(db_inits)

    db_id_str = "-".join(db_ids_initials)
    task_str = "-".join(train_configs['dataset_tasks'])
    r = lora_params.get("lora_r", "") if use_lora else 0
    alpha = lora_params.get("lora_alpha", "") if use_lora else 0

    epoch = int(training_params['num_train_epochs'])
    bs = int(training_params['per_device_train_batch_size'])
    gas=int(training_params["gradient_accumulation_steps"])
    learningrate=float(training_params["learning_rate"])

    use_reasoning = train_configs['use_reasoning']
    use_grpo = train_configs['use_grpo']
    use_cvd = train_configs["use_col_value_and_descriptions"]
    use_cvd_init = 't' if use_cvd else "f"

    # hf_repo_name = f"{base_model_id_without_user}_{t2s_dataset_name[0]}{data_mode[0]}_{db_id_str}_{task_str}"
    if use_grpo:
        hf_repo_name = f"{base_model_id_without_user}_{t2s_dataset_name[0]}{data_mode[0]}_{db_id_str}_{task_str}_r{r}_a{alpha}_e{epoch}_bs{bs}_gas{gas}_lr{learningrate}_fs{few_shot_cnt}{urifs}_cvd{use_cvd_init}_pt{ptn}_grpo" # Training with GRPO
    elif use_reasoning:
        hf_repo_name = f"{base_model_id_without_user}_{t2s_dataset_name[0]}{data_mode[0]}_{db_id_str}_{task_str}_r{r}_a{alpha}_e{epoch}_bs{bs}_gas{gas}_lr{learningrate}_fs{few_shot_cnt}{urifs}_cvd{use_cvd_init}_pt{ptn}_sftreason" # Reasoning with SFT
    else:
        hf_repo_name = f"{base_model_id_without_user}_{t2s_dataset_name[0]}{data_mode[0]}_{db_id_str}_{task_str}_r{r}_a{alpha}_e{epoch}_bs{bs}_gas{gas}_lr{learningrate}_fs{few_shot_cnt}{urifs}_cvd{use_cvd_init}_pt{ptn}"

    wandb_run_id = hf_repo_name

    return wandb_run_id

def train(args):
    ## Setup Logger
    train_logger = logging.getLogger("train_logger")
    train_logger.setLevel(logging.INFO)
    log_dir = Path(f"logs/train_{args.run_start_time}")
    logger_path = log_dir / "train_logs.log"
    logger_path.parent.mkdir(parents=True, exist_ok=True)
    train_logger_handler = logging.FileHandler(logger_path)
    train_logger_handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    train_logger.addHandler(train_logger_handler)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    train_logger.addHandler(console_handler)
    train_logger.propagate = False  # Avoid double logging

    wandb_run_id = create_wandb_run_id(args)
    # Try to login only if a key is present
    wandb_key = os.getenv("WANDB_API_KEY")
    train_logger.info("Trying to login wandb")
    if wandb_key:
        run = wandb.init(
            entity = 'schemaless-team',
            project='schemaless_v2',
            name=wandb_run_id,
            id=wandb_run_id,
            resume='allow',
            settings=wandb.Settings(init_timeout=600)
        )
        train_logger.info("Logged in Wandb and a run is initialized.")
    else:
        train_logger.warning("[WARN] WANDB_API_KEY not set. W&B will be disabled.")
        os.environ["WANDB_MODE"] = "disabled"

    ## Log full args into a separate file in the same directory
    args_log_path = log_dir / "train_args.json"
    with open(args_log_path, 'w') as f:
        # We convert Namespace to dict for pretty-printing
        yaml.dump(vars(args), f, default_flow_style=False)

    ## Train configurations
    train_configs = args.config['train']
    use_unsloth = bool(train_configs['use_unsloth'])
    use_reasoning = bool(train_configs['use_reasoning'])
    use_grpo = bool(train_configs['use_grpo'])
    train_logger.info(f"use_unsloth: {use_unsloth} | use_grpo: {use_grpo}")

    if use_unsloth and use_grpo:
        from training.train_unsloth_grpo import train_with_unsloth_gpro

        train_logger.info(f"GRPO training using unsloth.")
        train_with_unsloth_gpro(args)
    elif use_unsloth:
        from training.train_unsloth_sft import train_with_unsloth_sft

        train_logger.info(f"SFT training using unsloth.")
        train_with_unsloth_sft(args)
    else:
        from training.train_sft import train_sft

        train_logger.info(f"SFT training.")
        train_sft(args)


if __name__ == "__main__":
    load_dotenv()
    args = parse_arguments()
    update_args(args)
    train(args)