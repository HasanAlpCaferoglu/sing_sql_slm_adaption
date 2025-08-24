import os
import subprocess

# def pick_gpu_by_lowest_memory():
#     try:
#         # Run nvidia-smi and capture memory used for each GPU
#         result = subprocess.check_output(
#             ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,nounits,noheader"],
#             encoding="utf-8"
#         )
#         memory_used = [int(x) for x in result.strip().split("\n")]
#         selected_gpu = memory_used.index(min(memory_used))

#         os.environ["CUDA_VISIBLE_DEVICES"] = str(selected_gpu)
#         print(f"[INFO] Selected GPU {selected_gpu} (Memory used: {memory_used[selected_gpu]} MiB)")

#     except Exception as e:
#         print(f"[WARN] Failed to select GPU automatically: {e}")
#         print("[INFO] Defaulting to GPU 0")
#         os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# pick_gpu_by_lowest_memory() 

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
import wandb


import unsloth
from unsloth import FastLanguageModel
from unsloth import is_bfloat16_supported

import torch
import datasets
from transformers import TrainingArguments, DataCollatorForSeq2Seq
from transformers import EarlyStoppingCallback
from accelerate import Accelerator
from trl import SFTTrainer, GRPOConfig, GRPOTrainer

from utils.datasets.prepare_train_datasets import prepare_train_datasets

import requests
from transformers.utils import hub

hub.constants.DEFAULT_TIMEOUT = 60  # Increase timeout to 60 seconds
requests.adapters.DEFAULT_RETRIES = 5  # Retry downloading up to 5 times


def train_with_unsloth_gpro(args):
    """
    GRPO training using unsloth
    """
    # Get Train Logger
    train_logger = logging.getLogger("train_logger")
    ## Prepare Dataset
    train_dataset, dev_dataset = prepare_train_datasets(args=args)
    train_logger.info(f"Train Dataset Size: {len(train_dataset)}")
    train_logger.info(f"Dev Dataset Size: {len(dev_dataset)}")

    ## Observe Dataset
    train_logger.info("++++++++++++++++++++ OBSERVE DATASET INDEX:0 +++++++++++++++++++")
    train_logger.info(f"{train_dataset[0]}")
    train_logger.info("++++++++++++++++++++ OBSERVE DATASET INDEX:-1 +++++++++++++++++++")
    train_logger.info(f"{train_dataset[-1]}")
    train_logger.info("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    train_logger.info("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

    ## Train configurations
    train_configs = args.config['train']
    use_cvd = train_configs["use_col_value_and_descriptions"]
    use_reasoning = train_configs["use_reasoning"]
    use_unsloth = train_configs["use_unsloth"]
    use_grpo = train_configs["use_grpo"]
    base_model_name = train_configs['base_model_name']
    lora_params = train_configs.get('lora_params', {})
    training_params = train_configs['training_params']
    max_seq_length = training_params['max_seq_length']

    # Logging basic information
    total_steps = (
        len(train_dataset) // (training_params['per_device_train_batch_size'] *
                                training_params['gradient_accumulation_steps']) *
        training_params['num_train_epochs']
    )
    print(f"Total training steps: {total_steps} (Assuming packing=False and drop_last=False)")
    train_logger.info(f"Total training steps: {total_steps} (Assuming packing=False and drop_last=False)")

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
    r = lora_params.get("lora_r", "")
    alpha = lora_params.get("lora_alpha", "")

    epoch = int(training_params['num_train_epochs'])
    bs = int(training_params['per_device_train_batch_size'])
    gas=int(training_params["gradient_accumulation_steps"])
    learningrate=float(training_params["learning_rate"])
    hf_repo_name = f"{base_model_id_without_user}_{t2s_dataset_name[0]}{data_mode[0]}_{db_id_str}_{task_str}_r{r}_a{alpha}_e{epoch}_bs{bs}_gas{gas}_lr{learningrate}_grpo" # Training with GRPO
    if use_cvd:
        hf_repo_name = f"{hf_repo_name}_cvd" # Using use_col_value_and_descriptions in the database schema
    hf_repo_name = f"{os.getenv('HF_USER')}/{hf_repo_name}"
    local_output_dir = f"./{hf_repo_name}"  # this is where checkpoints will be saved

    # Decide train from scratch or resume
    resume_from_checkpoint = None
    train_logger.info(f"..Checking for local checkpoints under: {local_output_dir}")
    checkpoint_paths = sorted(Path(local_output_dir).glob("checkpoint-*"))
    if checkpoint_paths:
        resume_from_checkpoint = str(checkpoint_paths[-1])  # Use the latest one
        train_logger.info(f"✅ Found local checkpoint: {resume_from_checkpoint}. Resuming training.")
    else:
        train_logger.info("Proceeding to start training from scratch.")


    ## Load model and tokenizer using Unsloth
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name= resume_from_checkpoint if resume_from_checkpoint else base_model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=bool(train_configs.get('load_in_4bit', False)),
        max_lora_rank = int(lora_params["lora_r"]),
        fast_inference = True, # Fast inference does not yet work with RoPE Scaling.
        gpu_memory_utilization = 0.5, # Reduce if out of memory
    )

    # Prepare LoRA
    if bool(train_configs.get("use_lora", False)):
        train_logger.info("-LoRA will be used")
        model = FastLanguageModel.get_peft_model(
            model,
            r=int(lora_params["lora_r"]),
            lora_alpha=int(lora_params["lora_alpha"]),
            target_modules=lora_params["lora_target_modules"],
            lora_dropout=float(lora_params["lora_dropout"]),
            bias=lora_params.get("bias", "none"),
            use_gradient_checkpointing="unsloth",
            random_state=3407,
        )

    train_logger.info(f"training_params: \n{training_params}")
    save_steps = int(training_params.get("save_steps", 0))
    save_strategy = "steps" if save_steps > 0 else "no"
    train_logger.info(f"save_steps: {save_steps}, save_strategy: {save_strategy}")

    training_args = GRPOConfig(
        output_dir = local_output_dir,   # local directory path where model checkpoints are saved
        # use_vllm = True, # use vLLM for fast inference # Fast inference does not yet work with RoPE Scaling, so I couldn't use vllm
        learning_rate = float(training_params["learning_rate"]),
        adam_beta1 = 0.9,
        adam_beta2 = 0.99,
        weight_decay = float(training_params['weight_decay']), # 0.1,
        warmup_ratio = float(training_params['warmup_ratio']), # 0.1
        lr_scheduler_type = training_params['lr_scheduler_type'],
        optim = training_params['optim'],
        logging_steps = 1,
        bf16 = is_bfloat16_supported(),
        fp16 = not is_bfloat16_supported(),
        per_device_train_batch_size = int(training_params['per_device_train_batch_size']),
        gradient_accumulation_steps = int(training_params['gradient_accumulation_steps']), 
        num_generations = int(training_params['num_generations']), # Decrease if out of memory
        max_prompt_length = int(training_params['max_prompt_lenght']),
        max_completion_length = int(training_params['max_seq_length']) - int(training_params['max_prompt_lenght']),
        num_train_epochs = int(training_params['num_train_epochs']), 
        save_steps = int(training_params['save_steps']),
        max_grad_norm = float(training_params['max_grad_norm']),
        report_to=["wandb"]  # remove 'wandb' or 'tensorboard' if not used
    )
    
    # Getting reward function names
    reward_functions_names = training_params["reward_functions_names"]
    reward_funcs = []
    for func_name in reward_functions_names:
        if func_name in globals():
            func = globals()[func_name]
            if callable(func): # Check if it's a function (not a variable)
                reward_funcs.append(func)

    trainer = GRPOTrainer(
        model = model, 
        processing_class = tokenizer,
        reward_funcs = reward_funcs,
        args = training_args,
        train_dataset = train_dataset,
    )

    try:
        # Allow numpy pickling for RNG state 
        # This will allow both the constructor and the actual ndarray type to be safely loaded.
        encode = _codecs.encode  
        torch.serialization.add_safe_globals([
            np._core.multiarray._reconstruct,
            np.ndarray,
            np.dtype,
            np.dtypes.UInt32DType,
            encode,
        ])
        train_logger.info("Allowing numpy pickling for RNG state")
    except Exception as e:
        train_logger.warning(f"Could not register safe globals: {e}")

    ## Start training
    train_logger.info("Start training...")
    try:
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    except wandb.errors.UsageError as e:
        train_logger.warnign("W&B failed to initialize. Continuing without W&B.")
        trainer.args.report_to = []
    
    train_logger.info("Training completed. Saving model to Hugging Face Hub...")

    # Save final model by pushing to the hub
    model.push_to_hub(hf_repo_name, token = os.getenv("HF_TOKEN")) # Online saving
    tokenizer.push_to_hub(hf_repo_name, token = os.getenv("HF_TOKEN")) # Online saving
    train_logger.info("Model and tokenizer pushed to Hugging Face Hub.")

    return


