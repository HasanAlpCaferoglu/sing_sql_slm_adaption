import os
import subprocess
import torch

def pick_gpu_by_lowest_memory():

    print(f"Visible devices: {torch.cuda.device_count()}")  

    try:
        # Run nvidia-smi and capture memory used for each GPU
        result = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,nounits,noheader"],
            encoding="utf-8"
        )
        memory_used = [int(x) for x in result.strip().split("\n")]
        selected_gpu = memory_used.index(min(memory_used))

        os.environ["CUDA_VISIBLE_DEVICES"] = str(selected_gpu)
        print(f"[INFO] Selected GPU {selected_gpu} (Memory used: {memory_used[selected_gpu]} MiB)")

    except Exception as e:
        print(f"[WARN] Failed to select GPU automatically: {e}")
        print("[INFO] Defaulting to GPU 0")
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    print(f"Visible devices: {torch.cuda.device_count()}")  
    print(f"Using device: {torch.cuda.current_device()} - {torch.cuda.get_device_name(0)}")

pick_gpu_by_lowest_memory() 

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


import unsloth
from unsloth import FastLanguageModel
from unsloth import is_bfloat16_supported

import datasets
from transformers import TrainingArguments, DataCollatorForSeq2Seq
from transformers import EarlyStoppingCallback
from trl import SFTTrainer

import wandb

from utils.datasets.prepare_train_datasets import prepare_train_datasets

import requests
from transformers.utils import hub

hub.constants.DEFAULT_TIMEOUT = 60  # Increase timeout to 60 seconds
requests.adapters.DEFAULT_RETRIES = 5  # Retry downloading up to 5 times


def train_with_unsloth_sft(args):
    """
    Supervised fine-tuning using unsloth
    """
    # Use only one GPU with unsloth
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"

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

    ## Train configurations
    train_configs = args.config['train']
    prompt_temp_name = train_configs.get("prompt_temp_name", "")
    ptn = "ST" if prompt_temp_name == "slm_t2s" else "T"
    use_few_shot = bool(train_configs["use_few_shot"])
    few_shot_cnt = int(train_configs["few_shot_cnt"]) if use_few_shot else 0
    use_reasoning_in_few_shots = bool(train_configs.get("use_reasoning_in_few_shots", False)) if use_few_shot else False
    urifs = "t" if use_reasoning_in_few_shots else "f"
    use_cvd = train_configs["use_col_value_and_descriptions"]
    use_cvd_init = 't' if use_cvd else "f"
    use_reasoning = train_configs["use_reasoning"]
    use_grpo = train_configs["use_grpo"]
    base_model_name = train_configs['base_model_name']
    use_lora = bool(train_configs.get("use_lora", False))
    lora_params = train_configs.get('lora_params', {})
    training_params = train_configs['training_params']
    max_seq_length = training_params['max_seq_length']
    train_logger.info(f"train_configs: \n {train_configs}\n ")

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
    r = lora_params.get("lora_r", "") if use_lora else 0
    alpha = lora_params.get("lora_alpha", "") if use_lora else 0

    epoch = int(training_params['num_train_epochs'])
    bs = int(training_params['per_device_train_batch_size'])
    gas=int(training_params["gradient_accumulation_steps"])
    learningrate=float(training_params["learning_rate"])
    hf_repo_name = f"{base_model_id_without_user}_{t2s_dataset_name[0]}{data_mode[0]}_{db_id_str}_{task_str}_r{r}_a{alpha}_e{epoch}_bs{bs}_gas{gas}_lr{learningrate}_fs{few_shot_cnt}{urifs}_cvd{use_cvd_init}_pt{ptn}"
    if use_reasoning:
        hf_repo_name = f"{hf_repo_name}_sftreason" # Reasoning with SFT
    elif use_grpo:
        hf_repo_name = f"{hf_repo_name}_grpo" # Reasoning with SFT


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
    if "phi" in base_model_name or (resume_from_checkpoint and "phi" in resume_from_checkpoint):
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name= resume_from_checkpoint if resume_from_checkpoint else base_model_name,
            max_seq_length=max_seq_length,
            dtype=None,  # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
            load_in_4bit=bool(train_configs.get('load_in_4bit', False)),
            rope_scaling = None # Added for the phi models
            # device_map="auto" # remove auto mapping to prevent multi-gpu process
        )
    else:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name= resume_from_checkpoint if resume_from_checkpoint else base_model_name,
            max_seq_length=max_seq_length,
            dtype=None,  # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
            load_in_4bit=bool(train_configs.get('load_in_4bit', False)),
            # device_map="auto" # remove auto mapping to prevent multi-gpu process
        )
    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "left"

    # Prepare LoRA
    if bool(train_configs.get("use_lora", False)):
        train_logger.info("-LoRA will be used.")
        model = FastLanguageModel.get_peft_model(
            model,
            r=int(lora_params["lora_r"]),
            lora_alpha=int(lora_params["lora_alpha"]),
            lora_dropout=float(lora_params["lora_dropout"]),
            target_modules=lora_params["lora_target_modules"],
            bias=lora_params.get("bias", "none"),
            use_gradient_checkpointing=True,
            random_state=3407,
            use_rslora=False,
            loftq_config=None
        )

    train_logger.info(f"training_params: \n{training_params}")
    save_steps = int(training_params.get("save_steps", 0))
    save_strategy = "steps" if save_steps > 0 else "no"
    train_logger.info(f"save_steps: {save_steps}, save_strategy: {save_strategy}")

    train_logger.info(f"int(training_params['per_device_train_batch_size']): { int(training_params['per_device_train_batch_size']) }")
    # Define SFTTrainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        dataset_text_field='text',  # commented_out
        max_seq_length=max_seq_length, # commented_out
        data_collator = DataCollatorForSeq2Seq(tokenizer = tokenizer),
        # dataset_num_proc=train_configs['train_workers_num'], # commented_out
        packing=training_params.get('packing', False), # commented_out
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
        args=TrainingArguments(
            output_dir=local_output_dir,   # local directory path where model checkpoints are saved
            push_to_hub=True, 
            hub_token=os.getenv("HF_TOKEN"),
            hub_model_id=hf_repo_name,
            per_device_train_batch_size=1, # int(training_params['per_device_train_batch_size'])
            per_device_eval_batch_size=int(training_params["per_device_eval_batch_size"]),
            gradient_accumulation_steps=int(training_params["gradient_accumulation_steps"]),
            eval_accumulation_steps = int(training_params["eval_accumulation_steps"]),
            warmup_ratio=float(training_params["warmup_ratio"]),
            num_train_epochs=int(training_params["num_train_epochs"]),
            learning_rate=float(training_params["learning_rate"]),
            fp16 = not is_bfloat16_supported(),
            bf16 = is_bfloat16_supported(),
            logging_steps=int(training_params["logging_steps"]),
            save_strategy = save_strategy,
            save_steps=save_steps,
            save_total_limit=int(training_params["save_total_limit"]),
            load_best_model_at_end = True,
            lr_scheduler_type=training_params["lr_scheduler_type"],
            weight_decay=float(training_params["weight_decay"]),
            optim=training_params["optim"],
            eval_strategy="steps",
            run_name=hf_repo_name.split("/")[-1],
            report_to=["wandb"]  # remove 'wandb' or 'tensorboard' if not used,
        )
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
    train_logger.info("Starting training...")
    try:
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    except wandb.errors.UsageError as e:
        train_logger.warning("W&B failed to initialize. Continuing without W&B.")
        trainer.args.report_to = []
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    train_logger.info("Training completed. Saving model to Hugging Face Hub...")

    # Save final model by pushing to the hub
    model.push_to_hub(hf_repo_name, token = os.getenv("HF_TOKEN")) # Online saving
    tokenizer.push_to_hub(hf_repo_name, token = os.getenv("HF_TOKEN")) # Online saving
    train_logger.info("Model and tokenizer pushed to Hugging Face Hub.")

    return



