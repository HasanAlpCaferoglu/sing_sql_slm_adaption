import os
import re
import logging
from pathlib import Path
import time
from accelerate import Accelerator
from accelerate.utils import set_seed
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_scheduler,
    DataCollatorForSeq2Seq,
)
from torch.optim import AdamW
from torch.utils.data import ConcatDataset, random_split
from peft import get_peft_model, LoraConfig, TaskType
from tqdm.auto import tqdm
import torch
import wandb

from utils.datasets.prepare_train_datasets import prepare_train_datasets, get_dataset_instances

def train_sft(args):
    train_logger = logging.getLogger("train_logger")
    set_seed(args.config['seed'])

    train_configs = args.config['train']
    use_cvd = train_configs["use_col_value_and_descriptions"]
    use_reasoning = train_configs["use_reasoning"]
    base_model_name = train_configs['base_model_name']
    lora_params = train_configs.get('lora_params', {})
    training_params = train_configs['training_params']
    max_seq_length = training_params['max_seq_length']

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
    hf_repo_name = f"{base_model_id_without_user}_{t2s_dataset_name[0]}{data_mode[0]}_{db_id_str}_{task_str}_e{epoch}_bs{bs}_gas{gas}_lr{learningrate}"
    if use_reasoning:
        hf_repo_name = f"{hf_repo_name}_sftreason" # Reasoning with SFT

    if use_cvd:
        hf_repo_name = f"{hf_repo_name}_cvd" # Using use_col_value_and_descriptions in the database schema

    hf_repo_name = f"{os.getenv('HF_USER')}/{hf_repo_name}"
    local_output_dir = f"./{hf_repo_name}"  # this is where checkpoints will be saved

    # Decide train from scratch or resume
    def extract_checkpoint_number(path):
        try:
            return int(path.name.split("-")[-1])
        except (IndexError, ValueError):
            return -1  # Put malformed ones (if any) at the start
    
    resume_from_checkpoint = None
    train_logger.info(f"..Checking for local checkpoints under: {local_output_dir}")
    # checkpoint_paths = sorted(Path(local_output_dir).glob("checkpoint-*")) # lexical sorting (problem: ckpt-500 looks like ckpt-1000 )
    train_logger.info(f'Path(local_output_dir).glob("checkpoint-*"): {Path(local_output_dir).glob("checkpoint-*"),}')
    checkpoint_paths = sorted(
        Path(local_output_dir).glob("checkpoint-*"),
        key=extract_checkpoint_number
    )
    if checkpoint_paths:
        train_logger.info(f"Local checkpoints: {[str(ckpt_path_str) for ckpt_path_str in checkpoint_paths] }")
        resume_from_checkpoint = str(checkpoint_paths[-1])  # Use the latest one
        train_logger.info(f"✅ Found local checkpoint: {resume_from_checkpoint}. Resuming training.")
    else:
        train_logger.info("Proceeding to start training from scratch.")

    
    # Accelerator Instance
    accelerator = Accelerator(gradient_accumulation_steps=int(training_params['gradient_accumulation_steps']))
    device = accelerator.device

    # Print the current device info
    accelerator.print(f"Accelerator device: {device}")
    accelerator.print(f"Is distributed: {accelerator.distributed_type}")
    accelerator.print(f"Available devices: {torch.cuda.device_count()} GPUs")

    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            accelerator.print(f"CUDA Device {i}: {torch.cuda.get_device_name(i)}")

    ## -- Loading Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path = base_model_name,
        trust_remote_code=True
    ) 
    # Note that padding and truncation is handled in dataset
    # tokenizer.padding_side = "left"
    # tokenizer.truncation_side = "left"

    ## -- Loading Model and PEFT if set
    train_logger.info("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path = resume_from_checkpoint if resume_from_checkpoint else base_model_name, 
        trust_remote_code=True
    )

    if bool(train_configs.get("use_lora", False)):
        train_logger.info("Setting PEFT configs...")
        peft_config = LoraConfig(
            r=lora_params["lora_r"],
            lora_alpha=lora_params["lora_alpha"],
            target_modules=lora_params["lora_target_modules"],
            lora_dropout=lora_params["lora_dropout"],
            bias=lora_params.get("bias", "none"),
        )
        model = get_peft_model(model, peft_config)

    model.to(device) # Loading model to device
    model.gradient_checkpointing_enable() # Reduce memory requirement. It stores only a few key activations (checkpoints). During the backward pass, it recomputes the rest of the activations as needed.
    train_logger.info(f"Model ({model.name_or_path}) is loaded.")

    # enable gradient checkpointing to save GPU memory, but this action would slowdown the training speed 20-30%
    # model.gradient_checkpointing_enable()
    
    # -- Dataset
    dataset_instances = get_dataset_instances(
        args=args, 
        tokenizer=tokenizer, 
        max_tokens=int(training_params['max_seq_length'])
    )
    combined_dataset = ConcatDataset(dataset_instances)

    data_collator = DataCollatorForSeq2Seq(tokenizer, return_tensors="pt", padding=True)
    train_dataloader = DataLoader(
        combined_dataset,
        shuffle=True,
        collate_fn=data_collator,
        batch_size=int(training_params['per_device_train_batch_size'])
    )

    ## Optimizer
    optimizer = AdamW(model.parameters(), lr=float(training_params['learning_rate']))

    ## LR Scheduler
    num_training_steps = int(training_params['num_train_epochs']) * len(train_dataloader)
    lr_scheduler = get_scheduler(
        name=training_params['lr_scheduler_type'],
        optimizer=optimizer,
        num_warmup_steps=int(training_params['warmup_ratio'] * num_training_steps),
        num_training_steps=num_training_steps
    )

    # Prepare PyTorch objects (model, optimizer, scheduler, etc.) for distributed training
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    # Training Loop with save_steps logic
    save_steps = int(training_params.get("save_steps", 500))
    save_total_limit = int(training_params.get("save_total_limit", 2))

    train_logger.info("Starting Accelerate training loop...")
    progress_bar = tqdm(range(num_training_steps), disable=not accelerator.is_local_main_process)
    
    global_completed_steps = 0
    accumulation_loss = 0
    model.train()

    ## Resume training:  
    if resume_from_checkpoint:
        try:
            ## Load Optimizer and LR Scheduler states
            optimizer.load_state_dict(torch.load(Path(resume_from_checkpoint) / "optimizer.pt"))
            lr_scheduler.load_state_dict(torch.load(Path(resume_from_checkpoint) / "scheduler.pt"))

            # Load global_completed_steps from checkpoint folder name  Format: checkpoint-<global_completed_steps>
            global_completed_steps = int(Path(resume_from_checkpoint).name.split("-")[-1])

            # Compute resume_epoch and resume_batch_idx
            steps_per_epoch = len(train_dataloader) // int(training_params['gradient_accumulation_steps'])
            resume_epoch = global_completed_steps // steps_per_epoch
            resume_batch_idx = (global_completed_steps % steps_per_epoch) * int(training_params['gradient_accumulation_steps'])

            train_logger.info(f"Resuming from step: {global_completed_steps}")
            train_logger.info(f"Resume epoch: {resume_epoch}, batch index: {resume_batch_idx}")
        except Exception as e:
            train_logger.error(f"Failed to parse global_step from checkpoint path: {e}")

    st = time.time()
    for epoch in range(epoch):

        if resume_from_checkpoint and resume_epoch > epoch:
            accelerator.print(f"skip {epoch}-th epoch")
            continue

        for batch_idx, batch in enumerate(train_dataloader):

            if resume_from_checkpoint and resume_batch_idx > batch_idx:
                accelerator.print(f"skip {batch_idx}-th batch")
                continue

            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs.loss
                accumulation_loss += loss.detach().float()
                # when deepspeed is enabled, `accelerator.backward(loss)` is doing optimizer.step(), optimizer.zero_grad(), and grad accumulation automatically. 
                # see `if self.is_gradient_accumulation_boundary():` line in path-to-env/site-packages/deepspeed/runtime/engine.py
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # 'accelerator.sync_gradients' checks if the accelerator has performed an optimization step on the `total_batch_size` examples
            if accelerator.sync_gradients:
                global_completed_steps += 1
                progress_bar.update(1)
                accelerator.print("GPU 0, step {}, loss {}".format(global_completed_steps, accumulation_loss / accelerator.gradient_accumulation_steps))
                accelerator.print("GPU 0, step {}, lr state dict:".format(global_completed_steps), lr_scheduler.state_dict())
                accelerator.print(time.time()-st)
                st = time.time()

                if accelerator.is_main_process:
                    wandb.log({
                        "train/loss": loss.item(),
                        "train/learning_rate": lr_scheduler.get_last_lr()[0],
                        "train/step": global_completed_steps,
                        "train/epoch": epoch + batch_idx / len(train_dataloader),
                    }, step=global_completed_steps)



                if global_completed_steps % save_steps == 0 and accelerator.is_main_process:
                    # save checkpoint
                    ckpt_path = local_output_dir / f"checkpoint-{global_completed_steps}"
                    ckpt_path.mkdir(parents=True, exist_ok=True)
                    accelerator.unwrap_model(model).save_pretrained(ckpt_path, save_function=accelerator.save)
                    tokenizer.save_pretrained(ckpt_path)
                    train_logger.info(f"Checkpoint saved at step {global_completed_steps} to {ckpt_path}")

                    # save scheduler
                    scheduler_path = ckpt_path / "scheduler.pt"
                    torch.save(lr_scheduler.state_dict(), scheduler_path)

                    # save optimizer
                    optimizer_path = ckpt_path / "optimizer.pt"
                    torch.save(optimizer.state_dict(), optimizer_path)

                    # Remove older checkpoints
                    checkpoints = sorted(local_output_dir.glob("checkpoint-*"), key=os.path.getmtime)
                    if len(checkpoints) > save_total_limit:
                        for ckpt_to_remove in checkpoints[:-save_total_limit]:
                            train_logger.info(f"Removing old checkpoint: {ckpt_to_remove}")
                            os.system(f"rm -rf {ckpt_to_remove}")


    accelerator.print("Training finished!")
    return
