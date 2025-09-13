import os
import traceback
import subprocess
import copy
import torch


def pick_gpu_by_lowest_memory():
    try:
        print(torch.__version__)
        print(torch.cuda.is_available())
        print(torch.cuda.get_device_name(0))
        # Run nvidia-smi and capture memory used for each GPU
        result = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,nounits,noheader"],
            encoding="utf-8"
        )
        memory_used = [int(x) for x in result.strip().split("\n")]
        selected_gpu = memory_used.index(min(memory_used))
        selected_cuda = f"cuda:{selected_gpu}"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(selected_gpu)
        print(f"[INFO] Selected GPU {selected_gpu} (Memory used: {memory_used[selected_gpu]} MiB)")
        
    except Exception as e:
        print(f"[WARN] Failed to select GPU automatically: {e}")
        print("[INFO] Defaulting to GPU 0")
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        selected_cuda = f"cuda:0"

    return selected_cuda

device = pick_gpu_by_lowest_memory() 

import json
import logging
import math
import time
import random
import traceback
import threading
from pathlib import Path
from typing import List, Tuple, Dict, Any, Set, Iterable, Optional, Literal
from dataclasses import dataclass, field
from itertools import chain, combinations
from pydantic import BaseModel

import unsloth
from unsloth import FastLanguageModel

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from transformers import BitsAndBytesConfig

from utils.db_utils.execution import execute_sql, get_execution_status, compare_sqls
from utils.llm_utils.prompt_utils import load_template, load_template_examples
from utils.llm_utils.model import call_llm, parse_llm_output

from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
from utils.db_utils.schema_generator import DatabaseSchemaGenerator
from utils.db_utils.db_info_utils import get_db_all_tables, get_db_schema
from utils.db_utils.db_info import DatabaseGeneralInfo
from utils.db_utils.schema import DatabaseSchema
from utils.db_utils.execution import calculate_f1_score_for_sql
from utils.db_utils.sql_parser import get_sql_columns_dict
from utils.llm_utils.model import extract_xml_answer, extract_xml_reasoning, extract_response_part, extract_json, extract_sql_part
from utils.eval_utils.eval_utils import calculate_schema_metrics_for_single_schema
from utils.llm_utils.LLMService import LLMService
from utils.vdb_utils.T2SSyntheticVDBService import T2SSyntheticVDBService

file_lock = threading.Lock()

class EvalSynthRunner:
    def __init__(self, args: Any):
        self.args = args
        self.all_db_ids: List[str] = self._set_db_ids()
        self.db_ids: List[str] = self.args.config.get("db_ids", [])
        self.eval_dataset = self._load_dataset()

        # Set logger
        logger = logging.getLogger('eval')
        logger.setLevel(logging.INFO)
        logger_path = Path(f"logs/eval_{self.args.run_start_time}/eval_logs.log")
        logger_path.parent.mkdir(parents=True, exist_ok=True)
        logger_handler = logging.FileHandler(logger_path)
        logger_handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
        logger.addHandler(logger_handler)
        self.eval_logger = logger
        
        # eval configs
        eval_configs = self.args.config['evaluation']
        use_proprietary_model = eval_configs["use_proprietary_model"]
        if use_proprietary_model:
            self.model = None
            self.tokenizer = None
            self.model_name = eval_configs["proprietary_model_name"]
        else:
            self.model, self.tokenizer, self.model_name = self.load_language_model_and_tokenizer()

        # Initialize vector database service
        # self.vdb_service = self._construct_vdb_service() # unnecessary as we found the few-shots in prep stage

        self.results_dir = Path(f"results/")
        self.eval_results_dir = self._make_eval_results_dir()

        # Get already processed items
        self.processed_t2s_items: List[Dict[str, Any]] = self._load_previously_processed_items()
        self.processed_t2s_items_keys: Set[Tuple[str, int]] = self._get_previously_processed_items_keys()
        
        

    def _set_db_ids(self) -> List[str]:
        """
        Extract the database ids that is under consideration and sets as the attribute of the class
        """
        DBS_ROOT_DIR = self.args.dbs_root_dir
        if not DBS_ROOT_DIR.exists() or not DBS_ROOT_DIR.is_dir():
            raise ValueError(f"Invalid directory: {DBS_ROOT_DIR}")
    
        all_db_ids_list = [dir_.name for dir_ in DBS_ROOT_DIR.iterdir() if dir_.is_dir()]
        all_db_ids_list = sorted(all_db_ids_list)
        
        return all_db_ids_list
    
    def _load_dataset(self) -> List[Dict[str, Any]]:
        """
        Loads the dataset from the specified path.

        Args:
            data_path (str): Path to the data file.

        Returns:
            List[Dict[str, Any]]: The loaded dataset.
        """
        dataset = []

        dbs_root_dir = Path(self.args.dbs_root_dir) # databases 
        for db_id in self.db_ids:
            database_dir: Path = dbs_root_dir / db_id
            prep_dir_name = self.args.config.get("prep_dir_name", "")
            database_prep_dir: Path = database_dir / prep_dir_name
            if not database_prep_dir.exists():
                raise FileNotFoundError(f"Directory {database_prep_dir} not found.")
            
            sub_schemas_dir: Path = database_prep_dir / "sub_schemas"
            column_level_sub_schemas_dir: Path = sub_schemas_dir / "column_level"

            if not column_level_sub_schemas_dir.exists():
                raise FileNotFoundError(f"Directory {column_level_sub_schemas_dir} not found.")

            eval_configs = self.args.config['evaluation']
            eval_synth_mode = eval_configs['eval_synth_mode']
            
            db_eval_data_jsonl_path: Path = column_level_sub_schemas_dir / f"sub_schema_examples_{eval_synth_mode}_with_few_shots.jsonl"
            if not db_eval_data_jsonl_path.exists():
                raise FileNotFoundError(f"Directory {db_eval_data_jsonl_path} not found.")
            
            with open(db_eval_data_jsonl_path, 'r') as file:
                for line in file:
                    try:
                        item = json.loads(line)
                        dataset.append(item)
                    except:
                        continue

        return dataset
    
    def _make_eval_results_dir(self) -> Path:
        """
        Constructs eval results dir
        """


        if "/" in self.model_name:
            pure_model_name = self.model_name.split("/")[1]
        else:
            pure_model_name = self.model_name
        
        # Create Results Directory
        db_ids_str = ""
        for index, db_id in enumerate(self.db_ids):
            db_ids_str += f"{db_id}"
            if index != len(self.db_ids) - 1:
                db_ids_str += "_"

        eval_configs = self.args.config['evaluation']
        eval_synth_mode = eval_configs['eval_synth_mode']
        prompt_temp_name = str(eval_configs['prompt_temp_name'])
        if prompt_temp_name == "slm_t2s":
            ptn = "ST"
        elif prompt_temp_name == "csc_t2s":
            ptn = "CT"
        else:
            ptn = "T"
        ebm = "T" if bool(eval_configs["eval_base_model"]) else "F"
        upm = "T" if bool(eval_configs["use_proprietary_model"]) else "F"
        cvd = "T" if bool(eval_configs["use_col_value_and_descriptions"]) else "F"
        us = "T" if bool(eval_configs["use_schema"]) else "F"
        sc = ''.join(word[0] for word in eval_configs["schema_content"].split('_')).upper()
        ufs = "T" if bool(eval_configs["use_few_shot"]) else "F"
        fsc = int(eval_configs["few_shot_cnt"])
        urifs = "T" if bool(eval_configs["use_reasoning_in_few_shots"]) else "F"
        ur = "T" if bool(eval_configs["use_reasoning"]) else "F"
        en = int(eval_configs["eval_no"])
        temperatures = eval_configs["temperature"]
        temp = temperatures[0]
        can_count = len(temperatures)
    
        eval_sub_dir_name = f"{eval_synth_mode}_cc{can_count}_temp{temp}_ptn{ptn}_ebm{ebm}_upm{upm}_cvd{cvd}_us{us}_sc{sc}_ufs{ufs}_fsc{fsc}_urifs{urifs}_ur{ur}_en{en}"

        # eval_results_dir = Path(f"./results/{db_ids_str}/{pure_model_name}/{self.args.run_start_time}")
        eval_results_dir = Path(f"./results_synth/{db_ids_str}/{pure_model_name}/{eval_sub_dir_name}")
        eval_results_dir.mkdir(parents=True, exist_ok=True)
        print(f"Eval Results Dir: {eval_results_dir}")
        return eval_results_dir
    
    def _load_previously_processed_items(self) -> List[Dict[str, Any]]:
        """"
        Loads the previously processed data in evaluation set, iff some of the portions of the data is processed already.
        """
        t2s_items: List[Dict[str, Any]] = []
        t2s_items_file_path: Path = self.eval_results_dir / "t2s_items.json"
        if t2s_items_file_path.exists():
            with open(t2s_items_file_path, 'r') as file:
                t2s_items = json.load(file)

        print(f"Previously processed {len(t2s_items)} items loaded from {t2s_items_file_path} ")
        return t2s_items
    
    def _get_previously_processed_items_keys(self) -> Set[Tuple[str, int]]:
        """
        Build a fast-lookup set of (ss_id, example_no) for already processed items.

        Returns:
            Set[Tuple[str, int]]: set of unique keys.
        """
        processed_t2s_items_keys: Set[Tuple[str, int]] = set()
        for item in self.processed_t2s_items:
            ss_id = str(item.get("ss_id"))
            example_no = int(item.get("example_no"))
            processed_t2s_items_keys.add((ss_id, example_no))

        print(f"Previously processed {len(processed_t2s_items_keys)} item-keys loaded. ")
        return processed_t2s_items_keys
    
    @staticmethod
    def _item_key(item: Dict[str, Any]) -> Tuple[str, int]:
        """
        Extract the unique key for a dataset item.

        Raises:
            KeyError/ValueError if fields are missing or malformed.
        """
        ss_id = str(item["ss_id"])
        example_no = int(item["example_no"])
        return ss_id, example_no
    
    def _has_been_processed(self, item: Dict[str, Any]) -> bool:
        """Return True if item key is in the processed set."""
        try:
            return self._item_key(item) in self.processed_t2s_items_keys
        except (KeyError, ValueError):
            # If key malformed, treat as not processed so we can log/handle it.
            return False

    def iter_unprocessed(self, items: Iterable[Dict[str, Any]]) -> Iterable[Dict[str, Any]]:
        """Yield only unprocessed items."""
        for it in items:
            if not self._has_been_processed(it):
                yield it

    
    def load_language_model_and_tokenizer(self) -> Tuple:
        """"
        Loading language model and its tokenizer
        """
        self.eval_logger.info('Loading Model...') 
        print('Loading Model...') 
        ## Train configurations
        train_configs = self.args.config['train']
        prompt_temp_name = train_configs.get("prompt_temp_name", "")
        if prompt_temp_name == "slm_t2s":
            ptn = "ST"
        elif prompt_temp_name == "csc_t2s":
            ptn = "CT"
        else:
            ptn = "T"
        use_few_shot = bool(train_configs["use_few_shot"])
        few_shot_cnt = int(train_configs["few_shot_cnt"]) if use_few_shot else 0
        use_reasoning_in_few_shots = bool(train_configs.get("use_reasoning_in_few_shots", False)) if use_few_shot else False
        urifs_in_training = "t" if use_reasoning_in_few_shots else "f"
        use_cvd = bool(train_configs.get("use_col_value_and_descriptions", False))
        use_cvd_init = 't' if use_cvd else "f"
        use_grpo = bool(train_configs["use_grpo"])
        use_lora = bool(train_configs["use_lora"])
        use_reasoning = bool(train_configs["use_reasoning"])
        base_model_name = train_configs['base_model_name']
        self.base_model_name = base_model_name
        lora_params = train_configs.get('lora_params', {})
        training_params = train_configs['training_params']
        max_seq_length = training_params['max_seq_length']

        if ('gpt' in base_model_name) or ('gemini' in base_model_name):
            self.eval_logger.info(f'Model ({base_model_name}) is loaded.')
            return None, None, base_model_name

        eval_configs = self.args.config['evaluation']
        eval_base_model = eval_configs['eval_base_model']
        use_unsloth = bool(eval_configs['use_unsloth'])
        # use_unsloth_flash_attention_2 = bool(eval_configs['use_unsloth_flash_attention_2'])

        if eval_base_model:
            model_name = base_model_name
        else:
            # Extract model repo name parts
            data_mode = self.args.config['data_mode']
            t2s_dataset_name = self.args.config['dataset']
            base_model_id_without_user = base_model_name.split("/")[-1]
            db_ids_initials = []
            for db_name in self.args.db_ids:
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

            # Construct model repo name
            epoch = int(training_params['num_train_epochs'])
            bs = int(training_params['per_device_train_batch_size'])
            gas=int(training_params["gradient_accumulation_steps"])
            learningrate=float(training_params["learning_rate"])
            
            model_name = f"{base_model_id_without_user}_{t2s_dataset_name[0]}{data_mode[0]}_{db_id_str}_{task_str}_r{r}_a{alpha}_e{epoch}_bs{bs}_gas{gas}_lr{learningrate}_fs{few_shot_cnt}{urifs_in_training}_cvd{use_cvd_init}_pt{ptn}"
            if use_reasoning:
                model_name = f"{model_name}_sftreason" # Reasoning with SFT
            elif use_grpo:
                model_name = f"{model_name}_grpo"


            
            model_name = f"{os.getenv('HF_USER')}/{model_name}"
            self.eval_logger.info(f"=== Loading Model: {model_name} ")
            print(f"=== Loading Model: {model_name} ")

        # if use_unsloth and "qwen" not in model_name.lower():
        if use_unsloth:
            try:
                model, tokenizer = FastLanguageModel.from_pretrained(
                    model_name = model_name,
                    load_in_4bit = train_configs.get('load_in_4bit', False),
                    max_seq_length=max_seq_length 
                    # attn_implementation="flash_attention_2"
                    # max_seq_length = max_seq_length, # 'Qwen2Model' object has no attribute 'max_seq_length
                    # use_flash_attention_2 = use_unsloth_flash_attention_2 # Error: Both attn_implementation="eager" and `use_flash_attention_2=True` were used when loading the model, which are not compatible. We recommend to just use `attn_implementation="flash_attention_2"` when loading the model.
                )
                FastLanguageModel.for_inference(model) # 2x faster inference 
                model.eval()
            except Exception as e1:
                self.eval_logger.error(f"Unsloth couldn't find or load the model with the name of {model_name}. Error: {e1}")
                print(f"Unsloth couldn't find or load the model with the name of {model_name}. Error: {e1}")
                self.eval_logger.info(f"Trying AutoModelForCausalLM from transformers library")
                print(f"Trying AutoModelForCausalLM from transformers library")
                use_unsloth = False
        
        # if not use_unsloth or "qwen" in model_name.lower():
        if not use_unsloth:
            self.eval_logger.info(f"Trying AutoModelForCausalLM from transformers library")
            print(f"Trying AutoModelForCausalLM from transformers library")
            try:
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit = train_configs.get('load_in_4bit', False),
                    bnb_4bit_compute_dtype=torch.float16
                )
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    quantization_config=bnb_config,
                    device_map="auto"
                )
                model.eval()
            except Exception as e2:
                raise ValueError(f"There is no such model ({model_name}) in the hub. Check your config file and arrange accordingly. Error: {e2}")

        self.eval_logger.info('Model is loaded.')
        print('Model is loaded.')
        print("==="*50)
        print("==="*50)
        print("==="*50)
        self.eval_logger.info(f'Model name: {model_name}.')
        print(f'Model name: {model_name}.')
        print(f"Model config = {model.config}") # DELETE or COMMENT OUT LATER
        print(f"model max_position_embeddings = {getattr(model.config, 'max_position_embeddings')}") # DELETE or COMMENT OUT LATER
        print("==="*50)
        print("==="*50)
        print("==="*50)
        return model, tokenizer, model_name
    
    def _construct_vdb_service(self):
        """
        Initialize a T2SSyntheticVDBService object instance
        """
        vector_db_configs = self.args.config['vector_db']
        self.embedding_model_provider = str(vector_db_configs['model_provider'])
        self.embedding_model_name_or_path = str(vector_db_configs['model_name_or_path'])

        eval_configs = self.args.config['evaluation']
        keyword_extraction_llm_model = str(eval_configs['keyword_extraction_llm_model'])

        if keyword_extraction_llm_model:
            kex_model_name = keyword_extraction_llm_model
            kex_llm_model = None
            kex_llm_tokenizer = None
        else:
            kex_model_name = self.model_name
            kex_llm_model = self.model,
            kex_llm_tokenizer = self.tokenizer

        vdb_service = T2SSyntheticVDBService(
            dbs_root_dir=self.args.dbs_root_dir,
            embedding_model_provider=self.embedding_model_provider,
            embedding_model_name_or_path=self.embedding_model_name_or_path,
            llm_model_name=kex_model_name,
            llm_model=kex_llm_model,
            llm_tokenizer=kex_llm_tokenizer,
            logger=self.eval_logger
        )

        return vdb_service
    
    def _prepare_prompt(self, t2s_dict: Dict[str, Any], prompt_template: str) -> str:
        """
        Prepares the full prompt string for a given text-to-SQL item.
        This includes formatting the schema, few-shot examples, and the question.
        This function is called only ONCE per question to avoid redundant work.
        """
        ss_id: str = t2s_dict.get("ss_id")
        db_id = ss_id.split("-")[0]
        question = t2s_dict.get('question', '')
        evidence = t2s_dict.get('evidence', '')
        question = f"{question} Hint: {evidence}" if evidence else question

        eval_configs = self.args.config['evaluation']
        use_few_shot = bool(eval_configs['use_few_shot'])
        few_shot_cnt = int(eval_configs['few_shot_cnt']) if use_few_shot else 0
        use_reasoning_in_few_shots = bool(eval_configs['use_reasoning_in_few_shots'])
        use_schema = bool(eval_configs['use_schema'])
        use_cvd = bool(eval_configs['use_col_value_and_descriptions'])
        schema_content = str(eval_configs['schema_content'])
        prompt_temp_name = str(eval_configs["prompt_temp_name"])
        
        # --- 1. Prepare Few-Shot Examples ---
        few_shot_augmentation_string = ""
        if use_few_shot:
            few_shot_examples = t2s_dict.get('few_shot', {}).get('examples', [])[:few_shot_cnt]
            few_shot_string = ""
            for example_idx, example_dict in enumerate(few_shot_examples):
                synthetic_question = example_dict.get("question")
                synthetic_sql = example_dict.get("SQL")
                example_dac_reasoning = example_dict.get("dac_reasoning")
                example_string = f"Example {example_idx+1}:\n"
                example_string += f"Question: {synthetic_question}\n"
                if use_reasoning_in_few_shots:
                    example_string += f"<reasoning>{example_dac_reasoning}</reasoning>\n"
                example_string += f"<answer>{synthetic_sql}</answer>\n"
                few_shot_string += example_string + "\n"
            
            few_shot_instructions = "- Below example question and their corresponding SQL queries are given as an example. Read them carefully and analyze the example question intentions, understand the link between database items and question. These examples can help you to reach correct response.\n"
            few_shot_augmentation_string = "**EXAMPLES**\n" + few_shot_instructions + few_shot_string + "\n"

        # --- 2. Prepare Schema String ---
        schema_augmentation_string = ""
        if use_schema:
            db_info = DatabaseGeneralInfo(db_id=db_id, dbs_root_dir=self.args.dbs_root_dir)
            schema_generator = None
            
            if schema_content == "whole_schema":
                schema_generator = db_info.original_db_schema_generator
            elif schema_content == "ground_truth_schema":
                gt_sql = t2s_dict['SQL']
                gt_schema_dict: Dict[str, List[str]] = get_sql_columns_dict(db_path=db_info.db_path, sql=gt_sql)
                schema_structure = DatabaseSchema.from_schema_dict(gt_schema_dict)
                schema_generator = DatabaseSchemaGenerator(
                    tentative_schema=schema_structure, db_id=db_id, db_path=db_info.db_path
                )
            elif schema_content == "filtered_schema":
                filtered_schema_dict = t2s_dict.get('filtered_schema', {}).get('schema_dict', {})
                if not filtered_schema_dict:
                    self.eval_logger.info(f"Couldn't find filtered schema dictionary. Continuing with the full schema...")
                    filtered_schema_dict: Dict[str, List[str]] = get_db_schema(db_info.db_path)
                schema_structure = DatabaseSchema.from_schema_dict(filtered_schema_dict)
                schema_generator = DatabaseSchemaGenerator(
                    tentative_schema=schema_structure, db_id=db_id, db_path=db_info.db_path
                )

            if schema_generator:
                schema_string = schema_generator.generate_schema_string(
                    include_column_value_examples=use_cvd, include_value_description=use_cvd
                )
                column_meanings_str = schema_generator.get_column_profiles_string(with_keys=False, with_references=False) if use_cvd else ""
                
                schema_instruction = "- Deeply analyze the database schema and information related with the schema items. Link user question with the database items.\n"
                schema_augmentation_string = "**DATABASE SCHEMA INFORMATION**\n" + schema_instruction + schema_string + "\n"
                if use_cvd and column_meanings_str:
                    schema_augmentation_string += "**COLUMN INFORMATION**\n" + column_meanings_str + "\n"

        # --- 3. Assemble Final Prompt ---
        augmentation_string = few_shot_augmentation_string + schema_augmentation_string
        if prompt_temp_name == "slm_t2s" or prompt_temp_name == "csc_t2s" :
            prompt = prompt_template.format(AUGMENTATION=augmentation_string, QUESTION=question)
        elif prompt_temp_name == "t2s":
            prompt = prompt_template.format(DB_ID=db_id, AUGMENTATION=augmentation_string, QUESTION=question)
        else:
            # Fallback for other templates
            prompt = prompt_template.format(AUGMENTATION=augmentation_string, QUESTION=question)

        return prompt
    
    def _generate_and_evaluate_sql(self, prompt: str, t2s_dict: Dict[str, Any], ex_id: int) -> Dict[str, Any]:
        """
        Generates a single SQL query from a pre-built prompt, parses the output,
        and evaluates its correctness.
        """
        item_key: Tuple[str, int] = self._item_key(item=t2s_dict)
        item_key_str = f"{item_key[0]}_{item_key[1]}"
        ss_id: str = t2s_dict.get("ss_id")
        db_id = ss_id.split("-")[0]

        gt_sql = t2s_dict['SQL']
        
        occured_error = ""
        output_text = ""
        
        # --- 1. Set Generation Parameters ---
        eval_configs = self.args.config['evaluation']
        max_new_tokens = eval_configs['max_new_tokens']
        temperature = eval_configs['temperature'][ex_id]
        top_p = eval_configs['top_p'][ex_id]
        
        self.eval_logger.info(f"===============Question:{item_key} (in ex_{ex_id}) (temp:{temperature} - top_p:{top_p})===============")

        # --- 2. Generate SQL ---
        if ('gemini' in self.model_name) or ('gpt' in self.model_name):
            try:
                llm_service = LLMService(model_name=self.model_name, logger=self.eval_logger)
                response_object,  prompt_token_cnt, completion_token_cnt, total_token_cnt  = llm_service.call_llm(prompt=prompt, temperature=temperature, top_p=top_p)
                output_text = response_object.text
            except Exception as e:
                occured_error = f"Error during generation: {e}\n{traceback.format_exc()}"
                self.eval_logger.error(occured_error)
        else:
            try:
                max_seq_length = int(self.args.config['train']['training_params']['max_seq_length'])
                model_device = next(self.model.parameters()).device
                inputs = self.tokenizer(prompt, return_tensors='pt', truncation=True, max_length=max_seq_length).to(model_device)

                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        do_sample=False, 
                        pad_token_id=self.tokenizer.eos_token_id
                    )
                output_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                self.eval_logger.info(f"----------RESPONSE Question:{item_key} (in ex_{ex_id}):\n {output_text}")
            except Exception as e:
                occured_error = f"Error during generation: {e}\n{traceback.format_exc()}"
                self.eval_logger.error(occured_error)

        # --- 3. Parse and Evaluate SQL ---
        try:
            response_text = extract_response_part(output_text)
            predicted_sql = extract_sql_part(extract_xml_answer(response_text))
            reasoning = extract_xml_reasoning(response_text)
        except Exception as e:
            predicted_sql, reasoning = "", ""
            error_traceback = traceback.format_exc()
            occured_error += f"\nError during parsing: {e}\n{error_traceback}"
            self.eval_logger.error(f"Error during parsing for Q_KEY {item_key}: {e}\n{error_traceback}")

        ## Evaluating the correctness of predicted SQL query
        # Calculate Execution Accuracy for the predicted SQL
        try:
            db_path = self.args.dbs_root_dir / db_id / f"{db_id}.sqlite"
            comparison_dict = compare_sqls(db_path=db_path, predicted_sql=predicted_sql, ground_truth_sql=gt_sql)
            exec_res = comparison_dict['exec_res']
            exec_err = comparison_dict['exec_err']
            soft_f1_score = calculate_f1_score_for_sql(predicted_sql, gt_sql, db_path)
            f1_score = soft_f1_score if soft_f1_score else 0
        except Exception as e:
            self.eval_logger.error(f"Error during SQL evaluation for Q_KEY {item_key}. Error: {e}")
            exec_res, exec_err, f1_score = 0, str(e), 0

        self.eval_logger.info(f"----- PREDICTED_SQL: {predicted_sql} \n----- GT_SQL: {gt_sql} \n----- exec_res: {exec_res} | f1_score: {f1_score:.4f}")

        return {
            "ex_id": ex_id,
            "predicted_sql": predicted_sql,
            "reasoning": reasoning,
            "exec_res": exec_res,
            "exec_err": exec_err,
            "f1_score": f1_score,
            "occured_error": occured_error
        }
        
    
    def compute_bounds(self, t2s_dicts_with_translations: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Computes the upper and lower bound of EX values across all evaluations.
        """
        max_ex_scores = []
        min_ex_scores = []
        max_f1_scores = []
        min_f1_scores = []
        for t2s_dict in t2s_dicts_with_translations:
            translations = t2s_dict.get("translations", {})
            max_ex_score = 0.0
            max_f1_score = 0.0
            min_ex_score = 1.0
            min_f1_score = 1.0
            for translation_idx, translation_output in translations.items():
                ex_score = translation_output.get("exec_res", 0.0)
                if ex_score > max_ex_score:
                    max_ex_score = ex_score
                if ex_score < min_ex_score:
                    min_ex_score = ex_score

                f1_score = translation_output.get("f1_score", 0.0)
                if f1_score > max_f1_score:
                    max_f1_score = f1_score
                if f1_score < min_f1_score:
                    min_f1_score = f1_score
            
            max_ex_scores.append(max_ex_score)
            max_f1_scores.append(max_f1_score)
            min_ex_scores.append(min_ex_score)
            min_f1_scores.append(min_f1_score)


        ex_upper_bound = sum(max_ex_scores) / len(t2s_dicts_with_translations)
        f1_upper_bound = sum(max_f1_scores) / len(t2s_dicts_with_translations)
        ex_lower_bound = sum(min_ex_scores) / len(t2s_dicts_with_translations)
        f1_lower_bound = sum(min_f1_scores) / len(t2s_dicts_with_translations)

        bounds = {
            "ex_upper_bound": ex_upper_bound,
            "ex_lower_bound": ex_lower_bound,
            "f1_upper_bound": f1_upper_bound,
            "f1_lower_bound": f1_lower_bound
        }

        return bounds
    
    def find_best_translation_ids(self, t2s_dicts_with_translations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Find the best execution id to find best temperature and top_p
        """
        execution_ex_values = {}
        execution_f1_values = {}
        execution_performances = {}
        for t2s_dict in t2s_dicts_with_translations:
            translations = t2s_dict.get("translations", {})
            for translation_idx, translation_output in translations.items():
                if translation_idx not in execution_ex_values:
                    execution_ex_values[translation_idx] = []
                if translation_idx not in execution_f1_values:
                    execution_f1_values[translation_idx] = []
                
                ex_score = translation_output.get("exec_res", 0.0)
                f1_score = translation_output.get("f1_score", 0.0)
                execution_ex_values[translation_idx].append(ex_score)
                execution_f1_values[translation_idx].append(f1_score)
        
        t_id_with_max_ex_value = None
        max_ex_value = 0.0
        for translation_idx, ex_scores in execution_ex_values.items():
            avg_ex_score = sum(ex_scores) / len(ex_scores) if ex_scores else 0.0
            if avg_ex_score > max_ex_value:
                max_ex_value = avg_ex_score
                t_id_with_max_ex_value = translation_idx

        t_id_with_max_f1_value = None
        max_f1_value = 0.0
        for translation_idx, f1_scores in execution_f1_values.items():
            avg_f1_score = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
            if avg_f1_score > max_f1_value:
                max_f1_value = avg_f1_score
                t_id_with_max_f1_value = translation_idx
        
        best_translation_ids = {
            "best_translation_id_based_on_ex": t_id_with_max_ex_value,
            "max_ex_value": max_ex_value,
            "best_translation_id_based_on_f1": t_id_with_max_f1_value,
            "max_f1_value": max_f1_value
        }

        return best_translation_ids
        

    def evaluate_with_different_sampling_settings(self):
        # Write configs
        write_config_file_path = self.eval_results_dir / "config.json"
        with open(write_config_file_path, "w") as file:
            json.dump(self.args.config, file, indent=4)

        t2s_dicts_path = self.eval_results_dir / 't2s_items.json'
        eval_start_time = time.time()
        
        eval_configs = self.args.config['evaluation']
        temperature_list = eval_configs['temperature']
        generation_count = len(temperature_list)

        # Load prompt template ONCE before the loop
        prompt_temp_name = str(eval_configs["prompt_temp_name"])
        prompt_template = load_template(template_name=prompt_temp_name)
        if not bool(eval_configs['use_reasoning']):
            pt_parts = prompt_template.split('<reasoning>')
            if len(pt_parts) > 1:
                prompt_template = pt_parts[0] + pt_parts[1].split('</reasoning>')[1]

        for t2s_dict_idx, t2s_dict in enumerate(self.eval_dataset):
            has_been_processed = self._has_been_processed(item=t2s_dict)
            if has_been_processed:
                print(f"\n===\nAlready processed item {t2s_dict_idx + 1}/{len(self.eval_dataset)} | DB-SS-ID: {t2s_dict['ss_id']} | Example No: {t2s_dict['example_no']}\n===\n")
                continue
            print(f"\n===\nProcessing item {t2s_dict_idx + 1}/{len(self.eval_dataset)} | DB-SS-ID: {t2s_dict['ss_id']} | Example No: {t2s_dict['example_no']}\n===\n")
            
            # Step 1. Prepare the prompt string ONCE for the current question
            prompt = self._prepare_prompt(t2s_dict, prompt_template)
            
            process_t2s_dict = {
                "ss_id": t2s_dict.get("ss_id"),
                "example_no": t2s_dict.get("example_no"),
                "difficulty": t2s_dict.get("difficulty"),
                "SQL": t2s_dict.get("SQL"),
                "question": t2s_dict.get("question")
            }
            process_t2s_dict["translations"] = {}

            # --- REFINEMENT: Replaced ThreadPoolExecutor with a simple, more direct for loop ---
            # Step 2. Run multiple generations sequentially.
            for idx in range(generation_count):
                try:
                    translation_output = self._generate_and_evaluate_sql(
                        prompt,    # Pass the pre-built prompt
                        t2s_dict,
                        idx
                    )
                    process_t2s_dict["translations"][str(translation_output.get("ex_id"))] = translation_output
                except Exception as e:
                    self.eval_logger.error(f"An error occurred during SQL generation/evaluation for ex_id {idx}: {e}\n{traceback.format_exc()}")

            
            # Add the currenlty processed data item into the processed list
            item_key: Tuple[str, int] = self._item_key(item=t2s_dict)
            self.processed_t2s_items_keys.add(item_key)
            # Save progress after each item
            self.processed_t2s_items.append(process_t2s_dict)
            # processed_t2s_items_keys.add((ss_id, example_no))
            with open(t2s_dicts_path, 'w') as file:
                json.dump(self.processed_t2s_items, file, indent=4)

        # Final calculations
        bounds = self.compute_bounds(self.processed_t2s_items)
        best_translation_ids = self.find_best_translation_ids(self.processed_t2s_items)

        overall_eval_info = {"bounds": bounds, "best_translation_ids": best_translation_ids}
        overall_eval_info_path = self.eval_results_dir / "overall_eval_info.json"
        with open(overall_eval_info_path, 'w') as file:
            json.dump(overall_eval_info, file, indent=4)

        eval_duration = (time.time() - eval_start_time) / 60
        self.eval_logger.info(f"-- Overall Evaluation Duration: {eval_duration:.2f} minutes")
        print(f"Evaluation finished in {eval_duration:.2f} minutes.")


    def _parse_and_evaluate_single_sql(self, output_text: str, t2s_dict: Dict[str, Any], ex_id: int) -> Dict[str, Any]:
        """
        Parses a single generated text output and evaluates its correctness against the ground truth.
        """
        item_key: Tuple[str, int] = self._item_key(item=t2s_dict)
        item_key_str = f"{item_key[0]}_{item_key[1]}"
        ss_id: str = t2s_dict.get("ss_id")
        db_id = ss_id.split("-")[0]

        gt_sql = t2s_dict['SQL']
        predicted_sql = ""
        reasoning = ""
        exec_res = 0.0
        exec_err = ""
        f1_score = 0.0
        occured_error = ""

        # --- 1. Parse SQL from the generated text ---
        try:
            response_text = extract_response_part(output_text)
            answer_part = extract_xml_answer(response_text)
            # print(f"Answer tag content: {answer_part}") ## Delete later
            predicted_sql = extract_sql_part(answer_part)
            reasoning = extract_xml_reasoning(response_text)
        except Exception as e:
            predicted_sql, reasoning = "", ""
            error_traceback = traceback.format_exc()
            occured_error = f"\nError during parsing: {e}\n{error_traceback}"
            print(f"Error during parsing for Q_KEY {item_key}: {e}\n{error_traceback}")
            self.eval_logger.error(f"Error during parsing for Q_KEY {item_key}: {e}\n{error_traceback}")

        # --- 2. Evaluate the parsed SQL ---
        try:
            db_path = self.args.dbs_root_dir / db_id / f"{db_id}.sqlite"
            comparison_dict = compare_sqls(db_path=db_path, predicted_sql=predicted_sql, ground_truth_sql=gt_sql)
            exec_res = comparison_dict['exec_res']
            exec_err = comparison_dict['exec_err']
            soft_f1_score = calculate_f1_score_for_sql(predicted_sql, gt_sql, db_path)
            f1_score = soft_f1_score if soft_f1_score else 0
        except Exception as e:
            error_traceback = traceback.format_exc()
            print(f"Error during SQL evaluation for QQ_KEY_ID {item_key}. Error: {e}. \n {error_traceback}")
            self.eval_logger.error(f"Error during SQL evaluation for QQ_KEY_ID {item_key}. Error: {e}. \n {error_traceback}")
            exec_res = 0.0
            exec_err = str(e)
            f1_score = 0.0
            occured_error = str(e)

        self.eval_logger.info(f"\n\n-----ex_id{ex_id} \n PREDICTED_SQL: {predicted_sql} \n----- GT_SQL: {gt_sql} \n----- exec_res: {exec_res} | f1_score: {f1_score:.4f}")

        return {
            "ex_id": ex_id,
            "predicted_sql": predicted_sql,
            "reasoning": reasoning,
            "exec_res": exec_res,
            "exec_err": exec_err,
            "f1_score": f1_score,
            "occured_error": occured_error
        }
    
    def evaluate_with_single_sampling_setting(self):

        # Write configs
        write_config_file_path = self.eval_results_dir / "config.json"
        with open(write_config_file_path, "w") as file:
            json.dump(self.args.config, file, indent=4)

        t2s_dicts_path = self.eval_results_dir / 't2s_items.json'
        eval_start_time = time.time()
        
        eval_configs = self.args.config['evaluation']
        if isinstance(eval_configs['temperature'], list):
            temperature = eval_configs['temperature'][0]
            generation_count = len(eval_configs['temperature'])
        elif isinstance(eval_configs['temperature'], int):
            temperature = eval_configs['temperature']
            generation_count = 9 # HARDCODED

        if isinstance(eval_configs['top_p'], list):
            top_p = eval_configs['top_p'][0]
        elif isinstance(eval_configs['top_p'], int):
            top_p = eval_configs['top_p']

        max_new_tokens = eval_configs['max_new_tokens']
        
        # Load prompt template ONCE before the loop
        prompt_temp_name = str(eval_configs["prompt_temp_name"])
        prompt_template = load_template(template_name=prompt_temp_name)
        if not bool(eval_configs['use_reasoning']):
            pt_parts = prompt_template.split('<reasoning>')
            if len(pt_parts) > 1:
                prompt_template = pt_parts[0] + pt_parts[1].split('</reasoning>')[1]

        for t2s_dict_idx, t2s_dict in enumerate(self.eval_dataset):
            has_been_processed = self._has_been_processed(item=t2s_dict)
            if has_been_processed:
                print(f"\n\n ===\n*** Already processed item {t2s_dict_idx + 1}/{len(self.eval_dataset)} | DB-SS-ID: {t2s_dict['ss_id']} | Example No: {t2s_dict['example_no']} \n===\n")
                continue
            print(f"\n\n ===\n*** Processing item {t2s_dict_idx + 1}/{len(self.eval_dataset)} | DB-SS-ID: {t2s_dict['ss_id']} | Example No: {t2s_dict['example_no']} \n===\n")
            
            
            # Step 1. Prepare the prompt string ONCE.
            prompt = self._prepare_prompt(t2s_dict, prompt_template)
            if t2s_dict_idx < 2:
                print(f"PROMPT: {prompt}")
            
            process_t2s_dict = {
                "ss_id": t2s_dict.get("ss_id"),
                "example_no": t2s_dict.get("example_no"),
                "difficulty": t2s_dict.get("difficulty"),
                "SQL": t2s_dict.get("SQL"),
                "question": t2s_dict.get("question")
            }
            process_t2s_dict["translations"] = {}

            # --- BATCHED GENERATION: Generate all N sequences in a single, parallel call ---
            generated_outputs = []
            try:
                max_seq_length = int(self.args.config['train']['training_params']['max_seq_length'])
                model_device = next(self.model.parameters()).device
                inputs = self.tokenizer(prompt, return_tensors='pt', truncation=True, max_length=max_seq_length).to(model_device)

                with torch.no_grad():
                    # The key change is here!
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        do_sample=True, # Must be True for sampling # When do_sample is set to False, We got ":An error occurred during batched SQL generation: Greedy methods without beam search do not support `num_return_sequences` different than 1 (got 9)"
                        num_return_sequences=generation_count, # Generate N sequences
                        pad_token_id=self.tokenizer.eos_token_id
                    )
                # Decode all the generated sequences
                generated_outputs = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
                self.eval_logger.info(f"Successfully generated {len(generated_outputs)} sequences in a single batch for DB-SS-ID: {t2s_dict['ss_id']} | Example No: {t2s_dict['example_no']}")

            except Exception as e:
                error_reason = traceback.format_exc()
                self.eval_logger.error(f"An error occurred during batched SQL generation: {e}\n{traceback.format_exc()} \n {error_reason}")

            # Step 2. Parse and evaluate each generated sequence. 
            """
            # Before
            for idx, output_text in enumerate(generated_outputs):
                self.eval_logger.info(f"--- Parsing and evaluating sequence {idx+1}/{generation_count} ---")
                translation_output = self._parse_and_evaluate_single_sql(output_text, t2s_dict, idx)
                t2s_dict["translations"][str(idx)] = translation_output
            """
            max_workers = min(10, os.cpu_count() or 1) 
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all parsing/evaluation jobs to the thread pool
                futures = {
                    executor.submit(self._parse_and_evaluate_single_sql, output_text, t2s_dict, idx): idx
                    for idx, output_text in enumerate(generated_outputs)
                }

                # Process results as they are completed
                for future in as_completed(futures):
                    idx = futures[future]
                    try:
                        translation_output = future.result(timeout=120)
                        process_t2s_dict["translations"][str(translation_output.get("ex_id"))] = translation_output
                    except TimeoutError:
                        self.eval_logger.error(f"Worker for sequence {idx} timed out and is likely hung. Skipping this result.")
                        # Optionally, add a placeholder result for the timed-out query
                        process_t2s_dict["translations"][str(idx)] = {
                            "ex_id": idx,
                            "predicted_sql": "QUERY TIMED OUT",
                            "reasoning": "",
                            "exec_res": 0.0,
                            "exec_err": "Worker thread timed out after 120 seconds.",
                            "f1_score": 0.0,
                            "occured_error": "Worker thread timed out."
                        }

                    except Exception as e:
                        error_reason = traceback.format_exc()
                        idx = futures[future]
                        self.eval_logger.error(f"Error processing sequence {idx} in parallel: {e}\n{traceback.format_exc()} \n {error_reason}")
                        process_t2s_dict["translations"][str(idx)] = {
                            "ex_id": idx,
                            "predicted_sql": "Error",
                            "reasoning": "",
                            "exec_res": 0.0,
                            "exec_err": "Worker thread error.",
                            "f1_score": 0.0,
                            "occured_error": "Worker thread error."
                        }
            

            # Add the currenlty processed data item into the processed list
            item_key: Tuple[str, int] = self._item_key(item=t2s_dict)
            self.processed_t2s_items_keys.add(item_key)
            # Save progress after each item
            self.processed_t2s_items.append(process_t2s_dict)
            # processed_t2s_items_keys.add((ss_id, example_no))
            with open(t2s_dicts_path, 'w') as file:
                json.dump(self.processed_t2s_items, file, indent=4)

        # Final calculations
        bounds = self.compute_bounds(self.processed_t2s_items)
        best_translation_ids = self.find_best_translation_ids(self.processed_t2s_items)

        overall_eval_info = {"bounds": bounds, "best_translation_ids": best_translation_ids}
        self.eval_logger.info(f"overall_eval_info: \n {json.dumps(overall_eval_info, indent=4)}")
        overall_eval_info_path = self.eval_results_dir / "overall_eval_info.json"
        with open(overall_eval_info_path, 'w') as file:
            json.dump(overall_eval_info, file, indent=4)

        eval_duration = (time.time() - eval_start_time) / 60
        self.eval_logger.info(f"-- Overall Evaluation Duration: {eval_duration:.2f} minutes")
        print(f"Evaluation finished in {eval_duration:.2f} minutes.")

        return
    
    def _is_single_sampling_setting_used(self):
        
        eval_configs = self.args.config['evaluation']
        temperatures = eval_configs['temperature']
        top_ps = eval_configs['top_p']

        if isinstance(temperatures, int) and isinstance(top_ps, int):
            return True
        elif isinstance(temperatures, int) and isinstance(top_ps, list):
            raise ValueError("Type mismatch between temperatures and top-p. They must be same")
        elif isinstance(temperatures, list) and isinstance(top_ps, int):
            raise ValueError("Type mismatch between temperatures and top-p. They must be same")
        

        initial_temp_value = temperatures[0]
        for temp_value in temperatures:
            if temp_value != initial_temp_value:
                return False
            
        initial_topp_value = top_ps[0]
        for top_p_value in top_ps:
            if top_p_value != initial_topp_value:
                return False
            
        return True


    def evaluate(self):
        eval_configs = self.args.config['evaluation']
        use_few_shot = bool(eval_configs['use_few_shot'])
        few_shot_cnt = int(eval_configs['few_shot_cnt'])
        use_reasoning_in_few_shots = bool(eval_configs['use_reasoning_in_few_shots'])

        is_single_sampling_setting_used = self._is_single_sampling_setting_used()
        self.eval_logger.info(f"is_single_sampling_setting_used: {is_single_sampling_setting_used}")
        if is_single_sampling_setting_used:
            self.eval_logger.info("Running evaluate_with_single_sampling_setting")
            self.evaluate_with_single_sampling_setting()
        else:
            self.eval_logger.info("Running evaluate_with_different_sampling_settings")
            self.evaluate_with_different_sampling_settings()

            
          
        



