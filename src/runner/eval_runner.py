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
from typing import List, Tuple, Dict, Any, Optional, Literal
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

from concurrent.futures import ThreadPoolExecutor, as_completed
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

class EvalRunner:
    def __init__(self, args: Any):
        self.args = args
        self.all_db_ids: List[str] = self._set_db_ids()
        self.db_ids: List[str] = self.args.config.get("db_ids", [])
        self.dataset = self._load_dataset()
        self.eval_dataset = self._get_db_eval_dataset()

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

        data_path = Path(self.args.data_json_path)
        data_with_few_shots_file_name = data_path.stem + "_with_few_shots" + data_path.suffix 
        data_with_few_shots_path = data_path.parent / data_with_few_shots_file_name

        if data_with_few_shots_path.exists():
            if data_with_few_shots_path.suffix == ".json":  
                with open(data_with_few_shots_path, 'r') as file:
                    dataset = json.load(file)
            elif data_with_few_shots_path.suffix == ".jsonl":
                with open(data_with_few_shots_path, 'r') as file:
                    for line in file:
                        try:
                            example = json.loads(line)
                            if 'execution_status' in example:
                                if example.get('execution_status') == 'SYNTACTICALLY_CORRECT':
                                    dataset.append(example)
                            else:
                                dataset.append(example)
                        except:
                            continue
        else:
            raise ValueError(f"The {data_with_few_shots_path} doesn't exist.")

        return dataset
    
    def _get_db_eval_dataset(self) -> List[Dict[str, Any]]:
        eval_dataset = []
        for t2s_dict in self.dataset:
            # Check if the current dataset item is one of the considered database
            db_id = t2s_dict['db_id']
            if db_id in self.db_ids:
                eval_dataset.append(t2s_dict)

        return eval_dataset
    
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
        prompt_temp_name = str(eval_configs['prompt_temp_name'])
        ptn = "ST" if prompt_temp_name == "slm_t2s" else "T"
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
    
        eval_sub_dir_name = f"ptn{ptn}_ebm{ebm}_upm{upm}_cvd{cvd}_us{us}_sc{sc}_ufs{ufs}_fsc{fsc}_urifs{urifs}_ur{ur}_en{en}"

        # eval_results_dir = Path(f"./results/{db_ids_str}/{pure_model_name}/{self.args.run_start_time}")
        eval_results_dir = Path(f"./results/{db_ids_str}/{pure_model_name}/{eval_sub_dir_name}")
        eval_results_dir.mkdir(parents=True, exist_ok=True)

        return eval_results_dir

    
    def load_language_model_and_tokenizer(self) -> Tuple:
        """"
        Loading language model and its tokenizer
        """
        self.eval_logger.info('Loading Model...') 
        print('Loading Model...') 
        ## Train configurations
        train_configs = self.args.config['train']
        prompt_temp_name = train_configs.get("prompt_temp_name", "")
        ptn = "ST" if prompt_temp_name == "slm_t2s" else "T"
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

    def translate_text_to_sql(self, t2s_dict: Dict[str, Any], model=None, tokenizer=None, model_name: str = None, ex_id: int = 0) -> Dict[str, Any]:
        """
        Generating SQL query for a user question

        Arguments:
            t2s_dict (Dict[str, Any]): A dataset item 
        
        Returns:
            Dict[str, Any]: A new dictionary containing necessary keys as follows: ex_id, predicted_sql, reasoning, exec_res, exec_err, f1_score, output_text, occured_error
        """        
        q_id = t2s_dict['question_id']
        gt_sql = t2s_dict['SQL']
        db_id = t2s_dict['db_id']
        question = t2s_dict['question']
        evidence = t2s_dict.get('evidence', '')
        question = question + " Hint: " + evidence if evidence else question

        occured_error = ""
        output_text = ""
        predicted_sql = ""

        ## Training configurations
        train_configs = self.args.config['train']
        max_seq_length = int(train_configs['training_params']['max_seq_length'])

        ## Evaluation configurations
        eval_configs = self.args.config['evaluation']       

        eval_base_model = bool(eval_configs['eval_base_model'])
        use_reasoning = bool(eval_configs['use_reasoning'])
        output_format = eval_configs['output_format']
        use_cvd = bool(eval_configs['use_col_value_and_descriptions'])
        prompt_temp_name = str(eval_configs["prompt_temp_name"])
        use_schema = bool(eval_configs['use_schema'])
        schema_content = str(eval_configs['schema_content'])
        use_few_shot = bool(eval_configs['use_few_shot'])
        few_shot_cnt = int(eval_configs['few_shot_cnt']) if use_few_shot else 0
        use_reasoning_in_few_shots = bool(eval_configs['use_reasoning_in_few_shots'])
        max_new_tokens = eval_configs['max_new_tokens']
        # Determine the temperature and top_p parameters for the current run
        temperature = eval_configs['temperature'][ex_id]
        top_p = eval_configs['top_p'][ex_id]
        # self.eval_logger.info(f"temperature: {temperature} - top_p: {top_p}")
        self.eval_logger.info(f"===============Question:{q_id} (in ex_{ex_id}) (temp:{temperature} - top_p:{top_p})===============")
        self.eval_logger.info(f"Question and Hint: {question}")

        if eval_base_model:
            # Since base model doesn't trained for learning database schema, it should be given in prompt
            use_schema = True
            eval_configs['use_schema'] = True
        else:
            if "reason" in model_name:
                # if model name includes sftreason, then it is specifically designed to make reasoning 
                use_reasoning = True
                eval_configs['use_reasoning'] = True
            else:
                use_reasoning = False
                eval_configs['use_reasoning'] = False

        if 'cvd' in model_name:
            use_cvd = True
            eval_configs['use_col_value_and_descriptions'] = True
        else:
            use_cvd = bool(eval_configs['use_col_value_and_descriptions'] )
        # print(f"use cvd: {use_cvd}")
        
        similar_synthetic_examples = []
        if use_few_shot:
            # Get similar synthetic examples using vector db
            # similar_synthetic_examples = self.vdb_service.search_examples(question_and_hint=question, db_id=db_id, k=3)   # This slows down the process of evaluation, so i have prepared few-shots
            for examples in t2s_dict.get('few_shot', {}).get('examples', []):
                similar_synthetic_examples.append(examples)
        
        all_few_shot_examples = t2s_dict.get('few_shot', {}).get('examples', [])
        few_shot_examples = t2s_dict.get('few_shot', {}).get('examples', [])[:few_shot_cnt] # Get top n similar examples

        # self.eval_logger.info(f"+++++++ few shot examples ++++++\n {type(few_shot_examples)} \n{few_shot_examples}") # DELETE LATER

        few_shot_augmentation_string = ""
        if use_few_shot:
            few_shot_string = ""
            for example_idx, example_dict in enumerate(few_shot_examples):
                # self.eval_logger.info(f"{type(example_dict)}\n example_dict: {example_dict}") # DELETE OR COMMENT OUT LATER
                synthetic_question = example_dict.get("question")
                synthetic_sql = example_dict.get("SQL")
                example_dac_reasoning = example_dict.get("dac_reasoning")
                example_string = f"Example {example_idx+1}:\n"
                example_string += f"Question: {synthetic_question}\n"
                if use_reasoning_in_few_shots:
                    example_string += f"<think>{example_dac_reasoning}</think>\n"
                example_string += f"<answer>{synthetic_sql}</answer>\n"
                few_shot_string += example_string + "\n"
                ## IDEA: Can giving search keyword for each example increase the EX, as it might add where to focus on the examples? Or we may direct LLM to focus on that part.

            few_shot_instructions = "- Below example question and their corresponding SQL queries are given as an example. Read them carefully and analyze the example question intentions, understand the link between database items and question. These examples can help you to reach correct response.\n"
            few_shot_augmentation_string = "**EXAMPLES**\n" + few_shot_instructions + few_shot_string + "\n"
        

        used_schema_dict: Dict[str, List[str]] = {}
        schema_str = ""
        # construct schema string
        schema_augmentation_string = ""
        schema_string = ""
        column_meanings_str = ""
        if use_schema:
            db_info = DatabaseGeneralInfo(db_id=db_id, dbs_root_dir=self.args.dbs_root_dir)
            if schema_content == "whole_schema":
                schema_string = db_info.original_db_schema_generator.generate_schema_string(
                    include_column_value_examples=use_cvd,
                    include_value_description=use_cvd
                )
                if use_cvd:
                    # Construct column meanings 
                    column_meanings_str = db_info.original_db_schema_generator.get_column_profiles_string(with_keys=False, with_references=False)

            elif schema_content == "ground_truth_schema":
                gt_sql = t2s_dict['SQL']
                gt_schema_dict: Dict[str, List[str]] = get_sql_columns_dict(db_path=db_info.db_path, sql=gt_sql)
                schema_structure = DatabaseSchema.from_schema_dict(gt_schema_dict)
                schema_generator = DatabaseSchemaGenerator(
                    tentative_schema=schema_structure,
                    db_id=db_id,
                    db_path=db_info.db_path,
                    add_examples=False, 
                    add_random_examples=False  # making this True slow donw the process
                )
                schema_string = schema_generator.generate_schema_string(
                    include_column_value_examples=use_cvd, 
                    include_value_description=use_cvd
                    )
                if use_cvd:
                    # Construct column meanings 
                    column_meanings_str = schema_generator.get_column_profiles_string(with_keys=False, with_references=False)
            elif schema_content == "filtered_schema":
                filtered_schema_dict = t2s_dict.get('filtered_schema', {}).get('schema_dict', {})
                if not filtered_schema_dict:
                    self.eval_logger.info(f"Couldn't find filtered schema dictionary. Continuing with the full schema...")
                    filtered_schema_dict: Dict[str, List[str]] = get_db_schema(db_info.db_path)
                
                schema_structure = DatabaseSchema.from_schema_dict(filtered_schema_dict)
                schema_generator = DatabaseSchemaGenerator(
                    tentative_schema=schema_structure,
                    db_id=db_id,
                    db_path=db_info.db_path,
                    add_examples=use_cvd,
                    add_random_examples=use_cvd
                )
                schema_string = schema_generator.generate_schema_string(
                    include_column_value_examples=use_cvd,
                    include_value_description=use_cvd
                )
                if use_cvd:
                    # Construct column meanings 
                    column_meanings_str = schema_generator.get_column_profiles_string(with_keys=False, with_references=False)

            schema_instruction = "- Deeply analyze the database schema and information related with the schema items. Link user question with the database items.\n"
            schema_augmentation_string = "**DATABASE SCHEMA INFORMATION**\n" + schema_instruction +  schema_string + "\n"
            if use_cvd:
                schema_augmentation_string += "**COLUMN INFORMATION**\n" + column_meanings_str + "\n"

    

        # load prompt template
        prompt_template = load_template(template_name=prompt_temp_name)

        if not use_reasoning:
            pt = prompt_template.split('<think>')[0] + prompt_template.split('</think>')[1] 
            prompt_template = pt

        # Format the template
        if prompt_temp_name == "slm_t2s":
            augmentation_string = few_shot_augmentation_string + schema_augmentation_string
            prompt = prompt_template.format(
                AUGMENTATION = augmentation_string,
                QUESTION = question,
            )
        elif prompt_temp_name == "t2s":
            augmentation_string = few_shot_augmentation_string + schema_augmentation_string
            prompt = prompt_template.format(
                DB_ID = db_id,
                AUGMENTATION = augmentation_string,
                QUESTION = question,
            )


        # self.eval_logger.info(f"----------PROMPT: \n{prompt}") # DELETE OR COMMENT OUT LATER

        output_text = ""
        if ('gemini' in model_name) or ('gpt' in model_name):
            # translate text-to-sql using Google models
            try:
                llm_service = LLMService(model_name=model_name, logger=self.eval_logger)
                response_object,  prompt_token_cnt, completion_token_cnt, total_token_cnt = llm_service.call_llm(prompt=prompt, temperature=temperature, top_p=top_p)
                output_text = response_object.text
                
            except Exception as e:
                output_text = ""
                occured_error = f"Error is taken during generation: {e}\n{traceback.format_exc()}"
                self.eval_logger.error(f"Error is taken during generation: {e}\n{traceback.format_exc()}")

        else:

            try:
                # Tokenize and move to GPU
                model_device = next(model.parameters()).device
                inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=max_seq_length).to(model_device)
                prompt_token_count = inputs["input_ids"].shape[1]
                print(f"Prompt token count: {prompt_token_count}")

                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        do_sample=False,
                        pad_token_id=tokenizer.eos_token_id
                    )

                output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

                self.eval_logger.info(f"----------RESPONSE Question:{q_id} (in ex_{ex_id}) (temp:{temperature} - top_p:{top_p}):\n {output_text}")
                # if q_id < 2:
                #     # self.eval_logger.info(f"----------PROMPT: \n Prompt to question_id: {q_id}:\n {prompt}")
                #     self.eval_logger.info(f"----------RESPONSE: \n Response to question_id: {q_id}:\n {output_text}")

            except Exception as e:
                output_text = ""
                occured_error = f"Error is taken during generation: {e}\n{traceback.format_exc()}"
                self.eval_logger.error(f"Error is taken during generation: {e}\n{traceback.format_exc()}")

        try:
            # output_dict = parse_llm_output(output_text, model_name=model.name_or_path, output_format=output_format) # old
            response_text = extract_response_part(output_text)
            output_dict = {
                "reasoning": extract_xml_reasoning(response_text),
                "answer": extract_sql_part(extract_xml_answer(response_text))
            }

            predicted_sql = output_dict['answer']
            reasoning = output_dict['reasoning']
        except Exception as e:
            predicted_sql = ""
            reasoning = ""
            error_traceback = traceback.format_exc()
            occured_error = f"Error is taken during parsing: {e}\n{error_traceback}"
            self.eval_logger.error(f"Error is taken during parsing: {e}\n{error_traceback}")
        
        ## Evaluating the correctness of predicted SQL query
        # Calculate Execution Accuracy for the predicted SQL
        try: 
            db_path = self.args.dbs_root_dir / db_id / f"{db_id}.sqlite"
            comparison_dict = compare_sqls(db_path=db_path, predicted_sql=predicted_sql, ground_truth_sql=gt_sql)
            exec_res = comparison_dict['exec_res']
            exec_err = comparison_dict['exec_err']
            # Calculate Soft-F1 score for the predicted SQL
            soft_f1_score = calculate_f1_score_for_sql(predicted_sql, gt_sql, db_path)
            f1_score = soft_f1_score if soft_f1_score else 0
        except Exception as e:
            self.eval_logger.error(f"Error is taken while evaluating predicted SQL query. Error: {e}")
            exec_res = 0
            exec_err = str(e) if e else "There is an error wile calculating accuracy of predicted SQL"
            f1_score = 0
        
        self.eval_logger.info(f"===============Question:{q_id} (in ex_{ex_id}) (temp:{temperature} - top_p:{top_p})=============== \n----- PREDICTED_SQL: {predicted_sql} \n----- GT_SQL: {gt_sql} \n ----- exec_res: {exec_res} \n ----- f1_score: {f1_score} \n -----exec_err: {exec_err}")

        translation_output = {
            "ex_id": ex_id,
            "predicted_sql": predicted_sql,
            "reasoning": reasoning,
            "exec_res": exec_res,
            "exec_err": exec_err,
            "f1_score": f1_score,
            "occured_error": occured_error

        }
        return translation_output
        
    
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
        

    def evaluate(self):
        """"
        Evaluates the model performance
        """
        # Write configs
        used_config = self.args.config
        write_config_file_path = self.eval_results_dir / "config.json"
        with open(write_config_file_path, "w") as file:
            json.dump(used_config, file, indent=4)

        t2s_dicts_path = self.eval_results_dir / f't2s_items.json'

        eval_start_time = time.time()
        # Eval Configs
        eval_configs = self.args.config['evaluation']
        temperature_list = eval_configs['temperature']
        top_p_list = eval_configs['top_p']
        assert len(temperature_list) == len(top_p_list)

        generation_count = len(temperature_list)
        t2s_dicts_with_translations = []
        for t2s_dict_idx, t2s_dict in enumerate(self.eval_dataset):
            print(f"+++++++++++++++++++++++++++++++++++")
            print(f"++++++++++{t2s_dict_idx}++++++++++++")
            print(f"+++++++++++++++++++++++++++++++++++")
            # check if the current dataset item is one of the considered database
            db_id = t2s_dict['db_id']
            if db_id not in self.db_ids:
                continue
            
            
            t2s_dict["translations"] = {}
            # max_workers = min(len(generation_count), 4 * (os.cpu_count() or 4))
            # max_workers = generation_count
            max_workers = 1
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    executor.submit(
                        self.translate_text_to_sql,
                        t2s_dict,
                        self.model,
                        self.tokenizer,
                        self.model_name,
                        idx
                    )
                    for idx in range(generation_count)
                ]

                for future in futures:
                    try:
                        translation_output = future.result()  # Will block until that specific future finishes
                        t2s_dict["translations"][str(translation_output.get("ex_id"))] = translation_output
                    except Exception as e:
                        self.eval_logger.error(f"Error while translating question into SQL in parallel.\n{traceback.format_exc()}")
            
            # Add t2s_dict with translations
            
            t2s_dicts_with_translations.append({
                "question_id": t2s_dict.get("question_id", ""),
                "db_id": t2s_dict.get("db_id", ""),
                "question": t2s_dict.get("question", ""),
                "evidence": t2s_dict.get("evidence", ""),
                "SQL": t2s_dict.get("SQL", ""),
                "difficulty": t2s_dict.get("difficulty", ""),
                "translations": t2s_dict.get("translations", {}),

            })
            with open(t2s_dicts_path, 'w') as file:
                json.dump(t2s_dicts_with_translations, file, indent=4)



        ## Computing bounds
        bounds = self.compute_bounds(t2s_dicts_with_translations)
        ## Computing F1 bounds
        best_translation_ids = self.find_best_translation_ids(t2s_dicts_with_translations)

        overall_eval_info = {
            "bounds": bounds,
            "best_translation_ids": best_translation_ids
        }   

        overall_eval_info_path = self.eval_results_dir / "overall_eval_info.json"
        with open(overall_eval_info_path, 'w') as file:
            json.dump(overall_eval_info, file, indent=4)


        eval_end_time = time.time()
        eval_duration = (eval_end_time - eval_start_time) / 60
        self.eval_logger.info(f"-- Overall Evaluation Duration: {eval_duration} minutes")



            

            
          
            



