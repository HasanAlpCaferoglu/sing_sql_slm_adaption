import os
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



class EvalRunner:
    def __init__(self, args: Any):
        self.args = args
        self.all_db_ids: List[str] = self._set_db_ids()
        self.db_ids: List[str] = self.args.config.get("db_ids", [])
        self.dataset = self._load_dataset()
        self.eval_dataset = self._get_eval_dataset()

        # Set logger
        logger = logging.getLogger('eval')
        logger.setLevel(logging.INFO)
        logger_path = Path(f"logs/eval_{self.args.run_start_time}/eval_logs.log")
        logger_path.parent.mkdir(parents=True, exist_ok=True)
        logger_handler = logging.FileHandler(logger_path)
        logger_handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
        logger.addHandler(logger_handler)
        self.eval_logger = logger
        
        # Load model and tokenizer
        self.model, self.tokenizer, self.model_name = self.load_language_model_and_tokenizer()

        # Initialize vector database service
        self.vdb_service = self._construct_vdb_service()

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
        data_path = Path(self.args.data_json_path)
        data_with_few_shots_file_name = data_path.stem + "_with_few_shots" + data_path.suffix 
        data_with_few_shots_path = data_path.parent / data_with_few_shots_file_name

        if data_with_few_shots_path.exists():
            with open(data_with_few_shots_path, 'r') as file:
                dataset = json.load(file)
        else:
            with open(data_path, 'r') as file:
                dataset = json.load(file)

        return dataset
    
    def _get_eval_dataset(self) -> List[Dict[str, Any]]:
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
        eval_results_dir = Path(f"./results/{db_ids_str}/{pure_model_name}/{self.args.run_start_time}")
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
        use_cvd = train_configs["use_col_value_and_descriptions"]
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
        use_unsloth_flash_attention_2 = bool(eval_configs['use_unsloth_flash_attention_2'])

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
            r = lora_params.get("lora_r", "")
            alpha = lora_params.get("lora_alpha", "")

            # Construct model repo name
            epoch = int(training_params['num_train_epochs'])
            bs = int(training_params['per_device_train_batch_size'])
            gas=int(training_params["gradient_accumulation_steps"])
            learningrate=float(training_params["learning_rate"])
            if (not use_lora):
                model_name = f"{base_model_id_without_user}_{t2s_dataset_name[0]}{data_mode[0]}_{db_id_str}_{task_str}_e{epoch}_bs{bs}_gas{gas}_lr{learningrate}"
            else:
                model_name = f"{base_model_id_without_user}_{t2s_dataset_name[0]}{data_mode[0]}_{db_id_str}_{task_str}_r{r}_a{alpha}_e{epoch}_bs{bs}_gas{gas}_lr{learningrate}"

            if use_reasoning:
                model_name = f"{model_name}_sftreason" # Reasoning with SFT
            elif use_grpo:
                model_name = f"{model_name}_grpo"
            
            if use_cvd:
                model_name = f"{model_name}_cvd"

            
            model_name = f"{os.getenv('HF_USER')}/{model_name}"
            self.eval_logger.info(f"=== Loading Model: {model_name} ")
            print(f"=== Loading Model: {model_name} ")

        if use_unsloth:
            try:
                model, tokenizer = FastLanguageModel.from_pretrained(
                    model_name = model_name,
                    load_in_4bit = train_configs.get('load_in_4bit', False),
                    # attn_implementation="flash_attention_2"
                    # max_seq_length = max_seq_length, # 'Qwen2Model' object has no attribute 'max_seq_length
                    # use_flash_attention_2 = use_unsloth_flash_attention_2 # Error: Both attn_implementation="eager" and `use_flash_attention_2=True` were used when loading the model, which are not compatible. We recommend to just use `attn_implementation="flash_attention_2"` when loading the model.
                )
                FastLanguageModel.for_inference(model)
                model.eval()
            except Exception as e1:
                self.eval_logger.error(f"Unsloth couldn't find or load the model with the name of {model_name}. Error: {e1}")
                print(f"Unsloth couldn't find or load the model with the name of {model_name}. Error: {e1}")
                self.eval_logger.info(f"Trying AutoModelForCausalLM from transformers library")
                print(f"Trying AutoModelForCausalLM from transformers library")
                use_unsloth = False
        
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
        self.eval_logger.info(f'Model name: {model_name}.')
        print(f'Model name: {model_name}.')
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

    def translate_text_to_sql(self, t2s_dict: Dict[str, Any], model=None, tokenizer=None, model_name: str = None, eval_id: int = 0) -> Dict[str, Any]:
        """
        Generating SQL query for a user question

        Arguments:
            t2s_dict (Dict[str, Any]): A dataset item 
        
        Returns:
            Dict[str, Any]: A new dictionary containing predicted SQL query with other default information 
        """        
        q_id = t2s_dict['question_id']
        gt_sql = t2s_dict['SQL']
        db_id = t2s_dict['db_id']
        question = t2s_dict['question']
        evidence = t2s_dict.get('evidence', '')
        question = question + " Hint: " + evidence if evidence else question

        new_t2s_dict = copy.deepcopy(t2s_dict)
        new_t2s_dict['predicted_sql'] = ""
        predicted_sql = ""


        ## Evaluation configurations
        eval_configs = self.args.config['evaluation']        
        eval_base_model = bool(eval_configs['eval_base_model'])
        use_reasoning = bool(eval_configs['use_reasoning'])
        output_format = eval_configs['output_format']
        use_schema = bool(eval_configs['use_schema'])
        use_gt_schema = bool(eval_configs['use_gt_schema'])
        use_filtered_schema = bool(eval_configs['use_filtered_schema'])
        use_column_profile = bool(eval_configs['use_column_profile'])
        use_few_shot = bool(eval_configs['use_few_shot'])
        max_new_tokens = eval_configs['max_new_tokens']
        temperature = eval_configs['temperature'][eval_id]
        top_p = eval_configs['top_p'][eval_id]
        # self.eval_logger.info(f"temperature: {temperature} - top_p: {top_p}")
        self.eval_logger.info(f"===============Question:{q_id} (in eval_{eval_id}) (temp:{temperature} - top_p:{top_p})===============")
        self.eval_logger.info(f"Question and Hint: {question}")

        if eval_base_model:
            # Since base model doesn't trained for learning database schema, it should be given in prompt
            use_schema = True
            eval_configs['use_schema'] = True
        else:
            if "reason" in model_name:
                # if model name includes sftr, then it is specifically designed to make reasoning 
                use_reasoning = True
                eval_configs['use_reasoning'] = True
            else:
                use_reasoning = False
                eval_configs['use_reasoning'] = False
        
        if use_gt_schema or use_filtered_schema:
            use_schema = True
            eval_configs['use_schema'] = True

        if 'cvd' in model_name:
            use_cvd = True
            eval_configs['use_col_value_and_descriptions'] = True
        else:
            use_cvd = bool(eval_configs['use_col_value_and_descriptions'] )
        # print(f"use cvd: {use_cvd}")
        
        similar_synthetic_examples = []
        if use_few_shot or use_filtered_schema:
            # Get similar synthetic examples
            # similar_synthetic_examples = self.vdb_service.search_examples(question_and_hint=question, db_id=db_id, k=3) # This slows down the process of evaluation, so i have prepared few-shots
            for examples in t2s_dict.get('few_shot', {}).get('examples', []):
                similar_synthetic_examples.append(examples)
        
        few_shot_str = ""
        if use_few_shot and similar_synthetic_examples:
            few_shot_str += "\n *** Text-to-SQL Examples that contain similar keywords ***\n"
            for example_idx, example_dict in enumerate(similar_synthetic_examples):
                synthetic_question = example_dict.get("question")
                synthetic_sql = example_dict.get("SQL")
                few_shot_str += f"Example {example_idx+1}:\n"
                few_shot_str += f"Question: {synthetic_question}\n"
                few_shot_str += f"SQL: {synthetic_sql}\n\n"
                ## IDEA: Can giving search keyword for each example increase the EX, as it might add where to focus on the examples? Or we may direct LLM to focus on that part.

        used_schema_dict: Dict[str, List[str]] = {}
        schema_str = ""
        if use_schema:
            schema_str += "\n*** Database Schema Information ***\n"
            db_info = DatabaseGeneralInfo(db_id=db_id, dbs_root_dir=self.args.dbs_root_dir)
            if use_filtered_schema:
                # ##### USING ONLY EXAMPLES FOR SCHEMA FILTERING
                # filtered_schema_dict = T2SSyntheticVDBService.get_filtered_schema_dict_from_similar_examples(
                #     db_path=db_info.db_path, 
                #     similar_examples=similar_synthetic_examples
                # )
                ##### USING PRE-COMPUTED SCHEMA FILTERING INCLUDES LEVERAGES EXAMPLES TAKEN FROM VDB
                filtered_schema_dict = t2s_dict.get('filtered_schema', {}).get('schema_dict', {})
                if not filtered_schema_dict:
                    self.eval_logger.info(f"Couldn't find filtered schema dictionary. Continuing with the full schema...")
                    filtered_schema_dict: Dict[str, List[str]] = get_db_schema(db_info.db_path)
                
                used_schema_dict = filtered_schema_dict
                filtered_schema_structure = DatabaseSchema.from_schema_dict(filtered_schema_dict)
                filtered_schema_generator = DatabaseSchemaGenerator(
                    tentative_schema=filtered_schema_structure,
                    db_id=db_id,
                    db_path=db_info.db_path,
                    add_examples=use_cvd,
                    add_random_examples=use_cvd
                )
                db_schema_string = filtered_schema_generator.generate_schema_string(
                    include_column_value_examples=use_cvd,
                    include_value_description=use_cvd
                )
                # Adding column profiles into the schema part
                if use_column_profile:
                    column_profile_string = filtered_schema_generator.get_column_profiles_string( with_keys=False, with_references=False)
                    db_schema_string +=  "\n### Detailed Column Information: \n"
                    db_schema_string += column_profile_string

            elif use_gt_schema: # Use GT schema 
                gt_sql = t2s_dict['SQL']
                gt_schema_dict: Dict[str, List[str]] = get_sql_columns_dict(db_path=db_info.db_path, sql=gt_sql)
                used_schema_dict = gt_schema_dict
                gt_schema_structure = DatabaseSchema.from_schema_dict(gt_schema_dict)
                gt_schema_generator = DatabaseSchemaGenerator(
                    tentative_schema=gt_schema_structure,
                    db_id = db_id,
                    db_path=db_info.db_path,
                    add_examples=use_cvd,
                    add_random_examples=use_cvd
                )
                db_schema_string = gt_schema_generator.generate_schema_string(
                    include_column_value_examples=use_cvd,
                    include_value_description=use_cvd
                )
                # Adding column profiles into the schema part
                if use_column_profile:
                    column_profile_string = gt_schema_generator.get_column_profiles_string( with_keys=False, with_references=False)
                    db_schema_string +=  "\n ### Detailed Column Information: \n"
                    db_schema_string += column_profile_string

            else: # Use whole schema 
                full_db_schema_dict: Dict[str, List[str]] = get_db_schema(db_info.db_path)
                used_schema_dict = full_db_schema_dict
                full_schema_structure = DatabaseSchema.from_schema_dict(full_db_schema_dict)
                full_schema_generator = DatabaseSchemaGenerator(
                    tentative_schema = full_schema_structure,
                    db_id = db_id,
                    db_path= db_info.db_path,
                    add_examples=use_cvd, 
                    add_random_examples=use_cvd 
                )
                db_schema_string = full_schema_generator.generate_schema_string(
                    include_column_value_examples=use_cvd,
                    include_value_description=use_cvd  
                    )
                # Adding column profiles into the schema part
                if use_column_profile:
                    column_profile_string = full_schema_generator.get_column_profiles_string( with_keys=False, with_references=False)
                    db_schema_string +=  "\n ### Detailed Column Information: \n"
                    db_schema_string += column_profile_string
                

            schema_str += db_schema_string
            schema_str += "\n\n"
            # self.eval_logger.info(f"========== SCHEMA STRING =========\n {schema_str}")
        else:
            use_gt_schema = False
            eval_configs['use_gt_schema'] = False

        # compute the schema metrics
        if use_schema:
            gt_schema_dict: Dict[str, List[str]] = get_sql_columns_dict(db_path=db_info.db_path, sql=gt_sql)
            num_tp, num_fp, num_fn, s_recall, s_precision, s_f1 = calculate_schema_metrics_for_single_schema(
                used_schema_dict=used_schema_dict, 
                gt_schema_dict=gt_schema_dict
            )
            new_t2s_dict["prompt_schema_metrics"] = {
                "tp": num_tp,
                "fp": num_fp,
                "fn": num_fn,
                "recall": s_recall,
                "precision": s_precision,
                "f1": s_f1
            }
        else:
            new_t2s_dict["prompt_schema_metrics"] = {
                "tp": 0,
                "fp": 0,
                "fn": 0,
                "recall": 0.0,
                "precision": 0.0,
                "f1": 0.0
            }


        # load template
        prompt_template = load_template(template_name='train_t2s_schemaless')

        if not use_reasoning:
            prompt_template = prompt_template.split("### Respond in the following format:")[0] # removing <think></think> part
            prompt_template += "\nWhen you get to the final query, output the query string ONLY inside the xml delimiter <answer></answer>."

        ## Prompt Template formatting
        prompt_t = prompt_template.split("*** Question ***")[0]
        prompt_t += schema_str # Add schema if used
        prompt_t += few_shot_str # Add few-shot examples if used
        prompt_t += "\n*** Question ***\n"
        prompt_t += prompt_template.split("*** Question ***")[1]
        prompt_template = prompt_t
        
        prompt = prompt_template.format(
            DB_ID = db_id,
            QUESTION = question
        )
        # self.eval_logger.info(f"----------PROMPT: \n{prompt}")

        output_text = ""
        if ('gemini' in model_name) or ('gpt' in model_name):
            # translate text-to-sql using Google models
            try:
                llm_service = LLMService(model_name=model_name, logger=self.eval_logger)
                response_object,  prompt_token_cnt, completion_token_cnt, total_token_cnt = llm_service.call_llm(prompt=prompt, temperature=temperature, top_p=top_p)
                output_text = response_object.text
                new_t2s_dict['model_output'] = output_text
                
            except Exception as e:
                predicted_sql = ""
                new_t2s_dict['predicted_sql'] = ""
                new_t2s_dict['llm_error'] = f"{e}"
                self.eval_logger.error(f"Error is taken during generation: {e}")

        else:

            try:
                # Tokenize and move to GPU
                model_device = next(model.parameters()).device
                inputs = tokenizer(prompt, return_tensors='pt').to(model_device)
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
                new_t2s_dict['model_output'] = output_text

                if q_id < 2:
                    # self.eval_logger.info(f"----------PROMPT: \n Prompt to question_id: {q_id}:\n {prompt}")
                    self.eval_logger.info(f"----------RESPONSE: \n Response to question_id: {q_id}:\n {output_text}")

            except Exception as e:
                predicted_sql = ""
                new_t2s_dict['predicted_sql'] = ""
                new_t2s_dict['llm_error'] = f"{e}"
                self.eval_logger.error(f"Error is taken during generation: {e}")

        try:
            # output_dict = parse_llm_output(output_text, model_name=model.name_or_path, output_format=output_format) # old
            response_text = extract_response_part(output_text)
            output_dict = {
                "reasoning": extract_xml_reasoning(response_text),
                "answer": extract_sql_part(extract_xml_answer(response_text))
            }

            predicted_sql = output_dict['answer']
            new_t2s_dict['predicted_sql'] = predicted_sql
        except Exception as e:
            predicted_sql = ""
            new_t2s_dict['predicted_sql'] = ""
            new_t2s_dict['parsing_error'] = f"{e}"
            self.eval_logger.error(f"Error is taken during parsing: {e}")
        
        ## Evaluating the correctness of predicted SQL query
        # Calculate Execution Accuracy for the predicted SQL
        try: 
            db_path = self.args.dbs_root_dir / db_id / f"{db_id}.sqlite"
            comparison_dict = compare_sqls(db_path=db_path, predicted_sql=predicted_sql, ground_truth_sql=gt_sql)
            new_t2s_dict['exec_res'] = comparison_dict['exec_res']
            new_t2s_dict['exec_err'] = comparison_dict['exec_err']
            # Calculate Soft-F1 score for the predicted SQL
            soft_f1_score = calculate_f1_score_for_sql(predicted_sql, gt_sql, db_path)
            new_t2s_dict['f1_score'] = soft_f1_score if soft_f1_score else 0
        except Exception as e:
            self.eval_logger.error(f"Error is taken while evaluating predicted SQL query. Error: {e}")
            new_t2s_dict['exec_res'] = 0
            new_t2s_dict['exec_err'] = str(e) if e else "There is an error wile calculating accuracy of predicted SQL"
            new_t2s_dict['f1_score'] = 0
        
        self.eval_logger.info(f"----- PREDICTED_SQL: {predicted_sql}")
        self.eval_logger.info(f"----- exec_res: {new_t2s_dict['exec_res']}")
        self.eval_logger.info(f"----- f1_score: {new_t2s_dict['f1_score']}")
        self.eval_logger.info(f"----- prompt_schema_recall: {new_t2s_dict['prompt_schema_metrics']['recall']}")
        self.eval_logger.info(f"----- prompt_schema_precision: {new_t2s_dict['prompt_schema_metrics']['precision']}")
        return new_t2s_dict
        
    
    def evaluate_one_time(self, eval_id: int = 0) -> Dict:
        """
        Evaluating trained model on a given databases
        """
        # Dataset
        dataset = self.args.dataset.split('-')[0]

        t2s_dicts_path = self.eval_results_dir / f't2s_items_{eval_id}.json'
        predictions_path = self.eval_results_dir / f'predict_{self.args.data_mode}_{eval_id}.json'
        eval_metrics_path = self.eval_results_dir / f'eval_metrics_{eval_id}.json'


        ## read predict_data_mode.json file if exist
        if predictions_path.exists():
            with open(predictions_path, 'r') as f2:
                predicted_sql_queries = json.load(f2)
        else:
            predicted_sql_queries = []

        new_t2s_dicts = []
        simple_ex_values = []
        simple_f1_values = []
        moderate_ex_values = []
        moderate_f1_values = []
        challenging_ex_values = []
        challenging_f1_values = []
        all_ex_values = []
        all_f1_values = []
        prompt_schema_total_tp = 0
        prompt_schema_total_fp = 0
        prompt_schema_total_fn = 0

        # OLD FOR LOOP
        for t2s_dict in self.eval_dataset:
            # check if the current dataset item is one of the considered database
            db_id = t2s_dict['db_id']
            if db_id not in self.db_ids:
                continue

            try:
                new_t2s_dict = self.translate_text_to_sql(t2s_dict, model=self.model, tokenizer=self.tokenizer, model_name=self.model_name, eval_id=eval_id)
                predicted_sql = new_t2s_dict['predicted_sql']
                self.eval_logger.info(f"Predicted SQL: {predicted_sql}")
                new_t2s_dicts.append(new_t2s_dict)

                predicted_sql_str = predicted_sql + f"\t----- {dataset} -----\t{db_id}"
                predicted_sql_queries.append(predicted_sql_str)

                # save new_t2s_dicts into file
                with open(t2s_dicts_path, "w") as file:
                    json.dump(new_t2s_dicts, file, indent=4)

                # save predicted_sql_queries into file
                with open(predictions_path, "w") as file:
                    json.dump(predicted_sql_queries, file, indent=4)

                prompt_schema_metrics_dict = new_t2s_dict['prompt_schema_metrics']
                prompt_schema_total_tp += prompt_schema_metrics_dict['tp']
                prompt_schema_total_fp += prompt_schema_metrics_dict['fp']
                prompt_schema_total_fn += prompt_schema_metrics_dict['fn']
                
                all_ex_values.append(new_t2s_dict['exec_res'])
                all_f1_values.append(new_t2s_dict['f1_score'])
                if new_t2s_dict['difficulty'].strip().lower()=="simple":
                    simple_ex_values.append(new_t2s_dict['exec_res'])
                    simple_f1_values.append(new_t2s_dict['f1_score'])
                elif new_t2s_dict['difficulty'].strip().lower()=="moderate":
                    moderate_ex_values.append(new_t2s_dict['exec_res'])
                    moderate_f1_values.append(new_t2s_dict['f1_score'])
                elif new_t2s_dict['difficulty'].strip().lower()=="challenging":
                    challenging_ex_values.append(new_t2s_dict['exec_res'])
                    challenging_f1_values.append(new_t2s_dict['f1_score'])
                else:
                    self.eval_logger.error(f"Couldn't find the difficulty level for item: \n {new_t2s_dict}")

            except Exception as e:
                self.eval_logger.error(f"Unexpected error: {e}")
                self.eval_logger.error("Unexpected error:\n" + traceback.format_exc())
        
        #### CONCURRENCY #####
        # max_workers = 4
        # with ThreadPoolExecutor(max_workers=max_workers) as executor:
        #     futures = [
        #         executor.submit(
        #             self.translate_text_to_sql,
        #             t2s_dict,
        #             self.model,
        #             self.tokenizer,
        #             self.model_name,
        #             eval_id
        #         )
        #         for t2s_dict in self.eval_dataset
        #     ]

        #     # Get results in correct order
        #     for future in as_completed(futures):
        #         new_t2s_dict = future.result() # Will block until that specific future finishes
        #         new_t2s_dicts.append(new_t2s_dict)
        #         db_id = new_t2s_dicts['db_id']

        #         predicted_sql = new_t2s_dict['predicted_sql']
        #         predicted_sql_str = predicted_sql + f"\t----- {dataset} -----\t{db_id}"
        #         predicted_sql_queries.append(predicted_sql_str)
                
        #         all_ex_values.append(new_t2s_dict['exec_res'])
        #         all_f1_values.append(new_t2s_dict['f1_score'])
        #         if new_t2s_dict['difficulty'].strip().lower()=="simple":
        #             simple_ex_values.append(new_t2s_dict['exec_res'])
        #             simple_f1_values.append(new_t2s_dict['f1_score'])
        #         elif new_t2s_dict['difficulty'].strip().lower()=="moderate":
        #             moderate_ex_values.append(new_t2s_dict['exec_res'])
        #             moderate_f1_values.append(new_t2s_dict['f1_score'])
        #         elif new_t2s_dict['difficulty'].strip().lower()=="challenging":
        #             challenging_ex_values.append(new_t2s_dict['exec_res'])
        #             challenging_f1_values.append(new_t2s_dict['f1_score'])
        #         else:
        #             self.eval_logger.error(f"Couldn't find the difficulty level for item: \n {new_t2s_dict}")

        # save new_t2s_dicts into file
        with open(t2s_dicts_path, "w") as file:
            json.dump(new_t2s_dicts, file, indent=4)

        # save predicted_sql_queries into file
        with open(predictions_path, "w") as file:
            json.dump(predicted_sql_queries, file, indent=4)

        # ps == prompt schema
        eval_ps_precision = prompt_schema_total_tp / (prompt_schema_total_tp + prompt_schema_total_fp) if (prompt_schema_total_tp + prompt_schema_total_fp) > 0 else 0.0
        eval_ps_recall = prompt_schema_total_tp / (prompt_schema_total_tp + prompt_schema_total_fn) if (prompt_schema_total_tp + prompt_schema_total_fn) > 0 else 0.0
        eval_ps_f1 = 2 * eval_ps_precision * eval_ps_recall / (eval_ps_precision + eval_ps_recall) if (eval_ps_precision + eval_ps_recall) > 0 else 0.0

        simple_ex = sum(simple_ex_values) / len(simple_ex_values) * 100
        moderate_ex = sum(moderate_ex_values) / len(moderate_ex_values) * 100
        challenging_ex = sum(challenging_ex_values) / len(challenging_ex_values) * 100
        all_ex = sum([item['exec_res'] for item in new_t2s_dicts]) / len(new_t2s_dicts) * 100

        simple_f1 = sum(simple_f1_values) / len(simple_f1_values) * 100
        moderate_f1 = sum(moderate_f1_values) / len(moderate_f1_values) * 100
        challenging_f1 = sum(challenging_f1_values) / len(challenging_f1_values) * 100
        all_f1 = sum([item['f1_score'] for item in new_t2s_dicts]) / len(new_t2s_dicts) * 100
        eval_metrics = {
            "total_item_count": len(new_t2s_dicts),
            "simple_item_count": len(simple_ex_values),
            "moderate_item_count": len(moderate_ex_values),
            "challenging_item_count": len(challenging_ex_values),
            "correct_item_count_ex": sum([item['exec_res'] for item in new_t2s_dicts]),
            "EX": all_ex,
            "simple_ex": simple_ex,
            "moderate_ex": moderate_ex,
            "challenging_ex": challenging_ex,
            "F1": all_f1,
            "simple_f1": simple_f1,
            "moderate_f1": moderate_f1,
            "challenging_f1": challenging_f1,
            "prompt_schema_precision": eval_ps_precision,
            "prompt_schema_recall": eval_ps_recall,
            "prompt_schema_f1": eval_ps_f1  
        }
        self.eval_logger.info(f" Evaluation Metrics: {eval_metrics}")
        with open(eval_metrics_path, 'w') as file:
            json.dump(eval_metrics, file, indent=4)

        ## write config file
        config_path = self.eval_results_dir / 'config.json'
        with open(config_path, 'w') as file:
            json.dump(self.args.config, file, indent=4)


        return {
            "eval_metrics": eval_metrics,
            "all_ex_values": all_ex_values,
            "simple_ex_values": simple_ex_values,
            "moderate_ex_values": moderate_ex_values,
            "challenging_ex": challenging_ex,
            "all_f1_values": all_f1_values,
            "simple_f1_values":simple_f1_values,
            "moderate_f1": moderate_f1,
            "challenging_f1": challenging_f1,

        }

    def compute_ex_bounds(self, eval_outputs: List[Dict[str, Any]]) -> Tuple[float, float]:
        """
        Computes the upper and lower bound of EX values across all evaluations.
        """
        all_ex_lists = [output["all_ex_values"] for output in eval_outputs]
        list_length = len(all_ex_lists[0])

        upper_bound_ex_list = [
            max(ex_values[i] for ex_values in all_ex_lists)
            for i in range(list_length)
        ]
        lower_bound_ex_list = [
            min(ex_values[i] for ex_values in all_ex_lists)
            for i in range(list_length)
        ]

        ex_upper_bound = sum(upper_bound_ex_list) / list_length * 100
        ex_lower_bound = sum(lower_bound_ex_list) / list_length * 100

        return (ex_upper_bound, ex_lower_bound)


    def compute_f1_bounds(self, eval_outputs: List[Dict[str, Any]]) -> Tuple[float, float]:
        """
        Computes the upper and lower bound of F1 values across all evaluations.
        """
        all_f1_lists = [output["all_f1_values"] for output in eval_outputs]
        list_length = len(all_f1_lists[0])

        upper_bound_f1_list = [
            max(f1_values[i] for f1_values in all_f1_lists)
            for i in range(list_length)
        ]
        lower_bound_f1_list = [
            min(f1_values[i] for f1_values in all_f1_lists)
            for i in range(list_length)
        ]

        f1_upper_bound = sum(upper_bound_f1_list) / list_length * 100
        f1_lower_bound = sum(lower_bound_f1_list) / list_length * 100

        return (f1_upper_bound, f1_lower_bound)
    
    def find_best_eval_id_based_on_ex(self, eval_outputs: List[Dict[str, Any]]) -> int:

        best_eval_id = None
        best_ex = 0.0
        for id, eval_output in enumerate(eval_outputs):
            ex = eval_output['eval_metrics']['EX']
            if ex > best_ex:
                best_ex = ex
                best_eval_id = id

        return best_eval_id
    
    def find_best_eval_id_based_on_f1(self, eval_outputs: List[Dict[str, Any]]) -> int:

        best_eval_id = None
        best_f1_score = 0.0
        for id, eval_output in enumerate(eval_outputs):
            f1_score = eval_output['eval_metrics']['F1']
            if f1_score > best_f1_score:
                best_f1_score = f1_score
                best_eval_id = id

        return best_eval_id
        

    def evaluate(self):
        """"
        Evaluates the model performance
        """

        eval_start_time = time.time()
        # Eval Configs
        eval_configs = self.args.config['evaluation']
        temperature_list = eval_configs['temperature']
        top_p_list = eval_configs['top_p']
        assert len(temperature_list) == len(top_p_list)

        eval_count = len(temperature_list)
        eval_outputs = []
        for eval_id in range(eval_count):
            print(f"================= Eval:{eval_id} ================= ")
            self.eval_logger.info(f"================= Eval:{eval_id} ================= ")
            s_time = time.time()
            eval_output = self.evaluate_one_time(eval_id=eval_id)
            eval_outputs.append(eval_output)
            e_time = time.time()
            duration = (e_time - s_time) / 60
            self.eval_logger.info(f"-- Eval_{eval_id} Duration: {duration} minutes")


        # Compute upper and lower bounds
        ex_upper_bound, ex_lower_bound = self.compute_ex_bounds(eval_outputs)
        f1_upper_bound, f1_lower_bound = self.compute_f1_bounds(eval_outputs)

        # Find best eval metrics
        best_ex_eval_id = self.find_best_eval_id_based_on_ex(eval_outputs)
        best_ex_eval_metrics = eval_outputs[best_ex_eval_id]["eval_metrics"]
        best_f1_eval_id = self.find_best_eval_id_based_on_f1(eval_outputs)
        best_f1_eval_metrics = eval_outputs[best_f1_eval_id]["eval_metrics"]

        
        self.eval_logger.info(f"\n===\n")
        self.eval_logger.info(f"EX Upper Bound: {ex_upper_bound} | EX Lower Bound: {ex_lower_bound}")
        self.eval_logger.info(f"F1 Upper Bound: {f1_upper_bound} | F1 Lower Bound: {f1_lower_bound}")
        self.eval_logger.info(f"best_ex_eval_id: {best_ex_eval_id}")
        self.eval_logger.info(f"best_ex_eval_metrics: {best_ex_eval_metrics}")
        self.eval_logger.info(f"best_f1_eval_id: {best_f1_eval_id}")
        self.eval_logger.info(f"best_f1_eval_metrics: {best_f1_eval_metrics}")

        overall_eval_info = {
            "ex_upper_bound": ex_upper_bound,
            "ex_lower_bound": ex_lower_bound,
            "f1_upper_bound": f1_upper_bound,
            "f1_lower_bound": f1_lower_bound,
            "best_ex_eval_id": best_ex_eval_id,
            "best_f1_eval_id": best_f1_eval_id,
            "best_ex_eval_metrics": best_ex_eval_metrics,
            "best_f1_eval_metrics": best_f1_eval_metrics,
        }

        overall_eval_info_path = self.eval_results_dir / "overall_eval_info.json"
        with open(overall_eval_info_path, 'w') as file:
            json.dump(overall_eval_info, file, indent=4)

        eval_end_time = time.time()
        eval_duration = (eval_end_time - eval_start_time) / 60
        self.eval_logger.info(f"-- Overall Evaluation Duration: {eval_duration} minutes")


            

            
          
            



