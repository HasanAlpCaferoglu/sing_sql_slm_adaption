import os
import subprocess
import copy
import torch

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

class LocalEvalRunner:
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
            print("loading dev data with few shots:")
            with open(data_with_few_shots_path, 'r') as file:
                dataset = json.load(file)
        else:
            print("loading raw dev data (without few shots):")
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
        else:
            raise ValueError("Cannot run open-source model in local machine (MacOS)")

    
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

        ## Evaluation configurations
        eval_configs = self.args.config['evaluation']      
        prompt_template_level = eval_configs['prompt_template_level']  
        eval_base_model = bool(eval_configs['eval_base_model'])
        use_reasoning = bool(eval_configs['use_reasoning'])
        output_format = eval_configs['output_format']
        use_schema = bool(eval_configs['use_schema'])
        use_gt_schema = bool(eval_configs['use_gt_schema'])
        use_filtered_schema = bool(eval_configs['use_filtered_schema'])
        use_column_profile = bool(eval_configs['use_column_profile'])
        use_few_shot = bool(eval_configs['use_few_shot'])
        give_reasoning_for_few_shots = bool(eval_configs['give_reasoning_for_few_shots'])
        max_new_tokens = eval_configs['max_new_tokens']
        temperature = eval_configs['temperature'][ex_id]
        top_p = eval_configs['top_p'][ex_id]
        # self.eval_logger.info(f"temperature: {temperature} - top_p: {top_p}")
        # self.eval_logger.info(f"===============Question:{q_id} (in ex_{ex_id}) (temp:{temperature} - top_p:{top_p})===============")
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
            # Get similar synthetic examples using vector db
            # similar_synthetic_examples = self.vdb_service.search_examples(question_and_hint=question, db_id=db_id, k=3)   # This slows down the process of evaluation, so i have prepared few-shots
            for examples in t2s_dict.get('few_shot', {}).get('examples', []):
                similar_synthetic_examples.append(examples)
        
        few_shot_str = ""
        if use_few_shot and similar_synthetic_examples:
            few_shot_str += "\n *** Text-to-SQL Examples that contain similar keywords ***\n"
            for example_idx, example_dict in enumerate(similar_synthetic_examples):
                synthetic_question = example_dict.get("question")
                synthetic_sql = example_dict.get("SQL")
                example_reasoning = example_dict.get("dac_reasoning")
                few_shot_str += f"Example {example_idx+1}:\n"
                few_shot_str += f"Question: {synthetic_question}\n"
                if give_reasoning_for_few_shots:
                    few_shot_str += f"Reasoning: {example_reasoning}\n"
                few_shot_str += f"SQL: {synthetic_sql}\n\n"
                ## IDEA: Can giving search keyword for each example increase the EX, as it might add where to focus on the examples? Or we may direct LLM to focus on that part.

        used_schema_dict: Dict[str, List[str]] = {}
        schema_str = ""
        if use_schema:
            schema_str += "\n*** Database Schema Information ***\n"
            db_info = DatabaseGeneralInfo(db_id=db_id, dbs_root_dir=self.args.dbs_root_dir)
            if use_filtered_schema:
                ##### USING ONLY EXAMPLES FOR SCHEMA FILTERING
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

        # load template
        if prompt_template_level == "L1":
            prompt_template = load_template(template_name='train_t2s_schemaless')
        else:
            raise ValueError(f"NOT IMPLEMENTED YET.")

        if not use_reasoning:
            prompt_template = prompt_template.split("### Respond in the following format:")[0] # removing <reasoning></reasoning> part
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
        self.eval_logger.info(f"="*100)
        self.eval_logger.info(f"----------PROMPT: \n{prompt}")

        output_text = ""
        if ('gemini' in model_name) or ('gpt' in model_name):
            # translate text-to-sql using Google models
            try:
                llm_service = LLMService(model_name=model_name, logger=self.eval_logger)
                response_object,  prompt_token_cnt, completion_token_cnt, total_token_cnt = llm_service.call_llm(prompt=prompt, temperature=temperature, top_p=top_p)
                output_text = response_object.text
                print(f"output_text: {output_text}")
                print(f"prompt_token_cnt: {prompt_token_cnt} | completion_token_cnt: {completion_token_cnt},")
                
            except Exception as e:
                output_text = ""
                occured_error = f"Error is taken during generation: {e}"
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

                if q_id < 2:
                    # self.eval_logger.info(f"----------PROMPT: \n Prompt to question_id: {q_id}:\n {prompt}")
                    self.eval_logger.info(f"="*100)
                    self.eval_logger.info(f"----------RESPONSE: \n Response to question_id: {q_id}:\n {output_text}")

            except Exception as e:
                output_text = ""
                occured_error = f"Error is taken during generation: {e}"
                self.eval_logger.error(f"Error is taken during generation: {e}")

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
            occured_error = f"Rrror is taken during parsing: {e}"
            self.eval_logger.error(f"Error is taken during parsing: {e}")
        
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
        
        self.eval_logger.info(f"===============Question:{q_id} (in ex_{ex_id}) (temp:{temperature} - top_p:{top_p})=============== \n----- PREDICTED_SQL: {predicted_sql} \n ----- exec_res: {exec_res} \n ----- f1_score: {f1_score} \n -----exec_err: {exec_err}")

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
        for t2s_dict in self.eval_dataset:
            # check if the current dataset item is one of the considered database
            db_id = t2s_dict['db_id']
            if db_id not in self.db_ids:
                continue
            
            
            t2s_dict["translations"] = {}
            # max_workers = min(generation_count, 4 * (os.cpu_count() or 4))
            max_workers=1
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
                        self.eval_logger.error(f"Error while translation questions into sql parallel.")
            
            # Add t2s_dict with translations
            t2s_dicts_with_translations.append({
                "question_id": t2s_dict.get("question_id"),
                "db_id": t2s_dict.get("db_id"),
                "question": t2s_dict.get("question"),
                "evidence": t2s_dict.get("evidence"),
                "SQL": t2s_dict.get("SQL"),
                "difficulty": t2s_dict.get("difficulty"),
                "translations": t2s_dict.get("translations", {}),
            })
            with open(t2s_dicts_path, 'w') as file:
                json.dump(t2s_dicts_with_translations, file, indent=4)



        ## Computing bounds
        bounds = self.compute_bounds(t2s_dicts_with_translations)
        print(f"Bounds: \n {json.dumps(bounds, indent=4)}")
        ## Computing F1 bounds
        best_translation_ids = self.find_best_translation_ids(t2s_dicts_with_translations)
        print(f"best_translation_ids: \n {json.dumps(best_translation_ids, indent=4)}")

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


            

            
          
            



