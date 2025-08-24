import json
import logging
from pathlib import Path
import numpy
import torch
from torch.utils.data import IterableDataset, Dataset, DataLoader
from typing import Dict, Optional, Union, List, Literal, Any
from utils.llm_utils.prompt_utils import load_template, load_template_examples
from utils.db_utils.sql_parser import get_sql_columns_dict
from utils.db_utils.schema_generator import DatabaseSchemaGenerator
from utils.db_utils.db_info_utils import get_db_all_tables, get_db_schema
from utils.db_utils.db_info import DatabaseGeneralInfo
from utils.db_utils.schema import DatabaseSchema

# At the top of your dataset file
MAX_LENGTH = 0

def prepare_inputs_and_labels(prefix_seq: str, target_seq: str, tokenizer, max_tokens):
    """"
    Preparing input_ids, attention_mask and labels
    """
    train_logger = logging.getLogger("train_logger")
    # train_logger.info(f"prefix_seq: {prefix_seq}") # DELETE LATER
    # train_logger.info(f"target_seq: {target_seq}") # DELETE LATER

    bos_token_id = tokenizer.bos_token_id
    eos_token_id = tokenizer.eos_token_id
    pad_token_id = tokenizer.pad_token_id

    global MAX_LENGTH
    # prefix_ids = [tokenizer.bos_token_id] + tokenizer(prefix_seq, truncation=False)["input_ids"]
    prefix_ids = ([bos_token_id] if bos_token_id else []) + tokenizer(prefix_seq, truncation=False)["input_ids"]
    target_ids = tokenizer(target_seq, truncation=False)["input_ids"] + [tokenizer.eos_token_id]

    seq_len = len(prefix_ids) + len(target_ids)
    MAX_LENGTH = max(MAX_LENGTH, seq_len)  # Track max length
    if seq_len <= max_tokens: # Padding inputs with pad_token_id
        pad_length = max_tokens - seq_len
        input_ids = prefix_ids + target_ids + [tokenizer.pad_token_id] * pad_length
        attention_mask = [1] * seq_len + [0] * pad_length # Ignoring the padding tokens when performing (masked) self-attention
        labels = [-100] * len(prefix_ids) + target_ids + [-100] * pad_length
    else: # no padding
        print("the current input sequence exceeds the max_tokens, we will truncate it.")
        input_ids = prefix_ids + target_ids
        # input_ids = [tokenizer.bos_token_id]  + input_ids[-(max_tokens-1):] # pre-truncate input ids
        input_ids = ([bos_token_id] if bos_token_id else [])  + input_ids[-(max_tokens-1):] # pre-truncate input ids
        attention_mask = [1] * max_tokens
        
        labels = [-100] * len(prefix_ids) + target_ids # only target_ids produces gradients
        labels = labels[-max_tokens:] # pre-truncate labels

    # train_logger.info(f"input_ids: {input_ids}")  # DELETE LATER
    # train_logger.info(f"attention_mask: {attention_mask}")  # DELETE LATER
    # train_logger.info(f"labels: {labels}")  # DELETE LATER

    # print(f"============ MAX_LENGTH = {MAX_LENGTH} ============")
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.int64),
        "attention_mask": torch.tensor(attention_mask, dtype = torch.int64),
        "labels": torch.tensor(labels, dtype=torch.int64),
        "seq_len": seq_len
    }

class BirdTrainText2SQLWithSchema(Dataset):
    def __init__(self, dataset_root_path: Path,  use_grpo, use_unsloth, use_reasoning, use_cvd, tokenizer = None, max_tokens = 32768):
        # Attributes
        self.use_grpo = use_grpo
        self.use_unsloth = use_unsloth
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens
        self.use_reasoning = use_reasoning
        self.use_cvd = use_cvd

        # Path attributes
        self.dataset_root_path = dataset_root_path 
        self.dataset_train_mode_path = dataset_root_path / "bird-sql"
        self.data_json_path = self.dataset_train_mode_path / f"train.json"
        self.train_dbs_root_dir = self.dataset_train_mode_path / f"train_databases"
        self.column_meaning_path = self.dataset_train_mode_path / f"column_meaning.json"
        self.processed_column_meaning_path = self.dataset_train_mode_path / f"processed_column_meaning.json"

        self.dataset = self.load_dataset()

        prompt_template = load_template(template_name='train_t2s_schemaless')
        if not use_reasoning:
            pt = prompt_template.split('<think>')[0] + prompt_template.split('</think>')[1] 
            self.prompt_template = pt
        else:
            self.prompt_template = prompt_template

    def load_dataset(self):

        with open(self.data_json_path, 'r') as file:
            t2s_obj_train_dataset = json.load(file)
        
        return t2s_obj_train_dataset


    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        example = self.dataset[idx]
        db_id = example.get("db_id")
        question = example.get("question")
        evidence = example.get("evidence", "")
        sql = example.get("SQL")
        dac_reasoning = example.get("dac_reasoning", "") # reasoning when generating t2s pairs
        
        question = question + "Hint: " + evidence if evidence else question

        db_info = DatabaseGeneralInfo(db_id=db_id, dbs_root_dir=self.train_dbs_root_dir)
        db_schema_dict = get_db_schema(db_path=db_info.db_path)
        schema_structure = DatabaseSchema.from_schema_dict(db_schema_dict)
        schema_generator = DatabaseSchemaGenerator(
            tentative_schema=schema_structure,
            db_id=db_id,
            db_path=db_info.db_path,
            add_examples=bool(self.use_cvd), 
            add_random_examples=bool(self.use_cvd) 
        )
        
        schema_string = schema_generator.generate_schema_string(
            include_column_value_examples=bool(self.use_cvd), 
            include_value_description=bool(self.use_cvd)
        ) 

        input_seq = self.prompt_template.format(
            DB_ID = db_id,
            QUESTION = question,
            DB_SCHEMA = schema_string,
        )

        # prepare output sequences for SFT
        if self.use_reasoning:
            output_seq = f"\n<think>\n{dac_reasoning}\n</think>"
            output_seq = output_seq + f"\n<answer>\n{sql}\n</answer>"
        else:
            output_seq = f"\n<answer>\n{sql}\n</answer>"
            
        if self.use_unsloth and self.use_grpo:
            return {"prompt": input_seq, "answer": sql, "question": question, "db_path": str(db_info.db_path), "task": "btws"} # Need to conver db_path type to string due to serializability
        elif self.use_unsloth:
            input_and_output = input_seq + output_seq
            return {"text": input_and_output}

        return prepare_inputs_and_labels(prefix_seq=input_seq, target_seq=output_seq, tokenizer=self.tokenizer, max_tokens=self.max_tokens)

class Text2SQLDataset(Dataset):
    def __init__(self, 
                 t2s_dataset_path: Union[str, Path], 
                 db_info: DatabaseGeneralInfo, 
                 use_grpo: bool, 
                 use_unsloth: bool, 
                 use_schema: bool,
                 schema_content: Literal['filtered_schema', 'ground_truth_schema', 'whole_schema'], 
                 use_cvd: bool,
                 use_few_shot: bool,
                 few_shot_cnt: int,
                 use_reasoning_in_few_shots: bool,
                 use_reasoning: bool, 
                 tokenizer = None, 
                 max_tokens = 32768
                 ):
        self.t2s_dataset_path = Path(t2s_dataset_path)
        self.db_info = db_info
        self.use_grpo = use_grpo
        self.use_unsloth = use_unsloth
        self.use_schema = use_schema
        self.schema_content = schema_content
        self.use_cvd = use_cvd
        self.use_few_shot = use_few_shot
        self.few_shot_cnt = few_shot_cnt
        self.use_reasoning_in_few_shots = use_reasoning_in_few_shots
        self.use_reasoning = use_reasoning
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens

        self.dataset = self.load_dataset()

        prompt_template = load_template(template_name='t2s')

        if not use_reasoning:
            pt = prompt_template.split('<think>')[0] + prompt_template.split('</think>')[1] 
            self.prompt_template = pt
        else:
            self.prompt_template = prompt_template

    def load_dataset(self):
        print(f"Loading data from {self.t2s_dataset_path}...")
        original_dataset = []
        dataset = []
        dataset_path = self.t2s_dataset_path
        if dataset_path.suffix == ".json":
            with open(dataset_path, 'r') as file:
                original_dataset = json.load(file)
            for sub_schema_id, t2s_examples in original_dataset.items():
                db_id = sub_schema_id.split('-')[0]
                for example in t2s_examples:
                    if example.get('execution_status') == "SYNTACTICALLY_CORRECT":
                        example["db_id"] = db_id
                        dataset.append(example)

        elif dataset_path.suffix == ".jsonl":
            with open(dataset_path, 'r') as file:
                for line in file:
                    try:
                        example = json.loads(line)
                        if example.get('execution_status') == "SYNTACTICALLY_CORRECT":
                            dataset.append(example)
                    except:
                        continue
        
        print("Data is loaded.")
        return dataset
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        example = self.dataset[idx]
        db_id = self.db_info.db_id
        question = example.get("question")
        evidence = example.get("evidence", "")
        question = question + "Hint: " + evidence if evidence else question
        sql = example.get("SQL")
        dac_reasoning = example.get("dac_reasoning") 
        # Few shot part
        few_shot: Dict = example.get("few_shot", {})
        few_shot_examples: List[Dict[str, Any]] = few_shot.get("examples")
        # print(f"INFO: len(few_shot_examples): {len(few_shot_examples)}") # DELETE OR COMMENT OUT LATER
        # print(f"INFO: self.few_shot_cnt: {self.few_shot_cnt}") # DELETE OR COMMENT OUT LATER
        few_shot_examples = few_shot_examples[:self.few_shot_cnt]
        # print(f"INFO: After fewshot count filtration len(few_shot_examples): {len(few_shot_examples)}") # DELETE OR COMMENT OUT LATER
        # Schema part
        filtered_schema_dict = example.get('filtered_schema', {}).get("schema_dict", {})
        gt_schema_dict = example.get("gt_schema_dict", {})
        
        few_shot_augmentation_string = ""
        if self.use_few_shot:
            few_shot_string = ""
            for e_id, few_shot_example in enumerate(few_shot_examples):
                example_question = few_shot_example.get("question", "")
                example_sql = few_shot_example.get("SQL", "")
                example_dac_reasoning = few_shot_example.get("dac_reasoning", "")
                example_string = f"Example {e_id+1}:\n"
                example_string += f"Example User Question: {example_question}\n"
                if self.use_reasoning_in_few_shots:
                    example_string += f"<think>{example_dac_reasoning}</think>\n"
                example_string += f"<answer>{example_sql}</answer>\n"
                few_shot_string += example_string + "\n"
            
            few_shot_instructions = "- Below example question and their corresponding SQL queries are given as an example. Read them carefully and analyze the example question intentions, understand the link between database items and question. These examples can help you to reach correct response.\n"
            few_shot_augmentation_string = "**EXAMPLES**\n" + few_shot_instructions + few_shot_string + "\n"
        
        # construct schema string
        schema_augmentation_string = ""
        schema_string = ""
        column_meanings_str = ""
        if self.use_schema:
            if self.schema_content == "whole_schema":
                schema_string = self.db_info.original_db_schema_generator.generate_schema_string(
                    include_column_value_examples=bool(self.use_cvd),
                    include_value_description=bool(self.use_cvd)
                )
                if self.use_cvd:
                    # Construct column meanings 
                    column_meanings_str = self.db_info.original_db_schema_generator.get_column_profiles_string(with_keys=False, with_references=False)
                    
            elif self.schema_content == "ground_truth_schema":
                schema_structure = DatabaseSchema.from_schema_dict(gt_schema_dict)
                schema_generator = DatabaseSchemaGenerator(
                    tentative_schema=schema_structure,
                    db_id=db_id,
                    db_path=self.db_info.db_path,
                    add_examples=False, 
                    add_random_examples=False  # making this True slow donw the process
                )
                schema_string = schema_generator.generate_schema_string(
                    include_column_value_examples=bool(self.use_cvd), 
                    include_value_description=bool(self.use_cvd)) 
                if self.use_cvd:
                    # Construct column meanings 
                    column_meanings_str = schema_generator.get_column_profiles_string(with_keys=False, with_references=False)
                    
            elif self.schema_content == "filtered_schema":
                schema_structure = DatabaseSchema.from_schema_dict(filtered_schema_dict)
                schema_generator = DatabaseSchemaGenerator(
                    tentative_schema=schema_structure,
                    db_id=db_id,
                    db_path=self.db_info.db_path,
                    add_examples=False, 
                    add_random_examples=False # making this True slow donw the process
                )
                schema_string = schema_generator.generate_schema_string(
                    include_column_value_examples=bool(self.use_cvd), 
                    include_value_description=bool(self.use_cvd)) 
                if self.use_cvd:
                    # Construct column meanings 
                    column_meanings_str = schema_generator.get_column_profiles_string(with_keys=False, with_references=False)


        
            schema_instruction = "- Deeply analyze the database schema and information related with the schema items. Link user question with the database items.\n"
            schema_augmentation_string = "**DATABASE SCHEMA INFORMATION**\n" + schema_instruction +  schema_string + "\n"
            if self.use_cvd:
                schema_augmentation_string += "**COLUMN INFORMATION**\n" + column_meanings_str + "\n"

        # Format the template
        augmentation_string = few_shot_augmentation_string + schema_augmentation_string
        input_seq = self.prompt_template.format(
            DB_ID = db_id,
            AUGMENTATION = augmentation_string,
            QUESTION = question,
        )
    
        # prepare output sequences for SFT
        if self.use_reasoning:
            output_seq = f"\n<think>\n{dac_reasoning}\n</think>"
            output_seq = output_seq + f"\n<answer>\n{sql}\n</answer>"
        else:
            output_seq = f"\n<answer>\n{sql}\n</answer>"

        if self.use_unsloth and self.use_grpo:
            return {"prompt": input_seq, "answer": sql, "question": question, "db_path": str(self.db_info.db_path), "task": "t2sws"} # Need to conver db_path type to string due to serializability
        elif self.use_unsloth:
            input_and_output = input_seq + output_seq
            return {"text": input_and_output}

        return prepare_inputs_and_labels(prefix_seq=input_seq, target_seq=output_seq, tokenizer=self.tokenizer, max_tokens=self.max_tokens)
        


class Text2SQLSchemalessDataset(Dataset):
    def __init__(self, t2s_dataset_paths: Union[str, List[str]], db_info: DatabaseGeneralInfo, use_grpo, use_unsloth, use_reasoning, tokenizer = None, max_tokens = 32768):
        self.t2s_dataset_paths = t2s_dataset_paths if isinstance(t2s_dataset_paths, list) else [t2s_dataset_paths]
        self.db_info = db_info
        self.use_grpo = use_grpo
        self.use_unsloth = use_unsloth
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens
        self.use_reasoning = use_reasoning
        
        self.dataset = self.load_dataset()

        prompt_template = load_template(template_name='train_t2s_schemaless')
        if not use_reasoning:
            pt = prompt_template.split('<think>')[0] + prompt_template.split('</think>')[1] 
            self.prompt_template = pt
        else:
            self.prompt_template = prompt_template
    
    def load_dataset(self):
        original_dataset = []
        dataset = []
        for dataset_path in self.t2s_dataset_paths:
            if dataset_path.suffix == ".json":
                with open(dataset_path, 'r') as file:
                    original_dataset = json.load(file)
                for sub_schema_id, t2s_examples in original_dataset.items():
                    db_id = sub_schema_id.split('-')[0]
                    for example in t2s_examples:
                        if example.get('execution_status') == "SYNTACTICALLY_CORRECT":
                            example["db_id"] = db_id
                            dataset.append(example)

            elif dataset_path.suffix == ".jsonl":
                with open(dataset_path, 'r') as file:
                    for line in file:
                        try:
                            example = json.loads(line)
                            if example.get('execution_status') == "SYNTACTICALLY_CORRECT":
                                dataset.append(example)
                        except:
                            continue
            
        return dataset

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        example = self.dataset[idx]
        db_id = example.get("db_id")
        question = example.get("question")
        evidence = example.get("evidence", "")
        question = question + "Hint: " + evidence if evidence else question
        t2s_gen_reasoning = example.get("chain_of_thought_reasoning") # reasoning when generating t2s pairs
        sql = example.get("SQL")
        dac_reasoning = example.get("dac_reasoning") # Divide-and-Conquer Reasoning for a t2s

        input_seq = self.prompt_template.format(
            DB_ID = db_id,
            QUESTION = question,
        )
        
        # prepare output sequences for SFT
        if self.use_reasoning: # sft trainig with reasoning
            output_seq = f"\n<think>\n{dac_reasoning}\n</think>"
            output_seq = output_seq + f"\n<answer>\n{sql}\n</answer>"
        else:
            output_seq = f"\n<answer>\n{sql}\n</answer>"

        if self.use_unsloth and self.use_grpo:
            return {"prompt": input_seq, "answer": sql, "question": question, "db_path": str(self.db_info.db_path), "task": "t2s"} # Need to conver db_path type to string due to serializability
        elif self.use_unsloth:
            input_and_output = input_seq + output_seq
            return {"text": input_and_output}

        
        return prepare_inputs_and_labels(prefix_seq=input_seq, target_seq=output_seq, tokenizer=self.tokenizer, max_tokens=self.max_tokens)
    
class Text2SQLWithSchemaDataset(Dataset):
    def __init__(self, t2s_dataset_paths: Union[str, List[str]], db_info: DatabaseGeneralInfo, schema_string, use_grpo, use_unsloth, use_reasoning, tokenizer = None, max_tokens = 32768):
        self.t2s_dataset_paths = t2s_dataset_paths if isinstance(t2s_dataset_paths, list) else [t2s_dataset_paths]
        self.t2s_dataset_paths = [Path(dataset_path) for dataset_path in t2s_dataset_paths]
        self.db_info = db_info
        self.schema_string = schema_string
        self.dbs_root_dir = db_info.dbs_root_dir
        self.use_grpo = use_grpo
        self.use_unsloth = use_unsloth
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens
        self.use_reasoning = use_reasoning
        
        self.dataset = self.load_dataset()

        prompt_template = load_template(template_name='train_t2s_with_schema')
        if not use_reasoning:
            pt = prompt_template.split('<think>')[0] + prompt_template.split('</think>')[1] 
            self.prompt_template = pt
        else:
            self.prompt_template = prompt_template
    
    def load_dataset(self):
        original_dataset = []
        dataset = []
        for dataset_path in self.t2s_dataset_paths:
            if dataset_path.suffix == ".json":
                with open(dataset_path, 'r') as file:
                    original_dataset = json.load(file)
                for sub_schema_id, t2s_examples in original_dataset.items():
                    db_id = sub_schema_id.split('-')[0]
                    for example in t2s_examples:
                        if example.get('execution_status') == "SYNTACTICALLY_CORRECT":
                            example["db_id"] = db_id
                            dataset.append(example)

            elif dataset_path.suffix == ".jsonl":
                with open(dataset_path, 'r') as file:
                    for line in file:
                        try:
                            example = json.loads(line)
                            if example.get('execution_status') == "SYNTACTICALLY_CORRECT":
                                dataset.append(example)
                        except:
                            continue
            
        return dataset

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        example = self.dataset[idx]
        db_id = example.get("db_id")
        question = example.get("question")
        evidence = example.get("evidence", "")
        question = question + "Hint: " + evidence if evidence else question
        t2s_gen_reasoning = example.get("chain_of_thought_reasoning") # reasoning when generating t2s pairs
        sql = example.get("SQL")
        dac_reasoning = example.get("dac_reasoning") # Divide-and-Conquer Reasoning for a t2s
        
        input_seq = self.prompt_template.format(
            DB_ID = db_id,
            QUESTION = question,
            DB_SCHEMA = self.schema_string,
        )
        
        # prepare output sequences for SFT
        if self.use_reasoning:
            output_seq = f"\n<think>\n{dac_reasoning}\n</think>"
            output_seq = output_seq + f"\n<answer>\n{sql}\n</answer>"
        else:
            output_seq = f"\n<answer>\n{sql}\n</answer>"

        if self.use_unsloth and self.use_grpo:
            return {"prompt": input_seq, "answer": sql, "question": question, "db_path": str(self.db_info.db_path), "task": "t2sws"} # Need to conver db_path type to string due to serializability
        elif self.use_unsloth:
            input_and_output = input_seq + output_seq
            return {"text": input_and_output}

        return prepare_inputs_and_labels(prefix_seq=input_seq, target_seq=output_seq, tokenizer=self.tokenizer, max_tokens=self.max_tokens)

class SchemaLinkingSchemalessDataset(Dataset):
    def __init__(self, t2s_dataset_paths: Union[str, List[str]], db_info: DatabaseGeneralInfo, use_grpo, use_unsloth, use_reasoning, tokenizer = None, max_tokens = 32768):
        self.t2s_dataset_paths = t2s_dataset_paths if isinstance(t2s_dataset_paths, list) else [t2s_dataset_paths]
        self.db_info = db_info
        self.dbs_root_dir = db_info.dbs_root_dir
        self.use_grpo = use_grpo
        self.use_unsloth = use_unsloth
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens
        self.use_reasoning = use_reasoning

        self.dataset = self.load_dataset()

        prompt_template = load_template(template_name='train_sl_schemaless')
        if not use_reasoning:
            pt = prompt_template.split('<think>')[0] + prompt_template.split('</think>')[1] 
            self.prompt_template = pt
        else:
            self.prompt_template = prompt_template

    def load_dataset(self):
        original_dataset = []
        dataset = []
        for dataset_path in self.t2s_dataset_paths:
            if dataset_path.suffix == ".json":
                with open(dataset_path, 'r') as file:
                    original_dataset = json.load(file)
                for sub_schema_id, t2s_examples in original_dataset.items():
                    db_id = sub_schema_id.split('-')[0]
                    for example in t2s_examples:
                        if example.get('execution_status') == "SYNTACTICALLY_CORRECT":
                            example["db_id"] = db_id
                            dataset.append(example)

            elif dataset_path.suffix == ".jsonl":
                with open(dataset_path, 'r') as file:
                    for line in file:
                        try:
                            example = json.loads(line)
                            if example.get('execution_status') == "SYNTACTICALLY_CORRECT":
                                dataset.append(example)
                        except:
                            continue
            
        return dataset
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        example = self.dataset[idx]
        db_id = example.get("db_id")
        question = example.get("question")
        evidence = example.get("evidence", "")
        question = question + "Hint: " + evidence if evidence else question
        t2s_gen_reasoning = example.get("chain_of_thought_reasoning") # reasoning when generating t2s pairs
        sql = example.get("SQL")
        dac_reasoning = example.get("dac_reasoning") # Divide-and-Conquer Reasoning for a t2s

        db_path = self.dbs_root_dir / db_id / f"{db_id}.sqlite"
        
        sql_columns_dict = get_sql_columns_dict(db_path=db_path, sql=sql)
        sql_columns_dict_json_str = "```json{\n" + json.dumps(sql_columns_dict) + "\n}```"

        input_seq = self.prompt_template.format(
            DB_ID = db_id, 
            QUESTION = question
            )
        
        # prepare output sequences for SFT
        if self.use_reasoning:
            output_seq = f"\n<think>\n{dac_reasoning}\n</think>"
            output_seq = output_seq + f"\n<answer>\n{sql_columns_dict_json_str}\n</answer>"
        else:
            output_seq = f"\n<answer>\n{sql_columns_dict_json_str}\n</answer>"
        
        if self.use_unsloth and self.use_grpo:
            return {"prompt": input_seq, "answer": sql_columns_dict_json_str, "question": question, "db_path": str(self.db_info.db_path), "task": "sl"} # Need to conver db_path type to string due to serializability
        elif self.use_unsloth:
            input_and_output = input_seq + output_seq
            return {"text": input_and_output}
        

        return prepare_inputs_and_labels(prefix_seq=input_seq, target_seq=output_seq, tokenizer=self.tokenizer, max_tokens=self.max_tokens)
    

class SchemaLinkingWithSchemaDataset(Dataset):
    def __init__(self, t2s_dataset_paths: Union[str, List[str]], db_info: DatabaseGeneralInfo, schema_string:str, use_grpo, use_unsloth, use_reasoning, tokenizer = None, max_tokens = 32768):
        self.t2s_dataset_paths = t2s_dataset_paths if isinstance(t2s_dataset_paths, list) else [t2s_dataset_paths]
        self.db_info = db_info
        self.schema_string = schema_string
        self.dbs_root_dir = db_info.dbs_root_dir
        self.use_grpo = use_grpo
        self.use_unsloth = use_unsloth
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens
        self.use_reasoning = use_reasoning

        self.dataset = self.load_dataset()

        prompt_template = load_template(template_name='train_sl_with_schema')
        if not use_reasoning:
            pt = prompt_template.split('<think>')[0] + prompt_template.split('</think>')[1] 
            self.prompt_template = pt
        else:
            self.prompt_template = prompt_template

    def load_dataset(self):
        original_dataset = []
        dataset = []
        for dataset_path in self.t2s_dataset_paths:
            if dataset_path.suffix == ".json":
                with open(dataset_path, 'r') as file:
                    original_dataset = json.load(file)
                for sub_schema_id, t2s_examples in original_dataset.items():
                    db_id = sub_schema_id.split('-')[0]
                    for example in t2s_examples:
                        if example.get('execution_status') == "SYNTACTICALLY_CORRECT":
                            example["db_id"] = db_id
                            dataset.append(example)

            elif dataset_path.suffix == ".jsonl":
                with open(dataset_path, 'r') as file:
                    for line in file:
                        try:
                            example = json.loads(line)
                            if example.get('execution_status') == "SYNTACTICALLY_CORRECT":
                                dataset.append(example)
                        except:
                            continue
            
        return dataset
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        example = self.dataset[idx]
        db_id = example.get("db_id")
        question = example.get("question")
        evidence = example.get("evidence", "")
        question = question + "Hint: " + evidence if evidence else question
        t2s_gen_reasoning = example.get("chain_of_thought_reasoning") # reasoning when generating t2s pairs
        sql = example.get("SQL")
        dac_reasoning = example.get("dac_reasoning") # Divide-and-Conquer Reasoning for a t2s

        db_path = self.dbs_root_dir / db_id / f"{db_id}.sqlite"
        
        sql_columns_dict = get_sql_columns_dict(db_path=db_path, sql=sql)
        sql_columns_dict_json_str = "```json{\n" + json.dumps(sql_columns_dict) + "\n}```"

        input_seq = self.prompt_template.format(
            DB_ID = db_id, 
            DB_SCHEMA = self.schema_string,
            QUESTION = question
            )
        
        # prepare output sequences for SFT
        if self.use_reasoning:
            output_seq = f"\n<think>\n{dac_reasoning}\n</think>"
            output_seq = output_seq + f"\n<answer>\n{sql_columns_dict_json_str}\n</answer>"
        else:
            output_seq = f"\n<answer>\n{sql_columns_dict_json_str}\n</answer>"
        
        if self.use_unsloth and self.use_grpo:
            return {"prompt": input_seq, "answer": sql_columns_dict_json_str, "question": question, "db_path": str(self.db_info.db_path), "task": "slws"} # Need to conver db_path type to string due to serializability
        if self.use_unsloth:
            input_and_output = input_seq + output_seq
            return {"text": input_and_output}

        return prepare_inputs_and_labels(prefix_seq=input_seq, target_seq=output_seq, tokenizer=self.tokenizer, max_tokens=self.max_tokens)
    
class DatabaseCompletionDataset(Dataset):
    def __init__(self, db_completion_dataset_path: str, db_info: DatabaseGeneralInfo, use_grpo, use_unsloth: bool, tokenizer = None, max_tokens = 32768):
        self.db_completion_dataset_path = db_completion_dataset_path
        self.db_info = db_info
        self.db_id = db_info.db_id
        self.db_path = db_info.db_path
        self.use_grpo = use_grpo
        self.use_unsloth = use_unsloth
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens

        self.dataset = self.load_dataset()
        self.prompt_template = load_template(template_name='train_db_completion')


    def load_dataset(self):
        with open(self.db_completion_dataset_path, 'r') as file:
            schema_missing_parts = json.load(file)

        return schema_missing_parts
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        example = self.dataset[idx]
        missing_parts = example
        complete_schema = get_db_schema(self.db_path)

        incomplete_schema = complete_schema
        for table_name, column_list in missing_parts.items():
            for column_name in column_list:
                try:
                    incomplete_schema[table_name].remove(column_name)
                except ValueError as ve:
                    print(f"ValueError: {ve}")
                    pass
            
            ## After removing columns, if there is no remainig column in a table, remove the table
            if len(incomplete_schema[table_name]) == 0:
                incomplete_schema.pop(table_name, None)

        # Converting incomplete schema and missign parts into string
        incomplete_schema_str = "Below tables and their columns are given.\n"
        for table_name, column_list in incomplete_schema.items():
            incomplete_schema_str += f"{table_name}: {str(column_list)}\n"


        input_seq = self.prompt_template.format(
            DB_ID = self.db_id,
            PARTIAL_SCHEMA = incomplete_schema_str,
        )
        
        # prepare output sequences for SFT
        output_seq =  "```json{\n" + json.dumps(missing_parts) + "\n}```"
        
        if self.use_unsloth and self.use_grpo:
            return {"prompt": input_seq, "answer": json.dumps(missing_parts), "db_path": str(self.db_info.db_path), "task": "slws"} # Need to conver db_path type to string due to serializability
        if self.use_unsloth:
            input_and_output = input_seq + output_seq
            return {"text": input_and_output}
        
        return prepare_inputs_and_labels(prefix_seq=input_seq, target_seq=output_seq, tokenizer=self.tokenizer, max_tokens=self.max_tokens)