
import random
import numpy as np
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from typing import List, Dict, Any

import torch
from torch.utils.data import IterableDataset, Dataset, DataLoader
import datasets

from utils.datasets.datasets import Text2SQLDataset, Text2SQLSchemalessDataset, Text2SQLWithSchemaDataset, DatabaseCompletionDataset, SchemaLinkingSchemalessDataset, SchemaLinkingWithSchemaDataset, BirdTrainText2SQLWithSchema
from utils.db_utils.db_info_utils import get_db_all_tables, get_db_schema
from utils.db_utils.db_info import DatabaseGeneralInfo
from utils.db_utils.schema import DatabaseSchema
from utils.db_utils.schema_generator import DatabaseSchemaGenerator
    
def prepare_train_datasets(args, tokenizer=None, max_tokens=32768):
    dataset_root_path = Path(args.dataset_root_path)
    dbs_root_dir = Path(args.dbs_root_dir)
    train_configs = args.config['train']
    granularity_level = args.config['granularity_level']
    schema_content = str(train_configs['schema_content'])
    use_cvd = train_configs['use_col_value_and_descriptions']
    use_few_shot = bool(train_configs.get("use_few_shot", False))
    few_shot_cnt = int(train_configs.get("few_shot_cnt", 0)) if use_few_shot else 0
    use_reasoning_in_few_shots = bool(train_configs.get("use_reasoning_in_few_shots", False)) if use_few_shot else False
    use_grpo = train_configs['use_grpo']
    use_unsloth = train_configs['use_unsloth']
    use_reasoning = train_configs['use_reasoning']
    dataset_tasks = train_configs['dataset_tasks']
    print(f"use_cvd: {use_cvd}") # DELETE OR COMMENT OUT LATER
    print(f"few_shot_cnt: {few_shot_cnt}") # DELETE OR COMMENT OUT LATER

    rndm = random.Random(args.seed)

    train_data, dev_data = [], []
    # train_parts, dev_parts = [], []
    train_split_percentage = 0.95

    for db_id in args.db_ids:
        db_info = DatabaseGeneralInfo(db_id=db_id, dbs_root_dir=dbs_root_dir)
        db_schema_dict = get_db_schema(db_path=db_info.db_path)
        schema_structure = DatabaseSchema.from_schema_dict(db_schema_dict)
        schema_generator = DatabaseSchemaGenerator(
            tentative_schema=schema_structure,
            db_id=db_id,
            db_path=db_info.db_path,
            add_examples=bool(use_cvd), 
            add_random_examples=bool(use_cvd) 
        )
        db_schema_string_fixed_wo_examples = schema_generator.generate_schema_string(
            include_column_value_examples=bool(use_cvd), 
            include_value_description=bool(use_cvd)) 

        # t2s_dataset_path = db_info.db_prep_dir / "sub_schemas" / f"{granularity_level}_level" / f"sub_schema_examples.json"
        t2s_dataset_path = db_info.db_prep_dir / "sub_schemas" / f"{granularity_level}_level" / f"sub_schema_examples_train_with_few_shots.jsonl"
        db_completion_dataset_path = db_info.db_prep_dir / "db_completion" / "schema_missing_parts.json"

        if 'dc' in dataset_tasks:
            db_completion_dataset = DatabaseCompletionDataset(
                db_completion_dataset_path=db_completion_dataset_path,
                db_info=db_info,
                use_grpo=use_grpo, 
                use_unsloth=use_unsloth,
                tokenizer=tokenizer,
                max_tokens=max_tokens
            )
            split = int(train_split_percentage * len(db_completion_dataset))
            indices = list(range(len(db_completion_dataset)))
            rndm.shuffle(indices)
            train_data += [db_completion_dataset[i] for i in indices[:split]]
            dev_data += [db_completion_dataset[i] for i in indices[split:]]
            # observe data
            print(f"Observe DB Completion {db_completion_dataset[0]}")


        if 'sl' in dataset_tasks: # sl == schema linking
            sl_dataset = SchemaLinkingSchemalessDataset(
                t2s_dataset_paths=t2s_dataset_path, 
                db_info=db_info,
                use_grpo=use_grpo,
                use_unsloth=use_unsloth, 
                use_reasoning=use_reasoning,
                tokenizer=tokenizer,
                max_tokens=max_tokens
            )
            split = int(train_split_percentage * len(sl_dataset))
            indices = list(range(len(sl_dataset)))
            rndm.shuffle(indices)
            train_data += [sl_dataset[i] for i in indices[:split]]
            dev_data += [sl_dataset[i] for i in indices[split:]]
            print(f"Observe Schema Linking (Schemaless) {sl_dataset[0]}")
            

        if 'slws' in dataset_tasks: # slws == schema linking with schema
            slws_dataset = SchemaLinkingWithSchemaDataset(
                t2s_dataset_paths=t2s_dataset_path, 
                db_info=db_info,
                schema_string=db_schema_string_fixed_wo_examples,
                use_grpo=use_grpo, 
                use_unsloth=use_unsloth, 
                use_reasoning=use_reasoning,
                tokenizer=tokenizer,
                max_tokens=max_tokens
            )
            split = int(train_split_percentage * len(slws_dataset))
            indices = list(range(len(slws_dataset)))
            rndm.shuffle(indices)
            train_data += [slws_dataset[i] for i in indices[:split]]
            dev_data += [slws_dataset[i] for i in indices[split:]]
            print(f"Observe Schema Linking (With Schema) {slws_dataset[0]}")

        if 't2s' in dataset_tasks: # t2s == text-to-sql
            # t2s_dataset = Text2SQLSchemalessDataset(
            #     t2s_dataset_paths=t2s_dataset_path, 
            #     db_info=db_info,
            #     use_grpo=use_grpo,
            #     use_unsloth=use_unsloth, 
            #     use_reasoning=use_reasoning,
            #     tokenizer=tokenizer,
            #     max_tokens=max_tokens
            # )
            t2s_dataset = Text2SQLDataset(
                t2s_dataset_path=t2s_dataset_path,
                db_info=db_info,
                use_grpo=use_grpo,
                use_unsloth=use_unsloth, 
                use_schema=False, # use_schema is True because of the t2sws (text-to-sql with schema) task
                schema_content=schema_content,
                use_cvd = use_cvd,
                use_few_shot= use_few_shot,
                few_shot_cnt = few_shot_cnt,
                use_reasoning_in_few_shots = use_reasoning_in_few_shots,
                use_reasoning = use_reasoning,
                tokenizer=tokenizer,
                max_tokens=max_tokens
            )
            split = int(train_split_percentage * len(t2s_dataset))
            indices = list(range(len(t2s_dataset)))
            rndm.shuffle(indices)
            train_data += [t2s_dataset[i] for i in indices[:split]]
            dev_data += [t2s_dataset[i] for i in indices[split:]]
            print(f"Observe Text-to-SQL (Schemaless) {t2s_dataset[0]}")

        if 't2sws' in dataset_tasks: # t2sws == text-to-sql with schema
            # t2sws_dataset = Text2SQLWithSchemaDataset(
            #     t2s_dataset_paths=t2s_dataset_path, 
            #     db_info=db_info,
            #     schema_string=db_schema_string_fixed_wo_examples,
            #     use_grpo=use_grpo,
            #     use_unsloth=use_unsloth, 
            #     use_reasoning=use_reasoning,
            #     tokenizer=tokenizer,
            #     max_tokens=max_tokens
            # )
            print("Creating t2sws dataset...") # DELETE OR COMMENT OUT LATER
            t2sws_dataset = Text2SQLDataset(
                t2s_dataset_path=t2s_dataset_path,
                db_info=db_info,
                use_grpo=use_grpo,
                use_unsloth=use_unsloth, 
                use_schema=True, # use_schema is True because of the t2sws (text-to-sql with schema) task
                schema_content=schema_content,
                use_cvd = use_cvd,
                use_few_shot= use_few_shot,
                few_shot_cnt = few_shot_cnt,
                use_reasoning_in_few_shots = use_reasoning_in_few_shots,
                use_reasoning = use_reasoning,
                tokenizer=tokenizer,
                max_tokens=max_tokens
            )
            print("t2sws dataset created...") # DELETE OR COMMENT OUT LATER
            split = int(train_split_percentage * len(t2sws_dataset))
            indices = list(range(len(t2sws_dataset)))
            rndm.shuffle(indices)
            train_data += [t2sws_dataset[i] for i in indices[:split]]
            dev_data += [t2sws_dataset[i] for i in indices[split:]]
            print(f"Observe Text-to-SQL (With SChema) {t2sws_dataset[0]}")

    if 'btws' in dataset_tasks:  # btws = bird dataset train split with schema
        btws_dataset = BirdTrainText2SQLWithSchema(
            dataset_root_path=dataset_root_path,
            use_grpo=use_grpo,
            use_unsloth=use_unsloth,
            use_reasoning=use_reasoning,
            use_cvd=use_cvd,
            tokenizer=tokenizer,
            max_tokens=max_tokens
        )
        train_data += btws_dataset
        print(f"Observe Bird Train Text-to-SQL (With SChema) {btws_dataset[0]}")
    
    # Get portion of data
    data_portion = train_configs['data_portion']
    start_index_multiplier = float(data_portion[0])
    end_index_multiplier = float(data_portion[1])
    
    train_start_index = int(start_index_multiplier * len(train_data))
    train_end_index = int(end_index_multiplier  * len(train_data))
    train_data = train_data[train_start_index : train_end_index]
    rndm.shuffle(train_data) # in-place shuffle

    dev_start_index = int(start_index_multiplier * len(dev_data))
    dev_end_index = int(end_index_multiplier * len(dev_data))
    dev_data = dev_data[dev_start_index : dev_end_index]
    rndm.shuffle(dev_data) # in-place shuffle

    # Convert to HF Dataset
    train_dataset = datasets.Dataset.from_list(train_data)
    dev_dataset = datasets.Dataset.from_list(dev_data)
    

    # Shuffle train and dev sets
    # train_dataset = ConcatDataset(train_parts)
    # dev_dataset = ConcatDataset(dev_parts)

    return train_dataset, dev_dataset


def get_dataset_instances(args, tokenizer=None, max_tokens=32768) -> List[Dataset]:
    """
    Returns a list of dataset instances
    """
    dataset_root_path = Path(args.dataset_root_path)
    dbs_root_dir = Path(args.dbs_root_dir)

    train_configs = args.config['train']
    granularity_level = args.config['granularity_level']
    dataset_tasks = train_configs['dataset_tasks']
    
    schema_content = str(train_configs['schema_content'])
    use_few_shot = bool(train_configs['use_few_shot'])
    few_shot_cnt = int(train_configs['few_shot_cnt']) if use_few_shot else 0
    use_reasoning_in_few_shots = bool(train_configs['use_reasoning_in_few_shots']) if use_few_shot else False

    use_cvd = bool(train_configs['use_col_value_and_descriptions'])
    use_grpo = bool(train_configs['use_grpo'])
    use_unsloth = bool(train_configs['use_unsloth'])
    use_reasoning = bool(train_configs['use_reasoning'])
    print(f"use_cvd: {use_cvd}")

    rndm = random.Random(args.seed)

    dataset_instances: List[Dataset] = []

    for db_id in args.db_ids:
        db_info = DatabaseGeneralInfo(db_id=db_id, dbs_root_dir=dbs_root_dir)
        db_schema_dict = get_db_schema(db_path=db_info.db_path)
        schema_structure = DatabaseSchema.from_schema_dict(db_schema_dict)
        schema_generator = DatabaseSchemaGenerator(
            tentative_schema=schema_structure,
            db_id=db_id,
            db_path=db_info.db_path,
            add_examples=bool(use_cvd), 
            add_random_examples=bool(use_cvd) 
        )
        db_schema_string_fixed_wo_examples = schema_generator.generate_schema_string(
            include_column_value_examples=bool(use_cvd), 
            include_value_description=bool(use_cvd)) 

        # t2s_dataset_path = db_info.db_prep_dir / "sub_schemas" / f"{granularity_level}_level" / f"sub_schema_examples.json"
        t2s_dataset_path = db_info.db_prep_dir / "sub_schemas" / f"{granularity_level}_level" / f"sub_schema_examples_train_with_few_shots.json"
        db_completion_dataset_path = db_info.db_prep_dir / "db_completion" / "schema_missing_parts.json"

        if 'dc' in dataset_tasks:
            db_completion_dataset = DatabaseCompletionDataset(
                db_completion_dataset_path=db_completion_dataset_path,
                db_info=db_info,
                use_grpo=use_grpo, 
                use_unsloth=use_unsloth,
                tokenizer=tokenizer,
                max_tokens=max_tokens
            )
            dataset_instances.append(db_completion_dataset)

        if 'sl' in dataset_tasks: # sl == schema linking
            sl_dataset = SchemaLinkingSchemalessDataset(
                t2s_dataset_paths=t2s_dataset_path, 
                db_info=db_info,
                use_grpo=use_grpo,
                use_unsloth=use_unsloth, 
                use_reasoning=use_reasoning,
                tokenizer=tokenizer,
                max_tokens=max_tokens
            )
            dataset_instances.append(sl_dataset)

        if 'slws' in dataset_tasks: # slws == schema linking with schema
            slws_dataset = SchemaLinkingWithSchemaDataset(
                t2s_dataset_paths=t2s_dataset_path, 
                db_info=db_info,
                schema_string=db_schema_string_fixed_wo_examples,
                use_grpo=use_grpo, 
                use_unsloth=use_unsloth, 
                use_reasoning=use_reasoning,
                tokenizer=tokenizer,
                max_tokens=max_tokens
            )
            dataset_instances.append(slws_dataset)
            
            
        if 't2s' in dataset_tasks: # t2s == text-to-sql
            # t2s_dataset = Text2SQLSchemalessDataset(
            #     t2s_dataset_paths=t2s_dataset_path, 
            #     db_info=db_info,
            #     use_grpo=use_grpo,
            #     use_unsloth=use_unsloth, 
            #     use_reasoning=use_reasoning,
            #     tokenizer=tokenizer,
            #     max_tokens=max_tokens
            # )
            t2s_dataset = Text2SQLDataset(
                t2s_dataset_path=t2s_dataset_path,
                db_info=db_info,
                use_grpo=use_grpo,
                use_unsloth=use_unsloth, 
                use_schema=False, # use_schema is True because of the t2s (text-to-sql without schema) task
                schema_content=schema_content,
                use_cvd = use_cvd,
                use_few_shot= use_few_shot,
                few_shot_cnt = few_shot_cnt,
                use_reasoning_in_few_shots = use_reasoning_in_few_shots,
                use_reasoning = use_reasoning,
                tokenizer=tokenizer,
                max_tokens=max_tokens
            )
            dataset_instances.append(t2s_dataset)
            

        if 't2sws' in dataset_tasks: # t2sws == text-to-sql with schema
            # t2sws_dataset = Text2SQLWithSchemaDataset(
            #     t2s_dataset_paths=t2s_dataset_path, 
            #     db_info=db_info,
            #     schema_string=db_schema_string_fixed_wo_examples,
            #     use_grpo=use_grpo,
            #     use_unsloth=use_unsloth, 
            #     use_reasoning=use_reasoning,
            #     tokenizer=tokenizer,
            #     max_tokens=max_tokens
            # )
            t2sws_dataset = Text2SQLDataset(
                t2s_dataset_path=t2s_dataset_path,
                db_info=db_info,
                use_grpo=use_grpo,
                use_unsloth=use_unsloth, 
                use_schema=True, # use_schema is True because of the t2sws (text-to-sql with schema) task
                schema_content=schema_content,
                use_cvd = use_cvd,
                use_few_shot= use_few_shot,
                few_shot_cnt = few_shot_cnt,
                use_reasoning_in_few_shots = use_reasoning_in_few_shots,
                use_reasoning = use_reasoning,
                tokenizer=tokenizer,
                max_tokens=max_tokens
            )
            dataset_instances.append(t2sws_dataset)
    
    if 'btws' in dataset_tasks:  # btws = bird dataset train split with schema
        btws_dataset = BirdTrainText2SQLWithSchema(
            dataset_root_path=dataset_root_path,
            use_grpo=use_grpo,
            use_unsloth=use_unsloth,
            use_reasoning=use_reasoning,
            use_cvd=use_cvd,
            tokenizer=tokenizer,
            max_tokens=max_tokens
        )
        dataset_instances.append(t2sws_dataset)

    return dataset_instances