import os
import re
import json
import logging
import math
import torch
import time
import random
import gc
import shutil
import sqlite3
import threading
from pathlib import Path
from dataclasses import asdict
from collections import defaultdict
from typing import List, Tuple, Dict, Any, Union, Optional, Literal
from dataclasses import dataclass, field
from itertools import chain, combinations
from pydantic import BaseModel
from concurrent.futures import ThreadPoolExecutor, as_completed

from utils.db_utils.DatabaseDataTracker import DatabaseDataTracker

from utils.database_manager import DatabaseManager
from utils.db_utils.schema import DatabaseSchema
from utils.db_utils.schema_generator import DatabaseSchemaGenerator
from utils.db_utils.execution import execute_sql, get_execution_status
from utils.db_utils.db_info_utils import get_db_all_tables, get_db_schema
from utils.db_utils.db_info import DatabaseGeneralInfo
from utils.db_utils.helper import get_combinations, get_all_combinations
from utils.llm_utils.prompt_utils import load_template, load_template_examples
from utils.llm_utils.model import call_llm
from utils.db_utils.db_catalog.csv_utils import load_tables_description
from utils.db_utils.sql_parser import get_sql_columns_dict
from utils.output_structures.models import TextToSQLPair, SQLFix, TextToSQLReasoningGeneration
from utils.llm_utils.PreprocessLLMService import PreprocessLLMService
from utils.db_utils.db_values.preprocess_db_values import make_db_lsh

file_lock = threading.Lock()


class PreprocessRunner:
    PREPROCESS_ROOT_PATH = "preprocessed_databases"

    def __init__(self, args: Any):
        self.args = args
        self.db_ids: List[str] = self._set_db_ids()
        model_name  = self.args.config['preprocess']['sample_generator']['model']
        self.llm_service = PreprocessLLMService(model_name=model_name)
    
    def _set_db_ids(self):

        DBS_ROOT_DIR = self.args.dbs_root_dir
        if not DBS_ROOT_DIR.exists() or not DBS_ROOT_DIR.is_dir():
            raise ValueError(f"Invalid directory: {DBS_ROOT_DIR}")
    
        db_ids_list = [dir_.name for dir_ in DBS_ROOT_DIR.iterdir() if dir_.is_dir()]
        db_ids_list = sorted(db_ids_list)
        
        return db_ids_list
    
    def _get_all_combinations(self, list_of_items: List[Any]) -> List[List[str]]:
        """
        Extract all combinatioins of tables.
        """
        item_num = len(list_of_items)
        all_combinations = list(chain.from_iterable(combinations(list_of_items, r) for r in range(1, item_num + 1)))
        return all_combinations
    
    def _construct_table_level_db_sub_schemas(self, db_info: DatabaseGeneralInfo) -> List[DatabaseSchemaGenerator]:
        """
        Generating table level sub-schemas 

        Args:
            db_info (DatabaseGeneralInfo): An instance of DatabaseGeneralInfo for a specific database whose sub-schemas will be generated
        
        Returns:
            List[DatabaseSchemaGenerator]: List of DatabaseSchemaGenerator instances for generated table level sub-schemas
        """
        self.db_logger.info(f"*** STEP 1.1 START: {db_info.db_id.upper()} TABLE LEVEL SUB-SCHEMA CONSTRUCTION ***")

        db_table_level_sub_schemas_generators = []
        db_table_level_sub_schema_tcs = []

        table_level_ss_dir = db_info.sub_schemas_dir / 'table_level' 
        table_level_ss_dir.mkdir(parents=True, exist_ok=True)
        
        # database_manager = DatabaseManager(dataset=self.args.dataset, db_mode=self.args.data_mode, db_id=db_info.db_id)
        # db_original_schema_dict=DatabaseManager().get_db_schema()
        # schema_with_descriptions = load_tables_description(db_directory_path=db_info.db_directory, use_value_description=True)
        # db_schema_generator = DatabaseSchemaGenerator(
        #     tentative_schema=DatabaseSchema.from_schema_dict(db_original_schema_dict),
        #     schema_with_descriptions=DatabaseSchema.from_schema_dict_with_descriptions(schema_with_descriptions),
        #     db_id=db_info.db_id, 
        #     db_path=db_info.db_path)
        db_schema_generator = db_info.original_db_schema_generator
        self.db_logger.info(f"The database Complete Schema: \n {db_schema_generator.generate_schema_string(include_column_value_examples=True, include_value_description=True)}")

        sub_schemas = db_schema_generator.generate_table_level_sub_schemas()
        #TODO: You may generate sub-schemas utilizing LLMs especially for the ones whose PK and FKs are not properly set

        ## Generate sub-schema generator instances, append them into a list and save sub-schemas details into json file
        for tlsn, sub_schema_instance in enumerate(sub_schemas):
            self.db_logger.info(f"-Table Level Sub-Schema (tlsn={tlsn}): {list(sub_schema_instance.tables.keys())}")
            sub_schema_generator = DatabaseSchemaGenerator(
                tentative_schema=sub_schema_instance,
                schema_with_examples=sub_schema_instance,
                schema_with_descriptions=sub_schema_instance,
                db_id = db_info.db_id,
                db_path=db_info.db_path,
                tlsn=tlsn, # tlsn = table level schema no
                add_examples=True
            )
            schema_id = sub_schema_generator.schema_id
            schema_dict = sub_schema_generator.convert_schema_to_dict()
            schema_tc_dict = {k: v for k, v in schema_dict.items() if k != "schema"} # getting schema dict after removing detailed schema 
            db_table_level_sub_schema_tcs.append(schema_tc_dict)
            """
            if tlsn == 147 : ### DELETE LATER
                self.db_logger.info(f"=======TLSN = 147======") ### DELETE LATER
                self.db_logger.info(f"{sub_schema_instance.tables['tags'].columns['ExcerptPostId']}") ### DELETE LATER
                self.db_logger.info(f"=======TLSN = 147 SCHEMA STRING======") ### DELETE LATER
                self.db_logger.info(f"{sub_schema_generator.generate_schema_string(shuffle_cols=False)}") ### DELETE LATER
                self.db_logger.info(f"postHistory: \n{asdict(sub_schema_generator.schema_structure.tables['postHistory'])}") ### DELETE LATER
                self.db_logger.info(f"tags: \n{asdict(sub_schema_generator.schema_structure.tables['tags'])}") ### DELETE LATER
                self.db_logger.info(f"tags['ExcerptPostId']: \n{asdict(sub_schema_generator.schema_structure.tables['tags'].columns['ExcerptPostId'])}") ### DELETE LATER
                return ### DELETE LATER
            """
            
            ## Save Schema to a separate json file
            # sub_schema_json_file_path = table_level_ss_dir / f"{schema_id}.json"
            # sub_schema_generator.save_schema_to_json_file(sub_schema_json_file_path)

            ## Add sub_schema to a list
            db_table_level_sub_schemas_generators.append(sub_schema_generator)
        
        # Saving all table level ss dicts into a json file
        table_level_sub_schemas_json_path = table_level_ss_dir / f"sub_schemas.json"
        with open(table_level_sub_schemas_json_path, 'w') as file:
            json.dump(db_table_level_sub_schema_tcs, file, indent=4)
        
        self.db_logger.info(f"*** STEP 1.1 END: {db_info.db_id.upper()} TABLE LEVEL SUB-SCHEMA CONSTRUCTION ***")
        return db_table_level_sub_schemas_generators

    def _construct_column_level_db_sub_schemas(self, db_info: DatabaseGeneralInfo, db_table_level_sub_schemas_generators: List[DatabaseSchemaGenerator] ) -> None:
        """
        Generating column level sub-schemas from table level sub-schemas

        Args:
            db_info (DatabaseGeneralInfo): An instance of DatabaseGeneralInfo for a specific database whose sub-schemas will be generated
            db_table_level_sub_schemas_generators (List[DatabaseSchemaGenerator]): List of DatabaseSchemaGenerator objects for each table level sub-schemas
        
        Returns:
            None
        """
        self.db_logger.info(f"*** STEP 1.2 START: {db_info.db_id.upper()} COLUMN LEVEL SUB-SCHEMA CONSTRUCTION ***")

        sub_schema_generator_config =  self.args.config['preprocess']['sub_schema_generator']
        db_ss_configs = sub_schema_generator_config.get('db_ss_configs')
        current_db_ss_config = db_ss_configs.get(self.args.data_mode).get(db_info.db_id)

        window = current_db_ss_config.get('sliding_window_length')
        stride = current_db_ss_config.get('stride')
        table_level_ss_table_counts = current_db_ss_config.get('table_level_ss_table_counts')

        self.db_logger.info(f"table_level_ss_table_counts: {table_level_ss_table_counts}")
        # self.db_logger.info(f"type table_level_ss_table_counts: {type(table_level_ss_table_counts)}")
        

        db_column_level_sub_schemas_generators = []
        db_column_level_sub_schema_tcs = []
        column_level_ss_dir = db_info.sub_schemas_dir / 'column_level' 
        column_level_ss_dir.mkdir(parents=True, exist_ok=True)

        
        for tl_ss_generator in db_table_level_sub_schemas_generators:
            # tl_ss_generator stands for single table level sub-schema generator
            ss_table_cnt = len(list(tl_ss_generator.schema_structure.tables.keys()))
            if ss_table_cnt not in table_level_ss_table_counts:
                continue

            tlsn = tl_ss_generator.tlsn # tlsn == table level schema no
            sub_schemas = tl_ss_generator.generate_column_level_sub_schemas_via_sliding_window(window, stride)
            self.db_logger.info(f"-For Table Level Sub-Schema (tlsn={tlsn}), available Column Level Sub-Schema Number: {len(sub_schemas)}")
            for clsn, sub_schema_instance in enumerate(sub_schemas):
                self.db_logger.info(f"--Column Level Sub-Schema (tlsn={tlsn} - clsn={clsn}) {clsn}: \n {sub_schema_instance.to_dict(with_info=False)}")
                
                # # CHANGE IN JULY 5
                # sub_schema_generator = DatabaseSchemaGenerator(
                #     tentative_schema=sub_schema_instance,
                #     schema_with_examples=sub_schema_instance,
                #     schema_with_descriptions=sub_schema_instance,
                #     db_id = db_info.db_id,
                #     db_path=db_info.db_path,
                #     tlsn=tlsn,
                #     clsn=clsn,
                #     add_examples=True
                # )
                # schema_dict = sub_schema_generator.convert_schema_to_dict()
                # schema_tc_dict = {k: v for k, v in schema_dict.items() if k != "schema"} # getting schema dict after removing detailed schema 
                # # Add sub_schema to a list
                # db_column_level_sub_schemas_generators.append(sub_schema_generator)
                
                schema_tc_dict = {
                    "schema_id": f"{db_info.db_id}-{tlsn}-{clsn}",
                    "db_id": db_info.db_id,
                    "db_path": db_info.db_path,
                    "tables_and_columns": sub_schema_instance.to_dict(with_info=False)
                }


                db_column_level_sub_schema_tcs.append(schema_tc_dict)

                ## Save Schema to a separate json file
                # schema_id = sub_schema_generator.schema_id
                # sub_schema_json_file_path = column_level_ss_dir / f"{schema_id}.json"
                # sub_schema_generator.save_schema_to_json_file(sub_schema_json_file_path)

        
        info_str = f"""" ++++++++++ {db_info.db_id} Sub-Schema Infor+++++++++\nConfig: \ntable_level_ss_table_counts:{table_level_ss_table_counts} \nwindow: {window}, \nstride:{stride}\nTotal number of column level sub-schema for {db_info.db_id}: {len(db_column_level_sub_schema_tcs)}"""
        self.db_logger.info(info_str)

        # Saving all column level ss table column info dicts into a json file
        column_level_sub_schemas_json_path = column_level_ss_dir / f"sub_schemas.json"
        with open(column_level_sub_schemas_json_path, 'w') as file:
            json.dump(db_column_level_sub_schema_tcs, file, indent=4)

        # Calculating the table columns' counts and saving into a file
        tracker = DatabaseDataTracker(db_id=db_info.db_id, db_path=db_info.db_path)
        for ss_dict in db_column_level_sub_schema_tcs:
            tables_and_columns = ss_dict.get('tables_and_columns', '') 
            for t_name, columns in tables_and_columns.items():
                for c_name in columns:
                    tracker.increase_column_count(table_name=t_name, column_name=c_name)
        tracker.write(dir=column_level_ss_dir, file_name="data_columns_counts_ss.json")
           

        print(f"*** STEP 1.2 END: {db_info.db_id.upper()} COLUMN LEVEL SUB-SCHEMA CONSTRUCTION ***")
        return None

    def _construct_column_level_db_sub_schemas_v1(self, db_info: DatabaseGeneralInfo, db_table_level_sub_schemas_generators: List[DatabaseSchemaGenerator] ) -> List[DatabaseSchemaGenerator]:
        """
        Generating column level sub-schemas from table level sub-schemas

        Args:
            db_info (DatabaseGeneralInfo): An instance of DatabaseGeneralInfo for a specific database whose sub-schemas will be generated
        
        Returns:
            List[DatabaseSchemaGenerator]: List of DatabaseSchemaGenerator instances for generated column level sub-schemas
        """
        db_column_level_sub_schemas_generators = []
        column_level_ss_dir = db_info.sub_schemas_dir / 'column_level' 
        column_level_ss_dir.mkdir(parents=True, exist_ok=True)
        
        for tl_ss_generator in db_table_level_sub_schemas_generators:
            # tl_ss_generator stands for single table level sub-schema generator
            tlsn = tl_ss_generator.tlsn
            sub_schemas = tl_ss_generator.generate_column_level_sub_schemas()
            print(f"-For Table Level Sub-Schema (tlsn={tlsn}), available Column Level Sub-Schema Number: {len(sub_schemas)}")
            for clsn, sub_schema_instance in enumerate(sub_schemas):
                print(f"--Column Level Sub-Schema (tlsn={tlsn} - clsn={clsn}) {clsn}: \n {sub_schema_instance.to_dict(with_info=False)}")
                sub_schema_generator = DatabaseSchemaGenerator(
                    tentative_schema=sub_schema_instance,
                    schema_with_examples=sub_schema_instance,
                    schema_with_descriptions=sub_schema_instance,
                    db_id = db_info.db_id,
                    db_path=db_info.db_path,
                    tlsn=tlsn,
                    clsn=clsn,
                    add_examples=True
                )
                schema_id = sub_schema_generator.schema_id
                sub_schema_json_file_path = column_level_ss_dir / f"{schema_id}.json"
                # sub_schema_generator.save_schema_to_json_file(sub_schema_json_file_path)
                db_column_level_sub_schemas_generators.append(sub_schema_generator)

        return db_column_level_sub_schemas_generators
    
    def _load_db_sub_schemas(self, ss_dir: Path, granularity_level: str) -> List[DatabaseSchemaGenerator]:
        """
        This function loads existing database sub_schemas 
        """
        self.db_logger.info(f"Loading sub_schemas...")
        db_sub_schemas_generators = []
        sub_schemas_json_path = ss_dir / "sub_schemas.json"
        with open(sub_schemas_json_path, 'r') as file:
            ss_info_list = json.load(file)

        for ss_dict  in ss_info_list:
            schema_id = ss_dict.get("schema_id")
            schema_id_parts = schema_id.split("-")
            
            tlsn = schema_id_parts[1] if len(schema_id_parts) > 1 else ""
            clsn = schema_id_parts[2] if len(schema_id_parts) > 2 else ""
            ss_schema_structure = DatabaseSchema.from_schema_dict(ss_dict.get("tables_and_columns"))
            ss_generator = DatabaseSchemaGenerator(
                tentative_schema = ss_schema_structure,
                db_id = ss_dict.get('db_id'),
                db_path= ss_dict.get("db_path"),
                tlsn=tlsn,
                clsn=clsn,
                add_examples=True,
                add_random_examples=True
            )
            db_sub_schemas_generators.append(ss_generator)
        
        ### To check the loaded db_sub_schemas_generators, save them into sub_schemas_2.json file. COMMENT OUT THIS PART LATER
        # schemas_after_loading = []
        # for ss_generator in db_sub_schemas_generators:
        #     schema_dict = ss_generator.convert_schema_to_dict()
        #     schemas_after_loading.append(schema_dict)
        # json_file_path = ss_dir / "sub_schemas_2.json"
        # with open(json_file_path, 'w') as file:
        #     json.dump(schemas_after_loading, file, indent=4)

        return db_sub_schemas_generators
    
    def _load_ss_generator_from_ss_info(self, ss_dict: Dict, add_examples: bool = True, add_random_examples: bool = True) -> DatabaseSchemaGenerator:
        """"
        Generates DatabaseSchemaGenerator instance for a specific sub-schema using sub-schema info dict

        Args:
            ss_dict (Dict): Dictionary stroring information about a specific sub-schema
            db_info (DatabaseGeneralInfo): database genereal information instance

        Returns:
            DatabaseSchemaGenerator: Instance of DatabaseSchemaGenerator for a specific sub-schema
        """
        schema_id = ss_dict.get("schema_id")
        schema_id_parts = schema_id.split("-")
        
        tlsn = schema_id_parts[1] if len(schema_id_parts) > 1 else ""
        clsn = schema_id_parts[2] if len(schema_id_parts) > 2 else ""
        ss_schema_structure = DatabaseSchema.from_schema_dict(ss_dict.get("tables_and_columns"))
        ss_generator = DatabaseSchemaGenerator(
            tentative_schema = ss_schema_structure,
            db_id = ss_dict.get('db_id'),
            db_path= ss_dict.get("db_path"),
            tlsn=tlsn,
            clsn=clsn,
            add_examples=add_examples,
            add_random_examples=add_random_examples
        )

        return ss_generator
    

    def _construct_db_sub_schemas(self, db_info: DatabaseGeneralInfo) -> None:
        """"
        From the full schema it constructs the sub schemas of the database

        Args:
            db_info (DatabaseGeneralInfo): An instance of DatabaseGeneralInfo for a specific database whose sub-schemas will be generated

        Returns:
            None
        """
        sub_schema_generator_config =  self.args.config['preprocess']['sub_schema_generator']
        granularity_level = sub_schema_generator_config.get('granularity_level')

        if granularity_level != 'table' and granularity_level != 'column':
            raise ValueError("Wrong granularity level value! It can be either table or column.")

        db_sub_schemas_generators = []

        # Construct table level sub-schemas
        if granularity_level == "table":
            table_level_ss_dir = db_info.sub_schemas_dir / 'table_level' 
            if table_level_ss_dir.exists() and any(table_level_ss_dir.iterdir()):
                print("Table level sub-schemas exist. Loading sub-schemas...")
                print("No need to load sub-schema generators at this code.")
                # db_sub_schemas_generators = self._load_db_sub_schemas(table_level_ss_dir, granularity_level="table")
            else:
                db_sub_schemas_generators = self._construct_table_level_db_sub_schemas(db_info=db_info)

        # Construct column level sub-schemas
        if granularity_level == 'column':
            column_level_ss_dir = db_info.sub_schemas_dir / 'column_level' 
            if column_level_ss_dir.exists() and any(column_level_ss_dir.iterdir()):
                print("Column level sub-schemas exist.Loading sub-schemas...")
                print("No need to load sub-schema generators at this code.")
                # db_sub_schemas_generators = self._load_db_sub_schemas(column_level_ss_dir, granularity_level="column")
            else:
                db_sub_schemas_generators = self._construct_table_level_db_sub_schemas(db_info=db_info)
                self._construct_column_level_db_sub_schemas(db_info, db_sub_schemas_generators)
                gc.collect() # to free the memery

        # return db_sub_schemas_generators
        return
    
    def _check_and_fix_sql(self, t2s_pair: Dict, db_info:DatabaseGeneralInfo, sub_schema_str:str, column_meanings: str, column_values_str: str) -> Dict:
        """
        The function checks the sql in the Text-to-SQL pair object and try to fix it if the SQL has execution error
        """
        # self.db_logger.info(f"-- Stage: Check and Fix")
        caf_st = time.time()
        sample_generator_config = self.args.config['preprocess']['sample_generator']

        generated_sql = t2s_pair.get('SQL')
        execution_status, error_reason = get_execution_status(db_path=db_info.db_path, sql=generated_sql, fetch="one", timeout=60)
        t2s_pair['execution_status'] = execution_status
        t2s_pair['error_reason'] = 'None' if not error_reason else error_reason
        t2s_pair['is_fixed'] = "False"

        if execution_status == "SYNTACTICALLY_INCORRECT":  # only fix the SQL query if its execution status is SYNTACTICALLY_INCORRECT
            # Revise the SQL query one time
            t2s_pair = self.llm_service.fix_sql(
                t2s_pair=t2s_pair, 
                sub_schema_string=sub_schema_str, 
                column_meanings=column_meanings, 
                column_values_str=column_values_str
                )
            fixed_sql = t2s_pair['SQL'] 
            fixed_sql_execution_status, fixed_sql_error_reason = get_execution_status(db_path=db_info.db_path, sql=fixed_sql, fetch="one", timeout=60) # check fixed SQl executability
            t2s_pair['execution_status'] = fixed_sql_execution_status
            t2s_pair['error_reason'] = 'None' if not fixed_sql_error_reason else fixed_sql_error_reason
            t2s_pair['is_fixed'] = "True"
            caf_et = time.time()
            caf_duration = caf_et - caf_st
            self.db_logger.info(f"-- Stage: Check and Fix completed in {caf_duration} seconds (Fixing applied).")
            return t2s_pair
        else:
            caf_et = time.time()
            caf_duration = caf_et - caf_st
            self.db_logger.info(f"-- Stage: Check and Fix completed in {caf_duration} seconds (Fixing is not needed). ")
            return t2s_pair
        
    def _evaluate_logic_in_t2s_pair(self, t2s_pair: Dict, db_info: DatabaseGeneralInfo,  sub_schema_str:str, column_meanings: str, column_values_str: str) -> Dict:
        """
        This function evaluates the logic of the sql query and question.
        """
        if "is_logical" not in t2s_pair:
            el_st = time.time()
            evaluation = self.llm_service.evaluate_logic_in_single_t2s(t2s_pair=t2s_pair, sub_schema_string=sub_schema_str, column_meanings=column_meanings, column_values_str=column_values_str)
            t2s_pair["question_and_sql_logic_analysis"] = evaluation.get("question_and_sql_logic_analysis", "")
            t2s_pair["is_logical"] = evaluation.get("is_logical", "")
            el_duration = time.time() - el_st
            self.db_logger.info(f"-- Stage: Evaluate Logic in Text-to-SQL pair completed in {el_duration} seconds.")
        return t2s_pair
        
    def _generate_reasoning_for_t2s(self,  t2s_pair: Dict, original_full_schema_str: str) -> Dict:
        """
        The function generates Divide-and-Conquery reasoning for a given syntactically corret Text-to-SQL and corresponding full schema
        """
        if t2s_pair['execution_status'] == "SYNTACTICALLY_CORRECT" or t2s_pair['execution_status'] == "EMPTY_RESULT":
            # self.db_logger.info(f"-- Stage: Generate Reasoning")
            gr_st = time.time()
            t2s_pair = self.llm_service.generate_reasoning(t2s_pair=t2s_pair, original_full_schema_str=original_full_schema_str, logger=self.db_logger)
            gr_et = time.time()
            gr_duration = gr_et - gr_st
            self.db_logger.info(f"-- Stage: Generate Reasoning completed in {gr_duration} seconds.")
            return t2s_pair
        else:
            return t2s_pair
        
    # def _judge_generated_t2s_pairs(self, t2s_pairs: List[Dict], sub_schema_str: str, column_meaning_string: str) -> Dict:
    #     """"
    #     Evaluate the generated text-to-sql pairs
    #     """
    #     judge_start = time.time()
    #     evaluated_t2s_pairs = self.llm_service.evaluate_t2s_pairs(t2s_pairs, sub_schema_str=sub_schema_str, column_meaning_string=column_meaning_string)
    #     judge_duration = time.time() - judge_start
    #     self.db_logger.info(f"-- Stage: Judging generated Text-to-SQL examples completed in {judge_duration} seconds.")

    #     return evaluated_t2s_pairs
        
    def _process_generated_t2s_pair(self, t2s_pair:Dict, db_info:DatabaseGeneralInfo, sub_schema_str:str, original_full_schema_str: str, column_meanings: str, column_values_str: str):
        """
        Helper function to process a single t2s_pair.
        Process includes checking and fixing generated SQL query. Then generating Divide-and-Conquer reasoning
        """

        # Evaluate the logic
        if "is_logical" not in t2s_pair:
            t2s_pair = self._evaluate_logic_in_t2s_pair(
                t2s_pair=t2s_pair,
                db_info=db_info,
                sub_schema_str=sub_schema_str,
                column_meanings=column_meanings,
                column_values_str=column_values_str
            )

        if bool(t2s_pair.get("is_logical", False)):
            # Check and fix SQL
            t2s_pair = self._check_and_fix_sql(
                t2s_pair=t2s_pair,
                db_info=db_info,
                sub_schema_str=sub_schema_str,
                column_meanings=column_meanings,
                column_values_str=column_values_str
            )
            # Generate reasoning
            t2s_pair = self._generate_reasoning_for_t2s(
                t2s_pair=t2s_pair,
                original_full_schema_str=original_full_schema_str
            )

        return t2s_pair
     
    def _column_count_analysis(self, db_info: DatabaseGeneralInfo, jsonl_path: Path) -> DatabaseDataTracker:

        """
        The function analyze the column counts in the examples which are generated synthetically
        """

        tracker = DatabaseDataTracker(db_id=db_info.db_id, db_path=db_info.db_path)
        with open(jsonl_path, 'r', encoding='utf-8') as infile:
            for line in infile:
                try:
                    t2s_example_dict = json.loads(line)
                    if t2s_example_dict.get('execution_status', '') == 'SYNTACTICALLY_INCORRECT':
                        self.db_logger.info("Encountered SYNTACTICALLY_INCORRECT SQL. Continuing with the next one.")
                        continue
                    
                    sql_query = t2s_example_dict.get('SQL')
                    sql_columns_dict = get_sql_columns_dict(db_path=db_info.db_path, sql=sql_query)
                    # self.db_logger.info(f"SQL Columns Dict:\n{sql_columns_dict}")
                    for sql_t_name, sql_c_name_list in sql_columns_dict.items():
                        for sql_c_name in sql_c_name_list:
                            tracker.increase_column_count(table_name=sql_t_name, column_name=sql_c_name)

                except json.JSONDecodeError:
                    self.db_logger.warning(f"Skipping malformed line in {jsonl_path} file")
                except Exception as e:
                    self.db_logger.error(f"Unexpected error while processing a line: {e}", exc_info=True)


        ## Below code is valid if you were used json file for the text-to-sql examples
        # for ss_id, t2s_examples in ss_examples_dict.items():    
        #     for t2s_example_dict in t2s_examples:
        #         if t2s_example_dict.get('execution_status', '') == 'SYNTACTICALLY_INCORRECT':
        #             self.db_logger.info("Encountered SYNTACTICALLY_INCORRECT sql. Continuing with the next one.")
        #             continue
        #         sql_query = t2s_example_dict.get('SQL')
        #         sql_columns_dict = get_sql_columns_dict(db_path=db_info.db_path, sql=sql_query)
        #         # self.db_logger.info(f"SQL Columns Dict:\n{sql_columns_dict}")
        #         for sql_t_name, sql_c_name_list in sql_columns_dict.items():
        #             for sql_c_name in sql_c_name_list:
        #                 tracker.increase_column_count(table_name=sql_t_name, column_name=sql_c_name)
                
        return tracker


    
    def _generate_ss_t2s_examples(self, db_info: DatabaseGeneralInfo) -> bool:
        """ 
        Generating syntetic text-to-sql samples for each sub-schema
        """
        sample_generator_config = self.args.config['preprocess']['sample_generator']
        granularity_level = sample_generator_config['granularity_level']
        sub_schema_examples_jsonl_file_path = db_info.db_prep_dir / "sub_schemas" / f"{granularity_level}_level" / f"sub_schema_examples.jsonl"
        sample_count = sample_generator_config['sample_count_edf']
        column_value_cnt=sample_generator_config['column_value_cnt']
        eval_in_generation = sample_generator_config['eval_in_generation']
        # Get original full schema string for DAC reasoning
        original_full_schema_str = db_info.original_db_schema_generator.generate_schema_string(
            include_column_value_examples=True,
            include_value_description=True
            )
        
        column_level_ss_dir = db_info.sub_schemas_dir / 'column_level' 
        sub_schemas_json_path = column_level_ss_dir / "sub_schemas.json"
        with open(sub_schemas_json_path, 'r') as file:
            ss_info_list = json.load(file)

        ############################################################
        ##### STEP : TEXT-TO-SQL GENERATION USING SUB-SCHEMAS: #####
        ############################################################

        ### Initialize or load existing examples
        # self.db_logger.info("--Initialize or load existing examples") # When json is used instead of jsonl
        # if sub_schema_examples_json_file_path.exists():
        #     with open(sub_schema_examples_json_file_path, 'r') as file:
        #         ss_examples_dict = json.load(file)
        # else:
        #     ss_examples_dict: Dict[str, List[Dict]] = {}
        #     sub_schema_examples_json_file_path.parent.mkdir(parents=True, exist_ok=True)
        #     with open(sub_schema_examples_json_file_path, 'w') as file:
        #         json.dump(ss_examples_dict, file, indent=4)
        sub_schema_examples_jsonl_file_path.parent.mkdir(parents=True, exist_ok=True)
        existing_ss_ids = set()
        if sub_schema_examples_jsonl_file_path.exists():
            with open(sub_schema_examples_jsonl_file_path, 'r') as f:
                for line in f:
                    try:
                        obj = json.loads(line)
                        existing_ss_ids.add(obj.get('ss_id'))
                    except Exception:
                        continue

        proceed_next_step = False
        try:
            for idx, ss_info_dict in enumerate(ss_info_list):
                # Check if the sub_schema_id exist in ss_examples_dict. If yes, do not generate examples
                ss_t2s_generation_start_time = time.time()
                sub_schema_id = ss_info_dict.get('schema_id')
                # if sub_schema_id in ss_examples_dict.keys(): # When json is used instead of jsonl
                #     self.db_logger.info(f"Examples for the sub-schema (id = {sub_schema_id}) - ({idx+1}/{len(ss_info_list)}) have already been generated. Skipping to the next sub-schema...")
                #     if idx == len(ss_info_list) - 1:
                #         proceed_next_step = True
                #     continue
                # else:
                #     ss_examples_dict[sub_schema_id] = []
                if sub_schema_id in existing_ss_ids:
                    self.db_logger.info(f"Examples for the sub-schema (id = {sub_schema_id}) - ({idx+1}/{len(ss_info_list)}) have already been generated. Skipping to the next sub-schema...")
                    if idx == len(ss_info_list) - 1:
                        proceed_next_step = True
                    continue
                    
                sub_schema_generator: DatabaseSchemaGenerator = self._load_ss_generator_from_ss_info(ss_dict=ss_info_dict)
                self.db_logger.info(f"\n\nExample generation for {db_info.db_id} (ss_id = {sub_schema_id}) - {idx}/{len(ss_info_list)-1}")
                sub_schema_id = sub_schema_generator.schema_id
                
                sub_schema_string = sub_schema_generator.generate_schema_string(
                    include_column_value_examples=True, 
                    include_value_description=True, 
                    shuffle_cols=True, 
                    shuffle_tables=True
                    )
                sub_schema_string = sub_schema_string = "### DATABASE SCHEMA: \n" + sub_schema_string
                # column_meanings_str = sub_schema_generator.get_schema_column_meanings_string()
                column_meanings_str = sub_schema_generator.get_column_profiles_string(with_keys=False, with_references=False)
                column_values_str = sub_schema_generator.get_random_unique_column_values_string(value_cnt=sample_generator_config['column_value_cnt'])
                
                # self.db_logger.info(f"===== DATABASE ({db_info.db_id}-{sub_schema_id}) SUB-SCHEMA STRING ===== \n {sub_schema_string}")
                # self.db_logger.info(f"===== DATABASE ({db_info.db_id}-{sub_schema_id}) COLUMN PROFILES ===== \n {column_meanings_str}")

                # Generate Text-to-SQL examples
                self.db_logger.info("===== Initial Example generation =====")
                ieg_st = time.time()
                generated_t2s_examples: List[Dict] = self.llm_service.generate_sql_to_text_examples(
                    sub_schema_string=sub_schema_string,
                    sample_count=sample_count,
                    column_meanings=column_meanings_str,
                    column_values_str=column_values_str,
                    eval_in_generation=eval_in_generation
                    )
                ieg_et = time.time()
                ieg_duration = ieg_et - ieg_st
                self.db_logger.info(f"----Initial example generation completed in {ieg_duration} seconds")

                
                self.db_logger.info("===== Check Status - Fix - Generate DAC Reasoning =====")
                ### Checking execution status of generated SQL queries. Generating Divide-and-Conquer Reasoning. Then, adding into ss_examples_dict

                ### paralel process generated t2s pairs
                def process_pair_with_print(idx: int, pair: dict) -> tuple[int, dict]:
                    print(f"\n----- Processing Generated T2S Pair idx: {idx}")
                    result = self._process_generated_t2s_pair(
                        pair,
                        db_info,
                        sub_schema_string,
                        original_full_schema_str,
                        column_meanings_str,
                        column_values_str
                    )
                    return idx, result
                
                max_workers = min(len(generated_t2s_examples), 4 * (os.cpu_count() or 4)) # Cannot apply due to Gemini Free-tier limit
                max_workers = 12
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = [
                        executor.submit(
                            process_pair_with_print,
                            idx,
                            t2s_pair
                        )
                        for idx, t2s_pair in enumerate(generated_t2s_examples)
                    ]

                    for future in futures:
                        try:
                            idx, processed_t2s_pair = future.result()
                            processed_t2s_pair = {
                                'ss_id': sub_schema_id,
                                'example_no': idx,
                                **processed_t2s_pair
                            }
                            # Writing only the executable t2s examples to file (if its execution_status = SYNTACTICALLY_CORRECT)
                            # if processed_t2s_pair.get("execution_status") == "SYNTACTICALLY_CORRECT":
                            # Write all T2S pairs
                            self.db_logger.info(f"Writing processed T2S pair idx: {idx}")
                            with file_lock:
                                with open(sub_schema_examples_jsonl_file_path, 'a') as outfile:
                                    json_line = json.dumps(processed_t2s_pair)
                                    outfile.write(json_line + '\n')

                        except Exception as e:
                            self.db_logger.error(f"Error while processing T2S pair: {e}", exc_info=True)

                # ## Serial
                # for sub_t2s_pair_index, t2s_pair in enumerate(generated_t2s_examples):
                #     self.db_logger.info(f"----- Processing Generated T2S Pair ({sub_t2s_pair_index+1}/{len(generated_t2s_examples)})")
                #     try:
                #         processed_t2s_pair = self._process_generated_t2s_pair(
                #             t2s_pair=t2s_pair, 
                #             db_info=db_info, 
                #             sub_schema_str=sub_schema_string, 
                #             original_full_schema_str=original_full_schema_str, 
                #             column_meanings=column_meanings_str, 
                #             column_values_str=column_values_str
                #         )
                #         # ss_examples_dict[sub_schema_id].append(processed_t2s_pair)   # When json is used instead of jsonl
                #         processed_t2s_pair = {'ss_id': sub_schema_id, 'example_no': sub_t2s_pair_index, **processed_t2s_pair}
                #         # Write to .jsonl file
                #         with open(sub_schema_examples_jsonl_file_path, 'a') as outfile:
                #             json_line = json.dumps(processed_t2s_pair)
                #             outfile.write(json_line + '\n')

                #     except Exception as e:
                #         continue
                
                ### Write the updated dictionary to file
                ## Below code is valid if you use json
                # print("Writing whole ss_examples_dict into file")
                # with open(sub_schema_examples_json_file_path, 'w') as file:
                #     json.dump(ss_examples_dict, file, indent=4)


                ss_t2s_generation_end_time = time.time()
                ss_t2s_generation_duration = ss_t2s_generation_end_time - ss_t2s_generation_start_time
                self.db_logger.info(f"-Text-to-SQL examples for sub-schema whose id is {sub_schema_id} are generated in {ss_t2s_generation_duration} seconds and saved to {sub_schema_examples_jsonl_file_path}.")

                if idx == len(ss_info_list) - 1:
                    print("Text-to-SQL Generation (first step) is completed. For each sub-schema Text-to-SQL pairs generated. Proceeding to the next step. ")
                    proceed_next_step = True

                # if idx >= 3:
                #     break
                # del generated_t2s_examples
                gc.collect()
        except Exception as e:
            self.db_logger.error(f"Example generation process is stopped because of an error: {e}")
            if hasattr(e, "args") and e.args:
                error_message = str(e.args[0])
                if "database or disk is full" in error_message:
                    self.db_logger.critical("Disk is full. Halting further processing.")

                    total, used, free = shutil.disk_usage("/")
                    print(f"Disk total: {total // (1024**3)} GB")
                    print(f"Disk used: {used // (1024**3)} GB")
                    print(f"Disk free: {free // (1024**3)} GB")

                    raise RuntimeError("Halting due to disk being full.")
                    
                if "RESOURCE_EXHAUSTED" in error_message:
                    raise
        
        ########################################
        ##### STEP : COLUMN COUNT ANALYSIS #####
        ########################################
        if proceed_next_step:
            write_file_name = "data_columns_counts_synthetic_t2s.json"
            tracker = self._column_count_analysis(db_info=db_info, jsonl_path=sub_schema_examples_jsonl_file_path)
            tracker.write(dir=db_info.sub_schemas_dir, file_name=write_file_name)

        return proceed_next_step
    
    def _get_sub_schemas_including_low_columns(self, tracker: DatabaseDataTracker, db_info: DatabaseGeneralInfo) -> List[Tuple[str, str]]:
        """
        The function finds the low columns (columns whose counts in the synthetically generated examples are less than the threshold).
        Then, gets the ids of sub-schemas they are involved in.

        Args:

        Returns:
            List[Tuple[str, str]]: List of sub-schemas ids including low columns and the name of the low column (table_name.col_name)

        """
        # Determine low-columns
        sub_schema_generator_config =  self.args.config['preprocess']['sub_schema_generator']
        db_ss_configs = sub_schema_generator_config.get('db_ss_configs')
        current_db_ss_config = db_ss_configs.get(self.args.data_mode).get(db_info.db_id)
        min_column_example_cnt = current_db_ss_config.get('min_column_example_cnt')
        threshold = min_column_example_cnt
        low_columns: Dict[str, List[str]] = tracker.get_low_columns(threshold=threshold)
        self.db_logger.info(f"Low Columns Dict: \n {low_columns}")

        # Get dicts give info about sub_schemas
        column_level_ss_dir = db_info.sub_schemas_dir / 'column_level' 
        sub_schemas_json_path = column_level_ss_dir / "sub_schemas.json"
        with open(sub_schemas_json_path, 'r') as file:
            ss_info_list = json.load(file)

        ## Determine the sub-schemas including low-columns
        # For each low-column, select 10 sub-schema
        # ss_gens_for_column_focused_generation: List[Tuple[DatabaseSchemaGenerator, str]] = []
        ss_for_column_focused_generation: List[Tuple[str, str]] = []
        low_column_including_ss_gens: Dict[str, Dict[str, List[str]]] = {}
        for t_name_low, columns in low_columns.items():
            low_column_including_ss_gens[t_name_low] = {}
            for c_name_low in columns:
                low_column_including_ss_gens[t_name_low][c_name_low] = []
                for idx, ss_info_dict in enumerate(ss_info_list):
                    if len(low_column_including_ss_gens[t_name_low][c_name_low]) >= (threshold // 10):
                        break
                    # Check sub-schema includes the currentlu considered t_name_low and c_name_low
                    ss_id = ss_info_dict.get("schema_id")
                    sub_schema_generator: DatabaseSchemaGenerator = self._load_ss_generator_from_ss_info(ss_dict=ss_info_dict, add_examples=False, add_random_examples=False)
                    column_info = sub_schema_generator.schema_structure.get_column_info(table_name=t_name_low, column_name=c_name_low)
                    if column_info and len(low_column_including_ss_gens[t_name_low][c_name_low]) < (threshold // 10):
                        ## if column exist in the sub-schema and the sub-schema count existing that column lower than (threshold // 10), add it
                        ## This could be (threshold // (4*sample_generator_config['sample_count_edf']))
                        ## The reason of division is that it is expected that all the generated t2s examples in one prompt include the considered focused column
                        actual_table_name = sub_schema_generator.schema_structure.get_actual_table_name(t_name_low)
                        actual_column_name = sub_schema_generator.schema_structure.get_actual_column_name(table_name=actual_table_name, column_name=c_name_low)
                        low_column_including_ss_gens[t_name_low][c_name_low].append((ss_id, f"`{actual_table_name}`.`{actual_column_name}`"))
                        ss_for_column_focused_generation.append((ss_id, f"`{actual_table_name}`.`{actual_column_name}`"))

        return ss_for_column_focused_generation
    
    def _generate_column_focused_t2s_examples(self, db_info: DatabaseGeneralInfo, tracker: DatabaseDataTracker = None) -> bool:
        """ 
        Generating column specific syntetic text-to-sql samples
        """
        sample_generator_config = self.args.config['preprocess']['sample_generator']
        granularity_level = sample_generator_config['granularity_level']
        sub_schema_examples_jsonl_file_path = db_info.db_prep_dir / "sub_schemas" / f"{granularity_level}_level" / f"sub_schema_examples.jsonl"
        sample_count = sample_generator_config['sample_count_edf']
        column_value_cnt=sample_generator_config['column_value_cnt']
        eval_in_generation = sample_generator_config['eval_in_generation']
        original_full_schema_str = db_info.original_db_schema_generator.generate_schema_string(
            include_column_value_examples=True,
            include_value_description=True
            )
        
        # Load sub-schema info
        column_level_ss_dir = db_info.sub_schemas_dir / 'column_level' 
        sub_schemas_json_path = column_level_ss_dir / "sub_schemas.json"
        with open(sub_schemas_json_path, 'r') as file:
            ss_info_list = json.load(file)

        ## Load existing examples
        # Code below is for json usage
        # if sub_schema_examples_json_file_path.exists():
        #     with open(sub_schema_examples_json_file_path, 'r') as file:
        #         ss_examples_dict = json.load(file)
        # else:
        #     raise FileNotFoundError("Couldn't find sub-schema examples.")
        if not sub_schema_examples_jsonl_file_path.exists():
            raise FileNotFoundError("Couldn't find sub-schema examples.")

        ########################################
        ##### STEP : COLUMN COUNT ANALYSIS #####
        ########################################
        if not tracker:
            ## Analyzing the column counts in synthetically generated examples
            tracker = self._column_count_analysis(db_info=db_info, jsonl_path=sub_schema_examples_jsonl_file_path)

        ##########################################################################
        ##### STEP : SELECTING SUB-SCHEMAS FOR COLUMN FOCUSED T2S GENERATION #####
        ##########################################################################
        ss_for_column_focused_generation = self._get_sub_schemas_including_low_columns(
            tracker=tracker,
            db_info=db_info,
        )
        

        if len(ss_for_column_focused_generation) == 0:
            # If there is not SchemaGeneraotor instance that contains low columns, then return True
            self.db_logger.info(f"End of column focused generation.")
            proceed_next_step = True
            return proceed_next_step 
        else:
            self.db_logger.info(f"For column focused generation, {len(ss_for_column_focused_generation)} sub-schema will be used.")


        #########################################################
        ##### STEP : COLUMN FOCUSED TEXT-TO-SQL GENERATION: #####
        #########################################################

        proceed_next_step = False
        try:
            for idx, (ss_id, focused_column) in enumerate(ss_for_column_focused_generation):
                self.db_logger.info(f"Column focused example generation for {ss_id} - {idx}/{len(ss_for_column_focused_generation)-1}")
                gen_start_time = time.time()
                # Find the sub-schema dict info and
                ss_info_dict: Dict[str, Any] = None
                for idx, single_ss_info in enumerate(ss_info_list):
                    if ss_id == single_ss_info.get("schema_id"):
                        ss_info_dict = single_ss_info
                        break
                
                if not ss_info_dict:
                    self.db_logger.info(f"Couldn't find a schema with id {ss_id}")
                    continue
                
                # Generate sub-schema-generator object 
                sub_schema_generator: DatabaseSchemaGenerator = self._load_ss_generator_from_ss_info(ss_dict=ss_info_dict)
                sub_schema_id = sub_schema_generator.schema_id

                sub_schema_string = sub_schema_generator.generate_schema_string(
                    include_column_value_examples=True, 
                    include_value_description=True, 
                    shuffle_cols=True, 
                    shuffle_tables=True)
                sub_schema_string = "### DATABASE SCHEMA: \n" + sub_schema_string
                # column_meanings_str = sub_schema_generator.get_schema_column_meanings_string()
                column_meanings_str = sub_schema_generator.get_column_profiles_string(with_keys=False, with_references=False)
                column_values_str = sub_schema_generator.get_random_unique_column_values_string(value_cnt=sample_generator_config['column_value_cnt'])

                # generated_t2s_examples: List[Dict] = self.llm_service.generate_column_focused_text_to_sql_examples(
                #     sub_schema_string=sub_schema_string,
                #     focused_column=focused_column,
                #     sample_count=sample_generator_config['sample_count_edf'],
                #     column_meanings=column_meanings_str,
                #     column_values_str=column_values_str
                # )
                generated_t2s_examples: List[Dict] = self.llm_service.generate_column_focused_sql_to_text_examples(
                    sub_schema_string=sub_schema_string,
                    focused_column=focused_column,
                    sample_count=sample_generator_config['sample_count_edf'],
                    column_meanings=column_meanings_str,
                    column_values_str=column_values_str,
                    eval_in_generation=eval_in_generation
                )

                self.db_logger.info("===== Check Status - Fix - Generate DAC Reasoning =====")
                ### Checking execution status of generated SQL queries and adding into ss_examples_dict
                
                def process_pair_with_print(idx: int, pair: dict) -> tuple[int, dict]:
                    print(f"\n----- Processing Generated T2S Pair idx: {idx}")
                    result = self._process_generated_t2s_pair(
                        pair,
                        db_info,
                        sub_schema_string,
                        original_full_schema_str,
                        column_meanings_str,
                        column_values_str
                    )
                    return idx, result
                
                max_workers = min(len(generated_t2s_examples), 4 * (os.cpu_count() or 4)) # Cannot apply due to Gemini Free-tier limit
                max_workers = 12
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = [
                        executor.submit(
                            process_pair_with_print,
                            idx,
                            t2s_pair
                        )
                        for idx, t2s_pair in enumerate(generated_t2s_examples)
                    ]

                    for future in futures:
                        try:
                            idx, processed_t2s_pair = future.result()
                            processed_t2s_pair = {
                                'ss_id': sub_schema_id,
                                'example_no': idx,
                                **processed_t2s_pair
                            }
                            self.db_logger.info(f"Writing processed T2S pair idx: {idx}")
                            # Synchronized file write
                            with file_lock:
                                with open(sub_schema_examples_jsonl_file_path, 'a') as outfile:
                                    json_line = json.dumps(processed_t2s_pair)
                                    outfile.write(json_line + '\n')

                        except Exception as e:
                            self.db_logger.error(f"Error while processing T2S pair: {e}", exc_info=True)
                
                # ## Serial
                # for sub_t2s_pair_index, t2s_pair in enumerate(generated_t2s_examples):
                #     self.db_logger.info(f"----- Processing Generated T2S Pair ({sub_t2s_pair_index+1}/{len(generated_t2s_examples)})")
                #     try:
                #         processed_t2s_pair = self._process_generated_t2s_pair(
                #             t2s_pair=t2s_pair, 
                #             db_info=db_info, 
                #             sub_schema_str=sub_schema_string, 
                #             original_full_schema_str=original_full_schema_str, 
                #             column_meanings=column_meanings_str, 
                #             column_values_str=column_values_str
                #         )
                #         # ss_examples_dict[sub_schema_id].append(processed_t2s_pair)   # When json is used instead of jsonl
                #         processed_t2s_pair = {'ss_id': sub_schema_id, 'cf_example_no': sub_t2s_pair_index, 'focused_column': str(focused_column), **processed_t2s_pair}
                #         # Write to .jsonl file
                #         with open(sub_schema_examples_jsonl_file_path, 'a') as outfile:
                #             json_line = json.dumps(processed_t2s_pair)
                #             outfile.write(json_line + '\n')

                #     except Exception as e:
                #         continue
                duration = time.time() - gen_start_time
                self.db_logger.info(f"- Colum Focused Text-to-SQL examples for sub-schema whose id is {sub_schema_id} are generated in {duration} seconds and saved to {sub_schema_examples_jsonl_file_path}.")
                
                if idx == len(ss_for_column_focused_generation)-1:
                    proceed_next_step = True

        except Exception as e:
            self.db_logger.error(f"Column focused example generation process is stopped because of an error: {e}")
            if hasattr(e, "args") and e.args:
                error_message = str(e.args[0])
                if "RESOURCE_EXHAUSTED" in error_message:
                    raise

        
        ###########################################
        ##### STEP : RE-ANALYZE COLUMN COUNTS #####
        ###########################################
        ## Analyzing the column counts in synthetically generated examples
        tracker2 = self._column_count_analysis(db_info=db_info, jsonl_path=sub_schema_examples_jsonl_file_path)
        tracker2.write(dir=db_info.sub_schemas_dir, file_name="data_columns_counts_synthetic_t2s.json")
        
        return self._generate_column_focused_t2s_examples(db_info=db_info, tracker=tracker2)
    

    def _corect_duplicate_ids(self, db_info: DatabaseGeneralInfo):
        """
        Fixes duplicate (ss_id, example_no) pairs in the JSONL file due to synthetic 
        generation from sub-schemas. Ensures uniqueness by incrementing example_no 
        for duplicate (ss_id, example_no) pairs.

        Args:
            db_info (DatabaseGeneralInfo): Information about the database, including paths.

        Returns:
            bool: True if correction is applied and file is rewritten, False otherwise.
        """


        sample_generator_config = self.args.config['preprocess']['sample_generator']
        granularity_level = sample_generator_config['granularity_level']

        sub_schema_examples_jsonl_file_path = db_info.db_prep_dir / "sub_schemas" / f"{granularity_level}_level" / f"sub_schema_examples.jsonl"
        if not sub_schema_examples_jsonl_file_path.exists():
            return False
        
        # load data
        synthetic_t2s_data = []
        with open(sub_schema_examples_jsonl_file_path, 'r') as f:
            for line in f:
                try:
                    synthetic_t2s_obj = json.loads(line)
                    synthetic_t2s_data.append(synthetic_t2s_obj)
                except Exception:
                    continue

        # Track example_nos for each ss_id
        ssid_to_example_nos = defaultdict(set)
        seen_keys = set()
        corrected_data = []

        for obj in synthetic_t2s_data:
            ss_id = obj['ss_id']
            example_no = obj['example_no']

            key = (ss_id, example_no)
            if key in seen_keys:
                # Assign new example_no for duplicate
                max_example_no = max(ssid_to_example_nos[ss_id]) if ssid_to_example_nos[ss_id] else -1
                new_example_no = max_example_no + 1
                obj['example_no'] = new_example_no
                ssid_to_example_nos[ss_id].add(new_example_no)
                seen_keys.add((ss_id, new_example_no))
            else:
                ssid_to_example_nos[ss_id].add(example_no)
                seen_keys.add(key)
            
            corrected_data.append(obj)

        # Rewrite the file 
        with open(sub_schema_examples_jsonl_file_path, 'w') as f:
            for obj in corrected_data:
                f.write(json.dumps(obj) + '\n')
        
        return True
    
    def _filter_logical_and_syntactically_correct_pairs(self, db_info: DatabaseGeneralInfo):
        """
        Filters the syntactically correct and logical Text-to-SQL pairs
        """

        sample_generator_config = self.args.config['preprocess']['sample_generator']
        granularity_level = sample_generator_config['granularity_level']

        sub_schema_examples_jsonl_file_path = db_info.db_prep_dir / "sub_schemas" / f"{granularity_level}_level" / f"sub_schema_examples.jsonl"
        if not sub_schema_examples_jsonl_file_path.exists():
            return False
        
        # load data
        synthetic_t2s_data = []
        with open(sub_schema_examples_jsonl_file_path, 'r') as f:
            for line in f:
                try:
                    synthetic_t2s_obj = json.loads(line)
                    synthetic_t2s_data.append(synthetic_t2s_obj)
                except Exception:
                    continue
        
        # Copy the raw T2S pairs into sub_schema_examples_raw.jsonl file
        raw_sub_schema_examples_jsonl_file_path = db_info.db_prep_dir / "sub_schemas" / f"{granularity_level}_level" / f"sub_schema_examples_raw.jsonl"
        with open(raw_sub_schema_examples_jsonl_file_path, 'w') as f:
            for obj in synthetic_t2s_data:
                f.write(json.dumps(obj) + '\n')

        # Now filter the logical and syntactically correct T2s pairs
        with open(sub_schema_examples_jsonl_file_path, 'w') as f:
            for obj in synthetic_t2s_data:
                if bool(obj.get("is_logical", False)):
                    if obj.get("execution_status", "") == "SYNTACTICALLY_CORRECT":
                        f.write(json.dumps(obj) + '\n')


    def _load_sub_schema_examples(self, ss_examples_json_file_path: Path) -> str:
        """
        Loading text-to-sql pairs for the sub-schema
        Args:
            ss_examples_json_file_path (Path): The path to the JSON file that include examples for the specific sub_schema
        Returns:
            str: Single string containing text-to-sql pair examples for the sub-schema
        """
        if ss_examples_json_file_path.exists():
            try:
                with open(ss_examples_json_file_path, 'r') as file:
                    sub_schema_examples_dict = json.load(file)
            except:
                print(f"Couldn't read the sub-schema examples file {ss_examples_json_file_path}")
                sub_schema_examples_dict = {}
        else:
            print(f"Couldn't find such file {ss_examples_json_file_path}")
            sub_schema_examples_dict = {}
        
        t2s_pair_examples = ""
        for example_no_str, example_dict in sub_schema_examples_dict.items():
            t2s_pair_examples += f"# Question: {example_dict.get('question', '')}\n"
            t2s_pair_examples += f"# Reasoning for Corresponding SQL: {example_dict.get('chain_of_thought_reasoning', '')}\n"
            t2s_pair_examples += f"# SQL: {example_dict.get('SQL', '')}\n\n"

        return t2s_pair_examples
    
    def _create_db_completion_dataset(self, db_info: DatabaseGeneralInfo, data_threshold_cnt: int = 5000):
        """"
        Creates a dataset for LLM training. 
        The task will be asking LLM to complete missing tables and columns given some portion of database.
        """
        db_completion_dataset_configs = self.args.config['preprocess']['db_completion_dataset']['db_completion_configs']
        N_list = db_completion_dataset_configs[self.args.data_mode][db_info.db_id]['N']
        M_list = db_completion_dataset_configs[self.args.data_mode][db_info.db_id]['M']
        
        database_manager = DatabaseManager(dataset=self.args.dataset, db_mode=self.args.data_mode, db_id=db_info.db_id)
        db_original_schema_dict=DatabaseManager().get_db_schema()

        schema_with_descriptions = load_tables_description(db_directory_path=db_info.db_directory, use_value_description=True)
        db_schema_generator = DatabaseSchemaGenerator(
            tentative_schema=DatabaseSchema.from_schema_dict(db_original_schema_dict),
            schema_with_descriptions=DatabaseSchema.from_schema_dict_with_descriptions(schema_with_descriptions),
            db_id=db_info.db_id, 
            db_path=db_info.db_path)
        
        db_tables = list(db_original_schema_dict.keys())
        db_table_count = len(db_tables) 
        db_table_column_counts = {table: len(columns) for table, columns in db_original_schema_dict.items()}

        schemas_with_missing_parts: List[Dict[str, List[str]]] = [] 
        schema_missing_parts: List[Dict[str, List[str]]] = []


        ## Construct tables_n_combinations and tables_columns_m_combinations
        select_table_cnt = N_list
        self.db_logger.info(f"db_tables: {db_tables}")
        self.db_logger.info(f"select_table_cnt: {select_table_cnt}")
        tables_n_combinations = get_combinations(db_tables, samples=select_table_cnt)
        self.db_logger.info(f"{len(tables_n_combinations)} number of table combinations (with {select_table_cnt} number of tables) exist.")
        tables_columns_m_combinations = {}
        for table_name, table_columns in db_original_schema_dict.items():
            # select_column_cnt = max(2, round(len(table_columns) * M)) # this cause lots of combinations especially for the tables having large number of colums
            select_column_cnt = M_list
            tables_columns_m_combinations[table_name] = get_combinations(table_columns, samples=select_column_cnt)
            self.db_logger.info(f"For {table_name} {len(tables_columns_m_combinations[table_name])} number of columns combinations (with {select_column_cnt} number of columns) exist.")

        ##### ===== SUB-STEP 1: Column Completion ===== ######
        ## 1. Get combinations of N percent of tables from the schema 
        ## 2. For each selected tables, select combinations of M percent of columns 
        ## 3. Remove the selected columns only. 
        ## 4. It will be expected that removed columns will be generated by LLM
        column_completion_task_count = 0
        for selected_tables in tables_n_combinations:
            ## selecting combination count according to column having max number of combinations
            combination_cnt = 0
            for table in selected_tables:
                cnt = len(tables_columns_m_combinations[table])
                if cnt > combination_cnt:
                    combination_cnt = cnt

            for i in range(combination_cnt):
                missing_part_item_dict = {}
                for table in selected_tables:
                    combination_len = len(tables_columns_m_combinations[table])
                    index = i % combination_len
                    missing_part_item_dict[table] = tables_columns_m_combinations[table][index]
                
                schema_missing_parts.append(missing_part_item_dict)
                column_completion_task_count += 1

            
        self.db_logger.info(f"--{column_completion_task_count} number of table completion task available.")
        ## Setting a threshold for the large number of data
        if len(schema_missing_parts) > data_threshold_cnt:
            schema_missing_parts = random.sample(schema_missing_parts, k=data_threshold_cnt)
        
        ##### ===== SUB-STEP 2: Table Completion ===== ######
        ## 1. Randomly select N percent of tables from the schema (N will change according to table count)
        ## 2. Remove the selected tables. 
        ## 3. It is expected that removed tables with its columns will be generated by LLM. 
        
        for selected_tables in tables_n_combinations:
            missing_part_item_dict = {}
            for table in selected_tables:
                missing_part_item_dict[table] = db_original_schema_dict[table]
            
            schema_missing_parts.append(missing_part_item_dict)
        
        self.db_logger.info(f"--{len(tables_n_combinations)} number of table completion task.")

        ##### ===== Constructing Incomplete Schemas by Removing Missing Pargs ===== ######
        # incomplete_schemas = []
        # for missing_parts_dict in schema_missing_parts:
        #     schema = db_original_schema_dict.copy()
        #     # removing columns and tables
        #     for table_name, column_list in missing_parts_dict.items():
        #         for column_name in column_list:
        #             try:
        #                 schema[table_name].remove(column_name)
        #             except ValueError as ve:
        #                 print(f"ValueError: {ve}")
        #                 pass
        #         # after removing columns, if there is no remaining columns, then get rid of the table
        #         if len(schema[table_name]) == 0:
        #             schema.pop(table_name, None)
            
        #     incomplete_schemas.append(schema)
        
        ##### ===== Saving into file ===== ######
        ## Saving generated dataset into a file
        completion_dataset_dir = db_info.db_prep_dir / "db_completion"
        completion_dataset_json_path = completion_dataset_dir / "schema_missing_parts.json"

        completion_dataset_dir.mkdir(parents=True, exist_ok=True)
        with open(completion_dataset_json_path, "w", encoding="utf-8") as f:
            json.dump(schema_missing_parts, f, indent=4)
            


        self.db_logger.info(f"--{len(schema_missing_parts)} number of database completion case generated and saved into {completion_dataset_json_path}")

        ## Tracking the counts of columns
        tracker = DatabaseDataTracker(db_id=db_info.db_id, db_path=db_info.db_path)
        for missing_parts_dict in schema_missing_parts:
            for table_name, column_names in missing_parts_dict.items():
                for column_name in column_names:
                    tracker.increase_column_count(table_name, column_name)

        tracker.write(dir=completion_dataset_dir, file_name="data_columns_counts_db_completion.json")

        return
    
    def _split_t2s_dataset(self, db_info: DatabaseGeneralInfo) -> bool:
        """
        Splits the database into train-dev-test
        """
        split_ratio = 0.05 # 90% train, 5% dev, 5% test
        seed = self.args.config['seed']

        sample_generator_config = self.args.config['preprocess']['sample_generator']
        granularity_level = sample_generator_config['granularity_level']

        sub_schema_examples_jsonl_file_path = db_info.db_prep_dir / "sub_schemas" / f"{granularity_level}_level" / f"sub_schema_examples.jsonl"
        if not sub_schema_examples_jsonl_file_path.exists():
            return False
        
        # load data
        synthetic_t2s_data = []
        with open(sub_schema_examples_jsonl_file_path, 'r') as f:
            for line in f:
                try:
                    synthetic_t2s_obj = json.loads(line)
                    synthetic_t2s_data.append(synthetic_t2s_obj)
                except Exception:
                    continue

        if not synthetic_t2s_data:
            self.db_logger.info("No Synthetic Text-to-SQL data found.")
            return False
        
        # Step 2: Group by ss_id
        ss_id_groups: Dict[str, List[dict]] = defaultdict(list)
        for item in synthetic_t2s_data:
            ss_id = item.get("ss_id")
            if ss_id:
                ss_id_groups[ss_id].append(item)

        # Step 3: Select one from each ss_id group
        selected_for_dev_test = []
        for ss_id, group in ss_id_groups.items():
            selected_example = random.choice(group)
            selected_for_dev_test.append(selected_example)

        # Step 4: Split selected samples equally into dev and test
        random.shuffle(selected_for_dev_test)
        half = len(selected_for_dev_test) // 2
        dev_data = selected_for_dev_test[:half]
        test_data = selected_for_dev_test[half:]

        # Step 5: Remaining data (excluding selected_for_dev_test) → train
        selected_ids = {id(example) for example in selected_for_dev_test}
        train_data = [item for item in synthetic_t2s_data if id(item) not in selected_ids]

        # Save splits
        split_dir = sub_schema_examples_jsonl_file_path.parent
        train_file = split_dir / "sub_schema_examples_train.jsonl"
        dev_file = split_dir / "sub_schema_examples_dev.jsonl"
        test_file = split_dir / "sub_schema_examples_test.jsonl"

        with open(train_file, 'w') as f:
            for item in train_data:
                f.write(json.dumps(item) + "\n")

        with open(dev_file, 'w') as f:
            for item in dev_data:
                f.write(json.dumps(item) + "\n")

        with open(test_file, 'w') as f:
            for item in test_data:
                f.write(json.dumps(item) + "\n")

        self.db_logger.info("Train-Dev-Test splits are constructed.")

        return True
        

    def _add_foreign_keys_to_db(self, db_id: str, dbs_root_dir:Union[Path, str], fk_specs: List[Dict], run_fk_check: bool = True):
        """"
        Updates the database sqlite file by adding missing Foreing Keys

        Args:
            db_id (str): database name
            dbs_root_dir (str): root directory to access databases
            run_fk_check (bool): after patching, run PRAGMA foreign_key_check and print problems
        """

        def is_fk_already_present(conn: sqlite3.Connection, table:str, col:str, ref_table: str, ref_col: str) -> bool:
            """
            Check whether (table.col) already references (ref_table.ref_col).
            """
            rows = conn.execute(f'PRAGMA foreign_key_list("{table}");').fetchall()
            return any(r[3] == col and r[2] == ref_table and r[4] == ref_col for r in rows)
        
        def inject_fk_lines(create_sql: str, fk_lines: List[str]) -> str:
            """
            Insert FK lines just before the final ')' of a CREATE TABLE statement.
            """
            create_sql = create_sql.strip()
            # remove trailing ';' if present to be able to work consistently
            if create_sql.endswith(';'):
                create_sql = create_sql[:-1]
            
            # index of the last right-paren that closes the column/constraint list
            idx = create_sql.rfind(')')
            before, after = create_sql[:idx], create_sql[idx:]

            # ensure we end the definition list with a comma
            if not before.rstrip().endswith(','):
                before += ','

            fk_block = "\n    " + ",\n    ".join(fk_lines)
            return before + fk_block + after + ';'

        db_path = Path(dbs_root_dir) / db_id / f"{db_id}.sqlite"
        
        conn = sqlite3.connect(db_path) # Connect to DB
        conn.row_factory = sqlite3.Row # Configures the connection so that rows returned from queries behave like dictionaries.
        cur = conn.cursor() # Creates a cursor object used to execute SQL commands and fetch results.
        cur.execute("PRAGMA foreign_keys=OFF;") # PRAGMA command that turns off enforcement of foreign key constraints. Disabling foreign keys allows you to: Insert or delete data that would normally violate foreign key constraints. Modify the database schema (e.g., dropping or recreating tables) without constraint errors.

        # --- Print Initial DDL statements for all tables ---
        print("="*100)
        print("\n Initial DDL commands in the database:")
        cur.execute("""
            SELECT name, sql FROM sqlite_master
            WHERE type='table' AND name NOT LIKE 'sqlite_%'
            ORDER BY name;
        """)
        for name, sql in cur.fetchall():
            print(f"\n-- Table: {name} --")
            print(sql + ";")

        # Group specs per referencing table
        grouped = defaultdict(list)
        for spec in fk_specs:
            src = spec["referencing_column"]
            tgt = spec["referenced_column"]
            grouped[src["table"]].append((src["column"], tgt["table"], tgt["column"]))

        for table, constraints in grouped.items():
            # filter out the ones that are already there
            missing = [(c, rt, rc) for c, rt, rc in constraints 
                       if not is_fk_already_present(conn, table, c, rt, rc)]
            
            if not missing:
                continue # table is already OK

            # Capture original CREATE statemtn *before* renaming
            create_sql, = cur.execute(
                "SELECT sql FROM sqlite_master WHERE type='table' AND name=?;",
                (table,)
            ).fetchone()

            # Build FK lines for the missing constraints
            fk_lines = [f'FOREIGN KEY("{col}") REFERENCES "{ref_table}"("{ref_col}")'
                        for col, ref_table, ref_col in missing]
            patched_sql = inject_fk_lines(create_sql, fk_lines)

            # Create a **temporary replacement** table
            tmp_name = f"{table}_fkpatch_new"
            patched_sql_tmp = re.sub(
                rf'CREATE TABLE\s+["`]?{re.escape(table)}["`]?',
                f'CREATE TABLE "{tmp_name}"',
                patched_sql,
                count=1,
                flags=re.IGNORECASE
            )
            cur.execute(patched_sql_tmp)

            # Copy data
            cur.execute(f'INSERT INTO "{tmp_name}" SELECT * FROM "{table}";')

            # Drop original & rename replacement
            cur.execute(f'DROP TABLE "{table}";')
            cur.execute(f'ALTER TABLE "{tmp_name}" RENAME TO "{table}";')

            # # Rebuild table ------------------------------------------------------
            # tmp_name = f"{table}_old_for_fk_patch"
            # cur.execute(f'ALTER TABLE "{table}" RENAME TO "{tmp_name}";')
            # cur.execute(new_create_sql)
            # cur.execute(f'INSERT INTO "{table}" SELECT * FROM "{tmp_name}";')
            # cur.execute(f'DROP TABLE "{tmp_name}";')
            # # --------------------------------------------------------------------

        conn.commit()
        cur.execute("PRAGMA foreign_keys=ON;")

        # --- Print final DDL statements for all tables ---
        print("="*100)
        print("\n Final DDL commands in the database:")
        cur.execute("""
            SELECT name, sql FROM sqlite_master
            WHERE type='table' AND name NOT LIKE 'sqlite_%'
            ORDER BY name;
        """)
        for name, sql in cur.fetchall():
            print(f"\n-- Table: {name} --")
            print(sql + ";")

        # if run_fk_check:
        #     problems = cur.execute("PRAGMA foreign_key_check;").fetchall()
        #     if problems:
        #         print(f"⚠️⚠️⚠️ {len(problems)} Foreign-key violations detected:")
        #         for row in problems:
        #             print("⚠️", row)
        #     else:
        #         print("✅ All foreign-key constraints satisfied.")

        conn.close()

    def preprocess_dbs(self):
        """
        Preprocessing Databases:
        1. Creating Sub-Schemas
        2. Generating Text-to-SQL Examples for the Sub-Schemas
        3. Constructing Database Completion Dataset
        
        """
        DBS_ROOT_DIR = Path(self.args.dbs_root_dir)
        databases = self.db_ids
        # databases = databases[:1]                         #### IMPORTANT: REMOVE [:1]
        for db_id in databases:
            if db_id not in self.args.db_ids:
                continue
            preprocessing_start_time = time.time()
            # Create preprocess directory
            prep_dir_name: Path = DBS_ROOT_DIR / db_id / self.args.config['prep_dir_name']
            prep_dir_name.mkdir(parents=True, exist_ok=True)
            # Save config
            prep_config_dir: Path = prep_dir_name / "config" 
            prep_config_dir.mkdir(parents=True, exist_ok=True)
            prep_config_path = prep_config_dir / "config.json"
            with open(prep_config_path, 'w') as file:
                json.dump(self.args.config, file, indent=4)
            
            

            ###########################################
            #### STEP 0: ADD MISSING FOREIGN KEYS #####
            ###########################################
            fk_specs_for_dbs = {
                "card_games": [
                    {
                        "referencing_column": {"table": "cards", "column": "setCode"}, 
                        "referenced_column": {"table": "sets", "column": "code"}
                    }
                ]
            }
            print(f"*** STEP 0 START: {db_id.upper()} ADD MISSING FOREIGN KEYS ***")
            add_fk_start_time = time.time()
            if db_id in fk_specs_for_dbs:
                fk_specs = fk_specs_for_dbs.get(db_id, [])
                self._add_foreign_keys_to_db(db_id=db_id, dbs_root_dir=DBS_ROOT_DIR, fk_specs=fk_specs)
            add_fk_duration = time.time() - add_fk_start_time
            print(f"*** STEP 0 END: {db_id.upper()} ADD MISSING FOREIGN KEYS ***")
            print(f"Duration: {add_fk_duration}")

            # Create DatabaseGeneralInfo instance
            db_info = DatabaseGeneralInfo(db_id=db_id, dbs_root_dir=DBS_ROOT_DIR)
            
            # create db_logger
            db_logger = logging.getLogger(db_id)
            db_logger.setLevel(logging.INFO)
            logger_path = Path(f"logs/prep_{self.args.run_start_time}/{db_id}_prep.log")
            logger_path.parent.mkdir(parents=True, exist_ok=True)
            db_logger_handler = logging.FileHandler(logger_path)
            db_logger_handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
            db_logger.addHandler(db_logger_handler)

            self.db_logger = db_logger
            self.llm_service.set_logger(logger=db_logger)

            # self.db_logger.info(f"===== DATABASE ({db_id}) SCHEMA =====")
            self.db_logger.info(f"""{db_info.original_db_schema_generator.generate_schema_string(
                include_column_value_examples=True,
                include_value_description=True,
                shuffle_cols=False,
                shuffle_tables=False
                )
            }""")
            # self.db_logger.info(f"===== DATABASE ({db_id}) COLUMN PROFILES =====")
            original_full_db_col_profiles_str = db_info.original_db_schema_generator.get_column_profiles_string(with_keys=False, with_references=False)
            self.db_logger.info(original_full_db_col_profiles_str)
            
            ########################################
            #### STEP 1: CONSTRUCT SUB-SCHEMAS #####
            ########################################
            db_logger.info(f"*** STEP 1 START: {db_id.upper()} SUB-SCHEMA CONSTRUCTION ***")
            ss_gen_start_time = time.time()
            self._construct_db_sub_schemas(db_info)
            ss_gen_duration = time.time() - ss_gen_start_time
            db_logger.info(f"*** STEP 1 END: {db_id.upper()} SUB-SCHEMA CONSTRUCTION ***")
            db_logger.info(f"Duration: {ss_gen_duration}")

            #################################################################
            #### STEP 2.1: GENERATE TEXT2SQL EXAMPLES FOR EACH SUB-SCHEMA #####
            #################################################################
            try:
                db_logger.info(f"*** STEP 2.1 START: {db_id.upper()} SUB-SCHEMA TEXT2SQL EXAMPLE GENERATION ***")
                t2s_example_gen_start_time = time.time()
                proceed = self._generate_ss_t2s_examples(db_info)
                t2s_example_gen_duration = time.time() - t2s_example_gen_start_time
                db_logger.info(f"*** STEP 2.1 END: {db_id.upper()} SUB-SCHEMA TEXT2SQL EXAMPLE GENERATION ***")
                db_logger.info(f"Duration: {t2s_example_gen_duration}")
            except Exception as e:
                db_logger.error(f"Error occured: {e}")
                proceed = False

            ############################################################
            #### STEP 2.2: GENERATE COLUMN FOCUSED TEXT2SQL EXAMPLES #####
            ############################################################
            if proceed:
                db_logger.info(f"*** STEP 2.2 START: {db_id.upper()} COLUMN FOCUSED TEXT2SQL EXAMPLE GENERATION ***")
                cf_t2s_example_gen_start_time = time.time()
                proceed = self._generate_column_focused_t2s_examples(db_info)
                cf_t2s_example_gen_duration = time.time() - cf_t2s_example_gen_start_time
                db_logger.info(f"*** STEP 2.2 END: {db_id.upper()} COLUMN FOCUSED TEXT2SQL EXAMPLE GENERATION ***")
                db_logger.info(f"Duration: {cf_t2s_example_gen_duration}")
            else:
                db_logger.error(f"Cannot proceed with column focused text-to-sql example generation process, since fundamental text-to-sql generation has not been completed yet, i.e. data is not generated for each sub-schema yet.")
                proceed = False


            ############################################################
            #### STEP 2.3: CORRECT DUBLICATE SS_ID & EXAMPLE_NO  #####
            ############################################################
            if proceed:
                db_logger.info(f"*** STEP 2.3 START: {db_id.upper()} CORRECTING DUPLICATES IDS ***")
                duplicati_id_correction_start_time = time.time()
                self._corect_duplicate_ids(db_info)
                db_logger.info(f"*** STEP 2.3 END: {db_id.upper()} CORRECTING DUPLICATES IDS ***")
                duplicate_id_correction_duration = time.time() - duplicati_id_correction_start_time
                db_logger.info(f"Duration: {duplicate_id_correction_duration}")

            ############################################################
            #### STEP 2.4: FILTER LOGICAL AND SYNTACTICALLY CORRECT PAIRS   #####
            ############################################################
            if proceed:
                db_logger.info(f"*** STEP 2.4 START: {db_id.upper()} FILTERING LOGICAL AND SYNTACTICALLY CORRECT PAIRS  ***")
                f_start_time = time.time()
                self._filter_logical_and_syntactically_correct_pairs(db_info)
                db_logger.info(f"*** STEP 2.4 END: {db_id.upper()} FILTERING LOGICAL AND SYNTACTICALLY CORRECT PAIRS ***")
                f_duration = time.time() - f_start_time
                db_logger.info(f"Duration: {f_duration}")

            ############################################################
            #### STEP 2.5: SPLITTING TEXT2SQL EXAMPLES #####
            ############################################################
            # if proceed:
            #     db_logger.info(f"*** STEP 2.5 START: {db_id.upper()} SPLITTING TEXT2SQL EXAMPLES ***")
            #     splitting_start_time = time.time()
            #     proceed = self._split_t2s_dataset(db_info)
            #     splitting_duration = time.time() - splitting_start_time
            #     db_logger.info(f"*** STEP 2.5 END: {db_id.upper()} SPLITTING TEXT2SQL EXAMPLES ***")
            # else:
            #     db_logger.error(f"Cannot proceed with `splitting text-to-sql examples` process, since column focused text-to-sql generation has not been completed yet.")
            #     proceed = False


            ############################################################
            ##### STEP 4: CONSTRUCTING DATABASE COMPLETION DATASET #####
            ############################################################
            if proceed:
                db_logger.info(f"*** STEP 4 START: {db_id.upper()} DATABASE COMPLETION DATASET GENERATION ***")
                db_completion_dataset_construction_start_time = time.time()
                self._create_db_completion_dataset(db_info)
                db_completion_dataset_construction_duration = time.time() - db_completion_dataset_construction_start_time
                db_logger.info(f"*** STEP 4 END: {db_id.upper()} DATABASE COMPLETION DATASET GENERATION ***")
                db_logger.info(f"Duration: {db_completion_dataset_construction_duration}")
            else:
                db_logger.error(f"Cannot proceed with database completion dataset generation process, since previous steps have not beed completed yet.")



            preprocessing_duration = time.time() - preprocessing_start_time
            print(f"Preprocessing {db_id.upper()} duration: {preprocessing_duration }")
            print("\n\n\n")

        return
    
    def preprocess_db_values(self) -> None:
        """"
        Creating LSH for the database values
        """
        db_values_preprocess_config = self.args.config['preprocess']['db_values_preprocess']
        print("db_values_preprocess_config: ", db_values_preprocess_config)

        DBS_ROOT_DIR = Path(self.args.dbs_root_dir)
        databases = self.db_ids
        # databases = databases[:1]                         #### IMPORTANT: REMOVE [:1]
        for db_id in databases:
            if db_id not in self.args.db_ids:
                continue
            preprocessing_start_time = time.time()

            db_directory_path = DBS_ROOT_DIR / db_id
            make_db_lsh(db_directory_path=db_directory_path,
                        prep_dir_name=str(self.args.config['prep_dir_name']),
                        signature_size=int(db_values_preprocess_config['signature_size']), 
                        n_gram=int(db_values_preprocess_config['n_gram']),
                        threshold=int(db_values_preprocess_config['threshold']),
                        )
            
            duration = time.time() - preprocessing_start_time
            print(f"LSH for {db_id} created in {duration} seconds.")
    
    def preprocess_column_meanings(self)-> None:
        """
        Creating new Dictionary(Dict[str, Dict[str, Dict[str, str]]]) for column meanings and saves in the same directory as processed_column_meanings
        """
        column_meaning_path = self.args.column_meaning_path
        processed_column_meaning_path = self.args.processed_column_meaning_path
        
        if processed_column_meaning_path.exists():
            print("Column meanings have already been processed.")
            return

        if not column_meaning_path.exists():
            print("Column meaning document is missing.")
            return
        
        with open(column_meaning_path, 'r', encoding='utf-8') as file:
            column_meaning_dict = json.load(file)

        processed_column_meaning = {}
        for db_id_table_column, column_meaning in column_meaning_dict.items():
            column_info = db_id_table_column.split("|")
            db_id = column_info[0].strip()
            table_name = column_info[1].strip()
            column_name = column_info[2].strip()
            
            if not processed_column_meaning.get(db_id, None):
                processed_column_meaning[db_id] = {}
                processed_column_meaning[db_id][table_name] = {}
                processed_column_meaning[db_id][table_name][column_name] = column_meaning
            else:
                if not processed_column_meaning[db_id].get(table_name, None):
                    processed_column_meaning[db_id][table_name] = {}
                    processed_column_meaning[db_id][table_name][column_name] = column_meaning
                else:
                    processed_column_meaning[db_id][table_name][column_name] = column_meaning
        
        
        with open(processed_column_meaning_path, 'w', encoding='utf-8') as file:
            json.dump(processed_column_meaning, file, indent=4)
        
        print("Column meaning document processed")
        return