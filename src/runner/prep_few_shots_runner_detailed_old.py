""""
Before the training or evaluation, this is used to extracts the few-shots, using vector database
in order not to increase the train/eval time.
"""

import json
import time
from typing import Any, List, Dict, Union
from pathlib import Path
import logging

from utils.vdb_utils.T2SSyntheticVDBService import T2SSyntheticVDBService
from utils.llm_utils.PreprocessLLMService import PreprocessLLMService
from utils.eval_utils.eval_utils import calculate_schema_metrics_for_single_schema
from utils.db_utils.sql_parser import get_sql_columns_dict, get_filtered_schema_dict_from_similar_examples
from utils.llm_utils.prompt_utils import load_template
from utils.db_utils.db_info_utils import get_db_schema
from utils.db_utils.schema import DatabaseSchema
from utils.db_utils.db_info import DatabaseGeneralInfo
from utils.db_utils.helper import get_combinations
from utils.db_utils.entity_retrieval import EntityRetrieval

class PrepFewShotsRunnerDetailedOld:

    def __init__(self, args: Any):
        self.args = args
        self.prep_few_shots_configs = self.args.config['prep_few_shots']
        self.db_ids: List[str] = self.prep_few_shots_configs['db_ids']

        # Set logger
        logger = logging.getLogger('eval')
        logger.setLevel(logging.INFO)
        logger_path = Path(f"logs/prep_few_shots_{self.args.run_start_time}/prep_few_shots_logs.log")
        logger_path.parent.mkdir(parents=True, exist_ok=True)
        logger_handler = logging.FileHandler(logger_path)
        logger_handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
        logger.addHandler(logger_handler)
        self.prep_few_shot_logger = logger

        # Initialize vector database service
        self.vdb_service = self._construct_vdb_service()
    
    def _load_dataset(self, data_path: Union[Path, str]) -> List[Dict[str, Any]]:
        """
        Loads the dataset from the specified path.

        Args:
            data_path (str): Path to the data file.

        Returns:
            List[Dict[str, Any]]: The loaded dataset.
        """
        print(f"Loading data from {data_path}...")
        data_path = Path(data_path)
        if not data_path.exists():
            raise FileNotFoundError(f"The specified data path does not exist: {data_path}")

        if data_path.suffix == '.json':
            with open(data_path, 'r') as file:
                dataset = json.load(file)

        elif data_path.suffix == '.jsonl':
            dataset = []
            with open(data_path, 'r') as file:
                for line in file:
                    dataset.append(json.loads(line.strip()))
        
        else:
            raise ValueError(f"Unsupported file extension: {data_path.suffix}. Supported extensions are .json and .jsonl")

        print("Data is loaded.")
        return dataset

    def _write_dataset(self, data, data_path: Union[str, Path]) -> None:
        """
            Writes the dataset to the specified path in either JSON or JSONL format.

            Args:
                data (list): The dataset to write.
                data_path (Union[str, Path]): The path to the output file. Must end with .json or .jsonl.

            Raises:
                ValueError: If the file extension is not supported.
        """
        print("Writing data...")
        data_path = Path(data_path)
        data_path.parent.mkdir(parents=True, exist_ok=True)  # Ensure directory exists

        if data_path.suffix == ".json":
            with open(data_path, 'w') as f:
                json.dump(data, f, indent=4)

        elif data_path.suffix == ".jsonl":
            with open(data_path, 'w') as f:
                for item in data:
                    f.write(json.dumps(item) + "\n")
                    
        else:
            raise ValueError(f"Unsupported file extension: {data_path.suffix}. Supported extensions are .json and .jsonl")
        
        print(f"Data is written to {data_path}")
        return
    
    def _construct_vdb_service(self):
        """
        Initialize a T2SSyntheticVDBService object instance
        """
        print("Constructing VDB service...")
        prep_few_shots_configs = self.args.config['prep_few_shots']
        self.embedding_model_provider = str(prep_few_shots_configs['embedding_model_provider'])
        self.embedding_model_name_or_path = str(prep_few_shots_configs['embedding_model_name_or_path'])
        keyword_extraction_llm_model = str(prep_few_shots_configs['keyword_extraction_llm_model'])

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
            logger=self.prep_few_shot_logger
        )
        print("VDB service is constructed.")
        return vdb_service
    

    def prep_few_shots_for_single_t2s(self, t2s_dict: Dict[str, Any], mode: str) -> Dict[str, Any]: 
        
        k = self.args.config['prep_few_shots']['search_k']
        if mode == "eval":
            db_id = t2s_dict.get("db_id")
        elif mode == "train":
            ss_id = t2s_dict.get("ss_id")
            db_id = str(ss_id.split("-")[0]).strip()
        else:
            raise ValueError(f"Wrong mode. It can be either 'train' or 'eval'. ")
        question = t2s_dict.get("question")
        hint = t2s_dict.get("evidence", "")
        gt_sql = t2s_dict.get("SQL")
        question_and_hint = question + " Hint: " + hint if hint else question
        db_path = Path(self.args.dbs_root_dir) / db_id / f"{db_id}.sqlite"
        db_info: DatabaseGeneralInfo = DatabaseGeneralInfo(dbs_root_dir = self.args.dbs_root_dir, db_id=db_id)

        gt_schema_dict = get_sql_columns_dict(db_path=db_path, sql=gt_sql)

        question_keywords = self.vdb_service.extract_question_keywords(question_and_hint=question_and_hint)
        t2s_dict["question_keywords"] = question_keywords

        keyword_pair_combinations = get_combinations(list_of_items=question_keywords, samples=[2])
        keyword_pair_combinations = [ f"{ks[0]} and {ks[1]}" for ks in keyword_pair_combinations]
        print(f"keyword_pair_combinations: {keyword_pair_combinations}")


        if "few_shot" not in t2s_dict:
            t2s_dict["few_shot"] = {
                "question_and_hint_search": {
                    "examples": [],
                    "schema_recall": [],
                    "schema_precision": []
                },
                "keyword_pair_combinations_on_question":{
                    "examples": {},
                    "schema_recall": [],
                    "schema_precision": []
                },
                "keyword_pair_combinations_on_sql":{
                    "examples": {},
                    "schema_recall": [],
                    "schema_precision": []
                },
                "search_combination_all": {
                    "examples": [],
                    "schema_recall": [],
                    "schema_precision": []
                }
            }

        ##############################
        ### 1) question_and_hint search on questions
        ##############################
        
        if not t2s_dict['few_shot']['question_and_hint_search']['examples']:
            start_time = time.time()
            similar_synthetic_examples_1 = []
            similar_synthetic_examples_1 = self.vdb_service.content_based_example_search_over_vdb(
                db_id=db_id,
                query=question_and_hint,
                content_type="question",
                k=k,
            )
            print(f"{len(similar_synthetic_examples_1)} examples are found for question_and_hint search on questions")
            schema_dict_1 = get_filtered_schema_dict_from_similar_examples(
                db_path=db_path, 
                similar_examples=similar_synthetic_examples_1
            )
            num_tp, num_fp, num_fn, s_recall, s_precision, s_sf1 = calculate_schema_metrics_for_single_schema(
                used_schema_dict=schema_dict_1, 
                gt_schema_dict=gt_schema_dict
            )

            t2s_dict["few_shot"]["question_and_hint_search"]["examples"] = [{"ss_id":item.get("ss_id"), "example_no":item.get("example_no"), "question": item.get("question"), "SQL": item.get("SQL")} for item in similar_synthetic_examples_1]
            t2s_dict["few_shot"]["question_and_hint_search"]["schema_recall"] = s_recall
            t2s_dict["few_shot"]["question_and_hint_search"]["schema_precision"] = s_precision

            duration = time.time() - start_time
            print(f"Duration for question_and_hint search on questions: {duration} seconds")
        else:
            print(f"question_and_hint search on questions has been performed already.")
            similar_synthetic_examples_1 = t2s_dict["few_shot"]["question_and_hint_search"]["examples"]


        ###########################
        ### 4) keyword_pair_combinations_on_question
        ###########################
        if not t2s_dict['few_shot']['keyword_pair_combinations_on_question']['examples']: 
            start_time = time.time()
            similar_synthetic_examples_4 = []
            for keyword_pair in keyword_pair_combinations:
                extracted_examples = self.vdb_service.content_based_example_search_over_vdb(
                    db_id=db_id,
                    query=keyword_pair,
                    content_type="question",
                    k=k
                )
                t2s_dict["few_shot"]["keyword_pair_combinations_on_question"]["examples"][keyword_pair] = [{"ss_id":item.get("ss_id"), "example_no":item.get("example_no"), "question": item.get("question"), "SQL": item.get("SQL")} for item in extracted_examples]
                similar_synthetic_examples_4 += extracted_examples

            print(f"{len(similar_synthetic_examples_4)} examples are found for keyword_pair_combinations_on_question")
            schema_dict_4 = get_filtered_schema_dict_from_similar_examples(
                db_path=db_path, 
                similar_examples=similar_synthetic_examples_4
            )
            num_tp, num_fp, num_fn, s_recall, s_precision, s_f1 = calculate_schema_metrics_for_single_schema(
                used_schema_dict=schema_dict_4, 
                gt_schema_dict=gt_schema_dict
            )
            
            c_cnt_4 = 0
            for t_name, columns in schema_dict_4.items():
                    c_cnt_4 += len(columns)
            t2s_dict["few_shot"]["keyword_pair_combinations_on_question"]["schema_recall"] = s_recall
            t2s_dict["few_shot"]["keyword_pair_combinations_on_question"]["schema_precision"] = s_precision
            t2s_dict["few_shot"]["keyword_pair_combinations_on_question"]["column_cnt"] = c_cnt_4
            t2s_dict["few_shot"]["keyword_pair_combinations_on_question"]["schema_dict"] = schema_dict_4

            duration = time.time() - start_time
            print(f"Duration for keyword_pair_combinations_on_question: {duration} seconds")
        else:
            print("keyword_pair_combinations_on_question search on questions has been performed already.")
            similar_synthetic_examples_4 = []
            for keyword_pair, examples_dicts in t2s_dict['few_shot']['keyword_pair_combinations_on_question']['examples'].items():
                similar_synthetic_examples_4 += examples_dicts


        ###########################
        ### 5) keyword_pair_combinations_on_sql
        ###########################
        if not t2s_dict['few_shot']['keyword_pair_combinations_on_sql']['examples']: 
            start_time = time.time()
            similar_synthetic_examples_5 = []
            for keyword_pair in keyword_pair_combinations:
                extracted_examples = self.vdb_service.content_based_example_search_over_vdb(
                    db_id=db_id,
                    query=keyword_pair,
                    content_type="sql",
                    k=k
                )
                t2s_dict["few_shot"]["keyword_pair_combinations_on_sql"]["examples"][keyword_pair] = [{"ss_id":item.get("ss_id"), "example_no":item.get("example_no"), "question": item.get("question"), "SQL": item.get("SQL")} for item in extracted_examples]
                similar_synthetic_examples_5 += extracted_examples

            print(f"{len(similar_synthetic_examples_5)} examples are found for keyword_pair_combinations_on_sql")
            schema_dict_5 = get_filtered_schema_dict_from_similar_examples(
                db_path=db_path, 
                similar_examples=similar_synthetic_examples_5
            )
            num_tp, num_fp, num_fn, s_recall, s_precision, s_f1 = calculate_schema_metrics_for_single_schema(
                used_schema_dict=schema_dict_5, 
                gt_schema_dict=gt_schema_dict
            )

            c_cnt_5 = 0
            for t_name, columns in schema_dict_5.items():
                    c_cnt_5 += len(columns)
            t2s_dict["few_shot"]["keyword_pair_combinations_on_sql"]["schema_recall"] = s_recall
            t2s_dict["few_shot"]["keyword_pair_combinations_on_sql"]["schema_precision"] = s_precision
            t2s_dict["few_shot"]["keyword_pair_combinations_on_sql"]["column_cnt"] = c_cnt_5
            t2s_dict["few_shot"]["keyword_pair_combinations_on_sql"]["schema_dict"] = schema_dict_5

            duration = time.time() - start_time
            print(f"Duration for keyword_pair_combinations_on_sql: {duration} seconds")
        else:
            print("keyword_pair_combinations_on_sql search on questions has been performed already.")
            similar_synthetic_examples_5 = []
            for keyword_pair, examples_dicts in t2s_dict['few_shot']['keyword_pair_combinations_on_sql']['examples'].items():
                similar_synthetic_examples_5 += examples_dicts

        ###########################
        ### 7) Extracting Schema Using Prev Steps CombinationOverall
        ###########################
        start_time = time.time()
        synthetic_examples_combination = similar_synthetic_examples_1 +  similar_synthetic_examples_4 + similar_synthetic_examples_5
        unique_examples_all = {}
        for example in synthetic_examples_combination:
            key = f"{example.get('ss_id')}-{example.get('example_no')}"
            if key not in unique_examples_all:
                unique_examples_all[key] = example
        
        similar_synthetic_examples_all = list(unique_examples_all.values())

        schema_dict = get_filtered_schema_dict_from_similar_examples(
            db_path=db_path, 
            similar_examples=similar_synthetic_examples_all
        )
        num_tp, num_fp, num_fn, s_recall, s_precision, s_f1 = calculate_schema_metrics_for_single_schema(
            used_schema_dict=schema_dict, 
            gt_schema_dict=gt_schema_dict
        )

        c_cnt = 0
        for t_name, columns in schema_dict.items():
                c_cnt += len(columns)
        t2s_dict["few_shot"]["search_combination_all"]["examples"] = [{"ss_id":item.get("ss_id"), "example_no":item.get("example_no"), "question": item.get("question"), "SQL": item.get("SQL")}for item in similar_synthetic_examples_all]
        t2s_dict["few_shot"]["search_combination_all"]["schema_recall"] = s_recall
        t2s_dict["few_shot"]["search_combination_all"]["schema_precision"] = s_precision
        t2s_dict["few_shot"]["search_combination_all"]["column_cnt"] = c_cnt
        t2s_dict["few_shot"]["search_combination_all"]["schema_dict"] = schema_dict

        duration = time.time() - start_time
        print(f"Duration for Extracting Schema Using Previous Steps Combination : {duration} seconds")

        ###########################
        ### 8) Filter with LLM
        ###########################
        if not t2s_dict.get("llm_filtered_schema_dict_based_on_examples", None): 
            start_time = time.time()
            schema_filterer_llm_model = self.args.config['prep_few_shots']['schema_filterer_llm_model']
            llm_filtered_schema_dict = {}
            llm_service = PreprocessLLMService(model_name=schema_filterer_llm_model, logger=self.prep_few_shot_logger)
            
            db_full_schema_dict: Dict[str, List[str]] = get_db_schema(db_path=db_path)

            column_selection_reasoning = ""
            for table_name, columns in db_full_schema_dict.items():
                # Filter the table columns using LLM
                filtered_table_columns, reasoning = llm_service.filter_table_based_on_examples(
                    db_path=db_path, 
                    t2s_item=t2s_dict, 
                    t2s_examples=similar_synthetic_examples_all,
                    table_name=table_name
                )
                column_selection_reasoning += f"\n== {table_name} ==\n {reasoning} \n"
                print(f"LLM filtration reasoning: \n {reasoning} \n") # DELETE LATER OF COMMENT OUT LATER
                
                if filtered_table_columns:
                    ## Add all columns that are either PK or FK if not in filtered_table_columns
                    for column_name in columns:
                        is_connection: bool = db_info.original_db_schema_generator.is_connection(table_name=table_name, column_name=column_name)
                        if is_connection and column_name not in filtered_table_columns:
                            filtered_table_columns.append(column_name)

                    llm_filtered_schema_dict[table_name] = filtered_table_columns

            
            print(f"llm_filtered_schema_dict: \n {json.dumps(llm_filtered_schema_dict, indent=4)} \n") # DELETE LATER OF COMMENT OUT LATER

            num_tp, num_fp, num_fn, s_recall, s_precision, s_f1 = calculate_schema_metrics_for_single_schema(
                used_schema_dict=llm_filtered_schema_dict, 
                gt_schema_dict=gt_schema_dict
            )
            c_nct = 0
            for t_name, columns in llm_filtered_schema_dict.items():
                c_nct += len(columns)
            t2s_dict["llm_filtered_schema_dict_based_on_examples"] = {
                "schema_recall": s_recall,
                "schema_precision": s_precision,
                "column_cnt": c_nct,
                "column_selection_reasoning": column_selection_reasoning,
                "schema_dict": llm_filtered_schema_dict,
            }
            duration = time.time() - start_time
            print(f"Duration for Filter with LLM based on examples : {duration} seconds")
        else: 
            print("llm_filtered_schema_dict_based_on_examples search on questions has been performed already.")


        t2s_dict["gt_schema_dict"] = gt_schema_dict
        ###########################
        ### 9) Retrieve Entity via Similarity and LSH
        ###########################
        entity_retrieval = EntityRetrieval(args = self.args, db_id=db_id)
        selected_values, schema_dict_via_values = entity_retrieval.get_similar_entities(keywords=question_keywords)
        schema_dict_via_similar_columns = entity_retrieval.get_similar_columns(keywords=question_keywords, question_and_hint=question_and_hint)
        t2s_dict["entity_retrieval"] = {
            "selected_values": selected_values,
            "schema_dict_via_values": schema_dict_via_values,
            "schema_dict_via_similar_columns": schema_dict_via_similar_columns,
        }
        print(f"schema_dict_via_values: \n{json.dumps(schema_dict_via_values)} \n")
        print(f"schema_dict_via_similar_columns: \n{json.dumps(schema_dict_via_similar_columns)} \n")

        return t2s_dict


    def prep_eval_few_shots(self):

        print("Starting to prepare eval data few shots")
        eval_dataset_path = Path(self.args.data_json_path) 
        
        # Create new file name by adding '_new' before the extension
        eval_with_few_shots_file_name = eval_dataset_path.stem + "_with_few_shots" + eval_dataset_path.suffix
        eval_dataset_with_few_shots_path = eval_dataset_path.parent / eval_with_few_shots_file_name

        if eval_dataset_with_few_shots_path.exists():
            dataset = self._load_dataset(data_path=eval_dataset_with_few_shots_path)
        else:
            dataset = self._load_dataset(data_path=eval_dataset_path)


        eval_dataset_with_few_shots = []
        for idx, eval_t2s_item in enumerate(dataset):
            print(f"--- {idx}/{len(dataset)} --- ")
            db_id = eval_t2s_item.get("db_id")
            if db_id not in self.db_ids:
                eval_dataset_with_few_shots.append(eval_t2s_item)
                continue
            
            new_eval_t2s_item = self.prep_few_shots_for_single_t2s(t2s_dict=eval_t2s_item, mode="eval")
            eval_dataset_with_few_shots.append(new_eval_t2s_item)
        
        self._write_dataset(data = eval_dataset_with_few_shots, data_path=eval_dataset_with_few_shots_path)

        return
    
    def prep_train_few_shots(self):

        print("Starting to prepare training data few shots")
        dbs_root_dir = self.args.dbs_root_dir

        for db_id in self.db_ids:
            db_path = Path(self.args.dbs_root_dir) / db_id / f"{db_id}.sqlite"
            prep_schemaless_dir = Path(self.args.dbs_root_dir) / db_id / str(self.args.config['prep_dir_name'])
            sub_schema_examples_path = prep_schemaless_dir / "sub_schemas" / "column_level" / "sub_schema_examples_train.jsonl"

            sub_schema_examples_with_few_shots_file_name = sub_schema_examples_path.stem + "_with_few_shots" + sub_schema_examples_path.suffix
            sub_schema_examples_with_few_shots_path = sub_schema_examples_path.parent / sub_schema_examples_with_few_shots_file_name

            data: List[Dict[str, Any]] = self._load_dataset(data_path=sub_schema_examples_path)

            new_data: List[Dict[str, Any]] = []

            for idx, t2s_example_dict in enumerate(data):
                print(f"--- {idx}/{len(data)} --- ")
                new_t2s_dict = self.prep_few_shots_for_single_t2s(t2s_dict=t2s_example_dict, mode="train")
                new_data.append(new_t2s_dict)
                
            self._write_dataset(data=new_data, data_path=sub_schema_examples_with_few_shots_path)

        return
