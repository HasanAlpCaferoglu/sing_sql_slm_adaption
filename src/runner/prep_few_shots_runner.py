""""
Before the training or evaluation, this is used to extracts the few-shots, using vector database
in order not to increase the train/eval time.
"""

import json
import time
import copy
from typing import Any, List, Tuple, Dict, Union, Literal
from pathlib import Path
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import nltk
# Safe download: Only downloads if not already present
try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab", quiet=True, force=True)

from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
import nltk

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

file_lock = threading.Lock()

class PrepFewShotsRunner:

    def __init__(self, args: Any, method: Literal["bm25", "vdb"]= "bm25"):
        self.args = args
        self.prep_few_shots_configs = self.args.config['prep_few_shots']
        self.db_ids: List[str] = self.prep_few_shots_configs['db_ids']
        self.method = method

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

        # Synthetic Data Collection for databases
        self.db_synthetic_data_collection: Dict[str, List[Dict[str, Any]]] = {}
        
        # Key-value data for corpuses of each database
        self.db_question_corpuses: Dict[str, List[List[str]]] = {}
        self.db_sql_corpuses: Dict[str, List[List[str]]] = {}

        # BM25 engines
        self.db_bm25_question_engines: Dict[str, BM25Okapi] = {}
        self.db_bm25_sql_engines: Dict[str, BM25Okapi] = {}

        # Prepare db_synthetic_data_collection, corpuses and bm25 engines
        if self.method == "bm25":
            for db_id in self.db_ids:
                # Load db_id synthetic data
                prep_dir_name = str(self.args.config['prep_dir_name'])
                prep_dir = Path(self.args.dbs_root_dir) / db_id / prep_dir_name
                sub_schema_examples_path = prep_dir / "sub_schemas" / "column_level" / "sub_schema_examples_train.jsonl"
                synthetic_data = self._load_dataset(data_path=sub_schema_examples_path)
                self.db_synthetic_data_collection[db_id] = synthetic_data
                print("Synthetic data is loaded.")
                
                # prepare question corpus
                question_corpus = self._prepare_corpus(synthetic_data, content="question")
                self.db_question_corpuses[db_id] = question_corpus
                # prepare sql corpus
                sql_corpus = self._prepare_corpus(synthetic_data, content="sql")
                self.db_sql_corpuses[db_id] = sql_corpus
                print("Question and SQL corpuses are prepared.")

                # initialize BM25 engines
                # initialize bm25 question engine
                bm25_question_engine = BM25Okapi(question_corpus)
                self.db_bm25_question_engines[db_id] = bm25_question_engine
                # initialize bm25 sql engine
                bm25_sql_engine = BM25Okapi(sql_corpus)
                self.db_bm25_sql_engines[db_id] = bm25_sql_engine
                print("BM25 engines are initialized.")
    
    def set_method(self, method: Literal["bm25", "vdb"]):
        self.method = method
        print(f"Method is set to {self.method}")

    def _prepare_corpus(self, synthetic_data: List[Dict[str, Any]], content: Literal['question', 'sql']) -> List[str]:  
        """"
        Prepares a corpus containing only the documents
        """
        corpus = []
        for t2s_item in synthetic_data:
            if content == "question":
                corpus.append(t2s_item.get("question"))
            elif content == "sql":
                corpus.append(t2s_item.get("SQL"))
            else:
                raise ValueError("Wrong value of content attribute. It can be either `question` or `sql`")
        
        # Tokenize corpus and query with NLTK
        # Note that we don't apply lowercasing as the column names and keywords including uppercase letters may increaese the performance
        tokenized_corpus = [word_tokenize(doc) for doc in corpus]

        return tokenized_corpus

    
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
                    try:
                        obj = json.loads(line.strip())
                        dataset.append(obj)
                    except Exception as e:
                        continue
        
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
    
    def _append_json_line(self, obj: Dict, data_path: Union[str, Path]) -> None:
        """
        Write single json object (python dictionary) to the file as a single line
        """
        print("Writing single json object to a file as a single line")

        data_path = Path(data_path)
        data_path.parent.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
        with open(data_path, 'a') as f:
            f.write(json.dumps(obj) + "\n")
        
        print(f"Data is written to {data_path}")
    
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
    
    def _bm25_search(self, db_id:str, query: str, search_content: Literal["question", "sql"], top_k: int = 1) -> List[int]:
        """
        Performs BM25 search on a list of sentences using NLTK tokenization.
        Args:
            corpus (List[str]): List of input sentences.
            query (str): Keyword-based search query.
            top_k (int): Number of top documents to return.

        Returns:
            List[Tuple[int, Dict[str, Any]]]: List of (index, synthetic_data) tuples for top matches.
        """
        
        tokenized_query = word_tokenize(query)

        if search_content == "question":
            bm25_question_engine = self.db_bm25_question_engines[db_id]
            scores = bm25_question_engine.get_scores(tokenized_query)
            ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        elif search_content == "sql":
            bm25_sql_engine = self.db_bm25_sql_engines[db_id]
            scores = bm25_sql_engine.get_scores(tokenized_query)
            ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        
        # Filter out the exact match
        synthetic_data = self.db_synthetic_data_collection[db_id]
        result_indices = []
        for i in ranked_indices:
            if synthetic_data[i].get("question", "").lower().strip() != query.lower().strip():
                result_indices.append(i)
            if len(result_indices) == top_k:
                break

        return result_indices


    def prep_few_shots_for_single_t2s(self, t2s_dict: Dict[str, Any], mode: str, add_details: bool = False ) -> Dict[str, Any]: 
        if t2s_dict.get("filtered_schema", None) and t2s_dict.get("few_shot", None):
            print("The current t2s item is already processed")
            return t2s_dict
        
        prep_few_shot_start_time = time.time()
        prep_dir_name = str(self.args.config['prep_dir_name'])
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

        # Get synthetic_data
        db_synthetic_data = self.db_synthetic_data_collection.get(db_id, [])
        if not db_synthetic_data:
            raise RuntimeError(f"Synthetic data cannot loaded.")

        gt_schema_dict = get_sql_columns_dict(db_path=db_path, sql=gt_sql)

        question_keywords = self.vdb_service.extract_question_keywords(question_and_hint=question_and_hint)
        if add_details:
            t2s_dict["question_keywords"] = question_keywords

        keyword_pair_combinations = get_combinations(list_of_items=question_keywords, samples=[2])
        keyword_pair_combinations = [ f"{ks[0]} and {ks[1]}" for ks in keyword_pair_combinations]
        # print(f"keyword_pair_combinations: {keyword_pair_combinations}")

        # vdb_search_max_workers = 10


        if "few_shot" not in t2s_dict:
            t2s_dict["few_shot"] = {
                "examples": [],
                "schema_recall": 0,
                "schema_precision": 0
            }
            t2s_dict["entity_retrieval"] = {
                "selected_values": {},
                "schema_dict_via_similar_columns": {}
            }
            t2s_dict["filtered_schema"] = {
                "schema_dict": {},
                "schema_recall": 0,
                "schema_precision": 0
            }

        ##############################
        ### 1) question_and_hint search on questions
        ##############################
        
        start_time = time.time()
        similar_synthetic_examples_1 = []
        if self.method == "vdb":
            similar_synthetic_examples_1 = self.vdb_service.content_based_example_search_over_vdb(
                db_id=db_id,
                query=question_and_hint,
                content_type="question",
                k=k,
                prep_dir_name=prep_dir_name
            )
            for item in similar_synthetic_examples_1:
                item["search_keyword"] = question_and_hint
                item["search_content"] = "question"
        elif self.method == "bm25":
            search_result_indices = self._bm25_search(db_id=db_id, query=question_and_hint, search_content="question", top_k=k)
            for i in search_result_indices:
                synthetic_data_item = db_synthetic_data[i]
                synthetic_data_item["search_keyword"] = question_and_hint
                synthetic_data_item["search_content"] = "question"
                similar_synthetic_examples_1.append(synthetic_data_item)
        else:
            raise ValueError(f"Method can be either bm25 or vdb. method = {self.method}")

        duration = time.time() - start_time
        print(f"Duration for question_and_hint search on questions: {duration} seconds")
        
        ###########################
        ### Defining Search Keyword pair function for parallel search
        ###########################
        def search_keyword_pair(db_id: str, keyword_pair: str, content_type: str, k: int, prep_dir_name: str) -> list:
            """Perform content-based search for a given keyword pair."""
            extracted_examples = self.vdb_service.content_based_example_search_over_vdb(
                db_id=db_id,
                query=keyword_pair,
                content_type=content_type,
                k=k,
                prep_dir_name=prep_dir_name
            )

            for item in extracted_examples:
                item["search_keyword"] = keyword_pair  # Not clear what this empty key is for
                item["search_content"] = content_type
            return extracted_examples

        ###########################
        ### 2) keyword_pair_combinations_on_question
        ###########################
        start_time = time.time()
        similar_synthetic_examples_2 = []

        if self.method == "vdb":
            # with ThreadPoolExecutor(max_workers=vdb_search_max_workers) as executor:
            #     futures = [
            #         executor.submit(
            #             search_keyword_pair,
            #             db_id, # db_id
            #             keyword_pair, # query
            #             "question", # content_type
            #             k, # k
            #             prep_dir_name, # prep_dir_name
            #         )
            #         for keyword_pair in keyword_pair_combinations
            #     ]

            #     for future in as_completed(futures):
            #         extracted_examples = future.result()
            #         similar_synthetic_examples_2 += extracted_examples  

            for keyword_pair in keyword_pair_combinations:
                extracted_examples = self.vdb_service.content_based_example_search_over_vdb(
                    db_id=db_id,
                    query=keyword_pair,
                    content_type="question",
                    k=k,
                    prep_dir_name=prep_dir_name
                )

                for item in extracted_examples:
                    item["search_keyword"] = keyword_pair
                    item["search_content"] = "question"

                similar_synthetic_examples_2 += extracted_examples 

        elif self.method == "bm25":
            for keyword_pair in keyword_pair_combinations:

                search_result_indices = self._bm25_search(db_id=db_id, query=keyword_pair, search_content="question", top_k=k)
                for i in search_result_indices:
                    synthetic_data_item = db_synthetic_data[i]
                    synthetic_data_item["search_keyword"] = keyword_pair
                    synthetic_data_item["search_content"] = "question"
                    similar_synthetic_examples_2.append(synthetic_data_item)
        else:
            raise ValueError(f"Method can be either bm25 or vdb. method = {self.method}")

        duration = time.time() - start_time
        print(f"Duration for keyword_pair_combinations_on_question: {duration} seconds")
        

        ###########################
        ### 3) keyword_pair_combinations_on_sql
        ###########################
        start_time = time.time()
        similar_synthetic_examples_3 = []
        if self.method == "vdb":
            # with ThreadPoolExecutor(max_workers=vdb_search_max_workers) as executor:
            #     futures = [
            #         executor.submit(
            #             search_keyword_pair,
            #             db_id,
            #             keyword_pair,
            #             "sql",
            #             k,
            #             prep_dir_name
            #         )
            #         for keyword_pair in keyword_pair_combinations
            #     ]
                
            #     for future in as_completed(futures):
            #         extracted_examples = future.result()
            #         similar_synthetic_examples_3 += extracted_examples

            for keyword_pair in keyword_pair_combinations:
                extracted_examples = self.vdb_service.content_based_example_search_over_vdb(
                    db_id=db_id,
                    query=keyword_pair,
                    content_type="sql",
                    k=k,
                    prep_dir_name=prep_dir_name
                )
                for item in extracted_examples:
                    item["search_keyword"] = keyword_pair
                    item["search_content"] = "sql"

                similar_synthetic_examples_3 += extracted_examples

        elif self.method == "bm25":
            a = 1
            for keyword_pair in keyword_pair_combinations:
                search_result_indices = self._bm25_search(db_id=db_id, query=keyword_pair, search_content="sql", top_k=k)
                for i in search_result_indices:
                    synthetic_data_item = db_synthetic_data[i]
                    synthetic_data_item["search_keyword"] = keyword_pair
                    synthetic_data_item["search_content"] = "sql"
                    similar_synthetic_examples_3.append(synthetic_data_item)
        else:
            raise ValueError(f"Method can be either bm25 or vdb. method = {self.method}")

        duration = time.time() - start_time
        print(f"Duration for keyword_pair_combinations_on_sql: {duration} seconds")
        
        ###########################
        ### 4) Computing Similarity Between Extracted Examples and The User Question
        ###########################
        start_time = time.time()
        synthetic_examples_combination = similar_synthetic_examples_1 +  similar_synthetic_examples_2 + similar_synthetic_examples_3
        unique_examples_all = {}
        for example in synthetic_examples_combination:
            key = f"{example.get('ss_id')}-{example.get('example_no')}"
            if "ss_id" in t2s_dict:
                current_question_key = f"{t2s_dict.get('ss_id')}-{t2s_dict.get('example_no')}"
                if current_question_key == key: # if the question in the extracted example is same with the currently considered t2s item, exclude the example.
                    continue
            if example.get("question_similarity", 0) >= 0.99:
                continue
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
        
        few_shot_examples = []
        for item in similar_synthetic_examples_all:
            item_question = item.get("question")
            user_question = t2s_dict.get("question")
            try:
                question_similarity = T2SSyntheticVDBService.compute_text_similarity(item_question, user_question, embedding_model_provider=self.embedding_model_provider, embedding_model_name_or_path=self.embedding_model_name_or_path)
            except Exception as e:
                print("Couln't compute question similarity. So, it is taken as 0.")
                question_similarity = 0.0
            few_shot_examples.append({
                "ss_id":item.get("ss_id"), 
                "example_no":item.get("example_no"), 
                "question": item.get("question"), 
                "SQL": item.get("SQL"), 
                "dac_reasoning": item.get("dac_reasoning", ""), 
                "search_keyword": item.get("search_keyword"),
                "question_similarity": question_similarity
            })
        
        few_shot_examples.sort(key=lambda x: x["question_similarity"], reverse=True) # Sort few-shot examples
        if add_details:
            t2s_dict["few_shot"]["examples"] = few_shot_examples
        else:
            t2s_dict["few_shot"]["examples"] = [
                {
                    "ss_id": example_dict["ss_id"], 
                    "example_no": example_dict["example_no"], 
                    "search_keyword": example_dict['search_keyword'], 
                    "question_similarity": example_dict["question_similarity"]
                } 
                for example_dict in few_shot_examples
            ]
            
        t2s_dict["few_shot"]["schema_recall"] = s_recall
        t2s_dict["few_shot"]["schema_precision"] = s_precision
        # t2s_dict["few_shot"]["schema_dict"] = schema_dict

        duration = time.time() - start_time
        print(f"Duration for Extracting Schema Using Previous Steps Combination : {duration} seconds")

        ###########################
        ### 6) Retrieve Entity via Similarity and LSH
        ###########################
        # Get similar database values
        start_time = time.time()
        selected_values: Dict = {}
        schema_dict_via_values: Dict = {}
        entity_retrieval = EntityRetrieval(args = self.args, 
                                           db_id=db_id, 
                                           edit_distance_threshold=0.3, 
                                           embedding_similarity_threshold=0.6
                                        ) # Default CHESS values edit_distance_threshold=0.3 and embedding_similarity_threshold=0.6
        selected_values, schema_dict_via_values = entity_retrieval.get_similar_entities(keywords=question_keywords)
        # print(f"selected_values: {selected_values}")
        # print(f"schema_dict_via_values: {schema_dict_via_values}")
        t2s_dict["entity_retrieval"]["selected_values"] = selected_values
    

        # Get similar database columns
        schema_dict_via_similar_columns = entity_retrieval.get_similar_columns(keywords=question_keywords, question_and_hint=question_and_hint)
        t2s_dict["entity_retrieval"]["schema_dict_via_similar_columns"] =  schema_dict_via_similar_columns


        ###########################
        ### 7) Filter with LLM
        ###########################
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
            # print(f"LLM filtration reasoning: \n {reasoning} \n") # DELETE LATER OF COMMENT OUT LATER
            
            if filtered_table_columns:
                ## Add all columns that are either PK or FK if not in filtered_table_columns
                for column_name in columns:
                    is_connection: bool = db_info.original_db_schema_generator.is_connection(table_name=table_name, column_name=column_name)
                    if is_connection and column_name not in filtered_table_columns:
                        filtered_table_columns.append(column_name)

                llm_filtered_schema_dict[table_name] = filtered_table_columns
        
        
        print(f"llm_filtered_schema_dict: \n {json.dumps(llm_filtered_schema_dict, indent=4)} \n") # DELETE LATER OF COMMENT OUT LATER
        
        duration = time.time() - start_time
        print(f"Duration for Filter with LLM based on examples : {duration} seconds")


        ###########################
        ### 8) Update LLM Filtered Schema by Adding Similar Columns
        ###########################
        # Updating filtered schema dictionary by adding schema column obtained from similarity.
        filtered_schema_dict = copy.deepcopy(llm_filtered_schema_dict)
        for t_name, c_names in schema_dict_via_similar_columns.items():
            if t_name not in filtered_schema_dict:
                filtered_schema_dict[t_name] = c_names
            else:
                for c_name in c_names:
                    if c_name not in filtered_schema_dict[t_name]:
                        filtered_schema_dict[t_name].append(c_name)

        num_tp, num_fp, num_fn, s_recall, s_precision, s_sf1 = calculate_schema_metrics_for_single_schema(
            used_schema_dict=filtered_schema_dict,
            gt_schema_dict=gt_schema_dict
        )

        t2s_dict["filtered_schema"]['schema_dict'] = filtered_schema_dict
        t2s_dict["filtered_schema"]['schema_recall'] = s_recall
        t2s_dict["filtered_schema"]['schema_precision'] = s_precision

        if add_details:
            t2s_dict["gt_schema_dict"] = gt_schema_dict
        
        duration = time.time() - start_time
        print(f"Duration for Entity Retrieval: {duration} seconds")

        prep_few_shot_duration = time.time() - prep_few_shot_start_time
        print(f"+++++++++++++++++++Duration for Prep Few Shots For Single T2S: {prep_few_shot_duration} seconds.")
        return t2s_dict

    def prep_eval_few_shots(self):

        print("Starting to prepare eval data few shots")
        eval_dataset_path = Path(self.args.data_json_path) 
        
        # Create new file name by adding '_with_few_shots' before the extension
        eval_with_few_shots_file_name = eval_dataset_path.stem + "_with_few_shots" + eval_dataset_path.suffix
        eval_dataset_with_few_shots_path = eval_dataset_path.parent / eval_with_few_shots_file_name

        if eval_dataset_with_few_shots_path.exists():
            dataset = self._load_dataset(data_path=eval_dataset_with_few_shots_path)
        else:
            dataset = self._load_dataset(data_path=eval_dataset_path)


        def process_item(eval_t2s_item: Dict[str, Any]) -> Dict[str, Any]:
            db_id = eval_t2s_item.get("db_id")
            if db_id not in self.db_ids:
                return eval_t2s_item
            return self.prep_few_shots_for_single_t2s(t2s_dict=eval_t2s_item, mode="eval")

        eval_dataset_with_few_shots = copy.deepcopy(dataset)

        max_outer_threads = 12  # update this based on system and load
        with ThreadPoolExecutor(max_workers=max_outer_threads) as executor:
            futures = [
                executor.submit(lambda i, x: (i, process_item(x)), idx, item)
                for idx, item in enumerate(dataset)
            ]

            
            for future in futures:
                idx, result = future.result()
                print(f"================= {idx}/{len(dataset)} ================= ")
                # eval_dataset_with_few_shots.append(result)
                item_db_id = None
                if "db_id" in result:
                    item_db_id = result.get("db_id")
                elif "ss_id" in result:
                    item_db_id = result.get("ss_id").split("-")[0]
    
                if item_db_id in self.db_ids:
                    eval_dataset_with_few_shots[idx] = result # change the t2s item

                    with file_lock:
                        self._write_dataset(data = eval_dataset_with_few_shots, data_path=eval_dataset_with_few_shots_path)


        # for idx, eval_t2s_item in enumerate(dataset):
        #     print(f"--- {idx}/{len(dataset)} --- ")
        #     db_id = eval_t2s_item.get("db_id")
        #     if db_id not in self.db_ids:
        #         eval_dataset_with_few_shots.append(eval_t2s_item)
        #         continue
            
        #     new_eval_t2s_item = self.prep_few_shots_for_single_t2s(t2s_dict=eval_t2s_item, mode="eval")
        #     # eval_dataset_with_few_shots.append(new_eval_t2s_item)
        #     eval_dataset_with_few_shots[idx] = result # change the t2s item
        #     self._write_dataset(data = eval_dataset_with_few_shots, data_path=eval_dataset_with_few_shots_path)
        

        return
    
    def prep_synthetic_data_few_shots(self):

        print("Starting to prepare training data few shots")
        dbs_root_dir = Path(self.args.dbs_root_dir)

        for db_id in self.db_ids:
            db_directory_path = Path(self.args.dbs_root_dir) / db_id
            db_path = Path(self.args.dbs_root_dir) / db_id / f"{db_id}.sqlite"
            prep_dir_name = str(self.args.config['prep_dir_name'])
            prep_dir = Path(self.args.dbs_root_dir) / db_id / prep_dir_name
            
            split_names = ['dev', 'test', 'train']
            split_names = ['train']

            for split_name in split_names:
                split_start_time = time.time()
                print(f"\n\n\n ========== FEW SHOT PREP FOR SYNTHETIC DATA ({split_name} SPLIT) STARTED ==========\n\n\n")
                sub_schema_examples_path = prep_dir / "sub_schemas" / "column_level" / f"sub_schema_examples_{split_name}.jsonl"

                sub_schema_examples_with_few_shots_file_name = sub_schema_examples_path.stem + "_with_few_shots" + sub_schema_examples_path.suffix
                sub_schema_examples_with_few_shots_path = sub_schema_examples_path.parent / sub_schema_examples_with_few_shots_file_name

                print("Loading split data...")
                db_split_dataset: List[Dict[str, Any]] = self._load_dataset(data_path=sub_schema_examples_path)
                db_split_dataset_with_few_shot: List[Dict[str, Any]] = []
                if sub_schema_examples_with_few_shots_path.exists():
                    print("Loading existing split data (the ones with few-shots)...")
                    db_split_dataset_with_few_shot = self._load_dataset(data_path=sub_schema_examples_with_few_shots_path)

                already_processed_items_keys = {
                    (item["ss_id"], item["example_no"]) for item in db_split_dataset_with_few_shot
                }


                def process_item(idx, t2s_item: Dict[str, Any]) -> Tuple[int, Dict[str, Any]]:
                    """
                    process T2S item.

                    Returns:
                        int: index in the dataset
                        Dict[str, Any]: processed_t2s
                        bool: already_processed --> Whether if the item is processed or not (already processed)
                    """
                    ss_id = t2s_item.get("ss_id")
                    example_no = t2s_item.get("example_no")
                    key = (ss_id, example_no)
                    if key in already_processed_items_keys:
                        print(f"item at {idx} has already been processed. Moving to next one.")
                        return idx, t2s_item, True
                    db_id = ss_id.split("-")[0]
                    result =  self.prep_few_shots_for_single_t2s(t2s_dict=t2s_item, mode="train", add_details=True)
                    return idx, result, False


                ## Parallel processing
                max_outer_threads = 12  # update this based on system and load
                with ThreadPoolExecutor(max_workers=max_outer_threads) as executor:
                    futures = [
                        executor.submit(
                            process_item,
                            idx,
                            item
                        )
                        for idx, item in enumerate(db_split_dataset)
                    ]

                    for future in futures:
                        idx, result, is_processed_already = future.result()
                        print(f"================= {idx}/{len(db_split_dataset)} ================= ")
                        if is_processed_already:
                            print("Item has already been processed. Moving to next one")
                            continue
                        # raise error when filtered_schema schema_dict is empty
                        if result.get("filtered_schema", {}).get("schema_dict", {}) == {}:
                            raise RuntimeError(f"Unknown error for {idx}. Filtered Schema Dict couldn't be generated!")
                        
                        # db_split_dataset[idx] = result
                        # db_split_dataset[idx] = {
                        #     "ss_id": result.get("ss_id"),
                        #     "example_no": result.get("example_no"),
                        #     "difficulty": result.get("difficulty"),
                        #     "SQL": result.get("SQL"),
                        #     # "sql_analysis": data_item.get("sql_analysis"),
                        #     "question": result.get("question"),
                        #     # "question_and_sql_logic_analysis": result.get("question_and_sql_logic_analysis"),
                        #     "is_logical": result.get("is_logical"),
                        #     "execution_status": result.get("execution_status"),
                        #     # "error_reason": result.get("error_reason"),
                        #     # "is_fixed": result.get("is_fixed"),
                        #     "dac_reasoning": result.get("dac_reasoning"),
                        #     "few_shot": result.get("few_shot"),
                        #     "entity_retrieval": result.get("entity_retrieval"),
                        #     "filtered_schema": result.get("filtered_schema"),
                        # } # Change the t2s item 
                        
                        with file_lock:
                            # self._write_dataset(data=db_split_dataset, data_path=sub_schema_examples_with_few_shots_path)
                            self._append_json_line(obj=result, data_path=sub_schema_examples_with_few_shots_path)

                # ## Sequential processing
                # for idx, t2s_example_dict in enumerate(db_training_dataset):
                #     print(f"--- {idx}/{len(db_training_dataset)} --- ")
                #     new_t2s_dict = self.prep_few_shots_for_single_t2s(t2s_dict=t2s_example_dict, mode="train")
                #     # new_data.append(new_t2s_dict)
                #     db_training_dataset[idx] = new_t2s_dict
                #     self._write_dataset(data=db_training_dataset, data_path=sub_schema_examples_with_few_shots_path)

                split_duration = time.time() - split_start_time
                print(f"\n\n === Few-shot prep for synthetic data ({split_name} split) finished in {split_duration} seconds. \n\n")

        return
