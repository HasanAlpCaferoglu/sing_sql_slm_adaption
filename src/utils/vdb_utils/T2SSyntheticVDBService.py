import os
import re
import ast
import json
import copy
from pathlib import Path
from typing import Any, List, Dict, Optional
import logging
import time

from utils.db_utils.sql_parser import get_sql_columns_dict, get_filtered_schema_dict_from_similar_examples
from utils.db_utils.db_info import DatabaseGeneralInfo

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore, RetrievalMode
from qdrant_client import QdrantClient, models

from utils.llm_utils.LLMService import LLMService
from utils.llm_utils.prompt_utils import load_template

class T2SSyntheticVDBService:
    # Qdrant vector database service
    def __init__(self, dbs_root_dir, embedding_model_provider, embedding_model_name_or_path, llm_model_name, llm_model=None, llm_tokenizer=None, logger: Optional[logging.Logger] = None):
        self.dbs_root_dir = Path(dbs_root_dir)
        self.embedding_model_provider = embedding_model_provider
        self.embedding_model_name_or_path = embedding_model_name_or_path
        self.llm_model_name = llm_model_name
        self.llm_model = llm_model
        self.llm_tokenizer = llm_tokenizer
        self.logger = logger if logger else logging.getLogger("LLMService")

        self.vdb_embedding_fn = self._get_embedding_fn()

        self.cached_vector_stores: Dict[str, QdrantVectorStore] = {} # cache vector stores where keys are db_id and values are QdrantVectorStore
        self.cached_synthetic_examples: Dict[str, List[Dict[str, Any]]] = {} # cache the synthetic examples for db_ids

    def _get_embedding_fn(self):
        """"
        Gets the embedding model 
        """

        if self.embedding_model_provider.lower() == "google" or self.embedding_model_provider.lower() == "gemini":
            GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
            os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY
            embeddings = GoogleGenerativeAIEmbeddings(model=self.embedding_model_name_or_path)
            return embeddings
        elif self.embedding_model_provider.lower() == "sentence-transformer":
            embeddings = HuggingFaceEmbeddings(model_name=self.embedding_model_name_or_path)
            return embeddings
        else:
            raise ValueError(f"embedding_model_provider = {self.embedding_model_provider} has not been implemented yet.")


    def load_local_vdb_store(self, db_id, prep_dir_name: str = "prep_schemaless") -> QdrantVectorStore:
        
        if db_id in self.cached_vector_stores:
            return self.cached_vector_stores.get(db_id)
        
        prep_schemaless_dir = self.dbs_root_dir / db_id / prep_dir_name
        vdb_save_path = prep_schemaless_dir / f"vdb_{db_id}"
        collection_name = f"vdb_{db_id}"

        client = QdrantClient(path=vdb_save_path)
        # Check if collection exist
        if not client.collection_exists(collection_name):
            print(f"❌ Collection for {db_id} ({collection_name}) does not exist.")
            raise ValueError(f"❌ Collection for {db_id} ({collection_name}) does not exist.")
        
        collection_info = client.get_collection(collection_name)
        print(f"✅ Collection `{collection_name}` exist. Vector count: {collection_info.points_count}")

        vector_store = QdrantVectorStore(
            client=client,
            collection_name=collection_name,
            embedding=self.vdb_embedding_fn,
            vector_name="dense",
            retrieval_mode=RetrievalMode.DENSE
        )
        
        # cache vector store
        self.cached_vector_stores[db_id] = vector_store

        return vector_store
    
    def _load_synthetic_examples(self, db_id: str, prep_dir_name:str="prep_schemaless") -> List[Dict[str, Any]]:
        """"
        Get synthetic text-to-sql examples for the given db
        """
        if db_id in self.cached_synthetic_examples:
            return self.cached_synthetic_examples[db_id]
        
        prep_schemaless_dir = self.dbs_root_dir / db_id / prep_dir_name
        examples_jsonl_path = prep_schemaless_dir /  "sub_schemas" / "column_level" / "sub_schema_examples_train.jsonl"
        if not examples_jsonl_path.exists():
            print(f"Warning: File not found at {examples_jsonl_path}")
            self.cached_synthetic_examples[db_id] = []
            return []
        
        t2s_example_dicts: List[Dict[str, Any]] = []
        with open(examples_jsonl_path, 'r') as file:
            for line in file:
                try:
                    obj = json.loads(line)
                    t2s_example_dicts.append(obj)
                except Exception as e:
                    continue

        # cache synthetic examples
        self.cached_synthetic_examples[db_id] = t2s_example_dicts
        return t2s_example_dicts
    
    def extract_text_n_grams(self, text: str, n_values: List[int] = [3]) -> List[str]:
        """"
        Extracts the n-grams of the text
        """
        tokens = text.strip().split()
        n_grams = []

        for n in n_values:
            if n <= 0:
                continue
            for i in range(len(tokens) - n + 1):
                n_gram = " ".join(tokens[i:i + n])
                n_grams.append(n_gram)

        return n_grams

    def extract_question_keywords(self, question_and_hint: str) -> List[str]:
        """"
        Extracts the important keywords and phrases in the question.
        """
        def parse_for_list(input_str):
            # Match the first square-bracketed expression (non-greedy, allows newlines)
            match = re.search(r"\[.*?\]", input_str, re.DOTALL)
            if match:
                list_str = match.group(0)
                return list_str
            else:
                print("No list found in input string")
                return input_str
        
        print("Extracting question keywords...")
        question_keywords: List[str] = []

        prompt_template = load_template('extract_keyword')
        prompt = prompt_template.format(
            QUESTION_AND_HINT = question_and_hint
        )
        llm_service = LLMService(self.llm_model_name)
        response_object, *_ = llm_service.call_llm(prompt=prompt, llm_model=self.llm_model, llm_tokenizer=self.llm_tokenizer)
        output_text = response_object.text
        list_str = parse_for_list(output_text)
        
        # Parse LLM output
        try:
            question_keywords = ast.literal_eval(list_str)
        except Exception as e1:
            print("Couldn't parse the LLM response as a list.")
            print(f"ast.literal_eval error: {e1}")
            try:
                question_keywords = json.loads(output_text)
            except Exception as e2:
                print(f"Couln't parse the LLM response (convert list of strings). json.loads error: {e2}")
                print("Using n-grams for keywords...")
                question_keywords = self.extract_text_n_grams(question_and_hint)
                question_keywords.append(question_and_hint)

        print(f"Question keywords are extracted. Keywords: {question_keywords}")
        return question_keywords
    
    def _find_examples_from_metadata(self, t2s_example_dicts: List[Dict[str, Any]], metadatas: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Using metadata of the items extracted from similarity search, the complete information about the item will be found.
        """
        selected_ss_ids = [metadata.get("ss_id") for metadata in metadatas]
        selected_example_nos = [metadata.get("example_no") for metadata in metadatas]

        selected_examples: List[Dict[str, Any]] = []
        for t2s_dict in t2s_example_dicts:
            if not t2s_dict.get("ss_id") in selected_ss_ids:
                continue
            for idx, (selected_example_ss_id, selected_example_no) in enumerate(zip(selected_ss_ids, selected_example_nos)):
                if selected_example_ss_id == t2s_dict.get("ss_id") and selected_example_no == t2s_dict.get("example_no"):
                    t2s_dict_copy = copy.deepcopy(t2s_dict)
                    t2s_dict_copy['score'] = float(metadatas[idx].get('score'))
                    t2s_dict_copy['search_keyword'] = str(metadatas[idx].get('search_keyword'))
                    selected_examples.append(t2s_dict_copy)
    
        # Sort by score in descending order
        selected_examples.sort(key=lambda x: x['score'], reverse=True)

        return selected_examples
    
    def content_based_example_search_over_vdb(self, db_id: str, query:str, content_type:str='question', k:int=3, prep_dir_name:str="prep_schemaless", ss_id:str=None, example_no:int=None) -> List[Dict[str, Any]]: 
        """"
        For a given query search over vector database depending on the content.
        """

        if content_type not in ["question", "sql"]:
            raise ValueError("Wrong value for the content_type. It can be either 'question' or 'sql'.")

        vector_store = self.load_local_vdb_store(db_id=db_id, prep_dir_name=prep_dir_name)
        synthetic_examples = self._load_synthetic_examples(db_id=db_id, prep_dir_name=prep_dir_name)
        similar_t2s_examples: List[Dict[str, Any]] = []

        # set must_not filter
        must_not_filters = []
        if ss_id is not None and example_no is not None:
            must_not_filters.append(
                models.Filter(
                    must=[
                        models.FieldCondition(key="metadata.ss_id", match=models.MatchValue(value=ss_id)),
                        models.FieldCondition(key="metadata.example_no", match=models.MatchValue(value=example_no))
                    ]
                )
            )

        # Get the similar items using query-question similarity
        similar_items = vector_store.similarity_search_with_score(
            query=query,
            k=k,
            filter=models.Filter(
                must=[
                    models.FieldCondition(key="metadata.content_type", match=models.MatchValue(value=content_type))
                ],
                must_not=must_not_filters
            )
        )

        # Get metadata of the similar questions
        metadatas: List[Dict[str, Any]] = []
        for sim_item_content, sim_score in similar_items:
            metadata = copy.deepcopy(sim_item_content.metadata)
            metadata['search_content'] = content_type
            metadata['score'] = float(sim_score)
            metadata['search_query'] = query
            metadatas.append(metadata)

        # Find examples' complete information from metadata
        similar_t2s_examples = self._find_examples_from_metadata(synthetic_examples, metadatas)
        return similar_t2s_examples


    def example_search_over_vdb_using_keyword(self, db_id: str, keyword:str, k:int=3, ss_id:str=None, example_no:int=None) -> List[Dict[str, Any]]:
        
        similar_t2s_examples: List[Dict[str, Any]] = []
        # Get the similar items using query(keyword)-question similarity
        similar_t2s_examples_based_on_questions = self._content_based_example_search_over_vdb(
            db_id=db_id,
            query=keyword,
            content_type="question",
            k=k,
            ss_id=ss_id,
            example_no=example_no
        )
        # print(f"len(similar_t2s_examples_based_on_questions): {len(similar_t2s_examples_based_on_questions)}") # DELETE LATER OR COMMENT OUT LATER
        similar_t2s_examples += similar_t2s_examples_based_on_questions

        # Get the similar items using query(keyword)-SQL similarity
        similar_t2s_examples_based_on_sqls = self._content_based_example_search_over_vdb(
            db_id=db_id,
            query=keyword,
            content_type="sql",
            k=k,
            ss_id=ss_id,
            example_no=example_no
        )
        # print(f"len(similar_t2s_examples_based_on_sqls): {len(similar_t2s_examples_based_on_sqls)}") # DELETE LATER OR COMMENT OUT LATER
        similar_t2s_examples += similar_t2s_examples_based_on_sqls

        # print(f"len(similar_t2s_examples): {len(similar_t2s_examples)}") # DELETE LATER OR COMMENT OUT LATER
        return similar_t2s_examples
    

    def search_examples(self, question_and_hint: str, db_id: str, k:int=3, ss_id:str=None, example_no:int=None) -> List[Dict[str, Any]]:
        """
        Seach synthetic examples for a question.
        Search steps.
            1. Search similar Text-to-SQL examples using question_and_hint as vector search query and constraining search on the question content
            2. Search similar Text-to-SQL examples using keywords as vector search query in the question and SQL content
        """
        search_start_time = time.time()
        question_keywords = self.extract_question_keywords(question_and_hint)
        print("Starting to search similar synthetic examples using keywords...")
        similar_examples: List[Dict[str, Any]] = []

        extracted_examples = self.content_based_example_search_over_vdb(
            db_id=db_id, 
            query=question_and_hint, 
            content_type="question",
            k=k,
            ss_id=ss_id,
            example_no=example_no
        )
        similar_examples += extracted_examples

        for keyword in question_keywords:
            extracted_examples = self.example_search_over_vdb_using_keyword(
                db_id=db_id, 
                keyword=keyword, 
                k=k, 
                ss_id=ss_id, 
                example_no=example_no
            )
            similar_examples += extracted_examples

        search_end_time = time.time()
        search_duration = search_end_time - search_start_time
        print(f"Synthetic example search is completed in {search_duration} seconds. {len(similar_examples)} synthetic examples are extracted.")
        return similar_examples
    
    @staticmethod
    def get_filtered_schema_dict_from_similar_examples(db_path: Path, similar_examples: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        """"
        Constructs a filtered schema by parsing SQL queries in the similar examples
        """

        filtered_schema_tables_and_columns_dict: Dict[str, List[str]] = {}
        for t2s_dict in similar_examples:
            sql_query = t2s_dict.get("SQL")

            sql_columns_dict = get_sql_columns_dict(db_path=db_path, sql=sql_query)

            for table, columns in sql_columns_dict.items():
                if table not in filtered_schema_tables_and_columns_dict:
                    # Add new table with its columns
                    filtered_schema_tables_and_columns_dict[table] = list(columns)
                else:
                    # Add new columns without duplicating existing ones
                    for col in columns:
                        if col not in filtered_schema_tables_and_columns_dict[table]:
                            filtered_schema_tables_and_columns_dict[table].append(col)


        return filtered_schema_tables_and_columns_dict
    
    
    @staticmethod
    def compute_text_similarity(text1: str, text2: str, embedding_model_provider: str="google", embedding_model_name_or_path: str="models/embedding-001") -> float:
        """
        Compute cosine similarity between two text strings using the given embedding model.
        
        Args:
            text1 (str): First text.
            text2 (str): Second text.
            embedding_model_provider (str): Provider of the embedding model ('google' / 'gemini' / 'sentence-transformer').
            embedding_model_name_or_path (str): Model name or path for the embedding model.

        Returns:
            float: Cosine similarity score between text1 and text2.
        """
        from sklearn.metrics.pairwise import cosine_similarity
        import numpy as np

        # Load embedding model
        if embedding_model_provider.lower() in ["google", "gemini"]:
            GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
            os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY
            embedder = GoogleGenerativeAIEmbeddings(model=embedding_model_name_or_path)
        elif embedding_model_provider.lower() == "sentence-transformer":
            embedder = HuggingFaceEmbeddings(model_name=embedding_model_name_or_path)
        else:
            raise ValueError(f"Embedding provider '{embedding_model_provider}' not supported.")

        # Compute embeddings
        embedding1 = embedder.embed_query(text1)
        embedding2 = embedder.embed_query(text2)

        # Convert to numpy and compute cosine similarity
        similarity = cosine_similarity(
            np.array(embedding1).reshape(1, -1),
            np.array(embedding2).reshape(1, -1)
        )[0][0]

        return float(similarity)
