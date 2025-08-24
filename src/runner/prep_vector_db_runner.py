import os
import json
import logging
import time
import getpass
from uuid import uuid4
from pathlib import Path
from typing import Any, List, Dict
from dotenv import load_dotenv

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore, RetrievalMode
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import Distance, VectorParams



class PrepVDBRunner:
    def __init__(self, args: Any):
        self.args = args
        self.db_ids: List[str] = self.args.config.get("db_ids", [])
        self.index_method: str = args.config['index_method'] if args.config['index_method'] else "IndexHNSWFlat"
        self.model_provider: str = args.config['model_provider'] if args.config['model_provider'] else "google"
        self.model_name_or_path: str = args.config['model_name_or_path'] if args.config['model_name_or_path'] else "models/embedding-001"
        self.batch_size: int = args.config['batch_size'] if args.config['batch_size'] else 1

        logger = logging.getLogger('prepVDB')
        logger.setLevel(logging.INFO)
        logger_path = Path(f"logs/prep_vdb_{self.args.run_start_time}/prep_vdb_logs.log")
        logger_path.parent.mkdir(parents=True, exist_ok=True)
        logger_handler = logging.FileHandler(logger_path)
        logger_handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
        logger.addHandler(logger_handler)
        self.prep_vdb_logger = logger

        self.embedding_fn = self._get_embedding_fn()

    def _get_embedding_fn(self):
        """"
        gets embedding model depending on the provider and model name
        """
        if self.model_provider.lower() == "google" or self.model_provider.lower() == "gemini":
            GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
            os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY
            embeddings = GoogleGenerativeAIEmbeddings(model=self.model_name_or_path)
            return embeddings
        elif self.model_provider.lower() == "sentence-transformer":
            embeddings = HuggingFaceEmbeddings(model_name=self.model_name_or_path)
            return embeddings
        else:
            raise ValueError(f"model_provider = {self.model_provider} has not been implemented yet.")
    

    def _initialize_vector_store(self, vdb_save_path: Path, collection_name: str):
        """ 
        Initialize a vector store for a single database
        """
        vector_dim = len(self.embedding_fn.embed_query("hello world"))

        client = QdrantClient(path=vdb_save_path)
        # Create collection
        client.create_collection(
            collection_name=collection_name,
            vectors_config={"dense": VectorParams(size=vector_dim, distance=Distance.COSINE, on_disk=True)},
        )

        vector_store = QdrantVectorStore(
            client=client,
            collection_name=collection_name,
            embedding=self.embedding_fn,
            retrieval_mode=RetrievalMode.DENSE,
            vector_name="dense",
        )

        return vector_store

    def construct_and_save_single_vdb(self, db_id: str) -> QdrantClient:
        """
        Construct Vector Database for a single database
        """
        self.prep_vdb_logger.info(f"Constructing vector database for the {db_id}")
        prep_schemaless_dir = self.args.dbs_root_dir / db_id / str(self.args.config['prep_dir_name'])
        examples_jsonl_path = self.args.dbs_root_dir / db_id / str(self.args.config['prep_dir_name']) / "sub_schemas" / "column_level" / "sub_schema_examples_train.jsonl"

        vdb_save_path = prep_schemaless_dir / f"vdb_{db_id}"
        collection_name = f"vdb_{db_id}"

        if not examples_jsonl_path.exists():
            raise ValueError(f"!!!!!! Couldn't find the synthetic Text-to-SQL examples for the given database {db_id}.")

        vector_store: QdrantVectorStore = self._initialize_vector_store(vdb_save_path, collection_name)
        
        t2s_example_dicts: List[Dict[str, Any]] = []
        with open(examples_jsonl_path, 'r') as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    t2s_example_dicts.append(obj)
                except Exception:
                    continue
                    
        # t2s_example_dicts = t2s_example_dicts[:30] #### DELETE LATER or COMMENT OUT LATER - TRIAL PURPOSE
        batch_size = self.batch_size
        start_idx = 0
        end_idx = start_idx + batch_size
        while start_idx <= len(t2s_example_dicts) - 1:
            self.prep_vdb_logger.info(f"Adding {batch_size} examples (from {start_idx} to {end_idx})/{len(t2s_example_dicts)})")
            print(f"Adding {batch_size} examples (from {start_idx} to {end_idx})/{len(t2s_example_dicts)})")
            t2s_dicts: List[Dict[str, str]] = t2s_example_dicts[start_idx:end_idx]
            s_time = time.time()
            q_docs = []
            sql_docs = []
            q_docs_uuids = []
            sql_docs_uuids = []
            for t2s_obj in t2s_dicts:
                if not bool(t2s_obj.get("is_logical", False)):
                    self.prep_vdb_logger.info(f"The SQL is not logical.  Moving to the next example")
                    continue
                if t2s_obj.get("execution_status", None) != "SYNTACTICALLY_CORRECT":
                    self.prep_vdb_logger.info(f"The SQL execution result is {t2s_obj.get('execution_status')}, so NOT SYNTACTICALLY_CORRECT. Moving to the next example")
                    continue
                ss_id = t2s_obj.get("ss_id")
                example_no = t2s_obj.get("example_no")
                
                question = t2s_obj.get("question")
                SQL = t2s_obj.get("SQL")
                try:
                    # self.prep_vdb_logger.info(f"-----Question: {t2s_obj.get('question')} ||| ss_id: {t2s_obj.get('ss_id')}")
                    q_doc_uuid = str(uuid4())
                    q_docs_uuids.append(q_doc_uuid)
                    example_id = f"{ss_id}-e{example_no}"
                    q_docs.append(Document(page_content=t2s_obj.get("question"), metadata={"content_type": "question", "ss_id": t2s_obj.get("ss_id"), "example_no": example_no, "question": t2s_obj.get("question"), "SQL": t2s_obj.get('SQL') } ))
                    # q_docs_uuids.append(f"{ss_id}-e{example_no}-q")
                except Exception as e:
                    self.prep_vdb_logger.info(f"Couldn't construct a document for the Question in example {ss_id}-e{example_no}-q. Error: {e}")

                try:
                    # self.prep_vdb_logger.info(f"-----SQL: {t2s_obj.get('SQL')} ||| ss_id: {t2s_obj.get('ss_id')}")
                    sql_doc_uuid = str(uuid4())
                    sql_docs_uuids.append(sql_doc_uuid)
                    example_id = f"{ss_id}-e{example_no}"
                    sql_docs.append(Document(page_content=t2s_obj.get("SQL"), metadata={"content_type": "sql", "ss_id": t2s_obj.get("ss_id"), "example_no": example_no, "question": t2s_obj.get("question"), "SQL": t2s_obj.get('SQL') }))
                    # sql_docs_uuids.append(f"{ss_id}-e{example_no}-s")
                except Exception as e:
                    self.prep_vdb_logger.info(f"Couldn't construct a document for the SQL in example {ss_id}-e{example_no}-q. Error: {e}")

            if q_docs and q_docs_uuids:
                try:
                    time.sleep(0.5)
                    vector_store.add_documents(documents=q_docs, ids=q_docs_uuids)
                except Exception as e:
                    self.prep_vdb_logger.info(f"Couldn't add Question as document (from {start_idx} to {end_idx}). Error: {e}")
            if sql_docs and sql_docs_uuids:
                try:
                    time.sleep(0.5)
                    vector_store.add_documents(documents=sql_docs, ids=sql_docs_uuids)
                except Exception as e:
                    self.prep_vdb_logger.info(f"Couldn't add SQL as document (from {start_idx} to {end_idx}). Error: {e}")

            e_time = time.time()
            duration = e_time - s_time
            self.prep_vdb_logger.info(f"Adding {batch_size} examples to vector db completed in {duration} seconds.")
            start_idx += batch_size
            end_idx = start_idx+batch_size

        self.prep_vdb_logger.info(f"Vector database construction completed.")
        
        return vector_store
    
    def construct_vdbs(self):
        """"
        Construct Vector Databases for given databases
        """
        self.prep_vdb_logger.info("Starting vector database construction...")
        for db_id in self.db_ids:
            try:
                self.construct_and_save_single_vdb(db_id=db_id)
            except Exception as e:
                self.prep_vdb_logger.info(f"Vector DB couldn't be constructed for the {db_id} database: {e}")
            