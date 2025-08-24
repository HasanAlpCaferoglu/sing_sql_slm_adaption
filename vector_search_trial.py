import json
import os
import copy
from dotenv import load_dotenv
from pathlib import Path
from typing import List, Dict, Any
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore, RetrievalMode
from qdrant_client import QdrantClient, models

data_mode = "dev"
dataset = "bird-sql"
DATASET_ROOT_PATH=Path("../dataset")
DATASET_PATH = DATASET_ROOT_PATH / dataset
DATASET_MODE_ROOT_PATH= DATASET_PATH / "dev"
DBS_ROOT_DIR = DATASET_MODE_ROOT_PATH / f"{data_mode}_databases"

DATA = [
  {
    "question_id": 0,
    "db_id": "california_schools",
    "question": "What is the highest eligible free rate for K-12 students in the schools in Alameda County?",
    "evidence": "Eligible free rate for K-12 = `Free Meal Count (K-12)` / `Enrollment (K-12)`",
    "SQL": "SELECT `Free Meal Count (K-12)` / `Enrollment (K-12)` FROM frpm WHERE `County Name` = 'Alameda' ORDER BY (CAST(`Free Meal Count (K-12)` AS REAL) / `Enrollment (K-12)`) DESC LIMIT 1",
    "difficulty": "simple",
    "keywords": [
        "highest eligible free rate",
        "K-12 students",
        "schools",
        "Alameda County",
        "Eligible free rate for K-12",
        "Free Meal Count (K-12)",
        "Enrollment (K-12)"
    ]
  },
  {
    "question_id": 1,
    "db_id": "california_schools",
    "question": "Please list the lowest three eligible free rates for students aged 5-17 in continuation schools.",
    "evidence": "Eligible free rates for students aged 5-17 = `Free Meal Count (Ages 5-17)` / `Enrollment (Ages 5-17)`",
    "SQL": "SELECT `Free Meal Count (Ages 5-17)` / `Enrollment (Ages 5-17)` FROM frpm WHERE `Educational Option Type` = 'Continuation School' AND `Free Meal Count (Ages 5-17)` / `Enrollment (Ages 5-17)` IS NOT NULL ORDER BY `Free Meal Count (Ages 5-17)` / `Enrollment (Ages 5-17)` ASC LIMIT 3",
    "difficulty": "moderate"
  },
  {
    "question_id": 2,
    "db_id": "california_schools",
    "question": "Please list the zip code of all the charter schools in Fresno County Office of Education.",
    "evidence": "Charter schools refers to `Charter School (Y/N)` = 1 in the table fprm",
    "SQL": "SELECT T2.Zip FROM frpm AS T1 INNER JOIN schools AS T2 ON T1.CDSCode = T2.CDSCode WHERE T1.`District Name` = 'Fresno County Office of Education' AND T1.`Charter School (Y/N)` = 1",
    "difficulty": "simple"
  },
  {
    "question_id": 3,
    "db_id": "california_schools",
    "question": "What is the unabbreviated mailing street address of the school with the highest FRPM count for K-12 students?",
    "evidence": "",
    "SQL": "SELECT T2.MailStreet FROM frpm AS T1 INNER JOIN schools AS T2 ON T1.CDSCode = T2.CDSCode ORDER BY T1.`FRPM Count (K-12)` DESC LIMIT 1",
    "difficulty": "simple"
  },
]

def get_embedding_fn():
    """"
    gets embedding model depending on the provider and model name
    """
    
    model_provider = "google" 
    model_name_or_path = "models/embedding-001" 
    
    if model_provider.lower() == "google" or model_provider.lower() == "gemini":
        GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
        os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY
        embeddings = GoogleGenerativeAIEmbeddings(model=model_name_or_path)
        return embeddings
    elif model_provider.lower() == "sentence-transformer":
        embeddings = HuggingFaceEmbeddings(model_name=model_name_or_path)
        return embeddings
    else:
        raise ValueError(f"model_provider = {model_provider} has not been implemented yet.")
    

def load_local_vdb_store(db_id, vdb_embedding_fn):
    """"
    Loads the locally saved vector database index.
    """

    # Set local vector db index path 
    prep_schemaless_dir = DBS_ROOT_DIR / db_id / "prep_schemaless"
    vdb_save_path = prep_schemaless_dir / f"vdb_{db_id}"
    collection_name = f"vdb_{db_id}"

    client = QdrantClient(path=vdb_save_path)
    # Check if collection exists
    if not client.collection_exists(collection_name):
        print(f"❌ Collection {collection_name} does not exist.")
        raise SystemError(f"❌ Collection {collection_name} does not exist.")
    else:
        print(f"✅ Collection {collection_name} exist.")
    
    collection_info = client.get_collection(collection_name)
    print(f"✅ Collection `{collection_name}` loaded. Vector count: {collection_info.points_count}")

    vector_store = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=vdb_embedding_fn,
        vector_name="dense",
        retrieval_mode=RetrievalMode.DENSE # or write simply "dense"
    )

    return vector_store


def load_synthetic_examples(db_id):
    examples_jsonl_path = DBS_ROOT_DIR / db_id / "prep_schemaless" / "sub_schemas" / "column_level" / "sub_schema_examples.jsonl"
    t2s_example_dicts: List[Dict[Any]] = []
    with open(examples_jsonl_path, 'r') as file:
        for line in file:
            try:
                obj = json.loads(line)
                t2s_example_dicts.append(obj)
            except Exception as e:
                continue
    return t2s_example_dicts

def find_examples_from_metadata(t2s_example_dicts: List[Dict[str, Any]], metadatas: List[Dict[str, Any]]):
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

    return selected_examples 


def vector_search_trial():
    db_id = "california_schools"
    vdb_embedding_fn = get_embedding_fn()
    vector_store = load_local_vdb_store(db_id=db_id, vdb_embedding_fn=vdb_embedding_fn)
    synthetic_examples = load_synthetic_examples(db_id)

    data = DATA[:1]
    for idx, t2s_dict in enumerate(data):
        print("\n\n\n ========================================================================")
        question = t2s_dict.get("question")
        evidence = t2s_dict.get("evidence")
        question = question + f" Hint: {evidence}"
        print(f"- Question: {question}")
        print(f"- SQL; {t2s_dict.get('SQL')}\n")

        keywords = [
            "highest eligible free rate",
            "K-12 students",
            "schools",
            "Alameda County",
            "Eligible free rate for K-12",
            "Free Meal Count (K-12)",
            "Enrollment (K-12)"
        ]

        extracted_examples: List[Dict[str, Any]] = []
        for keyword_idx, keyword in enumerate(keywords):
            print(f"Searching for the {keyword} ({keyword_idx+1}/{len(keywords)})")
            similar_questions = vector_store.similarity_search_with_score(
                query=keyword, 
                k=3, 
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="metadata.content_type",
                            match=models.MatchValue(
                                value="question"
                            )
                        )
                    ]
                    # !!!!! WE CAN ALSO ADD FILTER NOT TO EXTRACT THE QUESTION THAT IS ASKED IN TRAINING
                    # WE CAN DO IT BY FILTERING SS_ID AND EXAMPLE_NO
                )
            )
            # print(f"similar_questions exist: {True if similar_questions else False}")

            q_metadatas: List[Dict[str, Any]] = []
            for sim_question, sim_score in similar_questions:  
                    metadata = copy.deepcopy(sim_question.metadata)
                    metadata['score'] = float(sim_score)
                    metadata['search_keyword'] = keyword
                    q_metadatas.append(metadata)
                    # print(f"\t* [SIM={sim_score:3f}] {sim_question.page_content} [{sim_question.metadata}]")
                    # print(f"example_no type: {type(sim_question.metadata.get('example_no'))}")

            similar_examples = find_examples_from_metadata(synthetic_examples, q_metadatas)
            extracted_examples += similar_examples
            # print("\n---\n")
            # for idx, sim_ex_dict in enumerate(similar_examples):
            #     print(f"{idx}---------------------")
            #     print(f"Q: {sim_ex_dict.get('question')}")
            #     print(f"SQL: {sim_ex_dict.get('SQL')}")
            #     print(f"score: {sim_ex_dict.get('score')}")

            similar_sqls = vector_store.similarity_search_with_score(
                query=keyword, 
                k=3, 
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="metadata.content_type",
                            match=models.MatchValue(
                                value="sql"
                            )
                        )
                    ]
                    # !!!!! WE CAN ALSO ADD FILTER NOT TO EXTRACT THE QUESTION THAT IS ASKED IN TRAINING
                    # WE CAN DO IT BY FILTERING SS_ID AND EXAMPLE_NO
                )
            )
            # print(f"similar_sqls exist: {True if similar_sqls else False}") 

            s_metadatas: List[Dict[str, Any]] = []
            for sim_sql, sim_score in similar_sqls:
                metadata = sim_sql.metadata
                metadata['score'] = float(sim_score)
                metadata['search_keyword'] = keyword
                s_metadatas.append(metadata)
        
            similar_examples = find_examples_from_metadata(synthetic_examples, s_metadatas)
            extracted_examples += similar_examples
            # print("\n---\n")
            # for idx, sim_ex_dict in enumerate(similar_examples):
            #     print(f"{idx}---------------------")
            #     print(f"Q: {sim_ex_dict.get('question')}")
            #     print(f"SQL: {sim_ex_dict.get('SQL')}")
            #     print(f"score: {sim_ex_dict.get('score')}")

        for idx, example_dict in enumerate(extracted_examples):
            print(f"{idx}---------------------")
            print(f"Q: {example_dict.get('question')}")
            print(f"SQL: {example_dict.get('SQL')}")
            print(f"score: {example_dict.get('score')}")
            print(f"search_keyword: {example_dict.get('search_keyword')}")

if __name__ == "__main__":
    load_dotenv()
    vector_search_trial()
    