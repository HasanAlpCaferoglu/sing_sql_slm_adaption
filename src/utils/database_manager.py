import os
import pickle
import time
from pathlib import Path
from typing import Callable, Dict, List, Any, Optional
from dotenv import load_dotenv
from threading import Lock

from utils.db_utils.execution import execute_sql, compare_sqls, validate_sql_query, aggregate_sqls, get_execution_status, subprocess_sql_executor
from utils.db_utils.db_info_utils import get_db_all_tables, get_table_all_columns, get_db_schema, get_db_joinables_manuel
from utils.db_utils.sql_parser import get_sql_tables, get_sql_columns_dict, get_sql_condition_literals
from utils.db_utils.schema import DatabaseSchema
from utils.db_utils.schema_generator import DatabaseSchemaGenerator
from utils.db_utils.db_catalog.csv_utils import load_tables_description
from utils.db_utils.db_values.search_db_values import query_lsh

load_dotenv(override=True)
DATASET_ROOT_PATH = Path(os.getenv("DATASET_ROOT_PATH"))
class DatabaseManager:

    """
    A singleton class t manage database operations including schema generation
    """
    _instance = None
    _lock = Lock()

    def __new__(cls, dataset=None, db_mode=None, db_id=None):
        if (db_mode is not None) and (db_id is not None):
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(DatabaseManager, cls).__new__(cls)
                    cls._instance._init(dataset, db_mode, db_id)
                elif cls._instance.db_id != db_id:
                    cls._instance._init(dataset, db_mode, db_id)
                return cls._instance
        else:
            if cls._instance is None:
                raise ValueError("DatabaseManager instance has not been initialized yet.")
            return cls._instance
        
    def _init(self, dataset:str, db_mode: str, db_id: str, prep_dir_name: str="prep_schemaless"):
        """
        Initializes the DatbaseManager instance

        Args:
            db_mode(str): The mode of the datbase ('train', 'dev' or 'test')
            db_id (str): The database identifier
        """
        self.dataset = dataset
        self.db_mode = db_mode
        self.db_id = db_id
        self.prep_dir_name = prep_dir_name
        self._set_paths()
        self.lsh = None
        self.minhashes = None
        
    def _set_paths(self) -> str:
        """
        The function sets the paths for the database files and directories
        """
        self.dataset_root_path = DATASET_ROOT_PATH
        if self.dataset=="bird":
            self.dataset_path = DATASET_ROOT_PATH / "bird-sql"
            self.dataset_mode_root_path = self.dataset_path / self.db_mode # == DB_ROOT_PATH
            self.dbs_root_dir = self.dataset_mode_root_path / f"{self.db_mode}_databases"
            self.db_directory_path = self.dbs_root_dir / self.db_id
            self.db_path = self.dataset_mode_root_path / f"{self.db_mode}_databases" / self.db_id / f"{self.db_id}.sqlite"
        else:
            raise ValueError(f"Your dataset is set to {self.dataset}. Currently this code is not support all datasets.")


    def set_lsh(self) -> str:
        """Sets the LSH and minhashes attributes by loading from pickle files."""
        # prep_dir_name = str(self.args.config['prep_dir_name']) #### ERROR 'DatabaseManager' object has no attribute 'args'
        prep_dir_name = self.prep_dir_name
        with self._lock:
            if self.lsh is None:
                try:
                    start_time = time.time()
                    with (self.db_directory_path / prep_dir_name / f"{self.db_id}_lsh.pkl").open("rb") as file:
                        self.lsh = pickle.load(file)
                    with (self.db_directory_path / prep_dir_name / f"{self.db_id}_minhashes.pkl").open("rb") as file:
                        self.minhashes = pickle.load(file)
                    duration = time.time() - start_time
                    print(f"Database ({self.db_id}) LSH and MinHashes are loaded in {duration} seconds.")
                    return "success"
                except Exception as e:
                    self.lsh = "error"
                    self.minhashes = "error"
                    print(f"Error loading LSH for {self.db_id}: {e}")
                    return "error"
            elif self.lsh == "error":
                return "error"
            else:
                return "success"
            
    def query_lsh(self, keyword: str, signature_size: int = 100, n_gram: int = 3, top_n: int = 10) -> Dict[str, List[str]]:
        """
        Queries the LSH for similar values to the given keyword.

        Args:
            keyword (str): The keyword to search for.
            signature_size (int, optional): The size of the MinHash signature. Defaults to 20.
            n_gram (int, optional): The n-gram size for the MinHash. Defaults to 3.
            top_n (int, optional): The number of top results to return. Defaults to 10.

        Returns:
            Dict[str, List[str]]: The dictionary of similar values.
        """
        
        lsh_status = self.set_lsh()
        if lsh_status == "success":
            return query_lsh(self.lsh, self.minhashes, keyword, signature_size, n_gram, top_n)
        else:
            raise Exception(f"Error loading LSH for {self.db_id}")


    @staticmethod
    def with_db_path(func: Callable):
        """
        Decorator to inject db_path as the first argument to the function.
        """
        def wrapper(self, *args, **kwargs):
            return func(self.db_path, *args, **kwargs)
        return wrapper
    
    @classmethod
    def add_methods_to_class(cls, funcs: List[Callable]):
        """
        Adds methods to the class with db_path automatically provided.

        Args:
            funcs (List(Callable)): List of functions to be added as methods
        """
        for func in funcs:
            method = cls.with_db_path(func)
            setattr(cls, func.__name__, method)

    
    def get_column_profiles(self, schema_with_examples: Dict[str, Dict[str, List[str]]],
                            use_value_description: bool, with_keys: bool, 
                            with_references: bool,
                            tentative_schema: Dict[str, List[str]] = None) -> Dict[str, Dict[str, str]]:
        """
        Generates column profiles for the schema.

        Args:
            schema_with_examples (Dict[str, List[str]]): Schema with example values.
            use_value_description (bool): Whether to use value descriptions.
            with_keys (bool): Whether to include keys.
            with_references (bool): Whether to include references.

        Returns:
            Dict[str, Dict[str, str]]: The dictionary of column profiles.
        """
        schema_with_descriptions = load_tables_description(self.db_directory_path, use_value_description)
        database_schema_generator = DatabaseSchemaGenerator(
            tentative_schema=DatabaseSchema.from_schema_dict(tentative_schema if tentative_schema else self.get_db_schema()),
            schema_with_examples=DatabaseSchema.from_schema_dict_with_examples(schema_with_examples),
            schema_with_descriptions=DatabaseSchema.from_schema_dict_with_descriptions(schema_with_descriptions),
            db_id=self.db_id,
            db_path=self.db_path,
            add_examples=True,
        )
        
        column_profiles = database_schema_generator.get_column_profiles(with_keys, with_references)
        return column_profiles
    
    
    def get_db_joinables_from_schema(self, tentative_schema: Optional[Dict[str, List[str]]] = None) ->  Dict[str, List[str]]:
        """
        Gets the database tables' joinable tables

        """
        database_schema_generator = DatabaseSchemaGenerator(
            tentative_schema=DatabaseSchema.from_schema_dict(tentative_schema if tentative_schema else self.get_db_schema()),
            db_id=self.db_id,
            db_path=self.db_path,
            add_examples=True,
        )
        
        return database_schema_generator.schema_structure.get_db_joinables_from_schema()
    
    

# List of functions to be added to the class
functions_to_add = [
    get_sql_tables,
    get_sql_columns_dict,
    get_sql_condition_literals,
    get_db_all_tables,
    get_table_all_columns,
    get_db_schema,
    get_db_joinables_manuel,
    subprocess_sql_executor,
    execute_sql, 
    compare_sqls,
    validate_sql_query,
    aggregate_sqls,
    get_execution_status
    ]

# Adding methods to the class
DatabaseManager.add_methods_to_class(functions_to_add)