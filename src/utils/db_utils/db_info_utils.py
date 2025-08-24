import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple
from itertools import chain, combinations
from dataclasses import dataclass, field
from func_timeout import func_timeout, FunctionTimedOut
from utils.db_utils.execution import execute_sql
from utils.db_utils.db_catalog.csv_utils import load_tables_description


def get_db_all_tables(db_path: str) -> List[str]:
    """
    Retrieves all table names from the database.
    
    Args:
        db_path (str): The path to the database file.
        
    Returns:
        List[str]: A list of table names.
    """
    try:
        raw_table_names = execute_sql(db_path, "SELECT name FROM sqlite_master WHERE type='table';")
        return [table[0].replace('\"', '').replace('`', '') for table in raw_table_names if table[0] != "sqlite_sequence"]
    except Exception as e:
        logging.error(f"Error in get_db_all_tables: {e}")
        raise e

def get_table_all_columns(db_path: str, table_name: str) -> List[str]:
    """
    Retrieves all column names for a given table.
    
    Args:
        db_path (str): The path to the database file.
        table_name (str): The name of the table.
        
    Returns:
        List[str]: A list of column names.
    """
    try:
        table_info_rows = execute_sql(db_path, f"PRAGMA table_info(`{table_name}`);")
        return [row[1].replace('\"', '').replace('`', '') for row in table_info_rows]
    except Exception as e:
        logging.error(f"Error in get_table_all_columns: {e}\nTable: {table_name}")
        raise e

def get_db_schema(db_path: str) -> Dict[str, List[str]]:
    """
    Retrieves the schema of the database.
    
    Args:
        db_path (str): The path to the database file.
        
    Returns:
        Dict[str, List[str]]: A dictionary mapping table names to lists of column names.
    """
    try:
        table_names = get_db_all_tables(db_path)
        return {table_name: get_table_all_columns(db_path, table_name) for table_name in table_names}
    except Exception as e:
        logging.error(f"Error in get_db_schema: {e}")
        raise e

def are_two_tables_joinable_manuel(db_path: str, t1_name: str, t2_name: str) -> Tuple[bool, Dict[str, Dict[str, Dict[str, str]]]]:
    """
    This function controls whether given two table names are joinable or not.

    Args:
        db_path (str): The path to the database file.
        tname_1 (str): Name of the first table.
        t2_name (str): Name of the second table.

    Returns:
        bool: wheter given two table is joinable or not
        Dict[str, Dict[str, List[Tuple(str, str)]]]: List of foreing keys 
    """

    t1_col_names = get_table_all_columns(db_path, t1_name)
    t2_col_names = get_table_all_columns(db_path, t2_name)
    fk_dict = {}
    for t1_col in t1_col_names:
        for t2_col in t2_col_names:
            sql_query = f"SELECT * FROM {t1_name} INNER JOIN {t2_name} ON {t1_name}.{t1_col} = {t2_name}.{t2_col}"
            try:
                execution_result = execute_sql(db_path=db_path, sql=sql_query, fetch="all")
            except FunctionTimedOut:
                print("Timeout in are_two_tables_joinable_manuel")
                continue
            print(f"query response for {sql_query}: \n {execution_result}")
            if (execution_result is None) or (execution_result == []):
                continue
            elif execution_result:
                if fk_dict.get(t1_name, None):
                    fk_dict[t1_name][t1_col] = (t2_name, t2_col)
                else:
                    fk_dict[t1_name] = {}
                    fk_dict[t1_name][t1_col] = (t2_name, t2_col)

                if fk_dict.get(t2_name, None):
                    fk_dict[t2_name][t2_col] = (t1_name, t1_col)
                else:
                    fk_dict[t2_name] = {}
                    fk_dict[t2_name][t2_col] = (t1_name, t1_col)
    
    if fk_dict != {}:
        return True, fk_dict
    else:
        return False, fk_dict

def get_db_joinables_manuel(db_path:str) -> Dict[str, List[str]]:
    """
    This function construct a dict storing each table's joinable tables

    Args: 
        db_path (str): The path to the database file.

    Returns:
        Dict[str, List[str]]: A dictionary where each table maps to a list of tables it can join with.
    """
    db_schema_dict = get_db_schema(db_path)
    db_joinables = {t_name: [] for t_name in db_schema_dict.keys()}
    
    table_combinations = combinations(db_schema_dict.keys(), 2)

    for t1_name, t2_name in table_combinations:
        is_joinable, _ = are_two_tables_joinable_manuel(db_path, t1_name, t2_name)
        if is_joinable:
            db_joinables[t1_name].append(t2_name)
            db_joinables[t2_name].append(t1_name)

    return db_joinables

def are_tables_joinable(table_names: List[str], db_joinables: Dict[str, List[str]]) -> bool:
    """
    This function controls whether given tables can be joined all together.
    To be able to join given all tables, two condition should be met:
        1) Each table should be joinable at least one table (otherthan itself) in the table set
        2) The number of joinable table pairs should be at least (table count in the set - 1)

    Args:
        table_names (List[str]): List of table names that are checked whether they can be joined or not
        joinables (Dict[str, List[str]]): A dictionary where each table maps to a list of tables it can join with.
    Retuns:
        bool: a boolean variable indicating given tables can be joinable all together
    """
    if len(table_names) == 1:
        return True
    
    condition_1 = False # represents the following requirement: Each table should be joinable at least one table (otherthan itself) in the table set
    condition_2 = False # represents the following requirement: The number of joinable table pairs should be at least (table count in the set - 1)

    table_cnt = len(table_names)
    min_pair_join_cnt = table_cnt - 1
    tables_joint_cnt_dict = {t_name: 0 for t_name in table_names}

    
    joinable_pair_cnt = 0
    for t1_ind in range(table_cnt - 1):
        t1_name = table_names[t1_ind]
        for t2_ind in range(t1_ind+1, table_cnt):
            t2_name = table_names[t2_ind]
            if t2_name in db_joinables.get(t1_name, None):
                joinable_pair_cnt += 1
                tables_joint_cnt_dict[t1_name] += 1
                tables_joint_cnt_dict[t2_name] += 1


    if joinable_pair_cnt < min_pair_join_cnt: 
        # Condition 2 unsatisfied
        return False

    for t_name, j_value in tables_joint_cnt_dict.items():
        # condition 1 unsatisfied
        if j_value == 0:
            return False    

    return True

