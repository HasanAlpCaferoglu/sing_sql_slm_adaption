""""
To run:
PYTHONPATH=src python -m data_exploration.explore_db_data
"""
from __future__ import annotations
import os
import json
from pathlib import Path
from typing import Union, Dict, Any, List, Literal, Tuple, Type
from utils.db_utils.db_info import DatabaseGeneralInfo
from utils.db_utils.schema import DatabaseSchema, TableSchema, ColumnInfo
from utils.db_utils.sql_parser import get_sql_columns_dict
from utils.db_utils.DatabaseDataTracker import DatabaseDataTracker

from google import genai

from sqlglot import parse
from sqlglot import exp
from sqlglot.errors import ParseError


DATASET_ROOT_PATH = Path("../dataset")
DATASET_DIR_PATH = DATASET_ROOT_PATH / "bird-sql"
DATA_MODE = 'dev'
DATASET_MODE_DIR_PATH = DATASET_DIR_PATH / f"{DATA_MODE}"
DBS_ROOT_DIR = DATASET_ROOT_PATH / "bird-sql" / f"{DATA_MODE}" / f"{DATA_MODE}_databases"
DB_NAME = "california_schools"

AGGREGATION_FUNCTIONS = ["AVG", "COUNT", "GROUP_CONCAT", "MAX", "MIN", "SUM", "TOTAL"]
WINDOW_FUNCTIONS = ["ROW_NUMBER", "RANK", "DENSE_RANK", "PERCENT_RANK", "CUME_DIST", "NTILE", "LAG", "LEAD", "FIRST_VALUE", "LAST_VALUE", "NTH_VALUE", "OVER ("]


GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
GOOGLE_GENAI_CLIENT = genai.Client(api_key=GEMINI_API_KEY)
GOOGLE_MODEL = "gemini-2.0-flash"

print(f"Current working directory: {os.getcwd()}")

def load_data(data_path: Union[Path, str]) -> List[Dict[str, Any]]:
    """
    Loads the dataset from the specified path.

    Args:
        data_path (str): Path to the data file.

    Returns:
        List[Dict[str, Any]]: The loaded dataset.
    """
    print(f"Loading data from {Path(data_path).resolve()}...")
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


def query_uses_window_functions(sql: str, *, dialect: str = "sqlite") -> bool:
    """
    Return True iff the SQL uses window functionality, determined via sqlglot's AST.

    This detects any of the following in any statement:
      1) An explicit Window node (e.g., ROW_NUMBER() OVER (...)).
      2) Any expression carrying an 'over' argument (some functions attach the window spec here).
      3) A WINDOW clause that defines named windows (e.g., WINDOW w AS (...)).

    Parameters
    ----------
    sql : str
        SQL string (may contain one or more statements).
    dialect : str, optional (default: "sqlite")
        Dialect hint for sqlglot's parser.

    Returns
    -------
    bool
        True if any statement contains window usage; otherwise False.

    Raises
    ------
    ValueError
        If the SQL cannot be parsed by sqlglot for the given dialect.
    """
    try:
        statements: List[exp.Expression] = parse(sql, read=dialect)
    except ParseError as e:
        raise ValueError(f"Failed to parse SQL with sqlglot (dialect='{dialect}'): {e}") from e

    for stmt in statements:
        # 1) Direct Window nodes anywhere in the tree
        if any(isinstance(node, exp.Window) for node in stmt.walk()):
            return True

        # 2) Any node with an 'over' argument (e.g., Function(..., over=Window(...)))
        for node in stmt.walk():
            if getattr(node, "args", None) and node.args.get("over"):
                return True

        # 3) Presence of a WINDOW clause (named windows)
        if stmt.args.get("windows"):
            return True

    return False

def query_uses_aggregate_functions(sql: str, *, dialect: str = "sqlite") -> bool:
    """
    Return True iff the SQL calls any aggregate function (e.g., COUNT, SUM, AVG, MIN, MAX,
    GROUP_CONCAT). Windowed aggregates like SUM(x) OVER (...) also count.
    """
    try:
        statements: List[exp.Expression] = parse(sql, read=dialect)
    except ParseError as e:
        raise ValueError(f"Failed to parse SQL with sqlglot (dialect='{dialect}'): {e}") from e

    # Core aggregates (SQLite) + a few cross-dialect extras (harmless if absent)
    known_agg_classes: Tuple[Type[exp.Expression], ...] = tuple(
        cls for cls in [
            getattr(exp, "Count", None),
            getattr(exp, "Sum", None),
            getattr(exp, "Avg", None),
            getattr(exp, "Min", None),
            getattr(exp, "Max", None),
            getattr(exp, "GroupConcat", None),
            getattr(exp, "ArrayAgg", None),
            getattr(exp, "StringAgg", None),
            getattr(exp, "Stddev", None),
            getattr(exp, "StddevPop", None),
            getattr(exp, "StddevSamp", None),
            getattr(exp, "VarPop", None),
            getattr(exp, "VarSamp", None),
            getattr(exp, "Variance", None),
            getattr(exp, "ApproxCountDistinct", None),
        ] if cls is not None
    )

    for stmt in statements:
        for node in stmt.walk():
            agg_attr = getattr(node, "is_aggregate", None)
            try:
                if bool(agg_attr() if callable(agg_attr) else agg_attr):
                    return True
            except Exception:
                pass  # fall through to class check

            if known_agg_classes and isinstance(node, known_agg_classes):
                return True

    return False

def count_joins_in_sql(sql: str, *, dialect: str = "sqlite") -> int:
    """
    Count the number of JOIN operations in the given SQL string using sqlglot's AST.

    What is counted
    ---------------
    1) **Explicit JOINs** (INNER/LEFT/RIGHT/FULL/CROSS/NATURAL/LATERAL) via `exp.Join` nodes.
    2) **Implicit comma joins** in FROM lists, e.g. `FROM a, b, c` counts as 2.

    The function traverses all statements (CTEs, subqueries, etc.) and returns the total.

    Parameters
    ----------
    sql : str
        SQL string, possibly containing multiple statements.
    dialect : str, optional
        Dialect hint for sqlglot parsing. Defaults to "sqlite".

    Returns
    -------
    int
        Total number of join operations detected across all statements.

    Raises
    ------
    ValueError
        If parsing fails for the provided dialect.
    """
    try:
        statements: List[exp.Expression] = parse(sql, read=dialect)
    except ParseError as e:
        raise ValueError(f"Failed to parse SQL with sqlglot (dialect='{dialect}'): {e}") from e

    total = 0

    for stmt in statements:
        # 1) Count explicit JOIN nodes anywhere in the tree.
        total += sum(1 for node in stmt.walk() if isinstance(node, exp.Join))

        # 2) Count implicit comma joins in FROM clauses: n items => n-1 joins.
        #    (sqlglot represents comma-separated sources as From(expressions=[...]))
        for from_node in stmt.find_all(exp.From):
            exprs = from_node.args.get("expressions") or []
            if len(exprs) > 1:
                total += len(exprs) - 1

    return total


def explore_data(data: List[Dict[str, Any]], dataset_name: Literal['bird', 'synthetic']) -> Dict[str, Any]:
    
    db_info = DatabaseGeneralInfo(dbs_root_dir=DBS_ROOT_DIR, db_id=DB_NAME)
    
    question_counts_per_level = {
        "overall": 0,
        "simple": 0,
        "moderate": 0,
        "challenging": 0,
        "window": 0
    }

    total_join_cnt_per_level = {
        "overall": 0,
        "simple": 0,
        "moderate": 0,
        "challenging": 0,
        "window": 0,
    }

    join_cnt_per_sql_per_level = {
        "overall": 0,
        "simple": 0,
        "moderate": 0,
        "challenging": 0,
        "window": 0,
    }
    
    total_token_cnt_per_level_for_sql = {
        "overall": 0,
        "simple": 0,
        "moderate": 0,
        "challenging": 0,
        "window": 0,
    }

    token_cnt_per_sql_per_level = {
        "overall": 0,
        "simple": 0,
        "moderate": 0,
        "challenging": 0,
        "window": 0,
    }
    
    total_token_cnt_per_level_for_question = {
        "overall": 0,
        "simple": 0,
        "moderate": 0,
        "challenging": 0,
        "window": 0,
    }

    token_cnt_per_question_per_level = {
        "overall": 0,
        "simple": 0,
        "moderate": 0,
        "challenging": 0,
        "window": 0,
    }

    aggregation_cnt_per_level = {
        "overall": 0,
        "simple": 0,
        "moderate": 0,
        "challenging": 0,
        "window": 0,
    }

    aggregation_percentage_per_level = {
        "overall": 0,
        "simple": 0,
        "moderate": 0,
        "challenging": 0,
        "window": 0,
    }

    db_data_tracker = DatabaseDataTracker(db_id=db_info.db_id, db_path=db_info.db_path)
    for idx, t2s_dict in enumerate(data):
        print(f"idx: {idx}")
        # if idx % 100 == 0:
        #     print(f"idx: {idx}")
            
        sql = t2s_dict.get("SQL")
        question = t2s_dict.get("question")
        ##### Question Count Per Difficulty Level #####
        question_counts_per_level["overall"] += 1
        level = t2s_dict.get("difficulty", "").lower()
        question_counts_per_level[f"{level}"] += 1
        # check if the question includes window function
        include_window_funtion = query_uses_window_functions(sql=sql)
        if level != "window" and include_window_funtion: # for bird dataset and the text-sql pairs in synthetic data containing window function in difficulty level other than window
            question_counts_per_level["window"] += 1

        ##### Join Counts #####
        sql_join_cnt = count_joins_in_sql(sql=sql)
        total_join_cnt_per_level["overall"] += sql_join_cnt
        total_join_cnt_per_level[level] += sql_join_cnt
        if level != "window" and include_window_funtion:
            total_join_cnt_per_level["window"] += sql_join_cnt

        ##### Token Count For SQL##### 
        sql_token_cnt = GOOGLE_GENAI_CLIENT.models.count_tokens(model=GOOGLE_MODEL , contents=sql).total_tokens
        total_token_cnt_per_level_for_sql["overall"] += sql_token_cnt
        total_token_cnt_per_level_for_sql[level] += sql_token_cnt
        if level != "window" and include_window_funtion:
            total_token_cnt_per_level_for_sql["window"] += sql_token_cnt
        
        ##### Token Count for Question #####
        question_token_cnt = GOOGLE_GENAI_CLIENT.models.count_tokens(model=GOOGLE_MODEL , contents=question).total_tokens
        total_token_cnt_per_level_for_question["overall"] += question_token_cnt
        total_token_cnt_per_level_for_question[level] += question_token_cnt
        if level != "window" and include_window_funtion:
            total_token_cnt_per_level_for_question["window"] += question_token_cnt
        
        ##### Aggregation Usage #####
        use_aggregate = query_uses_aggregate_functions(sql=sql)
        if use_aggregate:
            aggregation_cnt_per_level["overall"] += 1
            aggregation_cnt_per_level[level] += 1
            if level != "window" and include_window_funtion:
                aggregation_cnt_per_level["window"] += 1

        ##### Coverage #### 
        sql_columns_dict = get_sql_columns_dict(db_path=db_info.db_path, sql=sql)
        
        for sql_t_name, sql_c_name_list in sql_columns_dict.items():
            for sql_c_name in sql_c_name_list:
                db_data_tracker.increase_column_count(table_name=sql_t_name, column_name=sql_c_name)

    ### join_cnt_per_sql_per_level
    for dl in join_cnt_per_sql_per_level.keys():
            join_cnt_per_sql_per_level[dl] = total_join_cnt_per_level[dl] / question_counts_per_level[dl] if question_counts_per_level[dl] != 0 else 0

    ### token_cnt_per_sql_per_level
    for dl in token_cnt_per_sql_per_level.keys():
            token_cnt_per_sql_per_level[dl] = total_token_cnt_per_level_for_sql[dl] / question_counts_per_level[dl] if question_counts_per_level[dl] != 0 else 0

    ### token_cnt_per_question_per_level
    for dl in token_cnt_per_question_per_level.keys():
            token_cnt_per_question_per_level[dl] = total_token_cnt_per_level_for_question[dl] / question_counts_per_level[dl] if question_counts_per_level[dl] != 0 else 0

    ### aggregation_percentage_per_level
    for dl in aggregation_percentage_per_level.keys():
            aggregation_percentage_per_level[dl] = aggregation_cnt_per_level[dl] / question_counts_per_level[dl] if question_counts_per_level[dl] != 0 else 0


    data_exploration_dict = {
        "question_counts_per_level": question_counts_per_level,
        "total_join_cnt_per_level": total_join_cnt_per_level,
        "join_cnt_per_sql_per_level": join_cnt_per_sql_per_level,
        "total_token_cnt_per_level_for_sql": total_token_cnt_per_level_for_sql,
        "token_cnt_per_sql_per_level": token_cnt_per_sql_per_level,
        "total_token_cnt_per_level_for_question": total_token_cnt_per_level_for_question,
        "token_cnt_per_question_per_level": token_cnt_per_question_per_level,
        "aggregation_cnt_per_level": aggregation_cnt_per_level,
        "aggregation_percentage_per_level": aggregation_percentage_per_level,
        "db_columns_count_in_data": db_data_tracker.get_column_counts_lower()
    }

    return data_exploration_dict

def main():
    """
    Exploring the data
    """

    report_file_path = Path(f"./data_exploration/data_exploration_{DB_NAME}.json")
    report_file_path.parent.mkdir(parents=True, exist_ok=True)  # create folder if missing
    with open(report_file_path, "w") as file:
        json.dump({}, file, indent=4)

    # Explore the Bird Dev dataset
    print("Explore the Bird Dev dataset")
    bird_dev_json_file_path = DATASET_MODE_DIR_PATH / "dev.json"
    bird_dev_data = load_data(bird_dev_json_file_path)
    bird_dev_db_data = [t2s_dict for t2s_dict in bird_dev_data if t2s_dict['db_id'] == DB_NAME]
    bird_dev_db_data_exploration_report = explore_data(data=bird_dev_db_data, dataset_name='bird')


    # Explore the Synthetic Train Dataset
    print("Explore the Synthetic Train Dataset")
    synthetic_train_split_path = DBS_ROOT_DIR / f"{DB_NAME}" / "prep_schemaless" / "sub_schemas" / "column_level" / "sub_schema_examples_train.jsonl"
    synthetic_train_split_data = load_data(synthetic_train_split_path)
    synthetic_train_split_data_exploration_report = explore_data(data=synthetic_train_split_data, dataset_name="synthetic")

    # Explore the Synthetic Dev Dataset
    print("Explore the Synthetic Dev Dataset")
    synthetic_dev_split_path = DBS_ROOT_DIR / f"{DB_NAME}" / "prep_schemaless" / "sub_schemas" / "column_level" / "sub_schema_examples_dev.jsonl"
    synthetic_dev_split_data = load_data(synthetic_dev_split_path)
    synthetic_dev_split_data_exploration_report = explore_data(data=synthetic_dev_split_data, dataset_name="synthetic")
    
    # Explore the Synthetic Test Dataset
    print("Explore the Synthetic Test Dataset")
    synthetic_test_split_path = DBS_ROOT_DIR / f"{DB_NAME}" / "prep_schemaless" / "sub_schemas" / "column_level" / "sub_schema_examples_test.jsonl"
    synthetic_test_split_data = load_data(synthetic_test_split_path)
    synthetic_test_split_data_exploration_report = explore_data(data=synthetic_test_split_data, dataset_name="synthetic")


    data_exploration_dict  = {
        "database_name": DB_NAME,
        "bird_dev": bird_dev_db_data_exploration_report,
        "syn_train": synthetic_train_split_data_exploration_report,
        "syn_dev": synthetic_dev_split_data_exploration_report,
        "syn_test": synthetic_test_split_data_exploration_report
    }


    # Write report
    data_exploration_dir_path = Path(f"./data_exploration/data_exploration_details/{DB_NAME}")
    data_exploration_dir_path.mkdir(parents=True, exist_ok=True)
    report_file_path =  data_exploration_dir_path / f"data_exploration_{DB_NAME}.json"
    with open(report_file_path, "w") as file:
        json.dump(data_exploration_dict, file, indent=4)

    return

if __name__ == "__main__":
    main()