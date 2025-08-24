""""
To run
PYTHONPATH=src python -m data_exploration.schema_filtering_performance
"""

import os
import json
import traceback
from typing import Union, Dict, Any, List, Literal, Tuple, Type, Optional, Set
from pathlib import Path
from utils.db_utils.db_info_utils import get_db_all_tables, get_table_all_columns, get_db_schema
from utils.db_utils.sql_parser import get_filtered_schema_dict_from_similar_examples

from sqlglot import parse_one, exp
from sqlglot.optimizer.qualify import qualify


DATASET_ROOT_PATH = Path("../dataset")
DATASET_DIR_PATH = DATASET_ROOT_PATH / "bird-sql"
DATA_MODE = 'dev'
DATASET_MODE_DIR_PATH = DATASET_DIR_PATH / f"{DATA_MODE}"
DBS_ROOT_DIR = DATASET_ROOT_PATH / "bird-sql" / f"{DATA_MODE}" / f"{DATA_MODE}_databases"
DB_NAME = "california_schools"

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


def get_schema_tables_as_set(schema_dict: Dict[str, List[str]], lowercase: Optional[bool]=False) -> Set[str]:
    """
    For a given schema dict, extract schema tables as a set of table_name case-sensitively.
    
    Args:
        schema_dict: A dictionary where keys are table names and values are lists of column names.
        
    Returns:
        A set of strings in the format "table_name".
    """

    schema_tables_set = set()
    for table_name, columns in schema_dict.items():
        if lowercase:
            schema_tables_set.add(table_name.lower())
        else:
            schema_tables_set.add(table_name)

    return schema_tables_set

def get_schema_columns_as_set(schema_dict: Dict[str, List[str]], lowercase: Optional[bool]=False) -> Set[str]:
    """
    For a given schema dict, extract schema items as a set of table_name.column_name case-sensitively.
    
    Args:
        schema_dict: A dictionary where keys are table names and values are lists of column names.
        
    Returns:
        A set of strings in the format "table_name.column_name".
    """

    schema_items = set()
    for table_name, columns in schema_dict.items():
        for column_name in columns:
            current_schema_item = f"{table_name}.{column_name}"
            if lowercase:
                schema_items.add(current_schema_item.lower())
            else:
                schema_items.add(current_schema_item)
    return schema_items

def calculate_schema_metrics_for_single_schema( used_schema_dict: Dict[str, List[str]], gt_schema_dict: Dict[str, List[str]], scope: Literal["column", "table"]) -> Tuple[int, int, int, float, float, float]:
    """
    Given a schema dict that is used and its ground truth schema dict, compute true_positive (tp), false_positive (fp), false_negative (fn), recall, precision, and F1 score.

    Args:
        used_schema_dict (Dict[str, List[str]]): The schema dictionary used in the process.
        gt_schema_dict (Dict[str, List[str]]): The ground truth schema dictionary.

    Returns:
        Tuple[int, int, int, float, float, float]: recall, precision, f1
    """
    if used_schema_dict == {}:
        return 0.0, 0.0, 0.0
    
    if scope == "column":
        used_schema_items = get_schema_columns_as_set(used_schema_dict, lowercase=True)
        gt_schema_items = get_schema_columns_as_set(gt_schema_dict, lowercase=True)
    elif scope == "table":
        used_schema_items = get_schema_tables_as_set(used_schema_dict, lowercase=True)
        gt_schema_items = get_schema_tables_as_set(gt_schema_dict, lowercase=True)

    true_positives = used_schema_items & gt_schema_items
    num_tp = len(true_positives)
    num_fp = len(used_schema_items - gt_schema_items)
    num_fn = len(gt_schema_items - used_schema_items)

    precision = num_tp / (num_tp + num_fp) if (num_tp + num_fp) > 0 else 0.0
    recall = num_tp / (num_tp + num_fn) if (num_tp + num_fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return num_tp, num_fp, num_fn, round(recall, 2), round(precision, 2), round(f1, 2)

def measure_schema_performance(data: List[Dict[str, Any]], file_path: Union[Path, str], config: Literal['bm25', 'vdb']) -> Dict[str, Any]:

    db_path = DBS_ROOT_DIR/ f"{DB_NAME}"  / f"{DB_NAME}.sqlite"
    schema_performance = {
        "file_path": str(file_path),
        "config": config,
        "overall_schema_performance": {
            "t_num_tp": 0,
            "t_num_fp": 0,
            "t_num_fn": 0,
            "c_num_tp": 0,
            "c_num_fp": 0,
            "c_num_fn": 0,
            "table_precision": 0,
            "table_recall": 0,
            "column_precision": 0,
            "column_recall": 0,
            "column_strict_recall_rate": 0,
            "column_recalls": [],
        },
        "all_few_shot_schema_performance": {
            "t_num_tp": 0,
            "t_num_fp": 0,
            "t_num_fn": 0,
            "c_num_tp": 0,
            "c_num_fp": 0,
            "c_num_fn": 0,
            "table_precision": 0,
            "table_recall": 0,
            "column_precision": 0,
            "column_recall": 0,
            "column_strict_recall_rate": 0,
            "column_recalls": [],
        },
        "topk_few_shot_schema_performance": {
            "top_k": 6,
            "t_num_tp": 0,
            "t_num_fp": 0,
            "t_num_fn": 0,
            "c_num_tp": 0,
            "c_num_fp": 0,
            "c_num_fn": 0,
            "table_precision": 0,
            "table_recall": 0,
            "column_precision": 0,
            "column_recall": 0,
            "column_strict_recall_rate": 0,
            "column_recalls": [],
        }
    }

    for idx, t2s_dict in enumerate(data):
        # print(f"t2s_dict: {t2s_dict}")
        print(f"{idx}/{len(data)}")
        gt_schema_dict = t2s_dict.get("gt_schema_dict", {})
        filtered_schema_dict = t2s_dict.get("filtered_schema", {}).get("schema_dict", {})
        all_few_shot_examples: List[Dict[str, Any]] = t2s_dict.get("few_shot", {}).get("examples", [])

        ################################################
        # Overall Schema Performance Calculation Part
        ################################################
        t_num_tp_single, t_num_fp_single, t_num_fn_single, t_s_recall_single, t_s_precision_single, t_s_f1_single = calculate_schema_metrics_for_single_schema(used_schema_dict=filtered_schema_dict, gt_schema_dict=gt_schema_dict, scope="table") 
        c_num_tp_single, c_num_fp_single, c_num_fn_single, c_s_recall_single, c_s_precision_single, c_s_f1_single = calculate_schema_metrics_for_single_schema(used_schema_dict=filtered_schema_dict, gt_schema_dict=gt_schema_dict, scope="column") 

        schema_performance["overall_schema_performance"]["t_num_tp"] += t_num_tp_single
        schema_performance["overall_schema_performance"]["t_num_fp"] += t_num_fp_single
        schema_performance["overall_schema_performance"]["t_num_fn"] += t_num_fn_single

        schema_performance["overall_schema_performance"]["c_num_tp"] += c_num_tp_single
        schema_performance["overall_schema_performance"]["c_num_fp"] += c_num_fp_single
        schema_performance["overall_schema_performance"]["c_num_fn"] += c_num_fn_single

        schema_performance["overall_schema_performance"]["column_recalls"].append(c_s_recall_single)

        ################################################
        # All Few-Shot Schema Performance Calculation Part
        ################################################
        used_schema_dict = get_filtered_schema_dict_from_similar_examples(
            db_path=db_path, 
            similar_examples=all_few_shot_examples
        )
        t_num_tp_single, t_num_fp_single, t_num_fn_single, t_s_recall_single, t_s_precision_single, t_s_f1_single = calculate_schema_metrics_for_single_schema(used_schema_dict=used_schema_dict, gt_schema_dict=gt_schema_dict, scope="table") 
        c_num_tp_single, c_num_fp_single, c_num_fn_single, c_s_recall_single, c_s_precision_single, c_s_f1_single = calculate_schema_metrics_for_single_schema(used_schema_dict=used_schema_dict, gt_schema_dict=gt_schema_dict, scope="column") 

        schema_performance["all_few_shot_schema_performance"]["t_num_tp"] += t_num_tp_single
        schema_performance["all_few_shot_schema_performance"]["t_num_fp"] += t_num_fp_single
        schema_performance["all_few_shot_schema_performance"]["t_num_fn"] += t_num_fn_single

        schema_performance["all_few_shot_schema_performance"]["c_num_tp"] += c_num_tp_single
        schema_performance["all_few_shot_schema_performance"]["c_num_fp"] += c_num_fp_single
        schema_performance["all_few_shot_schema_performance"]["c_num_fn"] += c_num_fn_single

        schema_performance["all_few_shot_schema_performance"]["column_recalls"].append(c_s_recall_single)

        ################################################
        # Top-k Few-Shot Schema Performance Calculation Part
        ################################################
        k = 6
        top_k_few_shot_examples = all_few_shot_examples[: k]
        schema_performance["topk_few_shot_schema_performance"]["top_k"] = k
        used_schema_dict = get_filtered_schema_dict_from_similar_examples(
            db_path=db_path, 
            similar_examples=top_k_few_shot_examples
        )
        t_num_tp_single, t_num_fp_single, t_num_fn_single, t_s_recall_single, t_s_precision_single, t_s_f1_single = calculate_schema_metrics_for_single_schema(used_schema_dict=used_schema_dict, gt_schema_dict=gt_schema_dict, scope="table") 
        c_num_tp_single, c_num_fp_single, c_num_fn_single, c_s_recall_single, c_s_precision_single, c_s_f1_single = calculate_schema_metrics_for_single_schema(used_schema_dict=used_schema_dict, gt_schema_dict=gt_schema_dict, scope="column") 

        schema_performance["topk_few_shot_schema_performance"]["t_num_tp"] += t_num_tp_single
        schema_performance["topk_few_shot_schema_performance"]["t_num_fp"] += t_num_fp_single
        schema_performance["topk_few_shot_schema_performance"]["t_num_fn"] += t_num_fn_single

        schema_performance["topk_few_shot_schema_performance"]["c_num_tp"] += c_num_tp_single
        schema_performance["topk_few_shot_schema_performance"]["c_num_fp"] += c_num_fp_single
        schema_performance["topk_few_shot_schema_performance"]["c_num_fn"] += c_num_fn_single

        schema_performance["topk_few_shot_schema_performance"]["column_recalls"].append(c_s_recall_single)


    ################################################
    ## Computing the precision and recall
    ################################################

    # overall_schema_performance - table_precision & table_recall
    num_tp = schema_performance["overall_schema_performance"]["t_num_tp"]
    num_fp = schema_performance["overall_schema_performance"]["t_num_fp"]
    num_fn = schema_performance["overall_schema_performance"]["t_num_fn"]
    schema_performance["overall_schema_performance"]["table_precision"] = num_tp / (num_tp + num_fp) if (num_tp + num_fp) > 0 else 0.0
    schema_performance["overall_schema_performance"]["table_recall"] = num_tp / (num_tp + num_fn) if (num_tp + num_fn) > 0 else 0.0

    # overall_schema_performance - column_precision & column_recall
    num_tp = schema_performance["overall_schema_performance"]["c_num_tp"]
    num_fp = schema_performance["overall_schema_performance"]["c_num_fp"]
    num_fn = schema_performance["overall_schema_performance"]["c_num_fn"]
    schema_performance["overall_schema_performance"]["column_precision"] =  num_tp / (num_tp + num_fp) if (num_tp + num_fp) > 0 else 0.0
    schema_performance["overall_schema_performance"]["column_recall"] = num_tp / (num_tp + num_fn) if (num_tp + num_fn) > 0 else 0.0

    # overall_schema_performance - strict schema recall rate
    ones = 0
    column_recalls = schema_performance["overall_schema_performance"]["column_recalls"]
    for recall_value in column_recalls:
        if recall_value == 1.0:
            ones+=1
    schema_performance["overall_schema_performance"]["column_strict_recall_rate"] = ones / len(column_recalls) if len(column_recalls) > 0 else 0.0
    schema_performance["overall_schema_performance"]["column_recalls"] = []

    # all_few_shot_schema_performance - table_precision & table_recall
    num_tp = schema_performance["all_few_shot_schema_performance"]["t_num_tp"]
    num_fp = schema_performance["all_few_shot_schema_performance"]["t_num_fp"]
    num_fn = schema_performance["all_few_shot_schema_performance"]["t_num_fn"]
    schema_performance["all_few_shot_schema_performance"]["table_precision"] = num_tp / (num_tp + num_fp) if (num_tp + num_fp) > 0 else 0.0
    schema_performance["all_few_shot_schema_performance"]["table_recall"] = num_tp / (num_tp + num_fn) if (num_tp + num_fn) > 0 else 0.0
    
    # all_few_shot_schema_performance - column_precision & column_recall
    num_tp = schema_performance["all_few_shot_schema_performance"]["c_num_tp"]
    num_fp = schema_performance["all_few_shot_schema_performance"]["c_num_fp"]
    num_fn = schema_performance["all_few_shot_schema_performance"]["c_num_fn"]
    schema_performance["all_few_shot_schema_performance"]["column_precision"] = num_tp / (num_tp + num_fp) if (num_tp + num_fp) > 0 else 0.0
    schema_performance["all_few_shot_schema_performance"]["column_recall"] = num_tp / (num_tp + num_fn) if (num_tp + num_fn) > 0 else 0.0

    # all_few_shot_schema_performance - strict schema recall rate
    ones = 0
    column_recalls = schema_performance["all_few_shot_schema_performance"]["column_recalls"]
    for recall_value in column_recalls:
        if recall_value == 1.0:
            ones+=1
    schema_performance["all_few_shot_schema_performance"]["column_strict_recall_rate"] = ones / len(column_recalls) if len(column_recalls) > 0 else 0.0
    schema_performance["all_few_shot_schema_performance"]["column_recalls"] = []

    # topk_few_shot_schema_performance - table_precision & table_recall
    num_tp = schema_performance["topk_few_shot_schema_performance"]["t_num_tp"]
    num_fp = schema_performance["topk_few_shot_schema_performance"]["t_num_fp"]
    num_fn = schema_performance["topk_few_shot_schema_performance"]["t_num_fn"]
    schema_performance["topk_few_shot_schema_performance"]["table_precision"] = num_tp / (num_tp + num_fp) if (num_tp + num_fp) > 0 else 0.0
    schema_performance["topk_few_shot_schema_performance"]["table_recall"] = num_tp / (num_tp + num_fn) if (num_tp + num_fn) > 0 else 0.0
    
    # topk_few_shot_schema_performance - column_precision & column_recall
    num_tp = schema_performance["topk_few_shot_schema_performance"]["c_num_tp"]
    num_fp = schema_performance["topk_few_shot_schema_performance"]["c_num_fp"]
    num_fn = schema_performance["topk_few_shot_schema_performance"]["c_num_fn"]
    schema_performance["topk_few_shot_schema_performance"]["column_precision"] = num_tp / (num_tp + num_fp) if (num_tp + num_fp) > 0 else 0.0
    schema_performance["topk_few_shot_schema_performance"]["column_recall"] = num_tp / (num_tp + num_fn) if (num_tp + num_fn) > 0 else 0.0

    # topk_few_shot_schema_performance - strict schema recall rate
    ones = 0
    column_recalls = schema_performance["topk_few_shot_schema_performance"]["column_recalls"]
    for recall_value in column_recalls:
        if recall_value == 1.0:
            ones+=1
    schema_performance["topk_few_shot_schema_performance"]["column_strict_recall_rate"] = ones / len(column_recalls) if len(column_recalls) > 0 else 0.0
    schema_performance["topk_few_shot_schema_performance"]["column_recalls"] = []


    return schema_performance

def main():

    schema_performance_output_dir = Path(f"./data_exploration/schema_performance/{DB_NAME}")
    schema_performance_output_dir.mkdir(parents=True, exist_ok=True)

    # Explore the bird dev with few-shot
    print("Measure Schema Performance of dev_with_few_shot.json")
    bird_dev_with_fs_json_file_path = DATASET_MODE_DIR_PATH / "dev_with_few_shots.json"
    bird_dev_with_fs_data = load_data(bird_dev_with_fs_json_file_path)
    bird_dev_with_fs_db_data = [t2s_dict for t2s_dict in bird_dev_with_fs_data if t2s_dict['db_id'] == DB_NAME]
    schema_performance_1 = measure_schema_performance(data=bird_dev_with_fs_db_data, file_path=bird_dev_with_fs_json_file_path, config="bm25")
    schema_performance_1_file_path = schema_performance_output_dir / "bird_dev_with_few_shots_bm25.json"
    with open(schema_performance_1_file_path, "w") as file:
        json.dump(schema_performance_1, file, indent=4)
    
    # Explore the bird dev with few-shot where vector db is used for few-shot selection
    print("Measure Schema Performance of dev_with_few_shot_v10_important.json")
    bird_dev_with_vfs_json_file_path = DATASET_MODE_DIR_PATH / "dev_with_few_shots_v10_important.json"
    bird_dev_with_vfs_data = load_data(bird_dev_with_vfs_json_file_path)
    bird_dev_with_vfs_db_data = [t2s_dict for t2s_dict in bird_dev_with_vfs_data if t2s_dict['db_id'] == DB_NAME]
    schema_performance_2 = measure_schema_performance(data=bird_dev_with_vfs_db_data, file_path=bird_dev_with_vfs_json_file_path, config="vdb")
    schema_performance_2_file_path = schema_performance_output_dir / "bird_dev_with_few_shots_vdb.json"
    with open(schema_performance_2_file_path, "w") as file:
        json.dump(schema_performance_2, file, indent=4)

    # Explore the Synthetic Train Dataset
    print("Measure Schema Performance of Synthetic Train Dataset")
    synthetic_train_split_path = DBS_ROOT_DIR / f"{DB_NAME}" / "prep_schemaless" / "sub_schemas" / "column_level" / "sub_schema_examples_train_with_few_shots.jsonl"
    synthetic_train_split_data = load_data(synthetic_train_split_path)
    schema_performance_3 = measure_schema_performance(data=synthetic_train_split_data, file_path=synthetic_train_split_path, config="bm25")
    schema_performance_3_file_path = schema_performance_output_dir / "train_split.json"
    with open(schema_performance_3_file_path, "w") as file:
        json.dump(schema_performance_3, file, indent=4)

    # Explore the Synthetic Dev Dataset
    print("Measure Schema Performance of Synthetic Dev Dataset")
    synthetic_dev_split_path = DBS_ROOT_DIR / f"{DB_NAME}" / "prep_schemaless" / "sub_schemas" / "column_level" / "sub_schema_examples_dev_with_few_shots.jsonl"
    synthetic_dev_split_data = load_data(synthetic_dev_split_path)
    schema_performance_4 = measure_schema_performance(data=synthetic_dev_split_data, file_path=synthetic_dev_split_path, config="bm25")
    schema_performance_4_file_path = schema_performance_output_dir / "dev_split.json"
    with open(schema_performance_4_file_path, "w") as file:
        json.dump(schema_performance_4, file, indent=4)
    

    # Explore the Synthetic Dev Dataset
    print("Measure Schema Performance of Synthetic Test Dataset")
    synthetic_test_split_path = DBS_ROOT_DIR / f"{DB_NAME}" / "prep_schemaless" / "sub_schemas" / "column_level" / "sub_schema_examples_test_with_few_shots.jsonl"
    synthetic_test_split_data = load_data(synthetic_test_split_path)
    schema_performance_5 = measure_schema_performance(data=synthetic_test_split_data, file_path=synthetic_test_split_path, config="bm25")
    schema_performance_5_file_path = schema_performance_output_dir / "test_split.json"
    with open(schema_performance_5_file_path, "w") as file:
        json.dump(schema_performance_5, file, indent=4)
   

    return


if __name__ == "__main__":
    main()