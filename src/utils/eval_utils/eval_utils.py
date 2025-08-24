from typing import List, Dict, Tuple, Union, Any, Set
from pathlib import Path
from utils.db_utils.sql_parser import get_sql_columns_dict, get_schema_items_as_set, get_sql_schema_items_as_set



def calculate_schema_metrics_for_single_schema( used_schema_dict: Dict[str, List[str]], gt_schema_dict: Dict[str, List[str]]) -> Tuple[int, int, int, float, float, float]:
    """
    Given a schema dict that is used and its ground truth schema dict, compute true_positive (tp), false_positive (fp), false_negative (fn), recall, precision, and F1 score.

    Args:
        used_schema_dict (Dict[str, List[str]]): The schema dictionary used in the process.
        gt_schema_dict (Dict[str, List[str]]): The ground truth schema dictionary.

    Returns:
        Tuple[int, int, int, float, float, float]: recall, precision, f1
    """
    if used_schema_dict == {}:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    
    used_schema_items = get_schema_items_as_set(used_schema_dict, lowercase=True)
    gt_schema_items = get_schema_items_as_set(gt_schema_dict, lowercase=True)

    true_positives = used_schema_items & gt_schema_items
    num_tp = len(true_positives)
    num_fp = len(used_schema_items - gt_schema_items)
    num_fn = len(gt_schema_items - used_schema_items)

    precision = num_tp / (num_tp + num_fp) if (num_tp + num_fp) > 0 else 0.0
    recall = num_tp / (num_tp + num_fn) if (num_tp + num_fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return num_tp, num_fp, num_fn, round(recall, 2), round(precision, 2), round(f1, 2)


def calculate_schema_metrics_for_single_sql(db_path: Union[str, Path], sql: str, gt_sql: str) -> Tuple[float, float, float]:
    """
    For the schema used in a given SQL query and its ground truth compute recall, precision, and F1 score.

    Args:
        sql (str): Given sql query
        gt_sql (str): The ground truth sql

    Returns:
        Tuple[float, float, float]: recall, precision, f1
    """
    used_schema_dict = get_sql_columns_dict(db_path=db_path, sql=sql)
    gt_schema_dict = get_sql_columns_dict(db_path=db_path, sql=gt_sql)

    used_schema_items = get_schema_items_as_set(used_schema_dict, lowercase=True)
    gt_schema_items = get_schema_items_as_set(gt_schema_dict, lowercase=True)

    true_positives = used_schema_items & gt_schema_items
    num_tp = len(true_positives)
    num_fp = len(used_schema_items - gt_schema_items)
    num_fn = len(gt_schema_items - used_schema_items)

    precision = num_tp / (num_tp + num_fp) if (num_tp + num_fp) > 0 else 0.0
    recall = num_tp / (num_tp + num_fn) if (num_tp + num_fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return round(recall, 2), round(precision, 2), round(f1, 2)