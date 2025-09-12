import os
import json
import random
import logging
import sqlite3
from pathlib import Path
from typing import Any, Union, List, Dict, Tuple
from func_timeout import func_timeout, FunctionTimedOut
from multiprocessing import Process, Queue
import threading
from queue import Empty
from enum import Enum
from sqlglot import parse_one, exp

class TimeoutException(Exception):
    pass

def connect_db(sql_dialect, db_path):
    if sql_dialect == "SQLite":
        conn = sqlite3.connect(db_path)
    # elif sql_dialect == "MySQL":
    #     conn = connect_mysql()
    # elif sql_dialect == "PostgreSQL":
    #     conn = connect_postgresql()
    else:
        raise ValueError("Unsupported SQL dialect")
    return conn

def execute_and_calculate_function_on_sqls(predicted_sql, ground_truth, db_path, sql_dialect, calculate_func):
    conn = connect_db(sql_dialect, db_path)
    # Connect to the database
    cursor = conn.cursor()
    cursor.execute(predicted_sql)
    predicted_res = cursor.fetchall()
    cursor.execute(ground_truth)
    ground_truth_res = cursor.fetchall()
    conn.close()
    res = calculate_func(predicted_res, ground_truth_res)
    return res

def execute_sql(db_path: str, sql: str, fetch: Union[str, int] = "all", timeout: int = 60) -> Any:
    class QueryThread(threading.Thread):
        def __init__(self):
            threading.Thread.__init__(self)
            self.result = None
            self.exception = None

        def run(self):
            try:
                with sqlite3.connect(db_path, timeout=timeout) as conn:
                    cursor = conn.cursor()
                    cursor.execute(sql)
                    if fetch == "all":
                        self.result = cursor.fetchall()
                    elif fetch == "one":
                        self.result = cursor.fetchone()
                    elif fetch == "random":
                        samples = cursor.fetchmany(10)
                        self.result = random.choice(samples) if samples else []
                    elif isinstance(fetch, int):
                        self.result = cursor.fetchmany(fetch)
                    else:
                        raise ValueError("Invalid fetch argument. Must be 'all', 'one', 'random', or an integer.")
            except Exception as e:
                print(f"Error while executing SQL: \n{sql}\n Database Path: \n{db_path}")
                self.exception = e
    query_thread = QueryThread()
    query_thread.start()
    query_thread.join(timeout)
    if query_thread.is_alive():
        print(f"SQL query execution exceeded the timeout of {timeout} seconds. SQL: {sql}")
        raise TimeoutError(f"SQL query execution exceeded the timeout of {timeout} seconds.")
    if query_thread.exception:
        # logging.error(f"Error in execute_sql: {query_thread.exception}\nSQL: {sql}")
        raise query_thread.exception
    return query_thread.result


def _clean_sql(sql: str) -> str:
    """
    Cleans the SQL query by removing unwanted characters and whitespace.
    
    Args:
        sql (str): The SQL query string.
        
    Returns:
        str: The cleaned SQL query string.
    """
    return sql.replace('\n', ' ').replace('"', "'").strip("`.")

def task(queue, db_path, sql, fetch):
    try:
        result = execute_sql(db_path, sql, fetch)
        queue.put(result)
    except Exception as e:
        queue.put(e)

def subprocess_sql_executor(db_path: str, sql: str, timeout: int = 240):
    queue = Queue()
    process = Process(target=task, args=(queue, db_path, sql, "all"))
    process.start()
    process.join(timeout)
    if process.is_alive():
        process.terminate()
        process.join()
        print("Time out in subprocess_sql_executor")
        raise TimeoutError("Execution timed out.")
    else:
        try:
            result = queue.get_nowait()
        except Empty:
            raise Exception("No data returned from the process.")
        
        if isinstance(result, Exception):
            raise result
        return result

def _compare_sqls_outcomes(db_path: str, predicted_sql: str, ground_truth_sql: str) -> int:
    """
    Compares the outcomes of two SQL queries to check for equivalence.
    
    Args:
        db_path (str): The path to the database file.
        predicted_sql (str): The predicted SQL query.
        ground_truth_sql (str): The ground truth SQL query.
        
    Returns:
        int: 1 if the outcomes are equivalent, 0 otherwise.
    
    Raises:
        Exception: If an error occurs during SQL execution.
    """
    try:
        predicted_res = execute_sql(db_path, predicted_sql)
        ground_truth_res = execute_sql(db_path, ground_truth_sql)
        return int(set(predicted_res) == set(ground_truth_res))
    except Exception as e:
        logging.critical(f"Error comparing SQL outcomes: {e}")
        raise e

def compare_sqls(db_path: str, predicted_sql: str, ground_truth_sql: str, meta_time_out: int = 240) -> Dict[str, Union[int, str]]:
    """
    Compares predicted SQL with ground truth SQL within a timeout.
    
    Args:
        db_path (str): The path to the database file.
        predicted_sql (str): The predicted SQL query.
        ground_truth_sql (str): The ground truth SQL query.
        meta_time_out (int): The timeout for the comparison.
        
    Returns:
        dict: A dictionary with the comparison result and any error message.
    """
    predicted_sql = _clean_sql(predicted_sql)
    try:
        res = func_timeout(meta_time_out, _compare_sqls_outcomes, args=(db_path, predicted_sql, ground_truth_sql))
        error = "incorrect answer" if res == 0 else "--"
    except FunctionTimedOut:
        logging.warning("Comparison timed out.")
        error = "timeout"
        res = 0
    except Exception as e:
        logging.error(f"Error in compare_sqls: {e}")
        error = str(e)
        res = 0
    return {'exec_res': res, 'exec_err': error}

def validate_sql_query(db_path: str, sql: str, max_returned_rows: int = 30) -> Dict[str, Union[str, Any]]:
    """
    Validates an SQL query by executing it and returning the result.
    
    Args:
        db_path (str): The path to the database file.
        sql (str): The SQL query to validate.
        max_returned_rows (int): The maximum number of rows to return.
        
    Returns:
        dict: A dictionary with the SQL query, result, and status.
    """
    try:
        result = execute_sql(db_path, sql, fetch=max_returned_rows)
        return {"SQL": sql, "RESULT": result, "STATUS": "OK"}
    except Exception as e:
        logging.error(f"Error in validate_sql_query: {e}")
        return {"SQL": sql, "RESULT": str(e), "STATUS": "ERROR"}

def aggregate_sqls(db_path: str, sqls: List[str]) -> str:
    """
    Aggregates multiple SQL queries by validating them and clustering based on result sets.
    
    Args:
        db_path (str): The path to the database file.
        sqls (List[str]): A list of SQL queries to aggregate.
        
    Returns:
        str: The shortest SQL query from the largest cluster of equivalent queries.
    """
    results = [validate_sql_query(db_path, sql) for sql in sqls]
    clusters = {}

    # Group queries by unique result sets
    for result in results:
        if result['STATUS'] == 'OK':
            # Using a frozenset as the key to handle unhashable types like lists
            key = frozenset(tuple(row) for row in result['RESULT'])
            if key in clusters:
                clusters[key].append(result['SQL'])
            else:
                clusters[key] = [result['SQL']]
    
    if clusters:
        # Find the largest cluster
        largest_cluster = max(clusters.values(), key=len, default=[])
        # Select the shortest SQL query from the largest cluster
        if largest_cluster:
            return min(largest_cluster, key=len)
    
    logging.warning("No valid SQL clusters found. Returning the first SQL query.")
    return sqls[0]

def get_execution_status(db_path: str, sql: str, fetch: Union[str, int] = "all", execution_result: List = None, timeout: int = 120) -> Tuple[str]:
    """
    Determines the status of an SQL query execution result.
    
    Args:
        execution_result (List): The result of executing an SQL query.
        
    Returns:
        Tuple[str]: The status of the execution result and error reason
    """
    if not execution_result:
        try:
            execution_result = execute_sql(db_path, sql, fetch=fetch, timeout=timeout)
        except FunctionTimedOut:
            print("Timeout in get_execution_status")
            return ("SYNTACTICALLY_INCORRECT", "timeout")
        except Exception as e:
            return ("SYNTACTICALLY_INCORRECT", str(e))   
    if (execution_result is None) or (execution_result == []):
        return ("EMPTY_RESULT", "empty_result")
    
    return ("SYNTACTICALLY_CORRECT", None)

def run_with_timeout(func, *args, timeouts=[30, 50]):
    def wrapper(stop_event, *args):
        try:
            if not stop_event.is_set():
                result[0] = func(*args)
        except Exception as e:
            result[1] = e

    for attempt, timeout in enumerate(timeouts):
        result = [None, None]
        stop_event = threading.Event()
        thread = threading.Thread(target=wrapper, args=(stop_event, *args))
        thread.start()

        # Wait for the thread to complete or timeout
        thread.join(timeout)

        if thread.is_alive():
            logging.error(f"Function {func.__name__} timed out after {timeout} seconds on attempt {attempt + 1}/{len(timeouts)}")
            stop_event.set()  # Signal the thread to stop
            thread.join()  # Wait for the thread to recognize the stop event
            if attempt == len(timeouts) - 1:
                raise TimeoutException(
                    f"Function {func.__name__} timed out after {timeout} seconds on attempt {attempt + 1}/{len(timeouts)}"
                )
        else:
            if result[1] is not None:
                raise result[1]
            return result[0]

    raise TimeoutException(f"Function {func.__name__} failed to complete after {len(timeouts)} attempts")


def calculate_row_match(predicted_row, ground_truth_row):
    """
    Calculate the matching percentage for a single row.

    Args:
    predicted_row (tuple): The predicted row values.
    ground_truth_row (tuple): The actual row values from ground truth.

    Returns:
    float: The match percentage (0 to 1 scale).
    """
    total_columns = len(ground_truth_row)
    matches = 0
    element_in_pred_only = 0
    element_in_truth_only = 0
    for pred_val in predicted_row:
        if pred_val in ground_truth_row:
            matches += 1
        else:
            element_in_pred_only += 1
    for truth_val in ground_truth_row:
        if truth_val not in predicted_row:
            element_in_truth_only += 1
    match_percentage = matches / total_columns
    pred_only_percentage = element_in_pred_only / total_columns
    truth_only_percentage = element_in_truth_only / total_columns
    return match_percentage, pred_only_percentage, truth_only_percentage

def calculate_f1_score_from_execution_results(predicted, ground_truth):
    """
    Calculate the F1 score based on sets of predicted results and ground truth results,
    where each element (tuple) represents a row from the database with multiple columns.

    Args:
    predicted (set of tuples): Predicted results from SQL query.
    ground_truth (set of tuples): Actual results expected (ground truth).

    Returns:
    float: The calculated F1 score.
    """
    # if both predicted and ground_truth are empty, return 1.0 for f1_score
    if not predicted and not ground_truth:
        return 1.0

    try:
        # Drop duplicates
        predicted_set = set(predicted) if predicted else set()
        ground_truth_set = set(ground_truth)

        # convert back to list
        predicted = list(predicted_set)
        ground_truth = list(ground_truth_set)

        # Calculate matching scores for each possible pair
        match_scores = []
        pred_only_scores = []
        truth_only_scores = []
        for i, gt_row in enumerate(ground_truth):
            # rows only in the ground truth results
            if i >= len(predicted):
                match_scores.append(0)
                truth_only_scores.append(1)
                continue
            pred_row = predicted[i]
            match_score, pred_only_score, truth_only_score = calculate_row_match(
                pred_row, gt_row
            )
            match_scores.append(match_score)
            pred_only_scores.append(pred_only_score)
            truth_only_scores.append(truth_only_score)

        # rows only in the predicted results
        for i in range(len(predicted) - len(ground_truth)):
            match_scores.append(0)
            pred_only_scores.append(1)
            truth_only_scores.append(0)

        tp = sum(match_scores)
        fp = sum(pred_only_scores)
        fn = sum(truth_only_scores)

        precision = tp / (tp + fp) if tp + fp > 0 else 0
        recall = tp / (tp + fn) if tp + fn > 0 else 0

        f1_score = (
            2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
        )
        return f1_score
    except Exception as e:
        logging.error(f"Error in computing SOFT F1 score. {e}")
        return 0.0


def calculate_f1_score_for_sql(predicted_sql: str, ground_truth: str, db_path: Union[str, Path], sql_dialect: str = 'SQLite'):
    """"
    For a given sql query and its ground truth, function calculates the f1_score

    Args:
        predicted_sql (str): predicted SQL query for a user question in string format
        ground_truth (str): ground truth SQL query for a user question in string format
        db_path (Union[str, Path]): sql file path for the database
        sql_dialect (str): Dialect used for the sql
        calculate_func (str): Function that is going to be run on execution results of the SQLs


    """

    return execute_and_calculate_function_on_sqls(predicted_sql, ground_truth, db_path, sql_dialect, calculate_f1_score_from_execution_results)