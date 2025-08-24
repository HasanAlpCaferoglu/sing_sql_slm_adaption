import re
import logging
from pydantic import BaseModel

from utils.db_utils.execution import execute_sql, get_execution_status, compare_sqls, calculate_f1_score_for_sql
from utils.llm_utils.model import extract_xml_answer, extract_xml_reasoning, extract_response_part
from utils.db_utils.sql_parser import get_sql_tables, get_sql_columns_dict
from utils.llm_utils.prompt_utils import load_template
from utils.llm_utils.model import call_llm

# Get Train Logger
train_logger = logging.getLogger("train_logger")

def calculate_jaccard(list1: list[str], list2: list[str]):
    """
    The function calculates Jaccard similarity between two lists.

    Args:
        list1 (list[str]): List of string
        list2 (list[str]): List of string

    Returns:
        float: Jaccard similarity between list1 and list2
    """
    set1 = set(list1)
    set2 = set(list2)
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    return len(intersection) / len(union) if union else 0.0

def get_rlaif_score(prompt: str, model_name: str = "gemini-2.0-flash") -> float:
    """
    The function calls LLM to give a feedback on the predicted answer

    Args:
        prompt (str): prompt given to the LLM
        model_name (str): LLM model that is going to be called

    Returns:
        float: Reward given by an LLM
    """
    
    class AIScore(BaseModel):
        score: float

    try:
        response_object, prompt_token_cnt, completion_token_cnt, total_token_cnt = call_llm(prompt, model_name, response_json_schema=AIScore)
    except Exception as e:
        print(f"Error is taken during RLAIF. Error: {e}")
        return 0.0

    score_object = response_object.parsed
    score_dict = score_object.dict()
    score = score_dict.get("score", 0.0)
    return score

def generate_ngrams(text: str, n: int = 2) -> list[str]:
    """
    Generates n-grams from normalized tokens. Handles multiple spaces and tabs.

    Args:
        text (str): Input string.
        n (int): N-gram size.

    Returns:
        List[str]: List of n-gram strings.
    """
    # Normalize whitespace: remove leading/trailing and reduce all spaces/tabs/newlines to a single space
    normalized_text = re.sub(r'\s+', ' ', text.lower().strip())
    tokens = normalized_text.split()
    return [' '.join(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]


def syntax_check_reward(completions: list[str], db_path: list[str], weight: int = 1.0, **kwargs) -> list[float]:
    """
    This function gives a reward according to executability of the predicted SQL queries. 

    Args:
        completions (List[str]): Model responses
        db_path (List[str]): List of database paths

    Returns:
        List[float]: rewards for each predicted sql
    """
    # responses = [completion[0]['content'] for completion in completions] # conversational format
    responses = [extract_response_part(completion) for completion in completions] # standard format

    p_sqls = [extract_xml_answer(r) for r in responses]
    return [weight * 1.0 if get_execution_status(dp, p_sql) == "SYNTACTICALLY_CORRECT" else 0.0 for p_sql, dp in zip(p_sqls, db_path)]

def exec_acc_reward(completions: list[str], answer: list[str], db_path: list[str], task: list[str], weight: int = 3.0, **kwargs) -> list[float]:
    """
    This function gives a reward according to accuracy of the predicted SQL query

    Args:
        completions (List[str]): Model responses
        answer (List[str]): List of ground tructh SQL queries
        db_path (List[str]): List of database paths

    Returns:
        List[float]: rewards for each predicted sql
    """
    
    # responses = [completion[0]['content'] for completion in completions] # conversational format
    responses = [extract_response_part(completion) for completion in completions] # standard format

    p_sqls = [extract_xml_answer(r) for r in responses]
    
    rewards = []
    for p_sql, gt_sql, dp, t in zip(p_sqls, answer, db_path, task):
        if t == 't2s' or t == 't2sws':
            try:
                comparison = compare_sqls(dp, p_sql, gt_sql)
                reward = weight * float(comparison['exec_res'])
                rewards.append(reward)
            except:
                rewards.append(0.0)
        else:
            # Return None
            rewards.append(None)

    return rewards


def f1_acc_reward(completions: list[str], answer: list[str], db_path: list[str], task: list[str], weight: int = 1.0, **kwargs) -> list[float]:
    """
    This function gives a reward according to executability of the predicted SQL queries. 

    Args:
        completions (List[str]): Model responses
        answer (List[str]): List of ground tructh SQL queries
        db_path (List[str]): List of database paths

    Returns:
        List[float]: rewards for each predicted sql
    """
    # responses = [completion[0]['content'] for completion in completions] # conversational format
    responses = [extract_response_part(completion) for completion in completions] # standard format

    p_sqls = [extract_xml_answer(r) for r in responses]

    rewards = []
    for p_sql, gt_sql, dp, t in zip(p_sqls, answer, db_path, task):
        if t == 't2s' or t == 't2sws':
            try:
                reward = weight * calculate_f1_score_for_sql(p_sql, gt_sql, dp)
                rewards.append(reward)
            except:
                rewards.append(0.0)
        else:
            # Return None
            rewards.append(None)

    return rewards


def rlaif_reward(completions: list[str], answer: list[str], questions: list[str], task: list[str], weight: int = 2.0, **kwargs): # Reinforcement Learning from AI Feedback i.e. LLM-as-a-judge reward
    """
    This function gives a reward according to schama accuracy 

    Args:
        completions (List[str]): Model responses
        answer (List[str]): List of ground tructh SQL queries

    Returns:
        List[float]: rewards for each predicted sql
    """
    prompt_template = load_template(template_name='rlaif')
    # responses = [completion[0]['content'] for completion in completions] # conversational format
    responses = [extract_response_part(completion) for completion in completions] # standard format
    p_sqls = [extract_xml_answer(r) for r in responses]

    rlaif_rewards: list[float] = []
    for p_sql, gt_sql, question, hint, t in zip(p_sqls, answer, questions, task):
        if t == 't2s' or t == 't2sws':
            try:
                prompt = prompt_template.format(
                    QUESTION_AND_HINT = question,
                    GT_SQL = gt_sql,
                    PREDICTED_SQL = p_sql
                )
                score = get_rlaif_score(prompt)
                rlaif_rewards.append(weight*score)
            except:
                rlaif_rewards.append(0.0)
        else:
            # Return None
            rlaif_rewards.append(None)

    return rlaif_rewards


def schema_linking_reward(completions: list[str], answer: list[str], db_path: list[str], task: list[str], weight: int = 1.0, **kwargs) -> list[float]:
    """
    This function gives a reward according to schama accuracy 

    Args:
        completions (List[str]): Model responses
        answer (List[str]): List of ground tructh SQL queries
        db_path (List[str]): List of database paths

    Returns:
        List[float]: rewards for each predicted sql
    """
    # responses = [completion[0]['content'] for completion in completions] # conversational format
    responses = [extract_response_part(completion) for completion in completions] # standard format

    p_sqls = [extract_xml_answer(r) for r in responses]

    rewards: list[float] = []
    for p_sql, gt_sql, dp, t in zip(p_sqls, answer, db_path, task):
        if t == 't2s' or t == 't2sws':
            try:
                # Extracting schema items in the predicted sql query
                p_sql_schema_items: list[str] = []
                p_sql_columns_dict = get_sql_columns_dict(dp, p_sql)
                for t_name, c_names in p_sql_columns_dict.items():
                    p_sql_schema_items.append(t_name.strip())
                    for c_name in c_names:
                        p_sql_schema_items.append(f"{t_name.strip()}.{c_name.strip()}")

                # Extracting schema items in the ground truth sql query
                gt_sql_schema_items: list[str] = []
                gt_sql_columns_dict = get_sql_columns_dict(dp, gt_sql)
                for t_name, c_names in gt_sql_columns_dict.items():
                    gt_sql_schema_items.append(t_name.strip())
                    for c_name in c_names:
                        gt_sql_schema_items.append(f"{t_name.strip()}.{c_name.strip()}")

                reward = weight * calculate_jaccard(p_sql_schema_items, gt_sql_schema_items)
                rewards.append(reward)
            except:
                rewards.append(0.0)
        else:
            # Return None
            rewards.append(None)
    
    return rewards


def n_gram_similarity_reward(completions: list[str], answer: list[str], task: list[str], n: int = 2,  weight: int = 1.0, **kwargs) -> list[float]:
    """
    Computes Jaccard similarity between n-grams of predicted and gold SQL queries.

    Args:
        completions (List[str]): Model responses.
        answer (List[str]): Ground truth SQL queries.
        n (int): N-gram size (default: 2 for bigrams).

    Returns:
        List[float]: Jaccard similarity scores for each prediction.
    """
    # responses = [completion[0]['content'] for completion in completions] # conversational format
    responses = [extract_response_part(completion) for completion in completions] # standard format

    p_sqls = [extract_xml_answer(r) for r in responses]

    ngram_rewards = []
    for p_sql, gt_sql, t in zip(p_sqls, answer, task):
        if t == 't2s' or t == 't2sws':
            p_ngrams = set(generate_ngrams(p_sql, n))
            gt_ngrams = set(generate_ngrams(gt_sql, n))
            ngram_jaccard_similarity = calculate_jaccard(p_ngrams, gt_ngrams)
            ngram_rewards.append(weight * ngram_jaccard_similarity)
        else:
            # Return None
            ngram_rewards.append(None)

    return ngram_rewards


def format_reward(completions, weight: int = 1.0, **kwargs) -> list[float]:
    """
    This function checks if the completion has a specific format
    """
    pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
    # responses = [completion[0]['content'] for completion in completions] # conversational format
    responses = [extract_response_part(completion) for completion in completions] # standard format
    
    matches = [re.match(pattern, r) for r in responses]
    return [weight * 0.5 if match else 0.0 for match in matches]
