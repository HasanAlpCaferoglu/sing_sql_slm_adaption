from pydantic import BaseModel
from typing import Literal, Dict, List

class TextToSQLPair(BaseModel):
    question_plan: str
    question: str
    chain_of_thought_reasoning: str
    SQL: str
    difficulty: Literal["Simple", "Moderate", "Challenging", "Window"]

class SQLToTextPair(BaseModel):
    difficulty: Literal["Simple", "Moderate", "Challenging", "Window"]
    SQL: str
    sql_analysis: str
    question: str

class SQLToTextPairWithEval(BaseModel):
    difficulty: Literal["Simple", "Moderate", "Challenging", "Window"]
    SQL: str
    sql_analysis: str
    question: str
    question_and_sql_logic_analysis: str
    is_logical: bool

class SQLFix(BaseModel):
    correction_step: str
    corrected_sql: str

class SQLAndTextFix(BaseModel):
    correction_step: str
    corrected_sql: str
class TextToSQLReasoningGeneration(BaseModel):
    reasoning: str

class FilteredTable(BaseModel):
    table_name: str
    column_names: List[str]

class SchemaFilterBasedOnExamples(BaseModel):
    reasoning: str
    filtered_schema: List[FilteredTable]

class TableFilterBasedOnExamples(BaseModel):
    reasoning: str
    selected_columns: List[str]

class SingleTextToSQLPairEvaluation(BaseModel):
    question_and_sql_logic_analysis: str
    is_logical: bool


class TextToSQLPairsEvaluator(BaseModel):
    evaluation_outputs: list[SingleTextToSQLPairEvaluation]