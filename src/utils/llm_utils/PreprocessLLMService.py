import json
import logging
from typing import List, Dict, Optional, Any, Tuple, Union
from pathlib import Path
from utils.llm_utils.LLMService import LLMService
from utils.llm_utils.prompt_utils import load_template, load_template_examples
from utils.db_utils.schema_generator import DatabaseSchemaGenerator
from utils.db_utils.schema import DatabaseSchema
from utils.output_structures.models import TextToSQLPair, SQLToTextPair, SQLToTextPairWithEval, SQLFix, TextToSQLReasoningGeneration, SchemaFilterBasedOnExamples, TableFilterBasedOnExamples, SingleTextToSQLPairEvaluation
from utils.db_utils.db_info import DatabaseGeneralInfo
from utils.db_utils.sql_parser import get_sql_columns_dict, get_filtered_schema_dict_from_similar_examples
from utils.db_utils.db_info_utils import get_db_schema


class PreprocessLLMService(LLMService):
    def generate_text_to_sql_examples(self, sub_schema_string:str, sample_count: int, column_meanings:str ="", column_values_str:str="") -> List[Dict]:
        """
        Generate Text-to-SQL pairs for a sub-schema.
        
        Args:
            sub_schema_string (str): DDL form of sub-schema
            column_meaning (str): descriptions for columns that exist in sub-schema
            column_values_str (str): string for sub-schema columns values
            sample_count (int): Text-to-SQL pair numbers that are going to be generated for each difficulty level

        Returns:
            list[Dict]: Generated Text-to-SQL examples
        
        """
        prompt_template = load_template('sample_generator_t2s')
        prompt_examples = load_template_examples('sample_generator_t2s')


        prompt = prompt_template.format(
            N=sample_count,
            EXAMPLES=prompt_examples,
            SUB_SCHEMA=sub_schema_string,
            COLUMN_MEANINGS=column_meanings,
            COLUMN_VALUES=column_values_str
        )

        # self.logger.info(f"+++++++++++ PROMPT +++++++++++ \n {prompt}") ## COMMENT OUT LATER / DELETE LATER

        response, *_ = self.call_llm(prompt, response_schema=list[TextToSQLPair])
        if response is not None:
            try:
                t2s_examples =  [r.dict() for r in response.parsed]
                return t2s_examples
            except Exception as e:
                self.logger.info(f"Error is taken while parsing generated Text-to-SQL examples. {e}")
        
        return []
    
    def generate_sql_to_text_examples(self, sub_schema_string:str, sample_count: int, column_meanings:str ="", column_values_str:str="", eval_in_generation: bool = False) -> List[Dict]:
        """
        Generate SQL-to-Text pairs for a sub-schema.
        
        Args:
            sub_schema_string (str): DDL form of sub-schema
            column_meaning (str): descriptions for columns that exist in sub-schema
            column_values_str (str): string for sub-schema columns values
            sample_count (int): SQL-to-Text pair numbers that are going to be generated for each difficulty level

        Returns:
            list[Dict]: Generated SQL-to-Text examples
        
        """
        if eval_in_generation:
            print("Generating SQL-to-Text Examples with Evaluation")
            prompt_template = load_template('sample_generator_s2t_with_eval')
            prompt_examples = load_template_examples('sample_generator_s2t_with_eval')
        else:
            prompt_template = load_template('sample_generator_s2t')
            prompt_examples = load_template_examples('sample_generator_s2t')


        prompt = prompt_template.format(
            N=sample_count,
            EXAMPLES=prompt_examples,
            SUB_SCHEMA=sub_schema_string,
            COLUMN_MEANINGS=column_meanings,
            COLUMN_VALUES=column_values_str
        )

        # self.logger.info(f"+++++++++++ PROMPT +++++++++++ \n {prompt}") ## COMMENT OUT LATER / DELETE LATER

        # Generate S2T examples
        if eval_in_generation:
            response, *_ = self.call_llm(prompt, response_schema=list[SQLToTextPairWithEval])
        else:
            response, *_ = self.call_llm(prompt, response_schema=list[SQLToTextPair])
        if response is not None:
            try:
                t2s_examples: List[Dict] =  [r.dict() for r in response.parsed]
                return t2s_examples
            except Exception as e:
                self.logger.info(f"Error is taken while parsing generated Text-to-SQL examples. {e}")

        return []
    
    def generate_column_focused_text_to_sql_examples(self, sub_schema_string:str, focused_column:str, sample_count: int, column_meanings:str ="", column_values_str:str="") -> List[Dict]:
        """
        Generate Text-to-SQL pairs for a sub-schema.
        
        Args:
            sub_schema_string (str): DDL form of sub-schema
            column_meaning (str): descriptions for columns that exist in sub-schema
            column_values_str (str): string for sub-schema columns values
            sample_count (int): Text-to-SQL pair numbers that are going to be generated for each difficulty level

        Returns:
            list[Dict]: Generated Text-to-SQL examples
        
        """
        prompt_template = load_template('sample_generator_column_focused_t2s')
        prompt_examples = load_template_examples('sample_generator_t2s')


        prompt = prompt_template.format(
            N=sample_count,
            EXAMPLES=prompt_examples,
            SUB_SCHEMA=sub_schema_string,
            COLUMN_MEANINGS=column_meanings,
            COLUMN_VALUES=column_values_str,
            FOCUSED_COLUMN = focused_column
        )

        response, *_ = self.call_llm(prompt, response_schema=list[TextToSQLPair])
        if response is not None:
            try:
                t2s_examples =  [r.dict() for r in response.parsed]
                return t2s_examples
            except Exception as e:
                self.logger.info(f"Error is taken while parsing generated column focused  Text-to-SQL examples. {e}")
        
        return []
    
    def generate_column_focused_sql_to_text_examples(self, sub_schema_string:str, focused_column:str, sample_count: int, column_meanings:str ="", column_values_str:str="",  eval_in_generation: bool = False) -> List[Dict]:
        """
        Generate Text-to-SQL pairs for a sub-schema.
        
        Args:
            sub_schema_string (str): DDL form of sub-schema
            column_meaning (str): descriptions for columns that exist in sub-schema
            column_values_str (str): string for sub-schema columns values
            sample_count (int): Text-to-SQL pair numbers that are going to be generated for each difficulty level

        Returns:
            list[Dict]: Generated Text-to-SQL examples
        
        """
        if eval_in_generation:
            prompt_template = load_template('sample_generator_column_focused_s2t_with_eval')
            prompt_examples = load_template_examples('sample_generator_s2t_with_eval')
        else:
            prompt_template = load_template('sample_generator_column_focused_s2t')
            prompt_examples = load_template_examples('sample_generator_s2t')


        prompt = prompt_template.format(
            N=sample_count,
            EXAMPLES=prompt_examples,
            SUB_SCHEMA=sub_schema_string,
            COLUMN_MEANINGS=column_meanings,
            COLUMN_VALUES=column_values_str,
            FOCUSED_COLUMN = focused_column
        )

        if eval_in_generation:
            response, *_ = self.call_llm(prompt, response_schema=list[SQLToTextPairWithEval])
        else:
            response, *_ = self.call_llm(prompt, response_schema=list[SQLToTextPair])
        if response is not None:
            try:
                t2s_examples =  [r.dict() for r in response.parsed]
                return t2s_examples
            except Exception as e:
                self.logger.info(f"Error is taken while parsing generated column focused  Text-to-SQL examples. {e}")
        
        return []

    def fix_sql(self, t2s_pair: Dict, sub_schema_string:str, column_meanings:str ="", column_values_str:str="") -> Optional[Dict]:
        """
        Fix an erroneous SQL query.

        Args:
            t2s_pair (Dict): A dictionary including current Text-to-SQL conversation
            sub_schema_string (str): DDL form of sub-schema
            column_meaning (str): descriptions for columns that exist in sub-schema
            column_values_str (str): string for sub-schema columns values

        Returns:
            Dict: updated t2s_pair
        """
        prompt_template = load_template('sql_fixer')

        prompt = prompt_template.format(
            DATABASE_SCHEMA=sub_schema_string,
            COLUMN_MEANINGS=column_meanings,
            COLUMN_VALUES=column_values_str,
            QUESTION=t2s_pair.get('question', ''),
            SQL=t2s_pair.get('SQL', ''),
            ERROR_REASON=t2s_pair.get("error_reason", "")
        )

        response, *_ = self.call_llm(prompt, response_schema=SQLFix)
        try:
            fixed_sql = response.parsed.dict().get('corrected_sql', '')
            t2s_pair['initial_SQL'] =  t2s_pair['SQL'] 
            t2s_pair['initial_error_reason'] =  t2s_pair['error_reason'] 
            t2s_pair['SQL'] =  fixed_sql
            
            return t2s_pair
        except Exception as e:
            self.logger.info(f"Error is taken while parsing fixed SQL query. {e}")
            self.logger.info(f"Returning errorenous t2s_pair object.")
            return t2s_pair
        

    def generate_reasoning(self, t2s_pair: Dict, original_full_schema_str: str, logger: logging.Logger=None) -> Optional[Dict]:
        """
        Generate Divide-and-Conquer reasoning for a SQL query.
        
        Args:
            t2s_pair (Dict): A dictionary including current Text-to-SQL conversation
            original_full_schema_str (str): Original database schema string (full database schema string)
            
        Returns:
            Dict: updated t2s_pair

        """
        prompt_template = load_template('t2s_dac_reason_generator')
        
        prompt = prompt_template.format(
            DB_SCHEMA=original_full_schema_str,
            QUESTION= t2s_pair['question'],
            SQL=t2s_pair['SQL'] 
        )

        response, *_ = self.call_llm(prompt, response_schema=TextToSQLReasoningGeneration)
        try:
            dac_reasoning =  response.parsed.dict().get('reasoning', '')
            t2s_pair['dac_reasoning'] = dac_reasoning
            return t2s_pair
        except Exception as e:
            self.logger.info(f"Error is taken while parsing generated DAC reasoning. {e}")
            t2s_pair['dac_reasoning'] = dac_reasoning
            t2s_pair['dac_reasoning'] = ""
            return t2s_pair
        
    def filter_schema_based_on_examples(self, db_path: Union[str, Path], t2s_item: Dict[str, Any], t2s_examples: List[Dict[str, Any]], use_dac_reasoning_in_examples: Optional[bool] = False) -> Dict[str, List[str]]:
        """
        Filter the schema relaying on the extracted examples

        Args:
            t2s_item (Dict[str, Any]): Single Text-to-SQL item
            t2s_examples (List[Dict[str, Any]]): A list of text-to-sql examples containing the keywords of the user question
            give_dac_in_examples (bool): Flag to provide dac reasonings of the sqls

        Returns:
            Dict[str, List[str]]: filtered dictionary by LLM
        """
        db_id = t2s_item.get("db_id") if "db_id" in t2s_item else t2s_item.get("ss_id").split("-")[0]
        question = t2s_item.get("question")
        hint = t2s_item.get("evidence", "")
        question_and_hint = question + " Hint: " + hint if hint else question
        print(f"question_and_hint: {question_and_hint}")  ## DELETE LATER OR COMMENT OUT LATER

       
        examples_str = ""
        db_schema_str = ""
        
        # Get database schema and construct database schema string
        db_schema_str += "### Database Schema:\n"
        # schema_dict = get_filtered_schema_dict_from_similar_examples(db_path=db_path, similar_examples=t2s_examples)
        db_full_schema_dict: Dict[str, List[str]] = get_db_schema(db_path=db_path)
        schema_dict = db_full_schema_dict
        schema_structure = DatabaseSchema.from_schema_dict(schema_dict)
        schema_generator = DatabaseSchemaGenerator(
            tentative_schema= schema_structure,
            db_id=db_id,
            db_path=db_path,
            add_examples=True,
            add_random_examples=False
        )

        schema_string = schema_generator.generate_schema_string(
            include_column_value_examples=True,
            include_value_description=True,
            shuffle_cols=False,
            shuffle_tables=False
        )
        db_schema_str += schema_string
        db_schema_str += "\n"

        # Construct string for examples
        examples_str += "### Text-to-SQL Examples (Containing The Keywords in The User Question) \n"
        for idx, t2s_example in enumerate(t2s_examples):
            examples_str += f"Example {idx}:"
            examples_str += f"Question: {t2s_example.get('question')} \n"
            if use_dac_reasoning_in_examples:
                examples_str += f"SQL Reasoning: {t2s_example.get('dac_reasoning')} \n"
            examples_str += f"SQL: {t2s_example.get('SQL')} \n"



        prompt_template = load_template('filter_schema_based_on_examples')
        prompt = prompt_template.format(
            DB_SCHEMA = db_schema_str,
            EXAMPLES = examples_str,
            QUESTION_AND_HINT = question_and_hint
        )
        prompt.replace("{{", "{").replace("}}", "}")

        response, *_ = self.call_llm(prompt, response_schema=SchemaFilterBasedOnExamples)
        # print(f" LLM response for Schema Filtering Based on Examples: \n {response}") # DELETE LATER OR COMMENT OUT LATER
        
        try:
            list_of_filtered_table_columns: List[Dict] = response.parsed.dict().get("filtered_schema")
            filtered_schema_dict = {}
            for filtered_table in list_of_filtered_table_columns:
                t_name = filtered_table.get("table_name")
                if t_name and t_name not in filtered_schema_dict:
                    filtered_schema_dict[t_name] = filtered_table.get("column_names")
                # if filtered_table.table_name not in filtered_schema_dict:
                #     filtered_schema_dict[filtered_table.table_name] = filtered_table.column_names
            # print(f"filtered_schema_dict: \n {json.dumps(filtered_schema_dict, indent=4)}")
            return filtered_schema_dict
        except Exception as e:
            self.logger.info(f"Error is taken while parsing filtered schema. {e}")
            self.logger.info(f"Returning full db schema.")
            return db_full_schema_dict
            


    def filter_table_based_on_examples(self, db_path: Union[str, Path], t2s_item: Dict[str, Any], t2s_examples: List[Dict[str, Any]], table_name: str, use_dac_reasoning_in_examples: Optional[bool] = False):
        """
        Filter single table in a database relyin on both extracted examples and LLM reasoning capability.
        When filtering table, the full schema of a table will be given.

        Args:
            t2s_item (Dict[str, Any]): Single Text-to-SQL item
            t2s_examples (List[Dict[str, Any]]): A list of text-to-sql examples containing the keywords of the user question
            give_dac_in_examples (bool): Flag to provide dac reasonings of the sqls

        Returns:
            Dict[str, List[str]]: filtered dictionary by LLM
        """

        db_id = t2s_item.get("db_id") if "db_id" in t2s_item else t2s_item.get("ss_id").split("-")[0]
        question = t2s_item.get("question")
        hint = t2s_item.get("evidence", "")
        question_and_hint = question + " Hint: " + hint if hint else question
        # print(f"question_and_hint: {question_and_hint}")  ## DELETE LATER OR COMMENT OUT LATER

        examples_str = ""
        table_schema_str = ""
        
        # Get database schema and construct database schema string
        table_schema_str += "### Table Schema:\n"
        db_full_schema_dict: Dict[str, List[str]] = get_db_schema(db_path=db_path)
        table_schema_dict = {
            f"{table_name}": db_full_schema_dict.get(table_name)
        }
        schema_structure = DatabaseSchema.from_schema_dict(table_schema_dict)
        schema_generator = DatabaseSchemaGenerator(
            tentative_schema= schema_structure,
            db_id=db_id,
            db_path=db_path,
            add_examples=True,
            add_random_examples=False
        )

        schema_string = schema_generator.generate_schema_string(
            include_column_value_examples=True,
            include_value_description=True,
            shuffle_cols=False,
            shuffle_tables=False
        )

        table_schema_str += schema_string
        table_schema_str += "\n\n"

        # Using column profiles
        column_profile_string = schema_generator.get_column_profiles_string( with_keys=False, with_references=False)
        table_schema_str += "\n### Column Information: \n"
        table_schema_str += column_profile_string
        table_schema_str += "\n\n"

        # Construct string for examples
        examples_str += "### Text-to-SQL Examples (Containing The Keywords in The User Question) \n"
        ex_no = 1
        for idx, t2s_example in enumerate(t2s_examples):
            t2s_example_question = t2s_example.get('question')
            t2s_example_sql = t2s_example.get('SQL')
            t2s_example_used_tables_and_columns = get_sql_columns_dict(db_path=db_path, sql=t2s_example_sql)
            
            if table_name in t2s_example_used_tables_and_columns:
                used_column_list_in_example = t2s_example_used_tables_and_columns.get(table_name)
                examples_str += f"Example {ex_no}:"
                ex_no += 1
                examples_str += f"Question: {t2s_example_question}\n"
                if use_dac_reasoning_in_examples:
                    examples_str += f"SQL Reasoning: {t2s_example.get('dac_reasoning')} \n"
                examples_str += f"SQL: {t2s_example_sql} \n"
                examples_str += f"Used columns from the table {table_name}: {str(used_column_list_in_example)} \n\n"



        prompt_template = load_template('filter_table_based_on_examples')
        prompt = prompt_template.format(
            TABLE_SCHEMA = table_schema_str,
            EXAMPLES = examples_str,
            QUESTION_AND_HINT = question_and_hint
        )
        prompt.replace("{{", "{").replace("}}", "}")
        # print(f"PROMPT (filter_table_based_on_examples): \n{prompt}") # DELETE LATER OR COMMENT OUT LATER
        response, *_ = self.call_llm(prompt, response_schema=TableFilterBasedOnExamples)
        # print(f" LLM response for Schema Filtering Based on Examples: \n {response}") # DELETE LATER OR COMMENT OUT LATER
        
        try:
            selected_columns: List[str] = response.parsed.dict().get("selected_columns")
            selected_columns = [col.replace('`', '') for col in selected_columns]
            column_selection_reasoning: List[str] = response.parsed.dict().get("reasoning")
            return selected_columns, column_selection_reasoning
        except Exception as e:
            self.logger.info(f"Error is taken while parsing filtered table schema. {e}")
            self.logger.info(f"Returning the whole columns of a table.")
            return db_full_schema_dict.get(table_name), ""
    

    def evaluate_logic_in_single_t2s(self, t2s_pair: Dict, sub_schema_string:str, column_meanings:str ="", column_values_str:str=""):
        """"
        Evaluate logic in single t2s example
        """

        prompt_template = load_template("evaluate_logic_in_single_t2s")
        prompt_examples = load_template_examples('evaluate_logic_in_single_t2s')

        question = t2s_pair.get("question")
        sql = t2s_pair.get("SQL")
        evaluate_t2s_pairs_string = f"Text-to-SQL Pair to be evaluated: \n"
        evaluate_t2s_pairs_string += f"Question: {question} \n"
        evaluate_t2s_pairs_string += f"SQL to answer the question: {sql}\n\n"

        prompt = prompt_template.format(
            EXAMPLES = prompt_examples,
            SUB_SCHEMA = sub_schema_string,
            COLUMN_MEANINGS = column_meanings,
            COLUMN_VALUES = column_values_str,
            QUESTION = question,
            SQL_QUERY = sql
        )

        response, *_ = self.call_llm(prompt, response_schema=SingleTextToSQLPairEvaluation)
        if response is not None:
            try:
                evaluation = response.parsed.dict()
                return evaluation
            except Exception as e:
                self.logger.info(f"Error is taken while parsing the output of Text-to-SQL pair Evaluation.")
                return {
                    "question_and_sql_logic_analysis": "",
                    "is_logical": False
                }

            
    # def evaluate_t2s_pairs(self, generated_t2s_pairs: List[Dict],  sub_schema_string:str, column_meanings:str ="", column_values_str:str=""):
    #     """
    #     Evaluate generated t2s examples. (LLM as a Judge)
    #     """

    #     prompt_template = load_template("evaluate_generated_t2s_examples")
    #     prompt_examples = load_template_examples('evaluate_generated_t2s_examples')

    #     evaluate_t2s_pairs_string = "**Evaluate the following Text-to-SQL pairs one by one**"
    #     for idx, t2s_item in enumerate(generated_t2s_pairs):
    #         question = t2s_item.get("question")
    #         sql = t2s_item.get("SQL")
    #         evaluate_t2s_pairs_string += f"Text-to-SQL Pair to be evaluated {idx}: \n"
    #         evaluate_t2s_pairs_string += f"Question: {question} \n"
    #         evaluate_t2s_pairs_string += f"SQL to answer the question: {sql}\n\n"

    #     prompt = prompt_template.format(
    #         DATABASE_SCHEMA=sub_schema_string,
    #         COLUMN_MEANINGS=column_meanings,
    #         COLUMN_VALUES=column_values_str,
    #         EVALUATE_T2S_PAIRS=evaluate_t2s_pairs_string
    #     )

    #     response, *_ = self.call_llm(prompt, response_schema=SQLToTextPairsEvaluator)
    #     if response is not None:
    #         try:
    #             evaluated_t2s_examples = [r.dict() for r in response.parsed]
    #             return evaluated_t2s_examples
    #         except Exception as e:
    #             self.logger.info(f"Error is taken while parsing the output of generated Text-to-SQL pairs.")
        
