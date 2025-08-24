import os
import json
import re
import logging
import random
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from itertools import chain, combinations, product
from collections import Counter
from dataclasses import asdict

from utils.db_utils.execution import execute_sql
from utils.db_utils.db_info_utils import get_db_schema, get_db_all_tables, get_table_all_columns, are_tables_joinable
from utils.db_utils.schema import DatabaseSchema, TableSchema, ColumnInfo, get_primary_keys
from utils.db_utils.helper import get_all_combinations, get_combinations

class DatabaseSchemaGenerator:
    """
    Generates database schema with optional examples and descriptions.
    
    Attributes:
        db_id (str): The database identifier.
        db_path (str): The path to the database file.
        add_examples (bool): Flag to indicate whether to add examples.
        schema_structure (DatabaseSchema): The base schema structure.
        schema_with_examples (DatabaseSchema): The schema including examples.
        schema_with_descriptions (DatabaseSchema): The schema including descriptions.
    """
    CACHED_DB_SCHEMA: Dict[str, DatabaseSchema] = {}

    def __init__(self, tentative_schema: Optional[DatabaseSchema] = None, schema_with_examples: Optional[DatabaseSchema] = None,
                 schema_with_descriptions: Optional[DatabaseSchema] = None, db_id: Optional[str] = None, db_path: Optional[str] = None, 
                 tlsn: Optional[Union[str, int]] = "", clsn: Optional[Union[str, int]] = "",
                 add_examples: bool = True, add_random_examples: bool = True):
        self.db_id = db_id
        self.db_path = Path(db_path)

        self.tlsn = tlsn # tlsn == table level schema no
        self.clsn = clsn # clsn == column level schema no
        self.schema_id = db_id if db_id is not None else "unknown"
        if tlsn or tlsn==0:
            self.schema_id += f"-{tlsn}"
            if clsn or clsn==0:
                self.schema_id += f"-{clsn}"

        self.add_examples = add_examples
        self.add_random_examples = add_random_examples
        if not add_examples:
            self.add_random_examples = False
        
        if self.db_id not in DatabaseSchemaGenerator.CACHED_DB_SCHEMA:
            DatabaseSchemaGenerator._load_schema_into_cache(db_id=db_id, db_path=db_path)

        self.schema_structure = tentative_schema or DatabaseSchema()
        self.schema_with_examples = schema_with_examples or DatabaseSchema()
        self.schema_with_descriptions = schema_with_descriptions or DatabaseSchema()
        self._initialize_schema_structure()

    @staticmethod
    def _set_primary_keys(db_path: str, database_schema: DatabaseSchema) -> None:
        """
        Sets primary keys in the database schema.
        
        Args:
            db_path (str): The path to the database file.
            database_schema (DatabaseSchema): The database schema to update.
        """
        schema_with_primary_keys = {
            table_name: {
                col[1]: {"primary_key": True} for col in execute_sql(db_path, f"PRAGMA table_info(`{table_name}`)") if col[5] > 0
            }
            for table_name in database_schema.tables.keys()
        }
        database_schema.set_columns_info(schema_with_primary_keys)

    @staticmethod
    def _set_foreign_keys(db_path: str, database_schema: DatabaseSchema) -> None:
        """
        Sets foreign keys in the database schema.
        
        Args:
            db_path (str): The path to the database file.
            database_schema (DatabaseSchema): The database schema to update.
        """
        schema_with_references = {
            table_name: {
                column_name: {"foreign_keys": [], "referenced_by": []} for column_name in table_schema.columns.keys()
            }
            for table_name, table_schema in database_schema.tables.items()
        }

        for table_name, columns in schema_with_references.items():
            foreign_keys_info = execute_sql(db_path, f"PRAGMA foreign_key_list(`{table_name}`)")
            # print(f"foreign_keys_info for table {table_name}: \n {foreign_keys_info}\n") # DEBUG DELETE LATER
            for fk in foreign_keys_info:
                source_table = table_name
                source_column = database_schema.get_actual_column_name(table_name, fk[3]) #fk[3] is the column name
                destination_table = database_schema.get_actual_table_name(fk[2]) # fk[2] is the name of the table that is referenced
                destination_column = get_primary_keys(database_schema.tables[destination_table])[0] if not fk[4] else database_schema.get_actual_column_name(fk[2], fk[4]) # fk[4] is the column name that is referenced 

                schema_with_references[source_table][source_column]["foreign_keys"].append((destination_table, destination_column))
                schema_with_references[destination_table][destination_column]["referenced_by"].append((source_table, source_column))


        database_schema.set_columns_info(schema_with_references)

    @staticmethod
    def _add_foreign_keys_manual(db_id, schema_with_references: Dict[str, Dict[str, Dict[str, Any]]]) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """"
        Adds manually foreing keys that are missing in the database sqlite file.

        Args:
            schema_with_references(Dict[str, Dict[str, Dict[str, Any]]]): Dictionary for FKs.
        """

        # Format to add FK --> 
        # schema_with_references[source_table][source_column]["foreign_keys"].append((destination_table, destination_column))
        # schema_with_references[destination_table][destination_column]["referenced_by"].append((source_table, source_column))
        if db_id == "card_games":
            # cards JOIN sets ON cards.setCode = sets.code
            schema_with_references['cards']['setCode']["foreign_keys"].append(('sets', 'code'))
            schema_with_references['sets']['code']["foreign_keys"].append(('cards', 'setCode'))

        if db_id == "codebase_community":
            # badges - comments (badges JOIN comments ON badges.UserId = comments.UserId)
            schema_with_references['badges']['UserId']["foreign_keys"].append(('comments', 'UserId'))
            schema_with_references['comments']['UserId']["foreign_keys"].append(('badges', 'UserId'))
            # postHistory - votes (postHistory.PostId = votes.PostId)
            schema_with_references['postHistory']['PostId']["foreign_keys"].append(('votes', 'PostId'))
            schema_with_references['votes']['PostId']["foreign_keys"].append(('postHistory', 'PostId'))
            # postHistory - tags (postHistory.PostId = tags.ExcerptPostId)
            schema_with_references['postHistory']['PostId']["foreign_keys"].append(('tags', 'ExcerptPostId'))
            schema_with_references['tags']['ExcerptPostId']["foreign_keys"].append(('postHistory', 'PostId'))
            # postHistory - badges (postHistory.UserId = badges.UserId)
            schema_with_references['postHistory']['UserId']["foreign_keys"].append(('badges', 'UserId'))
            schema_with_references['badges']['UserId']["foreign_keys"].append(('postHistory', 'UserId'))



    @classmethod
    def _load_schema_into_cache(cls, db_id: str, db_path: str) -> None:
        """
        Loads database schema into cache.
        
        Args:
            db_id (str): The database identifier.
            db_path (str): The path to the database file.
        """
        db_schema = DatabaseSchema.from_schema_dict(get_db_schema(db_path))
        # schema_with_type = {
        #     table_name: {col[1]: {"type": col[2]} for col in execute_sql(db_path, f"PRAGMA table_info(`{table_name}`)", fetch="all")}
        #     for table_name in db_schema.tables.keys()
        # }
        schema_with_type = {}
        for table_name in db_schema.tables.keys():
            columns = execute_sql(db_path, f"PRAGMA table_info(`{table_name}`)", fetch="all")
            schema_with_type[table_name] = {}
            for col in columns:
                schema_with_type[table_name][col[1]] = {"type": col[2]} # Add Column Type
                unique_values = execute_sql(db_path, f"SELECT COUNT(*) FROM (SELECT DISTINCT `{col[1]}` FROM `{table_name}` LIMIT 21) AS subquery;", fetch="all", timeout=480)
                is_categorical = int(unique_values[0][0]) < 20
                unique_values = None  
                if is_categorical:
                    unique_values = execute_sql(db_path, f"SELECT DISTINCT `{col[1]}` FROM `{table_name}` WHERE `{col[1]}` IS NOT NULL", fetch="all", timeout=480)
                else:
                    unique_values = execute_sql(db_path, f"SELECT DISTINCT `{col[1]}` FROM `{table_name}` WHERE `{col[1]}` IS NOT NULL LIMIT 10", fetch="all", timeout=480)
                    unique_values = [val for val in unique_values if len(str(val)) < 50]

                schema_with_type[table_name][col[1]].update({"unique_values": unique_values}) # Add Unique Values
                try:
                    value_statics_query = f"""
                    SELECT 'Total count ' || COUNT(`{col[1]}`) || ' - Distinct count ' || COUNT(DISTINCT `{col[1]}`) || 
                        ' - Null count ' || SUM(CASE WHEN `{col[1]}` IS NULL THEN 1 ELSE 0 END) AS counts  
                    FROM (SELECT `{col[1]}` FROM `{table_name}` LIMIT 100000) AS limited_dataset;
                    """
                    value_statics = execute_sql(db_path, value_statics_query, "all", 480)
                    schema_with_type[table_name][col[1]].update({
                        "value_statics": str(value_statics[0][0]) if value_statics else None
                    }) # Add Value Statistics
                except Exception as e:
                    print(f"An error occurred while fetching statistics for {col[1]} in {table_name}: {e}")
                    schema_with_type[table_name][col[1]].update({"value_statics": None})
        db_schema.set_columns_info(schema_with_type)
        cls.CACHED_DB_SCHEMA[db_id] = db_schema
        cls._set_primary_keys(db_path, cls.CACHED_DB_SCHEMA[db_id])
        cls._set_foreign_keys(db_path, cls.CACHED_DB_SCHEMA[db_id])
   
    def _initialize_schema_structure(self) -> None:
        """
        Initializes the schema structure with table and column info, examples, and descriptions.
        """
        self._load_table_and_column_info()
        self._load_column_examples()
        self._load_column_descriptions()
        self._load_column_meanings()
        # print(f"Schema structure initialized for {self.db_id}")

    def _load_table_and_column_info(self) -> None:
        """
        Loads table and column information from cached schema.
        """
        self.schema_structure = DatabaseSchemaGenerator.CACHED_DB_SCHEMA[self.db_id].subselect_schema(self.schema_structure)
        self.schema_structure.add_info_from_schema(schema=self.CACHED_DB_SCHEMA[self.db_id], 
                                                   field_names=["type", "primary_key", "foreign_keys", "referenced_by", "unique_values", "value_statics"])              
                        
    def _load_column_examples(self, value_cnt: int = 5) -> None:
        """
        Loads examples for columns in the schema.
        """
        self.schema_structure.add_info_from_schema(schema=self.schema_with_examples, field_names=["examples"])
        
        for table_name, table_schema in self.schema_structure.tables.items():
            for column_name, column_info in table_schema.columns.items():
                if not self.add_random_examples:
                    # getting fixed column examples while creating schema generator instance
                    if not column_info.examples: 
                        examples = DatabaseSchemaGenerator.CACHED_DB_SCHEMA[self.db_id].get_column_info(table_name, column_name).unique_values
                        if examples:
                            column_info.examples = [str(x[0]) for x in examples][:5]
                        else:
                            examples = execute_sql(db_path=self.db_path,
                                            sql=f"SELECT DISTINCT `{column_name}` FROM `{table_name}` WHERE `{column_name}` IS NOT NULL LIMIT {value_cnt}",
                                            fetch="all", timeout=480)
                            column_info.examples = [str(x[0]) if isinstance(x, tuple) else str(x) for x in examples][:5]
                else:
                    # getting random column examples while creating schema generator instance
                    if (self.add_examples and not column_info.examples) or ((column_info.type.lower()) == "date" or ("date" in column_name.lower())):
                        ### To fetch totally N random values
                        example = execute_sql(db_path=self.db_path,
                                            sql=f"SELECT DISTINCT `{column_name}` FROM `{table_name}` ORDER BY RANDOM() LIMIT {value_cnt}",
                                            fetch="all", timeout=480)
                        example = [r[0] for r in example]
                        if example and len(str(example[0])) < 50:
                            column_info.examples = example
                    
                    if not column_info.value_statics:
                        value_statics = DatabaseSchemaGenerator.CACHED_DB_SCHEMA[self.db_id].get_column_info(table_name, column_name).value_statics
                        if value_statics:
                            column_info.value_statics = value_statics
                    

    def _load_column_descriptions(self) -> None:
        """
        Loads descriptions for columns in the schema.
        """
        self.schema_structure.add_info_from_schema(self.schema_with_descriptions, field_names=["original_column_name", "column_name", "column_description", "data_format", "value_description"])
    
    def _load_column_meanings(self) -> None:
        """
        Loads the meanings of columns in the schema.
        """
        # DATASET_MODE_ROOT_PATH = Path(os.getenv('DATASET_MODE_ROOT_PATH')) # DATASET_MODE_ROOT_PATH = DB_ROOT_PATH
        DATASET_MODE_ROOT_PATH = self.db_path.parent.parent.parent
        column_meaning_path = DATASET_MODE_ROOT_PATH / "processed_column_meaning.json"

        if not column_meaning_path.exists():
            print(f"Couldn't find {column_meaning_path} for loading column meanings.")
            return
        
        try:
            with open(column_meaning_path, 'r', encoding="utf-8") as file:
                column_meaning_dict: Dict[str, Any] = json.load(file)
                # print(f"column_meaning_dict: {column_meaning_dict}")
        except Exception as e:
            print(f"Couldn't read the column meaning file.")
            return

        for table_name, table_schema in self.schema_structure.tables.items():
            for column_name, column_info in table_schema.columns.items():
                db_column_meanings: Dict[str, Dict[str, str]] = column_meaning_dict.get(f"{self.db_id}", '')
                # print(f"db_column_meanings: {db_column_meanings}")
                table_columns_and_meanings: Dict[str, str] = db_column_meanings.get(table_name, '')
                column_info.column_meaning = table_columns_and_meanings.get(column_name, '')
        
        return

    def _extract_create_ddl_commands(self) -> Dict[str, str]:
        """
        Extracts DDL commands to create tables in the schema.
        
        Returns:
            Dict[str, str]: A dictionary mapping table names to their DDL commands.
        """
        ddl_commands = {}
        for table_name in self.schema_structure.tables.keys():
            create_prompt = execute_sql(db_path=self.db_path, 
                                        sql=f"SELECT sql FROM sqlite_master WHERE type='table' AND name='{table_name}';", 
                                        fetch="one")
            ddl_commands[table_name] = create_prompt[0] if create_prompt else ""
        return ddl_commands
    
    @staticmethod
    def _separate_column_definitions(column_definitions: str) -> List[str]:
        """
        Separates column definitions in a DDL command.
        
        Args:
            column_definitions (str): The column definitions as a single string.
            
        Returns:
            List[str]: A list of individual column definitions.
        """
        paranthesis_open = 0
        start_position = 0
        definitions = []
        for index, char in enumerate(column_definitions):
            if char == "(":
                paranthesis_open += 1
            elif char == ")":
                paranthesis_open -= 1
            if paranthesis_open == 0 and char == ",":
                definitions.append(column_definitions[start_position:index].strip())
                start_position = index + 1
        definitions.append(column_definitions[start_position:].strip())
        return definitions
    
    def is_connection(self, table_name: str, column_name: str, strictly_in_schema: bool = True) -> bool:
        """
        Checks if a column is a connection (primary key or foreign key).
        
        Args:
            table_name (str): The name of the table.
            column_name (str): The name of the column.
            
        Returns:
            bool: True if the column is a connection, False otherwise.
        """
        column_info = self.CACHED_DB_SCHEMA[self.db_id].get_column_info(table_name, column_name)
        if column_info is None:
            return False
        if column_info.primary_key:
            return True
        
        if strictly_in_schema: # This part only considers a column in a connection if the referenced table is in the schema
            for target_table, _ in column_info.foreign_keys:
                if self.schema_structure.get_table_info(target_table): # This if statement prevents a FK statement when the referenced table is not in sub-schema
                    return True
                
            for target_table, _ in column_info.referenced_by:
                if self.schema_structure.get_table_info(target_table): # This if statement prevents a FK statement when the referenced table is not in sub-schema
                    return True
        else: # Additional to strictly in schema condition, this part considers a column in a connection when it refers to a non-schema-existant table's column if an only if another table's column in schema refer to that specific column
            for target_table, _ in column_info.foreign_keys:
                if self.schema_structure.get_table_info(target_table): 
                    return True
                
            for target_table, _ in column_info.referenced_by:
                if self.schema_structure.get_table_info(target_table): 
                    return True
                
            for target_table, target_column in column_info.foreign_keys: # Check
                joinable_tables_on_third_table_column = self.schema_structure.get_table_joinables_on_third_table_column(
                    current_t_name=table_name, 
                    fk_t_name=target_table, 
                    fk_c_name=target_column)
                if joinable_tables_on_third_table_column:
                    return True
                
        # else: # This part considers a column in a connection if it refers to any table regardless of whether referenced table exists in the schema or not
        #     if column_info.foreign_keys:
        #         return True
        #     if column_info.referenced_by:
        #         return True

        for target_table_name, table_schema in self.schema_structure.tables.items():
            if table_name.lower() == target_table_name.lower():
                continue
            for target_column_name, target_column_info in table_schema.columns.items():
                if target_column_name.lower() == column_name.lower() and target_column_info.primary_key:
                    return True
        return False
    
    def _get_connections(self) -> Dict[str, List[str]]:
        """
        Retrieves connections between tables in the schema.
        
        Returns:
            Dict[str, List[str]]: A dictionary mapping table names to lists of connected columns.
        """
        connections = {}
        for table_name, table_schema in self.schema_structure.tables.items():
            connections[table_name] = []
            for column_name, column_info in self.CACHED_DB_SCHEMA[self.db_id].tables[table_name].columns.items():
                if self.is_connection(table_name, column_name):
                    connections[table_name].append(column_name)
        return connections
    
    def get_schema_with_connections(self) -> Dict[str, List[str]]:
        """
        Gets schema with connections included.
        
        Returns:
            Dict[str, List[str]]: The schema with connections included.
        """
        schema_structure_dict = self.schema_structure.to_dict()
        connections = self._get_connections()
        for table_name, connected_columns in connections.items():
            for column_name in connected_columns:
                if column_name.lower() not in [col.lower() for col in schema_structure_dict[table_name]]:
                    schema_structure_dict[table_name].append(column_name)
        return schema_structure_dict
    
    def _get_example_column_name_description(self, table_name: str, column_name: str, include_column_value_examples: bool = True, include_value_description: bool = True) -> str:
        """
        Retrieves example values and descriptions for a column.
        
        Args:
            table_name (str): The name of the table.
            column_name (str): The name of the column.
            include_value_description (bool): Flag to include value description.
            include_column_value_examples (bool): Flag to include column value examples.
            
        Returns:
            str: The example values and descriptions for the column.
        """
        example_part = ""
        name_string = ""
        description_string = ""
        value_statics_string = ""
        value_description_string = ""
        
        column_info = self.schema_structure.get_column_info(table_name, column_name)
        if column_info:
            if include_column_value_examples and column_info.examples:
                example_part = f" Example Values: {', '.join([f'`{str(x)}`' for x in column_info.examples])}"
            if column_info.value_statics:
                value_statics_string = f" Value Statics: {column_info.value_statics}"
            if column_info.column_name:
                if (column_info.column_name.lower() != column_name.lower()) and (column_info.column_name.strip() != ""):
                    name_string = f"| Column Name Meaning: {column_info.column_name}"
            if column_info.column_description:
                description_string = f"| Column Description: {column_info.column_description}"
            if column_info.value_description and include_value_description:
                value_description_string = f"| Value Description: {column_info.value_description}"
        
        description_part = f"{name_string} {description_string} {value_description_string}"
        joint_string = f" --{example_part} |{value_statics_string} {description_part}" if example_part and description_part else f" --{example_part or description_part or value_statics_string}"
        if joint_string == " --" or joint_string == " --  ":
            joint_string = ""
        return joint_string.replace("\n", " ") if joint_string else ""

    def generate_schema_string(self, include_column_value_examples: bool = True, include_value_description: bool = True, shuffle_cols: bool = True, shuffle_tables: bool = True) -> str:
        """
        Generates a schema string with descriptions and examples.
        
        Args:
            include_column_value_examples (bool): Flag to include column value examples
            include_value_description (bool): Flag to include value descriptions.
            shuffle_cols (bool): Flag to suffle columns 
            shuffle_tables (bool) Flag to shuffle tables
        
        Returns:
            str: The generated schema string.
        """
        # Shuffle DDL commands
        ddl_commands = self._extract_create_ddl_commands()
        if shuffle_tables:
            ddl_tables = list(ddl_commands.keys())
            random.shuffle(ddl_tables)
            ddl_commands = {table_name: ddl_commands[table_name] for table_name in ddl_tables}
            # ddl_commands = dict(random.sample(ddl_commands.items(), len(ddl_commands)))
            
        for table_name, ddl_command in ddl_commands.items():
            ddl_command = re.sub(r'\s+', ' ', ddl_command.strip())
            create_table_match = re.match(r'CREATE TABLE "?`?([\w -]+)`?"?\s*\((.*)\)', ddl_command, re.DOTALL)
            table = create_table_match.group(1).strip()
            if table != table_name:
                logging.warning(f"Table name mismatch: {table} != {table_name}")
            column_definitions = create_table_match.group(2).strip()
            targeted_columns = self.schema_structure.tables[table_name].columns
            schema_lines = [f"CREATE TABLE {table_name}", "("]
            definitions = DatabaseSchemaGenerator._separate_column_definitions(column_definitions)
            if shuffle_cols:
                definitions = random.sample(definitions, len(definitions))
            for column_def in definitions:
                column_def = column_def.strip()
                if any(keyword in column_def.lower() for keyword in ["foreign key", "primary key"]):
                    if "primary key" in column_def.lower():
                        new_column_def = f"\t{column_def},"
                        schema_lines.append(new_column_def)
                    if "foreign key" in column_def.lower(): 
                        new_column_def = f"\t{column_def},"
                        schema_lines.append(new_column_def)
                        # This for loop in if statement prevents a FK statement when the referenced table is not in sub-schema
                        # for t_name in self.schema_structure.tables.keys():
                        #     if t_name.lower() in column_def.lower():
                        #         new_column_def = f"\t{column_def},"
                        #         schema_lines.append(new_column_def)
                else:
                    if column_def.startswith('--'):
                        continue
                    if column_def.startswith('`'):
                        column_name = column_def.split('`')[1]
                    elif column_def.startswith('"'):
                        column_name = column_def.split('"')[1]
                    else:
                        column_name = column_def.split(' ')[0]
                        
                    if (column_name in targeted_columns) or self.is_connection(table_name, column_name):
                        new_column_def = f"\t{column_def},"
                        # new_column_def += self._get_example_column_name_description(table_name, column_name, include_column_value_examples, include_value_description)
                        desc_str = self._get_example_column_name_description(table_name, column_name, include_column_value_examples, include_value_description)
                        if desc_str and desc_str != " --  ": # if there is a description, add it
                            # print(f"There is desc_str and it is `{desc_str}`") # COMMENT OUT LATER
                            new_column_def += desc_str

                        schema_lines.append(new_column_def)
                    elif column_def.lower().startswith("unique"):
                        new_column_def = f"\t{column_def},"
                        schema_lines.append(new_column_def)
            schema_lines.append(");")
            ddl_commands[table_name] = '\n'.join(schema_lines)
        return "\n\n".join(ddl_commands.values())

    def get_column_profiles(self, with_keys: bool = False, with_references: bool = False) -> Dict[str, Dict[str, str]]:
        """
        Retrieves profiles for columns in the schema. 
        The output is a dictionary with table names as keys mapping to dictionaries with column names as keys and column profiles as values.
        
        Args:
            with_keys (bool): Flag to include primary keys and foreign keys.
            with_references (bool): Flag to include referenced columns.
            
        Returns:
            Dict[str, Dict[str, str]]: The column profiles.
        """
        column_profiles = {}
        for table_name, table_schema in self.schema_structure.tables.items():
            column_profiles[table_name] = {}
            for column_name, column_info in table_schema.columns.items():
                if with_keys or not (column_info.primary_key or column_info.foreign_keys or column_info.referenced_by):
                    column_profile = f"Table name: `{table_name}`\nOriginal column name: `{column_name}`\n"
                    if (column_info.column_name.lower().strip() != column_name.lower().strip()) and (column_info.column_name.strip() != ""):
                        column_profile += f"Expanded column name: `{column_info.column_name}`\n"
                    if column_info.type:
                        column_profile += f"Data type: {column_info.type}\n"
                    if column_info.column_description:
                        column_profile += f"Description: {column_info.column_description}\n"
                    if column_info.column_meaning:
                        column_profile += f"Column meaning: {column_info.column_meaning}"
                    if column_info.value_description:
                        column_profile += f"Value description: {column_info.value_description}\n"
                    if column_info.examples:
                        column_profile += f"Example of values in the column: {', '.join([f'`{str(x)}`' for x in column_info.examples])}\n"
                    if column_info.primary_key:
                        column_profile += "This column is a primary key.\n"
                    if with_references:
                        if column_info.foreign_keys:
                            column_profile += "This column references the following columns:\n"
                            for target_table, target_column in column_info.foreign_keys:
                                column_profile += f"    Table: `{target_table}`, Column: `{target_column}`\n"
                        if column_info.referenced_by:
                            column_profile += "This column is referenced by the following columns:\n"
                            for source_table, source_column in column_info.referenced_by:
                                column_profile += f"    Table: `{source_table}`, Column: `{source_column}`\n"
                    column_profiles[table_name][column_name] = column_profile
        return column_profiles
    
    def get_column_profiles_string(self, with_keys: bool = False, with_references: bool = False) -> str:
        """ 
        Construct a single string by concatenating all columns in the schema 

        Args:
            with_keys (bool): Flag to include primary keys and foreign keys.
            with_references (bool): Flag to include referenced columns.
            
        Returns:
            str: The string containing column profiles.
        """

        column_profiles_string = ""
        column_profiles_dict = self.get_column_profiles(with_keys, with_references)
        for table_name, table_dict in column_profiles_dict.items():
            for column_name, column_profile in table_dict.items():
                column_profiles_string += column_profile 
                column_profiles_string += "\n"

        return column_profiles_string
    
    def get_schema_column_meanings_string(self):
        """
        Concatenate all column meanings and return it as a string
        """

        column_meaning_string = "### DETAIL DESCRIPTIONS AND MEANINGS OF COLUMNS: \n"
        for table_name, table_schema in self.schema_structure.tables.items():
            for column_name, column_info in table_schema.columns.items():
                column_meaning_string += column_info.column_meaning + "\n"

        return column_meaning_string

    def get_first_n_unique_column_values_string(self, value_cnt:int=10) -> str:
        """
        Constructing a string for each column values and concatenating them.

        Args:
            value_cnt (20): the number of unique values to get for a single column
        
        Returns:
            str: A string containing values for each column in the schema
        """

        column_values_string = "### DATABASE COLUMN EXAMPLE VALUES"
        for table_name, table_schema in self.schema_structure.tables.items():
            for column_name, column_info in table_schema.columns.items():
                unique_values = execute_sql(self.db_path, f"SELECT DISTINCT `{column_name}` FROM `{table_name}` WHERE `{column_name}` IS NOT NULL LIMIT {str(value_cnt)}", fetch="all", timeout=480)
                unique_values = [val for val in unique_values if len(str(val)) < 50]
                column_values_string += f"# {len(unique_values)} unique values for `{column_name}` column in `{table_name}` table: {unique_values}\n"
        
        return column_values_string

    def get_random_unique_column_values_string(self, value_cnt:int=10) -> str:
        """
        Constructing a string for each column values and concatenating them.

        Args:
            value_cnt (20): the number of unique values to get for a single column
        
        Returns:
            str: A string containing values for each column in the schema
        """

        column_values_string = "### DATABASE COLUMN EXAMPLE VALUES"
        for table_name, table_schema in self.schema_structure.tables.items():
            for column_name, column_info in table_schema.columns.items():
                ## To fetch first-N * 5 values and then select N of them randomly
                # unique_values = execute_sql(self.db_path, f"SELECT DISTINCT `{column_name}` FROM `{table_name}` WHERE `{column_name}` IS NOT NULL LIMIT {str(value_cnt*5)}")
                # random_unique_values = random.sample(unique_values, value_cnt)

                ## To fetch totally N random values
                random_unique_values = execute_sql(db_path=self.db_path,
                                sql=f"SELECT DISTINCT `{column_name}` FROM `{table_name}` ORDER BY RANDOM() LIMIT {value_cnt}", timeout=480)
                random_unique_values = [r[0] for r in random_unique_values]
                random_unique_values = [val for val in random_unique_values if len(str(val)) < 50]
                column_values_string += f"#{len(random_unique_values)} unique values for {column_name} column in {table_name} table: {random_unique_values}\n"
        
        return column_values_string
    
    def _manual_db_joinables(self, db_joinables:Dict[str, List[str]]) -> Dict[str, List[str]]:
        """
        This function enlarge db_joinables which cannot capture joinable tables due to missing referential integrity.

        Args:
            db_joinables (Dict[str, List[str]]): A dictionary for storing list of tables a table in db can join
        
        Returns:
            Dict[str, List[str]]
        """
        db_logger = logging.getLogger(self.db_id)
        db_logger.info("*** Adding manuel joinables...")

        if self.db_id == "card_games":
            # cards JOIN sets ON cards.setCode = sets.code
            if 'sets' not in db_joinables['cards']:
                db_joinables['cards'].append('sets') 
            if 'cards' not in db_joinables['sets']:
                db_joinables['sets'].append("cards")
            # cards JOIN sets ON cards.setCode = set_translations.setCode
            if 'set_translations' not in db_joinables['cards']:
                db_joinables['cards'].append('set_translations') 
            if 'cards' not in db_joinables('set_translations'):
                db_joinables['set_translations'].append('cards')

        if self.db_id == "codebase_community":
            # badges - comments ( badges.UserId = comments.UserId)
            if 'comments' not in db_joinables['badges']:
                db_joinables['badges'].append('comments') 
            if 'badges' not in db_joinables['comments']:
                db_joinables['comments'].append('badges') 
            # postHistory - votes (postHistory.PostId = votes.PostId)
            if 'votes' not in db_joinables['postHistory']:
                db_joinables['postHistory'].append('votes') 
            if 'postHistory' not in db_joinables['votes']:
                db_joinables['votes'].append('postHistory')
            # postHistory - tags 
            if 'tags' not in db_joinables['postHistory']:
                db_joinables['postHistory'].append('tags') 
            if 'postHistory' not in db_joinables['tags']:
                db_joinables['tags'].append('postHistory') 
            # postHistory - badges
            if 'badges' not in db_joinables['postHistory']:
                db_joinables['postHistory'].append('badges') 
            if 'postHistory' not in db_joinables['badges']:
                db_joinables['badges'].append('postHistory') 
            

        return db_joinables
    
    def generate_table_level_sub_schemas(self, db_joinables:Optional[Dict[str, List[str]]] = None) -> List[DatabaseSchema]:
        """
        This function generates table level sub-schemas using existing schema

        Args:
            db_joinables (Dict[str, List[str]]): A dictionary for storing list of tables a table in db can join

        Returns
            List[DatabaseSchema]: List of sub-schema
        """
        db_logger = logging.getLogger(self.db_id)
        db_logger.info("===Generating Table level Sub-Schemas===")
        def update_foreign_keys(sub_schema: DatabaseSchema):
            """
            This function updates the foreign keys and referenced by attributes of schema by removing excessive tables 
            """
            ss_table_names = sub_schema.tables.keys()
            for table_name, table_schema in sub_schema.tables.items():
                for column_name, column_info in table_schema.columns.items():
                    foreign_keys = column_info.foreign_keys
                    updated_foreign_keys = [fk for fk in foreign_keys if fk[0] in ss_table_names]
                    column_info.foreign_keys = updated_foreign_keys
                    
                    referenced_by = column_info.referenced_by
                    updated_referenced_by = [rb for rb in referenced_by if rb[0] in ss_table_names]
                    column_info.referenced_by = updated_referenced_by
            return

        original_schema = self.schema_structure
        if not db_joinables:
            db_joinables = self.schema_structure.get_db_joinables_from_schema()
        
        # db_joinables = self._manual_db_joinables(db_joinables) # Adding manuel joinables due to missing referential integrity
        db_logger.info(f"========== ++++++++++ ========== ++++++++++") ## DELETE LATER / COMMENT OUT LATER
        db_logger.info(f"db_joinables \n {json.dumps(db_joinables, indent=4)}")  ## DELETE LATER / COMMENT OUT LATER
        db_logger.info(f"========== ++++++++++ ========== ++++++++++") ## DELETE LATER / COMMENT OUT LATER
        # raise ValueError("db_joinables observation purpose")  ## DELETE LATER / COMMENT OUT LATER
        
        db_all_tables = list(self.schema_structure.tables.keys())
        all_table_combinations = get_all_combinations(db_all_tables)
        
        tables_of_sub_schemas = []
        for tables_of_candidate_sub_schema in all_table_combinations:
            joinable = are_tables_joinable(table_names=list(tables_of_candidate_sub_schema), db_joinables=db_joinables)
            if joinable:
                tables_of_sub_schemas.append(tables_of_candidate_sub_schema)

        sub_schemas = []
        is_sub_schemas_contains_original_schema = False
        for ss_table_names in tables_of_sub_schemas:
            if Counter(ss_table_names) == Counter(db_all_tables):
                sub_schemas.append(original_schema) # adding original schema
                is_sub_schemas_contains_original_schema = True
                continue
            sub_schema_tables = {t_name: original_schema.tables.get(t_name) for t_name in ss_table_names}
            new_sub_schema = DatabaseSchema(sub_schema_tables)

            # update_foreign_keys(new_sub_schema) 
            # PROBLEM: This function (update_foreign_keys) causes not adding necessary columns for
            # sub-schemas with two tables and joinable on a column in another third table.
            sub_schemas.append(new_sub_schema)
    
        # Inserting original schema (whole schema) as sub_schema 
        if not is_sub_schemas_contains_original_schema:
            sub_schemas.insert(0, original_schema)

        return sub_schemas
    
    def generate_column_level_sub_schemas_via_sliding_window(self, window: int, stride: int, column_value_cnt: int = 5) -> List[DatabaseSchema]:
        """
        This function generates column level sub-schemas using existing schema with sliding window column selection approach

        Arguments:
            window (int): the window lenght, i.e. number of columns will be considered. 
            stride (int): the number of column that window is advanced
        """
        schema_structure = self.schema_structure
        schema_detailed_dict = self.convert_schema_to_dict()
        tc_names_dict = schema_detailed_dict.get('tables_and_columns') # tc_names_dict stands for table_column_names dict

        # All possible column combination approach.
        # # IMPORTANT: Note that this yield huge number of subschema since all column combinations are considered.
        # table_column_combinations = {
        #     table: get_all_combinations(columns) for table, columns in tc_names_dict.items()
        # }

        table_column_combinations = {}
        for table_name, columns in tc_names_dict.items():
            table_column_combinations[table_name] = []

            table_connections = [] # names of columns which are either PK or FK relation
            non_connection_columns = [] # columns exluding connections
            for column_name in columns:
                if self.is_connection(table_name, column_name, strictly_in_schema=False):
                    table_connections.append(column_name)
                else:
                    non_connection_columns.append(column_name)

            if not non_connection_columns:
                table_column_combinations[table_name].append(table_connections)
            else:
                # shuffle non_connection_columns. The reason is that a column can be exist in different sub-schemas; however, whithout random shuffling, we would always select the same column in same order.
                random.shuffle(non_connection_columns)
                window_start_index = 0
                window_end_index = window_start_index + window
                move = True
                while window_start_index <= len(non_connection_columns) - 1:
                    sub_column_list = non_connection_columns[window_start_index : window_end_index]
                    sub_column_list = table_connections + sub_column_list # adding connection columns into the list i.e. adding PKs and FKs
                    table_column_combinations[table_name].append(sub_column_list)
                    window_start_index = window_end_index
                    window_end_index += stride
                    # if window_end_index >= len(non_connection_columns) - 1 + (stride - 1):
                    #     move = False

        db_logger = logging.getLogger(self.db_id)
        # Generate all possible sub-schemas by taking the Cartesion product of column combinations across tables
        sub_schemas = []
        for combination in product(*table_column_combinations.values()):
            ss_tc_names_dict = {table: columns for table, columns in zip(tc_names_dict.keys(), combination)}

            new_sub_schema_tables = {}
            for table_name, columns in ss_tc_names_dict.items():
                table_schema_columns = {}
                for column_name in columns:
                    original_column_info_object = self.schema_structure.tables.get(table_name).columns.get(column_name)
                    column_info_copy = deepcopy(original_column_info_object)
                    table_schema_columns[column_name] = column_info_copy
                    ## Fetch new set of column values when constructin new sub-schema
                    new_examples = execute_sql(db_path=self.db_path,
                                                            sql=f"SELECT DISTINCT `{column_name}` FROM `{table_name}` ORDER BY RANDOM() LIMIT {column_value_cnt}",
                                                            fetch="all", timeout=480)
                    new_examples = [r[0] for r in new_examples]
                    db_logger.info(f"{table_name}-{column_name}: {new_examples}")
                    if new_examples and len(str(new_examples[0])) < 50:
                        table_schema_columns[column_name].examples = new_examples
                        # db_logger.info(f"{table_name}-{column_name} examples after update {table_schema_columns[column_name].examples}") # COMMENT OUT LATER
                new_sub_schema_tables[table_name] = TableSchema(table_schema_columns)
            new_sub_schema = DatabaseSchema(new_sub_schema_tables)
                    
            sub_schemas.append(new_sub_schema)

        return sub_schemas
    
    def generate_column_level_sub_schemas_v1(self) -> List[DatabaseSchema]:
        """
        This function generates column level sub-schemas using existing schema
        """
        schema_detailed_dict = self.convert_schema_to_dict()
        tc_names_dict = schema_detailed_dict.get('tables_and_columns') # tc_names_dict stands for table_column_names dict
        
        # All possible column combination approach.
        # # IMPORTANT: Note that this yield huge number of subschema since all column combinations are considered.
        table_column_combinations = {
            table: get_all_combinations(columns) for table, columns in tc_names_dict.items()
        }

        # Generate all possible sub-schemas by taking the Cartesion product of column combinations across tables
        sub_schemas = []
        for combination in product(*table_column_combinations.values()):
            ss_tc_names_dict = {table: columns for table, columns in zip(tc_names_dict.keys(), combination)}
            
            new_sub_schema_tables = {}
            for table, columns in ss_tc_names_dict.items():
                table_schema_columns = {}
                for column in columns:
                    table_schema_columns[column] = self.schema_structure.tables.get(table).columns.get(column)
                new_sub_schema_tables[table] = TableSchema(table_schema_columns)
            new_sub_schema = DatabaseSchema(new_sub_schema_tables)
                    
            sub_schemas.append(new_sub_schema)

        return sub_schemas

    
    def convert_schema_to_dict(self) -> Dict[str, Dict[str, Any]]:
        """
        Constructs a detailed dictionary for the schema including its db_id, db_path and schema_id
        """

        schema_dict = {
            "schema_id": str(self.schema_id),
            "db_id": str(self.db_id),
            "db_path": str(self.db_path),
            "tables_and_columns": self.schema_structure.to_dict(with_info=False),
            "schema": self.schema_structure.to_dict(with_info=True)
        }

        return schema_dict
    
    def save_schema_to_json_file(self, json_file_path: Path):
        """
        Saving Database Schema into a JSON file
        """

        json_file_path.touch(exist_ok=True)
        detailed_db_schema_dict = self.convert_schema_to_dict()
        with open(json_file_path, 'w') as file:
            json.dump(detailed_db_schema_dict, file, indent=4)