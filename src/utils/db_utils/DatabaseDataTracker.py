import os
import json
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List
from utils.db_utils.db_info_utils import get_db_all_tables, get_db_schema



@dataclass
class DatabaseDataTracker:
    """
    Tracks the database items and their counts for synthetic data
    """
    db_id: str
    db_path: str 
    tables_and_columns: Dict[str, List[str]] = field(init=False)
    tables_and_columns_lower: Dict[str, List[str]] = field(init=False)
    column_counts_lower: Dict[str, Dict[str, int]] = field(init=False)
    logger: logging = field(init=False)

    def __post_init__(self):
        self.logger = logging.getLogger(self.db_id)
        self.column_counts_lower = {}
        self.tables_and_columns = get_db_schema(db_path=self.db_path)
        self.tables_and_columns_lower = {}
        for table, column_list in self.tables_and_columns.items():
            column_list_lower = [col.lower() for col in column_list]
            self.tables_and_columns_lower[table.lower()] = column_list_lower
      
        for table_name, column_list in self.tables_and_columns_lower.items():
            self.column_counts_lower[table_name] = {}
            for column_name in column_list:
                self.column_counts_lower[table_name][column_name] = 0


    def increase_column_count(self, table_name:str, column_name:str, amount:int = 1) -> None:
        """
        Increase the count of a column
        """
        table_name = table_name.strip()
        column_name = column_name.strip()
        
        table_name_lower = table_name.lower().strip()
        column_name_lower = column_name.lower().strip()

        all_table_names_lower = self.tables_and_columns_lower.keys()
        if table_name_lower not in all_table_names_lower:
            # print(f"Cannot increase the column count because couldn't find the table {table_name}")
            return
        
        all_column_names_lower = self.tables_and_columns_lower[table_name_lower]
        if column_name_lower not in all_column_names_lower:
            # print(f"Cannot increase the column count because couldn't find the column {table_name}.{column_name}")
            return
        
        self.column_counts_lower[table_name_lower][column_name_lower] += amount

    def write(self, dir:Path, file_name: str ='data_column_counts.json') -> None:
        """
        Write the column counts into a JSON file
        """
        json_file_path = dir / file_name
        with open(json_file_path, 'w') as file:
            json.dump(self.column_counts_lower, file, indent=4)
        
        self.logger.info(f"Column count information is written in {json_file_path} file.")

    def get_column_counts_lower(self) -> Dict[str, Dict[str, int]]:
        return self.column_counts_lower
    
    def get_low_columns(self, threshold: int) -> Dict[str, List[str]]:
        """
        Function returns columns whose counts are less than the threshold

        Arguments:
            threshold (int): A threshold for the column counts

        Returns:
            Dict[str, List[str]]: Columns whose counts are less than the threshold.
        """
        low_columns_dict: Dict[str, List[str]] = {}
        for t_name, columns_dict in self.column_counts_lower.items():
            for c_name, c_count in columns_dict.items():
                if c_count < threshold:
                    if low_columns_dict.get(t_name, None):
                        low_columns_dict[t_name].append(c_name)
                    else:
                        low_columns_dict[t_name] = [c_name]

        return low_columns_dict