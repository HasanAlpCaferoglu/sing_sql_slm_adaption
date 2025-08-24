
from pathlib import Path
from dataclasses import dataclass, field

from utils.db_utils.db_info_utils import get_db_schema
from utils.db_utils.db_catalog.csv_utils import load_tables_description
from utils.db_utils.schema_generator import DatabaseSchemaGenerator
from utils.db_utils.schema import DatabaseSchema
@dataclass
class DatabaseGeneralInfo:
    """
    Represents necessary information for a specific database.
    """
    dbs_root_dir: Path  # General root path to the directory that includes databases.
    db_id: str          # Database name.
    prep_dir_name: str = "prep_schemaless"  # Default name for the preparation directory

    # Set default values in __post_init__
    db_directory: Path = field(init=False)
    db_path: Path = field(init=False)
    db_prep_dir: Path = field(init=False)
    original_db_schema_generator: DatabaseSchemaGenerator = field(init=False)

    def __post_init__(self):
        self.db_directory = self.dbs_root_dir / self.db_id
        self.db_path = self.dbs_root_dir / self.db_id / f"{self.db_id}.sqlite"
        self.db_prep_dir = self.dbs_root_dir / self.db_id / self.prep_dir_name
        self.sub_schemas_dir = self.dbs_root_dir / self.db_id / self.prep_dir_name / "sub_schemas"

        self.db_lsh_path = self.db_prep_dir / f'{self.db_id}_lsh.pkl'
        self.db_minhashes_path = self.db_prep_dir / f'{self.db_id}_minhashes.pkl'

        # Ensure the db_prep_dir exists, creating it if necessary
        self.db_prep_dir.mkdir(parents=True, exist_ok=True)
        self.sub_schemas_dir.mkdir(parents=True, exist_ok=True)

        db_original_schema_dict=get_db_schema(db_path=self.db_path)
        schema_with_descriptions = load_tables_description(db_directory_path=self.db_directory, use_value_description=True)
        self.original_db_schema_generator = DatabaseSchemaGenerator(
            tentative_schema=DatabaseSchema.from_schema_dict(db_original_schema_dict),
            schema_with_descriptions=DatabaseSchema.from_schema_dict_with_descriptions(schema_with_descriptions), 
            db_id=self.db_id, 
            db_path=self.db_path
        )
        