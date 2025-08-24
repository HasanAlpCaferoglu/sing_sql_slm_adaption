import os
import re
import logging
from typing import Any, List
from pathlib import Path


TEMPLATES_ROOT_PATH = "templates_and_examples"

def load_template(template_name: str) -> str:
    """
    Loads a template from a file.

    Args:
        template_name (str): The name of the template to load.

    Returns:
        str: The content of the template.
    """
    file_name = f"template_{template_name}.txt"
    template_path = os.path.join(TEMPLATES_ROOT_PATH, file_name)
    try:
        with open(template_path, "r") as file:
            template = file.read()
        logging.info(f"Template {template_name} loaded successfully.")
        return template
    except FileNotFoundError:
        logging.error(f"Template file not found: {template_path}")
        raise
    except Exception as e:
        logging.error(f"Error loading template {template_name}: {e}")
        raise

def load_template_examples(template_name: str) -> str:
    """
    Loads a examples from a file.

    Args:
        template_name (str): The name of the template to load.

    Returns:
        str: The content of the examples.
    """
    
    file_name = f"examples_{template_name}.txt"
    template_examples_file_path = os.path.join(TEMPLATES_ROOT_PATH, file_name)

    examples_content = ""
    if os.path.exists(template_examples_file_path):
        try:
            with open(template_examples_file_path, "r") as file:
                examples_content = file.read()
        except Exception as e:
            logging.error(f"Error loading template examples {template_name}: {e}")
            raise
    
    return examples_content


def extract_input_variables(template: str) -> List[str]:
        pattern = r'\{(.*?)\}'
        placeholders = re.findall(pattern, template)
        return placeholders


def fill_prompt_template(prompt_template: str, placeholders: List[str]):
    """
    Fills the placeholders in the prompt template

    Args:
        prompt_template (str): template for the llm prompt
    Returns:
        str: Completed prompt
    """

    prompt_template.format(
        
    )