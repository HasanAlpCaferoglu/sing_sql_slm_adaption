import logging
from typing import Any, Dict, List, Optional



from utils.llm_utils.model import call_llm
from utils.llm_utils.prompt_utils import load_template, load_template_examples


from pydantic import BaseModel


class LLMService:
    def __init__(self, model_name: str, logger: Optional[logging.Logger] = None):
        self.model_name = model_name
        self.logger = logger if logger else logging.getLogger("LLMService")

    def call_llm(self, prompt: str, response_schema: Any = None, llm_model="", llm_tokenizer="", max_tokens:int=32768, temperature: float=0, top_p: float=1) -> Optional[Any]:
        try:
            response_object, prompt_token_cnt, completion_token_cnt, total_token_cnt = call_llm(
                prompt=prompt, 
                model_name=self.model_name, 
                response_json_schema=response_schema, 
                llm_model=llm_model,
                llm_tokenizer=llm_tokenizer,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
            )
            return response_object,  prompt_token_cnt, completion_token_cnt, total_token_cnt
        except Exception as e:
            self.logger.error(f"LLM call failed: {e}. Trying one more time")
            try:
                response_object, prompt_token_cnt, completion_token_cnt, total_token_cnt = call_llm(
                    prompt=prompt, 
                    model_name=self.model_name, 
                    response_json_schema=response_schema, 
                    llm_model=llm_model,
                    llm_tokenizer=llm_tokenizer,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                )
                return response_object,  prompt_token_cnt, completion_token_cnt, total_token_cnt
            except Exception as e:
                self.logger.error(f"LLM call failed in second try: {e}.")
                return {}, 0, 0, 0
        
    def set_logger(self, logger: Optional[logging.Logger]):
        self.logger = logger