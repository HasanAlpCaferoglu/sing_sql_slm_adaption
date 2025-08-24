import os
import re
import json
from openai import OpenAI
from google import genai
from google.genai import types
from typing import Dict, Union


def extract_xml_answer(text: str) -> str:
    if ("<answer>" in text) and ("</answer>" in text):
        answer = text.split("<answer>")[-1]
        answer = answer.split("</answer>")[0]
    elif ("<answer>" in text) and ("</answer>" not in text):
        answer = text.split("<answer>")[-1]
    elif ("<answer>" not in text) and ("</answer>" in text):
        answer = text.split("</answer>")[0]
    elif ("<answer>" not in text) and ("</answer>" not in text):
        answer = ""
    
    return answer

def extract_xml_reasoning(text: str) -> str:
    if ("<reasoning>" in text) and ("</reasoning>" in text):
        reasoning = text.split("<reasoning>")[-1]
        reasoning = reasoning.split("</reasoning>")[0]
        return reasoning
    elif ("<reasoning>" not in text) and ("</reasoning>" in text):
        reasoning = text.split("</reasoning>")[0]
        return reasoning
    elif ("<reasoning>" in text) and ("</reasoning>" not in text):
        reasoning = reasoning = text.split("<reasoning>")[-1]
        return reasoning
    else:
        return ""
    
def extract_response_part(text: str) -> str:
    if ('### Now is your turn to respond in the above format:' in text) or ("### Answer:" in text):
        answer = text.split("### Answer:")[-1]
        answer = answer.split('### Now is your turn to respond in the above format:')[-1]
        return answer
    else:
        return text
    
def extract_json(text:str) -> Dict:
    text = text.replace("```json", "").replace("```", "").replace("json", "").replace("{{", "{").replace("}}", "}")
    return json.loads(text)

def extract_sql_part(text:str) -> str:
    if "```sqlite" in text:
        text = text.replace("```sqlite", "").replace("```", "").replace("\n", " ")

    if "```sql" in text:
        text = text.replace("```sql", "").replace("```", "").replace("\n", " ")
    
    return text


def parse_llm_output(response_object: Union[Dict, str], model_name: str, output_format: str = "json") -> Dict:
    """
    The function gets a LLM response object, and then it converts the content of it to the Python object.

    Arguments:
        response_object (Dict): LLM response object
    Returns:
        response_object (Dict): Response object whose content changed to dictionary
    """
    if output_format.lower().strip() == "json":
        if 'gpt' in model_name:
            response_object.choices[0].message.content = json.loads(response_object.choices[0].message.content)
        else:
            response_object = response_object.replace("```json", "").replace("```", "").replace("json", "").replace("{{", "{").replace("}}", "}")
            response_object = json.loads(response_object)

    elif output_format.lower().strip() == "xml":
        response_part_text = extract_response_part(response_object)

        reasoning = extract_xml_reasoning(response_part_text)
        answer = extract_xml_answer(response_part_text)
        if "sql" in answer:
            answer = extract_sql_part(answer)
        if "json" in answer:
            answer = extract_json(answer)

        response_object = {
            "reasoning": reasoning,
            "answer": response_object
        }

    return response_object
    


def call_llm(prompt: str, model_name: str, response_json_schema=None, llm_model="", llm_tokenizer="", max_tokens: int=32768, temperature: float=0, top_p: float=1,  n: int=1) -> Dict:
    """
    The functions creates chat response by using chat completion

    Arguments:
        prompt (str): prepared prompt 
        model (str): LLM model used to create chat completion
        max_tokens (int): The maximum number of tokens that can be generated in the chat completion
        temperature (float): Sampling temperature
        top_p (float): Nucleus sampling
        n (int): Number of chat completion for each input message

    Returns:
        response_object (Dict): Object returned by the model
    """

    ###################################
    ########## OPENAI MODELS ##########
    ###################################
    if "gpt" in model_name:
        # if model is OpenAI model
        prompt = prompt.replace("### Enclose your response within three backticks (```):", "")
        prompt = re.sub(r'### Respond in the JSON format as follows:*', '', prompt, flags=re.DOTALL).strip()
        # print("prompt: ", prompt)
        # construct messages
        messages = [
                {"role": "user", "content": prompt}
        ]

        client = OpenAI()
        response_object = client.chat.completions.create(
            model = model_name,
            messages=messages,
            max_completion_tokens = max_tokens,
            response_format = { "type": "json_object" },
            temperature = temperature,
            top_p = top_p,
            n=n,
            presence_penalty = 0.0,
            frequency_penalty = 0.0
        )
        # print("OpenAI: ", response_object)

        try:
            response_object = parse_llm_output(response_object, model_name)
        except Exception as e:
            print(f"Couldn't convert message content to dict. Error: {e}")
            pass

        prompt_token_cnt = response_object.usage.prompt_tokens
        completion_token_cnt = response_object.usage.completion_tokens
        total_token_cnt = response_object.usage.total_tokens

        content_object = response_object.choices[0].message.content
    
    ###################################
    ########## GOOGLE MODELS ##########
    ###################################
    elif 'gemini' in model_name:

        GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
        
        client = genai.Client(api_key=GEMINI_API_KEY)

        if response_json_schema:
            # Removing the structured output generation request from the prompt
            prompt =  re.sub(r'### Enclose your response.*', '', prompt, flags=re.DOTALL).strip()
            prompt = re.sub(r'### Respond in the JSON format as follows:*', '', prompt, flags=re.DOTALL).strip()
            # print(f"==== PROMPT GEMINI =====\n{prompt}")
            
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    max_output_tokens=max_tokens,
                    temperature=temperature,
                    topP=top_p, # Gemini API set topP=0.95 by default
                    responseMimeType='application/json',
                    responseSchema=response_json_schema
                )
            )

            content_object = response
            prompt_token_cnt = response.usage_metadata.prompt_token_count
            completion_token_cnt = response.usage_metadata.candidates_token_count
            total_token_cnt = response.usage_metadata.total_token_count

        else:
            
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    max_output_tokens=max_tokens,
                    temperature=temperature,
                    topP=top_p # Gemini API set topP=0.95 by default
                )
            )

            content_object = response
            prompt_token_cnt = response.usage_metadata.prompt_token_count
            completion_token_cnt = response.usage_metadata.candidates_token_count
            total_token_cnt = response.usage_metadata.total_token_count

    ########################################
    ########## OPEN SOURCE MODELS ##########
    ########################################
    else:

        # construct messages
        messages = [
                {"role": "user", "content": prompt}
        ],

        text = llm_tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        model_inputs = llm_tokenizer(text, return_tensors='pt').to(llm_model.device)

        generated_ids = llm_model.generate(
            **model_inputs,
            max_new_tokens = max_tokens,
            # top_p = top_p,
            # temperature, temperature
        )

        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]

        response = llm_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        # print(f"response: ", response)
        content_object = response
        # content_object = parse_llm_output(response_object=response, model_name=model_name)
        prompt_token_cnt = 0
        completion_token_cnt = 0
        total_token_cnt = 0

    # print(f"==Token Counts: \nPrompt Token Count: {prompt_token_cnt} \nCompletion Token Count: {completion_token_cnt}")
    return content_object, prompt_token_cnt, completion_token_cnt, total_token_cnt


