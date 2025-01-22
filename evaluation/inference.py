import urllib.request
import re
import json
from typing import List, Dict
import os
from rouge_score import rouge_scorer
import pandas as pd
from datasets import load_dataset
import ast
import time

# Request data goes here
# The example below assumes JSON formatting which may be updated
# depending on the format your endpoint expects.
# More information can be found here:
# https://docs.microsoft.com/azure/machine-learning/how-to-deploy-advanced-entry-script

params = {
        "temperature": 0.1,
        "max_new_tokens": 512,
        "do_sample": True,
        "return_full_text": False
    }

def format_input(messages : List[Dict], params):
    """
    Formats the input data and parameters for the inference request.

    Args:
        messages (List[Dict]): A list of dictionaries containing the messages.
        params (dict): A dictionary of parameters for the inference request.

    Returns:
        dict: A dictionary containing the formatted input data and parameters.
    """
    return {
        "input_data": messages,
        "params": params
    }

# sample = {
#     "input_data": 
#         [
#             {'role': 'system', 'content': 'You are an helpful assistant who has access to the following functions to help the user, you can use the functions if needed- { "name": "calculate_shipping_cost", "description": "Calculate the cost of shipping a package", "parameters": { "type": "object", "properties": { "weight": { "type": "number", "description": "The weight of the package in pounds" }, "destination": { "type": "string", "description": "The destination of the package" } }, "required": [ "weight", "destination" ] }}}"'},
#             {'role': 'user', 'content': 'Can you help me with shipping cost for a package?'},
#             {'role': 'assistant', 'content': 'Sure! I can help you with that. Please provide me with the weight and destination of the package.'},
#             {'role': 'user', 'content': 'The weight of the package is 10 pounds and the destination is New York.'}
#         ],
#     "params": params
# }

def run_inference(input_data):
    """
    Runs inference on the input data using the deployed model.

    Args:
        input_data (dict): A dictionary containing the input data for inference.

    Returns:
        dict: The response from the inference endpoint.
    """
    # Replace this with the URL for your deployed model
    url = 'https://llama-endpoint-ft.westus3.inference.ml.azure.com/score'
    # Replace this with the primary/secondary key, AMLToken, or Microsoft Entra ID token for the endpoint
    api_key = '' # Update it with the API key

    params = {
        "temperature": 0.1,
        "max_new_tokens": 512,
        "do_sample": True,
        "return_full_text": False
    }

    body = format_input(input_data, params)
    body = str.encode(json.dumps(body))

    if not api_key:
        raise Exception("A key should be provided to invoke the endpoint")


    headers = {'Content-Type':'application/json', 'Authorization':('Bearer '+ api_key)}

    req = urllib.request.Request(url, body, headers)

    try:
        response = urllib.request.urlopen(req)

        result = json.loads(response.read().decode("utf-8"))["result"]
    except urllib.error.HTTPError as error:
        print("The request failed with status code: " + str(error.code))

        # Print the headers - they include the requert ID and the timestamp, which are useful for debugging the failure
        print(error.info())
        print(error.read().decode("utf8", 'ignore'))

    return result

def parse_conversation(input_string):  
    
    ROLE_MAPPING = {"USER" : "user", "ASSISTANT" : "assistant", "SYSTEM" : "system", "FUNCTION RESPONSE" : "tool"}

    # Regular expression to split the conversation based on SYSTEM, USER, and ASSISTANT  
    pattern = r"(SYSTEM|USER|ASSISTANT|FUNCTION RESPONSE):"  
      
    # Split the input string and keep the delimiters  
    parts = re.split(pattern, input_string)  
      
    # Initialize the list to store conversation entries  
    conversation = []  
      
    # Iterate over the parts, skipping the first empty string  
    for i in range(1, len(parts), 2):  
        role = parts[i].strip()  
        content = parts[i + 1].strip()  
        content = content.replace("<|endoftext|>", "").strip()

        if content.startswith('<functioncall>'):  # build structured data for function call
                # try to turn function call from raw text to structured data
                content = content.replace('<functioncall>', '').strip()
                # replace single quotes with double quotes for valid JSON
                clean_content = content.replace("'{", '{').replace("'}", '}')
                data_json = json.loads(clean_content)
                # Make it compatible with openAI prompt format
                func_call = {'recipient_name': f"functions.{data_json['name']}", 'parameters': data_json['arguments']}
                content = {'tool_uses': [func_call]}
          
        # Append a dictionary with the role and content to the conversation list  
        conversation.append({"role": ROLE_MAPPING[role], "content": content})  
      
    return conversation  

def apply_chat_template(input_data):
        try:
            system_message = parse_conversation(input_data['system'])
            chat_message = parse_conversation(input_data['chat'])
            message = system_message + chat_message
            return message
        except Exception as e:
                print(str(e))
                return None
        
def get_multilevel_qna_pairs(message):
    prompts = []
    answers = []
    for i, item in enumerate(message):
        if item['role'] == 'assistant':
            prompts.append(message[:i])
            answers.append(item["content"])

    return prompts, answers  
        
# dataset = load_dataset("glaiveai/glaive-function-calling-v2", split="train")
# val_dataset = dataset.select(range(2000, 2200))

# processed_val_dataset = []
# for i in range(len(val_dataset)):
#     input_data = apply_chat_template(val_dataset[i])
#     if input_data is None:
#         continue
#     else:
#          processed_val_dataset.append(input_data)

# input_data = []
# answers = []
# for example in processed_val_dataset:
#     prompts, responses = get_multilevel_qna_pairs(example)
#     input_data.extend(prompts)
#     answers.extend(responses)

# with open("./eval_data_jsonl.jsonl", "w") as f:
#     for query, answer in zip(input_data, answers):
#         f.write(json.dumps({"query" : query, "answer" : answer}) + "\n")

def eval(query, answer):
    """
    Evaluate the performance of a model in selecting the correct function based on given prompts.

    Args:
        input_data (List) : List of input prompts for evaluation and benchmarking
        expected_output (List) : List of expected response

    Returns:
        df : Pandas Dataframe with input prompts, actual response, expected response, Match/No Match and ROUGE Score
    """
    # Initialize the ROUGE Scorer where llm response is not function-call
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True) 

    expected_output = answer
    # For generic model response without function-call, set a threshold to classify it as a match
    match_threshold_g = 0.75

    predicted_response = run_inference(query)

    is_func_call = False

    if predicted_response[1:12] == "'tool_uses'":
        is_func_call = True
        try:
            predicted_response = ast.literal_eval(predicted_response)
        except:
            predicted_response = predicted_response
        if isinstance(predicted_response, dict):
            predicted_functions = [func["recipient_name"] for func in predicted_response["tool_uses"]]
            predicted_function_args = [func["parameters"] for func in predicted_response["tool_uses"]]

            actual_functions = [func["recipient_name"] for func in expected_output["tool_uses"]]
            actual_function_args = [func["parameters"] for func in expected_output["tool_uses"]]

            fcall_match = predicted_functions == actual_functions
            fcall_args_match = predicted_function_args == actual_function_args
            match = "Yes" if fcall_match and fcall_args_match else "No"
    else:
        fmeasure_score = scorer.score(expected_output, predicted_response)['rougeL'].fmeasure 
        match = "Yes" if fmeasure_score >= match_threshold_g else "No"
    
    result = {
            "response": predicted_response,
            "fcall_match": fcall_match if is_func_call else "NA",
            "fcall_args_match": fcall_args_match if is_func_call else "NA",
            "match": match
        }
    
    return result
