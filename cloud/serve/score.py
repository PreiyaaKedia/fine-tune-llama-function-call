import os
import re
import json
import torch
import base64
import logging

from io import BytesIO
from transformers import AutoTokenizer, AutoProcessor, pipeline
from transformers import AutoModelForCausalLM, AutoProcessor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def init():
    """
    This function is called when the container is initialized/started, typically after create/update of the deployment.
    You can write the logic here to perform init operations like caching the model in memory
    """
    global model
    global tokenizer
    # AZUREML_MODEL_DIR is an environment variable created during deployment.
    # It is the path to the model folder (./azureml-models/$MODEL_NAME/$VERSION)
    # Please provide your model's folder name if there is one
    model_name_or_path = os.path.join(
        os.getenv("AZUREML_MODEL_DIR"), "outputs"
    )
    
    model_kwargs = dict(
        trust_remote_code=True,    
        device_map={"":0},
        torch_dtype="auto" 
    )
    
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **model_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)    

    logging.info("Loaded model.")
    
def run(json_data: str):
    logging.info("Request received")
    data = json.loads(json_data)
    input_data = data["input_data"]
    params = data['params']

    # pipe = pipeline("text-generation", model = model, tokenizer = tokenizer)
    # output = pipe(input_data, **params)
    # result = output[0]["generated_text"]
    # logging.info(f"Generated text : {result}")
    inputs = tokenizer.apply_chat_template(input_data, tokenize = True, add_generation_prompt = True, return_tensors = "pt").to("cuda")
    outputs = model.generate(input_ids = inputs, max_new_tokens = 1024, do_sample = True, temperature = 0.1)
    result = tokenizer.decode(outputs[0][inputs.shape[-1]:], skip_special_tokens = True)

    json_result = {"result" : result}

    return json_result