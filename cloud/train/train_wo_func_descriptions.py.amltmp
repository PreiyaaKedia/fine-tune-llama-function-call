import os
import argparse
import sys
import logging

import datasets
from datasets import load_dataset, load_from_disk
from peft import LoraConfig
import torch
import transformers
from trl import SFTTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig, TextStreamer
from unsloth.chat_templates import get_chat_template
from unsloth import FastLanguageModel, is_bfloat16_supported
from datasets import load_dataset
import yaml
import json
from datetime import datetime
import re
snapshot_date = datetime.now().strftime("%Y-%m-%d")

# print(os.getcwd())

# sys.path.append(os.path.abspath(os.path.join('..')))
# sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# with open('./config.yaml') as f:
#     d = yaml.load(f, Loader=yaml.FullLoader)

logger = logging.getLogger(__name__)

def load_model(args):

    model_name_or_path = args.model_name_or_path   
    model_kwargs = dict(
        trust_remote_code=True,
        #attn_implementation="flash_attention_2",  # loading the model with flash-attenstion support
        dtype = None,
        max_seq_length = args.max_seq_length,
        load_in_4bit = False,
    )
    model, tokenizer = FastLanguageModel.from_pretrained(model_name_or_path, **model_kwargs)
    return model, tokenizer

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

def prepare_dataset(tokenizer, args):
    
    # Create the cache_dir
    cache_dir = "./outputs/dataset"
    os.makedirs(cache_dir, exist_ok = True)

    # Load the dataset from disk
    train_dataset = load_from_disk(args.train_dir) 
    eval_dataset = load_from_disk(args.val_dir)

    column_names = list(train_dataset.features)

    def apply_chat_template(examples):
        conversations = []
        for system, chat in zip(examples["updated_system"], examples["chat"]):
            try:
                system_message = parse_conversation(system)
                chat_message = parse_conversation(chat)
                message = system_message + chat_message
                conversations.append(message)
            except Exception as e:
                print(e) 

        text = [tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=False) for message in conversations]
        return {"text": text}

    # process the dataseta and drop unused columns
    processed_train_dataset = train_dataset.map(apply_chat_template, cache_file_name = f"{cache_dir}/cache.arrow", batched = True, remove_columns=column_names)
    processed_eval_dataset = eval_dataset.map(apply_chat_template, cache_file_name = f"{cache_dir}/cache.arrow", batched = True, remove_columns=column_names)

    return processed_train_dataset, processed_eval_dataset
    
def main(args):

     ###################
    # Hyper-parameters
    ###################
    # Only overwrite environ if wandb param passed
    if len(args.wandb_project) > 0:
        os.environ['WANDB_API_KEY'] = args.wandb_api_key    
        os.environ["WANDB_PROJECT"] = args.wandb_project
    if len(args.wandb_watch) > 0:
        os.environ["WANDB_WATCH"] = args.wandb_watch
    if len(args.wandb_log_model) > 0:
        os.environ["WANDB_LOG_MODEL"] = args.wandb_log_model
        
    use_wandb = len(args.wandb_project) > 0 or ("WANDB_PROJECT" in os.environ and len(os.environ["WANDB_PROJECT"]) > 0) 
        
    training_config = {"per_device_train_batch_size" : args.train_batch_size,  # Controls the batch size per device
                       "per_device_eval_batch_size" : args.eval_batch_size,    # Controls the batch size for evaluation
                       "gradient_accumulation_steps" : args.grad_accum_steps,
                       "warmup_ratio" : args.warmup_ratio,  # Controls the ratio of warmup steps
                        "learning_rate" : args.learning_rate,  
                        "fp16" : not torch.cuda.is_bf16_supported(),
                        "bf16" : torch.cuda.is_bf16_supported(),
                        "optim" : "adamw_8bit",
                        "lr_scheduler_type" : args.lr_scheduler_type,
                        "output_dir" : args.output_dir,
                        "logging_steps": args.logging_steps,
                        "logging_strategy": "epoch",
                        "save_steps": args.save_steps,
                        "eval_strategy": "epoch",
                        "num_train_epochs": args.epochs,
                        # "load_best_model_at_end": True,
                        "save_only_model": False,
                        "seed" : 0
    }

    peft_config = {
        "r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "bias": "none",
        #"target_modules": "all-linear",
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "modules_to_save": None,
        "use_gradient_checkpointing": "unsloth",
        "use_rslora": False,
        "loftq_config": None,
    }

    checkpoint_dir = os.path.join(args.output_dir, "checkpoints")

    train_conf = TrainingArguments(
        **training_config,
        report_to="wandb" if use_wandb else "azure_ml",
        run_name=args.wandb_run_name if use_wandb else None,    
    )

    model, tokenizer = load_model(args)
    model = FastLanguageModel.get_peft_model(model, **peft_config)

    ###############
    # Setup logging
    ###############
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = train_conf.get_process_log_level()
    logger.setLevel(log_level)

    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process a small summary
    logger.warning(
        f"Process rank: {train_conf.local_rank}, device: {train_conf.device}, n_gpu: {train_conf.n_gpu}"
        + f" distributed training: {bool(train_conf.local_rank != -1)}, 16-bits training: {train_conf.fp16}"
    )
    logger.info(f"Training/evaluation parameters {train_conf}")
    logger.info(f"PEFT parameters {peft_config}") 

    # Load the dataset
    train_dataset, eval_dataset = prepare_dataset(tokenizer, args)

     ###########
    # Training
    ###########
    trainer = SFTTrainer(
        model=model,
        args=train_conf,
        tokenizer = tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        dataset_text_field="text",
        packing = False        # Can make training 5x faster for shorter responses
    )

    # Show current memory stats
    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    logger.info(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
    logger.info(f"{start_gpu_memory} GB of memory reserved.")

    last_checkpoint = None
    if os.path.isdir(checkpoint_dir):
        checkpoints = [os.path.join(checkpoint_dir, d) for d in os.listdir(checkpoint_dir)]
        if len(checkpoints) > 0:
            checkpoints.sort(key=os.path.getmtime, reverse=True)
            last_checkpoint = checkpoints[0]  
    
    trainer_stats = trainer.train(resume_from_checkpoint=last_checkpoint)

    #############
    # Evaluation
    #############
    tokenizer.padding_side = "left"
    metrics = trainer.evaluate()
    metrics["eval_samples"] = len(eval_dataset)
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)
    
    # ############
    # # Save model
    # ############
    os.makedirs(args.model_dir, exist_ok=True)

    if args.save_merged_model:
        print("Save PEFT model with merged 16-bit weights")
        model.save_pretrained_merged("outputs", tokenizer, save_method="merged_16bit")
    else:
        print(f"Save PEFT model: {args.model_dir}/model")    
        model.save_pretrained(f"{args.model_dir}/model")

    tokenizer.save_pretrained(args.model_dir)

def parse_args():
    # setup argparse
    parser = argparse.ArgumentParser()
    # curr_time = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")

    # hyperparameters
    parser.add_argument("--model_name_or_path", default="unsloth/Llama-3.2-3B-Instruct", type=str, help="Model name or path")    
    parser.add_argument("--train_dir", default="./dataset/train_dataset", type=str, help="Input directory for training")
    parser.add_argument("--val_dir", default="./dataset/val_dataset", type=str, help="Input directory for validation")
    parser.add_argument("--model_dir", default="./outputs", type=str, help="output directory for model")
    parser.add_argument("--epochs", default=10, type=int, help="number of epochs")
    parser.add_argument("--output_dir", default="./output_dir", type=str, help="directory to temporarily store when training a model")    
    parser.add_argument("--train_batch_size", default=10, type=int, help="training - mini batch size for each gpu/process")
    parser.add_argument("--eval_batch_size", default=10, type=int, help="evaluation - mini batch size for each gpu/process")
    parser.add_argument("--learning_rate", default=5e-06, type=float, help="learning rate")
    parser.add_argument("--logging_steps", default=2, type=int, help="logging steps")
    parser.add_argument("--save_steps", default=10, type=int, help="save steps")    
    parser.add_argument("--grad_accum_steps", default=2, type=int, help="gradient accumulation steps")
    parser.add_argument("--lr_scheduler_type", default="linear", type=str)
    parser.add_argument("--seed", default=0, type=int, help="seed")
    parser.add_argument("--warmup_ratio", default=0.2, type=float, help="warmup ratio")
    parser.add_argument("--max_seq_length", default=1024, type=int, help="max seq length")
    parser.add_argument("--save_merged_model", type=bool, default=True)
    
    # lora hyperparameters
    parser.add_argument("--lora_r", default=16, type=int, help="lora r")
    parser.add_argument("--lora_alpha", default=32, type=int, help="lora alpha")
    parser.add_argument("--lora_dropout", default=0, type=float, help="lora dropout")
    
    # wandb params
    parser.add_argument("--wandb_api_key", type=str, default="")
    parser.add_argument("--wandb_project", type=str, default="")
    parser.add_argument("--wandb_run_name", type=str, default="")
    parser.add_argument("--wandb_watch", type=str, default="gradients") # options: false | gradients | all
    parser.add_argument("--wandb_log_model", type=str, default="false") # options: false | true

    # parse args
    args = parser.parse_args()

    # return args
    return args

if __name__ == "__main__":
    args = parse_args()
    main(args)