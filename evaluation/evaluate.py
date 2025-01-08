
import json
import pathlib

# set environment variables before importing any other code
from dotenv import load_dotenv
load_dotenv()

import os
import pandas as pd
from pprint import pprint

from promptflow.core import AzureOpenAIModelConfiguration
from promptflow.evals.evaluate import evaluate
from inference import eval
from azure.ai.evaluation import BleuScoreEvaluator, GleuScoreEvaluator, MeteorScoreEvaluator, RougeScoreEvaluator, RougeType
from custom_evaluators.difference import DifferenceEvaluator

bleu = BleuScoreEvaluator()
glue = GleuScoreEvaluator()
meteor = MeteorScoreEvaluator(alpha = 0.9, beta = 3.0, gamma = 0.5)
rouge = RougeScoreEvaluator(rouge_type=RougeType.ROUGE_L)


# Define helper methods
def load_jsonl(path):
    with open(path, "r") as f:
        return [json.loads(line) for line in f.readlines()]

def run_evaluation(name = None, dataset_path = None):
    
    model_config = AzureOpenAIModelConfiguration(
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_version=os.environ["AZURE_OPENAI_API_VERSION"],
        azure_deployment=os.environ["AZURE_OPENAI_EVALUATION_DEPLOYMENT"]
    )

    # Initializing Evaluators
    difference_eval = DifferenceEvaluator(model_config)

    bleu = BleuScoreEvaluator()
    glue = GleuScoreEvaluator()
    meteor = MeteorScoreEvaluator(alpha = 0.9, beta = 3.0, gamma = 0.5)
    rouge = RougeScoreEvaluator(rouge_type=RougeType.ROUGE_L)

    data_path = str(pathlib.Path.cwd() / dataset_path)
    csv_output_path = str(pathlib.Path.cwd() / "./eval_results/eval_results.csv")
    output_path = str(pathlib.Path.cwd() / "./eval_results/eval_results.jsonl")

    result = evaluate(
        # target=copilot_qna,
        evaluation_name=name,
        data=data_path,
        target=eval,
        evaluators={
            "bleu": bleu,
            "gleu": glue,
            "meteor": meteor,
            "rouge" : rouge,
            "difference": difference_eval
        },
        evaluator_config=
        {"default": {
            # only provide additional input fields that target and data do not have
            "ground_truth": "${data.answer}",
            "query": "${data.query}",
            "response": "${target.response}",
            # "match": "${target.match}",
            # "fcall_match": "${target.fcall_match}",
            # "fcall_args_match": "${target.fcall_args_match}",
            # "latency": "${target.latency}",
            # "completeness": {"question": "${data.chat_input}"}
        }}
    )
    
    tabular_result = pd.DataFrame(result.get("rows"))
    tabular_result.to_csv(csv_output_path, index=False)
    tabular_result.to_json(output_path, orient="records", lines=True) 

    return result, tabular_result

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation-name", help="evaluation name used to log the evaluation to AI Studio", type=str)
    parser.add_argument("--dataset-path", help="Test dataset to use with evaluation", type=str, default="./input_data//eval_data.jsonl")
    args = parser.parse_args()

    evaluation_name = args.evaluation_name if args.evaluation_name else "slm-fc-evalation"
    dataset_path = args.dataset_path if args.dataset_path else "./evaluation/evaluation_dataset_small.jsonl"
    
    result, tabular_result = run_evaluation(name=evaluation_name, dataset_path=dataset_path)

    pprint("-----Summarized Metrics-----")
    pprint(result["metrics"])
    pprint("-----Tabular Result-----")
    pprint(tabular_result)
    pprint(f"View evaluation results in AI Studio: {result['studio_url']}")