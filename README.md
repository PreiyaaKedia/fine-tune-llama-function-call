# Fine-Tuning Small Language Models for Function Calling: An End-to-End Demonstration

This repository demonstrates an end-to-end process for fine-tuning small language models (SLMs) to effectively utilize function calling. It covers synthetic data generation, fine-tuning with and without function descriptions, inference using Azure Machine Learning (AzureML), and evaluation using Azure AI Evaluation tools.

## Overview

This project showcases how to:

*   Generate synthetic data tailored for goal-oriented user personas.
*   Fine-tune SLMs using the generated data, both with and without function descriptions.
*   Deploy and serve the fine-tuned models using AzureML.
*   Evaluate the performance of the models using Azure AI Evaluation tools and Prompt Flow.

## File Structure

*   **[`synthetic_data_generation.ipynb`](synthetic-data-generation/synthetic_data_generation.ipynb)**: Notebook for generating synthetic data.
*   **[`train_llama_function_calling.ipynb`](train_custom_llama_function_calling.ipynb)**: Notebook for fine-tuning an SLM with function descriptions.
*   **[`train_llama_function_calling_wo_description.ipynb`](train_llama_function_calling_wo_description.ipynb)**: Notebook for fine-tuning an SLM without function descriptions.
*   **[`serve_ft_llama_model.ipynb`](serve_custom_model_fc_inference.ipynb)**: Notebook for serving the fine-tuned model and performing inference.
*   **[`evaluation/evaluate.py`](evaluation/evaluate.py)**: Script for evaluating the fine-tuned models using Azure AI Evaluation.
*   **[`custom_evaluators/difference.py`](custom_evaluators/difference.py)**: Custom evaluator for comparing model outputs.
*   **[`cloud/`](cloud)**: Directory containing cloud deployment-related files (e.g., Dockerfile, conda environment and training scripts).
*   **[`llama-fc_config.yaml`](llama-fc_config.yaml)**: Configuration file for training with function descriptions.
*   **[`llama-fc-wo-descriptions_config.yaml`](llama-fc-wo-descriptions_config.yaml)**: Configuration file for training without function descriptions.
*   **[`requirements.txt`](requirements.txt)**: Lists the python dependencies.

## Components

### 1. Synthetic Data Generation ([`synthetic_data_generation.ipynb`](synthetic-data-generation/synthetic_data_generation.ipynb))

This notebook uses the Azure AI Evaluation `Simulator` class to generate synthetic conversations between users and an e-commerce assistant. The assistant is designed to handle multiple roles, including creating promo codes, tracking their usage, checking stock levels, and helping customers make shopping decisions.

*   **Key Steps:**
    *   Define user personas and tasks.
    *   Use the `Simulator` to generate conversations based on these personas and tasks.
    *   Format the generated data for fine-tuning.

### 2. Fine-Tuning ([`train_custom_llama_function_calling.ipynb`](train_custom_llama_function_calling.ipynb) and [`train_llama_function_calling_wo_description.ipynb`](train_llama_function_calling_wo_description.ipynb))

These notebooks demonstrate how to fine-tune an SLM using the generated synthetic data. One notebook focuses on fine-tuning with function descriptions, while the other focuses on fine-tuning without function descriptions.

*   **Key Steps:**
    *   Configure AzureML workspace details.
    *   Load and prepare the dataset.
    *   Create an AzureML environment (using a Dockerfile and conda environment).
    *   Create a compute cluster.
    *   Start the training job using the `command` function.
    *   Register the trained model in AzureML.

### 3. Serving and Inference ([`serve_custom_model_fc_inference.ipynb`](serve_custom_model_fc_inference.ipynb))

This notebook demonstrates how to deploy the fine-tuned model as an online endpoint in AzureML and perform inference.

*   **Key Steps:**
    *   Retrieve the registered model.
    *   Create an AzureML environment for serving.
    *   Define a scoring script (`score.py`) to load the model and perform inference.
    *   Create an online endpoint and deployment.
    *   Test the endpoint with sample data.

### 4. Evaluation ([`evaluation/evaluate.py`](evaluation/evaluate.py))

This script evaluates the performance of the fine-tuned models using Azure AI Evaluation tools and Prompt Flow.

*   **Key Steps:**
    *   Load the evaluation dataset.
    *   Define evaluators (e.g., BLEU, GLEU, METEOR, ROUGE).
    *   Run the evaluation using the `evaluate` function from Prompt Flow.
    *   Analyze the evaluation results.

## Configuration

The project uses YAML configuration files ([`llama-fc_config.yaml`](llama-fc_config.yaml) and [`llama-fc-wo-descriptions_config.yaml`](llama-fc-wo-descriptions_config.yaml)) to manage settings such as:

*   Azure subscription, resource group, and workspace details.
*   Data asset names and paths.
*   Model name and path.
*   Training parameters (e.g., epochs, batch size).
*   Environment names and compute cluster details.

Modify these files to suit your specific environment and requirements.

## Requirements

*   Azure subscription
*   AzureML workspace
*   Azure AI Evaluation
*   Python 3.10+
*   Required Python packages (see `requirements.txt`)

## Setup Instructions

1.  Clone the repository.
2.  Install the required Python packages:

    ```bash
    pip install -r requirements.txt
    ```

3.  Configure the YAML configuration files with your AzureML workspace details and other settings.
4.  Follow the instructions in each notebook to generate synthetic data, fine-tune the model, deploy the endpoint, and evaluate the results.

## Usage

1.  **Synthetic Data Generation:** Run the [`synthetic_data_generation.ipynb`](synthetic-data-generation/synthetic_data_generation.ipynb) notebook to generate synthetic data.
2.  **Fine-Tuning:** Run either [`train_custom_llama_function_calling.ipynb`](train_custom_llama_function_calling.ipynb) or [`train_llama_function_calling_wo_description.ipynb`](train_llama_function_calling_wo_description.ipynb) to fine-tune the model.
3.  **Serving:** Run the [`serve_custom_model_fc_inference.ipynb`](serve_custom_model_fc_inference.ipynb) notebook to deploy the model and create an online endpoint.
4.  **Evaluation:** Run the [`evaluation/evaluate.py`](evaluation/evaluate.py) script to evaluate the model's performance.

## Contributing

Contributions are welcome! Please submit a pull request with your proposed changes.