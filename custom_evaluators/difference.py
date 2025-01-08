# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

import os
import re
import json

import numpy as np

from promptflow.client import load_flow
from promptflow.core import AzureOpenAIModelConfiguration


class DifferenceEvaluator:
    def __init__(self, model_config: AzureOpenAIModelConfiguration):
        """
        Initialize an evaluator configured for a specific Azure OpenAI model.

        :param model_config: Configuration for the Azure OpenAI model.
        :type model_config: AzureOpenAIModelConfiguration

        **Usage**

        .. code-block:: python

            eval_fn = CompletenessEvaluator(model_config)
            result = eval_fn(
                question="What is (3+1)-4?",
                answer="First, the result within the first bracket is 3+1 = 4; then the next step is 4-4=0. The answer is 0",
                truth="0")
        """
        # TODO: Remove this block once the bug is fixed
        # https://msdata.visualstudio.com/Vienna/_workitems/edit/3151324
        if model_config.api_version is None:
            model_config.api_version = "2024-05-01-preview"

        prompty_model_config = {"configuration": model_config}
        current_dir = os.path.dirname(__file__)
        prompty_path = os.path.join(current_dir, "difference.prompty")
        assert os.path.exists(prompty_path), f"Please specify a valid prompty file for completeness metric! The following path does not exist:\n{prompty_path}"
        self._flow = load_flow(source=prompty_path, model=prompty_model_config)

    def __call__(self, *, response: str, ground_truth: str, **kwargs):
        """Evaluate correctness of the answer in the context.

        :param answer: The answer to be evaluated.
        :type answer: str
        :param context: The context in which the answer is evaluated.
        :type context: str
        :return: The correctness score.
        :rtype: dict
        """
        # Validate input parameters
        response = str(response or "")
        ground_truth = str(ground_truth or "")

        if not (response.strip()) or not (ground_truth.strip()):
            raise ValueError("All inputs including 'answer' must be non-empty strings.")

        # Run the evaluation flow
        output = self._flow(response=response, ground_truth=ground_truth)
        print(output)
        return json.loads(output)