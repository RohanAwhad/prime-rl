import logging
import re
from pathlib import Path

import verifiers as vf
from api_adapter.ifbench.eval_utils import (
    InputExample,
    normalize_instruction_kwargs,
    test_instruction_following_loose,
)
from datasets import Dataset
from loguru import logger as loguru_logger
from src.prompt import SYSTEM_PROMPT

logger = logging.getLogger("verifiers.ifbench")

_ADAPTER_RESPONSE_PATTERN = re.compile(r"<\|ADAPTER_RESPONSE_START\|>(.*)<\|ADAPTER_RESPONSE_END\|>", re.DOTALL)


def _extract_adapter_response(response: str) -> str:
    try:
        return _ADAPTER_RESPONSE_PATTERN.findall(response)[-1]
    except Exception:
        loguru_logger.exception("Error in _extract_adapter_response")
        return ""


def _reward_fn(completion: vf.Messages, parser: vf.Parser, state: vf.State, info: vf.Info, **kwargs) -> float:
    """
    - if response is CORRECT, and claude_reward is True, return 1.0
    - if response is CORRECT, and claude_reward is False, return 0.0
    - if response is an actual_answer,
        - and is same as draft_response, return 0.0  # we dont want the model to repeat the draft response
        - and is different from draft_response
            - and claude_reward is True, return 0.0  # because we want to punish the model for saying that draft response was incorrect.
            - and is incorrect, return 0.5  # we want to reward the model for saying that draft response was incorrect but also punish it slightly for generating the wrong answer.
            - and is correct, return 1.0  # we want to reward the model for saying that draft response was incorrect and generating the correct answer.

    - if claude_reward is True,
        - and response is CORRECT, return 1.0
        - and response is not CORRECT, return 0.0

    if response is CORRECT, and claude_reward is False, return 0.0
    if response is actual_answer,
        - and is same as draft_response, return 0.0
        - and is different from draft_response
            - and is incorrect, return 0.5
            - and is correct, return 1.0

    where,
    - response is the response from the model
    - claude_reward is the reward for the claude response
    - "CORRECT" is model's way of saying LGTM
    - "actual_answer" is the fixed final answer that the model generates
    """
    try:
        loguru_logger.debug(f"Completion: {completion[-1].content}")
        response = parser.parse_answer(completion) or ""
        loguru_logger.debug(f"Response: {response}")
        gt = eval(info["ground_truth"])
        loguru_logger.debug(f"GT: {gt}")
        input_example = InputExample(
            key=info["key"],
            instruction_id_list=gt[0]["instruction_id"],
            prompt=info["messages"][-1]["content"],
            kwargs=normalize_instruction_kwargs(gt[0]["kwargs"]),
        )
        loguru_logger.debug(f"claude reward: {info['claude_reward']}")

        # logs to state
        state["lgtm_count"] = 1 if response.strip() == "CORRECT" else 0
        state["fixme_count"] = 1 if response.strip() != "CORRECT" else 0

        if info["claude_reward"]:
            if response.strip() == "CORRECT":
                return 1.0
            else:
                return 0.0

        if response.strip() == "CORRECT":
            return 0.0
        if response.strip() == info["claude_response"]:
            return 0.0

        reward = 0.5
        prompt_to_response = {input_example.prompt: response}
        output_example = test_instruction_following_loose(input_example, prompt_to_response)
        loguru_logger.debug(f"output_example: {output_example}")
        return reward + (float(output_example.follow_all_instructions) / 2)
    except:
        loguru_logger.exception("Error in reward_fn")
        logger.exception("Error in reward_fn")
        return 0.0


def get_lgtm_count(state: vf.State, **kwargs):
    return state["lgtm_count"]


def get_fixme_count(state: vf.State, **kwargs):
    return state["fixme_count"]


def load_environment(
    data_path: str | None = None,
    system_prompt: str | None = None,
    **kwargs,
) -> vf.Environment:
    if data_path is None:
        data_path = str(Path(__file__).parent / "data" / "input_train_data_with_claude_response_5000_subset.jsonl")
    if system_prompt is None:
        system_prompt = SYSTEM_PROMPT

    dataset = Dataset.from_json(data_path)
    dataset = dataset.map(
        lambda x: {
            "question": (
                f"User Prompt: {x['messages'][0]['content']}\n<draft_response>{x['claude_response']}</draft_response>"
                # "/no_think"
            ),
            "answer": "",
            "info": {**x},
        }
    )
    # split dataset into train and val
    # 80-20 stratified split on claude_reward values
    lgtm_dataset = dataset.filter(lambda x: x["claude_reward"] == True)
    fixme_dataset = dataset.filter(lambda x: x["claude_reward"] == False)

    lgtm_train_dataset, lgtm_val_dataset = lgtm_dataset.train_test_split(test_size=0.2, seed=42).values()
    fixme_train_dataset, fixme_val_dataset = fixme_dataset.train_test_split(test_size=0.2, seed=42).values()

    from datasets import concatenate_datasets

    train_dataset = concatenate_datasets([lgtm_train_dataset, fixme_train_dataset])
    val_dataset = concatenate_datasets([lgtm_val_dataset, fixme_val_dataset])

    parser = vf.MaybeThinkParser(extract_fn=_extract_adapter_response)
    rubric = vf.Rubric(funcs=[_reward_fn, get_lgtm_count, get_fixme_count], weights=[1.0, 0.0, 0.0], parser=parser)

    return vf.SingleTurnEnv(
        dataset=train_dataset,
        eval_dataset=val_dataset,
        parser=parser,
        system_prompt=system_prompt,
        rubric=rubric,
    )
