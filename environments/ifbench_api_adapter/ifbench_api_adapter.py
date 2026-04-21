import logging
import re
from pathlib import Path

import verifiers as vf
from datasets import Dataset
from loguru import logger as loguru_logger

from api_adapter.ifbench.eval_utils import (
    InputExample,
    normalize_instruction_kwargs,
    test_instruction_following_loose,
)
from src.prompt import SYSTEM_PROMPT

logger = logging.getLogger("verifiers.ifbench")

_ADAPTER_RESPONSE_PATTERN = re.compile(
    r"<\|ADAPTER_RESPONSE_START\|>(.*)<\|ADAPTER_RESPONSE_END\|>", re.DOTALL
)


def _extract_adapter_response(response: str) -> str:
    try:
        return _ADAPTER_RESPONSE_PATTERN.findall(response)[-1]
    except Exception:
        loguru_logger.exception("Error in _extract_adapter_response")
        return ""


def _reward_fn(
    completion: vf.Messages,
    parser: vf.Parser,
    state: vf.State,
    info: vf.Info,
    **kwargs,
) -> float:
    """The ratio of instructions that have been followed."""
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
        if info["claude_reward"]:
            if response.strip() == "CORRECT":
                return 1.0
            else:
                return 0.0
        prompt_to_response = {input_example.prompt: response}
        output_example = test_instruction_following_loose(input_example, prompt_to_response)
        loguru_logger.debug(f"output_example: {output_example}")
        return float(output_example.follow_all_instructions)
    except Exception:
        loguru_logger.exception("Error in _reward_fn")
        logger.exception("Error in _reward_fn")
        return 0.0


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
    dataset = dataset.map(lambda x: {
        "question": (
            f"User Prompt: {x['messages'][0]['content']}\n"
            f"<draft_response>{x['claude_response']}</draft_response>\n"
            "/no_think"
        ),
        "answer": "",
        "info": {**x},
    })

    eval_dataset = dataset.select(range(10))

    parser = vf.MaybeThinkParser(extract_fn=_extract_adapter_response)
    rubric = vf.Rubric(funcs=[_reward_fn], weights=[1.0], parser=parser)

    return vf.SingleTurnEnv(
        dataset=dataset,
        eval_dataset=eval_dataset,
        parser=parser,
        system_prompt=system_prompt,
        rubric=rubric,
    )
