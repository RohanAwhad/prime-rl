"""
Cascade Refine Environment for two-stage LLM training.

Architecture: Query -> M1 (frozen drafter) -> M2 (trainable refiner) -> Answer

M1 generates an initial draft, M2 reviews and refines it.
Only M2 receives weight updates during RL training.
"""

import asyncio
from typing import Any

import verifiers as vf
from openai import AsyncOpenAI

DEFAULT_REFINE_TEMPLATE = """Original question: {query}

Draft answer:
{draft}

Review the draft answer above. If there are any errors or areas for improvement,
provide a corrected and refined answer. If the draft is correct, you may restate
it with any clarifications.

Refined answer:"""


class CascadeRefineEnv(vf.Environment):
    """
    Environment wrapper that adds M1 draft generation before M2 refinement.

    M1 is called within the environment to generate a draft.
    The prompt for M2 includes the original query + M1's draft.
    M1 stays frozen because it's not in the orchestrator's client list.
    """

    def __init__(
        self,
        base_env: vf.Environment,
        m1_base_url: str,
        m1_model: str,
        m1_sampling_args: dict[str, Any] | None = None,
        refine_template: str | None = None,
    ):
        """
        Args:
            base_env: The wrapped environment (e.g., reverse-text, gsm8k)
            m1_base_url: M1 server URL (e.g., "http://localhost:8001/v1")
            m1_model: M1 model name
            m1_sampling_args: Sampling parameters for M1 generation
            refine_template: Template for formatting prompt to M2
        """
        # Don't call super().__init__() - we delegate everything to base_env
        # Just set up the M1 client and store references
        self.base_env = base_env
        self.m1_client = AsyncOpenAI(base_url=m1_base_url, api_key="EMPTY")
        self.m1_model = m1_model
        self.m1_sampling_args = m1_sampling_args or {"temperature": 0.7, "max_tokens": 1024}
        self.refine_template = refine_template or DEFAULT_REFINE_TEMPLATE

        # Copy required attributes from base_env for compatibility
        self.rubric = base_env.rubric
        self.env_id = getattr(base_env, "env_id", None)
        self.env_args = getattr(base_env, "env_args", None)
        self.parser = getattr(base_env, "parser", None)
        self._cleanup_handlers = getattr(base_env, "_cleanup_handlers", [])
        self._teardown_handlers = getattr(base_env, "_teardown_handlers", [])

    def get_dataset(self, **kwargs) -> Any:
        """Delegate to base environment."""
        return self.base_env.get_dataset(**kwargs)

    def get_eval_dataset(self, **kwargs) -> Any:
        """Delegate to base environment."""
        return self.base_env.get_eval_dataset(**kwargs)

    async def setup_state(self, state: vf.State) -> vf.State:
        """Delegate setup_state to base environment."""
        return await self.base_env.setup_state(state)

    async def rollout(
        self,
        input: vf.RolloutInput,
        client: AsyncOpenAI,
        model: str,
        sampling_args: vf.SamplingArgs | None = None,
    ) -> vf.State:
        """
        Run a single rollout with M1 draft injection.

        1. Get M1 draft for the input
        2. Modify the prompt to include the draft
        3. Delegate to base_env.rollout
        """
        original_prompt = input.get("prompt", "")

        # Get M1 draft
        draft = await self._get_m1_draft(original_prompt)

        # Modify input with draft
        modified_input = vf.RolloutInput(**input)
        modified_input["prompt"] = self.refine_template.format(
            query=original_prompt,
            draft=draft,
        )
        modified_input["_original_prompt"] = original_prompt
        modified_input["_m1_draft"] = draft

        # Delegate to base environment
        return await self.base_env.rollout(
            input=modified_input,
            client=client,
            model=model,
            sampling_args=sampling_args,
        )

    async def _get_m1_draft(self, prompt: str) -> str:
        """Call M1 to generate a draft response."""
        response = await self.m1_client.chat.completions.create(
            model=self.m1_model,
            messages=[{"role": "user", "content": prompt}],
            **self.m1_sampling_args,
        )
        return response.choices[0].message.content or ""

    async def run_group(
        self,
        group_inputs: list[vf.RolloutInput],
        client: AsyncOpenAI,
        model: str,
        **kwargs,
    ) -> list[vf.State]:
        """
        Run cascade generation:
        1. Extract original prompts from inputs
        2. Call M1 for each to get drafts (in parallel)
        3. Modify prompts to include drafts
        4. Delegate to base_env.run_group with modified prompts
        """
        # Get M1 drafts for all inputs in parallel
        original_prompts = [inp.get("prompt", "") for inp in group_inputs]
        drafts = await asyncio.gather(*[self._get_m1_draft(p) for p in original_prompts])

        # Create modified inputs with draft included in prompt
        modified_inputs = []
        for inp, draft in zip(group_inputs, drafts):
            modified_inp = vf.RolloutInput(**inp)
            original_prompt = inp.get("prompt", "")
            modified_inp["prompt"] = self.refine_template.format(
                query=original_prompt,
                draft=draft,
            )
            # Store original data for reference/logging
            modified_inp["_original_prompt"] = original_prompt
            modified_inp["_m1_draft"] = draft
            modified_inputs.append(modified_inp)

        # Run M2 generation via base environment
        return await self.base_env.run_group(
            group_inputs=modified_inputs,
            client=client,
            model=model,
            **kwargs,
        )
