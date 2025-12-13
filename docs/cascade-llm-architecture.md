# Cascade LLM Architecture: Draft → Refine

## Overview

Two-stage LLM architecture where:
- **Model 1 (M1)**: Frozen "drafter" - generates initial answer
- **Model 2 (M2)**: Trainable "refiner" - reviews and rewrites the answer

```
Query → M1 (frozen) → M2 (trainable) → Answer
```

## Architecture Design

### Data Flow

```
┌─────────────┐
│   Query     │
└──────┬──────┘
       │
       ▼
┌─────────────────────────────────────────────┐
│           Custom Environment                 │
│  ┌─────────────────────────────────────┐    │
│  │ 1. Receive query                    │    │
│  │ 2. Call M1 (frozen) → get draft     │    │
│  │ 3. Format prompt: query + draft     │    │
│  │ 4. Return prompt for M2             │    │
│  └─────────────────────────────────────┘    │
└──────┬──────────────────────────────────────┘
       │
       │ prompt = f"{query}\n\nDraft:\n{m1_draft}\n\nReview and refine:"
       ▼
┌─────────────┐
│ M2 (vLLM)   │ ◄── trainable, receives weight updates
│  Refiner    │
└──────┬──────┘
       │
       │ refined_answer
       ▼
┌─────────────┐
│  Verifier   │ ◄── scores the refined answer
│  (reward)   │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│   Trainer   │ ◄── RL training on M2 only
└─────────────┘
```

### Component Responsibilities

| Component | Role | Training |
|-----------|------|----------|
| M1 (Drafter) | Generate initial answer from query | Frozen |
| M2 (Refiner) | Review draft, rewrite with corrections | Trainable |
| Environment | Chain M1 → M2, format prompts | N/A |
| Verifier | Score final refined answer | N/A |

## Implementation Approach

### Option 1: Environment-based (Recommended)

The cleanest approach within PRIME-RL's existing architecture.

**Components:**
1. **M1 Server**: Separate vLLM instance, frozen (no `update_weights` calls)
2. **M2 Server**: Standard PRIME-RL inference server (receives weight updates)
3. **Custom Environment**: Wraps M1 calls, formats prompts for M2
4. **Standard Orchestrator/Trainer**: Unchanged, trains M2

**Pros:**
- Minimal changes to PRIME-RL core
- Clean separation of concerns
- M1 can be a different model size/architecture

**Cons:**
- Two inference servers needed
- Network latency between M1 and environment

### Option 2: Single-server with prefix caching

**Components:**
1. **Single vLLM server**: Hosts both M1 and M2 (if same architecture)
2. **Custom generation logic**: M1 generates, then M2 refines
3. **Weight updates**: Only update M2's weights (via LoRA or partial update)

**Pros:**
- Single server, simpler deployment
- Can share KV cache

**Cons:**
- More invasive changes to inference server
- Complex weight management if models differ

## Detailed Design: Environment-based Approach

### Custom Environment Structure

```python
# src/prime_rl/envs/cascade_refine.py

class CascadeRefineEnv:
    """
    Environment that chains M1 (drafter) → M2 (refiner).

    M1 is called within the environment to generate a draft.
    The environment returns the combined prompt for M2.
    M2's output is then scored by the verifier.
    """

    def __init__(
        self,
        m1_base_url: str,  # e.g., "http://localhost:8001/v1"
        m1_model: str,     # model name for M1
        verifier: Callable,  # scoring function
        draft_prompt_template: str,
        refine_prompt_template: str,
    ):
        self.m1_client = openai.AsyncOpenAI(base_url=m1_base_url)
        self.m1_model = m1_model
        self.verifier = verifier
        self.draft_prompt_template = draft_prompt_template
        self.refine_prompt_template = refine_prompt_template

    async def get_prompt(self, query: str) -> str:
        """Generate M1 draft and return combined prompt for M2."""
        # 1. Format draft prompt
        draft_prompt = self.draft_prompt_template.format(query=query)

        # 2. Call M1 (frozen) to get draft
        response = await self.m1_client.chat.completions.create(
            model=self.m1_model,
            messages=[{"role": "user", "content": draft_prompt}],
        )
        m1_draft = response.choices[0].message.content

        # 3. Format refine prompt for M2
        refine_prompt = self.refine_prompt_template.format(
            query=query,
            draft=m1_draft,
        )
        return refine_prompt

    def score(self, query: str, m2_output: str) -> float:
        """Score M2's refined answer."""
        return self.verifier(query, m2_output)
```

### Prompt Templates

```python
DRAFT_PROMPT_TEMPLATE = """{query}"""

REFINE_PROMPT_TEMPLATE = """Original question: {query}

Draft answer:
{draft}

Review the draft answer above. If there are any errors or areas for improvement,
provide a corrected and refined answer. If the draft is correct, you may restate
it with any clarifications.

Refined answer:"""
```

### Infrastructure Setup

```
┌─────────────────────────────────────────────────────────────┐
│                        PRIME-RL Cluster                      │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐   │
│  │ M1 Server    │    │ M2 Server    │    │ Trainer      │   │
│  │ (frozen)     │    │ (trainable)  │    │ (FSDP)       │   │
│  │ port: 8001   │    │ port: 8000   │    │              │   │
│  └──────────────┘    └──────────────┘    └──────────────┘   │
│         ▲                   ▲                   ▲            │
│         │                   │                   │            │
│         │                   │    weight updates │            │
│         │                   │◄──────────────────┘            │
│         │                   │                                │
│  ┌──────┴───────────────────┴───────────────────────────┐   │
│  │                    Orchestrator                       │   │
│  │  - Loads queries from dataset                        │   │
│  │  - Calls environment (which calls M1)                │   │
│  │  - Sends prompt to M2                                │   │
│  │  - Collects rollouts, sends to trainer               │   │
│  └───────────────────────────────────────────────────────┘   │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Config Structure

```toml
# configs/cascade/m1_infer.toml
[server]
port = 8001

[model]
name = "Qwen/Qwen3-8B"  # or any frozen model

# No weight updates - this server is frozen
```

```toml
# configs/cascade/m2_infer.toml
[server]
port = 8000

[model]
name = "Qwen/Qwen3-8B"  # trainable model

# Standard weight update config
```

```toml
# configs/cascade/orch.toml
[client]
base_url = "http://localhost:8000/v1"  # M2 server

[environment]
type = "cascade_refine"
m1_base_url = "http://localhost:8001/v1"
m1_model = "Qwen/Qwen3-8B"

[dataset]
# query dataset
```

```toml
# configs/cascade/train.toml
# Standard RL training config for M2
[model]
name = "Qwen/Qwen3-8B"

[training]
# GRPO/AIPO config
```

## Training Dynamics

### What M2 Learns

Through RL training, M2 learns to:
1. **Identify errors** in M1's draft
2. **Correct mistakes** (factual, logical, formatting)
3. **Improve clarity** when draft is unclear
4. **Preserve correctness** when draft is already good

### Reward Signal

The verifier scores the **final refined answer**, not the draft.

```python
reward = verifier(query, m2_refined_answer)
```

This incentivizes M2 to:
- Fix M1's mistakes (higher reward than leaving errors)
- Not break correct answers (penalty for introducing errors)

### Potential Reward Shaping

```python
# Optional: reward for improvement over draft
draft_score = verifier(query, m1_draft)
refined_score = verifier(query, m2_refined)

# Reward M2 for improvement
reward = refined_score + alpha * (refined_score - draft_score)
```

## Implementation Details (Based on Codebase Exploration)

### Key Integration Points

1. **`generate_group()` in `src/prime_rl/utils/vf.py`** (lines 13-31)
   - Calls `env.run_group()` with client, model, and sampling args
   - This is where rollouts are generated
   - The environment controls prompt formatting

2. **`setup_clients()` in `src/prime_rl/utils/client.py`** (lines 14-31)
   - Creates OpenAI clients from `client_config.base_url` (list of URLs)
   - Only these clients receive weight updates via `update_weights()`
   - **M1 should NOT be in this list** → stays frozen

3. **`EnvConfig` in `src/prime_rl/orchestrator/config.py`** (lines 192-197)
   ```python
   class EnvConfig(BaseConfig):
       id: str = "reverse-text"
       args: dict = {}  # <-- Pass M1 config here
       name: str | None = None
   ```

4. **Environment loading** in orchestrator (line 115-125 of `orchestrator.py`)
   ```python
   env = vf.EnvGroup(
       envs=[vf.load_environment(env.id, **env.args) for env in config.env],
       ...
   )
   ```

### How to Keep M1 Frozen

The `update_weights()` function (client.py:104-126) updates ALL clients in `admin_clients`:
```python
await asyncio.gather(*[_update_weights(admin_client, weight_dir_posix) for admin_client in admin_clients])
```

**Solution**: Don't include M1 in the orchestrator's `client.base_url` list.
- M1 runs as a separate vLLM server
- Only M2's URL is in `client.base_url`
- M1 is accessed directly by the custom environment via its own client

### Custom Environment Implementation

```python
# src/prime_rl/envs/cascade_refine.py

import verifiers as vf
from openai import AsyncOpenAI
from typing import Any

class CascadeRefineEnv(vf.Environment):
    """
    Environment wrapper that adds M1 draft generation before M2 refinement.

    M1 is called within the environment to generate a draft.
    The prompt for M2 includes the original query + M1's draft.
    """

    def __init__(
        self,
        base_env: vf.Environment,  # The wrapped environment (e.g., gsm8k)
        m1_base_url: str,          # M1 server URL (e.g., "http://localhost:8001/v1")
        m1_model: str,             # M1 model name
        m1_sampling_args: dict | None = None,
        refine_template: str = "Original question: {query}\n\nDraft answer:\n{draft}\n\nReview and refine:",
    ):
        super().__init__()
        self.base_env = base_env
        self.m1_client = AsyncOpenAI(base_url=m1_base_url, api_key="EMPTY")
        self.m1_model = m1_model
        self.m1_sampling_args = m1_sampling_args or {"temperature": 0.7, "max_tokens": 1024}
        self.refine_template = refine_template

    def get_dataset(self, **kwargs):
        """Delegate to base environment."""
        return self.base_env.get_dataset(**kwargs)

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
        client: AsyncOpenAI,  # This is M2's client
        model: str,           # This is M2's model
        **kwargs,
    ) -> list[vf.State]:
        """
        1. Extract original prompts
        2. Call M1 for each to get drafts
        3. Modify prompts to include drafts
        4. Call base_env.run_group with modified prompts
        """
        # Get M1 drafts for all inputs
        import asyncio

        original_prompts = [inp.get("prompt", "") for inp in group_inputs]
        drafts = await asyncio.gather(*[self._get_m1_draft(p) for p in original_prompts])

        # Create modified inputs with draft included
        modified_inputs = []
        for inp, draft in zip(group_inputs, drafts):
            modified_inp = vf.RolloutInput(**inp)
            original_prompt = inp.get("prompt", "")
            modified_inp["prompt"] = self.refine_template.format(
                query=original_prompt,
                draft=draft,
            )
            # Store original for reference
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

    def score(self, state: vf.State) -> float:
        """Delegate scoring to base environment (scores M2's refined output)."""
        return self.base_env.score(state)


def load_cascade_refine_env(
    base_env_id: str,
    m1_base_url: str,
    m1_model: str,
    m1_sampling_args: dict | None = None,
    refine_template: str | None = None,
    **base_env_args,
) -> CascadeRefineEnv:
    """Factory function to create a CascadeRefineEnv."""
    base_env = vf.load_environment(base_env_id, **base_env_args)
    kwargs = {
        "base_env": base_env,
        "m1_base_url": m1_base_url,
        "m1_model": m1_model,
    }
    if m1_sampling_args:
        kwargs["m1_sampling_args"] = m1_sampling_args
    if refine_template:
        kwargs["refine_template"] = refine_template
    return CascadeRefineEnv(**kwargs)
```

### Config Example

```toml
# configs/cascade/orch.toml

[client]
# ONLY M2 - M1 is NOT here (so it stays frozen)
base_url = ["http://localhost:8000/v1"]

[model]
name = "Qwen/Qwen3-8B"

[[env]]
id = "cascade_refine"  # Custom environment ID
args = {
    base_env_id = "gsm8k",  # The actual task environment
    m1_base_url = "http://localhost:8001/v1",  # M1 server (frozen)
    m1_model = "Qwen/Qwen3-8B",
    refine_template = "Question: {query}\n\nDraft:\n{draft}\n\nReview and provide corrected answer:"
}

[sampling]
temperature = 1.0
max_tokens = 1024
```

### Startup Script

```bash
#!/bin/bash
# start_cascade_training.sh

# Terminal 1: M1 (frozen drafter) - port 8001
uv run inference @ configs/cascade/m1_infer.toml &
M1_PID=$!

# Terminal 2: M2 (trainable refiner) - port 8000
uv run inference @ configs/cascade/m2_infer.toml &
M2_PID=$!

# Wait for servers
sleep 30

# Terminal 3: Trainer
uv run trainer @ configs/cascade/train.toml &
TRAINER_PID=$!

# Terminal 4: Orchestrator
uv run orchestrator @ configs/cascade/orch.toml

# Cleanup
kill $M1_PID $M2_PID $TRAINER_PID
```

## Implementation Checklist

- [ ] Register `cascade_refine` environment with verifiers
- [ ] Implement `CascadeRefineEnv` class
- [ ] Create M1/M2 inference configs
- [ ] Create orchestrator config with cascade environment
- [ ] Create trainer config for M2
- [ ] Test with simple environment (e.g., reverse-text)
- [ ] Validate M1 stays frozen (no weight updates)
- [ ] Add logging for M1 drafts
- [ ] Run full RL training loop

## Open Questions

1. **Same or different model for M1/M2?**
   - Same: simpler, can share weights initially
   - Different: M1 can be larger/more capable drafter

2. **Multi-turn refinement?**
   - Single pass: query → draft → refine → done
   - Multi-turn: query → draft → refine → refine → ... → done

3. **Draft visibility in training?**
   - Should trainer see M1's draft for analysis/logging?
   - Useful for debugging M2's refinement behavior
   - Can store in `state["_m1_draft"]` for logging

4. **Inference-time options?**
   - Use M1+M2 cascade always?
   - Or just M2 standalone after training?

## Alternative: Simpler Wrapper Approach

If the full environment wrapping is too complex, a simpler approach:

1. **Pre-generate all M1 drafts** before training
2. **Store drafts in dataset** alongside queries
3. **Use standard environment** with modified prompts

```python
# Preprocessing script
async def generate_drafts(dataset, m1_client, m1_model):
    for example in dataset:
        draft = await m1_client.chat.completions.create(...)
        example["m1_draft"] = draft.choices[0].message.content
    return dataset
```

This is simpler but:
- M1 drafts are static (same for all training)
- Can't adapt M1 behavior during training
- Useful for initial experiments
