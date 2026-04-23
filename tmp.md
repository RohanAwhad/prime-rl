# RCA: Orchestrator crash at step 268

## Fatal error

```
Attempt 3/3 at step 268 filtered out all 1024 rollouts - crashing orchestrator
RuntimeError: All 1024 rollouts were filtered out on 3 consecutive attempts at step 268
```

## Root cause: reward hacking / model collapse

The model learned to always output `CORRECT` regardless of input. This works for ~half the dataset (where `claude_reward=True`, reward=1.0) but scores 0.0 on the other half (where `claude_reward=False`). Once the model fully collapsed to this strategy, all rollouts in a batch produced zero reward, got filtered out, and the orchestrator crashed after 3 consecutive failed attempts.

## Evidence

- **Entropy collapse**: dropped from ~0.17 early training to 0.001-0.005 by step 267
- **Reward logs**: model responding `CORRECT` to prompts where `claude_reward: False`, scoring 0.0
- **effective_batch_size** in wandb summary: 0.0625 (nearly all rollouts filtered)
- **Grad norm spike**: step 188 saw grad norm jump to 3.5 (vs typical 0.05-0.15), likely the tipping point

## Why it happened

1. Saying `CORRECT` is the easiest path to reward on ~50% of data (claude_reward=True examples)
2. Learning rate (5e-6) is high for RL, allowing rapid convergence to this degenerate policy
3. No entropy regularization or KL penalty to prevent collapse
4. With 64 rollouts per example and only 16 unique prompts per batch, low diversity amplifies the collapse

## Possible fixes

- Lower learning rate (e.g. 1e-6 or 2e-6)
- Reduce `rollouts_per_example` to get more diverse prompts per batch
- Add a format reward giving partial credit for producing `<|ADAPTER_RESPONSE_START|>...<|ADAPTER_RESPONSE_END|>` tags with non-trivial content (not just `CORRECT`)
- Add entropy bonus or KL penalty to prevent collapse
- Consider a reward shaping that penalizes saying `CORRECT` when the draft response is actually wrong
