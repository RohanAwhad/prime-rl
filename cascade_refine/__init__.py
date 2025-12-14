"""
Cascade Refine Environment - Top-level package for verifiers integration.

This package makes cascade_refine directly importable for verifiers.load_environment().
The actual implementation lives in src/prime_rl/envs/cascade_refine/.
"""

from prime_rl.envs.cascade_refine import CascadeRefineEnv, DEFAULT_REFINE_TEMPLATE, load_environment

__all__ = ["load_environment", "CascadeRefineEnv", "DEFAULT_REFINE_TEMPLATE"]
