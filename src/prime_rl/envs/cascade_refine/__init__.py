"""
Cascade Refine Environment Package.

This package provides a two-stage LLM environment:
Query -> M1 (frozen drafter) -> M2 (trainable refiner) -> Answer

Usage:
    In orchestrator config:

    [[env]]
    id = "cascade_refine"
    args = {
        base_env_id = "reverse-text",
        m1_base_url = "http://localhost:8001/v1",
        m1_model = "Qwen/Qwen3-0.6B"
    }
"""

from typing import Any

import verifiers as vf

from .env import CascadeRefineEnv, DEFAULT_REFINE_TEMPLATE

__all__ = ["load_environment", "CascadeRefineEnv", "DEFAULT_REFINE_TEMPLATE"]


def load_environment(
    base_env_id: str,
    m1_base_url: str,
    m1_model: str,
    m1_sampling_args: dict[str, Any] | None = None,
    refine_template: str | None = None,
    **base_env_args,
) -> CascadeRefineEnv:
    """
    Factory function to create a CascadeRefineEnv.

    This function is the entry point called by verifiers when loading
    the 'cascade_refine' environment.

    Args:
        base_env_id: ID of the base environment to wrap (e.g., "reverse-text", "gsm8k")
        m1_base_url: URL of the M1 (frozen drafter) server
        m1_model: Model name for M1
        m1_sampling_args: Optional sampling parameters for M1
        refine_template: Optional custom template for the refine prompt
        **base_env_args: Additional arguments passed to the base environment

    Returns:
        CascadeRefineEnv instance wrapping the base environment
    """
    base_env = vf.load_environment(base_env_id, **base_env_args)
    return CascadeRefineEnv(
        base_env=base_env,
        m1_base_url=m1_base_url,
        m1_model=m1_model,
        m1_sampling_args=m1_sampling_args,
        refine_template=refine_template,
    )
