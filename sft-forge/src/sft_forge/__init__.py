"""sft-forge — a resumable workflow for generating SFT training data for LLMs.

Typical programmatic use::

    from sft_forge import load_config, run_pipeline

    cfg = load_config("configs/general.yaml", overrides=["seeds.limit=20"])
    result = run_pipeline(cfg)
    print(result.stats["counts"], result.paths["dataset_chatml"])
"""

from __future__ import annotations

__version__ = "0.1.0"

from .cache import ResponseCache
from .client import Budget, BudgetExceeded, LLMClient
from .config import ConfigError, load_config, load_run_config, save_resolved_config
from .export import export_records
from .filters import apply_filters
from .pipeline import RunResult, StageRunner, build_client, preview, run_pipeline
from .providers import Provider, build_provider
from .schemas import Record, Seed
from .seeds import build_seeds, load_taxonomy

__all__ = [
    "Budget",
    "BudgetExceeded",
    "ConfigError",
    "LLMClient",
    "Provider",
    "Record",
    "ResponseCache",
    "RunResult",
    "Seed",
    "StageRunner",
    "__version__",
    "apply_filters",
    "build_client",
    "build_provider",
    "build_seeds",
    "export_records",
    "load_config",
    "load_run_config",
    "load_taxonomy",
    "preview",
    "run_pipeline",
    "save_resolved_config",
]
