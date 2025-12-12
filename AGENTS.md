# Repository Guidelines for AI Agents

**Build/Test/Lint Commands**
- Install: `uv sync && uv sync --extra fa` (Python 3.12, flash-attn optional)
- Lint: `uv run ruff check src tests` (line length 140, import sorting)
- Tests: `uv run pytest` (add `-m "not slow"` to skip slow, `-m "gpu"` for GPU tests)
- Single test: `uv run pytest path/to/test_file.py::test_name -v`
- Pre-commit: `uv run pre-commit run --all-files` (ruff format + lint)
- Debug training: `uv run torchrun --nproc_per_node=2 src/zeroband/train.py @ configs/training/debug.toml`
- Debug inference: `uv run python src/zeroband/infer.py @ configs/inference/debug.toml`

**Code Style**
- Python 3.12 with explicit type hints; 4‑space indent; 140‑char line limit
- snake_case functions/modules; CamelCase classes; descriptive filenames
- Functional‑style helpers; state in simple containers (dataclasses/pydantic)
- Imports sorted/unused‑free (Ruff enforces); no `# type: ignore` spam
- Error handling: fail fast; no `try/except` unless explicitly requested
- Tests: co‑locate under `tests/<scope>/`; use `@pytest.mark.slow/gpu` markers
- Minimal diffs; avoid refactoring unrelated code; update configs/tests together

**Project Layout**
- `src/zeroband/`: core (`train.py`, `infer.py`, `eval.py`) + `training/`, `inference/`, `rewards/`, `utils/`, `eval/`
- `configs/`: TOML presets for training/inference/debug
- `tests/unit/`, `integration/`, `e2e/` with shared `conftest.py`

**Commit/PR Guidelines**
- Prefix: `feat:`, `fix:`, `refactor:`; include PR numbers `(#123)`
- Include test/lint commands used; note GPU/TP/DP settings when relevant
