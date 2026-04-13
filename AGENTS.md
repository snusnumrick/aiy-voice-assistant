# Repository Guidance

## Project overview
- This repository contains a Python-based voice assistant for Google AIY Voice Kit V2 hardware.
- The main application code lives in `src/`; tests live in `tests/`; operational helper scripts live in `scripts/`.
- The project uses Poetry metadata in `pyproject.toml`, but local workflows may also use the existing virtualenv/`uv run` setup.

## Working areas
- Prefer editing files in `src/`, `tests/`, `scripts/`, and `docs/`.
- Treat `aiy/`, `google/`, `yandex/`, and similar third-party/vendor-style directories as external code unless the task explicitly requires changes there.
- Do not commit or rewrite local state, caches, logs, or reports unless the user explicitly asks: `.venv/`, `venv/`, `.uv-cache/`, `.pytest_cache/`, `.ruff_cache/`, `logs/`, `reports/`, `wizard_reports/`, `__pycache__/`.

## Configuration and secrets
- Configuration is layered: `config.json` < `user.json` < `APP_...` environment variables < direct constructor args.
- Never add secrets to tracked files. Keep API keys and passwords in `.env` or other untracked local config.
- Be careful when editing startup or ops scripts because they may run `git pull`, system package commands, networking checks, and device-specific setup.

## Code conventions
- Match the existing code style: Python 3.9-compatible, simple module-level functions/classes, and concise docstrings where the file already uses them.
- Keep changes focused and minimal; fix the root cause without broad refactors.
- Preserve existing public names, even when they contain legacy typos such as `responce_player`, unless the task explicitly includes renaming.
- Ruff is configured with line length 100 and lint rules `E`, `F`, `I`, `UP` (with `E501` ignored).

## Validation
- Prefer targeted validation first, then broader checks if needed.
- Recommended test commands:
  - `uv run pytest tests/test_config.py`
  - `uv run pytest tests/test_tools.py`
  - `uv run pytest`
- If relevant, run `uv run ruff check src tests scripts`.

## Agent expectations
- Read `README.md` and the directly affected modules before making substantial changes.
- When modifying behavior, update or add focused tests in `tests/` if an adjacent test pattern already exists.
- Call out hardware-specific or Raspberry Pi-specific assumptions when they prevent local verification.
