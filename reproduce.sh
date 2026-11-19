#!/usr/bin/env bash
set -euo pipefail

uv sync --quiet
uv run pytest -q

if [[ -n "${GOOGLE_API_KEY:-}" || -n "${DOTENV_FILE:-}" || -f .env ]]; then
  uv run python -m src.benchmark
else
  echo "GOOGLE_API_KEY not set; offline tests passed and live Gemini 3.5 Flash Lite benchmark was skipped."
fi
