#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXT_ENV="/data1/workspace/airulan/conda_envs/dexbotic_oft_fix"
PY310_HEADERS="/data1/workspace/airulan/.conda-envs/test310/include/python3.10"

export PATH="$EXT_ENV/bin:${PATH:-}"
export PYTHONPATH="$ROOT/.deps/py310:$EXT_ENV/lib/python3.10/site-packages:${PYTHONPATH:-}"
export TORCH_EXTENSIONS_DIR="$ROOT/.cache/torch_extensions"
export CPLUS_INCLUDE_PATH="$PY310_HEADERS:${CPLUS_INCLUDE_PATH:-}"

exec /usr/bin/python3.10 "$@"

