#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${ROOT_DIR}/.venv"

if [ ! -d "${VENV_DIR}" ]; then
  python3 -m venv "${VENV_DIR}"
fi

"${VENV_DIR}/bin/python" -m pip install -U pip
"${VENV_DIR}/bin/python" -m pip install -r "${ROOT_DIR}/requirements.txt"

exec "${VENV_DIR}/bin/streamlit" run "${ROOT_DIR}/code/pp_recommender_app.py"
