#!/usr/bin/env bash
# DocSearch: one-command start on macOS / Linux
set -e
cd "$(dirname "$0")"
if [ ! -d venv ]; then
  python3 -m venv venv
  . venv/bin/activate
  pip install --upgrade pip
  pip install -r requirements.txt
else
  . venv/bin/activate
fi
[ -f .env ] || cp .env.example .env
streamlit run app.py
