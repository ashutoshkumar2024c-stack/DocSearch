@echo off
rem DocSearch: one-click start on Windows
cd /d "%~dp0"
if not exist venv (
  echo Creating virtual environment...
  python -m venv venv || goto :error
  call venv\Scripts\activate
  python -m pip install --upgrade pip
  pip install -r requirements.txt || goto :error
) else (
  call venv\Scripts\activate
)
if not exist .env copy .env.example .env >nul
rem
streamlit run app.py
goto :eof
:error
echo Setup failed. See the README for manual steps.
pause
