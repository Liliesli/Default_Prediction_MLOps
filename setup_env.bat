@echo off
echo [+] Creating virtual environment...
python -m venv .venv

echo [+] Activating virtual environment...
call .venv\Scripts\activate

echo [+] Upgrading pip & installing build tools...
pip install --upgrade pip setuptools build

echo [+] Installing dependencies from pyproject.toml...
pip install .

echo [✓] Done! To activate: call .venv\Scripts\activate
