echo "[+] Creating virtual environment..."
python3 -m venv .venv

echo "[+] Activating virtual environment..."
source .venv/bin/activate

echo "[+] Upgrading pip & installing build tools..."
pip install --upgrade pip setuptools build

echo "[+] Installing dependencies from pyproject.toml..."
pip install .

echo "[✓] Done! To activate: source .venv/bin/activate"