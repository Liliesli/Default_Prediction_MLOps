# mlops/mlops_config.py
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODEL_OUTPUT_DIR = PROJECT_ROOT / ".models"

# 필요한 디렉토리 생성
for dir_path in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODEL_OUTPUT_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)