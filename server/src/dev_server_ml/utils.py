from pathlib import Path

def find_ckpt(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"not found: {path}")

    return str(path) 
