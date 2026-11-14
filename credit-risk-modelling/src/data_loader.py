"""Load and validate the UCI Default of Credit Card Clients dataset (id=350)."""
from pathlib import Path
import hashlib
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"  # raw dataset location (not committed)
XLS_NAME = "default of credit card clients.xls"
TARGET = "default payment next month"
EXPECTED_SHAPE = (30000, 25)


def sha256(path: Path) -> str:
    """Return the SHA-256 hex digest of a file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_raw(path: Path | None = None) -> pd.DataFrame:
    """Read the raw XLS (header is on the second row) and validate its shape."""
    path = path or DATA_DIR / XLS_NAME
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python scripts/download_data.py` or see data/README.md."
        )
    try:  # calamine is the supported engine going forward; xlrd is deprecated in pandas
        df = pd.read_excel(path, header=1, engine="calamine")
    except ImportError:
        df = pd.read_excel(path, header=1, engine="xlrd")
    if df.shape != EXPECTED_SHAPE:
        raise ValueError(f"Unexpected shape {df.shape}, expected {EXPECTED_SHAPE}")
    if set(df[TARGET].unique()) != {0, 1}:
        raise ValueError("Target is not binary 0/1")
    return df
