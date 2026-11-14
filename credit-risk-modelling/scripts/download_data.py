"""Download the UCI 'Default of Credit Card Clients' dataset into data/raw/.

Usage (from the repository root):
    python scripts/download_data.py

The script downloads the official UCI archive, extracts the Excel file and prints its SHA-256 and
the checksum of the archive used to build this project. A differing archive checksum triggers a
warning, not an error, because a re-packaged archive can differ while the data are identical;
`src.data_loader.load_raw()` independently validates the shape and the target.
"""
import argparse
import hashlib
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

URL = "https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip"
XLS_NAME = "default of credit card clients.xls"
ARCHIVE_SHA256 = "56c885f84457f6680f8438f02bfcdac9579323d8a94465ee5f26e32baa727602"
DEST = Path(__file__).resolve().parents[1] / "data" / "raw"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--url", default=URL, help="archive URL (default: official UCI link)")
    ap.add_argument("--force", action="store_true", help="re-download even if the file exists")
    args = ap.parse_args()

    target = DEST / XLS_NAME
    if target.exists() and not args.force:
        print(f"Already present: {target}")
        return 0
    DEST.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        archive = Path(tmp) / "dataset.zip"
        print(f"Downloading {args.url}")
        try:
            urllib.request.urlretrieve(args.url, archive)
        except Exception as exc:  # network errors vary widely
            print(f"Download failed: {exc}\nDownload it manually from "
                  "https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients "
                  f"and place '{XLS_NAME}' in {DEST}", file=sys.stderr)
            return 1
        digest = sha256(archive)
        if digest != ARCHIVE_SHA256:
            print(f"WARNING: archive SHA-256 differs from the one used for this project:\n  got      {digest}\n  expected {ARCHIVE_SHA256}")
        with zipfile.ZipFile(archive) as z:
            if XLS_NAME not in z.namelist():
                print(f"'{XLS_NAME}' not found in archive; contents: {z.namelist()}", file=sys.stderr)
                return 1
            z.extract(XLS_NAME, DEST)
    print(f"Saved {target}\nFile SHA-256: {sha256(target)}")
    print("Verify with: python -c \"from src.data_loader import load_raw; print(load_raw().shape)\"  # expect (30000, 25)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
