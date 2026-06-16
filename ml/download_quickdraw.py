"""Download selected Quick, Draw! bitmap .npy files.

Usage:
    python ml/download_quickdraw.py
"""

import json
from pathlib import Path
from urllib.request import urlretrieve

ROOT_DIR = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
NPY_DIR = ARTIFACTS_DIR / "quickdraw_npy"
CLASSES_PATH = Path(__file__).resolve().parent / "classes.json"
BASE_URL = "https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap"


def main() -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    NPY_DIR.mkdir(parents=True, exist_ok=True)

    with open(CLASSES_PATH, "r", encoding="utf-8") as f:
        class_names = json.load(f)

    for name in class_names:
        target = NPY_DIR / f"{name}.npy"
        if target.exists():
            print(f"[SKIP] {target.name} already exists")
            continue

        url_name = name.replace(" ", "%20")
        url = f"{BASE_URL}/{url_name}.npy"
        print(f"[DOWNLOAD] {name} <- {url}")
        urlretrieve(url, target)
        print(f"[DONE] {target}")

    print("All selected QuickDraw classes are downloaded.")


if __name__ == "__main__":
    main()
