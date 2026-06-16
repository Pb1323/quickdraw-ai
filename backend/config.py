from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "model.pt"
CLASS_NAMES_PATH = ARTIFACTS_DIR / "class_names.json"
DEFAULT_CLASS_PATH = ROOT_DIR / "ml" / "classes.json"

FRONTEND_DIR = ROOT_DIR / "frontend"
STATIC_DIR = FRONTEND_DIR / "static"
INDEX_HTML = FRONTEND_DIR / "index.html"

MAX_IMAGE_BYTES = 1_500_000
BLANK_PIXEL_RATIO_THRESHOLD = 0.003
