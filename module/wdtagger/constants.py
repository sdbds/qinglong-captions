from pathlib import Path

from rich.console import Console

from utils.console_util import print_exception


console = Console(color_system="truecolor", force_terminal=True)

IMAGE_MIME_FILTER = "mime LIKE 'image/%'"
IMAGE_SIZE = 448

DEFAULT_WD14_TAGGER_REPO = "SmilingWolf/wd-v1-4-moat-tagger-v2"
FILES = ["model.onnx", "selected_tags.csv"]
CL_FILES = ["cl_tagger_1_02/model.onnx", "cl_tagger_1_02/tag_mapping.json"]
CSV_FILE = "selected_tags.csv"
JSON_FILE = "cl_tagger_1_02/tag_mapping.json"
PARENTS_CSV = "tag_implications.csv"

CONFIG_DIR = Path(__file__).resolve().parents[2] / "config"
CONFIG = {}
SERIES_EXCLUDE_LIST = set()

try:
    from config.loader import load_config

    CONFIG = load_config(str(CONFIG_DIR))
    SERIES_EXCLUDE_LIST = set(CONFIG.get("wdtagger", {}).get("series_exclude_list", []))
except Exception as e:
    print_exception(console, e, prefix="Error loading config, using default empty exclude list")
