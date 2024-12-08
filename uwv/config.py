from pathlib import Path

from black.comments import ProtoComment
from dotenv import load_dotenv
from loguru import logger

# Load environment variables from .env file if it exists
load_dotenv()

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

MODELS_DIR = PROJ_ROOT / "models"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
OUTPUT_DIR = PROJ_ROOT / "output"

CBS_OPENDATA_BASE_URL = "https://opendata.cbs.nl/ODataApi/odata"
CBS_OPENDATA_EXTERNAL_DATA_DIR = EXTERNAL_DATA_DIR / "cbs"
CBS_OPENDATA_PROCESSED_DATA_DIR = PROCESSED_DATA_DIR / "cbs"
CBS80072NED = "80072ned"
CBS_OPENDATA_TABLE_LIST = [CBS80072NED]

BASELINE_CSV = PROCESSED_DATA_DIR / "baseline.csv"
BASELINE_PARQUET = PROCESSED_DATA_DIR / "baseline.parquet"
BASELINE_MEAN_ERRORS_CSV = PROCESSED_DATA_DIR / "baseline_mean_errors.csv"
BASELINE_MEAN_ERRORS_PARQUET = PROCESSED_DATA_DIR / "baseline_mean_errors.parquet"

# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove()
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass
