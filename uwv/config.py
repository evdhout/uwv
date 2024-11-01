from pathlib import Path

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

CBS_OPENDATA_BASE_URL = "https://opendata.cbs.nl"
CBS_OPENDATA_API_URL = f"{CBS_OPENDATA_BASE_URL}/ODataApi/odata"
CBS_OPENDATA_BULK_URL = f"{CBS_OPENDATA_BASE_URL}/ODataFeed/odata"
CBS_OPENDATA_EXTERNAL_DATA_DIR = EXTERNAL_DATA_DIR / "cbs"
CBS_OPENDATA_PROCESSED_DATA_DIR = PROCESSED_DATA_DIR / "cbs"
CBS80072NED = "80072ned"
CBS80472NED = "80472ned"
CBS80590NED = "80590ned"
CBS81588NED = "81588ned"
CBS81589NED = "81589ned"
CBS83147NED = "83147ned"
CBS83148NED = "83148ned"
CBS83149NED = "83149ned"
CBS83156NED = "83156ned"
CBS83157NED = "83157ned"
CBS83158NED = "83158ned"
CBS83159NED = "83159ned"
CBS84437NED = "84437ned"
CBS83451NED = "83451ned"
CBS85663NED = "85663ned"
CBS85928NED = "85928ned"

CBS_OPENDATA_TABLE_LIST = [
    CBS80072NED,
    CBS80472NED,
    CBS80590NED,
    CBS81588NED,
    CBS81589NED,
    CBS83147NED,
    CBS83148NED,
    CBS83149NED,
    CBS83156NED,
    CBS83157NED,
    CBS83158NED,
    CBS83159NED,
    CBS84437NED,
    CBS83451NED,
    CBS85663NED,
    CBS85928NED,
]

# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove()
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass
