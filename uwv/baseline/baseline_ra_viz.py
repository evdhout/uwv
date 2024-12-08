import math

import numpy as np
import pandas as pd
import typer
from loguru import logger

from uwv.config import BASELINE_PARQUET, OUTPUT_DIR

app = typer.Typer(no_args_is_help=True)
baseline: pd.DataFrame




def read_parquet():
    global baseline
    try:
        baseline = pd.read_parquet(BASELINE_PARQUET)
    except FileNotFoundError:
        logger.error(f"File {BASELINE_PARQUET} not found.")
        raise typer.Abort()


@app.command()
def main(force: bool = False):
    read_parquet()




if __name__ == "__main__":
    app()