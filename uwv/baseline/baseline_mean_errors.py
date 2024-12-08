import math

import pandas as pd
import typer
from loguru import logger

from uwv.config import BASELINE_PARQUET, BASELINE_MEAN_ERRORS_CSV, BASELINE_MEAN_ERRORS_PARQUET

PREDICTION_NAME = {"le": "linear extrapolation", "ra": "rolling average"}
PREFIXES = ["le", "ra"]

app = typer.Typer(no_args_is_help=True)

baseline: pd.DataFrame


def calculate_mean_error() -> {str: pd.DataFrame}:
    """
    Calculates the mean of absolute and squared errors for prediction data grouped
    by 'sbi', 'sbi_title', and 'period_quarter_number'. It transforms the squared
    errors into root mean squared errors and renames the columns accordingly.
    Utilizes predefined `PREFIXES`, and logs the process for each prediction name.

    :return: A dictionary where keys are prefixes and values are DataFrames
        containing mean absolute and root mean squared errors, grouped by
        'sbi', 'sbi_title', and 'period_quarter_number'.
    :rtype: dict[str, pd.DataFrame]
    """
    dfs: {str: pd.DataFrame} = {}
    for prefix in PREFIXES:
        logger.info(
            f"Calculating means of absolute and squared errors for {PREDICTION_NAME[prefix]}"
        )
        df = baseline.groupby(["sbi", "sbi_title", "period_quarter_number"], observed=True)[
            [f"{prefix}_absolute_error", f"{prefix}_squared_error"]
        ].mean()
        df[f"{prefix}_squared_error"] = df[f"{prefix}_squared_error"].apply(lambda x: math.sqrt(x))
        df = df.rename(
            columns={
                f"{prefix}_squared_error": f"root_mean_squared_error_{prefix}",
                f"{prefix}_absolute_error": f"mean_absolute_error_{prefix}",
            }
        )
        dfs[prefix] = df

    return dfs


@app.command()
def main(force: bool = False):
    """
    Main command line interface function for processing data files. This function
    either reads existing processed files or calculates mean absolute errors and
    root mean squared errors before saving the results in both parquet and csv
    formats. The operation can be forced with a True value for the force parameter
    to overwrite existing files.

    :raises FileNotFoundError: If the baseline parquet file does not exist.
    :param force: A boolean flag that forces the recalculation of data even if
                  existing processed files are present. Defaults to False.
    :return: None
    """
    global baseline
    try:
        baseline = pd.read_parquet(BASELINE_PARQUET)
    except FileNotFoundError:
        logger.error(f"File {BASELINE_PARQUET} not found.")
        raise typer.Abort()

    if BASELINE_MEAN_ERRORS_PARQUET.exists() and not force:
        logger.info(f"File {BASELINE_MEAN_ERRORS_PARQUET} exists and {force=}.")
        raise typer.Abort()

    logger.info(f"Calculating mean absolute errors and root mean squared errors")
    dfs = calculate_mean_error()

    merged_df = dfs["le"].join(dfs["ra"], how="inner")

    merged_df.to_parquet(BASELINE_MEAN_ERRORS_PARQUET)
    logger.info(f"Saved {BASELINE_MEAN_ERRORS_PARQUET}")
    merged_df.to_csv(BASELINE_MEAN_ERRORS_CSV)
    logger.info(f"Saved {BASELINE_MEAN_ERRORS_CSV}")


if __name__ == "__main__":
    app()
