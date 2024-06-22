import math

import numpy as np
import pandas as pd
import typer
from loguru import logger

from uwv.config import CBS_OPENDATA_PROCESSED_DATA_DIR, CBS80072NED, PROCESSED_DATA_DIR

PREDICTION_NAME = {'le': 'linear extrapolation', 'ra': 'rolling average'}
PREFIXES = ['le', 'ra']

app = typer.Typer(no_args_is_help=True)
slp: pd.DataFrame


def get_previous_slp(row: pd.DataFrame, year_delta: int):
    return slp[(slp.sbi == row.sbi) &
               (slp.period_year == row.period_year - year_delta) &
               (slp.period_quarter_number == row.period_quarter_number)
               ]['sick_leave_percentage'].values[0]


def get_prediction_linear_extrapolation(row: pd.DataFrame) -> int or pd.NA:
    if row.period_year < 1999:
        return pd.NA
    q2 = get_previous_slp(row, year_delta=2)
    q1 = get_previous_slp(row, year_delta=1)
    return q1 + (q1 - q2)


def get_prediction_rolling_average(row: pd.DataFrame) -> int or pd.NA:
    if row.period_year < 2000:
        return pd.NA
    q3 = get_previous_slp(row, year_delta=3)
    q2 = get_previous_slp(row, year_delta=2)
    q1 = get_previous_slp(row, year_delta=1)
    return (q1 + q2 + q3) / 3


def calculate_absolute_error(row: pd.DataFrame, prediction: str) -> float:
    return abs(row.sick_leave_percentage - row[prediction])


def calculate_squared_error(row: pd.DataFrame, prediction: str) -> float:
    return (row.sick_leave_percentage - row[prediction]) ** 2


def calculate_errors():
    for prefix in PREFIXES:
        logger.info(f"Calculating absolute error for {PREDICTION_NAME[prefix]}")

        prediction = f"{prefix}_prediction"
        slp[f'{prefix}_absolute_error'] = slp.apply(lambda row: calculate_absolute_error(row, prediction), axis=1)

        logger.info(f"Calculating squared error for {PREDICTION_NAME[prefix]}")
        slp[f'{prefix}_squared_error'] = slp.apply(lambda row: calculate_squared_error(row, prediction), axis=1)


def round_error_values():
    for prefix in PREFIXES:
        logger.info(f"Rounding absolute and squared error for {PREDICTION_NAME[prefix]}")

        absolute_error = f"{prefix}_absolute_error"
        slp[absolute_error] = slp[absolute_error].apply(
            lambda x: np.round(x, decimals=1) if not pd.isna(x) else pd.NA)

        squared_error = f"{prefix}_squared_error"
        slp[squared_error] = slp[squared_error].apply(
            lambda x: np.round(x, decimals=2) if not pd.isna(x) else pd.NA)


def round_predictions():
    for prefix in PREFIXES:
        logger.info(f"Rounding predictions for {PREDICTION_NAME[prefix]}")
        prediction = f"{prefix}_prediction"
        slp[prediction] = slp[prediction].apply(lambda x: np.round(x, decimals=1) if not pd.isna(x) else pd.NA)


def show_total_errors():
    slp_total = slp[slp.sbi == 'T001081']

    for prefix in PREFIXES:
        print(f'Mean absolute error of {PREDICTION_NAME[prefix]} for total on quarter number')
        print(slp_total.groupby('period_quarter_number')[f'{prefix}_absolute_error'].mean())
        print('Root mean squared error of total on quarter number')
        print(slp_total.groupby('period_quarter_number')[f'{prefix}_squared_error']
              .mean()
              .apply(lambda x: math.sqrt(x)))


@app.command()
def main(force: bool = False, info: bool = False):
    global slp
    try:
        slp = pd.read_parquet(CBS_OPENDATA_PROCESSED_DATA_DIR / f"{CBS80072NED}.parquet")
    except FileNotFoundError:
        logger.error(f"File {CBS_OPENDATA_PROCESSED_DATA_DIR} / {CBS80072NED}.parquet not found.")
        raise typer.Abort()

    baseline_parquet = PROCESSED_DATA_DIR / f"baseline-{CBS80072NED}.parquet"
    baseline_csv = PROCESSED_DATA_DIR / f"baseline-{CBS80072NED}.csv"

    if baseline_parquet.is_file() and not force:
        logger.error(f"Baseline file {baseline_parquet} already exists and {force=}")
        raise typer.Abort

    logger.info("Calculating prediction linear extrapolation")
    slp['le_prediction'] = slp.apply(lambda row: get_prediction_linear_extrapolation(row), axis=1)

    logger.info("Calculating prediction rolling average")
    slp['ra_prediction'] = slp.apply(lambda row: get_prediction_rolling_average(row), axis=1)

    round_predictions()
    calculate_errors()
    round_error_values()

    slp.to_csv(baseline_csv, index=False)
    slp.to_parquet(baseline_parquet, index=False)

    if info:
        show_total_errors()


if __name__ == '__main__':
    app()
