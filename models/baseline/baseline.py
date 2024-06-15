import pandas as pd
import numpy as np
from loguru import logger
import math
# baseline model is Q-4 + (Q-4 - Q-8) and rolling average of Q-12, Q-8 and Q-4

from uwv.config import CBS_OPENDATA_PROCESSED_DATA_DIR, CBS80072NED, DATA_DIR

slp = pd.read_parquet(CBS_OPENDATA_PROCESSED_DATA_DIR / f"{CBS80072NED}.parquet")


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


def main():
    logger.info("Calculating prediction linear extrapolation")
    slp['le_prediction'] = slp.apply(lambda row: get_prediction_linear_extrapolation(row), axis=1)
    slp.le_prediction = slp.le_prediction.apply(lambda x: np.round(x, decimals=1) if not pd.isna(x) else pd.NA)
    logger.info("Calculating absolute error linear extrapolation")
    slp['le_absolute_error'] = slp.apply(lambda row: calculate_absolute_error(row, 'le_prediction'), axis=1)
    slp.le_absolute_error = slp.le_absolute_error.apply(
        lambda x: np.round(x, decimals=1) if not pd.isna(x) else pd.NA)
    logger.info("Calculating squared error linear extrapolation")
    slp['le_squared_error'] = slp.apply(lambda row: calculate_squared_error(row, 'le_prediction'), axis=1)
    slp.le_squared_error = slp.le_squared_error.apply(
        lambda x: np.round(x, decimals=2) if not pd.isna(x) else pd.NA)

    logger.info("Calculating prediction rolling average")
    slp['ra_prediction'] = slp.apply(lambda row: get_prediction_rolling_average(row), axis=1)
    slp.ra_prediction = slp.ra_prediction.apply(lambda x: np.round(x, decimals=1) if not pd.isna(x) else pd.NA)
    logger.info("Calculating absolute error rolling average")
    slp['ra_absolute_error'] = slp.apply(lambda row: calculate_absolute_error(row, 'ra_prediction'), axis=1)
    slp.ra_absolute_error = slp.ra_absolute_error.apply(
        lambda x: np.round(x, decimals=1) if not pd.isna(x) else pd.NA)
    logger.info("Calculating squared error rolling average")
    slp['ra_squared_error'] = slp.apply(lambda row: calculate_squared_error(row, 'ra_prediction'), axis=1)
    slp.ra_squared_error = slp.ra_squared_error.apply(lambda x: np.round(x, decimals=2) if not pd.isna(x) else pd.NA)

    slp.to_csv(DATA_DIR / f"{CBS80072NED}-baseline.csv", index=False)

    slp_total = slp[slp.sbi == 'T001081']

    print('Mean absolute error of linear extrapolation for total on quarter number')
    print(slp_total.groupby('period_quarter_number')['le_absolute_error'].mean())
    print('Root mean squared error of total on quarter number')
    print(slp_total.groupby('period_quarter_number')['le_squared_error'].mean().apply(lambda x: math.sqrt(x)))

    print('Mean absolute error of rolling average for total on quarter number')
    print(slp_total.groupby('period_quarter_number')['ra_absolute_error'].mean())
    print('Root mean squared error of total on quarter number')
    print(slp_total.groupby('period_quarter_number')['ra_squared_error'].mean().apply(lambda x: math.sqrt(x)))


if __name__ == '__main__':
    main()
