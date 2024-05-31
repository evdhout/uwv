import pandas as pd
import numpy as np
from loguru import logger
import math
# baseline model is Q-4 + (Q-4 - Q-8)

from uwv.config import CBS_OPENDATA_PROCESSED_DATA_DIR, CBS80072NED, DATA_DIR

slp = pd.read_parquet(CBS_OPENDATA_PROCESSED_DATA_DIR / f"{CBS80072NED}.parquet")


def get_prediction(row: pd.DataFrame) -> int or pd.NA:
    if row.period_year < 1999:
        return pd.NA
    q8 = slp[(slp.sbi == row.sbi) &
             (slp.period_year == row.period_year - 2) &
             (slp.period_quarter_number == row.period_quarter_number)
             ]['sick_leave_percentage'].values[0]
    q4 = slp[(slp.sbi == row.sbi) &
             (slp.period_year == row.period_year - 1) &
             (slp.period_quarter_number == row.period_quarter_number)
             ]['sick_leave_percentage'].values[0]
    return q4 + (q4 - q8)


def calculate_absolute_error(row: pd.DataFrame) -> float:
    return abs(row.sick_leave_percentage - row.prediction)


def calculate_squared_error(row: pd.DataFrame) -> float:
    return (row.sick_leave_percentage - row.prediction) ** 2


def main():
    logger.info("Calculating predication")
    slp['prediction'] = slp.apply(lambda row: get_prediction(row), axis=1)
    slp.prediction = slp.prediction.apply(lambda x: np.round(x, decimals=1) if not pd.isna(x) else pd.NA)
    logger.info("Calculating absolute error")
    slp['absolute_error'] = slp.apply(lambda row: calculate_absolute_error(row), axis=1)
    slp.absolute_error = slp.absolute_error.apply(lambda x: np.round(x, decimals=1) if not pd.isna(x) else pd.NA)
    logger.info("Calculating squared error")
    slp['squared_error'] = slp.apply(lambda row: calculate_squared_error(row), axis=1)
    slp.squared_error = slp.squared_error.apply(lambda x: np.round(x, decimals=2) if not pd.isna(x) else pd.NA)

    slp.to_csv(DATA_DIR / f"{CBS80072NED}-baseline.csv", index=False)

    slp_total = slp[slp.sbi == 'T001081']

    print('Mean absolute error of total on quarter number')
    print(slp_total.groupby('period_quarter_number')['absolute_error'].mean())
    print('Root mean squared error of total on quarter number')
    print(slp_total.groupby('period_quarter_number')['squared_error'].mean().apply(lambda x: math.sqrt(x)))


if __name__ == '__main__':
    main()
