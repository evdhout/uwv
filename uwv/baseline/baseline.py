import math

import numpy as np
import pandas as pd
import typer
from loguru import logger

from uwv.config import CBS_OPENDATA_PROCESSED_DATA_DIR, CBS80072NED, BASELINE_CSV, BASELINE_PARQUET

PREDICTION_NAME = {"le": "linear extrapolation", "ra": "rolling average"}
PREFIXES = ["le", "ra"]

app = typer.Typer(no_args_is_help=True)
slp: pd.DataFrame


def get_previous_slp(row: pd.DataFrame, year_delta: int):
    """
    Retrieves the sick leave percentage from a previous year for a given row of
    data, based on the specified `year_delta`. The function filters the `slp`
    dataframe to match records with the same `sbi` and `period_quarter_number`
    and subtracts `year_delta` from `period_year` to find the corresponding
    sick leave percentage.

    :param row: The DataFrame row containing the columns `sbi`, `period_year`,
                and `period_quarter_number`.
    :type row: pd.DataFrame
    :param year_delta: The number of years to look back for the sick leave
                       percentage.
    :type year_delta: int
    :return: The sick leave percentage from the `slp` DataFrame for the
             specified year delta.
    :rtype: float
    """
    return slp[
        (slp.sbi == row.sbi)
        & (slp.period_year == row.period_year - year_delta)
        & (slp.period_quarter_number == row.period_quarter_number)
    ]["sick_leave_percentage"].values[0]


def get_prediction_linear_extrapolation(row: pd.DataFrame) -> int or pd.NA:
    """
    Calculates a prediction for a given row of data by using linear
    extrapolation based on previously observed values. For rows where
    the `period_year` is less than 1999, a missing value indicator
    (pd.NA) is returned.

    This function uses the difference between the last two available
    years' SLP values to predict the next value. If the computation
    cannot proceed due to insufficient data (i.e., `period_year`
    predates 1999), no prediction is made.

    :param row:
        A row from a DataFrame, expected to include a `period_year`
        and SLP values for previous years.
    :return:
        The predicted SLP value derived from linear extrapolation
        or pd.NA if conditions are not met for prediction.
    """
    if row.period_year < 1999:
        return pd.NA
    q2 = get_previous_slp(row, year_delta=2)
    q1 = get_previous_slp(row, year_delta=1)
    return q1 + (q1 - q2)


def get_prediction_rolling_average(row: pd.DataFrame) -> int or pd.NA:
    """
    Calculate the rolling average prediction value based on historical data.
    The function checks if the row's period year is earlier than 2000 and
    returns 'pd.NA' if true. For other cases, it computes the average of the
    prediction values of the three previous years.

    :param row: A DataFrame row containing the historical data with a
                'period_year' field and prediction values.
    :return: The rolling average prediction value as an integer if calculated,
             otherwise returns 'pd.NA'.
    """
    if row.period_year < 2000:
        return pd.NA
    q3 = get_previous_slp(row, year_delta=3)
    q2 = get_previous_slp(row, year_delta=2)
    q1 = get_previous_slp(row, year_delta=1)
    return (q1 + q2 + q3) / 3


def calculate_absolute_error(row: pd.DataFrame, prediction: str) -> float:
    """
    Calculate the absolute error between the actual sick leave percentage and
    a predicted percentage from a specified column in the provided DataFrame
    row. This function is useful in evaluating the accuracy of predictions
    made by a model or system by comparing them to actual observed values.

    :param row: A pandas DataFrame row containing an 'sick_leave_percentage'
                column and a column specified by 'prediction'.
    :type row: pd.DataFrame
    :param prediction: The name of the column in 'row' that contains the
                       predicted sick leave percentage.
    :type prediction: str
    :return: The absolute error between the actual sick leave percentage and
             the predicted value from the specified column.
    :rtype: float
    """
    return abs(row.sick_leave_percentage - row[prediction])


def calculate_squared_error(row: pd.DataFrame, prediction: str) -> float:
    """
    Computes the squared error between the actual sick leave percentage and
    the predicted value from a DataFrame row. This method is useful
    in evaluating the performance of predictive models by quantifying
    the error magnitude.

    :param row: A DataFrame row containing the actual sick leave percentage
    :type row: pd.DataFrame
    :param prediction: The column name in the DataFrame that holds
                       the predicted values
    :type prediction: str
    :return: The squared error, which is a non-negative float representing
             the squared difference between the actual and predicted values
    :rtype: float
    """
    return (row.sick_leave_percentage - row[prediction]) ** 2


def calculate_errors():
    """
    Calculate absolute and squared errors for predictions.

    This function iterates through a set of prediction prefixes and calculates
    the absolute and squared errors for predictions associated with each prefix.
    It logs the calculation process and updates the `slp` DataFrame with new
    columns for absolute and squared errors for each prediction.

    :raises KeyError: If a prefix in `PREFIXES` does not exist in `PREDICTION_NAME`.
    :raises Exception: If there's an issue applying the error calculation functions
                       to the DataFrame.

    :return: None
    """
    for prefix in PREFIXES:
        logger.info(f"Calculating absolute error for {PREDICTION_NAME[prefix]}")

        prediction = f"{prefix}_prediction"
        slp[f"{prefix}_absolute_error"] = slp.apply(
            lambda row: calculate_absolute_error(row, prediction), axis=1
        )

        logger.info(f"Calculating squared error for {PREDICTION_NAME[prefix]}")
        slp[f"{prefix}_squared_error"] = slp.apply(
            lambda row: calculate_squared_error(row, prediction), axis=1
        )


def round_error_values():
    """
    Rounds the error values in the dataset for both absolute and squared errors.

    This function iterates through a predefined list of prefixes and rounds the
    corresponding error values in a dataset. The absolute errors are rounded to
    one decimal place and squared errors are rounded to two decimal places.
    The rounding is applied conditionally based on whether the value is NA or not.

    :raises ValueError: If the dataset contains any invalid data types.
    :raises KeyError: If any of the expected columns are missing from the dataset.
    """
    for prefix in PREFIXES:
        logger.info(f"Rounding absolute and squared error for {PREDICTION_NAME[prefix]}")

        absolute_error = f"{prefix}_absolute_error"
        slp[absolute_error] = slp[absolute_error].apply(
            lambda x: np.round(x, decimals=1) if not pd.isna(x) else pd.NA
        )

        squared_error = f"{prefix}_squared_error"
        slp[squared_error] = slp[squared_error].apply(
            lambda x: np.round(x, decimals=2) if not pd.isna(x) else pd.NA
        )


def round_predictions():
    """
    Adjusts the predictions by rounding them to one decimal place for each
    prediction in the specified prefixes. This function iterates over a
    predefined list of prefixes, logging the action for each prediction name,
    and applies rounding to the corresponding prediction values in a DataFrame,
    except for missing values.

    :return: None
    """
    for prefix in PREFIXES:
        logger.info(f"Rounding predictions for {PREDICTION_NAME[prefix]}")
        prediction = f"{prefix}_prediction"
        slp[prediction] = slp[prediction].apply(
            lambda x: np.round(x, decimals=1) if not pd.isna(x) else pd.NA
        )


def show_total_errors():
    """
    Displays the mean absolute error and root mean squared error for a given
    prediction prefix across different periods in a dataset filtered by a
    specific service level prediction (sbi).

    The function filters the dataset using a specific sbi code and calculates
    both mean absolute error and root mean squared error for various
    prediction prefixes over quarters. This information is crucial for
    evaluating the accuracy of model predictions over time.

    :return: None
    """
    slp_total = slp[slp.sbi == "T001081"]

    for prefix in PREFIXES:
        print(f"Mean absolute error of {PREDICTION_NAME[prefix]} for total on quarter number")
        print(slp_total.groupby("period_quarter_number")[f"{prefix}_absolute_error"].mean())
        print("Root mean squared error of total on quarter number")
        print(
            slp_total.groupby("period_quarter_number")[f"{prefix}_squared_error"]
            .mean()
            .apply(lambda x: math.sqrt(x))
        )


@app.command()
def main(force: bool = False, info: bool = False):
    """
    Main function of the application, executed via command line.

    This function calculates the predictions using linear extrapolation and rolling
    average from the parquet file with sick leave percentages.
    The results are saved in both CSV and Parquet file formats. If the output
    file already exists and the `force` parameter is not set to True, the process
    aborts.
    Additionally, it can provide mean error information on all quarters.

    :raises FileNotFoundError: If the input file does not exist.
    :param force: Boolean flag that determines whether to overwrite existing
                  baseline files if they exist.
    :type force: bool
    :param info: Boolean flag that determines whether to display total errors
                 information after processing.
    :type info: bool
    :return: None
    """
    global slp
    try:
        slp = pd.read_parquet(CBS_OPENDATA_PROCESSED_DATA_DIR / f"{CBS80072NED}.parquet")
    except FileNotFoundError:
        logger.error(f"File {CBS_OPENDATA_PROCESSED_DATA_DIR} / {CBS80072NED}.parquet not found.")
        raise typer.Abort()

    if BASELINE_PARQUET.is_file() and not force:
        logger.error(f"Baseline file {BASELINE_PARQUET} already exists and {force=}")
        raise typer.Abort

    logger.info("Calculating prediction linear extrapolation")
    slp["le_prediction"] = slp.apply(lambda row: get_prediction_linear_extrapolation(row), axis=1)

    logger.info("Calculating prediction rolling average")
    slp["ra_prediction"] = slp.apply(lambda row: get_prediction_rolling_average(row), axis=1)

    round_predictions()
    calculate_errors()
    round_error_values()

    slp.to_csv(BASELINE_CSV, index=False)
    slp.to_parquet(BASELINE_PARQUET, index=False)

    if info:
        show_total_errors()


if __name__ == "__main__":
    app()
