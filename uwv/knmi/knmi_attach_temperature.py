import pandas as pd

from uwv.knmi.knmi_helper_functions import get_knmi_filename


def attach_knmi_temperature(df: pd.DataFrame) -> pd.DataFrame:
    """
    Attach KNMI temperature data to the input DataFrame based on the period.

    This function reads KNMI temperature data from a parquet file, constructs a period
    identifier for both the input DataFrame and the KNMI DataFrame, and merges the temperature
    data based on this period identifier.

    :param df: Input DataFrame containing columns 'period_year' and 'period_quarter_number'
               to create the period identifier.
    :type df: pd.DataFrame
    :return: DataFrame with KNMI temperature data attached based on the period.
    :rtype: pd.DataFrame
    """
    knmi: pd.DataFrame = pd.read_parquet(get_knmi_filename())

    knmi["period"] = df.apply(
        lambda row: f"{row['period_year']}KW0{row['period_quarter_number']}", axis=1
    )

    result_df = df.merge(knmi, on="period", how="left", suffixes=(None, "_knmi"))
    result_df = result_df.drop(
        columns=[
            "period_year_knmi",
            "period_quarter_number_knmi",
        ]
    )
    result_df["avg_temp"] = result_df["avg_temp"].astype("float64")

    return result_df
