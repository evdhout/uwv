import requests
import pandas as pd
from loguru import logger
from pathlib import Path
from uwv.config import (
    KNMI_AVG_TEMP_MONTH_URL,
    KNMI_AVG_TEMP,
    KNMI_INTERIM_DATA_DIR,
    KNMI_EXTERNAL_DATA_DIR,
    KNMI_PROCESSED_DATA_DIR,
)


def make_knmi_directories():
    KNMI_EXTERNAL_DATA_DIR.mkdir(parents=True, exist_ok=True)
    KNMI_INTERIM_DATA_DIR.mkdir(parents=True, exist_ok=True)
    KNMI_PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)


def get_knmi_avg_montly_temp():
    response = requests.get(KNMI_AVG_TEMP_MONTH_URL)
    if not response.ok:
        raise Exception(f"Url {KNMI_AVG_TEMP_MONTH_URL} returned {response.status_code}")

    with open(KNMI_EXTERNAL_DATA_DIR / KNMI_AVG_TEMP, "wb") as knmi_file:
        knmi_file.write(response.content)


def filter_knmi_avg_monthly_temp():
    with open(KNMI_EXTERNAL_DATA_DIR / KNMI_AVG_TEMP, "r") as knmi_file:
        # remove all lines which preceed the data
        with open(KNMI_INTERIM_DATA_DIR / KNMI_AVG_TEMP, "w") as knmi_interim_file:
            knmi_interim_file.writelines(
                [
                    line.replace(" ", "")
                    for line in filter(lambda line: line[:4] in ["260,", "STN,"], knmi_file)
                ]
            )


def get_knmi_avg_monthly_temp() -> pd.DataFrame:
    df = pd.read_csv(KNMI_INTERIM_DATA_DIR / KNMI_AVG_TEMP, dtype="Int64", na_values=["      "])
    df.columns = map(str.lower, df.columns)
    return df


if __name__ == "__main__":
    make_knmi_directories()
    if not Path(KNMI_INTERIM_DATA_DIR / KNMI_AVG_TEMP).exists():
        logger.info("Downloading KNMI average temperature data")
        get_knmi_avg_montly_temp()
        filter_knmi_avg_monthly_temp()
    pd_knmi = get_knmi_avg_monthly_temp()

    # remove the stn (station name) as we only have De Bilt
    pd_knmi = pd_knmi.drop("stn", axis=1)

    pd_knmi["Q1"] = (pd_knmi.jan + pd_knmi.feb + pd_knmi.mar) / 3
    pd_knmi["Q2"] = (pd_knmi.apr + pd_knmi.may + pd_knmi.jun) / 3
    pd_knmi["Q3"] = (pd_knmi.jul + pd_knmi.aug + pd_knmi.sep) / 3
    pd_knmi["Q4"] = (pd_knmi.oct + pd_knmi.nov + pd_knmi.dec) / 3

    pd_knmi = pd_knmi.rename(columns={"yyyy": "period_year"})

    pd_knmi_long = pd_knmi.melt(
        id_vars="period_year",
        value_vars=["Q1", "Q2", "Q3", "Q4"],
        var_name="period_quarter_number",
        value_name="avg_temp",
        ignore_index=True,
    )

    pd_knmi_long.period_quarter_number = pd_knmi_long.period_quarter_number.str[1]
    pd_knmi_long.period_quarter_number = pd_knmi_long.period_quarter_number.astype("Int64")

    pd_knmi_long.to_parquet(KNMI_PROCESSED_DATA_DIR / f"{KNMI_AVG_TEMP}.parquet")

    pd_knmi.info()
    pd_knmi_long.info()
    print(pd_knmi_long)
