from loguru import logger
import pandas as pd


from uwv.config import CBS_OPENDATA_EXTERNAL_DATA_DIR, CBS_OPENDATA_PROCESSED_DATA_DIR, CBS80072NED


def process_cbs_opendata_80072ned(overwrite: bool = False):
    external_data_dir = CBS_OPENDATA_EXTERNAL_DATA_DIR / CBS80072NED
    csv_file = CBS_OPENDATA_PROCESSED_DATA_DIR / f"{CBS80072NED}.csv"
    parquet_file = CBS_OPENDATA_PROCESSED_DATA_DIR / f"{CBS80072NED}.parquet"

    if parquet_file.exists() and not overwrite:
        logger.error(f"{parquet_file} already exists and {overwrite=}, processing aborted")
        return
    else:
        CBS_OPENDATA_PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    logger.debug(f"Processing CBS OpenData tabel {CBS80072NED}")

    logger.trace("Merge SBI df with groups and keep group title only")
    sbi: pd.DataFrame = pd.read_json(external_data_dir / f"BedrijfskenmerkenSBI2008.json")

    cat_groups: pd.DataFrame = pd.read_json(external_data_dir / f"CategoryGroups.json")

    sbi_cat_groups = (
        pd.merge(
            left=sbi,
            right=cat_groups,
            left_on="CategoryGroupID",
            right_on="ID",
            how="left",
            suffixes=(None, "_cat_group"),
        )
        .drop(["DimensionKey", "Description_cat_group", "ParentID", "ID"], axis="columns")
        .rename(
            columns={
                "CategoryGroupID": "category_group_id",
                "Title_cat_group": "category_group_title",
            }
        )
    )

    logger.trace("Processing period key format into separate columns")
    # The periods table describes the periods
    # the period key is yyyy(KW|JJ)qq
    # yyyy is the year
    # KW indicates it is a quarter : qq is the quarter number
    # JJ indicates it is a year : qq is equal to 00
    # split the period key into separate columns for easy slicing further down the line
    periods: pd.DataFrame = pd.read_json(external_data_dir / f"Perioden.json")
    periods[["period_year", "period_type", "period_quarter_number"]] = periods["Key"].str.extract(
        r"(\d+)(KW|JJ)(\d+)", expand=True
    )
    periods["period_year"] = periods["period_year"].astype(int)
    periods["period_quarter_number"] = periods["period_quarter_number"].astype(int)
    periods["period_quarter"] = periods.apply(
        lambda row: row["period_year"] * 10 + row["period_quarter_number"], axis=1
    )

    logger.trace(
        "Merge sick_leave_percentage with periods and keep period status and description and calculated fields"
    )
    sick_leave_percentage: pd.DataFrame = pd.read_json(external_data_dir / f"TypedDataSet.json")

    slp_periods = (
        pd.merge(
            left=sick_leave_percentage,
            right=periods,
            left_on="Perioden",
            right_on="Key",
            how="left",
            suffixes=(None, "_periods"),
        )
        .drop(labels=["Key", "Description"], axis="columns")
        .rename(columns={"Title": "period_title", "Status": "period_status"})
    )

    logger.trace("Merge slp_periods with sbi information")
    slp_periods_sbi = (
        pd.merge(
            left=slp_periods,
            right=sbi_cat_groups,
            left_on="BedrijfskenmerkenSBI2008",
            right_on="Key",
            how="left",
            suffixes=(None, "_sbi"),
        )
        .drop(labels=["Key"], axis="columns")
        .rename(columns={"Title": "sbi_title", "Description": "sbi_description"})
    )

    # there is a trailing space in the SBI category id, remove it
    slp_periods_sbi["BedrijfskenmerkenSBI2008"] = slp_periods_sbi[
        "BedrijfskenmerkenSBI2008"
    ].apply(lambda x: x.strip())

    logger.trace("Converting column names to uniform column")
    slp_periods_sbi = slp_periods_sbi.rename(
        columns={
            "ID": "id",
            "BedrijfskenmerkenSBI2008": "sbi",
            "Perioden": "period",
            "Ziekteverzuimpercentage_1": "sick_leave_percentage",
        }
    )

    # sbi_description is too verbose to keep
    slp_periods_sbi = slp_periods_sbi.drop("sbi_description", axis=1)

    # Convert columns to categorical
    categorical_columns = [
        "sbi",
        "period",
        "period_title",
        "period_status",
        "period_type",
        "sbi_title",
        "category_group_title",
    ]

    for column in categorical_columns:
        slp_periods_sbi[column] = slp_periods_sbi[column].astype("category")

    # COVID periode from first quarter of 2020 until second quarter of 2022 inclusive
    slp_periods_sbi["covid"] = slp_periods_sbi.period_quarter.map(lambda x: 20200 <= x <= 20222)

    slp_periods_sbi.to_parquet(parquet_file)
    slp_periods_sbi.to_csv(csv_file, index=False)
    logger.trace("%s and %s have been saved" % (parquet_file, csv_file))
