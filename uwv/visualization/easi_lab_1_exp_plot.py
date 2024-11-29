import pandas as pd
import plotly.express as px

from uwv.config import CBS80072NED, CBS_OPENDATA_PROCESSED_DATA_DIR, OUTPUT_DIR, RAW_DATA_DIR
from uwv.translations.sbi_translate import sbi_translate

pd.set_option("mode.copy_on_write", True)


def get_data(period: str = "JJ", max_year: int = 2022) -> pd.DataFrame:
    """Load the data from disk

    If the datafile already exists, just read the parquet and return the dataframe.
    Otherwise, create the datafile.

    :param period: either 'JJ' for yearly, or 'KW' for quarterly
    :param max_year: what is the max year to use
    :return: the Pandas dataframe with the requested data
    """
    selection_path = (
        CBS_OPENDATA_PROCESSED_DATA_DIR / f"{CBS80072NED}_{period}_lte{max_year}.parquet"
    )
    if selection_path.exists():
        return pd.read_parquet(selection_path)
    else:
        sbi_with_parents_path = (
            CBS_OPENDATA_PROCESSED_DATA_DIR / f"{CBS80072NED}_with_parents.parquet"
        )
        if sbi_with_parents_path.exists():
            slp_full: pd.DataFrame = pd.read_parquet(sbi_with_parents_path)
        else:
            # the hierarchy file links each sbi category to its parent category
            sbi_hierarchy = pd.read_csv(
                RAW_DATA_DIR / "sbi80072ned_hierarchy.csv", dtype={"Key": "str"}
            )[["Key", "ParentKey", "ParentTitle"]].copy()
            sbi_hierarchy.columns = sbi_hierarchy.columns.str.lower()
            sbi_hierarchy["key"] = sbi_hierarchy["key"].astype("category")

            slp: pd.DataFrame = pd.read_parquet(
                CBS_OPENDATA_PROCESSED_DATA_DIR / f"{CBS80072NED}.parquet"
            )

            slp_full: pd.DataFrame = slp.merge(
                sbi_hierarchy,
                how="left",
                left_on="sbi",
                right_on="key",
                suffixes=("", "_hierarchy"),
                validate="many_to_one",
            )

            # sector 301000 (landbouw) doesn't have branches.
            # So we will set its parent title to itself, treating the sector as a branch
            slp_full.loc[slp_full.sbi == "301000", "parenttitle"] = (
                "A Landbouw, bosbouw en visserij"
            )

            # translate into english
            # slp_full.sbi_title.cat.rename_categories(sbi_dutch_to_english)
            slp_full.sbi_title = slp_full.apply(lambda row: sbi_translate(row.sbi_title), axis=1)
            slp_full.sbi_title = slp_full.sbi_title.astype("category")
            slp_full.parenttitle = slp_full.apply(
                lambda row: sbi_translate(row.parenttitle), axis=1
            )
            slp_full.parenttitle = slp_full.parenttitle.astype("category")
            slp_full.to_parquet(sbi_with_parents_path)
            slp_full.to_csv(sbi_with_parents_path.with_suffix(".csv"), index=False)

    slp_full = slp_full[slp_full.period_year <= max_year]

    # sector 301000 (landbouw) doesn't have branches
    # select it anyway and treat this sector as a branch
    slp_selection: pd.DataFrame = slp_full[
        (slp_full.period_type == period)
        & ((slp_full.category_group_id == 3) | (slp_full.sbi == "301000"))
    ].copy(deep=True)
    slp_selection.sbi_title.cat = slp_selection.sbi_title.cat.remove_unused_categories()

    slp_selection.to_parquet(selection_path)
    slp_selection.to_csv(selection_path.with_suffix(".csv"), index=False)

    return slp_selection


def easi_lab_1_exp_plot(order: str = "descending"):
    slp_year = get_data(period="JJ", max_year=2022)
    slp_quarter = get_data(period="KW", max_year=2022)

    for df, label in zip((slp_year, slp_quarter), ("year", "quarter")):
        fig = px.box(
            df,
            x="sick_leave_percentage",
            y="sbi_title",
            orientation="h",
            color="parenttitle",
            title="No relation to sick leave percentage within business sectors",
            labels={
                "sbi_title": "Branch",
                "parenttitle": "Sector",
                "sick_leave_percentage": "Sick leave percentage",
            },
        )
        fig.update_yaxes(categoryorder=f"median {order}")
        fig.write_html(OUTPUT_DIR / f"easi_lab_1_exp_plot_{label}_{order}.html")

    for q in [1, 2, 3, 4]:
        fig2 = px.box(
            slp_quarter[slp_quarter.period_quarter_number == q],
            x="sick_leave_percentage",
            y="sbi_title",
            orientation="h",
            color="parenttitle",
            title=f"Quarter {q}",
            labels={
                "sbi_title": "Branch",
                "parenttitle": "Sector",
                "sick_leave_percentage": "Sick leave percentage",
            },
        )
        fig2.update_yaxes(categoryorder="median ascending")
        fig2.write_html(OUTPUT_DIR / f"easi_lab_1_exp_plot_quarter_{q}.html")


if __name__ == "__main__":
    easi_lab_1_exp_plot()
