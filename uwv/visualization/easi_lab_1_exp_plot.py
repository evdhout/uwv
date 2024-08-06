import pandas as pd
import plotly.express as px
import seaborn as sns
from loguru import logger

from uwv.config import RAW_DATA_DIR, OUTPUT_DIR, CBS_OPENDATA_PROCESSED_DATA_DIR, CBS80072NED

pd.set_option("mode.copy_on_write", True)


def easi_lab_1_exp_plot():
    sbi_hierarchy = pd.read_csv(RAW_DATA_DIR / "sbi80072ned_hierarchy.csv", dtype={'Key': 'str'})[["Key", "ParentKey", "ParentTitle"]].copy()
    sbi_hierarchy.columns = sbi_hierarchy.columns.str.lower()
    sbi_hierarchy['key'] = sbi_hierarchy['key'].astype('category')
    logger.info("sbi_hierarchy")
    sbi_hierarchy.info()
    print(sbi_hierarchy.head())

    slp: pd.DataFrame = pd.read_parquet(CBS_OPENDATA_PROCESSED_DATA_DIR / f"{CBS80072NED}.parquet")
    logger.info("slp.info")
    slp.info()
    print(slp.sbi.cat.categories)

    slp_full: pd.DataFrame = slp.merge(sbi_hierarchy,
                                       how="left",
                                       left_on='sbi',
                                       right_on='key',
                                       suffixes=('', '_hierarchy'),
                                       validate='many_to_one')
    logger.info("slp.full")
    logger.info(slp_full.columns)
    slp_full.info()

    print(slp_full[slp_full.key.notnull()]['sbi'].unique())

    slp_full = slp_full[slp.period_year <= 2022]
    slp_full.info()
    slp_full.to_csv(RAW_DATA_DIR / "slp_full.csv", index=False)
    print(slp_full[slp_full.sbi == "301000"])

    slp_year: pd.DataFrame = slp_full[(slp_full.period_type == "JJ")
                                      & ((slp_full.category_group_id == 3)
                                         | (slp_full.sbi == "301000"))].copy(deep=True)
    slp_year.sbi_title.cat = slp_year.sbi_title.cat.remove_unused_categories()
    slp_year.loc[slp_year.sbi == '301000', "parenttitle"] = 'A Landbouw, bosbouw en visserij'
    logger.info(slp_year.sbi.value_counts())

    fig = px.box(slp_year,
                 x="sick_leave_percentage",
                 y="sbi_title",
                 orientation="h",
                 color='parenttitle',
                 title='Geen relatie tussen bedrijfstakken en bedrijfssectoren in ziektepercentage',
                 labels={
                     'sbi_title': 'Bedrijfstak',
                     'parenttitle': 'Bedrijfssector',
                     'sick_leave_percentage': 'Ziektepercentage'
                 })
    fig.update_yaxes(categoryorder="median ascending")
    fig.show()


if __name__ == '__main__':
    easi_lab_1_exp_plot()
