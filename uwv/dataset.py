import typer
from loguru import logger

from uwv.cbs.get_cbs_opendata_table import get_cbs_opendata_table
from uwv.cbs.process_cbs_opendata_80072ned import process_cbs_opendata_80072ned
from uwv.config import CBS_OPENDATA_TABLE_LIST
from uwv.knmi.get_knmi_weather_data import get_knmi_data

app = typer.Typer(no_args_is_help=True)


@app.command()
def main(overwrite: bool = False):
    for table in CBS_OPENDATA_TABLE_LIST:
        logger.info(f"Downloading {table=} with {overwrite=} if newer version available")
        get_cbs_opendata_table(table, overwrite=overwrite)

    # post processing CBS datasets
    logger.info("Generating 80072ned with COVID indicator")
    process_cbs_opendata_80072ned(overwrite=overwrite)

    # get KNMI dataset
    logger.info("Collecting KNMI dataset")
    get_knmi_data(overwrite=overwrite)

    logger.info("Generating augmented dataset with additional columns")
    logger.critical("Generating augmented columns not implemented yet")


if __name__ == "__main__":
    app()
