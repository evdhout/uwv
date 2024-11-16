import pandas as pd
import typer
from loguru import logger

from uwv.cbs.cbs_helper_functions import get_cbs_parquet_file
from uwv.config import CBS80072NED, UWV_MODEL_PARQUET
from uwv.knmi.knmi_attach_temperature import attach_knmi_temperature

app = typer.Typer()


@app.command()
def main(overwrite: bool = False):
    if UWV_MODEL_PARQUET.exists() and not overwrite:
        logger.critical(f"{UWV_MODEL_PARQUET} exists and {overwrite=}")
        raise typer.Abort()

    logger.info("Generating selected features...")

    logger.trace("Open the final main dataset (80072ned) to add features to")
    df = pd.read_parquet(get_cbs_parquet_file(CBS80072NED))

    ## process the KNMI weather data
    logger.trace("Adding KNMI temperature data to the dataset")
    df = attach_knmi_temperature(df)

    logger.trace("Writing parquet file to disk")
    df.to_parquet(UWV_MODEL_PARQUET)

    logger.success(f"Feature generation complete. File saved in {UWV_MODEL_PARQUET}.")


if __name__ == "__main__":
    app()
