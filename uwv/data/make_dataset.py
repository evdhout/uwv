import typer
from loguru import logger
from uwv.config import CBS_OPENDATA_TABLE_LIST, CBS80072NED
from uwv.data.cbs.get_cbs_opendata_table import get_cbs_opendata_table
from uwv.data.cbs.process_cbs_opendata_80072ned import process_cbs_opendata_80072ned

app = typer.Typer(no_args_is_help=True)


@app.command()
def main(overwrite: bool = False):
    for table in CBS_OPENDATA_TABLE_LIST:
        logger.info(f"Processing {table=} with {overwrite=}")
        if get_cbs_opendata_table(table, overwrite=overwrite) or overwrite:
            if table == CBS80072NED:
                logger.info(f"Performing detailed processing on {table}")
                process_cbs_opendata_80072ned(overwrite=overwrite)


if __name__ == "__main__":
    app()
