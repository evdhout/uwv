import typer
from typing_extensions import Annotated
from loguru import logger

from uwv.config import CBS_OPENDATA_EXTERNAL_DATA_DIR
from uwv.data.cbs.is_cbs_opendata_table_updated import is_cbs_opendata_table_updated
from uwv.data.cbs.download_cbs_opendata_table import download_cbs_opendata_table


def get_cbs_opendata_table(
    cbs_table_id: Annotated[str, typer.Argument()],
    overwrite: bool = False,
) -> bool:
    """
    Get cbs table by provided table_id

    :param cbs_table_id: the id of the cbs opendata table
    :param overwrite: overwrite existing files if they exists
    :return: None
    """

    cbs_table_path = CBS_OPENDATA_EXTERNAL_DATA_DIR / cbs_table_id

    if not cbs_table_path.is_dir():
        logger.debug(f"Creating CBS OpenData table {cbs_table_id} data path: {cbs_table_path}")
        cbs_table_path.mkdir(exist_ok=True, parents=True)
    else:
        if is_cbs_opendata_table_updated(cbs_table_id):
            if not overwrite:
                logger.error(f"New version online but {overwrite=}, tables not updated")
                return False
        else:
            return False

    download_cbs_opendata_table(cbs_table_id)

    return True


if __name__ == "__main__":
    typer.run(get_cbs_opendata_table)
