from pathlib import Path
import json
import typer
from loguru import logger

from uwv.config import CBS_OPENDATA_BASE_URL, CBS_OPENDATA_EXTERNAL_DATA_DIR
from uwv.data.cbs.cbs_helper_functions import get_dict_from_json_response


def is_cbs_opendata_table_updated(table_id: str) -> bool:
    """
    Check whether there is a new version of the CBS OpenData table online

    :param table_id: cbs id of the table
    :return:
    """

    table_infos_file = CBS_OPENDATA_EXTERNAL_DATA_DIR / table_id / f"{table_id}_TableInfos.json"
    try:
        with open(table_infos_file, "rb") as f:
            table_infos = json.load(f)
    except FileNotFoundError:
        logger.error(f"TableInfos file not found: {table_infos_file}")
        raise typer.Abort(1)
    except OSError as e:
        raise e

    logger.info(
        f'Checking if table {table_id} has changed. Local modification date = {table_infos["Modified"]} '
    )
    table_infos_new = get_dict_from_json_response(f"{CBS_OPENDATA_BASE_URL}/{table_id}/TableInfos")
    if table_infos["Modified"] != table_infos_new["Modified"]:
        logger.info(f"Table {table_id} has changed")
        logger.info(f'Modification date: {table_infos_new["Modified"]}')
        logger.info(f'Reason for modification: {table_infos_new["ReasonDelivery"]}')
        logger.info(f"Please archive folder {CBS_OPENDATA_EXTERNAL_DATA_DIR / table_id} to proceed")
        return True
    else:
        logger.info(f"Table {table_id} has not been modified")
        logger.info(f'Local {table_infos["Modified"]} == {table_infos_new["Modified"]} Remotely')
        logger.info(f"TableInfos Modified value has NOT changed, we are done")
        return False


if __name__ == "__main__":
    pass
