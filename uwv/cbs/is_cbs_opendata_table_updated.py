import json
import typer
from loguru import logger

from uwv.config import CBS_OPENDATA_API_URL, CBS_OPENDATA_EXTERNAL_DATA_DIR
from uwv.cbs.cbs_helper_functions import get_dict_from_json_response


def is_cbs_opendata_table_updated(table_id: str) -> bool:
    """
    Check whether there is a new version of the CBS OpenData table online

    :param table_id: cbs id of the table
    :return:
    """

    table_infos_file = CBS_OPENDATA_EXTERNAL_DATA_DIR / table_id / f"TableInfos.json"
    try:
        with open(table_infos_file, "rb") as f:
            table_infos = json.load(f)[0]
    except FileNotFoundError:
        logger.error(f"TableInfos file not found: {table_infos_file}")
        raise typer.Abort(1)
    except OSError as e:
        raise e

    table_infos_remote = get_dict_from_json_response(
        f"{CBS_OPENDATA_API_URL}/{table_id}/TableInfos"
    )
    table_infos_modified = table_infos["Modified"]
    table_infos_remote_modified = table_infos_remote["Modified"]

    logger.trace(
        f"Checking if table {table_id} has changed. Local mod date {table_infos_modified}"
    )
    if table_infos_modified != table_infos_remote_modified:
        logger.trace(f"Table {table_id} has changed")
        logger.trace(f"Modification date: {table_infos_modified}")
        logger.trace(f'Reason for modification: {table_infos_remote["ReasonDelivery"]}')
        logger.trace(
            f"Please archive folder {CBS_OPENDATA_EXTERNAL_DATA_DIR / table_id} to proceed"
        )
        return True
    else:
        logger.trace(f"Table {table_id} has not been modified")
        logger.trace(f"Local {table_infos_modified} == {table_infos_remote_modified} Remotely")
        return False


if __name__ == "__main__":
    pass
