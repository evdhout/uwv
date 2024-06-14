import json
from loguru import logger

from uwv.config import CBS_OPENDATA_BASE_URL, CBS_OPENDATA_EXTERNAL_DATA_DIR
from uwv.data.cbs.cbs_helper_functions import (
    get_dict_from_json_response,
    get_dataframe_from_json_response,
)


def download_cbs_opendata_table(table_id: str):
    cbs_table_path = CBS_OPENDATA_EXTERNAL_DATA_DIR / table_id
    logger.info(f"Retrieving CBS OpenData table {table_id}")
    table_base = get_dataframe_from_json_response(f"{CBS_OPENDATA_BASE_URL}/{table_id}")
    table_base.to_csv(f"{cbs_table_path}/{table_id}.csv", index=False)

    for _, row in table_base.iterrows():
        logger.info("Getting %s for %s" % (row["name"], row["url"]))

        if row["name"] == "TableInfos":
            table = get_dict_from_json_response(row["url"])
            with open(cbs_table_path / f'{table_id}_{row["name"]}.json', "w") as outfile:
                json.dump(table, outfile)
        else:
            table = get_dataframe_from_json_response(row["url"])
            table.to_csv(f'{cbs_table_path}/{table_id}_{row["name"]}.csv', index=False)


if __name__ == "__main__":
    pass
