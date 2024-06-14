import requests
import pandas as pd

from uwv.config import CBS_OPENDATA_EXTERNAL_DATA_DIR


def _get_response(url):
    response = requests.get(url)
    if response.ok:
        return response
    else:
        raise Exception(f"Url {url} returned {response.status_code}")


def get_dataframe_from_json_response(url: str) -> pd.DataFrame:
    return pd.DataFrame([value for value in _get_response(url).json()["value"]])


def get_dict_from_json_response(url: str) -> dict:
    return _get_response(url).json()["value"][0]


def get_dataframe_value(
    df: pd.DataFrame, select_column: str, select_value: str, get_column: str
) -> str:
    return df[df[select_column] == select_value][get_column].values[0]


def get_url_value(df: pd.DataFrame, name: str) -> str:
    return get_dataframe_value(df=df, select_column="name", select_value=name, get_column="url")


def is_cbs_data_path(table_id: str):
    table_path = CBS_OPENDATA_EXTERNAL_DATA_DIR / table_id
    return table_path.is_dir()


def create_cbs_data_path_if_not_exists(table_id: str):
    if not is_cbs_data_path(table_id):
        table_path = CBS_OPENDATA_EXTERNAL_DATA_DIR / table_id
        table_path.mkdir(exist_ok=True, parents=True)


if __name__ == "__main__":
    pass
