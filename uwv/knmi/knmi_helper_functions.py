from pathlib import Path

from uwv.config import (
    KNMI_AVG_TEMP,
    KNMI_EXTERNAL_DATA_DIR,
    KNMI_INTERIM_DATA_DIR,
    KNMI_PROCESSED_DATA_DIR,
)


def make_knmi_directories():
    """
    Create directories for KNMI data organization.

    This function ensures that the directories required for KNMI external, interim,
    and processed data are created. If the directories already exist, they will
    not be created again.

    :return: None
    """
    KNMI_EXTERNAL_DATA_DIR.mkdir(parents=True, exist_ok=True)
    KNMI_INTERIM_DATA_DIR.mkdir(parents=True, exist_ok=True)
    KNMI_PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)


def get_knmi_filename(process_level: str = "processed") -> Path or None:
    """
    Generates a file path based on the process level of KNMI data. The function
    returns a path to the average temperature file in the specified directory based
    on process level: 'processed', 'interim', or 'external'. If the process level
    does not match any of these, it will return None.

    :param process_level: The level of processing for the KNMI data, which can be
                          'processed', 'interim', or 'external'. If an unknown
                          level is provided, None is returned.
    :type process_level: str
    :return: A Path object pointing to the relevant file, or None if the level
             is not recognized.
    :rtype: Path or None
    """
    match process_level:
        case "processed":
            return KNMI_PROCESSED_DATA_DIR / f"{KNMI_AVG_TEMP}.parquet"
        case "interim":
            return KNMI_INTERIM_DATA_DIR / f"{KNMI_AVG_TEMP}.csv"
        case "external":
            return KNMI_EXTERNAL_DATA_DIR / f"{KNMI_AVG_TEMP}.txt"
        case _:
            return None
