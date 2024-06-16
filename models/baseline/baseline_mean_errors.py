import math

import numpy as np
import pandas as pd
import typer
from loguru import logger

from uwv.config import CBS80072NED, PROCESSED_DATA_DIR

PREDICTION_NAME = {'le': 'linear extrapolation', 'ra': 'rolling average'}
PREFIXES = ['le', 'ra']

app = typer.Typer(no_args_is_help=True)

baseline: pd.DataFrame


def calculate_mean_absolute_error() -> {str: pd.DataFrame}:
    dfs: {str: pd.DataFrame} = {}
    for prefix in PREFIXES:
        logger.info(f'Calculating means of absolute and squared errors for {PREDICTION_NAME[prefix]}')
        df = baseline.groupby(['sbi', 'sbi_title', 'period_quarter_number'],
                              observed=True)[[f'{prefix}_absolute_error', f'{prefix}_squared_error']].mean()
        df[f'{prefix}_squared_error'] = df[f'{prefix}_squared_error'].apply(lambda x: math.sqrt(x))
        df = df.rename(columns={f'{prefix}_squared_error': f'root_mean_squared_error_{prefix}',
                                f'{prefix}_absolute_error': f'mean_absolute_error_{prefix}'})
        dfs[prefix] = df

    return dfs


@app.command()
def main(force: bool = False):
    global baseline
    try:
        baseline = pd.read_parquet(PROCESSED_DATA_DIR / f'baseline-{CBS80072NED}.parquet')
    except FileNotFoundError:
        logger.error(f"File {PROCESSED_DATA_DIR}/baseline-{CBS80072NED}.parquet not found.")
        raise typer.Abort()

    parquet_filename = PROCESSED_DATA_DIR / f'baseline-mean-errors-{CBS80072NED}.parquet'
    csv_filename = PROCESSED_DATA_DIR / f'baseline-mean-errors-{CBS80072NED}.csv'

    if parquet_filename.exists() and not force:
        logger.info(f'File {parquet_filename} exists and {force=}.')
        raise typer.Abort()

    logger.info(f'Calculating mean absolute errors and root mean squared errors')
    dfs = calculate_mean_absolute_error()

    merged_df = dfs['le'].join(dfs['ra'], how='inner')

    merged_df.to_parquet(parquet_filename)
    logger.info(f'Saved {parquet_filename}')
    merged_df.to_csv(csv_filename)
    logger.info(f'Saved {csv_filename}')


if __name__ == '__main__':
    app()
