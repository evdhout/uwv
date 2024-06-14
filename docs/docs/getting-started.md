Getting started
===============

## Initial installation of data 

For an initial download of the dataset run

```bash
python uwv/data/make_dataset.py
```

This wil download the CBS tables specified in uwv/config.py into the data/external directory
and process them into parquet files in the data/processed directory. 

## Getting a single CBS OpenData table

If you know the ID of a CBS OpenData table, you can get it using

```bash
python uwv/data/cbs/get_cbs_opendata_table.py TABLE_ID
```

## Updating data

Both `get_cbs_opendata_table.py` checks whether the table has been modified since
it was last downloaded and will only download if the modification dates do not match. 
This also applies to `make_dataset.py` as it only calls `get_cbs_opendata_table` iteratively.

Both accept an `overwrite` flag which defaults to false. Setting it will overwrite existing files. 
