Getting started
===============

## Initial download of data 

For an initial download of the dataset run

```bash
python uwv/dataset.py [--overwrite]
```

This wil download the CBS tables specified in uwv/config.py into the data/external directory
and process them into parquet files in the data/processed directory. 

### Getting a single CBS OpenData table

If you know the ID of a CBS OpenData table, you can get it using

```bash
python uwv/data/cbs/get_cbs_opendata_table.py TABLE_ID
```

## Initial generation of datafile with all features

```bash
python uwv/features.py [--overwrite]
```

This wil generate the datafile `data/processed/uwv_data.parquet`. It will add the selected features
to the main tabel 80072ned which is used to train the models.

## Updating data

`get_cbs_opendata_table.py` checks whether the table has been modified since
it was last downloaded and will only download if the modification dates do not match. 
This also applies to `dataset.py` as it only calls `get_cbs_opendata_table` iteratively.

`features.py` also accept an overwrite flag, to prevent accidentally overwriting your datafile.

Both accept an `overwrite` flag which defaults to false. Setting it will overwrite existing files. 
