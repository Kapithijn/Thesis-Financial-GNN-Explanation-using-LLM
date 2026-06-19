# Dataset Directory

This directory only contains placeholders. Do not commit raw datasets, processed graph caches, model checkpoints, or generated outputs.

Expected dataset locations:

```text
elliptic_dataset/
elliptic_temporal_dataset/
dgraphfin_dataset/raw/DGraphFin.zip
tfinance_dataset/raw/
ibm_aml_dataset/raw/HI-Small_Trans.csv.zip
```

The loader code in `Data_File.py` reads paths relative to this `data` directory.
