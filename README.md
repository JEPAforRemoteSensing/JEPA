# JEPA

## Datasets

### BEN-14K dataset

[BEN-14K](https://www.kaggle.com/datasets/narendraaironi/bigearthnet-14k) is a subset of BigEarthNet-v2 with 13,683 images with a ~52-24-24 train-test-validation split.

It has the following directory structure:
```
dataset
|-- BigEarthNet-S1
    |-- train
        |-- <Sentinel-ID>_MSIL2A_<YYYYMMDD>T<HHMMSS>_N9999_<Rooo>_<Txxxxxx>_<H-Order>_<V-Order>.tif
        |-- ...
    |-- validation
        |-- ...
    |-- test
        |-- ...
```
#### Construction
* All images with `metadata.parquet`'s country label as Serbia and taken during the summer (on or before 31st August, 2017) were taken
* B01 and B09 bands of S2 were removed
* Directory structure was rearranged using split labels from `metadata.parquet`
* Original BENv2 dataset had 1 geoTIFF file per band. In BEN-14K, all bands are combined into a single geoTIFF file.

---
