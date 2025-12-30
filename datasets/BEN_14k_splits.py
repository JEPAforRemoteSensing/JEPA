import pandas as pd
import os
import rasterio
from rasterio.plot import show
import matplotlib.pyplot as plt
import shutil

ben_serbia = pd.read_parquet('BEN_full/metadata.parquet')

ben_14k_s1_path = "BEN_full/BigEarthNet-S1"
ben_14k_s2_path = "BEN_full/BigEarthNet-S2"

def combine_tiffs(band_files, output_path):
    with rasterio.open(band_files[0]) as src:
        profile = src.profile
        profile.update(count=len(band_files))
    with rasterio.open(output_path, 'w', **profile) as dst:
        for i, f in enumerate(band_files, 1):
            dst.write(rasterio.open(f).read(1), i)

# src = rasterio.open("multispectral.tif")
# show(src, cmap='terrain')

def make_ms_tiff():
    c = 0
    for dir in os.scandir(ben_14k_s2_path):
        if not dir.is_dir():
            continue

        for subdir in os.scandir(dir):
            if not subdir.is_dir():
                continue

            band_files = []
            for f in os.scandir(subdir):
                if f.name.endswith('B01.tif') or f.name.endswith('B09.tif'):
                    continue
                band_files.append(f.path)

            band_files.sort()
            
            combine_tiffs(band_files, f"{ben_14k_s2_path}/{subdir.name}.tif")
            c += 1

            if c % 100 == 0:
                print(c / 138.63)
            shutil.rmtree(subdir.path)

make_ms_tiff()
