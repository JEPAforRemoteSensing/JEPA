import rasterio
import matplotlib.pyplot as plt
from rasterio.plot import show

src = rasterio.open("BEN_full/BigEarthNet-S2/S2A_MSIL2A_20170803T094031_N9999_R036_T34TCR_65_19.tif")
print(src.count)
bgr_bands = src.read()

rgb_bands = bgr_bands[[2, 1, 0], :, :]

# print(src.width, src.height)
show(rgb_bands, adjust='linear')
