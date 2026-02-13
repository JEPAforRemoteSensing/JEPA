import os
from lightning.data import optimize
import torch
import tifffile
import pandas as pd

data_path1 = "data/BEN_14k/BigEarthNet-S1/validation"
data_path2 = "data/BEN_14k/BigEarthNet-S2/validation"
metadata = pd.read_parquet('data/BEN_14k/serbia_metadata.parquet')
metadata1 = metadata[metadata['split'] == 'validation']['s1_name'].to_list()
metadata2 = metadata[metadata['split'] == 'validation']['patch_id'].to_list()

s1_min = torch.tensor([-40, -30], dtype=torch.float32).view(-1, 1, 1)
s1_max = torch.tensor([-1.03, 6.72], dtype=torch.float32).view(-1, 1, 1)
s2_min = torch.tensor([0], dtype=torch.float32).view(-1, 1, 1)
s2_max = torch.tensor([3500, 4000, 4300, 4300, 5000, 6000, 6600, 5400, 5200, 6300], dtype=torch.float32).view(-1, 1, 1)

def compress(idx):
    mc_img_c_h_w1 = torch.from_numpy(tifffile.imread(f"{os.path.join(data_path1, metadata1[idx])}.tif")).float()
    mc_img_c_h_w2 = torch.from_numpy(tifffile.imread(f"{os.path.join(data_path2, metadata2[idx])}.tif")).float()

    mc_img_c_h_w1 = torch.clamp(mc_img_c_h_w1, min=s1_min, max=s1_max)
    mc_img_c_h_w2 = torch.clamp(mc_img_c_h_w2, min=s2_min, max=s2_max)
        
    mc_img_c_h_w1 = (mc_img_c_h_w1 - s1_min) / (s1_max - s1_min)
    mc_img_c_h_w2 = (mc_img_c_h_w2 - s2_min) / (s2_max - s2_min)

    labels = metadata[metadata['s1_name'] == metadata1[idx]].iloc[0]["one_hot_labels"]

    return mc_img_c_h_w1, mc_img_c_h_w2, torch.tensor(labels, dtype=torch.float32)


if __name__ == "__main__":

    optimize(
        fn=compress,
        inputs=list(range(len(metadata1))),
        output_dir="data/opt_BEN_14k/validation",
        num_workers=4,
        chunk_bytes="128MB",
        compression='zstd'
    )