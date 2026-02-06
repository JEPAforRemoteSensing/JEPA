import os
import tifffile
import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

class MultiChannelDataset(torch.utils.data.Dataset):
    def __init__(self, root1, root2, metadata, split, transform=None):
        self.data_path1 = os.path.join(root1, split)
        self.data_path2 = os.path.join(root2, split)
        self.transform = transform
        self.split = split
        self.metadata = pd.read_parquet(metadata)
        self.metadata1 = self.metadata[self.metadata['split'] == split]['s1_name'].to_list()
        self.metadata2 = self.metadata[self.metadata['split'] == split]['patch_id'].to_list()

    def __getitem__(self, idx):
        mc_img_c_h_w1 = torch.from_numpy(tifffile.imread(f"{os.path.join(self.data_path1, self.metadata1[idx])}.tif")).float()
        mc_img_c_h_w2 = torch.from_numpy(tifffile.imread(f"{os.path.join(self.data_path2, self.metadata2[idx])}.tif")).float()
        c1 = mc_img_c_h_w1.shape[0]
        c2 = mc_img_c_h_w2.shape[0]

        s1_min = torch.tensor([-40, -30], dtype=torch.float32).view(-1, 1, 1)
        s1_max = torch.tensor([-1.03, 6.72], dtype=torch.float32).view(-1, 1, 1)
        s2_min = torch.tensor([0], dtype=torch.float32).view(-1, 1, 1)
        s2_max = torch.tensor([3500, 4000, 4300, 4300, 5000, 6000, 6600, 5400, 5200, 6300], dtype=torch.float32).view(-1, 1, 1)

        mc_img_c_h_w1 = torch.clamp(mc_img_c_h_w1, min=s1_min, max=s1_max)
        mc_img_c_h_w2 = torch.clamp(mc_img_c_h_w2, min=s2_min, max=s2_max)
        
        mc_img_c_h_w1 = (mc_img_c_h_w1 - s1_min) / (s1_max - s1_min)
        mc_img_c_h_w2 = (mc_img_c_h_w2 - s2_min) / (s2_max - s2_min)

        if self.split == 'train':
            return torch.split(self.transform(torch.cat((mc_img_c_h_w1, mc_img_c_h_w2), dim=0)), [c1, c2], dim=0)
        else:
            return torch.split(self.transform(torch.cat((mc_img_c_h_w1, mc_img_c_h_w2), dim=0)), [c1, c2], dim=0), idx
    
    def __len__(self):
        return len(self.metadata1)
    
    def plot(self, idx):
        s1_img = tifffile.imread(f"{os.path.join(self.data_path1, self.metadata1[idx])}.tif")
        s2_img = tifffile.imread(f"{os.path.join(self.data_path2, self.metadata2[idx])}.tif")

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        vv = s1_img[0]  # VV polarization
        vh = s1_img[1]  # VH polarization
        
        vv_vh_ratio = vv - vh
        s1_rgb = np.stack([vv, vh, vv_vh_ratio], axis=-1)
        
        axes[0].imshow(s1_rgb)
        axes[0].set_title(f'Sentinel-1 (SAR)\n{self.metadata1[idx]}', fontsize=10)
        axes[0].axis('off')
        
        rgb = s2_img[[2, 1, 0], :, :]  # Convert BGR to RGB
        
        rgb_norm = rgb / rgb.max()
        rgb_norm = np.transpose(rgb_norm, (1, 2, 0))  # CHW to HWC
        
        axes[1].imshow(rgb_norm)
        axes[1].set_title(f'Sentinel-2 (RGB)\n{self.metadata2[idx]}', fontsize=10)
        axes[1].axis('off')
        
        plt.tight_layout()
        
        return fig

class LeJEPADataset(MultiChannelDataset):
    def __init__(self, root1, root2, metadata, split, transform_s1, transform_s2, num_views=8):
        super().__init__(root1, root2, metadata, split)
        self.transform_s1 = transform_s1
        self.transform_s2 = transform_s2
        self.V = num_views

    def __getitem__(self, idx):
        mc_img_c_h_w1 = torch.from_numpy(tifffile.imread(f"{os.path.join(self.data_path1, self.metadata1[idx])}.tif")).float()
        mc_img_c_h_w2 = torch.from_numpy(tifffile.imread(f"{os.path.join(self.data_path2, self.metadata2[idx])}.tif")).float()

        s1_min = torch.tensor([-40, -30], dtype=torch.float32).view(-1, 1, 1)
        s1_max = torch.tensor([-1.03, 6.72], dtype=torch.float32).view(-1, 1, 1)
        s2_min = torch.tensor([0], dtype=torch.float32).view(-1, 1, 1)
        s2_max = torch.tensor([3500, 4000, 4300, 4300, 5000, 6000, 6600, 5400, 5200, 6300], dtype=torch.float32).view(-1, 1, 1)

        mc_img_c_h_w1 = torch.clamp(mc_img_c_h_w1, min=s1_min, max=s1_max)
        mc_img_c_h_w2 = torch.clamp(mc_img_c_h_w2, min=s2_min, max=s2_max)
        
        mc_img_c_h_w1 = (mc_img_c_h_w1 - s1_min) / (s1_max - s1_min)
        mc_img_c_h_w2 = (mc_img_c_h_w2 - s2_min) / (s2_max - s2_min)

        labels = self.metadata[self.metadata['s1_name'] == self.metadata1[idx]].iloc[0]["one_hot_labels"]

        views_s1 = torch.stack([self.transform_s1(mc_img_c_h_w1) for _ in range(self.V)])
        views_s2 = torch.stack([self.transform_s2(mc_img_c_h_w2) for _ in range(self.V)])
        return views_s1, views_s2, torch.tensor(labels, dtype=float)
    
class XJEPADataset(MultiChannelDataset):
    def __init__(self, root1, root2, metadata, split, transform_s1, transform_s2):
        super().__init__(root1, root2, metadata, split)
        self.transform_s1 = transform_s1
        self.transform_s2 = transform_s2

    def __getitem__(self, idx):
        mc_img_c_h_w1 = torch.from_numpy(tifffile.imread(f"{os.path.join(self.data_path1, self.metadata1[idx])}.tif")).float()
        mc_img_c_h_w2 = torch.from_numpy(tifffile.imread(f"{os.path.join(self.data_path2, self.metadata2[idx])}.tif")).float()

        s1_min = torch.tensor([-40, -30], dtype=torch.float32).view(-1, 1, 1)
        s1_max = torch.tensor([-1.03, 6.72], dtype=torch.float32).view(-1, 1, 1)
        s2_min = torch.tensor([0], dtype=torch.float32).view(-1, 1, 1)
        s2_max = torch.tensor([3500, 4000, 4300, 4300, 5000, 6000, 6600, 5400, 5200, 6300], dtype=torch.float32).view(-1, 1, 1)

        mc_img_c_h_w1 = torch.clamp(mc_img_c_h_w1, min=s1_min, max=s1_max)
        mc_img_c_h_w2 = torch.clamp(mc_img_c_h_w2, min=s2_min, max=s2_max)
        
        mc_img_c_h_w1 = (mc_img_c_h_w1 - s1_min) / (s1_max - s1_min)
        mc_img_c_h_w2 = (mc_img_c_h_w2 - s2_min) / (s2_max - s2_min)

        labels = self.metadata[self.metadata['s1_name'] == self.metadata1[idx]].iloc[0]["one_hot_labels"]
        view_s1 = self.transform_s1(mc_img_c_h_w1)
        view_s2 = self.transform_s2(mc_img_c_h_w2) 
        return view_s1, view_s2, torch.tensor(labels, dtype=float)
