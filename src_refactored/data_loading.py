import os
import tifffile
import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)

# Normalization constants (cached globally to avoid recreation)
S1_MIN = torch.tensor([-40, -30], dtype=torch.float32).view(-1, 1, 1)
S1_MAX = torch.tensor([-1.03, 6.72], dtype=torch.float32).view(-1, 1, 1)
S2_MIN = torch.tensor([0], dtype=torch.float32).view(-1, 1, 1)
S2_MAX = torch.tensor([3500, 4000, 4300, 4300, 5000, 6000, 6600, 5400, 5200, 6300], dtype=torch.float32).view(-1, 1, 1)

class MultiChannelDataset(torch.utils.data.Dataset):
    def __init__(self, root1, root2, metadata, split, transform=None, cache_in_memory=False):
        self.data_path1 = os.path.join(root1, split)
        self.data_path2 = os.path.join(root2, split)
        self.transform = transform
        self.split = split
        self.metadata = pd.read_parquet(metadata)
        self.metadata1 = self.metadata[self.metadata['split'] == split]['s1_name'].to_list()
        self.metadata2 = self.metadata[self.metadata['split'] == split]['patch_id'].to_list()
        
        # Pre-compute label lookup dict (O(1) instead of O(n) pandas lookup)
        self.label_lookup = {}
        for _, row in self.metadata[self.metadata['split'] == split].iterrows():
            self.label_lookup[row['s1_name']] = row.get("one_hot_labels", None)
        
        self.cache_in_memory = cache_in_memory
        self.data_cache = {}
        if cache_in_memory:
            logger.info(f"Caching {len(self)} samples in memory ({self.split} split)...")
            for idx in range(len(self)):
                self._load_and_cache(idx)
            logger.info(f"Cached {len(self.data_cache)} samples in memory")

    def _load_and_cache(self, idx):
        """Load and cache a single sample."""
        if idx in self.data_cache:
            return
        mc_img_c_h_w1 = torch.from_numpy(tifffile.imread(f"{os.path.join(self.data_path1, self.metadata1[idx])}.tif")).float()
        mc_img_c_h_w2 = torch.from_numpy(tifffile.imread(f"{os.path.join(self.data_path2, self.metadata2[idx])}.tif")).float()
        
        # Normalize and clamp
        mc_img_c_h_w1 = torch.clamp(mc_img_c_h_w1, min=S1_MIN, max=S1_MAX)
        mc_img_c_h_w2 = torch.clamp(mc_img_c_h_w2, min=S2_MIN, max=S2_MAX)
        mc_img_c_h_w1 = (mc_img_c_h_w1 - S1_MIN) / (S1_MAX - S1_MIN)
        mc_img_c_h_w2 = (mc_img_c_h_w2 - S2_MIN) / (S2_MAX - S2_MIN)
        
        self.data_cache[idx] = (mc_img_c_h_w1, mc_img_c_h_w2, mc_img_c_h_w1.shape[0], mc_img_c_h_w2.shape[0])

    def __getitem__(self, idx):
        if self.cache_in_memory:
            mc_img_c_h_w1, mc_img_c_h_w2, c1, c2 = self.data_cache[idx]
        else:
            # Load from disk with global normalization constants
            mc_img_c_h_w1 = torch.from_numpy(tifffile.imread(f"{os.path.join(self.data_path1, self.metadata1[idx])}.tif")).float()
            mc_img_c_h_w2 = torch.from_numpy(tifffile.imread(f"{os.path.join(self.data_path2, self.metadata2[idx])}.tif")).float()
            c1 = mc_img_c_h_w1.shape[0]
            c2 = mc_img_c_h_w2.shape[0]
            
            mc_img_c_h_w1 = torch.clamp(mc_img_c_h_w1, min=S1_MIN, max=S1_MAX)
            mc_img_c_h_w2 = torch.clamp(mc_img_c_h_w2, min=S2_MIN, max=S2_MAX)
            mc_img_c_h_w1 = (mc_img_c_h_w1 - S1_MIN) / (S1_MAX - S1_MIN)
            mc_img_c_h_w2 = (mc_img_c_h_w2 - S2_MIN) / (S2_MAX - S2_MIN)

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
    def __init__(self, root1, root2, metadata, split, transform_s1, transform_s2, num_views=4, cache_in_memory=False):
        super().__init__(root1, root2, metadata, split, cache_in_memory=cache_in_memory)
        self.transform_s1 = transform_s1
        self.transform_s2 = transform_s2
        self.V = num_views

    def __getitem__(self, idx):
        if self.cache_in_memory:
            mc_img_c_h_w1, mc_img_c_h_w2, _, _ = self.data_cache[idx]
        else:
            # Load from disk with global constants
            mc_img_c_h_w1 = torch.from_numpy(tifffile.imread(f"{os.path.join(self.data_path1, self.metadata1[idx])}.tif")).float()
            mc_img_c_h_w2 = torch.from_numpy(tifffile.imread(f"{os.path.join(self.data_path2, self.metadata2[idx])}.tif")).float()
            
            mc_img_c_h_w1 = torch.clamp(mc_img_c_h_w1, min=S1_MIN, max=S1_MAX)
            mc_img_c_h_w2 = torch.clamp(mc_img_c_h_w2, min=S2_MIN, max=S2_MAX)
            mc_img_c_h_w1 = (mc_img_c_h_w1 - S1_MIN) / (S1_MAX - S1_MIN)
            mc_img_c_h_w2 = (mc_img_c_h_w2 - S2_MIN) / (S2_MAX - S2_MIN)
        
        # Fast dict lookup instead of metadata filtering
        labels = self.label_lookup.get(self.metadata1[idx], None)
        
        # Reuse same input image for multiple views (faster than applying transform multiple times)
        views_s1 = torch.stack([self.transform_s1(mc_img_c_h_w1) for _ in range(self.V)])
        views_s2 = torch.stack([self.transform_s2(mc_img_c_h_w2) for _ in range(self.V)])
        return views_s1, views_s2, torch.tensor(labels, dtype=torch.float32)

