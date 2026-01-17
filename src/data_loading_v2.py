import os
import tifffile
import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def load_data(
    root1,
    root2,
    metadata,
    split,
    batch_size,
    shuffle,
    num_workers,
    collate_fn,
    pin_memory=True,
    drop_last=True,
    transform=None,
    persistent_workers=False,
    prefetch_factor=None,
):
    dataset = MultiChannelDataset(root1, root2, metadata=metadata, split=split, transform=transform)
    
    # Build dataloader kwargs
    loader_kwargs = {
        'batch_size': batch_size,
        'shuffle': shuffle,
        'num_workers': num_workers,
        'collate_fn': collate_fn,
        'pin_memory': pin_memory,
        'drop_last': drop_last,
    }
    
    # Add optional optimizations
    if num_workers > 0 and persistent_workers:
        loader_kwargs['persistent_workers'] = True
    if num_workers > 0 and prefetch_factor is not None:
        loader_kwargs['prefetch_factor'] = prefetch_factor
    
    data_loader = torch.utils.data.DataLoader(dataset, **loader_kwargs)
    
    return data_loader

class MultiChannelDataset(torch.utils.data.Dataset):
    def __init__(self, root1, root2, metadata, split, transform):
        self.data_path1 = os.path.join(root1, split)
        self.data_path2 = os.path.join(root2, split)
        self.transform = transform

        self.metadata = pd.read_parquet(metadata)
        self.metadata1 = self.metadata[self.metadata['split'] == split]['s1_name'].to_list()
        self.metadata2 = self.metadata[self.metadata['split'] == split]['patch_id'].to_list()

    def __getitem__(self, idx):
        mc_img_c_h_w1 = torch.from_numpy(tifffile.imread(f"{os.path.join(self.data_path1, self.metadata1[idx])}.tif")).float()
        mc_img_c_h_w2 = torch.from_numpy(tifffile.imread(f"{os.path.join(self.data_path2, self.metadata2[idx])}.tif")).float()
        c1 = mc_img_c_h_w1.shape[0]
        c2 = mc_img_c_h_w2.shape[0]

        return torch.split(self.transform(torch.cat((mc_img_c_h_w1, mc_img_c_h_w2), dim=0)), [c1, c2], dim=0)
    
    def __len__(self):
        return len(self.metadata1)
    
    def plot(self, idx):
        s1_img = tifffile.imread(f"{os.path.join(self.data_path1, self.metadata1[idx])}.tif")
        s2_img = tifffile.imread(f"{os.path.join(self.data_path2, self.metadata2[idx])}.tif")
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        vv = s1_img[0]  # VV polarization
        vh = s1_img[1]  # VH polarization
        
        vv_norm = (vv - vv.min()) / (vv.max() - vv.min())
        vh_norm = (vh - vh.min()) / (vh.max() - vh.min())
        
        vv_vh_ratio = np.clip(vv_norm / (vh_norm + 1e-10), 0, 1)
        s1_rgb = np.stack([vv_norm, vh_norm, vv_vh_ratio], axis=-1)
        
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

        
        
# class SADIDataset(Mult)


"""
Expects following directory structure:
dataset
|-- train
|   |-- img_1.tiff
|   |-- img_2.tiff
|   |-- ...
|-- test
    |-- ...

Dataloader returns mc_img_b_n_c_h_w tensors

option 1
    1 img -> transform to n views -> mask each view m times -> create a batch -> mc_img_b_m_n_c_h_w
option 2
    1 img -> transform once -> mask m times -> create a batch -> mc_img_b_m_1_c_h_w

collator returns B_num-masks_P size of masks_pred and masks_enc where P is number of patches to keep.
Usually, num-masks is 1 for context, and more than 1 for target.

forward_target applies the masks to each embedding of image in batch, giving B*num-masks_P_D where D is embedding dimension.


My data is of size B_n_c_h_w as of now. But, it should be B_c_h_w which is then converted to B_P_D instead of B_n_P_D

num_views is same as ncon + npred but without masking and no difference between context and target encoders.
I need to combine collator and transform functions into 1. 
But wait, random.py already implements a masking analog of transforms. 

"""
