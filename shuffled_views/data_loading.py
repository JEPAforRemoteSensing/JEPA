import os
import tifffile
import torch
from rasterio.plot import show
import rasterio

def load_data(
    root,
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
    dataset = MultiChannelDataset(root, split, transform=transform)
    
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
    def __init__(self, root, split, num_views = 6, metadata= None, transform=None, shuffle = None):
        self.data_path = os.path.join(root, split)
        self.transform = transform
        self.num_views = num_views
        self.shuffle = shuffle
        # If metadata is provided, use it for lazy-loading, else build the metadata from directory
        self.metadata = metadata
        if self.metadata is None:
            self.metadata = []
            for mc_img in os.scandir(self.data_path):
                self.metadata.append(mc_img.name)

    def __getitem__(self, idx):
        mc_img_c_h_w = torch.from_numpy(tifffile.imread(os.path.join(self.data_path, self.metadata[idx]))).float()
        for _ in range(self.num_views):
            mc_img_c_h_w = self.transform(mc_img_c_h_w) if self.transform else mc_img_c_h_w
            
    
    def __len__(self):
        return len(self.metadata)
    
    def plot(self, idx):
        sample = rasterio.open(os.path.join(self.data_path, self.metadata[idx]))
        img = sample.read()[[2, 1, 0], :, :]
        show(img, adjust='linear')
        
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
