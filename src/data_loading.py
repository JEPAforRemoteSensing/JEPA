import os
import tifffile
import torch

def load_data(
    root,
    split,
    batch_size,
    shuffle,
    num_workers,
    collate_fn,
    pin_memory=True,
    drop_last=True,
    num_views=1,
    transform=None
):
    dataset = MultiChannelDataset(root, split, num_views=num_views, transform=transform)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        drop_last=drop_last)
    
    return data_loader

class MultiChannelDataset(torch.utils.data.Dataset):
    def __init__(self, root, split, metadata= None, num_views=1, transform=None):
        self.data_path = os.path.join(root, split)
        self.num_views = num_views
        self.transform = transform

        # If metadata is provided, use it for lazy-loading, else build the metadata from directory
        self.metadata = metadata
        if self.metadata is None:
            self.metadata = []
            for mc_img in os.scandir(self.data_path):
                self.metadata.append(mc_img.name)

    def __getitem__(self, idx):
        mc_img_c_h_w = tifffile.imread(os.path.join(self.data_path, self.metadata[idx]))
        mc_img_n_c_h_w = torch.stack(
            [self.transform(mc_img_c_h_w, self.num_views) if self.transform else mc_img_c_h_w for _ in range(self.num_views)],
            dim=0)
        return mc_img_n_c_h_w
    
    def __len__(self):
        return len(self.metadata)


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
"""
