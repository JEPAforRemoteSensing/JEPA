import os
import tifffile
import torch

class MultiChannelDataset(torch.utils.data.IterableDataset):
    def __init__(self, root, split, num_views=1, transform=None):
        super(MultiChannelDataset, self).__init__()

        self.data_path = os.path.join(root, split)
        self.num_views = num_views
        self.transform = transform

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        worker_id = 0
        num_workers = 1
        # Supports multi-processing data loading
        if worker_info is not None:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
        
        # For fast subdirectory scanning using an iterator
        with os.scandir(self.data_path) as mc_imgs:
            for idx, mc_img in enumerate(mc_imgs):

                if idx % num_workers != worker_id:
                    continue

                # Yields a num_views x channels x height x width tensor
                mc_img_c_h_w = torch.from_numpy(tifffile.imread(mc_img.path))

                views = torch.stack(
                    [self.transform(mc_img_c_h_w) if self.transform else mc_img_c_h_w for _ in range(self.num_views)],
                    dim=0)
                
                yield views


"""
Expects following directory structure:
dataset
|-- train
|   |-- img_1.tiff
|   |-- img_2.tiff
|   |-- ...
|-- test
    |-- ...
"""
