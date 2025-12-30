from src.data_loading import load_data, MultiChannelDataset
from src.masks import MultiBlockMaskCollator
from src.transforms import make_transforms

def main():
    
    train_ds = MultiChannelDataset(
        root='/Users/narendraaironi/projects/biplab/BEN_14k/BigEarthNet-S2',
        split='train',
        transform=make_transforms(num_channels=10),
    )

    print(f"Dataset length: {len(train_ds)}")
    sample = train_ds[0]
    print(train_ds.metadata[0])
    print(f"Sample shape: {sample.shape}")

if __name__ == "__main__":
    main()
