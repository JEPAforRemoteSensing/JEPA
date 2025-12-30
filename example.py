import os
import subprocess
import zipfile
from pathlib import Path
from src.data_loading import load_data, MultiChannelDataset
from src.masks import MultiBlockMaskCollator
from src.transforms import make_transforms_rgb, make_transforms

def download_dataset_from_kaggle(dataset_name, download_path):
    """Download dataset from Kaggle if it doesn't exist."""
    print(f"Dataset not found. Downloading from Kaggle: {dataset_name}")
    
    # Create download directory if it doesn't exist
    os.makedirs(download_path, exist_ok=True)
    
    # Download dataset using kaggle CLI
    try:
        subprocess.run(
            ["kaggle", "datasets", "download", "-d", dataset_name, "-p", download_path],
            check=True
        )
        print(f"Dataset downloaded to {download_path}")
        
        # Unzip the dataset
        zip_files = list(Path(download_path).glob("*.zip"))
        for zip_file in zip_files:
            print(f"Extracting {zip_file}...")
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall(download_path)
            # Remove zip file after extraction
            zip_file.unlink()
            print(f"Extraction complete!")
            
    except subprocess.CalledProcessError as e:
        print(f"Error downloading dataset: {e}")
        raise
    except Exception as e:
        print(f"Error processing dataset: {e}")
        raise

def main():
    # Define paths
    dataset_root = os.path.join(os.getcwd(), 'data', 'BEN_14k')
    download_path = os.path.join(os.getcwd(), 'data')
    kaggle_dataset = "narendraaironi/bigearthnet-14k"
    
    # Check if dataset exists, if not download it
    if not os.path.exists(dataset_root):
        download_dataset_from_kaggle(kaggle_dataset, download_path)
    else:
        print(f"Dataset already exists at {dataset_root}")
    
    bigearthnet_s2_root = os.path.join(dataset_root, 'BigEarthNet-S2')
    train_ds = MultiChannelDataset(
        root=bigearthnet_s2_root,
        split='train',
        transform=make_transforms_rgb(num_channels=3),
    )

    print(f"Dataset length: {len(train_ds)}")
    sample = train_ds[0]
    print(train_ds.metadata[0])
    print(f"Sample shape: {sample.shape}")

    data_loader = load_data(
        root=bigearthnet_s2_root,
        split='train',
        batch_size=4,
        shuffle=True,
        num_workers=2,
        collate_fn=MultiBlockMaskCollator()
    )

    

if __name__ == "__main__":
    main()
