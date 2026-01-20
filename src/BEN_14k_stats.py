"""
Calculate BEN_14K statistics (mean, std, min, max) for each channel
of both S1 (Sentinel-1) and S2 (Sentinel-2) modalities.
"""

import os
import sys
import torch
import numpy as np
from tqdm import tqdm
from torch.nn import Identity

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_loading_v2 import load_data


def identity_collate_fn(batch):
    """Simple collate function that stacks S1 and S2 tensors separately."""
    s1_tensors = [item[0] for item in batch]
    s2_tensors = [item[1] for item in batch]
    return torch.stack(s1_tensors), torch.stack(s2_tensors)


class IdentityTransform:
    """Identity transform that returns the input unchanged."""
    def __call__(self, x):
        return x


def calculate_stats(dataloader, s1_channels=2, s2_channels=10):
    """
    Calculate running statistics (mean, std, min, max) for S1 and S2 modalities.
    
    Uses Welford's online algorithm for numerically stable computation.
    
    Args:
        dataloader: DataLoader yielding (s1_tensor, s2_tensor) tuples
        s1_channels: Number of channels in S1 (default: 2 for VV, VH)
        s2_channels: Number of channels in S2 (default: 10)
    
    Returns:
        Dictionary containing stats for both modalities
    """
    # Initialize running statistics for S1
    s1_count = 0
    s1_mean = torch.zeros(s1_channels, dtype=torch.float64)
    s1_M2 = torch.zeros(s1_channels, dtype=torch.float64)  # For variance calculation
    s1_min = torch.full((s1_channels,), float('inf'), dtype=torch.float64)
    s1_max = torch.full((s1_channels,), float('-inf'), dtype=torch.float64)
    
    # Initialize running statistics for S2
    s2_count = 0
    s2_mean = torch.zeros(s2_channels, dtype=torch.float64)
    s2_M2 = torch.zeros(s2_channels, dtype=torch.float64)
    s2_min = torch.full((s2_channels,), float('inf'), dtype=torch.float64)
    s2_max = torch.full((s2_channels,), float('-inf'), dtype=torch.float64)
    
    print("Calculating statistics...")
    for s1_batch, s2_batch in tqdm(dataloader, desc="Processing batches"):
        # S1 batch shape: (B, C, H, W) -> need per-channel stats
        # Reshape to (C, B*H*W) for per-channel computation
        s1_batch = s1_batch.double()
        s2_batch = s2_batch.double()
        
        # Process S1
        B, C1, H, W = s1_batch.shape
        s1_pixels = s1_batch.permute(1, 0, 2, 3).reshape(C1, -1)  # (C, N)
        
        # Update min/max
        batch_s1_min = s1_pixels.min(dim=1).values
        batch_s1_max = s1_pixels.max(dim=1).values
        s1_min = torch.minimum(s1_min, batch_s1_min)
        s1_max = torch.maximum(s1_max, batch_s1_max)
        
        # Welford's online algorithm for mean and variance
        batch_size = s1_pixels.shape[1]
        batch_mean = s1_pixels.mean(dim=1)
        batch_var = s1_pixels.var(dim=1, unbiased=False)
        
        delta = batch_mean - s1_mean
        new_count = s1_count + batch_size
        s1_mean = s1_mean + delta * batch_size / new_count
        s1_M2 = s1_M2 + batch_var * batch_size + delta ** 2 * s1_count * batch_size / new_count
        s1_count = new_count
        
        # Process S2
        B, C2, H, W = s2_batch.shape
        s2_pixels = s2_batch.permute(1, 0, 2, 3).reshape(C2, -1)  # (C, N)
        
        # Update min/max
        batch_s2_min = s2_pixels.min(dim=1).values
        batch_s2_max = s2_pixels.max(dim=1).values
        s2_min = torch.minimum(s2_min, batch_s2_min)
        s2_max = torch.maximum(s2_max, batch_s2_max)
        
        # Welford's online algorithm
        batch_size = s2_pixels.shape[1]
        batch_mean = s2_pixels.mean(dim=1)
        batch_var = s2_pixels.var(dim=1, unbiased=False)
        
        delta = batch_mean - s2_mean
        new_count = s2_count + batch_size
        s2_mean = s2_mean + delta * batch_size / new_count
        s2_M2 = s2_M2 + batch_var * batch_size + delta ** 2 * s2_count * batch_size / new_count
        s2_count = new_count
    
    # Calculate final standard deviation
    s1_std = torch.sqrt(s1_M2 / s1_count)
    s2_std = torch.sqrt(s2_M2 / s2_count)
    
    return {
        's1': {
            'mean': s1_mean.numpy(),
            'std': s1_std.numpy(),
            'min': s1_min.numpy(),
            'max': s1_max.numpy(),
            'num_pixels': s1_count,
            'channel_names': ['VV', 'VH']
        },
        's2': {
            'mean': s2_mean.numpy(),
            'std': s2_std.numpy(),
            'min': s2_min.numpy(),
            'max': s2_max.numpy(),
            'num_pixels': s2_count,
            'channel_names': ['B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B11', 'B12']
        }
    }


def print_stats(stats):
    """Pretty print the computed statistics."""
    print("\n" + "=" * 80)
    print("BEN_14K Dataset Statistics")
    print("=" * 80)
    
    for modality in ['s1', 's2']:
        mod_stats = stats[modality]
        mod_name = "Sentinel-1 (SAR)" if modality == 's1' else "Sentinel-2 (MSI)"
        
        print(f"\n{mod_name}")
        print("-" * 60)
        print(f"{'Channel':<10} {'Mean':>15} {'Std':>15} {'Min':>15} {'Max':>15}")
        print("-" * 60)
        
        for i, ch_name in enumerate(mod_stats['channel_names']):
            print(f"{ch_name:<10} {mod_stats['mean'][i]:>15.4f} {mod_stats['std'][i]:>15.4f} "
                  f"{mod_stats['min'][i]:>15.4f} {mod_stats['max'][i]:>15.4f}")
        
        print(f"\nTotal pixels processed: {mod_stats['num_pixels']:,}")
    
    print("\n" + "=" * 80)
    
    # Print as lists for easy copy-paste into code
    print("\nFor use in normalization transforms:")
    print("-" * 60)
    
    print("\nS1 (Sentinel-1):")
    print(f"  mean = {list(stats['s1']['mean'])}")
    print(f"  std  = {list(stats['s1']['std'])}")
    
    print("\nS2 (Sentinel-2):")
    print(f"  mean = {list(stats['s2']['mean'])}")
    print(f"  std  = {list(stats['s2']['std'])}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Calculate BEN_14K dataset statistics')
    parser.add_argument('--s1_root', type=str, default='data/BEN_14k/BigEarthNet-S1',
                        help='Root directory for Sentinel-1 data')
    parser.add_argument('--s2_root', type=str, default='data/BEN_14k/BigEarthNet-S2',
                        help='Root directory for Sentinel-2 data')
    parser.add_argument('--metadata', type=str, default='data/BEN_14k/serbia_metadata.parquet',
                        help='Path to metadata parquet file')
    parser.add_argument('--split', type=str, default='train',
                        choices=['train', 'validation', 'test'],
                        help='Dataset split to calculate stats for')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for data loading')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--save_path', type=str, default=None,
                        help='Path to save stats as .npz file (optional)')
    parser.add_argument('--all_splits', action='store_true',
                        help='Calculate stats across all splits (train, val, test)')
    
    args = parser.parse_args()
    
    # Create identity transform
    transform = IdentityTransform()
    
    splits = ['train', 'validation', 'test'] if args.all_splits else [args.split]
    
    all_stats = {}
    
    for split in splits:
        print(f"\n{'='*80}")
        print(f"Processing split: {split}")
        print(f"{'='*80}")
        
        # Load data using data_loading_v2
        dataloader = load_data(
            root1=args.s1_root,
            root2=args.s2_root,
            metadata=args.metadata,
            split=split,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=identity_collate_fn,
            pin_memory=False,
            drop_last=False,
            transform=transform
        )
        
        print(f"Dataset size: {len(dataloader.dataset)} samples")
        
        # Calculate statistics
        stats = calculate_stats(dataloader)
        all_stats[split] = stats
        
        # Print results
        print_stats(stats)
    
    # Save stats if requested
    if args.save_path:
        save_dict = {}
        for split, stats in all_stats.items():
            for modality in ['s1', 's2']:
                prefix = f"{split}_{modality}"
                save_dict[f"{prefix}_mean"] = stats[modality]['mean']
                save_dict[f"{prefix}_std"] = stats[modality]['std']
                save_dict[f"{prefix}_min"] = stats[modality]['min']
                save_dict[f"{prefix}_max"] = stats[modality]['max']
        
        np.savez(args.save_path, **save_dict)
        print(f"\nStatistics saved to: {args.save_path}")


if __name__ == '__main__':
    main()
