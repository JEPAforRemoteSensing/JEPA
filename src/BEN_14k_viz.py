"""
Visualize BEN_14K pixel value distributions for each band
of both S1 (Sentinel-1) and S2 (Sentinel-2) modalities.
Creates histograms of raw pixel values without any scaling or normalization.
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from transforms import make_transforms

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


def collect_pixel_values(dataloader, s1_channels=2, s2_channels=10, sample_fraction=0.1):
    """
    Collect pixel values for histogram computation.
    
    Args:
        dataloader: DataLoader yielding (s1_tensor, s2_tensor) tuples
        s1_channels: Number of channels in S1 (default: 2 for VV, VH)
        s2_channels: Number of channels in S2 (default: 10)
        sample_fraction: Fraction of pixels to sample for histogram (to manage memory)
    
    Returns:
        Dictionary containing sampled pixel values for both modalities
    """
    # Initialize lists to collect values for each channel
    s1_values = [[] for _ in range(s1_channels)]
    s2_values = [[] for _ in range(s2_channels)]
    
    print("Collecting pixel values...")
    for s1_batch, s2_batch in tqdm(dataloader, desc="Processing batches"):
        # S1 batch shape: (B, C, H, W)
        B, C1, H, W = s1_batch.shape
        
        # Sample a fraction of pixels to manage memory
        n_pixels = B * H * W
        n_samples = max(1, int(n_pixels * sample_fraction))
        
        # Process S1
        for c in range(C1):
            channel_pixels = s1_batch[:, c, :, :].flatten().numpy()
            # Random sampling
            indices = np.random.choice(len(channel_pixels), size=min(n_samples, len(channel_pixels)), replace=False)
            s1_values[c].append(channel_pixels[indices])
        
        # Process S2
        B, C2, H, W = s2_batch.shape
        for c in range(C2):
            channel_pixels = s2_batch[:, c, :, :].flatten().numpy()
            indices = np.random.choice(len(channel_pixels), size=min(n_samples, len(channel_pixels)), replace=False)
            s2_values[c].append(channel_pixels[indices])
    
    # Concatenate all collected values
    s1_values = [np.concatenate(vals) for vals in s1_values]
    s2_values = [np.concatenate(vals) for vals in s2_values]
    
    return {
        's1': {
            'values': s1_values,
            'channel_names': ['VV', 'VH']
        },
        's2': {
            'values': s2_values,
            'channel_names': ['B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B11', 'B12']
        }
    }


def plot_histograms(pixel_data, bins=100, save_path=None, split='train'):
    """
    Create histogram plots for each band.
    
    Args:
        pixel_data: Dictionary containing pixel values for S1 and S2
        bins: Number of bins for histograms
        save_path: Path to save the figure (optional)
        split: Dataset split name for title
    """
    # Create figure for S1 (2 channels)
    fig_s1, axes_s1 = plt.subplots(1, 2, figsize=(12, 4))
    fig_s1.suptitle(f'Sentinel-1 (SAR) Pixel Value Distributions - {split}', fontsize=14)
    
    # Percentiles to compute
    percentiles = [99, 99.9, 99.99, 99.9999, 99.999999]
    
    s1_data = pixel_data['s1']
    for i, (ax, ch_name) in enumerate(zip(axes_s1, s1_data['channel_names'])):
        values = s1_data['values'][i]
        ax.hist(values, bins=bins, color='steelblue', alpha=0.7, edgecolor='black', linewidth=0.5)
        ax.set_title(f'{ch_name}')
        ax.set_xlabel('Pixel Value')
        ax.set_ylabel('Frequency')
        ax.grid(True, alpha=0.3)
        
        # Calculate percentiles
        pct_values = np.percentile(values, percentiles)
        
        # Add statistics text
        stats_text = f'Mean: {np.mean(values):.2f}\nStd: {np.std(values):.2f}\nMin: {np.min(values):.2f}\nMax: {np.max(values):.2f}'
        ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Add percentile text on the left side
        pct_text = 'Percentiles:\n' + '\n'.join([f'{p}%: {v:.2f}' for p, v in zip(percentiles, pct_values)])
        ax.text(0.02, 0.95, pct_text, transform=ax.transAxes, fontsize=8,
                verticalalignment='top', horizontalalignment='left',
                bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.5))
    
    plt.tight_layout()
    
    if save_path:
        s1_path = save_path.replace('.png', '_S1.png')
        fig_s1.savefig(s1_path, dpi=150, bbox_inches='tight')
        print(f"Saved S1 histogram to: {s1_path}")
    
    # Create figure for S2 (10 channels) - 2 rows x 5 columns
    fig_s2, axes_s2 = plt.subplots(2, 5, figsize=(20, 8))
    fig_s2.suptitle(f'Sentinel-2 (MSI) Pixel Value Distributions - {split}', fontsize=14)
    
    s2_data = pixel_data['s2']
    axes_flat = axes_s2.flatten()
    
    for i, (ax, ch_name) in enumerate(zip(axes_flat, s2_data['channel_names'])):
        values = s2_data['values'][i]
        ax.hist(values, bins=bins, color='forestgreen', alpha=0.7, edgecolor='black', linewidth=0.5)
        ax.set_title(f'{ch_name}')
        ax.set_xlabel('Pixel Value')
        ax.set_ylabel('Frequency')
        ax.grid(True, alpha=0.3)
        
        # Calculate percentiles
        pct_values = np.percentile(values, percentiles)
        
        # Add statistics text
        stats_text = f'Mean: {np.mean(values):.2f}\nStd: {np.std(values):.2f}\nMin: {np.min(values):.2f}\nMax: {np.max(values):.2f}'
        ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, fontsize=8,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Add percentile text on the left side
        pct_text = 'Percentiles:\n' + '\n'.join([f'{p}%: {v:.2f}' for p, v in zip(percentiles, pct_values)])
        ax.text(0.02, 0.95, pct_text, transform=ax.transAxes, fontsize=6,
                verticalalignment='top', horizontalalignment='left',
                bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.5))
    
    plt.tight_layout()
    
    if save_path:
        s2_path = save_path.replace('.png', '_S2.png')
        fig_s2.savefig(s2_path, dpi=150, bbox_inches='tight')
        print(f"Saved S2 histogram to: {s2_path}")
    
    plt.show()
    
    return fig_s1, fig_s2


def plot_combined_histograms(pixel_data, bins=100, save_path=None, split='train'):
    """
    Create a combined histogram plot showing all bands together.
    
    Args:
        pixel_data: Dictionary containing pixel values for S1 and S2
        bins: Number of bins for histograms
        save_path: Path to save the figure (optional)
        split: Dataset split name for title
    """
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    fig.suptitle(f'BEN_14K Pixel Value Distributions (Raw Values) - {split}', fontsize=16)
    
    # Plot S1 channels in first row (first 2 subplots)
    s1_data = pixel_data['s1']
    for i, ch_name in enumerate(s1_data['channel_names']):
        ax = axes[0, i]
        values = s1_data['values'][i]
        ax.hist(values, bins=bins, color='steelblue', alpha=0.7, edgecolor='black', linewidth=0.5)
        ax.set_title(f'S1 - {ch_name}')
        ax.set_xlabel('Pixel Value')
        ax.set_ylabel('Frequency')
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots in first row
    for i in range(2, 4):
        axes[0, i].axis('off')
    
    # Plot S2 channels in remaining subplots
    s2_data = pixel_data['s2']
    s2_positions = [(0, 2), (0, 3), (1, 0), (1, 1), (1, 2), (1, 3), (2, 0), (2, 1), (2, 2), (2, 3)]
    
    for i, ch_name in enumerate(s2_data['channel_names']):
        row, col = s2_positions[i]
        ax = axes[row, col]
        values = s2_data['values'][i]
        ax.hist(values, bins=bins, color='forestgreen', alpha=0.7, edgecolor='black', linewidth=0.5)
        ax.set_title(f'S2 - {ch_name}')
        ax.set_xlabel('Pixel Value')
        ax.set_ylabel('Frequency')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        combined_path = save_path.replace('.png', '_combined.png')
        fig.savefig(combined_path, dpi=150, bbox_inches='tight')
        print(f"Saved combined histogram to: {combined_path}")
    
    plt.show()
    
    return fig


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize BEN_14K pixel value distributions')
    parser.add_argument('--s1_root', type=str, default='data/BEN_14k/BigEarthNet-S1',
                        help='Root directory for Sentinel-1 data')
    parser.add_argument('--s2_root', type=str, default='data/BEN_14k/BigEarthNet-S2',
                        help='Root directory for Sentinel-2 data')
    parser.add_argument('--metadata', type=str, default='data/BEN_14k/serbia_metadata.parquet',
                        help='Path to metadata parquet file')
    parser.add_argument('--split', type=str, default='train',
                        choices=['train', 'validation', 'test'],
                        help='Dataset split to visualize')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for data loading')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--sample_fraction', type=float, default=0.1,
                        help='Fraction of pixels to sample for histograms (to manage memory)')
    parser.add_argument('--bins', type=int, default=100,
                        help='Number of bins for histograms')
    parser.add_argument('--save_path', type=str, default=None,
                        help='Path to save histogram plots (e.g., histograms.png)')
    parser.add_argument('--combined', action='store_true',
                        help='Create a single combined plot with all bands')
    
    args = parser.parse_args()
    
    # Create identity transform (no scaling/normalization)
    transform = make_transforms(num_channels=12)
    
    print(f"\n{'='*80}")
    print(f"Visualizing pixel distributions for split: {args.split}")
    print(f"Sample fraction: {args.sample_fraction}")
    print(f"{'='*80}")
    
    # Load data using data_loading_v2
    dataloader = load_data(
        root1=args.s1_root,
        root2=args.s2_root,
        metadata=args.metadata,
        split=args.split,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=identity_collate_fn,
        pin_memory=False,
        drop_last=False,
        transform=transform
    )
    
    print(f"Dataset size: {len(dataloader.dataset)} samples")
    
    # Collect pixel values
    pixel_data = collect_pixel_values(
        dataloader, 
        sample_fraction=args.sample_fraction
    )
    
    # Print summary
    print(f"\nCollected pixel samples:")
    print(f"  S1: {[len(v) for v in pixel_data['s1']['values']]} samples per channel")
    print(f"  S2: {[len(v) for v in pixel_data['s2']['values']]} samples per channel")
    
    # Print percentile statistics
    percentiles = [99, 99.9, 99.99, 99.9999, 99.999999]
    
    print(f"\n{'='*100}")
    print("Percentile Statistics (pixel values at which X% of pixels are found)")
    print(f"{'='*100}")
    
    print(f"\nSentinel-1 (SAR):")
    print("-" * 80)
    header = f"{'Channel':<10}" + "".join([f"{p}%".rjust(15) for p in percentiles]) + f"{'Max':>15}"
    print(header)
    print("-" * 80)
    for i, ch_name in enumerate(pixel_data['s1']['channel_names']):
        values = pixel_data['s1']['values'][i]
        pct_values = np.percentile(values, percentiles)
        row = f"{ch_name:<10}" + "".join([f"{v:>15.2f}" for v in pct_values]) + f"{np.max(values):>15.2f}"
        print(row)
    
    print(f"\nSentinel-2 (MSI):")
    print("-" * 80)
    print(header)
    print("-" * 80)
    for i, ch_name in enumerate(pixel_data['s2']['channel_names']):
        values = pixel_data['s2']['values'][i]
        pct_values = np.percentile(values, percentiles)
        row = f"{ch_name:<10}" + "".join([f"{v:>15.2f}" for v in pct_values]) + f"{np.max(values):>15.2f}"
        print(row)
    
    print(f"\n{'='*100}")
    
    # Create plots
    if args.combined:
        plot_combined_histograms(
            pixel_data, 
            bins=args.bins, 
            save_path=args.save_path,
            split=args.split
        )
    else:
        plot_histograms(
            pixel_data, 
            bins=args.bins, 
            save_path=args.save_path,
            split=args.split
        )


if __name__ == '__main__':
    main()
