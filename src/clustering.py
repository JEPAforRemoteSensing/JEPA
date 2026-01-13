"""
Clustering and Dimensionality Reduction Analysis for I-JEPA Embeddings

This script:
1. Loads trained encoders (S1 and S2) from a checkpoint
2. Extracts embeddings using ONLY the encoders (no predictor)
3. Performs dimensionality reduction (t-SNE, UMAP, PCA) on combined embeddings
4. Visualizes the results to understand the learned representation space
"""

import os
import sys
import argparse
import logging
from tqdm import tqdm

import numpy as np
import torch
import torch.nn.functional as F
import tifffile
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    print("Warning: UMAP not installed. Install with: pip install umap-learn")

from encoder import build_encoder

# Logging setup
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)


class EmbeddingDataset(torch.utils.data.Dataset):
    """Dataset for extracting embeddings from images."""
    
    def __init__(self, root, split, transform=None, modality='s2'):
        self.data_path = os.path.join(root, split)
        self.transform = transform
        self.modality = modality
        self.metadata = []
        
        for img_file in os.scandir(self.data_path):
            if img_file.name.endswith('.tif'):
                self.metadata.append(img_file.name)
    
    def __getitem__(self, idx):
        filename = self.metadata[idx]
        img_path = os.path.join(self.data_path, filename)
        img = torch.from_numpy(tifffile.imread(img_path)).float()
        
        # BGR to RGB for S2, keep as-is for S1
        if self.modality == 's2':
            img = img[[2, 1, 0], :, :]
        
        if self.transform:
            img = self.transform(img)
        
        # Return image, patch_id
        patch_id = filename.replace('.tif', '')
        return img, patch_id
    
    def __len__(self):
        return len(self.metadata)


def make_eval_transform(num_channels, crop_size=224):
    """Create evaluation transform (no augmentation, just resize and normalize)."""
    from torchvision.transforms import v2
    
    if num_channels == 3:
        # RGB normalization for S2
        normalization = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    else:
        # Default normalization for S1
        normalization = ([0] * num_channels, [1] * num_channels)
    
    transform = v2.Compose([
        v2.Resize((crop_size, crop_size)),
        v2.ToTensor(),
        v2.Normalize(mean=normalization[0], std=normalization[1])
    ])
    return transform


def load_encoders(checkpoint_path, device, model_name='vit_base', patch_size=16, 
                  crop_size=224, in_chans1=2, in_chans2=3):
    """Load both encoders from checkpoint."""
    
    encoder1, embed_dim1 = build_encoder(
        model_name=model_name,
        img_size=crop_size,
        patch_size=patch_size,
        in_chans=in_chans1,
    )
    
    encoder2, embed_dim2 = build_encoder(
        model_name=model_name,
        img_size=crop_size,
        patch_size=patch_size,
        in_chans=in_chans2,
    )
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    def _clean_state_dict(sd: dict):
        """Remove common wrapper prefixes from state_dict keys."""
        new_sd = {}
        for k, v in sd.items():
            new_k = k
            if new_k.startswith('module.'):
                new_k = new_k[len('module.'):]
            new_k = new_k.replace('_orig_mod.', '')
            new_k = new_k.replace('module.', '')
            new_sd[new_k] = v
        return new_sd

    if 'encoder1' in checkpoint and 'encoder2' in checkpoint:
        sd1 = checkpoint['encoder1']
        sd2 = checkpoint['encoder2']

        if any(k.startswith('_orig_mod.') or k.startswith('module.') for k in sd1.keys()):
            sd1 = _clean_state_dict(sd1)

        if any(k.startswith('_orig_mod.') or k.startswith('module.') for k in sd2.keys()):
            sd2 = _clean_state_dict(sd2)

        encoder1.load_state_dict(sd1)
        encoder2.load_state_dict(sd2)
        logger.info("Loaded encoder1 (S1) and encoder2 (S2) from checkpoint")
    else:
        raise ValueError("encoder1 and encoder2 not found in checkpoint")
    
    encoder1 = encoder1.to(device).eval()
    encoder2 = encoder2.to(device).eval()
    
    return encoder1, encoder2


def extract_embeddings(encoder, data_loader, device, max_samples=None):
    """
    Extract embeddings using ONLY the encoder (no predictor).
    
    Args:
        encoder: The encoder model (S1 or S2)
        data_loader: DataLoader for the dataset
        device: torch device
        max_samples: Maximum number of samples to extract (None for all)
    
    Returns:
        embeddings: (N, embed_dim) tensor of embeddings
        patch_ids: List of patch IDs
    """
    embeddings = []
    patch_ids = []
    total_samples = 0
    
    with torch.no_grad():
        for images, pids in tqdm(data_loader, desc="Extracting embeddings"):
            if max_samples and total_samples >= max_samples:
                break
                
            images = images.to(device, non_blocking=True)
            
            # Get encoder output (B, num_patches, embed_dim)
            # NO PREDICTOR USED - only raw encoder output
            h = encoder(images)
            
            # Global average pooling to get (B, embed_dim)
            h = h.mean(dim=1)
            
            # L2 normalize for better clustering
            h = F.normalize(h, p=2, dim=-1)
            
            embeddings.append(h.cpu())
            patch_ids.extend(pids)
            total_samples += len(pids)
            
            if max_samples and total_samples >= max_samples:
                break
    
    embeddings = torch.cat(embeddings, dim=0)
    
    if max_samples:
        embeddings = embeddings[:max_samples]
        patch_ids = patch_ids[:max_samples]
    
    return embeddings, patch_ids


def perform_pca(embeddings, n_components=2):
    """Perform PCA dimensionality reduction."""
    logger.info(f"Performing PCA with {n_components} components...")
    pca = PCA(n_components=n_components)
    reduced = pca.fit_transform(embeddings.numpy())
    explained_var = pca.explained_variance_ratio_
    logger.info(f"PCA explained variance: {explained_var}")
    return reduced, explained_var


def perform_tsne(embeddings, n_components=2, perplexity=30, max_iter=1000):
    """Perform t-SNE dimensionality reduction."""
    logger.info(f"Performing t-SNE with perplexity={perplexity}, max_iter={max_iter}...")
    tsne = TSNE(n_components=n_components, perplexity=perplexity, 
                max_iter=max_iter, random_state=42, init='pca')
    reduced = tsne.fit_transform(embeddings.numpy())
    return reduced


def perform_umap(embeddings, n_components=2, n_neighbors=15, min_dist=0.1):
    """Perform UMAP dimensionality reduction."""
    if not UMAP_AVAILABLE:
        logger.warning("UMAP not available. Skipping UMAP analysis.")
        return None
    
    logger.info(f"Performing UMAP with n_neighbors={n_neighbors}, min_dist={min_dist}...")
    reducer = umap.UMAP(n_components=n_components, n_neighbors=n_neighbors, 
                        min_dist=min_dist, random_state=42)
    reduced = reducer.fit_transform(embeddings.numpy())
    return reduced


def visualize_embeddings(embeddings_dict, labels, title, save_path=None):
    """
    Visualize embeddings with different colors for S1 and S2.
    
    Args:
        embeddings_dict: Dict with 'pca', 'tsne', 'umap' keys containing 2D embeddings
        labels: Array of labels (0 for S1, 1 for S2)
        title: Plot title prefix
        save_path: Optional path to save the figure
    """
    n_methods = sum(1 for v in embeddings_dict.values() if v is not None)
    fig, axes = plt.subplots(1, n_methods, figsize=(6 * n_methods, 5))
    
    if n_methods == 1:
        axes = [axes]
    
    colors = ['#1f77b4', '#ff7f0e']  # Blue for S1, Orange for S2
    labels_text = ['S1 (Sentinel-1)', 'S2 (Sentinel-2)']
    
    ax_idx = 0
    for method_name, embeddings in embeddings_dict.items():
        if embeddings is None:
            continue
            
        ax = axes[ax_idx]
        
        # Plot S1 and S2 separately for legend
        for modality_idx in [0, 1]:
            mask = labels == modality_idx
            ax.scatter(
                embeddings[mask, 0], 
                embeddings[mask, 1],
                c=colors[modality_idx],
                label=labels_text[modality_idx],
                alpha=0.6,
                s=30
            )
        
        ax.set_title(f'{title} - {method_name.upper()}')
        ax.set_xlabel('Component 1')
        ax.set_ylabel('Component 2')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        ax_idx += 1
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Figure saved to {save_path}")
    
    plt.show()


def main(args):
    """Main function for clustering analysis."""
    
    # Device setup
    if torch.backends.mps.is_available():
        device = torch.device('mps')
        logger.info("Using Apple MPS (Metal Performance Shaders)")
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    logger.info(f"Using device: {device}")
    
    # Load encoders
    logger.info(f"Loading encoders from {args.checkpoint}")
    encoder1, encoder2 = load_encoders(
        args.checkpoint,
        device,
        model_name=args.model_name,
        patch_size=args.patch_size,
        crop_size=args.crop_size,
        in_chans1=args.in_chans_s1,
        in_chans2=args.in_chans_s2,
    )
    
    # Create transforms
    transform_s1 = make_eval_transform(num_channels=args.in_chans_s1, crop_size=args.crop_size)
    transform_s2 = make_eval_transform(num_channels=args.in_chans_s2, crop_size=args.crop_size)
    
    # Create datasets
    logger.info("Creating datasets...")
    dataset_s1 = EmbeddingDataset(
        root=args.data_root_s1,
        split=args.split,
        transform=transform_s1,
        modality='s1'
    )
    
    dataset_s2 = EmbeddingDataset(
        root=args.data_root_s2,
        split=args.split,
        transform=transform_s2,
        modality='s2'
    )
    
    logger.info(f"S1 dataset size: {len(dataset_s1)}")
    logger.info(f"S2 dataset size: {len(dataset_s2)}")
    
    # Create data loaders
    loader_s1 = torch.utils.data.DataLoader(
        dataset_s1,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    loader_s2 = torch.utils.data.DataLoader(
        dataset_s2,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Extract embeddings using ONLY the encoders (no predictor!)
    logger.info("Extracting S1 embeddings (encoder only, no predictor)...")
    embeddings_s1, patch_ids_s1 = extract_embeddings(
        encoder1, loader_s1, device, max_samples=args.max_samples
    )
    
    logger.info("Extracting S2 embeddings (encoder only, no predictor)...")
    embeddings_s2, patch_ids_s2 = extract_embeddings(
        encoder2, loader_s2, device, max_samples=args.max_samples
    )
    
    logger.info(f"S1 embeddings shape: {embeddings_s1.shape}")
    logger.info(f"S2 embeddings shape: {embeddings_s2.shape}")
    
    # Combine embeddings
    combined_embeddings = torch.cat([embeddings_s1, embeddings_s2], dim=0)
    logger.info(f"Combined embeddings shape: {combined_embeddings.shape}")
    
    # Create labels (0 for S1, 1 for S2)
    labels = np.array([0] * len(embeddings_s1) + [1] * len(embeddings_s2))
    
    # Perform dimensionality reduction
    embeddings_dict = {}
    
    # PCA
    pca_result, explained_var = perform_pca(combined_embeddings, n_components=2)
    embeddings_dict['pca'] = pca_result
    
    # t-SNE
    tsne_result = perform_tsne(
        combined_embeddings, 
        perplexity=min(args.perplexity, len(combined_embeddings) - 1),
        max_iter=args.tsne_max_iter
    )
    embeddings_dict['tsne'] = tsne_result
    
    # UMAP
    if UMAP_AVAILABLE:
        umap_result = perform_umap(
            combined_embeddings,
            n_neighbors=min(args.n_neighbors, len(combined_embeddings) - 1),
            min_dist=args.min_dist
        )
        embeddings_dict['umap'] = umap_result
    else:
        embeddings_dict['umap'] = None
    
    # Visualize
    save_path = args.output if args.output else None
    visualize_embeddings(
        embeddings_dict, 
        labels, 
        f"I-JEPA Encoder Embeddings ({args.split})",
        save_path=save_path
    )
    
    # Save embeddings if requested
    if args.save_embeddings:
        output_dir = os.path.dirname(args.save_embeddings) or '.'
        os.makedirs(output_dir, exist_ok=True)
        
        np.savez(
            args.save_embeddings,
            embeddings_s1=embeddings_s1.numpy(),
            embeddings_s2=embeddings_s2.numpy(),
            patch_ids_s1=patch_ids_s1,
            patch_ids_s2=patch_ids_s2,
            labels=labels,
            pca=pca_result,
            tsne=tsne_result,
            umap=embeddings_dict.get('umap'),
            pca_explained_variance=explained_var
        )
        logger.info(f"Embeddings saved to {args.save_embeddings}")
    
    # Print summary statistics
    logger.info("\n" + "="*50)
    logger.info("SUMMARY")
    logger.info("="*50)
    logger.info(f"Total S1 embeddings: {len(embeddings_s1)}")
    logger.info(f"Total S2 embeddings: {len(embeddings_s2)}")
    logger.info(f"Combined embeddings: {len(combined_embeddings)}")
    logger.info(f"Embedding dimension: {embeddings_s1.shape[1]}")
    logger.info(f"PCA explained variance (2 components): {sum(explained_var)*100:.2f}%")
    
    # Compute some basic statistics about the embedding space
    s1_mean = embeddings_s1.mean(dim=0)
    s2_mean = embeddings_s2.mean(dim=0)
    modality_distance = F.cosine_similarity(s1_mean.unsqueeze(0), s2_mean.unsqueeze(0)).item()
    logger.info(f"Cosine similarity between S1 and S2 centroids: {modality_distance:.4f}")
    
    return embeddings_dict, labels


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Clustering analysis for I-JEPA embeddings')
    
    # Data paths
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to the model checkpoint')
    parser.add_argument('--data_root_s1', type=str, required=True,
                        help='Root directory for S1 data')
    parser.add_argument('--data_root_s2', type=str, required=True,
                        help='Root directory for S2 data')
    parser.add_argument('--split', type=str, default='test',
                        choices=['train', 'validation', 'test'],
                        help='Data split to use')
    
    # Model parameters
    parser.add_argument('--model_name', type=str, default='vit_base',
                        help='Encoder model name')
    parser.add_argument('--patch_size', type=int, default=16,
                        help='Patch size')
    parser.add_argument('--crop_size', type=int, default=224,
                        help='Input crop size')
    parser.add_argument('--in_chans_s1', type=int, default=2,
                        help='Number of input channels for S1')
    parser.add_argument('--in_chans_s2', type=int, default=3,
                        help='Number of input channels for S2')
    
    # Data loading
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for extraction')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Maximum samples per modality (None for all)')
    
    # Dimensionality reduction parameters
    parser.add_argument('--perplexity', type=int, default=30,
                        help='t-SNE perplexity')
    parser.add_argument('--tsne_max_iter', type=int, default=1000,
                        help='t-SNE max iterations')
    parser.add_argument('--n_neighbors', type=int, default=15,
                        help='UMAP n_neighbors')
    parser.add_argument('--min_dist', type=float, default=0.1,
                        help='UMAP min_dist')
    
    # Output
    parser.add_argument('--output', type=str, default=None,
                        help='Path to save the visualization figure')
    parser.add_argument('--save_embeddings', type=str, default=None,
                        help='Path to save embeddings as .npz file')
    
    args = parser.parse_args()
    main(args)
