"""
I-JEPA Testing Script

Test the trained I-JEPA model by:
1. Encoding query images from test set
2. Finding top-k similar images from validation set
3. Computing F1 score based on multi-label classification using one-hot labels
"""

import os
import sys
import argparse
import logging
from tqdm import tqdm

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import tifffile
import matplotlib.pyplot as plt

from encoder import build_encoder, get_num_patches
from transforms import make_transforms_rgb, make_transforms

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
        
        # Return image, patch_id, and path
        patch_id = filename.replace('.tif', '')
        return img, patch_id, img_path
    
    def __len__(self):
        return len(self.metadata)


def load_models(checkpoint_path, device, model_name='vit_base', patch_size=16, crop_size=224, in_chans1=2, in_chans2=3):
    """Load both encoders from checkpoint or return None for random baseline."""
    
    if checkpoint_path.lower() == 'random':
        logger.info("Using random baseline - no models loaded")
        return None, None

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
            # Remove common wrappers like 'module.' or custom wrappers like '_orig_mod.'
            if new_k.startswith('module.'):
                new_k = new_k[len('module.'):]
            # Remove occurrences of the custom prefix anywhere in the key
            new_k = new_k.replace('_orig_mod.', '')
            new_k = new_k.replace('module.', '')
            new_sd[new_k] = v
        return new_sd

    if 'encoder1' in checkpoint and 'encoder2' in checkpoint:
        sd1 = checkpoint['encoder1']
        sd2 = checkpoint['encoder2']

        # If keys are wrapped (e.g. starting with '_orig_mod.'), clean them
        if any(k.startswith('_orig_mod.') or k.startswith('module.') for k in sd1.keys()):
            sd1 = _clean_state_dict(sd1)

        if any(k.startswith('_orig_mod.') or k.startswith('module.') for k in sd2.keys()):
            sd2 = _clean_state_dict(sd2)

        # Load with the cleaned state dicts
        encoder1.load_state_dict(sd1)
        encoder2.load_state_dict(sd2)
        logger.info("Loaded encoder1 and encoder2 from checkpoint (cleaned keys if needed)")
    else:
        raise ValueError("encoder1 and encoder2 not found in checkpoint")
    
    encoder1 = encoder1.to(device).eval()
    encoder2 = encoder2.to(device).eval()
    
    return encoder1, encoder2


def extract_embeddings(encoder, data_loader, device):
    """Extract embeddings for all images in the data loader."""
    
    embeddings = []
    patch_ids = []
    img_paths = []
    
    with torch.no_grad():
        for images, pids, paths in tqdm(data_loader, desc="Extracting embeddings"):
            images = images.to(device, non_blocking=True)
            
            # Get encoder output (B, num_patches, embed_dim)
            h = encoder(images)
            
            # Global average pooling to get (B, embed_dim)
            h = h.mean(dim=1)
            
            # L2 normalize for cosine similarity
            h = F.normalize(h, p=2, dim=-1)
            
            embeddings.append(h.cpu())
            patch_ids.extend(pids)
            img_paths.extend(paths)
    
    embeddings = torch.cat(embeddings, dim=0)
    return embeddings, patch_ids, img_paths


def build_knn_index(gallery_embeddings, k=10, metric='cosine'):
    """
    Build a KNN index using scikit-learn's NearestNeighbors.
    
    Args:
        gallery_embeddings: (N, embed_dim) gallery embeddings
        k: number of neighbors to retrieve
        metric: distance metric ('cosine', 'euclidean', etc.)
    
    Returns:
        Fitted NearestNeighbors model
    """
    from sklearn.neighbors import NearestNeighbors
    
    knn = NearestNeighbors(n_neighbors=k, metric=metric, algorithm='auto')
    knn.fit(gallery_embeddings.numpy())
    return knn


def get_topk_indices_knn(knn, query_embeddings, k=10):
    """
    Get top-k most similar gallery indices for each query using KNN.
    
    Args:
        knn: Fitted NearestNeighbors model
        query_embeddings: (N, embed_dim) query embeddings
        k: number of neighbors to retrieve
    
    Returns:
        topk_idx: (N, k) indices of top-k neighbors
        topk_distances: (N, k) distances to top-k neighbors
    """
    distances, indices = knn.kneighbors(query_embeddings.numpy(), n_neighbors=k)
    
    # Convert distances to similarity scores (for cosine: similarity = 1 - distance)
    # Note: sklearn's cosine metric returns distance, not similarity
    similarities = 1 - distances
    
    return torch.from_numpy(indices), torch.from_numpy(similarities)


def load_labels(metadata_path, patch_ids, modality='s2'):
    """Load one-hot labels from metadata parquet file.
    
    Args:
        metadata_path: Path to parquet file
        patch_ids: List of patch IDs (filenames without .tif)
        modality: 's1' or 's2' - determines which column to use for lookup
    """
    
    df = pd.read_parquet(metadata_path)
    
    # Create mapping based on modality
    patch_to_labels = {}
    if modality == 's1':
        # Use s1_name column for S1 images
        for _, row in df.iterrows():
            patch_to_labels[row['s1_name']] = np.array(row['one_hot_labels'])
    else:
        # Use patch_id column for S2 images
        for _, row in df.iterrows():
            patch_to_labels[row['patch_id']] = np.array(row['one_hot_labels'])
    
    # Get labels for requested patch_ids
    labels = []
    for pid in patch_ids:
        if pid in patch_to_labels:
            labels.append(patch_to_labels[pid])
        else:
            logger.warning(f"Patch {pid} not found in metadata")
            labels.append(np.zeros(19))  # 19 classes
    
    return np.stack(labels)


def compute_retrieval_metrics(query_labels, gallery_labels, topk_indices):
    """
    Compute retrieval precision and recall based on label overlap.
    
    Precision = (1/|X^final|) * sum_r (|L_q ∩ L_r| / |L_r|)
    Recall = (1/|X^final|) * sum_r (|L_q ∩ L_r| / |L_q|)
    
    Where:
        - X^final is the set of top-k retrieved items
        - L_q is the label set of the query
        - L_r is the label set of retrieved item r
    
    Args:
        query_labels: (N, num_classes) one-hot ground truth labels for queries
        gallery_labels: (M, num_classes) one-hot labels for gallery/validation set
        topk_indices: (N, k) indices of top-k retrieved items for each query
    
    Returns:
        Dictionary with precision, recall, and F1 scores
    """
    num_queries = query_labels.shape[0]
    k = topk_indices.shape[1]
    
    total_precision = 0.0
    total_recall = 0.0
    
    for i in range(num_queries):
        L_q = query_labels[i]  # Query label set (one-hot)
        query_precision = 0.0
        query_recall = 0.0
        
        for j in range(k):
            retrieved_idx = topk_indices[i, j]
            L_r = gallery_labels[retrieved_idx]  # Retrieved item label set
            
            # Compute intersection: |L_q ∩ L_r|
            intersection = np.sum(np.logical_and(L_q, L_r))
            
            # |L_r| - number of labels in retrieved item
            L_r_size = np.sum(L_r)
            # |L_q| - number of labels in query
            L_q_size = np.sum(L_q)
            
            # Precision contribution: |L_q ∩ L_r| / |L_r|
            if L_r_size > 0:
                query_precision += intersection / L_r_size
            
            # Recall contribution: |L_q ∩ L_r| / |L_q|
            if L_q_size > 0:
                query_recall += intersection / L_q_size
        
        # Average over k retrieved items
        total_precision += query_precision / k
        total_recall += query_recall / k
    
    # Average over all queries
    avg_precision = total_precision / num_queries
    avg_recall = total_recall / num_queries
    
    # F1 score
    if avg_precision + avg_recall > 0:
        f1 = 2 * avg_precision * avg_recall / (avg_precision + avg_recall)
    else:
        f1 = 0.0
    
    return {
        'precision': avg_precision,
        'recall': avg_recall,
        'f1': f1,
    }
def visualize_retrieval(query_path, retrieved_path, query_modality, retrieved_modality):
    """Display query and retrieved images side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    # Load and display query image
    query_img = tifffile.imread(query_path)
    if query_modality == 's2':
        query_img = query_img[[2, 1, 0], :, :]  # BGR to RGB
        query_img = np.transpose(query_img, (1, 2, 0))
        query_img = (query_img - query_img.min()) / (query_img.max() - query_img.min())
    else:  # s1
        # Show VV channel for S1
        query_img = query_img[0, :, :]
    
    axes[0].imshow(query_img, cmap='gray' if query_modality == 's1' else None)
    axes[0].set_title(f'Query ({query_modality.upper()})')
    axes[0].axis('off')
    
    # Load and display retrieved image
    retrieved_img = tifffile.imread(retrieved_path)
    if retrieved_modality == 's2':
        retrieved_img = retrieved_img[[2, 1, 0], :, :]  # BGR to RGB
        retrieved_img = np.transpose(retrieved_img, (1, 2, 0))
        retrieved_img = (retrieved_img - retrieved_img.min()) / (retrieved_img.max() - retrieved_img.min())
    else:  # s1
        # Show VV channel for S1
        retrieved_img = retrieved_img[0, :, :]
    
    axes[1].imshow(retrieved_img, cmap='gray' if retrieved_modality == 's1' else None)
    axes[1].set_title(f'Retrieved ({retrieved_modality.upper()})')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.show()


def run_evaluation_with_embeddings(args, mode, data_dict):
    """Run evaluation using pre-extracted embeddings."""
    
    q_mod, g_mod = mode.split('_')
    
    query_data = data_dict[q_mod]['test']
    gallery_data = data_dict[g_mod]['validation']
    
    logger.info(f"\n--- Evaluating mode: {mode} ---")
    
    if query_data['embeddings'] is None:
        # Random retrieval baseline
        num_queries = len(query_data['patch_ids'])
        num_gallery = len(gallery_data['patch_ids'])
        logger.info(f"Generating random retrieval for {num_queries} queries from {num_gallery} items")
        
        # Generate random indices for each query
        topk_idx = []
        for _ in range(num_queries):
            topk_idx.append(torch.randperm(num_gallery)[:args.top_k])
        topk_idx = torch.stack(topk_idx)
        topk_sim = torch.zeros_like(topk_idx).float()
    else:
        # Build KNN index
        knn = build_knn_index(gallery_data['embeddings'], k=args.top_k, metric='cosine')
        
        # Get top-k indices
        topk_idx, topk_sim = get_topk_indices_knn(knn, query_data['embeddings'], k=args.top_k)
    
    # Compute metrics
    metrics = compute_retrieval_metrics(
        query_data['labels'], 
        gallery_data['labels'], 
        topk_idx.numpy()
    )
    
    # Print results
    logger.info(f"Mode {mode} RESULTS: Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}, F1: {metrics['f1']:.4f}")
    
    # Examples
    if args.show_examples > 0:
        for i in range(min(args.show_examples, len(query_data['patch_ids']))):
            query_pid = query_data['patch_ids'][i]
            logger.info(f"Example {i+1} [Q: {query_pid}, Labels: {query_data['text_labels'][i]}]")
            
            top_idx = topk_idx[i, 0].item()
            ret_pid = gallery_data['patch_ids'][top_idx]
            logger.info(f"  Top-1: {ret_pid} (sim={topk_sim[i, 0]:.4f}), Labels: {gallery_data['text_labels'][top_idx]}")
            
            visualize_retrieval(query_data['paths'][i], gallery_data['paths'][top_idx], q_mod, g_mod)
            
    return metrics


def main(args):
    """Main testing function."""
    
    device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
    logger.info(f"Using device: {device}")
    
    # Load models
    encoders = load_models(
        checkpoint_path=args.checkpoint,
        device=device,
        model_name=args.model_name,
        patch_size=args.patch_size,
        crop_size=args.crop_size,
        in_chans1=args.in_chans1,
        in_chans2=args.in_chans2,
    )
    
    transform_s1 = make_transforms(num_channels=2)
    transform_s2 = make_transforms_rgb(num_channels=3)
    
    modes = ['s1_s1', 's2_s2', 's1_s2', 's2_s1'] if args.mode == 'all' else [args.mode]
    
    # Identify which modalities we need
    needed_modalities = set()
    for m in modes:
        q_mod, g_mod = m.split('_')
        needed_modalities.add(q_mod)
        needed_modalities.add(g_mod)
    
    # Extract all necessary embeddings once
    data_dict = {}
    df = pd.read_parquet(args.metadata_path)
    
    for mod in needed_modalities:
        data_dict[mod] = {}
        encoder = None
        if encoders is not None:
            encoder = encoders[0] if mod == 's1' else encoders[1]
        
        root = args.data_root_s1 if mod == 's1' else args.data_root_s2
        transform = transform_s1 if mod == 's1' else transform_s2
        
        for split in ['test', 'validation']:
            if encoder is not None:
                logger.info(f"Extracting {mod.upper()} {split} embeddings...")
                ds = EmbeddingDataset(root=root, split=split, transform=transform, modality=mod)
                loader = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
                
                emb, pids, paths = extract_embeddings(encoder, loader, device)
            else:
                logger.info(f"Collecting {mod.upper()} {split} metadata (random baseline)...")
                pids = []
                paths = []
                data_path = os.path.join(root, split)
                for img_file in os.scandir(data_path):
                    if img_file.name.endswith('.tif'):
                        pids.append(img_file.name.replace('.tif', ''))
                        paths.append(img_file.path)
                emb = None

            labels = load_labels(args.metadata_path, pids, modality=mod)
            
            # Map pids to text labels for display
            id_col = 's1_name' if mod == 's1' else 'patch_id'
            lookup = dict(zip(df[id_col], df['labels']))
            text_labels = [lookup.get(p, []) for p in pids]
            
            data_dict[mod][split] = {
                'embeddings': emb,
                'patch_ids': pids,
                'paths': paths,
                'labels': labels,
                'text_labels': text_labels
            }

    results = {}
    for m in modes:
        results[m] = run_evaluation_with_embeddings(args, m, data_dict)
        
    if len(modes) > 1:
        logger.info("\n" + "="*30 + "\nFINAL SUMMARY\n" + "="*30)
        for m, m_results in results.items():
            logger.info(f"{m:6}: F1 = {m_results['f1']:.4f}")
        logger.info("="*30)

    return results


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='I-JEPA Testing')
    
    # Model
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--model_name', type=str, default='vit_base',
                        choices=['vit_tiny', 'vit_small', 'vit_base', 'vit_large', 'vit_huge', 'vit_giant'])
    parser.add_argument('--patch_size', type=int, default=16)
    parser.add_argument('--in_chans1', type=int, default=2)
    parser.add_argument('--in_chans2', type=int, default=3)
    parser.add_argument('--crop_size', type=int, default=224)
    
    # Data
    parser.add_argument('--data_root_s1', type=str, required=True, help='Path to BigEarthNet-S1 folder')
    parser.add_argument('--data_root_s2', type=str, required=True, help='Path to BigEarthNet-S2 folder')
    parser.add_argument('--metadata_path', type=str, required=True, help='Path to serbia_metadata.parquet')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=4)
    
    # Mode
    parser.add_argument('--mode', type=str, required=True, 
                        choices=['s1_s1', 's2_s2', 's1_s2', 's2_s1', 'all'],
                        help='Inference mode: s1_s1, s2_s2, s1_s2, s2_s1, or all')
    
    # Retrieval
    parser.add_argument('--top_k', type=int, default=10, help='Number of top similar images to retrieve')
    
    # Output
    parser.add_argument('--show_examples', type=int, default=3, help='Number of example retrievals to show')
    
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args)
