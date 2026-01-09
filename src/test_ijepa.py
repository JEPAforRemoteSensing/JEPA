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

from encoder import build_encoder, get_num_patches
from transforms import make_transforms_rgb

# Logging setup
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)


class EmbeddingDataset(torch.utils.data.Dataset):
    """Dataset for extracting embeddings from images."""
    
    def __init__(self, root, split, transform=None):
        self.data_path = os.path.join(root, split)
        self.transform = transform
        self.metadata = []
        
        for img_file in os.scandir(self.data_path):
            if img_file.name.endswith('.tif'):
                self.metadata.append(img_file.name)
    
    def __getitem__(self, idx):
        filename = self.metadata[idx]
        img = torch.from_numpy(tifffile.imread(os.path.join(self.data_path, filename))).float()
        # BGR to RGB (channels 2, 1, 0)
        img = img[[2, 1, 0], :, :]
        if self.transform:
            img = self.transform(img)
        
        # Return image and patch_id (filename without .tif extension)
        patch_id = filename.replace('.tif', '')
        return img, patch_id
    
    def __len__(self):
        return len(self.metadata)


def load_model(checkpoint_path, device, model_name='vit_base', patch_size=16, crop_size=224, in_chans=3):
    """Load the trained encoder from checkpoint."""
    
    encoder, embed_dim = build_encoder(
        model_name=model_name,
        img_size=crop_size,
        patch_size=patch_size,
        in_chans=in_chans,
    )
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Load target encoder (EMA encoder used for inference)
    if 'target_encoder' in checkpoint:
        encoder.load_state_dict(checkpoint['target_encoder'])
        logger.info("Loaded target encoder from checkpoint")
    elif 'encoder' in checkpoint:
        encoder.load_state_dict(checkpoint['encoder'])
        logger.info("Loaded encoder from checkpoint")
    else:
        raise ValueError("No encoder found in checkpoint")
    
    encoder = encoder.to(device)
    encoder.eval()
    
    return encoder


def extract_embeddings(encoder, data_loader, device):
    """Extract embeddings for all images in the data loader."""
    
    embeddings = []
    patch_ids = []
    
    with torch.no_grad():
        for images, pids in tqdm(data_loader, desc="Extracting embeddings"):
            images = images.to(device, non_blocking=True)
            
            # Get encoder output (B, num_patches, embed_dim)
            h = encoder(images)
            
            # Global average pooling to get (B, embed_dim)
            h = h.mean(dim=1)
            
            # L2 normalize for cosine similarity
            h = F.normalize(h, p=2, dim=-1)
            
            embeddings.append(h.cpu())
            patch_ids.extend(pids)
    
    embeddings = torch.cat(embeddings, dim=0)
    return embeddings, patch_ids


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


def load_labels(metadata_path, patch_ids):
    """Load one-hot labels from metadata parquet file."""
    
    df = pd.read_parquet(metadata_path)
    
    # Create mapping from patch_id to one_hot_labels
    patch_to_labels = {}
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


def main(args):
    """Main testing function."""
    
    # Device setup
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
        logger.warning("CUDA and MPS not available, using CPU")
    
    logger.info(f"Using device: {device}")
    
    # Load model
    logger.info(f"Loading model from {args.checkpoint}")
    encoder = load_model(
        checkpoint_path=args.checkpoint,
        device=device,
        model_name=args.model_name,
        patch_size=args.patch_size,
        crop_size=args.crop_size,
        in_chans=args.in_chans,
    )
    
    # Create transforms
    transform = make_transforms_rgb(num_channels=3)
    
    # Create datasets
    logger.info("Creating datasets...")
    test_dataset = EmbeddingDataset(
        root=args.data_root,
        split='test',
        transform=transform,
    )
    val_dataset = EmbeddingDataset(
        root=args.data_root,
        split='validation',
        transform=transform,
    )
    
    logger.info(f"Test dataset: {len(test_dataset)} images")
    logger.info(f"Validation dataset: {len(val_dataset)} images")
    
    # Create data loaders
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    
    # Extract embeddings
    logger.info("Extracting test embeddings...")
    test_embeddings, test_patch_ids = extract_embeddings(encoder, test_loader, device)
    
    logger.info("Extracting validation embeddings...")
    val_embeddings, val_patch_ids = extract_embeddings(encoder, val_loader, device)
    
    logger.info(f"Test embeddings shape: {test_embeddings.shape}")
    logger.info(f"Validation embeddings shape: {val_embeddings.shape}")
    
    # Build KNN index from validation embeddings
    logger.info(f"Building KNN index with k={args.top_k}...")
    knn = build_knn_index(val_embeddings, k=args.top_k, metric='cosine')
    
    # Get top-k indices using KNN
    logger.info(f"Finding top-{args.top_k} similar images using KNN...")
    topk_idx, topk_sim = get_topk_indices_knn(knn, test_embeddings, k=args.top_k)
    
    # Load labels
    logger.info(f"Loading labels from {args.metadata_path}")
    test_labels = load_labels(args.metadata_path, test_patch_ids)
    val_labels = load_labels(args.metadata_path, val_patch_ids)
    
    logger.info(f"Test labels shape: {test_labels.shape}")
    logger.info(f"Validation labels shape: {val_labels.shape}")
    
    # Compute retrieval metrics based on label overlap
    logger.info("Computing retrieval metrics...")
    metrics = compute_retrieval_metrics(test_labels, val_labels, topk_idx.numpy())

    logger.info("=" * 50)
    logger.info("RESULTS")
    logger.info("=" * 50)
    logger.info(f"Top-K: {args.top_k}")
    logger.info(f"Precision: {metrics['precision']:.4f}")
    logger.info(f"Recall: {metrics['recall']:.4f}")
    logger.info(f"F1 Score: {metrics['f1']:.4f}")
    logger.info("=" * 50)
    
    # Print some examples
    if args.show_examples > 0:
        logger.info("\nExample retrievals:")
        df = pd.read_parquet(args.metadata_path)
        patch_to_labels_text = {row['patch_id']: row['labels'] for _, row in df.iterrows()}
        
        for i in range(min(args.show_examples, len(test_patch_ids))):
            query_pid = test_patch_ids[i]
            query_labels_text = patch_to_labels_text.get(query_pid, [])
            
            logger.info(f"\nQuery: {query_pid}")
            logger.info(f"  Labels: {query_labels_text}")
            logger.info(f"  Top-{args.top_k} retrieved:")
            
            for j, idx in enumerate(topk_idx[i][:5].numpy()):  # Show top 5
                retrieved_pid = val_patch_ids[idx]
                retrieved_labels = patch_to_labels_text.get(retrieved_pid, [])
                sim = topk_sim[i][j].item()
                logger.info(f"    {j+1}. {retrieved_pid} (sim={sim:.4f})")
                logger.info(f"       Labels: {retrieved_labels}")
    
    return metrics


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='I-JEPA Testing')
    
    # Model
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--model_name', type=str, default='vit_base',
                        choices=['vit_tiny', 'vit_small', 'vit_base', 'vit_large', 'vit_huge', 'vit_giant'])
    parser.add_argument('--patch_size', type=int, default=16)
    parser.add_argument('--in_chans', type=int, default=3)
    parser.add_argument('--crop_size', type=int, default=224)
    
    # Data
    parser.add_argument('--data_root', type=str, required=True, help='Path to BigEarthNet-S2 folder')
    parser.add_argument('--metadata_path', type=str, required=True, help='Path to serbia_metadata.parquet')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=4)
    
    # Retrieval
    parser.add_argument('--top_k', type=int, default=10, help='Number of top similar images to retrieve')
    
    # Output
    parser.add_argument('--show_examples', type=int, default=3, help='Number of example retrievals to show')
    
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args)
