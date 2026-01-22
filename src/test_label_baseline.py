"""
Label-based KNN Retrieval Baseline

Test retrieval performance using only label information:
1. Load labels for images from test set using serbia_metadata.parquet
2. Load labels for images from validation set
3. Use KNN on one-hot label vectors to find top-k similar images
4. Compute F1 score based on multi-label classification using same metrics as I-JEPA
"""

import os
import sys
import argparse
import logging
from tqdm import tqdm

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import tifffile

# Logging setup
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)


def load_split_metadata(data_root, split, metadata_df, modality='s2'):
    """
    Load metadata for all images in a split.
    
    Args:
        data_root: Path to BigEarthNet-S1 or BigEarthNet-S2 folder
        split: 'test' or 'validation'
        metadata_df: DataFrame with metadata
        modality: 's1' or 's2'
    
    Returns:
        Dictionary with patch_ids, labels (one-hot), text_labels, and paths
    """
    data_path = os.path.join(data_root, split)
    
    # Get all .tif files in the split
    tif_files = []
    for img_file in os.scandir(data_path):
        if img_file.name.endswith('.tif'):
            tif_files.append((img_file.name.replace('.tif', ''), img_file.path))
    
    # Sort for consistency
    tif_files.sort(key=lambda x: x[0])
    
    patch_ids = [pid for pid, _ in tif_files]
    paths = [path for _, path in tif_files]
    
    # Create mapping from metadata
    id_col = 's1_name' if modality == 's1' else 'patch_id'
    patch_to_labels = {}
    patch_to_text_labels = {}
    
    for _, row in metadata_df.iterrows():
        patch_to_labels[row[id_col]] = np.array(row['one_hot_labels'])
        patch_to_text_labels[row[id_col]] = row['labels']
    
    # Get labels for requested patch_ids
    labels = []
    text_labels = []
    for pid in patch_ids:
        if pid in patch_to_labels:
            labels.append(patch_to_labels[pid])
            text_labels.append(patch_to_text_labels[pid])
        else:
            logger.warning(f"Patch {pid} not found in metadata")
            labels.append(np.zeros(19))  # 19 classes
            text_labels.append([])
    
    labels = np.stack(labels)
    
    return {
        'patch_ids': patch_ids,
        'paths': paths,
        'labels': labels,
        'text_labels': text_labels
    }


def build_label_knn_index(gallery_labels, k=10, metric='cosine'):
    """
    Build a KNN index using label vectors.
    
    Args:
        gallery_labels: (N, num_classes) one-hot label vectors
        k: number of neighbors to retrieve
        metric: distance metric ('cosine', 'euclidean', 'jaccard', etc.)
    
    Returns:
        Fitted NearestNeighbors model
    """
    from sklearn.neighbors import NearestNeighbors
    
    knn = NearestNeighbors(n_neighbors=k, metric=metric, algorithm='auto')
    knn.fit(gallery_labels)
    return knn


def get_topk_indices_knn(knn, query_labels, k=10):
    """
    Get top-k most similar gallery indices for each query using KNN on labels.
    
    Args:
        knn: Fitted NearestNeighbors model
        query_labels: (N, num_classes) query label vectors
        k: number of neighbors to retrieve
    
    Returns:
        topk_idx: (N, k) indices of top-k neighbors
        topk_distances: (N, k) distances to top-k neighbors
    """
    distances, indices = knn.kneighbors(query_labels, n_neighbors=k)
    
    # Convert distances to similarity scores (for cosine: similarity = 1 - distance)
    similarities = 1 - distances
    
    return torch.from_numpy(indices), torch.from_numpy(similarities)


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


def run_evaluation(args, mode, data_dict):
    """Run evaluation for a specific mode."""
    
    q_mod, g_mod = mode.split('_')
    
    query_data = data_dict[q_mod]['test']
    gallery_data = data_dict[g_mod]['validation']
    
    logger.info(f"\n--- Evaluating mode: {mode} ---")
    logger.info(f"Query set size: {len(query_data['patch_ids'])}")
    logger.info(f"Gallery set size: {len(gallery_data['patch_ids'])}")
    
    # Build KNN index on gallery labels
    knn = build_label_knn_index(gallery_data['labels'], k=args.top_k, metric=args.distance_metric)
    
    # Get top-k indices using query labels
    topk_idx, topk_sim = get_topk_indices_knn(knn, query_data['labels'], k=args.top_k)
    
    # Compute metrics
    metrics = compute_retrieval_metrics(
        query_data['labels'], 
        gallery_data['labels'], 
        topk_idx.numpy()
    )
    
    # Print results
    logger.info(f"Mode {mode} RESULTS: Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}, F1: {metrics['f1']:.4f}")
    
    # Show examples
    if args.show_examples > 0:
        for i in range(min(args.show_examples, len(query_data['patch_ids']))):
            query_pid = query_data['patch_ids'][i]
            logger.info(f"\nExample {i+1} [Q: {query_pid}, Labels: {query_data['text_labels'][i]}]")
            
            for rank in range(min(3, args.top_k)):  # Show top-3
                top_idx = topk_idx[i, rank].item()
                ret_pid = gallery_data['patch_ids'][top_idx]
                logger.info(f"  Top-{rank+1}: {ret_pid} (sim={topk_sim[i, rank]:.4f}), Labels: {gallery_data['text_labels'][top_idx]}")
            
            # Visualize top-1 match
            if args.visualize:
                top_idx = topk_idx[i, 0].item()
                visualize_retrieval(
                    query_data['paths'][i], 
                    gallery_data['paths'][top_idx], 
                    q_mod, 
                    g_mod
                )
    
    return metrics


def main(args):
    """Main evaluation function."""
    
    logger.info("=" * 50)
    logger.info("Label-based KNN Retrieval Baseline")
    logger.info("=" * 50)
    
    # Load metadata
    logger.info(f"Loading metadata from {args.metadata_path}")
    metadata_df = pd.read_parquet(args.metadata_path)
    logger.info(f"Loaded {len(metadata_df)} entries from metadata")
    
    # Determine modes to evaluate
    modes = ['s1_s1', 's2_s2', 's1_s2', 's2_s1'] if args.mode == 'all' else [args.mode]
    
    # Identify which modalities we need
    needed_modalities = set()
    for m in modes:
        q_mod, g_mod = m.split('_')
        needed_modalities.add(q_mod)
        needed_modalities.add(g_mod)
    
    # Load data for all needed modalities and splits
    data_dict = {}
    for mod in needed_modalities:
        data_dict[mod] = {}
        root = args.data_root_s1 if mod == 's1' else args.data_root_s2
        
        for split in ['test', 'validation']:
            logger.info(f"Loading {mod.upper()} {split} metadata...")
            data_dict[mod][split] = load_split_metadata(
                root, split, metadata_df, modality=mod
            )
            logger.info(f"  Found {len(data_dict[mod][split]['patch_ids'])} images")
    
    # Run evaluation for each mode
    results = {}
    for m in modes:
        results[m] = run_evaluation(args, m, data_dict)
    
    # Print summary
    if len(modes) > 1:
        logger.info("\n" + "="*50)
        logger.info("FINAL SUMMARY")
        logger.info("="*50)
        for m, m_results in results.items():
            logger.info(f"{m:6}: Precision={m_results['precision']:.4f}, Recall={m_results['recall']:.4f}, F1={m_results['f1']:.4f}")
        logger.info("="*50)
    
    return results


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Label-based KNN Retrieval Baseline')
    
    # Data
    parser.add_argument('--data_root_s1', type=str, required=True, 
                        help='Path to BigEarthNet-S1 folder')
    parser.add_argument('--data_root_s2', type=str, required=True, 
                        help='Path to BigEarthNet-S2 folder')
    parser.add_argument('--metadata_path', type=str, required=True, 
                        help='Path to serbia_metadata.parquet')
    
    # Mode
    parser.add_argument('--mode', type=str, required=True, 
                        choices=['s1_s1', 's2_s2', 's1_s2', 's2_s1', 'all'],
                        help='Evaluation mode: s1_s1, s2_s2, s1_s2, s2_s1, or all')
    
    # Retrieval
    parser.add_argument('--top_k', type=int, default=10, 
                        help='Number of top similar images to retrieve')
    parser.add_argument('--distance_metric', type=str, default='cosine',
                        choices=['cosine', 'euclidean', 'manhattan', 'hamming', 'jaccard'],
                        help='Distance metric for KNN')
    
    # Output
    parser.add_argument('--show_examples', type=int, default=3, 
                        help='Number of example retrievals to show')
    parser.add_argument('--visualize', action='store_true',
                        help='Visualize retrieval examples')
    
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args)
