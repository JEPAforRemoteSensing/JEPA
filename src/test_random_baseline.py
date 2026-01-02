"""
Random Baseline Testing Script

Test a randomly initialized model to establish baseline performance.
Uses the same evaluation protocol as test_ijepa.py for fair comparison.
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


def create_random_model(device, model_name='vit_base', patch_size=16, crop_size=224, in_chans=3, seed=42):
    """Create a randomly initialized encoder (no pretrained weights)."""
    
    # Set seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    encoder, embed_dim = build_encoder(
        model_name=model_name,
        img_size=crop_size,
        patch_size=patch_size,
        in_chans=in_chans,
    )
    
    # Reinitialize weights randomly (in case build_encoder uses any pretrained weights)
    def _init_weights(m):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
        elif isinstance(m, torch.nn.LayerNorm):
            torch.nn.init.ones_(m.weight)
            torch.nn.init.zeros_(m.bias)
        elif isinstance(m, torch.nn.Conv2d):
            torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
    
    encoder.apply(_init_weights)
    logger.info(f"Created randomly initialized {model_name} encoder (seed={seed})")
    
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


def compute_similarity(query_embeddings, gallery_embeddings):
    """Compute cosine similarity between query and gallery embeddings."""
    similarity = torch.mm(query_embeddings, gallery_embeddings.t())
    return similarity


def get_topk_indices(similarity, k=10):
    """Get top-k most similar gallery indices for each query."""
    topk_sim, topk_idx = torch.topk(similarity, k=k, dim=1, largest=True)
    return topk_idx, topk_sim


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


def compute_f1_score(query_labels, predicted_labels):
    """Compute F1 score for multi-label classification."""
    from sklearn.metrics import f1_score, precision_score, recall_score
    
    # Convert to binary predictions (threshold at 0.5 for votes)
    pred_binary = (predicted_labels >= 0.5).astype(int)
    
    f1_micro = f1_score(query_labels, pred_binary, average='micro', zero_division=0)
    f1_macro = f1_score(query_labels, pred_binary, average='macro', zero_division=0)
    f1_samples = f1_score(query_labels, pred_binary, average='samples', zero_division=0)
    
    precision_micro = precision_score(query_labels, pred_binary, average='micro', zero_division=0)
    recall_micro = recall_score(query_labels, pred_binary, average='micro', zero_division=0)
    
    return {
        'f1_micro': f1_micro,
        'f1_macro': f1_macro,
        'f1_samples': f1_samples,
        'precision_micro': precision_micro,
        'recall_micro': recall_micro,
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
    
    # Create random model
    logger.info("Creating randomly initialized model...")
    encoder = create_random_model(
        device=device,
        model_name=args.model_name,
        patch_size=args.patch_size,
        crop_size=args.crop_size,
        in_chans=args.in_chans,
        seed=args.seed,
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
    
    # Compute similarity
    logger.info("Computing similarity matrix...")
    similarity = compute_similarity(test_embeddings, val_embeddings)
    
    # Get top-k indices
    logger.info(f"Finding top-{args.top_k} similar images...")
    topk_idx, topk_sim = get_topk_indices(similarity, k=args.top_k)
    
    # Load labels
    logger.info(f"Loading labels from {args.metadata_path}")
    test_labels = load_labels(args.metadata_path, test_patch_ids)
    val_labels = load_labels(args.metadata_path, val_patch_ids)
    
    logger.info(f"Test labels shape: {test_labels.shape}")
    logger.info(f"Validation labels shape: {val_labels.shape}")
    
    # Compute predicted labels based on top-k retrieval
    logger.info("Computing predicted labels from top-k retrieval...")
    predicted_labels = np.zeros_like(test_labels, dtype=float)
    
    for i in range(len(test_patch_ids)):
        topk_indices = topk_idx[i].numpy()
        topk_labels = val_labels[topk_indices]
        predicted_labels[i] = topk_labels.mean(axis=0)
        
    # Compute F1 scores
    logger.info("Computing F1 scores...")
    metrics = compute_f1_score(test_labels, predicted_labels)
    
    logger.info("=" * 50)
    logger.info("RANDOM BASELINE RESULTS")
    logger.info("=" * 50)
    logger.info(f"Model: {args.model_name} (randomly initialized, seed={args.seed})")
    logger.info(f"Top-K: {args.top_k}")
    logger.info(f"F1 Score (Micro): {metrics['f1_micro']:.4f}")
    logger.info(f"F1 Score (Macro): {metrics['f1_macro']:.4f}")
    logger.info(f"F1 Score (Samples): {metrics['f1_samples']:.4f}")
    logger.info(f"Precision (Micro): {metrics['precision_micro']:.4f}")
    logger.info(f"Recall (Micro): {metrics['recall_micro']:.4f}")
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
            
            for j, idx in enumerate(topk_idx[i][:5].numpy()):
                retrieved_pid = val_patch_ids[idx]
                retrieved_labels = patch_to_labels_text.get(retrieved_pid, [])
                sim = topk_sim[i][j].item()
                logger.info(f"    {j+1}. {retrieved_pid} (sim={sim:.4f})")
                logger.info(f"       Labels: {retrieved_labels}")
    
    return metrics


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Random Baseline Testing')
    
    # Model
    parser.add_argument('--model_name', type=str, default='vit_base',
                        choices=['vit_tiny', 'vit_small', 'vit_base', 'vit_large', 'vit_huge', 'vit_giant'])
    parser.add_argument('--patch_size', type=int, default=16)
    parser.add_argument('--in_chans', type=int, default=3)
    parser.add_argument('--crop_size', type=int, default=224)
    parser.add_argument('--seed', type=int, default=42, help='Random seed for model initialization')
    
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
