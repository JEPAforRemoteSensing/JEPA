"""
I-JEPA V2 Testing Script

Test the trained I-JEPA v2 model (dual encoder) by:
1. Encoding query images from test set
2. Finding top-k similar images from validation set
3. Computing F1 score based on multi-label classification using one-hot labels

Supports:
- Unimodal testing (S1 only or S2 only)
- Cross-modal testing (query from one modality, gallery from another)
- Fused testing (concatenated S1+S2 embeddings)
"""

import os
import sys
import argparse
import logging
from tqdm import tqdm

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import tifffile

from encoder import build_encoder, get_num_patches
from transforms import make_transforms_rgb, make_transforms

# Logging setup
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)


class EmbeddingDatasetS1(torch.utils.data.Dataset):
    """Dataset for extracting embeddings from S1 (2-channel) images."""
    
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
        # S1 has 2 channels, no conversion needed
        if self.transform:
            img = self.transform(img)
        
        # Return image and patch_id (filename without .tif extension)
        patch_id = filename.replace('.tif', '')
        return img, patch_id
    
    def __len__(self):
        return len(self.metadata)


class EmbeddingDatasetS2(torch.utils.data.Dataset):
    """Dataset for extracting embeddings from S2 (RGB) images."""
    
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


class PairedEmbeddingDataset(torch.utils.data.Dataset):
    """Dataset for extracting paired S1+S2 embeddings (same patch_id in both modalities)."""
    
    def __init__(self, root_s1, root_s2, split, transform_s1=None, transform_s2=None):
        self.data_path_s1 = os.path.join(root_s1, split)
        self.data_path_s2 = os.path.join(root_s2, split)
        self.transform_s1 = transform_s1
        self.transform_s2 = transform_s2
        
        # Find common patch_ids between S1 and S2
        s1_files = {f.name.replace('.tif', '') for f in os.scandir(self.data_path_s1) if f.name.endswith('.tif')}
        s2_files = {f.name.replace('.tif', '') for f in os.scandir(self.data_path_s2) if f.name.endswith('.tif')}
        
        self.common_ids = sorted(list(s1_files & s2_files))
        logger.info(f"Found {len(self.common_ids)} common patch_ids in {split} split")
    
    def __getitem__(self, idx):
        patch_id = self.common_ids[idx]
        filename = f"{patch_id}.tif"
        
        # Load S1
        img_s1 = torch.from_numpy(tifffile.imread(os.path.join(self.data_path_s1, filename))).float()
        if self.transform_s1:
            img_s1 = self.transform_s1(img_s1)
        
        # Load S2
        img_s2 = torch.from_numpy(tifffile.imread(os.path.join(self.data_path_s2, filename))).float()
        img_s2 = img_s2[[2, 1, 0], :, :]  # BGR to RGB
        if self.transform_s2:
            img_s2 = self.transform_s2(img_s2)
        
        return img_s1, img_s2, patch_id
    
    def __len__(self):
        return len(self.common_ids)


def load_model_v2(checkpoint_path, device, model_name='vit_base', patch_size=16, crop_size=224, 
                  in_chans1=2, in_chans2=3, pred_emb_dim=384):
    """Load the trained encoders and probe from checkpoint."""
    
    # Build encoder1 (S1)
    encoder1, embed_dim1 = build_encoder(
        model_name=model_name,
        img_size=crop_size,
        patch_size=patch_size,
        in_chans=in_chans1,
    )
    
    # Build encoder2 (S2)
    encoder2, embed_dim2 = build_encoder(
        model_name=model_name,
        img_size=crop_size,
        patch_size=patch_size,
        in_chans=in_chans2,
    )
    
    # Build probe
    probe = nn.Sequential(
        nn.Linear(embed_dim1 + embed_dim2, 2048),
        nn.LayerNorm(2048),
        nn.ReLU(inplace=True),
        nn.Linear(2048, 2048),
        nn.LayerNorm(2048),
        nn.ReLU(inplace=True),
        nn.Linear(2048, pred_emb_dim)
    )
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Load encoder1
    if 'encoder1' in checkpoint:
        encoder1.load_state_dict(checkpoint['encoder1'])
        logger.info("Loaded encoder1 from checkpoint")
    else:
        raise ValueError("No encoder1 found in checkpoint")
    
    # Load encoder2
    if 'encoder2' in checkpoint:
        encoder2.load_state_dict(checkpoint['encoder2'])
        logger.info("Loaded encoder2 from checkpoint")
    else:
        raise ValueError("No encoder2 found in checkpoint")
    
    # Load probe
    if 'probe' in checkpoint:
        probe.load_state_dict(checkpoint['probe'])
        logger.info("Loaded probe from checkpoint")
    else:
        logger.warning("No probe found in checkpoint, using random initialization")
    
    encoder1 = encoder1.to(device)
    encoder2 = encoder2.to(device)
    probe = probe.to(device)
    
    encoder1.eval()
    encoder2.eval()
    probe.eval()
    
    return encoder1, encoder2, probe, embed_dim1, embed_dim2


def extract_embeddings_single(encoder, data_loader, device):
    """Extract embeddings for a single modality."""
    
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


def extract_embeddings_paired(encoder1, encoder2, data_loader, device, use_probe=False, probe=None):
    """Extract embeddings for paired S1+S2 images."""
    
    embeddings = []
    patch_ids = []
    
    with torch.no_grad():
        for images_s1, images_s2, pids in tqdm(data_loader, desc="Extracting paired embeddings"):
            images_s1 = images_s1.to(device, non_blocking=True)
            images_s2 = images_s2.to(device, non_blocking=True)
            
            # Get encoder outputs
            h1 = encoder1(images_s1)  # (B, num_patches, embed_dim1)
            h2 = encoder2(images_s2)  # (B, num_patches, embed_dim2)
            
            # Global average pooling
            h1 = h1.mean(dim=1)  # (B, embed_dim1)
            h2 = h2.mean(dim=1)  # (B, embed_dim2)
            
            if use_probe and probe is not None:
                # Use probe for fused representation
                h = probe(torch.cat([h1, h2], dim=-1))
            else:
                # Concatenate embeddings
                h = torch.cat([h1, h2], dim=-1)
            
            # L2 normalize for cosine similarity
            h = F.normalize(h, p=2, dim=-1)
            
            embeddings.append(h.cpu())
            patch_ids.extend(pids)
    
    embeddings = torch.cat(embeddings, dim=0)
    return embeddings, patch_ids


def build_knn_index(gallery_embeddings, k=10, metric='cosine'):
    """Build a KNN index using scikit-learn's NearestNeighbors."""
    from sklearn.neighbors import NearestNeighbors
    
    knn = NearestNeighbors(n_neighbors=k, metric=metric, algorithm='auto')
    knn.fit(gallery_embeddings.numpy())
    return knn


def get_topk_indices_knn(knn, query_embeddings, k=10):
    """Get top-k most similar gallery indices for each query using KNN."""
    distances, indices = knn.kneighbors(query_embeddings.numpy(), n_neighbors=k)
    
    # Convert distances to similarity scores (for cosine: similarity = 1 - distance)
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


def run_unimodal_test(args, encoder, data_root, transform, device, modality_name):
    """Run unimodal retrieval test (S1 or S2 only)."""
    
    logger.info(f"\n{'='*50}")
    logger.info(f"UNIMODAL TEST: {modality_name}")
    logger.info(f"{'='*50}")
    
    # Create datasets
    if modality_name == 'S1':
        DatasetClass = EmbeddingDatasetS1
    else:
        DatasetClass = EmbeddingDatasetS2
    
    test_dataset = DatasetClass(root=data_root, split='test', transform=transform)
    val_dataset = DatasetClass(root=data_root, split='validation', transform=transform)
    
    logger.info(f"Test dataset: {len(test_dataset)} images")
    logger.info(f"Validation dataset: {len(val_dataset)} images")
    
    # Create data loaders
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )
    
    # Extract embeddings
    test_embeddings, test_patch_ids = extract_embeddings_single(encoder, test_loader, device)
    val_embeddings, val_patch_ids = extract_embeddings_single(encoder, val_loader, device)
    
    logger.info(f"Test embeddings shape: {test_embeddings.shape}")
    logger.info(f"Validation embeddings shape: {val_embeddings.shape}")
    
    # Build KNN index and get top-k
    knn = build_knn_index(val_embeddings, k=args.top_k, metric='cosine')
    topk_idx, topk_sim = get_topk_indices_knn(knn, test_embeddings, k=args.top_k)
    
    # Load labels and compute predictions
    test_labels = load_labels(args.metadata_path, test_patch_ids)
    val_labels = load_labels(args.metadata_path, val_patch_ids)
    
    predicted_labels = np.zeros_like(test_labels, dtype=float)
    for i in range(len(test_patch_ids)):
        topk_indices = topk_idx[i].numpy()
        topk_labels = val_labels[topk_indices]
        predicted_labels[i] = topk_labels.mean(axis=0)
    
    # Compute F1 scores
    metrics = compute_f1_score(test_labels, predicted_labels)
    
    logger.info(f"Results for {modality_name}:")
    logger.info(f"  F1 Score (Micro): {metrics['f1_micro']:.4f}")
    logger.info(f"  F1 Score (Macro): {metrics['f1_macro']:.4f}")
    logger.info(f"  F1 Score (Samples): {metrics['f1_samples']:.4f}")
    logger.info(f"  Precision (Micro): {metrics['precision_micro']:.4f}")
    logger.info(f"  Recall (Micro): {metrics['recall_micro']:.4f}")
    
    return metrics


def run_cross_modal_test(args, encoder_query, encoder_gallery, 
                         data_root_query, data_root_gallery,
                         transform_query, transform_gallery,
                         device, query_modality, gallery_modality):
    """Run cross-modal retrieval test (query from one modality, gallery from another)."""
    
    logger.info(f"\n{'='*50}")
    logger.info(f"CROSS-MODAL TEST: {query_modality} -> {gallery_modality}")
    logger.info(f"{'='*50}")
    
    # Create datasets
    if query_modality == 'S1':
        QueryDatasetClass = EmbeddingDatasetS1
    else:
        QueryDatasetClass = EmbeddingDatasetS2
    
    if gallery_modality == 'S1':
        GalleryDatasetClass = EmbeddingDatasetS1
    else:
        GalleryDatasetClass = EmbeddingDatasetS2
    
    test_dataset = QueryDatasetClass(root=data_root_query, split='test', transform=transform_query)
    val_dataset = GalleryDatasetClass(root=data_root_gallery, split='validation', transform=transform_gallery)
    
    logger.info(f"Query (test) dataset: {len(test_dataset)} images from {query_modality}")
    logger.info(f"Gallery (val) dataset: {len(val_dataset)} images from {gallery_modality}")
    
    # Create data loaders
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )
    
    # Extract embeddings
    test_embeddings, test_patch_ids = extract_embeddings_single(encoder_query, test_loader, device)
    val_embeddings, val_patch_ids = extract_embeddings_single(encoder_gallery, val_loader, device)
    
    logger.info(f"Query embeddings shape: {test_embeddings.shape}")
    logger.info(f"Gallery embeddings shape: {val_embeddings.shape}")
    
    # Build KNN index and get top-k
    knn = build_knn_index(val_embeddings, k=args.top_k, metric='cosine')
    topk_idx, topk_sim = get_topk_indices_knn(knn, test_embeddings, k=args.top_k)
    
    # Load labels and compute predictions
    test_labels = load_labels(args.metadata_path, test_patch_ids)
    val_labels = load_labels(args.metadata_path, val_patch_ids)
    
    predicted_labels = np.zeros_like(test_labels, dtype=float)
    for i in range(len(test_patch_ids)):
        topk_indices = topk_idx[i].numpy()
        topk_labels = val_labels[topk_indices]
        predicted_labels[i] = topk_labels.mean(axis=0)
    
    # Compute F1 scores
    metrics = compute_f1_score(test_labels, predicted_labels)
    
    logger.info(f"Results for {query_modality} -> {gallery_modality}:")
    logger.info(f"  F1 Score (Micro): {metrics['f1_micro']:.4f}")
    logger.info(f"  F1 Score (Macro): {metrics['f1_macro']:.4f}")
    logger.info(f"  F1 Score (Samples): {metrics['f1_samples']:.4f}")
    logger.info(f"  Precision (Micro): {metrics['precision_micro']:.4f}")
    logger.info(f"  Recall (Micro): {metrics['recall_micro']:.4f}")
    
    return metrics


def run_fused_test(args, encoder1, encoder2, probe, device, use_probe=False):
    """Run fused (S1+S2) retrieval test."""
    
    mode_name = "FUSED (with probe)" if use_probe else "FUSED (concatenated)"
    logger.info(f"\n{'='*50}")
    logger.info(f"{mode_name} TEST")
    logger.info(f"{'='*50}")
    
    # Create transforms
    transform_s1 = make_transforms(num_channels=2)
    transform_s2 = make_transforms_rgb(num_channels=3)
    
    # Create paired datasets
    test_dataset = PairedEmbeddingDataset(
        root_s1=args.data_root_s1, root_s2=args.data_root_s2,
        split='test', transform_s1=transform_s1, transform_s2=transform_s2,
    )
    val_dataset = PairedEmbeddingDataset(
        root_s1=args.data_root_s1, root_s2=args.data_root_s2,
        split='validation', transform_s1=transform_s1, transform_s2=transform_s2,
    )
    
    logger.info(f"Test dataset: {len(test_dataset)} paired images")
    logger.info(f"Validation dataset: {len(val_dataset)} paired images")
    
    # Create data loaders
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )
    
    # Extract embeddings
    test_embeddings, test_patch_ids = extract_embeddings_paired(
        encoder1, encoder2, test_loader, device, use_probe=use_probe, probe=probe
    )
    val_embeddings, val_patch_ids = extract_embeddings_paired(
        encoder1, encoder2, val_loader, device, use_probe=use_probe, probe=probe
    )
    
    logger.info(f"Test embeddings shape: {test_embeddings.shape}")
    logger.info(f"Validation embeddings shape: {val_embeddings.shape}")
    
    # Build KNN index and get top-k
    knn = build_knn_index(val_embeddings, k=args.top_k, metric='cosine')
    topk_idx, topk_sim = get_topk_indices_knn(knn, test_embeddings, k=args.top_k)
    
    # Load labels and compute predictions
    test_labels = load_labels(args.metadata_path, test_patch_ids)
    val_labels = load_labels(args.metadata_path, val_patch_ids)
    
    predicted_labels = np.zeros_like(test_labels, dtype=float)
    for i in range(len(test_patch_ids)):
        topk_indices = topk_idx[i].numpy()
        topk_labels = val_labels[topk_indices]
        predicted_labels[i] = topk_labels.mean(axis=0)
    
    # Compute F1 scores
    metrics = compute_f1_score(test_labels, predicted_labels)
    
    logger.info(f"Results for {mode_name}:")
    logger.info(f"  F1 Score (Micro): {metrics['f1_micro']:.4f}")
    logger.info(f"  F1 Score (Macro): {metrics['f1_macro']:.4f}")
    logger.info(f"  F1 Score (Samples): {metrics['f1_samples']:.4f}")
    logger.info(f"  Precision (Micro): {metrics['precision_micro']:.4f}")
    logger.info(f"  Recall (Micro): {metrics['recall_micro']:.4f}")
    
    return metrics


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
    
    # Load models
    logger.info(f"Loading models from {args.checkpoint}")
    encoder1, encoder2, probe, embed_dim1, embed_dim2 = load_model_v2(
        checkpoint_path=args.checkpoint,
        device=device,
        model_name=args.model_name,
        patch_size=args.patch_size,
        crop_size=args.crop_size,
        in_chans1=args.in_chans1,
        in_chans2=args.in_chans2,
        pred_emb_dim=args.pred_emb_dim,
    )
    
    # Create transforms
    transform_s1 = make_transforms(num_channels=2)
    transform_s2 = make_transforms_rgb(num_channels=3)
    
    all_results = {}
    
    # Run requested tests
    if args.test_mode in ['all', 'unimodal', 's1']:
        # Unimodal S1 test
        metrics_s1 = run_unimodal_test(
            args, encoder1, args.data_root_s1, transform_s1, device, 'S1'
        )
        all_results['unimodal_s1'] = metrics_s1
    
    if args.test_mode in ['all', 'unimodal', 's2']:
        # Unimodal S2 test
        metrics_s2 = run_unimodal_test(
            args, encoder2, args.data_root_s2, transform_s2, device, 'S2'
        )
        all_results['unimodal_s2'] = metrics_s2
    
    if args.test_mode in ['all', 'cross_modal', 's1_to_s2']:
        # Cross-modal S1 -> S2 test
        metrics_s1_to_s2 = run_cross_modal_test(
            args, encoder1, encoder2,
            args.data_root_s1, args.data_root_s2,
            transform_s1, transform_s2,
            device, 'S1', 'S2'
        )
        all_results['cross_modal_s1_to_s2'] = metrics_s1_to_s2
    
    if args.test_mode in ['all', 'cross_modal', 's2_to_s1']:
        # Cross-modal S2 -> S1 test
        metrics_s2_to_s1 = run_cross_modal_test(
            args, encoder2, encoder1,
            args.data_root_s2, args.data_root_s1,
            transform_s2, transform_s1,
            device, 'S2', 'S1'
        )
        all_results['cross_modal_s2_to_s1'] = metrics_s2_to_s1
    
    if args.test_mode in ['all', 'fused']:
        # Fused (concatenated) test
        metrics_fused_concat = run_fused_test(
            args, encoder1, encoder2, probe, device, use_probe=False
        )
        all_results['fused_concat'] = metrics_fused_concat
        
        # Fused (with probe) test
        metrics_fused_probe = run_fused_test(
            args, encoder1, encoder2, probe, device, use_probe=True
        )
        all_results['fused_probe'] = metrics_fused_probe
    
    # Print summary
    logger.info("\n" + "="*70)
    logger.info("SUMMARY OF ALL RESULTS")
    logger.info("="*70)
    logger.info(f"{'Test Mode':<25} {'F1 Micro':<12} {'F1 Macro':<12} {'F1 Samples':<12}")
    logger.info("-"*70)
    for test_name, metrics in all_results.items():
        logger.info(f"{test_name:<25} {metrics['f1_micro']:.4f}       {metrics['f1_macro']:.4f}       {metrics['f1_samples']:.4f}")
    logger.info("="*70)
    
    return all_results


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='I-JEPA V2 Testing (Dual Encoder)')
    
    # Model
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--model_name', type=str, default='vit_base',
                        choices=['vit_tiny', 'vit_small', 'vit_base', 'vit_large', 'vit_huge', 'vit_giant'])
    parser.add_argument('--patch_size', type=int, default=16)
    parser.add_argument('--in_chans1', type=int, default=2, help='Number of input channels for S1 encoder')
    parser.add_argument('--in_chans2', type=int, default=3, help='Number of input channels for S2 encoder')
    parser.add_argument('--pred_emb_dim', type=int, default=384, help='Probe output dimension')
    parser.add_argument('--crop_size', type=int, default=224)
    
    # Data
    parser.add_argument('--data_root_s1', type=str, default='data/BEN_14k/BigEarthNet-S1',
                        help='Path to BigEarthNet-S1 folder')
    parser.add_argument('--data_root_s2', type=str, default='data/BEN_14k/BigEarthNet-S2',
                        help='Path to BigEarthNet-S2 folder')
    parser.add_argument('--metadata_path', type=str, required=True, help='Path to serbia_metadata.parquet')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=4)
    
    # Retrieval
    parser.add_argument('--top_k', type=int, default=10, help='Number of top similar images to retrieve')
    
    # Test mode
    parser.add_argument('--test_mode', type=str, default='all',
                        choices=['all', 'unimodal', 'cross_modal', 'fused', 
                                 's1', 's2', 's1_to_s2', 's2_to_s1'],
                        help='Which tests to run')
    
    # Output
    parser.add_argument('--show_examples', type=int, default=0, help='Number of example retrievals to show')
    
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args)
