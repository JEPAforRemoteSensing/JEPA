import os
import sys
import torch
import logging
import argparse
from models import MMLeJEPA
from masks import EvalCollator
from torch.amp import autocast
from data_loading import MultiChannelDataset
from transforms import make_transforms_test

# Logging setup
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)

def load_checkpoint(checkpoint_path, model, device='cpu'):
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    # Load to CPU first to avoid MPS float64 issues
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model'])
    logger.info("Loaded model weights")
    return checkpoint.get('epoch', 0)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='LeJEPA Evaluation')
    
    # Data
    parser.add_argument('--in_chans1', type=int, default=2)
    parser.add_argument('--in_chans2', type=int, default=10)
    parser.add_argument('--data_root1', type=str, default='data/BEN_14k/BigEarthNet-S1')
    parser.add_argument('--data_root2', type=str, default='data/BEN_14k/BigEarthNet-S2')
    parser.add_argument('--metadata', type=str, default='data/BEN_14k/serbia_metadata.parquet')
    parser.add_argument('--num_workers', type=int, default=4)
    
    # Evaluation
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--use_amp', action='store_true')
    parser.add_argument('--top_k', type=int, default=5)
    parser.add_argument('--compile', action='store_true')
    
    return parser.parse_args()


def main(args):
    # Device setup
    if torch.backends.mps.is_available():
        device = torch.device('mps')
        logger.info("Using Apple MPS (Metal Performance Shaders)")
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
        logger.warning("GPU not available, using CPU")
    
    logger.info(f"Using device: {device}")

    if device.type == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        logger.info("Enabled TF32 and cuDNN benchmarking")

    # Prepare data
    test_transform = make_transforms_test(12, 120)
    test_collate_fn = EvalCollator()
    test_dataset = MultiChannelDataset(args.data_root1, args.data_root2, metadata=args.metadata, split='test', transform=test_transform)
    val_dataset = MultiChannelDataset(args.data_root1, args.data_root2, metadata=args.metadata, split='validation', transform=test_transform)
    
    test_data_loader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers, 
        collate_fn=test_collate_fn, 
        pin_memory=True, 
        drop_last=False, 
        persistent_workers=True if args.num_workers > 0 else False, 
        prefetch_factor=4 if args.num_workers > 0 else None
    )
    
    val_data_loader = torch.utils.data.DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers, 
        collate_fn=test_collate_fn, 
        pin_memory=True, 
        drop_last=False, 
        persistent_workers=True if args.num_workers > 0 else False, 
        prefetch_factor=4 if args.num_workers > 0 else None
    )

    # Initialize model
    model = MMLeJEPA(
        in_chans1=args.in_chans1,
        in_chans2=args.in_chans2,
    ).to(device)

    if args.compile:
        model = torch.compile(model, mode="reduce-overhead")
    
    # Load checkpoint
    epoch = load_checkpoint(args.checkpoint, model, device)
    logger.info(f"Evaluating checkpoint from epoch {epoch}")

    # Automatic Mixed Precision setup
    use_amp = args.use_amp and device.type == 'cuda'

    # Evaluation
    model.eval()
    val_embs1, val_embs2, val_idxs = [], [], []

    with torch.inference_mode():
        logger.info("Building retrieval index from validation set...")
        # Build retrieval index from validation set
        for images1, images2, idxs in val_data_loader:
            images1 = images1.to(device, non_blocking=True)
            images2 = images2.to(device, non_blocking=True)
            actual_bs = images1.shape[0]
            with autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                yhat = model(images1, images2)
                yhat_s1 = yhat[:actual_bs].clone()
                yhat_s2 = yhat[actual_bs:].clone()
            val_embs1.append(yhat_s1)
            val_embs2.append(yhat_s2)
            val_idxs.append(idxs)
        
        val_embs1 = torch.cat(val_embs1)
        val_embs2 = torch.cat(val_embs2)
        val_idxs = torch.cat(val_idxs)
        logger.info(f"Built index with {val_embs1.shape[0]} validation samples")

        # Get labels for validation set
        val_meta = val_dataset.metadata[val_dataset.metadata['split'] == 'validation']
        test_meta = test_dataset.metadata[test_dataset.metadata['split'] == 'test']
        
        def get_val_labels(idx, modality):
            name = val_dataset.metadata1[idx] if modality == 1 else val_dataset.metadata2[idx]
            col = 's1_name' if modality == 1 else 'patch_id'
            return val_meta[val_meta[col] == name]['labels'].iloc[0]

        # s1->s1, s2->s2, s1->s2, s2->s1
        metrics = {k: {'prec': 0, 'rec': 0} for k in ['s1s1', 's2s2', 's1s2', 's2s1']}
        
        # Classification metrics
        all_predictions = []
        all_labels = []
        num_classes = 19
        class_tp = torch.zeros(num_classes)
        class_fp = torch.zeros(num_classes)
        class_fn = torch.zeros(num_classes)
        
        n = 0

        logger.info("Querying with test set...")
        # Query with test set
        for images1, images2, q_idxs in test_data_loader:
            images1 = images1.to(device, non_blocking=True)
            images2 = images2.to(device, non_blocking=True)
            actual_bs = images1.shape[0]
            with autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                qhat = model(images1, images2)
                qhat_s1 = qhat[:actual_bs].clone()
                qhat_s2 = qhat[actual_bs:].clone()

            # Get predictions and ground truth for classification metrics
            preds = (qhat_s1.sigmoid() > 0.5).cpu()
            for idx in q_idxs:
                gt_labels = torch.tensor(test_meta[test_meta['s1_name'] == test_dataset.metadata1[idx.item()]]['one_hot_labels'].iloc[0], dtype=torch.float32)
                all_labels.append(gt_labels)
            all_predictions.append(preds)

            # Cosine similarity and top-k for all 4 combinations
            # [B, k] tensor of indices
            topk_s1s1 = (qhat_s1 @ val_embs1.T).topk(args.top_k, dim=-1).indices
            topk_s2s2 = (qhat_s2 @ val_embs2.T).topk(args.top_k, dim=-1).indices
            topk_s1s2 = (qhat_s1 @ val_embs2.T).topk(args.top_k, dim=-1).indices
            topk_s2s1 = (qhat_s2 @ val_embs1.T).topk(args.top_k, dim=-1).indices

            for i in range(qhat_s1.shape[0]):
                q_idx = q_idxs[i].item()
                L_q = set(test_meta[test_meta['s1_name'] == test_dataset.metadata1[q_idx]]['labels'].iloc[0])

                for mode, topk, ret_mod in [('s1s1', topk_s1s1, 1), ('s2s2', topk_s2s2, 2), 
                                             ('s1s2', topk_s1s2, 2), ('s2s1', topk_s2s1, 1)]:
                    for k in range(args.top_k):
                        r_idx = val_idxs[topk[i, k]].item()
                        L_r = set(get_val_labels(r_idx, ret_mod))
                        if L_r: metrics[mode]['prec'] += len(L_q & L_r) / len(L_r)
                        if L_q: metrics[mode]['rec'] += len(L_q & L_r) / len(L_q)
                n += 1

        # Compute retrieval metrics
        k = args.top_k
        f1s = {}
        logger.info(f"\n{'='*70}")
        logger.info(f"RETRIEVAL METRICS (top-{k})")
        logger.info("=" * 70)
        for mode in metrics:
            p, r = metrics[mode]['prec']/(n*k), metrics[mode]['rec']/(n*k)
            f1s[mode] = 2*p*r/(p+r+1e-8)
            metrics[mode] = {'prec': p, 'rec': r, 'f1': f1s[mode]}
            logger.info(f"{mode:6s} | Precision: {p:.4f} | Recall: {r:.4f} | F1: {f1s[mode]:.4f}")
        
        logger.info("=" * 70)
        logger.info(f"Summary: s1→s1: {f1s['s1s1']:.4f} | s2→s2: {f1s['s2s2']:.4f} | s1→s2: {f1s['s1s2']:.4f} | s2→s1: {f1s['s2s1']:.4f}")
        logger.info(f"Average Retrieval F1: {sum(f1s.values())/len(f1s):.4f}")

        # Compute classification metrics
        all_predictions = torch.cat(all_predictions, dim=0)
        all_labels = torch.stack(all_labels, dim=0)
        
        # Exact match accuracy (all labels must match)
        exact_match = (all_predictions == all_labels).all(dim=1).float().mean().item()
        
        # Hamming accuracy (per-label accuracy)
        hamming_acc = (all_predictions == all_labels).float().mean().item()
        
        # Per-class metrics
        for c in range(num_classes):
            pred_c = all_predictions[:, c]
            label_c = all_labels[:, c]
            class_tp[c] = ((pred_c == 1) & (label_c == 1)).sum()
            class_fp[c] = ((pred_c == 1) & (label_c == 0)).sum()
            class_fn[c] = ((pred_c == 0) & (label_c == 1)).sum()
        
        class_precision = class_tp / (class_tp + class_fp + 1e-8)
        class_recall = class_tp / (class_tp + class_fn + 1e-8)
        class_f1 = 2 * class_precision * class_recall / (class_precision + class_recall + 1e-8)
        
        # Macro and micro averaged metrics
        macro_precision = class_precision.mean().item()
        macro_recall = class_recall.mean().item()
        macro_f1 = class_f1.mean().item()
        
        micro_precision = class_tp.sum() / (class_tp.sum() + class_fp.sum() + 1e-8)
        micro_recall = class_tp.sum() / (class_tp.sum() + class_fn.sum() + 1e-8)
        micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall + 1e-8)
        
        logger.info(f"\n{'='*70}")
        logger.info("CLASSIFICATION METRICS")
        logger.info("=" * 70)
        logger.info(f"Exact Match Accuracy:  {exact_match:.4f}")
        logger.info(f"Hamming Accuracy:      {hamming_acc:.4f}")
        logger.info(f"Macro Precision:       {macro_precision:.4f}")
        logger.info(f"Macro Recall:          {macro_recall:.4f}")
        logger.info(f"Macro F1:              {macro_f1:.4f}")
        logger.info(f"Micro Precision:       {micro_precision:.4f}")
        logger.info(f"Micro Recall:          {micro_recall:.4f}")
        logger.info(f"Micro F1:              {micro_f1:.4f}")
        
        # Label names (BigEarthNet-19 class names)
        label_names = [
            'Urban fabric', 'Industrial or commercial units', 'Arable land',
            'Permanent crops', 'Pastures', 'Complex cultivation patterns',
            'Land principally occupied by agriculture',
            'Agro-forestry areas', 'Broad-leaved forest', 'Coniferous forest',
            'Mixed forest', 'Natural grassland and sparsely vegetated areas',
            'Moors, heathland and sclerophyllous vegetation',
            'Transitional woodland, shrub', 'Beaches, dunes, sands',
            'Inland wetlands', 'Coastal wetlands', 'Inland waters',
            'Marine waters'
        ]
        
        logger.info(f"\n{'='*70}")
        logger.info("PER-CLASS METRICS")
        logger.info("=" * 70)
        logger.info(f"{'Class':<45} {'Prec':<7} {'Rec':<7} {'F1':<7} {'Supp':<6}")
        logger.info("-" * 70)
        for c in range(num_classes):
            support = (all_labels[:, c] == 1).sum().item()
            logger.info(f"{label_names[c]:<45} {class_precision[c]:.4f}  {class_recall[c]:.4f}  {class_f1[c]:.4f}  {support:>5}")
        logger.info("=" * 70)


if __name__ == "__main__":
    args = parse_args()
    main(args)
