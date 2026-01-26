import torch
import argparse
from transforms import make_transforms_test
from masks import EvalCollator
from data_loading import MultiChannelDataset
import logging
import sys
from models import MEMPJepa
from torch.amp import autocast

# Logging setup
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)


def load_checkpoint(checkpoint_path, model, device='cpu'):
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    model.load_state_dict(checkpoint['model'])
    logger.info("Loaded model weights")    


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='I-JEPA Training')
    
    # Data preprocessing
    parser.add_argument('--in_chans1', type=int, default=2)
    parser.add_argument('--in_chans2', type=int, default=10)
    parser.add_argument('--data_root1', type=str, default='data/BEN_14k/BigEarthNet-S1')
    parser.add_argument('--data_root2', type=str, default='data/BEN_14k/BigEarthNet-S2')
    parser.add_argument('--metadata', type=str, default='data/BEN_14k/serbia_metadata.parquet')
    parser.add_argument('--checkpoint', type=str)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--patch_size', type=int, default=16)
    parser.add_argument('--crop_size', type=int, default=96)
    
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--use_amp', action='store_true')    
    parser.add_argument('--top_k', type=int, default=5)
    
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
        # Enable TF32 for Ampere+ GPUs (3.5x speedup on matrix ops)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True  # Auto-tune kernels
        logger.info("Enabled TF32 and cuDNN benchmarking")

    # Data preparation
    eval_collate_fn = EvalCollator()
    test_transform = make_transforms_test(12, args.crop_size)
    test_dataset = MultiChannelDataset(args.data_root1, args.data_root2, metadata=args.metadata, split='test', transform=test_transform)
    val_dataset = MultiChannelDataset(args.data_root1, args.data_root2, metadata=args.metadata, split='validation', transform=test_transform)
    test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=eval_collate_fn, pin_memory=True, drop_last=False, persistent_workers=True if args.num_workers > 0 else False, prefetch_factor=4 if args.num_workers > 0 else None)
    val_data_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=eval_collate_fn, pin_memory=True, drop_last=False, persistent_workers=True if args.num_workers > 0 else False, prefetch_factor=4 if args.num_workers > 0 else None)        

    # Eval
    model = MEMPJepa(
        in_chans1=args.in_chans1,
        in_chans2=args.in_chans2,
        patch_size=args.patch_size,
        img_size=args.crop_size
    )
    load_checkpoint(args.checkpoint, model, device)
    model = model.to(device)
    use_amp = args.use_amp and device.type == 'cuda'
    
    model.eval()
    val_embs1, val_embs2, val_idxs = [], [], []

    with torch.inference_mode():
        # Build retrieval index from validation set
        for images1, images2, idxs in val_data_loader:
            images1 = images1.to(device, non_blocking=True)
            images2 = images2.to(device, non_blocking=True)
            with autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                z1, z2 = model(images1, images2)
            val_embs1.append(z1.mean(dim=1))  # [B, D]
            val_embs2.append(z2.mean(dim=1))
            val_idxs.append(idxs)
        
        val_embs1 = torch.cat(val_embs1)
        val_embs2 = torch.cat(val_embs2)
        val_idxs = torch.cat(val_idxs)

        # Get labels for validation set
        val_meta = val_dataset.metadata[val_dataset.metadata['split'] == 'validation']
        test_meta = test_dataset.metadata[test_dataset.metadata['split'] == 'test']
        
        def get_val_labels(idx, modality):
            name = val_dataset.metadata1[idx] if modality == 1 else val_dataset.metadata2[idx]
            col = 's1_name' if modality == 1 else 'patch_id'
            return val_meta[val_meta[col] == name]['labels'].iloc[0]
        # s1->s1, s2->s2, s1->s2, s2->s1
        metrics = {k: {'prec': 0, 'rec': 0} for k in ['s1s1', 's2s2', 's1s2', 's2s1']}
        n = 0

        # Query with test set
        for images1, images2, q_idxs in test_data_loader:
            images1 = images1.to(device, non_blocking=True)
            images2 = images2.to(device, non_blocking=True)
            with autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                q1, q2 = model(images1, images2)
            q1 = q1.mean(dim=1)
            q2 = q2.mean(dim=1)

            # Cosine similarity and top-k for all 4 combinations
            # [B, k] tensor of indices
            topk_s1s1 = (q1 @ val_embs1.T).topk(args.top_k, dim=-1).indices
            topk_s2s2 = (q2 @ val_embs2.T).topk(args.top_k, dim=-1).indices
            topk_s1s2 = (q1 @ val_embs2.T).topk(args.top_k, dim=-1).indices
            topk_s2s1 = (q2 @ val_embs1.T).topk(args.top_k, dim=-1).indices

            for i in range(q1.shape[0]):
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

        # Compute F1 for all modes
        k = args.top_k
        f1s = {}
        for mode in metrics:
            p, r = metrics[mode]['prec']/(n*k), metrics[mode]['rec']/(n*k)
            f1s[mode] = 2*p*r/(p+r+1e-8)
            metrics[mode] = {'prec': p, 'rec': r, 'f1': f1s[mode]}
        
        logger.info(f"Eval - s1→s1: {f1s['s1s1']:.4f} | s2→s2: {f1s['s2s2']:.4f} | s1→s2: {f1s['s1s2']:.4f} | s2→s1: {f1s['s2s1']:.4f}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
