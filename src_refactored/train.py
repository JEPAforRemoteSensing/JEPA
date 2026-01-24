import os
import sys
import time
import torch
import logging
import argparse
import numpy as np
import wandb

from data_loading import MultiChannelDataset
from transforms import make_transforms, make_transforms_test
from masks import RandomMaskCollator, EvalCollator
from models import MEMPJepa
from losses import MEMPLoss
from torch.amp import GradScaler, autocast

# Logging setup
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)

# Global seed
_GLOBAL_SEED = 0
np.random.seed(_GLOBAL_SEED)
torch.manual_seed(_GLOBAL_SEED)

def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None, device='cpu'):
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    model.load_state_dict(checkpoint['model'])
    logger.info("Loaded model weights")
    
    if optimizer is not None and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
        logger.info("Loaded optimizer state")
    
    if scheduler is not None and 'scheduler' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler'])
        logger.info("Loaded scheduler state")
    
    start_epoch = checkpoint.get('epoch', 0) + 1
    loss = checkpoint.get('loss', None)
    
    logger.info(f"Resuming from epoch {start_epoch}, previous loss: {loss}")
    
    return start_epoch

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='I-JEPA Training')
    
    # Data preprocessing
    parser.add_argument('--in_chans1', type=int, default=2)
    parser.add_argument('--in_chans2', type=int, default=10)
    parser.add_argument('--data_root1', type=str, default='data/BEN_14k/BigEarthNet-S1')
    parser.add_argument('--data_root2', type=str, default='data/BEN_14k/BigEarthNet-S2')
    parser.add_argument('--metadata', type=str, default='data/BEN_14k/serbia_metadata.parquet')
    parser.add_argument('--num_workers', type=int, default=4)
    
    # Training
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--warmup_epochs', type=int, default=10)
    parser.add_argument('--use_amp', action='store_true')
    parser.add_argument('--lamb', type=float, default=0.02, help='Weight for SIGReg loss')
    parser.add_argument('--gamma', type=float, default=1.0, help='Weight for invariance loss')
    
    # Logging/Saving
    parser.add_argument('--output_dir', type=str, default='./checkpoints')
    parser.add_argument('--log_freq', type=int, default=10)
    parser.add_argument('--save_freq', type=int, default=20)
    parser.add_argument('--eval_freq', type=int, default=20)
    parser.add_argument('--top_k', type=int, default=5)
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume training from')
    
    # Weights & Biases
    parser.add_argument('--wandb_enabled', action='store_true', help='Enable wandb logging')
    parser.add_argument('--wandb_project', type=str, default='ijepa', help='Wandb project name')
    parser.add_argument('--wandb_run_name', type=str, default=None, help='Wandb run name')
    
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

    # Initialize wandb
    wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name,
        config=vars(args),
        mode="online" if args.wandb_enabled else "disabled",
    )

    # Prepare training data
    transform = make_transforms(12)
    train_dataset = MultiChannelDataset(args.data_root1, args.data_root2, metadata=args.metadata, split='train', transform=transform)
    
    collate_fn = RandomMaskCollator()
    data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True, drop_last=True, persistent_workers=True if args.num_workers > 0 else False, prefetch_factor=4 if args.num_workers > 0 else None)
    
    # Prepare validation data
    eval_collate_fn = EvalCollator()
    test_transform = make_transforms_test(12)
    test_dataset = MultiChannelDataset(args.data_root1, args.data_root2, metadata=args.metadata, split='test', transform=test_transform)
    val_dataset = MultiChannelDataset(args.data_root1, args.data_root2, metadata=args.metadata, split='validation', transform=test_transform)
    test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=eval_collate_fn, pin_memory=True, drop_last=False, persistent_workers=True if args.num_workers > 0 else False, prefetch_factor=4 if args.num_workers > 0 else None)
    val_data_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=eval_collate_fn, pin_memory=True, drop_last=False, persistent_workers=True if args.num_workers > 0 else False, prefetch_factor=4 if args.num_workers > 0 else None)

    iterations_per_epoch = len(data_loader)

    # Initialize model
    model = MEMPJepa(
        in_chans1=args.in_chans1,
        in_chans2=args.in_chans2,
    ).to(device)
    
    # Define loss, optimizer and scheduler
    warmup_iters = args.warmup_epochs * iterations_per_epoch
    total_iters = args.epochs * iterations_per_epoch
    
    def lr_lambda(current_iter):
        if current_iter < warmup_iters:
            return current_iter / warmup_iters
        else:
            progress = (current_iter - warmup_iters) / (total_iters - warmup_iters)
            return 0.5 * (1 + np.cos(np.pi * progress))
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    loss_fn = MEMPLoss().to(device)

    # Automatic Mixed Precision setup (disabled for MPS - not fully supported)
    use_amp = args.use_amp and device.type == 'cuda'
    scaler = GradScaler(device=device, enabled=use_amp)

    # Checkpoint loading
    start_epoch = 1
    if args.resume is not None:
        start_epoch = load_checkpoint(args.resume, model, optimizer, scheduler, device)
    
    # Training loop
    logger.info("Starting training...")
    for epoch in range(start_epoch, args.epochs + 1):

        model.train()
        epoch_start = time.time()
        for itr, (images1, images2, masks_enc, masks_pred) in enumerate(data_loader):

            # images1, images2: [B, C, H, W]
            images1 = images1.to(device, non_blocking=True)
            images2 = images2.to(device, non_blocking=True)
            # masks_enc: [B, num_keep]
            masks_enc = [m.to(device, non_blocking=True) for m in masks_enc]
            # masks_pred: [B, N-num_keep]
            masks_pred = [m.to(device, non_blocking=True) for m in masks_pred]

            optimizer.zero_grad()

            # Forward pass
            with autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                # z_ctx1, z_ctx2: B, N_ctx, D
                # z_tgt1, z_tgt2: B, N_tgt, D
                # z_tgt1_pred, z_tgt2_pred: B, N_tgt, D
                z_ctx1, z_ctx2, z_tgt1, z_tgt2, z_tgt1_pred, z_tgt2_pred = model(images1, images2, masks_enc, masks_pred)

                inv_loss, sigreg_loss, probe1_loss, probe2_loss = loss_fn(z_ctx1, z_ctx2, z_tgt1, z_tgt2, z_tgt1_pred, z_tgt2_pred)

                loss = args.gamma * inv_loss + args.lamb * sigreg_loss + probe1_loss + probe2_loss
            
            # Backward pass
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            #Logging
            if itr % args.log_freq == 0:
                logger.info(
                    f'Epoch [{epoch}][{itr}/{len(data_loader)}] '
                    f'Loss: {loss:.4f} '
                    f'LR: {scheduler.get_last_lr()[0]:.6f}'
                )
                
                wandb.log({
                    'train/loss': loss,
                    'train/probe1_loss': probe1_loss,
                    'train/probe2_loss': probe2_loss,
                    'train/sigreg_loss': sigreg_loss,
                    'train/inv_loss': inv_loss,
                    'train/lr': scheduler.get_last_lr()[0],
                    'epoch': epoch,
                })
        
        epoch_time = time.time() - epoch_start
        logger.info(f'Epoch {epoch} completed in {epoch_time:.2f}s ({epoch_time/60:.2f}m). Avg Loss: {loss:.4f}')
        
        wandb.log({
            'epoch': epoch,
            'train/epoch_loss': loss,
        })

        # Eval
        if epoch % args.eval_freq == 0 or epoch == args.epochs:
            model.eval()
            val_embs1, val_embs2, val_idxs = [], [], []

            with torch.inference_mode():
                # Build retrieval index from validation set
                for images1, images2, idxs in val_data_loader:
                    images1 = images1.to(device, non_blocking=True)
                    images2 = images2.to(device, non_blocking=True)
                    with autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                        z1, z2 = model(images1, images2)
                    val_embs1.append(z1.float().flatten(1))  # [B, N*D]
                    val_embs2.append(z2.float().flatten(1))
                    val_idxs.append(idxs)
                
                val_embs1 = torch.nn.functional.normalize(torch.cat(val_embs1), dim=-1)
                val_embs2 = torch.nn.functional.normalize(torch.cat(val_embs2), dim=-1)
                val_idxs = torch.cat(val_idxs)

                # Get labels for validation set
                val_meta = val_dataset.metadata[val_dataset.metadata['split'] == 'validation']
                test_meta = test_dataset.metadata[test_dataset.metadata['split'] == 'test']
                
                def get_val_labels(idx, modality):
                    name = val_dataset.metadata1[idx] if modality == 1 else val_dataset.metadata2[idx]
                    col = 's1_name' if modality == 1 else 'patch_id'
                    row = val_meta[val_meta[col] == name]
                    return set(row['labels'].iloc[0]) if len(row) > 0 else set()

                # s1->s1, s2->s2, s1->s2, s2->s1
                metrics = {k: {'prec': 0, 'rec': 0} for k in ['s1s1', 's2s2', 's1s2', 's2s1']}
                n = 0

                # Query with test set
                for images1, images2, q_idxs in test_data_loader:
                    images1 = images1.to(device, non_blocking=True)
                    images2 = images2.to(device, non_blocking=True)
                    with autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                        q1, q2 = model(images1, images2)
                    q1 = torch.nn.functional.normalize(q1.float().flatten(1), dim=-1)
                    q2 = torch.nn.functional.normalize(q2.float().flatten(1), dim=-1)

                    # Cosine similarity and top-k for all 4 combinations
                    topk_s1s1 = (q1 @ val_embs1.T).topk(args.top_k, dim=-1).indices
                    topk_s2s2 = (q2 @ val_embs2.T).topk(args.top_k, dim=-1).indices
                    topk_s1s2 = (q1 @ val_embs2.T).topk(args.top_k, dim=-1).indices
                    topk_s2s1 = (q2 @ val_embs1.T).topk(args.top_k, dim=-1).indices

                    for i in range(q1.size(0)):
                        q_idx = q_idxs[i].item()
                        L_q = set(test_meta[test_meta['s1_name'] == test_dataset.metadata1[q_idx]]['labels'].iloc[0])

                        for mode, topk, ret_mod in [('s1s1', topk_s1s1, 1), ('s2s2', topk_s2s2, 2), 
                                                     ('s1s2', topk_s1s2, 2), ('s2s1', topk_s2s1, 1)]:
                            for k in range(args.top_k):
                                r_idx = val_idxs[topk[i, k]].item()
                                L_r = get_val_labels(r_idx, ret_mod)
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
                wandb.log({f'eval/{m}_{k}': v for m in metrics for k, v in metrics[m].items()} | {'epoch': epoch})


        # Save checkpoint
        if epoch % args.save_freq == 0 or epoch == args.epochs:
            checkpoint = {
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'loss': loss,
            }
            save_path = os.path.join(args.output_dir, f'{type(model).__name__}_ep{epoch}.pth')
            os.makedirs(args.output_dir, exist_ok=True)
            torch.save(checkpoint, save_path)
            logger.info(f'Saved checkpoint to {save_path}')
            
            if args.wandb_enabled:
                wandb.save(save_path)
    
    logger.info("Training completed!")
    wandb.finish()


if __name__ == "__main__":
    args = parse_args()
    main(args)
