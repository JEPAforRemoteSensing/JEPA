import copy
import logging
import os
import sys
import argparse
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb

from encoder import build_encoder, get_num_patches
from predictor import build_predictor_for_encoder
from data_loading import load_data
from masks import MultiBlockMaskCollator
from transforms import make_transforms_rgb
from utils import apply_masks, repeat_interleave_batch

# Logging setup
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)

# Global seed
_GLOBAL_SEED = 0
np.random.seed(_GLOBAL_SEED)
torch.manual_seed(_GLOBAL_SEED)
torch.backends.cudnn.benchmark = True


class AverageMeter:
    """Computes and stores the average and current value."""
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def init_model(
    device,
    model_name='vit_base',
    patch_size=16,
    crop_size=224,
    in_chans=3,
    pred_depth=6,
    pred_emb_dim=384,
):
    """
    Initialize encoder and predictor models.
    
    Args:
        device: torch device
        model_name: encoder model name
        patch_size: patch size for ViT
        crop_size: input image size
        in_chans: number of input channels
        pred_depth: predictor depth
        pred_emb_dim: predictor embedding dimension
        
    Returns:
        encoder: context encoder
        predictor: predictor network
    """
    # Build encoder
    encoder, embed_dim = build_encoder(
        model_name=model_name,
        img_size=crop_size,
        patch_size=patch_size,
        in_chans=in_chans,
    )
    encoder = encoder.to(device)
    
    # Calculate number of patches
    num_patches = get_num_patches(crop_size, patch_size)
    
    # Build predictor
    predictor = build_predictor_for_encoder(
        encoder_name=model_name,
        num_patches=num_patches,
        embed_dim=embed_dim,
        predictor_embed_dim=pred_emb_dim,
        depth=pred_depth,
    )
    predictor = predictor.to(device)
    
    logger.info(f"Encoder: {model_name}, embed_dim={embed_dim}, num_patches={num_patches}")
    logger.info(f"Predictor: depth={pred_depth}, embed_dim={pred_emb_dim}")
    
    return encoder, predictor


def init_optimizer(
    encoder,
    predictor,
    lr=1e-4,
    weight_decay=0.05,
    warmup_epochs=10,
    num_epochs=100,
    iterations_per_epoch=1000,
):
    """
    Initialize optimizer and learning rate scheduler.
    
    Args:
        encoder: context encoder
        predictor: predictor network
        lr: learning rate
        weight_decay: weight decay
        warmup_epochs: number of warmup epochs
        num_epochs: total number of epochs
        iterations_per_epoch: iterations per epoch
        
    Returns:
        optimizer: AdamW optimizer
        scheduler: cosine annealing scheduler with warmup
    """
    # Combine parameters
    param_groups = [
        {'params': encoder.parameters()},
        {'params': predictor.parameters()},
    ]
    
    optimizer = torch.optim.AdamW(param_groups, lr=lr, weight_decay=weight_decay)
    
    # Cosine annealing with warmup
    warmup_iters = warmup_epochs * iterations_per_epoch
    total_iters = num_epochs * iterations_per_epoch
    
    def lr_lambda(current_iter):
        if current_iter < warmup_iters:
            return current_iter / warmup_iters
        else:
            progress = (current_iter - warmup_iters) / (total_iters - warmup_iters)
            return 0.5 * (1 + np.cos(np.pi * progress))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    return optimizer, scheduler


def train_one_epoch(
    encoder,
    predictor,
    target_encoder,
    data_loader,
    optimizer,
    scheduler,
    momentum_scheduler,
    device,
    epoch,
    use_amp=False,
    log_freq=10,
):
    """
    Train for one epoch.
    
    Args:
        encoder: context encoder
        predictor: predictor network
        target_encoder: EMA target encoder
        data_loader: training data loader
        optimizer: optimizer
        scheduler: learning rate scheduler
        momentum_scheduler: EMA momentum scheduler
        device: torch device
        epoch: current epoch
        use_amp: whether to use automatic mixed precision
        log_freq: logging frequency
        
    Returns:
        avg_loss: average loss for the epoch
    """
    encoder.train()
    predictor.train()
    target_encoder.eval()
    
    loss_meter = AverageMeter()
    scaler = torch.amp.GradScaler('cpu') if use_amp else None  # Note: MPS doesn't support GradScaler yet
    
    for itr, (images, masks_enc, masks_pred) in enumerate(data_loader):
        # Move to device
        images = images.to(device, non_blocking=True)
        
        masks_enc = [m.to(device, non_blocking=True) for m in masks_enc]
        masks_pred = [m.to(device, non_blocking=True) for m in masks_pred]
        
        # Forward pass
        with torch.amp.autocast(device_type=device.type, enabled=use_amp and device.type != 'mps'):
            # Target representations (no gradient)
            with torch.no_grad():
                h = target_encoder(images)
                h = F.layer_norm(h, (h.size(-1),))
                B = len(h)
                # print("h shape:", h.shape)
                # Extract target patches
                h = apply_masks(h, masks_pred)
                # h = repeat_interleave_batch(h, B, repeat=len(masks_enc))
            
            # Context representations
            z = encoder(images, masks_enc)
            z = predictor(z, masks_enc, masks_pred)
            
            # Loss (smooth L1)
            loss = F.smooth_l1_loss(z, h)
        
        # Backward pass
        optimizer.zero_grad()
        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        
        scheduler.step()
        
        # EMA update of target encoder
        with torch.no_grad():
            m = next(momentum_scheduler)
            for param_q, param_k in zip(encoder.parameters(), target_encoder.parameters()):
                param_k.data.mul_(m).add_((1. - m) * param_q.detach().data)
        
        # Logging
        loss_meter.update(loss.item())
        
        if itr % log_freq == 0:
            logger.info(
                f'Epoch [{epoch}][{itr}/{len(data_loader)}] '
                f'Loss: {loss_meter.avg:.4f} '
                f'LR: {scheduler.get_last_lr()[0]:.6f}'
            )
            
            # Log to wandb
            wandb.log({
                'train/loss': loss_meter.val,
                'train/loss_avg': loss_meter.avg,
                'train/lr': scheduler.get_last_lr()[0],
                'train/ema_momentum': m,
                'epoch': epoch,
            })
    
    return loss_meter.avg


def main(args):
    """Main training function."""
    
    # Device setup
    if torch.backends.mps.is_available():
        device = torch.device('mps')
        logger.info("Using Apple MPS (Metal Performance Shaders)")
    elif torch.cuda.is_available():
        device = torch.device('cuda')
        torch.cuda.set_device(0)
    else:
        device = torch.device('cpu')
        logger.warning("GPU not available, using CPU")
    
    logger.info(f"Using device: {device}")
    
    # Initialize wandb
    wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name,
        config=vars(args),
        mode="online" if args.wandb_enabled else "disabled",
    )
    
    # Initialize models
    encoder, predictor = init_model(
        device=device,
        model_name=args.model_name,
        patch_size=args.patch_size,
        crop_size=args.crop_size,
        in_chans=args.in_chans,
        pred_depth=args.pred_depth,
        pred_emb_dim=args.pred_emb_dim,
    )
    
    # Create target encoder (EMA of encoder)
    target_encoder = copy.deepcopy(encoder)
    for p in target_encoder.parameters():
        p.requires_grad = False
    
    # Create mask collator
    mask_collator = MultiBlockMaskCollator()
    
    # Create transforms
    transform = make_transforms_rgb(num_channels=3)
    
    # Create data loader
    data_loader = load_data(
        root=args.data_root,
        split='train',
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=mask_collator,
        pin_memory=True,
        drop_last=True,
        transform=transform,
    )
    
    iterations_per_epoch = len(data_loader)
    
    # Initialize optimizer and scheduler
    optimizer, scheduler = init_optimizer(
        encoder=encoder,
        predictor=predictor,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        num_epochs=args.epochs,
        iterations_per_epoch=iterations_per_epoch,
    )
    
    # Momentum schedule for EMA
    ema_start, ema_end = args.ema
    total_iters = args.epochs * iterations_per_epoch
    momentum_scheduler = (
        ema_start + i * (ema_end - ema_start) / total_iters
        for i in range(total_iters + 1)
    )
    
    # Training loop
    logger.info("Starting training...")
    for epoch in range(1, args.epochs + 1):
        avg_loss = train_one_epoch(
            encoder=encoder,
            predictor=predictor,
            target_encoder=target_encoder,
            data_loader=data_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            momentum_scheduler=momentum_scheduler,
            device=device,
            epoch=epoch,
            use_amp=args.use_amp,
            log_freq=args.log_freq,
        )
        
        logger.info(f'Epoch {epoch} completed. Avg Loss: {avg_loss:.4f}')
        
        # Log epoch summary to wandb
        wandb.log({
            'epoch': epoch,
            'train/epoch_loss': avg_loss,
        })
        
        # Save checkpoint
        if epoch % args.save_freq == 0 or epoch == args.epochs:
            checkpoint = {
                'epoch': epoch,
                'encoder': encoder.state_dict(),
                'predictor': predictor.state_dict(),
                'target_encoder': target_encoder.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'loss': avg_loss,
            }
            save_path = os.path.join(args.output_dir, f'checkpoint_ep{epoch}.pth')
            os.makedirs(args.output_dir, exist_ok=True)
            torch.save(checkpoint, save_path)
            logger.info(f'Saved checkpoint to {save_path}')
            
            # Log checkpoint to wandb
            if args.wandb_enabled:
                wandb.save(save_path)
    
    logger.info("Training completed!")
    wandb.finish()


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='I-JEPA Training')
    
    # Model
    parser.add_argument('--model_name', type=str, default='vit_base',
                        choices=['vit_tiny', 'vit_small', 'vit_base', 'vit_large', 'vit_huge', 'vit_giant'])
    parser.add_argument('--patch_size', type=int, default=16)
    parser.add_argument('--in_chans', type=int, default=3)
    parser.add_argument('--pred_depth', type=int, default=6)
    parser.add_argument('--pred_emb_dim', type=int, default=384)
    
    # Data
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--crop_size', type=int, default=224)
    parser.add_argument('--crop_scale', type=float, nargs=2, default=[0.3, 1.0])
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=4)
    
    # Augmentation
    parser.add_argument('--horizontal_flip', action='store_true')
    parser.add_argument('--color_distortion', action='store_true')
    parser.add_argument('--gaussian_blur', action='store_true')
    
    # Masking
    parser.add_argument('--enc_mask_scale', type=float, nargs=2, default=[0.85, 1.0])
    parser.add_argument('--pred_mask_scale', type=float, nargs=2, default=[0.15, 0.2])
    parser.add_argument('--aspect_ratio', type=float, nargs=2, default=[0.75, 1.5])
    parser.add_argument('--num_enc_masks', type=int, default=1)
    parser.add_argument('--num_pred_masks', type=int, default=4)
    parser.add_argument('--min_keep', type=int, default=10)
    parser.add_argument('--allow_overlap', action='store_true')
    
    # Training
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--warmup_epochs', type=int, default=10)
    parser.add_argument('--ema', type=float, nargs=2, default=[0.996, 1.0])
    parser.add_argument('--use_amp', action='store_true')
    
    # Logging/Saving
    parser.add_argument('--output_dir', type=str, default='./checkpoints')
    parser.add_argument('--log_freq', type=int, default=10)
    parser.add_argument('--save_freq', type=int, default=50)
    
    # Weights & Biases
    parser.add_argument('--wandb_enabled', action='store_true', help='Enable wandb logging')
    parser.add_argument('--wandb_project', type=str, default='ijepa', help='Wandb project name')
    parser.add_argument('--wandb_run_name', type=str, default=None, help='Wandb run name')
    
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args)
