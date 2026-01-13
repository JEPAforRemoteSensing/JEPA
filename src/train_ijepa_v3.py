import copy
import logging
import os
import sys
import argparse
from functools import partial
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb

from encoder import build_encoder, get_num_patches
from predictor import build_predictor_for_encoder
from data_loading import load_data
from masks import RandomMaskCollator
from transforms import make_transforms_rgb, make_transforms
from utils import apply_masks, repeat_interleave_batch
from vision_transformer import SharedPredictor

# Logging setup
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)

# Global seed
_GLOBAL_SEED = 0
np.random.seed(_GLOBAL_SEED)
torch.manual_seed(_GLOBAL_SEED)

# JIT-compiled helper for SIGReg computation
@torch.jit.script
def _sigreg_compute(proj: torch.Tensor, t: torch.Tensor, phi: torch.Tensor, 
                    weights: torch.Tensor, device: torch.device) -> torch.Tensor:
    """JIT-compiled SIGReg computation for better performance."""
    # Create random projection matrix
    A = torch.randn(proj.size(-1), 256, device=device, dtype=proj.dtype)
    A = F.normalize(A, p=2.0, dim=0)
    
    # Fused operations for efficiency
    x_t = torch.matmul(proj, A).unsqueeze(-1) * t
    cos_part = (x_t.cos().mean(-3) - phi).square()
    sin_part = x_t.sin().mean(-3).square()
    err = cos_part + sin_part
    statistic = torch.matmul(err, weights) * proj.size(-2)
    return statistic.mean()


class SIGReg(torch.nn.Module):
    def __init__(self, knots=17, device=None):
        super().__init__()
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)
        self.device = device

    def forward(self, proj):
        # # Optimized: Use F.normalize instead of manual normalization
        # A = torch.randn(proj.size(-1), 256, device=self.device, dtype=proj.dtype)
        # A = F.normalize(A, p=2, dim=0)
        # # Fused operations for efficiency
        # x_t = torch.matmul(proj, A).unsqueeze(-1).mul_(self.t)
        # cos_part = x_t.cos().mean(-3).sub_(self.phi).square_()
        # sin_part = x_t.sin().mean(-3).square_()
        # err = cos_part.add_(sin_part)
        # statistic = torch.matmul(err, self.weights).mul_(proj.size(-2))
        # return statistic.mean()
        return _sigreg_compute(proj, self.t, self.phi, self.weights, self.device)


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
    in_chans1=3,
    in_chans2=3,
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
    encoder1, embed_dim1 = build_encoder(
        model_name=model_name,
        img_size=crop_size,
        patch_size=patch_size,
        in_chans=in_chans1,
    )
    encoder1 = encoder1.to(device)
# B N E (1, 98, 784)
    # Build encoder
    encoder2, embed_dim2 = build_encoder(
        model_name=model_name,
        img_size=crop_size,
        patch_size=patch_size,
        in_chans=in_chans2,
    )
    encoder2 = encoder2.to(device)
    
    # Calculate number of patches
    num_patches = get_num_patches(crop_size, patch_size)
    
    # Build predictor
    # predictor = build_predictor_for_encoder(
    #     encoder_name=model_name,
    #     num_patches=num_patches,
    #     embed_dim=embed_dim1 + embed_dim2,
    #     predictor_embed_dim=pred_emb_dim,
    #     depth=pred_depth,
    # )
    # predictor = predictor.to(device)
    probe = SharedPredictor(
        num_patches=num_patches,
        embed_dim=embed_dim1,
        predictor_embed_dim=pred_emb_dim,
        depth=pred_depth,
    ).to(device)
    
    logger.info(f"Encoder1: {model_name}, embed_dim={embed_dim1}, num_patches={num_patches}")
    logger.info(f"Encoder2: {model_name}, embed_dim={embed_dim2}, num_patches={num_patches}")
    logger.info(f"Predictor: depth={pred_depth}, embed_dim={pred_emb_dim}")
    
    # Enable gradient checkpointing for memory efficiency
    if hasattr(encoder1, 'set_grad_checkpointing'):
        encoder1.set_grad_checkpointing(True)
        encoder2.set_grad_checkpointing(True)
        logger.info("Enabled gradient checkpointing")
    
    # Compile models with torch.compile for speedup (PyTorch 2.0+)
    # Using 'reduce-overhead' mode instead of 'max-autotune' to avoid CUDA graphs tensor overwriting issues
    if hasattr(torch, 'compile'):
        try:
            encoder1 = torch.compile(encoder1, mode='reduce-overhead')
            encoder2 = torch.compile(encoder2, mode='reduce-overhead')
            probe = torch.compile(probe, mode='reduce-overhead')
            logger.info("Models compiled with torch.compile (reduce-overhead mode)")
        except Exception as e:
            logger.warning(f"torch.compile failed: {e}. Continuing without compilation.")
    
    return encoder1, encoder2, probe


def init_optimizer(
    encoder1,
    encoder2,
    probe,
    lr=1e-4,
    weight_decay=0.05,
    warmup_epochs=10,
    num_epochs=100,
    iterations_per_epoch=1000,
):
    """
    Initialize optimizer and learning rate scheduler.
    
    Args:
        encoder1: first context encoder
        encoder2: second context encoder
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
        {'params': encoder1.parameters()},
        {'params': encoder2.parameters()},
        {'params': probe.parameters()},
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
    device,
    encoder1,
    encoder2,
    probe,
    data_loader1,
    data_loader2,
    optimizer,
    scheduler,
    epoch,
    use_amp=False,
    log_freq=10,
    lamb=0.5,
    gamma=1.0,
    sigreg=None,
    accumulation_steps=1,
    batch_size=64,
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
    encoder1.train()
    encoder2.train()
    probe.train()
    
    loss_meter = AverageMeter()
    probe_loss_meter = AverageMeter()
    lejepa_loss_meter = AverageMeter()
    scaler = torch.amp.GradScaler('cuda') if (use_amp and device.type == 'cuda') else None
    
    # Timing metrics
    data_time = AverageMeter()
    forward_time = AverageMeter()
    backward_time = AverageMeter()
    step_time = AverageMeter()
    epoch_start = time.time()
    
    for itr, ((images1, masks_enc1, masks_pred1), (images2, masks_enc2, masks_pred2)) in enumerate(zip(data_loader1, data_loader2)):
        iter_start = time.time()
        
        # Move to device
        images1 = images1.to(device, non_blocking=True)
        images2 = images2.to(device, non_blocking=True)
        
        masks_enc1 = [m.to(device, non_blocking=True) for m in masks_enc1]
        masks_pred1 = [m.to(device, non_blocking=True) for m in masks_pred1]
        masks_enc2 = [m.to(device, non_blocking=True) for m in masks_enc2]
        masks_pred2 = [m.to(device, non_blocking=True) for m in masks_pred2]
        
        data_time.update(time.time() - iter_start)
        forward_start = time.time()
        
        # Forward pass
        with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp and device.type != 'mps'):
            # Target representations (no gradient) - properly cached
            with torch.no_grad():
                z_target1 = encoder1(images1, masks_pred1).detach().clone()
                z_target2 = encoder2(images2, masks_pred2).detach().clone()
            
            # Context representations
            z_context1 = encoder1(images1, masks_enc1).clone()
            z_context2 = encoder2(images2, masks_enc2).clone()
            
            # Optimized: use cat instead of stack for better memory layout
            # Clone prevents CUDA graphs tensor overwriting issues
            encoder_emb = torch.cat([
                z_context1.unsqueeze(0), 
                z_target1.unsqueeze(0), 
                z_context2.unsqueeze(0), 
                z_target2.unsqueeze(0)
            ], dim=0)
            
            sigreg_loss = sigreg(encoder_emb)
            lejepa_loss = sigreg_loss * lamb
            inv_loss = F.smooth_l1_loss(z_context1, z_context2) * gamma
            
            # Prediction
            y_1_2 = probe(z_context1, masks_enc1, masks_pred2, z_context2)
            y_2_1 = probe(z_context2, masks_enc2, masks_pred1, z_context1)

            probe_loss = F.smooth_l1_loss(y_1_2, z_target2) + F.smooth_l1_loss(y_2_1, z_target1)
            loss = (lejepa_loss + probe_loss + inv_loss) / accumulation_steps  # Scale for gradient accumulation
        
        forward_time.update(time.time() - forward_start)
        backward_start = time.time()
        
        # Backward pass
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        
        backward_time.update(time.time() - backward_start)
        
        # Gradient accumulation: only step every N iterations
        if (itr + 1) % accumulation_steps == 0:
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
        
        step_time.update(time.time() - iter_start)
        
        # Logging (unscale loss for logging)
        loss_meter.update(loss.item() * accumulation_steps)
        probe_loss_meter.update(probe_loss.item())
        lejepa_loss_meter.update(lejepa_loss.item())
        
        if itr % log_freq == 0:
            logger.info(
                f'Epoch [{epoch}][{itr}/{len(data_loader1)}] '
                f'Loss: {loss_meter.avg:.4f} '
                f'LR: {scheduler.get_last_lr()[0]:.6f} | '
                f'Time: data={data_time.avg*1000:.1f}ms fwd={forward_time.avg*1000:.1f}ms '
                f'bwd={backward_time.avg*1000:.1f}ms step={step_time.avg*1000:.1f}ms'
            )
            
            # Log to wandb
            wandb.log({
                'train/loss': loss_meter.val,
                'train/probe_loss': probe_loss_meter.val,
                'train/lejepa_loss': lejepa_loss_meter.val,
                'train/loss_avg': loss_meter.avg,
                'train/lr': scheduler.get_last_lr()[0],
                'timing/data_time_ms': data_time.avg * 1000,
                'timing/forward_time_ms': forward_time.avg * 1000,
                'timing/backward_time_ms': backward_time.avg * 1000,
                'timing/step_time_ms': step_time.avg * 1000,
                'timing/samples_per_sec': batch_size / step_time.avg if step_time.avg > 0 else 0,
                'epoch': epoch,
            })
    
    epoch_time = time.time() - epoch_start
    logger.info(f'Epoch {epoch} completed in {epoch_time:.2f}s ({epoch_time/60:.2f}m). '
                f'Throughput: {len(data_loader1) * batch_size / epoch_time:.2f} samples/sec')
    
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
    
    # Initialize models
    encoder1, encoder2, probe = init_model(
        device=device,
        model_name=args.model_name,
        patch_size=args.patch_size,
        crop_size=args.crop_size,
        in_chans1=args.in_chans1,
        in_chans2=args.in_chans2,
        pred_depth=args.pred_depth,
        pred_emb_dim=args.pred_emb_dim,
    )
    
    # Create mask collator
    mask_collator1 = RandomMaskCollator()
    mask_collator2 = RandomMaskCollator()
    
    # Create transforms
    transform1 = make_transforms(num_channels=2)
    transform2 = make_transforms_rgb(num_channels=3)
    
    # Create data loaders with optimized settings
    # Note: persistent_workers keeps workers alive between epochs (reduces overhead)
    # prefetch_factor controls how many batches to prefetch per worker
    data_loader1 = load_data(
        root=args.data_root1,
        split='train',
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=mask_collator1,
        pin_memory=True,
        drop_last=True,
        transform=transform1,
        persistent_workers=True if args.num_workers > 0 else False,
        prefetch_factor=4 if args.num_workers > 0 else None,
    )
    
    data_loader2 = load_data(
        root=args.data_root2,
        split='train',
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=mask_collator2,
        pin_memory=True,
        drop_last=True,
        transform=transform2,
        persistent_workers=True if args.num_workers > 0 else False,
        prefetch_factor=4 if args.num_workers > 0 else None,
    )

    iterations_per_epoch = len(data_loader1)
    
    # Initialize optimizer and scheduler
    optimizer, scheduler = init_optimizer(
        encoder1=encoder1,
        encoder2=encoder2,
        probe=probe,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        num_epochs=args.epochs,
        iterations_per_epoch=iterations_per_epoch,
    )

    sigreg = SIGReg(device=device).to(device)
    
    # Training loop
    logger.info("Starting training...")
    logger.info(f"Effective batch size: {args.batch_size * args.accumulation_steps}")
    for epoch in range(1, args.epochs + 1):
        avg_loss = train_one_epoch(
            device=device,
            encoder1=encoder1,
            encoder2=encoder2,
            probe=probe,
            data_loader1=data_loader1,
            data_loader2=data_loader2,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            use_amp=args.use_amp,
            log_freq=args.log_freq,
            lamb=args.lamb,
            gamma=args.gamma,
            sigreg=sigreg,
            accumulation_steps=args.accumulation_steps,
            batch_size=args.batch_size,
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
                'encoder1': encoder1.state_dict(),
                'encoder2': encoder2.state_dict(),
                'probe': probe.state_dict(),
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
    parser.add_argument('--in_chans1', type=int, default=2)
    parser.add_argument('--in_chans2', type=int, default=3)
    parser.add_argument('--pred_depth', type=int, default=4)
    parser.add_argument('--pred_emb_dim', type=int, default=768)
    
    # Data
    parser.add_argument('--data_root1', type=str, default='data/BEN_14k/BigEarthNet-S1')
    parser.add_argument('--data_root2', type=str, default='data/BEN_14k/BigEarthNet-S2')
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
    parser.add_argument('--use_amp', action='store_true', help='Use automatic mixed precision (bfloat16 on CUDA)')
    parser.add_argument('--accumulation_steps', type=int, default=1, help='Gradient accumulation steps (effective_batch = batch * steps)')
    parser.add_argument('--lamb', type=float, default=0.02, help='Weight for SIGReg loss vs invariance loss')
    parser.add_argument('--gamma', type=float, default=1.0, help='Weight for invariance loss')
    
    # Logging/Saving
    parser.add_argument('--output_dir', type=str, default='./checkpoints')
    parser.add_argument('--log_freq', type=int, default=10)
    parser.add_argument('--save_freq', type=int, default=10)
    
    # Weights & Biases
    parser.add_argument('--wandb_enabled', action='store_true', help='Enable wandb logging')
    parser.add_argument('--wandb_project', type=str, default='ijepa', help='Wandb project name')
    parser.add_argument('--wandb_run_name', type=str, default=None, help='Wandb run name')
    
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args)
