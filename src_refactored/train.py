import os
import sys
import time
import torch
import logging
import argparse
import numpy as np
import wandb

from data_loading import MultiChannelDataset
from transforms import make_transforms
from masks import RandomMaskCollator
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
    
    model.load_state_dict(checkpoint['encoder1'])
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

    # Prepare data
    transform = make_transforms(12)
    dataset = MultiChannelDataset(args.data_root1, args.data_root2, metadata=args.metadata, split='train', transform=transform)
    
    collate_fn = RandomMaskCollator()
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True, drop_last=True, persistent_workers=True if args.num_workers > 0 else False, prefetch_factor=4 if args.num_workers > 0 else None)
    
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
    loss_fn = MEMPLoss(args.lamb, args.gamma)

    # Automatic Mixed Precision setup
    scaler = GradScaler(device=device, enabled=args.use_amp)

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
            iter_start = time.time()

            images1 = images1.to(device, non_blocking=True)
            images2 = images2.to(device, non_blocking=True)
            masks_enc = [m.to(device, non_blocking=True) for m in masks_enc]
            masks_pred = [m.to(device, non_blocking=True) for m in masks_pred]

            optimizer.zero_grad()

            # Forward pass
            with autocast(device_type=device.type, dtype=torch.bfloat16, enabled=args.use_amp):
                z_ctx1, z_tgt1, z_ctx2, z_tgt2 = model(images1, images2, masks_enc, masks_pred)

                sigreg_loss, probe1_loss, probe2_loss = loss_fn(z_ctx1, z_tgt1, z_ctx2, z_tgt2)
                loss = sigreg_loss + probe1_loss + probe2_loss
            
            # Backward pass
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

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
                    'train/loss_avg': loss,
                    'train/lr': scheduler.get_last_lr()[0],
                    'epoch': epoch,
                })
        
        epoch_time = time.time() - epoch_start
        logger.info(f'Epoch {epoch} completed in {epoch_time:.2f}s ({epoch_time/60:.2f}m). Avg Loss: {loss:.4f}')
        
        wandb.log({
            'epoch': epoch,
            'train/epoch_loss': loss,
        })

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


# TODO: Think about how to divide predictors into another shared encoder, and a probe, so that probe is used for eval
# TODO: Finish losses.py and models.py




def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='I-JEPA Training')
    
    parser.add_argument('--in_chans1', type=int, default=2)
    parser.add_argument('--in_chans2', type=int, default=10)
    
    parser.add_argument('--data_root1', type=str, default='data/BEN_14k/BigEarthNet-S1')
    parser.add_argument('--data_root2', type=str, default='data/BEN_14k/BigEarthNet-S2')
    parser.add_argument('--metadata', type=str, default='data/BEN_14k/serbia_metadata.parquet')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=4)
    
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
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume training from')
    
    # Weights & Biases
    parser.add_argument('--wandb_enabled', action='store_true', help='Enable wandb logging')
    parser.add_argument('--wandb_project', type=str, default='ijepa', help='Wandb project name')
    parser.add_argument('--wandb_run_name', type=str, default=None, help='Wandb run name')
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
