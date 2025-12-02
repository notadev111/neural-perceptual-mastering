"""
Training script for neural perceptual audio mastering.

Supports phased training:
- Phase 1A: Parametric EQ only
- Phase 1B: EQ + Residual
- Phase 1C: Adaptive bands + Residual
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import yaml
from pathlib import Path
import argparse
from tqdm import tqdm
import time
import wandb
import torchaudio

from models import (
    MasteringModel_Phase1A,
    MasteringModel_Phase1B,
    MasteringModel_Phase1C
)
from losses import CombinedLoss
from data_loader import get_dataloaders


def train_epoch(model, train_loader, criterion, optimizer, device, epoch, writer, use_wandb=True):
    """Train for one epoch."""
    model.train()

    total_loss = 0.0
    loss_components = {'spectral': 0.0, 'perceptual': 0.0,
                       'loudness': 0.0, 'param_reg': 0.0}

    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')

    for batch_idx, (unmastered, mastered) in enumerate(pbar):
        unmastered = unmastered.to(device)
        mastered = mastered.to(device)

        optimizer.zero_grad()

        # Forward pass
        output, params = model(unmastered)

        # Compute loss
        if 'eq_gains' in params:
            eq_params = (
                params['eq_frequencies'],
                params['eq_gains'],
                params['eq_q_factors']
            )
            loss, loss_dict = criterion(output, mastered, eq_params)
        else:
            loss, loss_dict = criterion(output, mastered)

        # Backward pass
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        # Logging
        total_loss += loss.item()
        for key in loss_components:
            if key in loss_dict:
                loss_components[key] += loss_dict[key]

        # Update progress bar
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        # Log to tensorboard
        global_step = epoch * len(train_loader) + batch_idx
        writer.add_scalar('train/batch_loss', loss.item(), global_step)

        # Log to wandb
        if use_wandb and batch_idx % 10 == 0:  # Log every 10 batches
            wandb.log({
                'train/batch_loss': loss.item(),
                'train/batch_spectral': loss_dict.get('spectral', 0),
                'train/batch_perceptual': loss_dict.get('perceptual', 0),
                'train/batch_loudness': loss_dict.get('loudness', 0),
                'global_step': global_step
            })

    # Epoch averages
    avg_loss = total_loss / len(train_loader)
    for key in loss_components:
        loss_components[key] /= len(train_loader)

    return avg_loss, loss_components


def validate(model, val_loader, criterion, device, epoch, writer, use_wandb=True, log_audio=False):
    """Validate the model."""
    model.eval()

    total_loss = 0.0
    loss_components = {'spectral': 0.0, 'perceptual': 0.0,
                       'loudness': 0.0, 'param_reg': 0.0}

    # Store one batch for audio logging
    sample_unmastered = None
    sample_mastered = None
    sample_output = None

    with torch.no_grad():
        for batch_idx, (unmastered, mastered) in enumerate(tqdm(val_loader, desc='Validation')):
            unmastered = unmastered.to(device)
            mastered = mastered.to(device)

            # Forward pass
            output, params = model(unmastered)

            # Compute loss
            if 'eq_gains' in params:
                eq_params = (
                    params['eq_frequencies'],
                    params['eq_gains'],
                    params['eq_q_factors']
                )
                loss, loss_dict = criterion(output, mastered, eq_params)
            else:
                loss, loss_dict = criterion(output, mastered)

            total_loss += loss.item()
            for key in loss_components:
                if key in loss_dict:
                    loss_components[key] += loss_dict[key]

            # Save first batch for audio logging
            if batch_idx == 0 and log_audio:
                sample_unmastered = unmastered[0].cpu()  # First sample
                sample_mastered = mastered[0].cpu()
                sample_output = output[0].cpu()

    # Averages
    avg_loss = total_loss / len(val_loader)
    for key in loss_components:
        loss_components[key] /= len(val_loader)

    # Log to tensorboard
    writer.add_scalar('val/total_loss', avg_loss, epoch)
    for key, value in loss_components.items():
        writer.add_scalar(f'val/{key}', value, epoch)

    # Log to wandb
    if use_wandb:
        log_dict = {
            'val/total_loss': avg_loss,
            'val/spectral': loss_components['spectral'],
            'val/perceptual': loss_components['perceptual'],
            'val/loudness': loss_components['loudness'],
            'val/param_reg': loss_components['param_reg'],
            'epoch': epoch
        }

        # Log audio samples every 5 epochs
        if log_audio and sample_output is not None and epoch % 5 == 0:
            log_dict['audio/unmastered'] = wandb.Audio(sample_unmastered.numpy(), sample_rate=44100, caption="Unmastered")
            log_dict['audio/target'] = wandb.Audio(sample_mastered.numpy(), sample_rate=44100, caption="Target Mastered")
            log_dict['audio/predicted'] = wandb.Audio(sample_output.numpy(), sample_rate=44100, caption="Model Output")

        wandb.log(log_dict)

    return avg_loss, loss_components


def save_checkpoint(model, optimizer, epoch, loss, config, checkpoint_dir, phase):
    """Save model checkpoint."""
    checkpoint_path = Path(checkpoint_dir) / f'phase_{phase}_epoch_{epoch}.pt'
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'config': config
    }, checkpoint_path)
    
    print(f"Checkpoint saved: {checkpoint_path}")


def main():
    parser = argparse.ArgumentParser(description='Train neural mastering model')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to config file')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--no-wandb', action='store_true',
                        help='Disable Weights & Biases logging')
    parser.add_argument('--wandb-project', type=str, default='neural-mastering',
                        help='W&B project name')
    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Determine phase
    phase = config['model'].get('phase', '1B')
    print(f"\n{'='*60}")
    print(f"Training Phase {phase}")
    print(f"{'='*60}\n")

    # Initialize Weights & Biases
    use_wandb = not args.no_wandb
    if use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=f"phase_{phase}_{time.strftime('%Y%m%d_%H%M%S')}",
            config=config,
            tags=[f"phase_{phase}"],
            notes=f"Training Phase {phase} - Neural Perceptual Audio Mastering"
        )
        print(f"Weights & Biases initialized: {wandb.run.url}")
        print(f"Project: {args.wandb_project}")
        print()
    
    # Create model
    if phase == '1A':
        model = MasteringModel_Phase1A(config).to(device)
        print("Model: Parametric EQ only (white-box)")
    elif phase == '1B':
        model = MasteringModel_Phase1B(config).to(device)
        print("Model: EQ + Residual (hybrid)")
    elif phase == '1C':
        model = MasteringModel_Phase1C(config).to(device)
        print("Model: Adaptive bands + Residual (novel)")
    else:
        raise ValueError(f"Unknown phase: {phase}")
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {num_params:,}")

    # Watch model with wandb
    if use_wandb:
        wandb.watch(model, log='all', log_freq=100)

    # Create dataloaders
    print("\nLoading data...")
    train_loader, val_loader, test_loader = get_dataloaders(config)
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    
    # Loss function
    criterion = CombinedLoss(config).to(device)
    
    # Optimizer
    lr = config['training']['learning_rate']
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # Tensorboard
    log_dir = Path('runs') / f'phase_{phase}_{time.strftime("%Y%m%d_%H%M%S")}'
    writer = SummaryWriter(log_dir)
    print(f"\nTensorboard logs: {log_dir}")
    
    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume:
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"\nResumed from epoch {start_epoch}")
    
    # Training loop
    num_epochs = config['training']['epochs']
    save_every = config['checkpoint']['save_every']
    checkpoint_dir = Path('checkpoints') / f'phase_{phase}'
    
    best_val_loss = float('inf')
    
    print("\n" + "="*60)
    print("Starting Training")
    print("="*60 + "\n")
    
    for epoch in range(start_epoch, num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")

        # Train
        train_loss, train_components = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch, writer, use_wandb=use_wandb
        )

        # Validate (log audio every 5 epochs)
        log_audio = (epoch % 5 == 0)
        val_loss, val_components = validate(
            model, val_loader, criterion, device, epoch, writer,
            use_wandb=use_wandb, log_audio=log_audio
        )

        # Update learning rate
        scheduler.step(val_loss)

        # Log epoch results
        print(f"\nEpoch {epoch+1} Results:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"    - Spectral: {train_components['spectral']:.4f}")
        print(f"    - Perceptual: {train_components['perceptual']:.4f}")
        print(f"    - Loudness: {train_components['loudness']:.4f}")
        print(f"    - Param Reg: {train_components['param_reg']:.4f}")
        print(f"  Val Loss: {val_loss:.4f}")
        print(f"    - Spectral: {val_components['spectral']:.4f}")
        print(f"    - Perceptual: {val_components['perceptual']:.4f}")
        print(f"    - Loudness: {val_components['loudness']:.4f}")
        print(f"    - Param Reg: {val_components['param_reg']:.4f}")

        writer.add_scalar('train/epoch_loss', train_loss, epoch)
        writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch)

        # Log epoch summary to wandb
        if use_wandb:
            wandb.log({
                'train/epoch_loss': train_loss,
                'learning_rate': optimizer.param_groups[0]['lr'],
                'epoch': epoch
            })
        
        # Save checkpoint
        if (epoch + 1) % save_every == 0:
            save_checkpoint(model, optimizer, epoch, val_loss, config,
                            checkpoint_dir, phase)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = checkpoint_dir / 'best_model.pt'
            save_checkpoint(model, optimizer, epoch, val_loss, config,
                            checkpoint_dir.parent, f"{phase}_best")
            print(f"  New best model! Val loss: {val_loss:.4f}")
    
    print("\n" + "="*60)
    print("Training Complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print("="*60)

    writer.close()

    # Finish wandb run
    if use_wandb:
        wandb.finish()
        print("\nWeights & Biases run finished")


if __name__ == '__main__':
    main()
