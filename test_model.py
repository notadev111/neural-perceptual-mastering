"""
Test the trained model on the test set.
Loads best checkpoint and evaluates on held-out test data.
"""

import torch
import yaml
from pathlib import Path
from src.models import (
    MasteringModel_Phase1A,
    MasteringModel_Phase1B,
    MasteringModel_Phase1C
)
from src.data_loader import get_dataloaders
from src.losses import CombinedLoss
from tqdm import tqdm


def test_model(config_path, checkpoint_path):
    """
    Test model on test set.

    Args:
        config_path: Path to config file (e.g., configs/phase1a.yaml)
        checkpoint_path: Path to best checkpoint
    """
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    print("=" * 70)
    print("MODEL TESTING")
    print("=" * 70)
    print(f"\nConfig: {config_path}")
    print(f"Checkpoint: {checkpoint_path}")

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load data (only need test loader)
    print("\nLoading test data...")
    _, _, test_loader = get_dataloaders(config)
    print(f"Test batches: {len(test_loader)}")

    # Build model
    print("\nBuilding model...")
    phase = config['model']['phase']

    if phase == '1A':
        model = MasteringModel_Phase1A(config).to(device)
        print("Model: Phase 1A - Parametric EQ only (white-box)")
    elif phase == '1B':
        model = MasteringModel_Phase1B(config).to(device)
        print("Model: Phase 1B - EQ + Residual (hybrid)")
    elif phase == '1C':
        model = MasteringModel_Phase1C(config).to(device)
        print("Model: Phase 1C - Adaptive bands + Residual (novel)")
    else:
        raise ValueError(f"Unknown phase: {phase}")

    # Load checkpoint
    print(f"\nLoading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Print checkpoint info
    print(f"Checkpoint epoch: {checkpoint.get('epoch', 'unknown')}")
    val_loss = checkpoint.get('val_loss', None)
    if val_loss is not None:
        print(f"Checkpoint val loss: {val_loss:.4f}")
    else:
        print(f"Checkpoint val loss: unknown")

    # Setup loss
    loss_fn = CombinedLoss(config).to(device)

    # Test
    print("\n" + "=" * 70)
    print("TESTING")
    print("=" * 70)

    model.eval()
    test_losses = []
    test_spectral = []
    test_perceptual = []
    test_loudness = []
    test_param_reg = []

    with torch.no_grad():
        for batch_idx, (unmastered, mastered) in enumerate(tqdm(test_loader, desc="Testing")):
            unmastered = unmastered.to(device)
            mastered = mastered.to(device)

            # Forward pass
            output, params = model(unmastered)

            # Extract EQ params as tuple for loss function
            if 'eq_frequencies' in params:
                eq_params = (
                    params['eq_frequencies'],
                    params['eq_gains'],
                    params['eq_q_factors']
                )
            else:
                eq_params = None

            # Compute loss
            loss, loss_dict = loss_fn(output, mastered, eq_params)

            # Store losses
            test_losses.append(loss.item())
            test_spectral.append(loss_dict['spectral'])
            test_perceptual.append(loss_dict['perceptual'])
            test_loudness.append(loss_dict['loudness'])
            if 'param_reg' in loss_dict:
                test_param_reg.append(loss_dict['param_reg'])

    # Compute averages
    avg_loss = sum(test_losses) / len(test_losses)
    avg_spectral = sum(test_spectral) / len(test_spectral)
    avg_perceptual = sum(test_perceptual) / len(test_perceptual)
    avg_loudness = sum(test_loudness) / len(test_loudness)
    avg_param_reg = sum(test_param_reg) / len(test_param_reg) if test_param_reg else 0.0

    # Print results
    print("\n" + "=" * 70)
    print("TEST RESULTS")
    print("=" * 70)
    print(f"\nTest Loss: {avg_loss:.4f}")
    print(f"  - Spectral:    {avg_spectral:.4f}")
    print(f"  - Perceptual:  {avg_perceptual:.4f}")
    print(f"  - Loudness:    {avg_loudness:.4f}")
    if test_param_reg:
        print(f"  - Param Reg:   {avg_param_reg:.4f}")

    print("\n" + "=" * 70)
    print("COMPARISON")
    print("=" * 70)

    # Compare to validation
    val_loss = checkpoint.get('val_loss', None)
    if val_loss:
        print(f"\nValidation Loss: {val_loss:.4f}")
        print(f"Test Loss:       {avg_loss:.4f}")
        print(f"Difference:      {avg_loss - val_loss:+.4f}")

        if abs(avg_loss - val_loss) < 0.01:
            print("\n[EXCELLENT] Test loss matches validation - good generalization!")
        elif avg_loss < val_loss + 0.05:
            print("\n[GOOD] Test loss close to validation - model generalizes well")
        else:
            print("\n[WARNING] Test loss significantly different from validation")

    print("\n" + "=" * 70)

    return {
        'test_loss': avg_loss,
        'spectral': avg_spectral,
        'perceptual': avg_perceptual,
        'loudness': avg_loudness,
        'param_reg': avg_param_reg
    }


if __name__ == '__main__':
    import sys

    # Default paths
    config_path = "configs/phase1a.yaml"
    checkpoint_path = "checkpoints/phase_1A_best_epoch_11.pt"

    # Allow command line override
    if len(sys.argv) > 1:
        checkpoint_path = sys.argv[1]
    if len(sys.argv) > 2:
        config_path = sys.argv[2]

    results = test_model(config_path, checkpoint_path)
