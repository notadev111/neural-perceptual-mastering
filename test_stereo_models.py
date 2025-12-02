"""
Quick test script to verify stereo models work correctly.
"""

import torch
import yaml
from pathlib import Path

# Import models
from src.models import (
    MasteringModel_Phase1A,
    MasteringModel_Phase1B,
    MasteringModel_Phase1C
)

def test_model(model_class, config, phase_name):
    """Test a model with stereo input."""
    print(f"\n{'='*60}")
    print(f"Testing {phase_name}")
    print(f"{'='*60}")

    # Create model
    model = model_class(config)
    model.eval()

    # Create dummy stereo input [batch=2, channels=2, samples=220500 (5s at 44.1kHz)]
    batch_size = 2
    channels = 2  # Stereo
    samples = 220500  # 5 seconds at 44.1kHz

    dummy_audio = torch.randn(batch_size, channels, samples)
    print(f"Input shape: {dummy_audio.shape}")

    # Forward pass
    try:
        with torch.no_grad():
            output, params = model(dummy_audio)

        print(f"Output shape: {output.shape}")
        print(f"Expected: [{batch_size}, {channels}, {samples}]")

        # Verify output shape
        assert output.shape == (batch_size, channels, samples), \
            f"Output shape mismatch! Got {output.shape}"

        # Check parameters
        print(f"Parameters extracted:")
        for key in params:
            if isinstance(params[key], torch.Tensor):
                print(f"  {key}: {params[key].shape}")

        print(f"[OK] {phase_name} works with stereo!")
        return True

    except Exception as e:
        print(f"[FAILED] {phase_name}: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("="*60)
    print("Stereo Model Compatibility Test")
    print("="*60)

    # Load config
    config_path = Path('configs/phase1a.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Test all three phases
    results = {}

    results['Phase 1A'] = test_model(
        MasteringModel_Phase1A,
        config,
        "Phase 1A (EQ only)"
    )

    results['Phase 1B'] = test_model(
        MasteringModel_Phase1B,
        config,
        "Phase 1B (EQ + Residual)"
    )

    results['Phase 1C'] = test_model(
        MasteringModel_Phase1C,
        config,
        "Phase 1C (Adaptive EQ + Residual)"
    )

    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    for phase, passed in results.items():
        status = "[OK]" if passed else "[FAILED]"
        print(f"{status} {phase}")

    all_passed = all(results.values())
    if all_passed:
        print("\n[SUCCESS] All models are stereo-compatible!")
    else:
        print("\n[WARNING] Some models failed. Check errors above.")

    return all_passed


if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)
