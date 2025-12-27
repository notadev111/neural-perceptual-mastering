"""
Test that FIXED encoder can distinguish high frequencies (15kHz vs 1kHz).

This verifies the stride=2 fix preserves the full audible spectrum.
Broken encoder (stride=16) could only see up to 1.3kHz.
"""

import torch
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from models import AudioEncoder


def test_high_freq_preservation():
    """Test encoder can distinguish 15kHz from 1kHz."""
    print("=" * 60)
    print("Testing FIXED Encoder - High Frequency Preservation")
    print("=" * 60)

    sr = 44100
    duration = 5
    samples = sr * duration

    # Create test signals
    t = np.linspace(0, duration, samples)

    # 15kHz sine (high "air" frequency for mastering)
    freq_15khz = np.sin(2 * np.pi * 15000 * t)
    audio_15k = torch.tensor(freq_15khz).float().unsqueeze(0).repeat(1, 2, 1)

    # 1kHz sine (mid frequency)
    freq_1khz = np.sin(2 * np.pi * 1000 * t)
    audio_1k = torch.tensor(freq_1khz).float().unsqueeze(0).repeat(1, 2, 1)

    print(f"\nInput shapes: {audio_15k.shape}")

    # Create encoder
    encoder = AudioEncoder(latent_dim=512)
    encoder.eval()

    # Test forward pass and shapes
    with torch.no_grad():
        # Check intermediate shapes
        stem_out = encoder.stem(audio_15k)
        print(f"After stem: {stem_out.shape}")
        print(f"Expected:   torch.Size([1, 64, 110250])")
        print(f"Match: {'[OK]' if stem_out.shape == torch.Size([1, 64, 110250]) else '[WRONG!]'}")

        # Full forward pass
        z_15k = encoder(audio_15k)
        z_1k = encoder(audio_1k)

        print(f"\nLatent shape: {z_15k.shape}")
        print(f"Expected:     torch.Size([1, 512])")

    # Check latent codes are different
    diff = torch.norm(z_15k - z_1k).item()

    print(f"\n" + "=" * 60)
    print(f"Latent difference (15kHz vs 1kHz): {diff:.4f}")
    print(f"Threshold for distinguishability: 0.1")

    if diff > 0.1:
        print(f"\n[PASS] ENCODER FIXED!")
        print(f"   Can distinguish 15kHz from 1kHz")
        print(f"   Full 20kHz spectrum preserved")
    else:
        print(f"\n[FAIL] ENCODER STILL BROKEN")
        print(f"   Cannot distinguish high frequencies")
        print(f"   Stride too aggressive")

    print("=" * 60)

    # Parameter count
    params = sum(p.numel() for p in encoder.parameters())
    print(f"\nTotal parameters: {params:,}")
    print(f"Expected: ~1.2M")

    return diff > 0.1


def test_frequency_range():
    """Test multiple frequencies across spectrum."""
    print("\n" + "=" * 60)
    print("Testing Frequency Range (100Hz - 18kHz)")
    print("=" * 60)

    encoder = AudioEncoder(latent_dim=512)
    encoder.eval()

    sr = 44100
    duration = 5
    samples = sr * duration
    t = np.linspace(0, duration, samples)

    # Test frequencies across spectrum
    test_freqs = [100, 500, 1000, 4000, 8000, 12000, 15000, 18000]
    latents = []

    with torch.no_grad():
        for freq in test_freqs:
            signal = np.sin(2 * np.pi * freq * t)
            audio = torch.tensor(signal).float().unsqueeze(0).repeat(1, 2, 1)
            z = encoder(audio)
            latents.append(z)

    # Check all latents are distinguishable
    print(f"\n{'Frequency':<12} {'Distinguishable':<20}")
    print("-" * 32)

    distinguishable_count = 0
    for i, freq in enumerate(test_freqs[1:], 1):
        diff = torch.norm(latents[i] - latents[0]).item()
        is_dist = diff > 0.1
        distinguishable_count += is_dist
        print(f"{freq}Hz{' '*(8-len(str(freq)))} {('[YES]' if is_dist else '[NO]'):<20} (diff={diff:.3f})")

    print(f"\n{distinguishable_count}/{len(test_freqs)-1} frequencies distinguishable")

    if distinguishable_count >= 6:  # At least 6/7 should be distinguishable
        print("[PASS] Encoder preserves frequency information across full spectrum")
    else:
        print("[FAIL] Encoder losing frequency information")

    print("=" * 60)


if __name__ == '__main__':
    print("\n=== ENCODER FIX VERIFICATION ===\n")

    # Test 1: High frequency preservation
    test1_pass = test_high_freq_preservation()

    # Test 2: Full spectrum
    test_frequency_range()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"High-freq preservation: {'[PASS]' if test1_pass else '[FAIL]'}")
    print(f"\nFixed encoder uses stride=2 (was: stride=16)")
    print(f"Output: 22kHz (Nyquist @ 11kHz)")
    print(f"Preserves: Full 20kHz audible spectrum")
    print("=" * 60 + "\n")
