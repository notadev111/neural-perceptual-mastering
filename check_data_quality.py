"""
Diagnostic script to check data quality and identify training issues.
"""

import torch
import torchaudio
from pathlib import Path
import numpy as np

def compute_rms_db(audio):
    """Compute RMS in dB."""
    rms = torch.sqrt(torch.mean(audio ** 2))
    return 20 * torch.log10(rms + 1e-7).item()

def check_data_directory(data_dir):
    """Check data quality."""
    data_dir = Path(data_dir)

    print("=" * 60)
    print("DATA QUALITY CHECK")
    print("=" * 60)

    # Check directory structure
    unmastered_dir = data_dir / 'unmastered'
    mastered_dir = data_dir / 'mastered'

    if not unmastered_dir.exists():
        print(f"\n[ERROR] Unmastered directory not found: {unmastered_dir}")
        return
    if not mastered_dir.exists():
        print(f"\n[ERROR] Mastered directory not found: {mastered_dir}")
        return

    # Count files
    unmastered_files = sorted(unmastered_dir.glob('*.wav'))
    mastered_files = sorted(mastered_dir.glob('*.wav'))

    print(f"\n1. FILE COUNT")
    print(f"   Unmastered: {len(unmastered_files)}")
    print(f"   Mastered: {len(mastered_files)}")

    if len(unmastered_files) != len(mastered_files):
        print(f"   [WARNING] Mismatch in file count!")
    else:
        print(f"   [OK] File counts match")

    # Check first 5 pairs
    print(f"\n2. FILE PAIRING (first 5)")
    for i in range(min(5, len(unmastered_files))):
        um_name = unmastered_files[i].name
        m_name = mastered_files[i].name
        match = "✓" if um_name == m_name else "✗"
        print(f"   {match} {um_name} <-> {m_name}")

    # Analyze audio properties
    print(f"\n3. AUDIO ANALYSIS (sampling 5 random pairs)")

    sample_indices = np.random.choice(len(unmastered_files), min(5, len(unmastered_files)), replace=False)

    rms_diffs = []
    spectral_diffs = []

    for idx in sample_indices:
        um_file = unmastered_files[idx]
        m_file = mastered_files[idx]

        # Load audio
        um_audio, um_sr = torchaudio.load(str(um_file))
        m_audio, m_sr = torchaudio.load(str(m_file))

        # Check properties
        print(f"\n   File: {um_file.name}")
        print(f"   - Unmastered: {um_audio.shape} @ {um_sr}Hz")
        print(f"   - Mastered:   {m_audio.shape} @ {m_sr}Hz")

        # RMS difference
        um_rms = compute_rms_db(um_audio)
        m_rms = compute_rms_db(m_audio)
        rms_diff = m_rms - um_rms
        rms_diffs.append(rms_diff)

        print(f"   - RMS (dB): unmastered={um_rms:.1f}, mastered={m_rms:.1f}, diff={rms_diff:.1f}")

        # Spectral difference (simple FFT comparison)
        um_fft = torch.abs(torch.fft.rfft(um_audio[0, :44100]))  # First second
        m_fft = torch.abs(torch.fft.rfft(m_audio[0, :44100]))
        spectral_diff = torch.mean(torch.abs(um_fft - m_fft)).item()
        spectral_diffs.append(spectral_diff)

        print(f"   - Spectral diff: {spectral_diff:.4f}")

        # Check if they're identical
        if torch.allclose(um_audio, m_audio, atol=1e-5):
            print(f"   [WARNING] Pre and post are IDENTICAL!")

    # Summary statistics
    print(f"\n4. SUMMARY STATISTICS")
    print(f"   Average RMS difference: {np.mean(rms_diffs):.2f} dB (± {np.std(rms_diffs):.2f})")
    print(f"   Average spectral difference: {np.mean(spectral_diffs):.4f}")

    # Diagnose issues
    print(f"\n5. DIAGNOSIS")

    if np.mean(rms_diffs) > 5:
        print(f"   [WARNING] Large RMS difference ({np.mean(rms_diffs):.1f} dB)")
        print(f"             This suggests mastering is mostly just loudness boost")
        print(f"             Model will learn volume, not EQ/dynamics")
    elif np.abs(np.mean(rms_diffs)) < 0.5:
        print(f"   [OK] RMS differences are small ({np.mean(rms_diffs):.1f} dB)")
        print(f"        LUFS normalization will force model to learn EQ")

    if np.std(rms_diffs) > 3:
        print(f"   [WARNING] High variance in RMS differences ({np.std(rms_diffs):.1f} dB)")
        print(f"             Inconsistent mastering across segments")
        print(f"             Will cause training instability")

    if np.mean(spectral_diffs) < 0.01:
        print(f"   [WARNING] Very small spectral differences")
        print(f"             Pre/post might be too similar")
        print(f"             Model has nothing to learn")

    # Check diversity (are all segments from same song?)
    print(f"\n6. DIVERSITY CHECK")
    unique_prefixes = set()
    for f in unmastered_files[:10]:
        # Extract base name (before segment number)
        # Assuming naming like "song_segment_001.wav"
        prefix = '_'.join(f.stem.split('_')[:-1]) if '_' in f.stem else f.stem
        unique_prefixes.add(prefix)

    print(f"   Detected {len(unique_prefixes)} unique song prefixes in first 10 files")
    if len(unique_prefixes) == 1:
        print(f"   [WARNING] All segments appear to be from the SAME SONG!")
        print(f"             Need diverse training data (multiple songs/genres)")
        print(f"             Current data will cause severe overfitting")
    else:
        print(f"   [OK] Multiple songs detected")

    print("\n" + "=" * 60)

if __name__ == '__main__':
    import sys

    # Default to UCL server path
    data_dir = "/home/zceeddu/neural-perceptual-mastering/data/processed/train"

    # Or use local path if provided
    if len(sys.argv) > 1:
        data_dir = sys.argv[1]

    check_data_quality(data_dir)
