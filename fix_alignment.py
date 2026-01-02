"""
Auto-align pre/post mastering pairs using cross-correlation.
Fixes time offsets by shifting audio to align precisely.
"""

import torch
import torchaudio
from pathlib import Path
import numpy as np
from scipy import signal
import shutil
from tqdm import tqdm

def find_alignment_offset(audio1, audio2, sr=44100):
    """
    Find optimal alignment offset using cross-correlation.

    Returns:
        offset_samples: Number of samples to shift audio2 to align with audio1
    """
    # Use first channel, up to 10 seconds for alignment
    max_samples = min(sr * 10, audio1.shape[-1], audio2.shape[-1])
    sig1 = audio1[0, :max_samples].numpy()
    sig2 = audio2[0, :max_samples].numpy()

    # Normalize
    sig1 = sig1 / (np.std(sig1) + 1e-8)
    sig2 = sig2 / (np.std(sig2) + 1e-8)

    # Cross-correlation
    correlation = signal.correlate(sig1, sig2, mode='same')

    # Find peak
    center = len(correlation) // 2
    peak_idx = np.argmax(np.abs(correlation))
    offset_samples = peak_idx - center

    return offset_samples

def align_audio(audio, offset_samples):
    """
    Shift audio by offset_samples to align.

    Args:
        audio: [channels, samples] tensor
        offset_samples: Number of samples to shift
                       Positive = shift right (delay)
                       Negative = shift left (advance)

    Returns:
        aligned_audio: [channels, samples] tensor
    """
    if offset_samples == 0:
        return audio

    channels, samples = audio.shape

    if offset_samples > 0:
        # Shift right (audio is ahead, need to delay it)
        # Pad at start, trim at end
        aligned = torch.cat([
            torch.zeros(channels, offset_samples),
            audio[:, :-offset_samples]
        ], dim=1)
    else:
        # Shift left (audio is behind, need to advance it)
        # Trim at start, pad at end
        offset_samples = abs(offset_samples)
        aligned = torch.cat([
            audio[:, offset_samples:],
            torch.zeros(channels, offset_samples)
        ], dim=1)

    return aligned

def fix_data_alignment(data_dir, backup=True, threshold_ms=1.0):
    """
    Auto-align all pre/post mastering pairs.

    Args:
        data_dir: Path to data directory
        backup: If True, backup original files before fixing
        threshold_ms: Only fix pairs with offset > threshold (default: 1ms)
    """
    data_dir = Path(data_dir)

    print("=" * 70)
    print("AUTO-ALIGNMENT FIX")
    print("=" * 70)

    unmastered_dir = data_dir / 'unmastered'
    mastered_dir = data_dir / 'mastered'

    if not unmastered_dir.exists() or not mastered_dir.exists():
        print("\n[ERROR] Data directories not found")
        return

    # Backup original files
    if backup:
        backup_dir = data_dir / 'backup_before_alignment'
        backup_dir.mkdir(exist_ok=True)

        backup_unmastered = backup_dir / 'unmastered'
        backup_mastered = backup_dir / 'mastered'

        print(f"\n[BACKUP] Creating backup at: {backup_dir}")

        if not backup_unmastered.exists():
            shutil.copytree(unmastered_dir, backup_unmastered)
            print(f"  Backed up unmastered files")

        if not backup_mastered.exists():
            shutil.copytree(mastered_dir, backup_mastered)
            print(f"  Backed up mastered files")

    # Get all file pairs
    unmastered_files = sorted(unmastered_dir.glob('*.wav'))
    mastered_files = sorted(mastered_dir.glob('*.wav'))

    print(f"\n[PROCESSING] Aligning {len(unmastered_files)} pairs...")
    print(f"  Threshold: {threshold_ms}ms")
    print()

    fixed_count = 0
    skipped_count = 0

    for um_file, m_file in tqdm(zip(unmastered_files, mastered_files), total=len(unmastered_files)):
        # Load audio
        try:
            um_audio, um_sr = torchaudio.load(str(um_file))
            m_audio, m_sr = torchaudio.load(str(m_file))
        except Exception as e:
            print(f"\n[ERROR] Failed to load {um_file.name}: {e}")
            continue

        # Find offset
        offset_samples = find_alignment_offset(um_audio, m_audio, um_sr)
        offset_ms = abs(offset_samples / um_sr * 1000)

        # Skip if offset is small
        if offset_ms < threshold_ms:
            skipped_count += 1
            continue

        # Align mastered audio to match unmastered
        # We align the mastered (post) to the unmastered (pre) as reference
        m_audio_aligned = align_audio(m_audio, offset_samples)

        # Ensure same length after alignment
        min_len = min(um_audio.shape[-1], m_audio_aligned.shape[-1])
        um_audio = um_audio[:, :min_len]
        m_audio_aligned = m_audio_aligned[:, :min_len]

        # Save aligned mastered file
        torchaudio.save(str(m_file), m_audio_aligned, um_sr)

        fixed_count += 1

    print(f"\n[DONE] Alignment complete!")
    print(f"  Fixed: {fixed_count} pairs")
    print(f"  Skipped (already aligned): {skipped_count} pairs")
    print(f"  Total: {len(unmastered_files)} pairs")

    # Verify alignment
    print(f"\n[VERIFY] Re-checking alignment...")

    offsets_after = []
    for um_file, m_file in unmastered_files[:20]:  # Check first 20
        try:
            um_audio, um_sr = torchaudio.load(str(um_file))
            m_audio, m_sr = torchaudio.load(str(m_file))

            offset = find_alignment_offset(um_audio, m_audio, um_sr)
            offset_ms = abs(offset / um_sr * 1000)
            offsets_after.append(offset_ms)
        except:
            continue

    if offsets_after:
        mean_offset_after = np.mean(offsets_after)
        max_offset_after = np.max(offsets_after)

        print(f"  Mean offset after fix: {mean_offset_after:.3f}ms")
        print(f"  Max offset after fix:  {max_offset_after:.3f}ms")

        if mean_offset_after < 1.0:
            print(f"\n[SUCCESS] Alignment fixed successfully!")
        else:
            print(f"\n[WARN] Some pairs still misaligned")

    print("=" * 70)

    if backup:
        print(f"\n[INFO] Original files backed up to: {backup_dir}")
        print(f"       To restore: copy files from backup back to data/")

    print()

if __name__ == '__main__':
    import sys

    # Default to UCL server path
    data_dir = "/home/zceeddu/neural-perceptual-mastering/data/processed/train"

    if len(sys.argv) > 1:
        data_dir = sys.argv[1]

    # Get user confirmation
    print("=" * 70)
    print("This script will:")
    print("  1. Backup your original files to backup_before_alignment/")
    print("  2. Auto-align mastered files to match unmastered files")
    print("  3. Overwrite the mastered files with aligned versions")
    print("=" * 70)

    response = input("\nProceed with alignment fix? (yes/no): ")

    if response.lower() in ['yes', 'y']:
        fix_data_alignment(data_dir, backup=True, threshold_ms=1.0)
    else:
        print("Aborted.")
