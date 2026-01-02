"""
Precise time alignment check for pre/post mastering pairs.
Detects even microsecond-level time offsets using cross-correlation.
"""

import torch
import torchaudio
from pathlib import Path
import numpy as np
from scipy import signal

def find_time_offset(audio1, audio2, sr=44100):
    """
    Find time offset between two audio signals using cross-correlation.

    Returns:
        offset_samples: Number of samples audio2 is shifted relative to audio1
                       Positive = audio2 is delayed
                       Negative = audio2 is ahead
        offset_ms: Offset in milliseconds
        correlation_peak: Peak correlation value (0-1, higher = better aligned)
    """
    # Use first channel, first 5 seconds for alignment check
    max_samples = min(sr * 5, audio1.shape[-1], audio2.shape[-1])
    sig1 = audio1[0, :max_samples].numpy()
    sig2 = audio2[0, :max_samples].numpy()

    # Normalize signals
    sig1 = sig1 / (np.std(sig1) + 1e-8)
    sig2 = sig2 / (np.std(sig2) + 1e-8)

    # Compute cross-correlation
    correlation = signal.correlate(sig1, sig2, mode='same')

    # Find peak
    center = len(correlation) // 2
    peak_idx = np.argmax(np.abs(correlation))
    offset_samples = peak_idx - center

    # Convert to milliseconds
    offset_ms = (offset_samples / sr) * 1000

    # Get correlation strength at peak
    correlation_peak = np.abs(correlation[peak_idx]) / len(sig1)

    return offset_samples, offset_ms, correlation_peak

def check_alignment(data_dir):
    """Check time alignment of all pre/post pairs."""
    data_dir = Path(data_dir)

    print("=" * 70)
    print("PRECISE TIME ALIGNMENT CHECK")
    print("=" * 70)

    unmastered_dir = data_dir / 'unmastered'
    mastered_dir = data_dir / 'mastered'

    if not unmastered_dir.exists() or not mastered_dir.exists():
        print("\n[ERROR] Data directories not found")
        return

    unmastered_files = sorted(unmastered_dir.glob('*.wav'))
    mastered_files = sorted(mastered_dir.glob('*.wav'))

    print(f"\nAnalyzing {len(unmastered_files)} pairs for time alignment...")
    print()

    offsets_samples = []
    offsets_ms = []
    correlations = []
    misaligned_pairs = []

    for um_file, m_file in zip(unmastered_files, mastered_files):
        # Load audio
        try:
            um_audio, um_sr = torchaudio.load(str(um_file))
            m_audio, m_sr = torchaudio.load(str(m_file))
        except Exception as e:
            print(f"[ERROR] Failed to load {um_file.name}: {e}")
            continue

        # Check sample rates match
        if um_sr != m_sr:
            print(f"[WARN] {um_file.name}: Sample rate mismatch ({um_sr} vs {m_sr})")
            continue

        # Find time offset
        offset_samples, offset_ms, corr_peak = find_time_offset(um_audio, m_audio, um_sr)

        offsets_samples.append(offset_samples)
        offsets_ms.append(offset_ms)
        correlations.append(corr_peak)

        # Flag significant misalignments
        # Threshold: >1ms offset OR low correlation
        if abs(offset_ms) > 1.0 or corr_peak < 0.7:
            misaligned_pairs.append({
                'file': um_file.name,
                'offset_ms': offset_ms,
                'offset_samples': offset_samples,
                'correlation': corr_peak
            })

    # Summary statistics
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)

    if not offsets_ms:
        print("\n[ERROR] No valid pairs analyzed")
        return

    print(f"\nTime Offset Statistics (Total: {len(offsets_ms)} pairs)")
    print(f"  Mean offset:    {np.mean(offsets_ms):+.3f} ms  ({np.mean(offsets_samples):+.1f} samples)")
    print(f"  Std deviation:  {np.std(offsets_ms):.3f} ms  ({np.std(offsets_samples):.1f} samples)")
    print(f"  Min offset:     {np.min(offsets_ms):+.3f} ms  ({np.min(offsets_samples):+.0f} samples)")
    print(f"  Max offset:     {np.max(offsets_ms):+.3f} ms  ({np.max(offsets_samples):+.0f} samples)")

    print(f"\nCorrelation Statistics")
    print(f"  Mean:  {np.mean(correlations):.3f}")
    print(f"  Min:   {np.min(correlations):.3f}")
    print(f"  Max:   {np.max(correlations):.3f}")

    # Categorize alignment quality
    print(f"\nAlignment Quality Breakdown")
    perfect = sum(1 for o in offsets_ms if abs(o) < 0.1)  # <0.1ms = perfect
    good = sum(1 for o in offsets_ms if 0.1 <= abs(o) < 1.0)  # <1ms = good
    acceptable = sum(1 for o in offsets_ms if 1.0 <= abs(o) < 5.0)  # <5ms = acceptable
    bad = sum(1 for o in offsets_ms if abs(o) >= 5.0)  # >=5ms = bad

    print(f"  Perfect (<0.1ms):     {perfect} ({perfect/len(offsets_ms)*100:.1f}%)")
    print(f"  Good (0.1-1ms):       {good} ({good/len(offsets_ms)*100:.1f}%)")
    print(f"  Acceptable (1-5ms):   {acceptable} ({acceptable/len(offsets_ms)*100:.1f}%)")
    print(f"  Bad (>5ms):           {bad} ({bad/len(offsets_ms)*100:.1f}%)")

    # Report problematic pairs
    if misaligned_pairs:
        print(f"\n[WARN] Found {len(misaligned_pairs)} potentially misaligned pairs:")
        print(f"{'File':<40} {'Offset (ms)':<15} {'Offset (samples)':<20} {'Correlation':<15}")
        print("-" * 90)

        # Sort by worst offset first
        misaligned_pairs.sort(key=lambda x: abs(x['offset_ms']), reverse=True)

        for pair in misaligned_pairs[:20]:  # Show worst 20
            print(f"{pair['file']:<40} {pair['offset_ms']:+8.2f}      "
                  f"{pair['offset_samples']:+8.0f}            {pair['correlation']:.3f}")

        if len(misaligned_pairs) > 20:
            print(f"... and {len(misaligned_pairs) - 20} more")

    # Final verdict
    print(f"\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)

    max_abs_offset = np.max(np.abs(offsets_ms))
    mean_abs_offset = np.mean(np.abs(offsets_ms))

    if max_abs_offset < 1.0 and mean_abs_offset < 0.5:
        print("[PASS] Excellent alignment!")
        print(f"       All pairs aligned within 1ms")
        print(f"       Mean offset: {mean_abs_offset:.3f}ms")
        verdict = True
    elif max_abs_offset < 5.0 and mean_abs_offset < 2.0:
        print("[ACCEPTABLE] Alignment is acceptable for training")
        print(f"       Max offset: {max_abs_offset:.2f}ms")
        print(f"       Mean offset: {mean_abs_offset:.3f}ms")
        print(f"       Training should work, but may learn time-shift artifacts")
        verdict = True
    else:
        print("[WARN] Significant time misalignment detected")
        print(f"       Max offset: {max_abs_offset:.2f}ms")
        print(f"       Mean offset: {mean_abs_offset:.3f}ms")
        print(f"       {len(misaligned_pairs)} pairs have issues")
        print(f"       Consider re-aligning or removing problematic pairs")
        verdict = False

    print("=" * 70)

    # Recommendations
    print("\nRECOMMENDATIONS:")

    if mean_abs_offset < 0.5:
        print("  [OK] Alignment is excellent - proceed with training")
    elif mean_abs_offset < 2.0:
        print("  [OK] Alignment is good enough - training should work")
        print("       Model may learn minor time-shift artifacts")
    else:
        print("  [ACTION] Consider fixing alignment:")
        print("          1. Use cross-correlation to auto-align pairs")
        print("          2. Manually check worst offenders")
        print("          3. Remove severely misaligned pairs")

    if np.mean(correlations) < 0.7:
        print(f"  [INFO] Low correlation (mean={np.mean(correlations):.3f}) is likely due to:")
        print("         - Heavy EQ/compression (GOOD - model has work to learn)")
        print("         - NOT time misalignment if offsets are small")

    print()
    return verdict

if __name__ == '__main__':
    import sys

    # Default to UCL server path
    data_dir = "/home/zceeddu/neural-perceptual-mastering/data/processed/train"

    if len(sys.argv) > 1:
        data_dir = sys.argv[1]

    check_alignment(data_dir)
