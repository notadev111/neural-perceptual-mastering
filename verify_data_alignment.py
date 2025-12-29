"""
Comprehensive data verification script.
Checks:
1. File pairing and naming alignment
2. Audio length matching
3. Sample rate consistency
4. Channel count (stereo)
5. Amplitude/RMS differences
6. Spectral content differences
7. Correlation (temporal alignment)
"""

import torch
import torchaudio
from pathlib import Path
import numpy as np
from scipy import signal as sp_signal

def compute_rms_db(audio):
    """Compute RMS in dB."""
    rms = torch.sqrt(torch.mean(audio ** 2))
    return 20 * torch.log10(rms + 1e-7).item()

def compute_cross_correlation(audio1, audio2):
    """Compute cross-correlation to check temporal alignment."""
    # Use first channel, first 44100 samples (1 second)
    sig1 = audio1[0, :44100].numpy()
    sig2 = audio2[0, :44100].numpy()

    # Compute cross-correlation
    correlation = np.correlate(sig1, sig2, mode='full')
    max_corr = np.max(np.abs(correlation))

    # Normalize
    norm = np.sqrt(np.sum(sig1**2) * np.sum(sig2**2))
    if norm > 0:
        max_corr = max_corr / norm

    return max_corr

def verify_data_alignment(data_dir):
    """Comprehensive data verification."""
    data_dir = Path(data_dir)

    print("=" * 70)
    print("AUDIO DATA ALIGNMENT VERIFICATION")
    print("=" * 70)

    # Get directories
    unmastered_dir = data_dir / 'unmastered'
    mastered_dir = data_dir / 'mastered'

    if not unmastered_dir.exists():
        print(f"\n[ERROR] Unmastered directory not found: {unmastered_dir}")
        return False
    if not mastered_dir.exists():
        print(f"\n[ERROR] Mastered directory not found: {mastered_dir}")
        return False

    # Get all files
    unmastered_files = sorted(unmastered_dir.glob('*.wav'))
    mastered_files = sorted(mastered_dir.glob('*.wav'))

    print(f"\n1. FILE COUNT CHECK")
    print(f"   Unmastered files: {len(unmastered_files)}")
    print(f"   Mastered files:   {len(mastered_files)}")

    if len(unmastered_files) != len(mastered_files):
        print(f"   [FAIL] File count mismatch!")
        return False
    else:
        print(f"   [PASS] File counts match ({len(unmastered_files)} pairs)")

    # Check file naming alignment
    print(f"\n2. FILE NAMING ALIGNMENT")
    naming_errors = 0
    for i, (um_file, m_file) in enumerate(zip(unmastered_files, mastered_files)):
        if um_file.name != m_file.name:
            print(f"   [FAIL] Mismatch at index {i}:")
            print(f"          Unmastered: {um_file.name}")
            print(f"          Mastered:   {m_file.name}")
            naming_errors += 1

    if naming_errors == 0:
        print(f"   [PASS] All {len(unmastered_files)} filenames match")
    else:
        print(f"   [FAIL] {naming_errors} naming mismatches found")
        return False

    # Detailed audio analysis
    print(f"\n3. AUDIO PROPERTIES ANALYSIS")
    print(f"   Analyzing all {len(unmastered_files)} pairs...")
    print()

    issues = []
    rms_diffs = []
    correlations = []
    spectral_diffs = []

    for i, (um_file, m_file) in enumerate(zip(unmastered_files, mastered_files)):
        # Load audio
        try:
            um_audio, um_sr = torchaudio.load(str(um_file))
            m_audio, m_sr = torchaudio.load(str(m_file))
        except Exception as e:
            issues.append(f"   [ERROR] Failed to load {um_file.name}: {e}")
            continue

        # Check 1: Sample rate
        if um_sr != 44100:
            issues.append(f"   [WARN] {um_file.name}: Unmastered SR={um_sr} (expected 44100)")
        if m_sr != 44100:
            issues.append(f"   [WARN] {um_file.name}: Mastered SR={m_sr} (expected 44100)")
        if um_sr != m_sr:
            issues.append(f"   [FAIL] {um_file.name}: Sample rate mismatch ({um_sr} vs {m_sr})")

        # Check 2: Channels (stereo)
        if um_audio.shape[0] != 2:
            issues.append(f"   [WARN] {um_file.name}: Unmastered has {um_audio.shape[0]} channels (expected 2)")
        if m_audio.shape[0] != 2:
            issues.append(f"   [WARN] {um_file.name}: Mastered has {m_audio.shape[0]} channels (expected 2)")

        # Check 3: Length matching
        um_len = um_audio.shape[-1]
        m_len = m_audio.shape[-1]
        len_diff = abs(um_len - m_len)

        if len_diff > 100:  # Allow small differences (padding/rounding)
            issues.append(f"   [FAIL] {um_file.name}: Length mismatch ({um_len} vs {m_len}, diff={len_diff})")
        elif len_diff > 0:
            issues.append(f"   [WARN] {um_file.name}: Small length diff ({len_diff} samples)")

        # Trim to same length for analysis
        min_len = min(um_len, m_len)
        um_audio = um_audio[:, :min_len]
        m_audio = m_audio[:, :min_len]

        # Check 4: RMS difference
        um_rms = compute_rms_db(um_audio)
        m_rms = compute_rms_db(m_audio)
        rms_diff = m_rms - um_rms
        rms_diffs.append(rms_diff)

        # Check 5: Temporal alignment (cross-correlation)
        if min_len >= 44100:  # At least 1 second
            corr = compute_cross_correlation(um_audio, m_audio)
            correlations.append(corr)

            if corr < 0.7:  # Low correlation suggests misalignment
                issues.append(f"   [WARN] {um_file.name}: Low correlation ({corr:.3f}), possible misalignment")

        # Check 6: Spectral difference
        if min_len >= 44100:
            um_fft = torch.abs(torch.fft.rfft(um_audio[0, :44100]))
            m_fft = torch.abs(torch.fft.rfft(m_audio[0, :44100]))
            spectral_diff = torch.mean(torch.abs(um_fft - m_fft)).item()
            spectral_diffs.append(spectral_diff)

            # Check if identical (no mastering)
            if spectral_diff < 0.001:
                issues.append(f"   [WARN] {um_file.name}: Pre/post nearly identical (diff={spectral_diff:.6f})")

        # Check 7: Completely identical files
        if torch.allclose(um_audio, m_audio, atol=1e-6):
            issues.append(f"   [FAIL] {um_file.name}: Pre and post are IDENTICAL!")

    # Print issues
    if issues:
        print("   Issues found:")
        for issue in issues[:20]:  # Show first 20 issues
            print(issue)
        if len(issues) > 20:
            print(f"   ... and {len(issues) - 20} more issues")
    else:
        print("   [PASS] No issues found in audio properties")

    # Statistical summary
    print(f"\n4. STATISTICAL SUMMARY")

    if rms_diffs:
        print(f"\n   RMS Level Differences (Mastered - Unmastered):")
        print(f"   - Mean:   {np.mean(rms_diffs):+.2f} dB")
        print(f"   - Std:    {np.std(rms_diffs):.2f} dB")
        print(f"   - Min:    {np.min(rms_diffs):+.2f} dB")
        print(f"   - Max:    {np.max(rms_diffs):+.2f} dB")

        if abs(np.mean(rms_diffs)) > 6:
            print(f"   [WARN] Large average RMS difference ({np.mean(rms_diffs):.1f} dB)")
            print(f"          Mastering may be mostly loudness boost")
        else:
            print(f"   [GOOD] Moderate RMS differences (EQ/compression, not just volume)")

        if np.std(rms_diffs) > 4:
            print(f"   [WARN] High RMS variance ({np.std(rms_diffs):.1f} dB)")
            print(f"          Inconsistent mastering across segments")

    if correlations:
        print(f"\n   Temporal Alignment (Cross-Correlation):")
        print(f"   - Mean:   {np.mean(correlations):.3f}")
        print(f"   - Min:    {np.min(correlations):.3f}")
        print(f"   - Max:    {np.max(correlations):.3f}")

        low_corr_count = sum(1 for c in correlations if c < 0.7)
        if low_corr_count > 0:
            print(f"   [WARN] {low_corr_count} pairs with correlation < 0.7")
            print(f"          These may be misaligned or very different")
        else:
            print(f"   [GOOD] All pairs well-aligned (correlation > 0.7)")

    if spectral_diffs:
        print(f"\n   Spectral Differences:")
        print(f"   - Mean:   {np.mean(spectral_diffs):.4f}")
        print(f"   - Min:    {np.min(spectral_diffs):.4f}")
        print(f"   - Max:    {np.max(spectral_diffs):.4f}")

        if np.mean(spectral_diffs) < 0.01:
            print(f"   [WARN] Very small spectral differences")
            print(f"          Model may have little to learn")
        else:
            print(f"   [GOOD] Meaningful spectral differences present")

    # Final verdict
    print(f"\n5. FINAL VERDICT")
    print("   " + "=" * 66)

    critical_issues = [i for i in issues if '[FAIL]' in i]
    warnings = [i for i in issues if '[WARN]' in i]

    if critical_issues:
        print(f"   [FAIL] {len(critical_issues)} critical issues found")
        print(f"          Fix these before training!")
        verdict = False
    elif len(warnings) > len(unmastered_files) * 0.1:  # More than 10% warnings
        print(f"   [WARN] {len(warnings)} warnings found")
        print(f"          Data may have quality issues, but training can proceed")
        print(f"          Monitor training carefully")
        verdict = True
    else:
        print(f"   [PASS] Data alignment verified!")
        print(f"          All {len(unmastered_files)} pairs are properly aligned")
        print(f"          Ready for training")
        verdict = True

    print("   " + "=" * 66)
    print()

    return verdict

if __name__ == '__main__':
    import sys

    # Default to UCL server path
    data_dir = "/home/zceeddu/neural-perceptual-mastering/data/processed/train"

    # Override with command line argument if provided
    if len(sys.argv) > 1:
        data_dir = sys.argv[1]

    # Run verification
    success = verify_data_alignment(data_dir)

    # Exit code
    sys.exit(0 if success else 1)
