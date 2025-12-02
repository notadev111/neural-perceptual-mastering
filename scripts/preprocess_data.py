"""
Preprocess audio data for training.

This script:
1. Loads pre-mastered and post-mastered audio files
2. Splits them into fixed-length segments (e.g., 5 seconds)
3. Saves segments to organized train/val/test directories
4. Preserves stereo channels throughout
"""

import os
import sys
from pathlib import Path
import torchaudio
import torch
import numpy as np

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))


def split_audio_into_segments(audio, segment_length_samples, overlap=0.0):
    """
    Split audio into fixed-length segments.

    Args:
        audio: [channels, samples] tensor
        segment_length_samples: Length of each segment in samples
        overlap: Overlap ratio (0.0 = no overlap, 0.5 = 50% overlap)

    Returns:
        List of [channels, segment_length_samples] tensors
    """
    segments = []
    hop_length = int(segment_length_samples * (1 - overlap))

    num_samples = audio.shape[-1]
    start = 0

    while start + segment_length_samples <= num_samples:
        segment = audio[:, start:start + segment_length_samples]
        segments.append(segment)
        start += hop_length

    # Optionally include last partial segment (padded)
    if start < num_samples and (num_samples - start) > segment_length_samples * 0.5:
        # Only include if at least 50% of segment length
        last_segment = audio[:, start:]
        pad_length = segment_length_samples - last_segment.shape[-1]
        last_segment = torch.nn.functional.pad(last_segment, (0, pad_length))
        segments.append(last_segment)

    return segments


def preprocess_dataset(
    pre_dir,
    post_dir,
    output_dir,
    segment_length=5.0,
    sample_rate=44100,
    overlap=0.0,
    train_split=1.0  # Use all 10 songs for training initially
):
    """
    Preprocess entire dataset.

    Args:
        pre_dir: Directory containing pre-mastered audio
        post_dir: Directory containing post-mastered audio
        output_dir: Output directory for processed segments
        segment_length: Length of segments in seconds
        sample_rate: Target sample rate
        overlap: Overlap ratio between segments
        train_split: Ratio of songs to use for training (rest for val/test)
    """
    pre_dir = Path(pre_dir)
    post_dir = Path(post_dir)
    output_dir = Path(output_dir)

    # Create output directories
    train_pre_dir = output_dir / 'train' / 'unmastered'
    train_post_dir = output_dir / 'train' / 'mastered'
    val_pre_dir = output_dir / 'val' / 'unmastered'
    val_post_dir = output_dir / 'val' / 'mastered'
    test_pre_dir = output_dir / 'test' / 'unmastered'
    test_post_dir = output_dir / 'test' / 'mastered'

    for dir_path in [train_pre_dir, train_post_dir, val_pre_dir,
                     val_post_dir, test_pre_dir, test_post_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)

    # Find all audio files
    pre_files = sorted(list(pre_dir.glob('*.wav')) + list(pre_dir.glob('*.flac')) +
                      list(pre_dir.glob('*.mp3')))
    post_files = sorted(list(post_dir.glob('*.wav')) + list(post_dir.glob('*.flac')) +
                       list(post_dir.glob('*.mp3')))

    print(f"Found {len(pre_files)} pre-mastered files")
    print(f"Found {len(post_files)} post-mastered files")

    if len(pre_files) != len(post_files):
        print(f"WARNING: Mismatch in number of files!")
        print("Pre-mastered files:", [f.name for f in pre_files])
        print("Post-mastered files:", [f.name for f in post_files])
        return

    # Match files by name
    file_pairs = []
    for pre_file in pre_files:
        # Try to find matching post file
        matching_post = None
        for post_file in post_files:
            if pre_file.stem == post_file.stem:
                matching_post = post_file
                break

        if matching_post:
            file_pairs.append((pre_file, matching_post))
        else:
            print(f"WARNING: No match found for {pre_file.name}")

    print(f"\nMatched {len(file_pairs)} file pairs")

    segment_length_samples = int(segment_length * sample_rate)

    # Split into train/val/test
    num_train = int(len(file_pairs) * train_split)

    # For initial training with all 10 songs, put everything in train
    train_pairs = file_pairs[:num_train] if num_train < len(file_pairs) else file_pairs
    val_pairs = file_pairs[num_train:] if num_train < len(file_pairs) else []
    test_pairs = []  # No test set initially

    print(f"\nDataset split:")
    print(f"  Training: {len(train_pairs)} songs")
    print(f"  Validation: {len(val_pairs)} songs")
    print(f"  Test: {len(test_pairs)} songs")

    # Process each split
    for split_name, pairs, pre_out_dir, post_out_dir in [
        ('train', train_pairs, train_pre_dir, train_post_dir),
        ('val', val_pairs, val_pre_dir, val_post_dir),
        ('test', test_pairs, test_pre_dir, test_post_dir)
    ]:
        if len(pairs) == 0:
            continue

        print(f"\nProcessing {split_name} set...")
        total_segments = 0

        for idx, (pre_file, post_file) in enumerate(pairs):
            print(f"  [{idx+1}/{len(pairs)}] {pre_file.stem}")

            try:
                # Load pre-mastered audio
                try:
                    pre_audio, pre_sr = torchaudio.load(str(pre_file))
                except Exception as e:
                    # Fallback to scipy
                    from scipy.io import wavfile
                    pre_sr, pre_data = wavfile.read(str(pre_file))
                    pre_audio = torch.from_numpy(pre_data.T).float()
                    # Normalize if int format
                    if pre_data.dtype == np.int16:
                        pre_audio = pre_audio / 32768.0
                    elif pre_data.dtype == np.int32:
                        pre_audio = pre_audio / 2147483648.0

                # Load post-mastered audio
                try:
                    post_audio, post_sr = torchaudio.load(str(post_file))
                except Exception as e:
                    # Fallback to scipy
                    from scipy.io import wavfile
                    post_sr, post_data = wavfile.read(str(post_file))
                    post_audio = torch.from_numpy(post_data.T).float()
                    # Normalize if int format
                    if post_data.dtype == np.int16:
                        post_audio = post_audio / 32768.0
                    elif post_data.dtype == np.int32:
                        post_audio = post_audio / 2147483648.0

                # Resample if needed
                if pre_sr != sample_rate:
                    resampler = torchaudio.transforms.Resample(pre_sr, sample_rate)
                    pre_audio = resampler(pre_audio)

                if post_sr != sample_rate:
                    resampler = torchaudio.transforms.Resample(post_sr, sample_rate)
                    post_audio = resampler(post_audio)

                # Ensure stereo
                if pre_audio.shape[0] == 1:
                    pre_audio = pre_audio.repeat(2, 1)
                elif pre_audio.shape[0] > 2:
                    pre_audio = pre_audio[:2, :]

                if post_audio.shape[0] == 1:
                    post_audio = post_audio.repeat(2, 1)
                elif post_audio.shape[0] > 2:
                    post_audio = post_audio[:2, :]

                # Ensure same length
                min_len = min(pre_audio.shape[-1], post_audio.shape[-1])
                pre_audio = pre_audio[:, :min_len]
                post_audio = post_audio[:, :min_len]

                # Split into segments
                pre_segments = split_audio_into_segments(
                    pre_audio, segment_length_samples, overlap
                )
                post_segments = split_audio_into_segments(
                    post_audio, segment_length_samples, overlap
                )

                print(f"    Created {len(pre_segments)} segments ({segment_length}s each)")

                # Save segments
                for seg_idx, (pre_seg, post_seg) in enumerate(zip(pre_segments, post_segments)):
                    seg_name = f"{pre_file.stem}_seg{seg_idx:04d}.wav"

                    # Save with fallback to scipy
                    try:
                        torchaudio.save(
                            str(pre_out_dir / seg_name),
                            pre_seg,
                            sample_rate
                        )
                        torchaudio.save(
                            str(post_out_dir / seg_name),
                            post_seg,
                            sample_rate
                        )
                    except Exception as e:
                        # Fallback to scipy
                        from scipy.io import wavfile
                        pre_data = (pre_seg.numpy().T * 32767).astype(np.int16)
                        post_data = (post_seg.numpy().T * 32767).astype(np.int16)
                        wavfile.write(str(pre_out_dir / seg_name), sample_rate, pre_data)
                        wavfile.write(str(post_out_dir / seg_name), sample_rate, post_data)

                total_segments += len(pre_segments)

            except Exception as e:
                print(f"    ERROR processing {pre_file.name}: {e}")
                continue

        print(f"  Total {split_name} segments: {total_segments}")

    print("\nPreprocessing complete!")
    print(f"\nOutput saved to: {output_dir}")


if __name__ == '__main__':
    # Configuration
    PRE_DIR = r"C:\Users\danie\Documents\!ELEC0030 Project\neural-perceptual-master training\pre\converted"
    POST_DIR = r"C:\Users\danie\Documents\!ELEC0030 Project\neural-perceptual-master training\post\converted"
    OUTPUT_DIR = r"C:\Users\danie\Documents\!ELEC0030 Project\neural-perceptual-master training\processed"

    SEGMENT_LENGTH = 5.0  # seconds
    SAMPLE_RATE = 44100
    OVERLAP = 0.0  # No overlap (0.0), or 50% overlap (0.5) for more data
    TRAIN_SPLIT = 1.0  # Use all 10 songs for training initially

    print("=" * 60)
    print("Audio Preprocessing for Neural Perceptual Mastering")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  Pre-mastered directory: {PRE_DIR}")
    print(f"  Post-mastered directory: {POST_DIR}")
    print(f"  Output directory: {OUTPUT_DIR}")
    print(f"  Segment length: {SEGMENT_LENGTH}s")
    print(f"  Sample rate: {SAMPLE_RATE}Hz")
    print(f"  Overlap: {OVERLAP * 100}%")
    print(f"  Training split: {TRAIN_SPLIT * 100}%")
    print()

    # Run preprocessing
    preprocess_dataset(
        pre_dir=PRE_DIR,
        post_dir=POST_DIR,
        output_dir=OUTPUT_DIR,
        segment_length=SEGMENT_LENGTH,
        sample_rate=SAMPLE_RATE,
        overlap=OVERLAP,
        train_split=TRAIN_SPLIT
    )
