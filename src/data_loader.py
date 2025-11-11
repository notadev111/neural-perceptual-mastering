"""
Data loading for neural perceptual audio mastering.

Self-supervised learning:
- Segments long tracks into 5-10s chunks
- Pre/post mastering pairs
- Audio augmentation during training
"""

import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio
import os
import random
from pathlib import Path


class MasteringDataset(Dataset):
    """
    Dataset of pre/post mastering pairs.
    
    Expects directory structure:
    data/
        unmastered/
            song1.wav
            song2.wav
            ...
        mastered/
            song1.wav
            song2.wav
            ...
    
    Each file should be 44.1kHz stereo (will be converted to mono).
    """
    def __init__(self, data_dir, segment_length=5.0, sample_rate=44100,
                 augment=False, subset='train', normalize_lufs=True, target_lufs=-14.0):
        """
        Args:
            data_dir: Path to data directory
            segment_length: Length of audio segments in seconds
            sample_rate: Target sample rate (44.1kHz)
            augment: Whether to apply data augmentation
            subset: 'train', 'val', or 'test'
            normalize_lufs: If True, normalize both pre/post to same LUFS
                           This forces model to learn EQ/dynamics, not just volume
            target_lufs: Target LUFS level in dB (default: -14.0 for streaming)
        """
        self.data_dir = Path(data_dir)
        self.segment_length = segment_length
        self.sample_rate = sample_rate
        self.segment_samples = int(segment_length * sample_rate)
        self.augment = augment
        self.subset = subset
        self.normalize_lufs = normalize_lufs
        self.target_lufs = target_lufs
        
        # Find all audio files
        self.unmastered_files = sorted(
            (self.data_dir / 'unmastered').glob('*.wav')
        )
        self.mastered_files = sorted(
            (self.data_dir / 'mastered').glob('*.wav')
        )
        
        # Verify pairs match
        assert len(self.unmastered_files) == len(self.mastered_files), \
            "Mismatch between unmastered and mastered files"
        
        for um_file, m_file in zip(self.unmastered_files, self.mastered_files):
            assert um_file.stem == m_file.stem, \
                f"File pair mismatch: {um_file.stem} != {m_file.stem}"
        
        print(f"Found {len(self.unmastered_files)} audio pairs for {subset}")
        
        # Load all audio into memory (if dataset is small)
        # For large datasets, load on-the-fly instead
        self.audio_pairs = []
        self._load_audio()
        
    def _load_audio(self):
        """Load all audio pairs into memory."""
        print(f"Loading audio files for {self.subset}...")
        
        for um_file, m_file in zip(self.unmastered_files, self.mastered_files):
            # Load unmastered
            um_audio, um_sr = torchaudio.load(um_file)
            um_audio = self._preprocess(um_audio, um_sr)
            
            # Load mastered
            m_audio, m_sr = torchaudio.load(m_file)
            m_audio = self._preprocess(m_audio, m_sr)
            
            # Ensure same length
            min_len = min(um_audio.shape[-1], m_audio.shape[-1])
            um_audio = um_audio[:, :min_len]
            m_audio = m_audio[:, :min_len]
            
            self.audio_pairs.append((um_audio, m_audio, um_file.stem))
        
        print(f"Loaded {len(self.audio_pairs)} audio pairs")
    
    def _compute_lufs(self, audio):
        """
        Compute approximate LUFS (simplified ITU-R BS.1770).
        
        Args:
            audio: [1, samples] tensor
        Returns:
            lufs: float (dB)
        """
        # RMS over entire signal (simplified LUFS)
        rms = torch.sqrt(torch.mean(audio ** 2))
        
        # Convert to dB (with small epsilon to avoid log(0))
        lufs_db = 20 * torch.log10(rms + 1e-7)
        
        return lufs_db.item()
    
    def _normalize_lufs(self, audio, target_lufs=-14.0):
        """
        Normalize audio to target LUFS level.
        
        This ensures pre/post mastering pairs are at the same loudness,
        forcing the model to learn EQ/compression/etc. rather than just volume.
        
        Args:
            audio: [1, samples] tensor
            target_lufs: Target LUFS in dB (default: -14.0 for streaming)
        Returns:
            normalized_audio: [1, samples] tensor
        """
        # Compute current LUFS
        current_lufs = self._compute_lufs(audio)
        
        # Calculate gain needed to reach target
        gain_db = target_lufs - current_lufs
        gain_linear = 10 ** (gain_db / 20)
        
        # Apply gain
        normalized = audio * gain_linear
        
        # Safety limiter (prevent clipping)
        max_val = torch.max(torch.abs(normalized))
        if max_val > 0.99:  # Leave 1% headroom
            normalized = normalized * (0.99 / max_val)
        
        return normalized
    
    def _preprocess(self, audio, sr):
        """
        Preprocess audio:
        - Resample to target sample rate
        - Convert to mono
        - LUFS normalize (if enabled)
        - Peak normalize (if LUFS not enabled)
        """
        # Resample if needed
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            audio = resampler(audio)
        
        # Convert to mono
        if audio.shape[0] > 1:
            audio = torch.mean(audio, dim=0, keepdim=True)
        
        # LUFS normalization (recommended for training)
        if hasattr(self, 'normalize_lufs') and self.normalize_lufs:
            audio = self._normalize_lufs(audio, target_lufs=self.target_lufs)
        else:
            # Peak normalization (fallback)
            max_val = torch.max(torch.abs(audio))
            if max_val > 0:
                audio = audio / max_val
        
        return audio
    
    def _augment(self, unmastered, mastered):
        """
        Apply data augmentation (random gain, polarity flip).
        
        Args:
            unmastered: [1, samples]
            mastered: [1, samples]
        """
        # Random gain (-3dB to +3dB)
        if random.random() < 0.5:
            gain_db = random.uniform(-3, 3)
            gain_linear = 10 ** (gain_db / 20)
            unmastered = unmastered * gain_linear
            mastered = mastered * gain_linear
        
        # Random polarity flip
        if random.random() < 0.3:
            unmastered = -unmastered
            mastered = -mastered
        
        return unmastered, mastered
    
    def __len__(self):
        """
        Return total number of segments across all songs.
        
        Each song can yield multiple segments.
        """
        total_segments = 0
        for um_audio, _, _ in self.audio_pairs:
            num_segments = um_audio.shape[-1] // self.segment_samples
            total_segments += max(1, num_segments)
        
        return total_segments
    
    def __getitem__(self, idx):
        """
        Get a random segment from a random song.
        
        For efficiency, we sample randomly rather than deterministically.
        
        Returns:
            unmastered: [1, segment_samples]
            mastered: [1, segment_samples]
        """
        # Randomly select a song
        song_idx = random.randint(0, len(self.audio_pairs) - 1)
        um_audio, m_audio, song_name = self.audio_pairs[song_idx]
        
        # Randomly select a segment
        max_start = max(0, um_audio.shape[-1] - self.segment_samples)
        start_idx = random.randint(0, max_start) if max_start > 0 else 0
        end_idx = start_idx + self.segment_samples
        
        # Extract segment
        um_segment = um_audio[:, start_idx:end_idx]
        m_segment = m_audio[:, start_idx:end_idx]
        
        # Pad if needed (for last segment)
        if um_segment.shape[-1] < self.segment_samples:
            pad_len = self.segment_samples - um_segment.shape[-1]
            um_segment = F.pad(um_segment, (0, pad_len))
            m_segment = F.pad(m_segment, (0, pad_len))
        
        # Augmentation
        if self.augment and self.subset == 'train':
            um_segment, m_segment = self._augment(um_segment, m_segment)
        
        return um_segment, m_segment


def get_dataloaders(config):
    """
    Create train, validation, and test dataloaders.
    
    Args:
        config: Configuration dictionary
    
    Returns:
        train_loader, val_loader, test_loader
    """
    data_dir = config['data']['data_dir']
    segment_length = config['data']['segment_length']
    sample_rate = config['data']['sample_rate']
    batch_size = config['training']['batch_size']
    num_workers = config['data'].get('num_workers', 4)
    
    # LUFS normalization settings
    normalize_lufs = config['data'].get('normalize_lufs', True)  # Default: True
    target_lufs = config['data'].get('target_lufs', -14.0)       # Default: -14 LUFS
    
    # Create datasets
    train_dataset = MasteringDataset(
        data_dir=data_dir,
        segment_length=segment_length,
        sample_rate=sample_rate,
        augment=config['data']['augment'],
        subset='train',
        normalize_lufs=normalize_lufs,
        target_lufs=target_lufs
    )
    
    val_dataset = MasteringDataset(
        data_dir=data_dir,
        segment_length=segment_length,
        sample_rate=sample_rate,
        augment=False,
        subset='val',
        normalize_lufs=normalize_lufs,
        target_lufs=target_lufs
    )
    
    test_dataset = MasteringDataset(
        data_dir=data_dir,
        segment_length=segment_length,
        sample_rate=sample_rate,
        augment=False,
        subset='test',
        normalize_lufs=normalize_lufs,
        target_lufs=target_lufs
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


def create_dummy_dataset(output_dir, num_songs=3, duration=30.0, sample_rate=44100):
    """
    Create a dummy dataset for testing.
    
    Args:
        output_dir: Where to save dummy files
        num_songs: Number of song pairs to create
        duration: Duration of each song in seconds
        sample_rate: Sample rate
    """
    import numpy as np
    
    output_dir = Path(output_dir)
    (output_dir / 'unmastered').mkdir(parents=True, exist_ok=True)
    (output_dir / 'mastered').mkdir(parents=True, exist_ok=True)
    
    print(f"Creating {num_songs} dummy song pairs...")
    
    for i in range(num_songs):
        # Generate random audio
        samples = int(duration * sample_rate)
        
        # Unmastered: raw audio with less processing
        unmastered = np.random.randn(2, samples).astype(np.float32) * 0.3
        
        # Mastered: simulate compression + EQ boost
        mastered = unmastered * 1.5  # Louder
        mastered = np.tanh(mastered * 2) / 2  # Compression
        
        # Save
        um_path = output_dir / 'unmastered' / f'song{i+1:03d}.wav'
        m_path = output_dir / 'mastered' / f'song{i+1:03d}.wav'
        
        torchaudio.save(str(um_path), torch.tensor(unmastered), sample_rate)
        torchaudio.save(str(m_path), torch.tensor(mastered), sample_rate)
    
    print(f"Dummy dataset created at {output_dir}")


if __name__ == '__main__':
    # Test dataset creation
    print("Creating dummy dataset for testing...")
    create_dummy_dataset('data', num_songs=3, duration=30.0)
    
    # Test dataloader
    print("\nTesting dataloader...")
    config = {
        'data': {
            'data_dir': 'data',
            'segment_length': 5.0,
            'sample_rate': 44100,
            'augment': True,
            'num_workers': 0
        },
        'training': {
            'batch_size': 2
        }
    }
    
    train_loader, val_loader, test_loader = get_dataloaders(config)
    
    # Get one batch
    unmastered, mastered = next(iter(train_loader))
    print(f"Batch shapes: {unmastered.shape}, {mastered.shape}")
    print("Dataloader test successful!")
