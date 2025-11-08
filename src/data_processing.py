import librosa
import soundfile as sf
import numpy as np
from pathlib import Path

class AudioProcessor:
    def __init__(self, target_sr=44100, segment_length=5.0):
        self.target_sr = target_sr
        self.segment_length = segment_length
        self.segment_samples = int(segment_length * target_sr)
    
    def load_pair(self, unmastered_path, mastered_path):
        """Load and align unmastered/mastered pair"""
        # Load audio
        unmastered, sr1 = librosa.load(unmastered_path, sr=self.target_sr, mono=False)
        mastered, sr2 = librosa.load(mastered_path, sr=self.target_sr, mono=False)
        
        # Ensure stereo
        if unmastered.ndim == 1:
            unmastered = np.stack([unmastered, unmastered])
        if mastered.ndim == 1:
            mastered = np.stack([mastered, mastered])
        
        # Align lengths (in case they differ slightly)
        min_length = min(unmastered.shape[1], mastered.shape[1])
        return unmastered[:, :min_length], mastered[:, :min_length]
    
    def segment_audio(self, unmastered, mastered, overlap=0.5):
        """Cut into overlapping segments"""
        hop = int(self.segment_samples * (1 - overlap))
        segments_unmastered = []
        segments_mastered = []
        
        for start in range(0, unmastered.shape[1] - self.segment_samples, hop):
            end = start + self.segment_samples
            segments_unmastered.append(unmastered[:, start:end])
            segments_mastered.append(mastered[:, start:end])
        
        return segments_unmastered, segments_mastered
    
    def augment(self, audio, aug_type='pitch'):
        """Apply augmentation"""
        if aug_type == 'pitch':
            # Pitch shift ±2 semitones
            n_steps = np.random.uniform(-2, 2)
            return librosa.effects.pitch_shift(audio, sr=self.target_sr, n_steps=n_steps)
        
        elif aug_type == 'time':
            # Time stretch 0.95x to 1.05x
            rate = np.random.uniform(0.95, 1.05)
            return librosa.effects.time_stretch(audio, rate=rate)
        
        return audio

# Usage:
processor = AudioProcessor(target_sr=44100, segment_length=5.0)