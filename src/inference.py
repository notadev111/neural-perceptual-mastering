"""
Inference script for neural perceptual audio mastering.

Process unmastered audio through trained model.
"""

import torch
import torchaudio
import argparse
import yaml
from pathlib import Path
import numpy as np

from models import (
    MasteringModel_Phase1A,
    MasteringModel_Phase1B,
    MasteringModel_Phase1C
)


def load_model(checkpoint_path, config_path, device):
    """Load trained model from checkpoint."""
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    phase = config['model'].get('phase', '1B')
    
    # Create model
    if phase == '1A':
        model = MasteringModel_Phase1A(config)
    elif phase == '1B':
        model = MasteringModel_Phase1B(config)
    elif phase == '1C':
        model = MasteringModel_Phase1C(config)
    else:
        raise ValueError(f"Unknown phase: {phase}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"Loaded Phase {phase} model from epoch {checkpoint['epoch']}")
    
    return model, config


def preprocess_audio(audio_path, target_sr=44100):
    """
    Load and preprocess audio file.
    
    Args:
        audio_path: Path to audio file
        target_sr: Target sample rate
    
    Returns:
        audio: [1, 1, samples] tensor
        sr: Sample rate
        original_length: Original audio length
    """
    # Load audio
    audio, sr = torchaudio.load(audio_path)
    
    original_length = audio.shape[-1]
    
    # Resample if needed
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(sr, target_sr)
        audio = resampler(audio)
        sr = target_sr
    
    # Convert to mono
    if audio.shape[0] > 1:
        audio = torch.mean(audio, dim=0, keepdim=True)
    
    # Normalize
    max_val = torch.max(torch.abs(audio))
    if max_val > 0:
        audio = audio / max_val
    
    # Add batch dimension
    audio = audio.unsqueeze(0)  # [1, 1, samples]
    
    return audio, sr, original_length


def process_audio(model, audio, segment_length=5.0, sample_rate=44100, device='cpu'):
    """
    Process long audio by segmenting.
    
    Args:
        model: Trained model
        audio: [1, 1, samples] tensor
        segment_length: Length of segments in seconds
        sample_rate: Sample rate
        device: torch device
    
    Returns:
        output: [1, 1, samples] processed audio
        params: Dictionary of extracted parameters
    """
    segment_samples = int(segment_length * sample_rate)
    total_samples = audio.shape[-1]
    
    # If audio is shorter than segment length, process directly
    if total_samples <= segment_samples:
        audio = audio.to(device)
        with torch.no_grad():
            output, params = model(audio)
        return output.cpu(), params
    
    # Process in segments
    output_segments = []
    all_params = {'eq_frequencies': [], 'eq_gains': [], 'eq_q_factors': []}
    
    num_segments = int(np.ceil(total_samples / segment_samples))
    
    print(f"Processing {num_segments} segments...")
    
    for i in range(num_segments):
        start_idx = i * segment_samples
        end_idx = min((i + 1) * segment_samples, total_samples)
        
        # Extract segment
        segment = audio[:, :, start_idx:end_idx]
        
        # Pad if needed
        if segment.shape[-1] < segment_samples:
            pad_len = segment_samples - segment.shape[-1]
            segment = torch.nn.functional.pad(segment, (0, pad_len))
        
        # Process
        segment = segment.to(device)
        with torch.no_grad():
            output_segment, params = model(segment)
        
        # Remove padding if added
        if end_idx - start_idx < segment_samples:
            output_segment = output_segment[:, :, :end_idx - start_idx]
        
        output_segments.append(output_segment.cpu())
        
        # Store parameters
        all_params['eq_frequencies'].append(params['eq_frequencies'].cpu())
        all_params['eq_gains'].append(params['eq_gains'].cpu())
        all_params['eq_q_factors'].append(params['eq_q_factors'].cpu())
        
        print(f"  Segment {i+1}/{num_segments} complete")
    
    # Concatenate segments
    output = torch.cat(output_segments, dim=-1)
    
    # Average parameters across segments
    avg_params = {
        'eq_frequencies': torch.mean(torch.cat(all_params['eq_frequencies'], dim=0), dim=0),
        'eq_gains': torch.mean(torch.cat(all_params['eq_gains'], dim=0), dim=0),
        'eq_q_factors': torch.mean(torch.cat(all_params['eq_q_factors'], dim=0), dim=0)
    }
    
    return output, avg_params


def postprocess_audio(audio, original_sr, target_sr):
    """
    Postprocess audio (denormalize, resample back if needed).
    
    Args:
        audio: [1, 1, samples] tensor
        original_sr: Original sample rate
        target_sr: Target sample rate used during processing
    
    Returns:
        audio: [channels, samples] tensor ready to save
    """
    # Remove batch dimension
    audio = audio.squeeze(0)
    
    # Resample back to original if needed
    if original_sr != target_sr:
        resampler = torchaudio.transforms.Resample(target_sr, original_sr)
        audio = resampler(audio)
    
    # Ensure values are in [-1, 1]
    audio = torch.clamp(audio, -1.0, 1.0)
    
    return audio


def print_eq_parameters(params):
    """Print EQ parameters in human-readable format."""
    print("\n" + "="*60)
    print("EQ Parameters:")
    print("="*60)
    
    freqs = params['eq_frequencies'].numpy()
    gains = params['eq_gains'].numpy()
    qs = params['eq_q_factors'].numpy()
    
    for i, (freq, gain, q) in enumerate(zip(freqs, gains, qs)):
        print(f"Band {i+1}: {freq:.1f} Hz, {gain:+.2f} dB, Q={q:.2f}")
    
    print("="*60)


def main():
    parser = argparse.ArgumentParser(description='Process audio through trained mastering model')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to config file')
    parser.add_argument('--input', type=str, required=True,
                        help='Path to input audio file')
    parser.add_argument('--output', type=str, required=True,
                        help='Path to save output audio')
    parser.add_argument('--segment_length', type=float, default=5.0,
                        help='Segment length in seconds (default: 5.0)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    args = parser.parse_args()
    
    # Setup
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print(f"\nLoading model from {args.checkpoint}...")
    model, config = load_model(args.checkpoint, args.config, device)
    
    sample_rate = config['data']['sample_rate']
    
    # Load input audio
    print(f"\nLoading input audio: {args.input}")
    audio, sr, original_length = preprocess_audio(args.input, sample_rate)
    print(f"Audio loaded: {audio.shape[-1]/sample_rate:.2f}s, {sample_rate}Hz")
    
    # Process audio
    print("\nProcessing audio...")
    output, params = process_audio(
        model, audio, args.segment_length, sample_rate, device
    )
    
    # Print EQ parameters
    if 'eq_frequencies' in params:
        print_eq_parameters(params)
    
    # Postprocess
    output = postprocess_audio(output, sr, sample_rate)
    
    # Save output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    torchaudio.save(str(output_path), output, sr)
    print(f"\nOutput saved to: {output_path}")
    
    # Statistics
    print("\nProcessing Statistics:")
    print(f"  Input peak: {torch.max(torch.abs(audio)).item():.4f}")
    print(f"  Output peak: {torch.max(torch.abs(output)).item():.4f}")
    print(f"  Duration: {output.shape[-1]/sr:.2f}s")


if __name__ == '__main__':
    main()
