"""
Evaluation metrics for neural perceptual audio mastering.

Metrics:
- Multi-scale STFT distance
- Mel spectral distance (perceptual)
- LUFS/loudness matching
- A-weighted error
- Signal-to-Noise Ratio (SNR)
- EQ parameter analysis
"""

import torch
import torch.nn.functional as F
import torchaudio
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from models import (
    MasteringModel_Phase1A,
    MasteringModel_Phase1B,
    MasteringModel_Phase1C
)
from data_loader import get_dataloaders
from losses import MultiScaleSTFTLoss, LUFSLoss, AWeightedLoss


def compute_snr(pred, target):
    """
    Compute Signal-to-Noise Ratio in dB.
    
    Args:
        pred: [batch, 1, samples]
        target: [batch, 1, samples]
    
    Returns:
        snr_db: [batch]
    """
    signal_power = torch.mean(target ** 2, dim=-1)
    noise_power = torch.mean((pred - target) ** 2, dim=-1)
    
    snr = signal_power / (noise_power + 1e-7)
    snr_db = 10 * torch.log10(snr + 1e-7)
    
    return snr_db.squeeze(1)


def compute_mel_distance(pred, target, sample_rate=44100, n_mels=128):
    """
    Compute mel-frequency spectral distance.
    
    Perceptual metric from Välimäki et al. 2016.
    
    Args:
        pred: [batch, 1, samples]
        target: [batch, 1, samples]
    
    Returns:
        mel_dist: [batch]
    """
    n_fft = 2048
    
    # Mel spectrogram transform
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        n_mels=n_mels
    ).to(pred.device)
    
    # Compute mel spectrograms
    pred_mel = mel_transform(pred)
    target_mel = mel_transform(target)
    
    # Log mel spectrograms
    pred_log_mel = torch.log(pred_mel + 1e-7)
    target_log_mel = torch.log(target_mel + 1e-7)
    
    # L2 distance
    mel_dist = torch.sqrt(torch.mean((pred_log_mel - target_log_mel) ** 2, dim=(1, 2)))
    
    return mel_dist


def analyze_eq_parameters(model, data_loader, device, phase):
    """
    Analyze EQ parameters learned by the model.
    
    Useful for understanding what the model is learning.
    
    Args:
        model: Trained model
        data_loader: DataLoader
        device: torch device
        phase: Model phase (1A, 1B, 1C)
    
    Returns:
        stats: Dictionary of statistics
    """
    model.eval()
    
    all_freqs = []
    all_gains = []
    all_qs = []
    all_band_weights = []
    
    with torch.no_grad():
        for unmastered, mastered in tqdm(data_loader, desc='Analyzing EQ'):
            unmastered = unmastered.to(device)
            
            output, params = model(unmastered)
            
            all_freqs.append(params['eq_frequencies'].cpu())
            all_gains.append(params['eq_gains'].cpu())
            all_qs.append(params['eq_q_factors'].cpu())
            
            if 'band_weights' in params:
                all_band_weights.append(params['band_weights'].cpu())
    
    # Concatenate
    all_freqs = torch.cat(all_freqs, dim=0)  # [N, num_bands]
    all_gains = torch.cat(all_gains, dim=0)
    all_qs = torch.cat(all_qs, dim=0)
    
    # Statistics
    stats = {
        'freq_mean': all_freqs.mean(dim=0).numpy(),
        'freq_std': all_freqs.std(dim=0).numpy(),
        'gain_mean': all_gains.mean(dim=0).numpy(),
        'gain_std': all_gains.std(dim=0).numpy(),
        'q_mean': all_qs.mean(dim=0).numpy(),
        'q_std': all_qs.std(dim=0).numpy(),
    }
    
    if all_band_weights:
        all_band_weights = torch.cat(all_band_weights, dim=0)
        stats['band_weights_mean'] = all_band_weights.mean(dim=0).numpy()
        stats['band_weights_std'] = all_band_weights.std(dim=0).numpy()
        stats['active_bands'] = (all_band_weights > 0.5).float().mean(dim=0).numpy()
    
    return stats


def visualize_eq_curve(freqs, gains, qs, sample_rate=44100, num_points=1000):
    """
    Visualize the frequency response of an EQ.
    
    Args:
        freqs: [num_bands] - center frequencies
        gains: [num_bands] - gains in dB
        qs: [num_bands] - Q factors
        sample_rate: Sample rate
        num_points: Number of frequency points to plot
    
    Returns:
        fig: Matplotlib figure
    """
    # Frequency range (20Hz to 20kHz, log scale)
    freq_range = np.logspace(np.log10(20), np.log10(20000), num_points)
    
    # Initialize frequency response
    magnitude_db = np.zeros(num_points)
    
    # Add contribution from each band
    for freq, gain, q in zip(freqs, gains, qs):
        freq = freq.item() if torch.is_tensor(freq) else freq
        gain = gain.item() if torch.is_tensor(gain) else gain
        q = q.item() if torch.is_tensor(q) else q
        
        # Peaking filter magnitude response
        omega = 2 * np.pi * freq_range / sample_rate
        omega0 = 2 * np.pi * freq / sample_rate
        
        # Simplified magnitude calculation
        bandwidth = omega0 / q
        bell = gain / (1 + ((omega - omega0) / (bandwidth / 2)) ** 2)
        
        magnitude_db += bell
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.semilogx(freq_range, magnitude_db, linewidth=2)
    ax.axhline(0, color='k', linestyle='--', alpha=0.3)
    
    # Mark band centers
    for freq, gain in zip(freqs, gains):
        freq_val = freq.item() if torch.is_tensor(freq) else freq
        gain_val = gain.item() if torch.is_tensor(gain) else gain
        ax.plot(freq_val, gain_val, 'ro', markersize=8)
    
    ax.set_xlabel('Frequency (Hz)', fontsize=12)
    ax.set_ylabel('Gain (dB)', fontsize=12)
    ax.set_title('EQ Frequency Response', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([20, 20000])
    
    return fig


def evaluate_model(model, data_loader, device, phase, save_dir=None):
    """
    Comprehensive evaluation of trained model.
    
    Args:
        model: Trained model
        data_loader: Test data loader
        device: torch device
        phase: Model phase
        save_dir: Directory to save plots and results
    
    Returns:
        metrics: Dictionary of evaluation metrics
    """
    model.eval()
    
    # Initialize metrics
    stft_loss_fn = MultiScaleSTFTLoss().to(device)
    lufs_loss_fn = LUFSLoss().to(device)
    a_weighted_loss_fn = AWeightedLoss().to(device)
    
    all_stft_loss = []
    all_mel_dist = []
    all_lufs_error = []
    all_a_weighted_error = []
    all_snr = []
    
    print("Evaluating model...")
    
    with torch.no_grad():
        for unmastered, mastered in tqdm(data_loader, desc='Testing'):
            unmastered = unmastered.to(device)
            mastered = mastered.to(device)
            
            # Forward pass
            output, params = model(unmastered)
            
            # Compute metrics
            stft_loss = stft_loss_fn(output, mastered)
            mel_dist = compute_mel_distance(output, mastered)
            lufs_error = lufs_loss_fn(output, mastered)
            a_weighted_error = a_weighted_loss_fn(output, mastered)
            snr = compute_snr(output, mastered)
            
            all_stft_loss.append(stft_loss.item())
            all_mel_dist.append(mel_dist.mean().item())
            all_lufs_error.append(lufs_error.item())
            all_a_weighted_error.append(a_weighted_error.item())
            all_snr.append(snr.mean().item())
    
    # Compute statistics
    metrics = {
        'stft_loss': {
            'mean': np.mean(all_stft_loss),
            'std': np.std(all_stft_loss)
        },
        'mel_distance': {
            'mean': np.mean(all_mel_dist),
            'std': np.std(all_mel_dist)
        },
        'lufs_error': {
            'mean': np.mean(all_lufs_error),
            'std': np.std(all_lufs_error)
        },
        'a_weighted_error': {
            'mean': np.mean(all_a_weighted_error),
            'std': np.std(all_a_weighted_error)
        },
        'snr_db': {
            'mean': np.mean(all_snr),
            'std': np.std(all_snr)
        }
    }
    
    # Print results
    print("\n" + "="*60)
    print(f"Phase {phase} Evaluation Results")
    print("="*60)
    print(f"STFT Loss: {metrics['stft_loss']['mean']:.4f} ± {metrics['stft_loss']['std']:.4f}")
    print(f"Mel Distance: {metrics['mel_distance']['mean']:.4f} ± {metrics['mel_distance']['std']:.4f}")
    print(f"LUFS Error: {metrics['lufs_error']['mean']:.4f} ± {metrics['lufs_error']['std']:.4f}")
    print(f"A-weighted Error: {metrics['a_weighted_error']['mean']:.4f} ± {metrics['a_weighted_error']['std']:.4f}")
    print(f"SNR: {metrics['snr_db']['mean']:.2f} ± {metrics['snr_db']['std']:.2f} dB")
    print("="*60)
    
    # Analyze EQ parameters
    if phase in ['1A', '1B', '1C']:
        print("\nAnalyzing EQ parameters...")
        eq_stats = analyze_eq_parameters(model, data_loader, device, phase)
        metrics['eq_stats'] = eq_stats
        
        print("\nEQ Parameter Statistics:")
        print(f"Frequency centers (Hz): {eq_stats['freq_mean']}")
        print(f"Average gains (dB): {eq_stats['gain_mean']}")
        print(f"Average Q factors: {eq_stats['q_mean']}")
        
        if 'active_bands' in eq_stats:
            print(f"Active band usage: {eq_stats['active_bands']}")
    
    # Save visualizations
    if save_dir and phase in ['1A', '1B', '1C']:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Plot average EQ curve
        fig = visualize_eq_curve(
            eq_stats['freq_mean'],
            eq_stats['gain_mean'],
            eq_stats['q_mean']
        )
        fig.savefig(save_dir / f'phase_{phase}_eq_curve.png', dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        # Plot metric distributions
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        axes[0, 0].hist(all_stft_loss, bins=30, edgecolor='black')
        axes[0, 0].set_title('STFT Loss Distribution')
        axes[0, 0].set_xlabel('Loss')
        
        axes[0, 1].hist(all_mel_dist, bins=30, edgecolor='black')
        axes[0, 1].set_title('Mel Distance Distribution')
        axes[0, 1].set_xlabel('Distance')
        
        axes[1, 0].hist(all_lufs_error, bins=30, edgecolor='black')
        axes[1, 0].set_title('LUFS Error Distribution')
        axes[1, 0].set_xlabel('Error (dB)')
        
        axes[1, 1].hist(all_snr, bins=30, edgecolor='black')
        axes[1, 1].set_title('SNR Distribution')
        axes[1, 1].set_xlabel('SNR (dB)')
        
        plt.tight_layout()
        fig.savefig(save_dir / f'phase_{phase}_metrics.png', dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        print(f"\nVisualizations saved to {save_dir}")
    
    return metrics


if __name__ == '__main__':
    import argparse
    import yaml
    
    parser = argparse.ArgumentParser(description='Evaluate trained model')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to config file')
    parser.add_argument('--save_dir', type=str, default='evaluation_results',
                        help='Directory to save results')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    phase = config['model'].get('phase', '1B')
    
    # Load model
    print(f"Loading Phase {phase} model...")
    if phase == '1A':
        model = MasteringModel_Phase1A(config).to(device)
    elif phase == '1B':
        model = MasteringModel_Phase1B(config).to(device)
    elif phase == '1C':
        model = MasteringModel_Phase1C(config).to(device)
    
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load test data
    _, _, test_loader = get_dataloaders(config)
    
    # Evaluate
    metrics = evaluate_model(model, test_loader, device, phase, args.save_dir)
    
    # Save metrics
    import json
    metrics_path = Path(args.save_dir) / f'phase_{phase}_metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x)
    
    print(f"\nMetrics saved to {metrics_path}")
