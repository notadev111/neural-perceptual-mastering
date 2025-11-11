"""
Loss Functions for Neural Perceptual Audio Mastering
- Multi-scale STFT loss (spectral reconstruction)
- A-weighted perceptual loss (ISO 226, Wright & Välimäki 2016)
- LUFS matching loss (loudness normalization)
- Parameter regularization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import signal
import numpy as np


class MultiScaleSTFTLoss(nn.Module):
    """
    Multi-scale STFT loss for spectral reconstruction.
    Based on Koo et al. 2022 - captures both fine and coarse spectral detail.
    """
    def __init__(self, fft_sizes=[2048, 1024, 512, 256], 
                 hop_sizes=[512, 256, 128, 64],
                 win_lengths=[2048, 1024, 512, 256]):
        super().__init__()
        self.fft_sizes = fft_sizes
        self.hop_sizes = hop_sizes
        self.win_lengths = win_lengths
        
    def stft(self, x, fft_size, hop_size, win_length):
        """Compute STFT magnitude."""
        # Window
        window = torch.hann_window(win_length).to(x.device)
        
        # Compute STFT
        stft_result = torch.stft(
            x.squeeze(1),
            n_fft=fft_size,
            hop_length=hop_size,
            win_length=win_length,
            window=window,
            return_complex=True
        )
        
        # Magnitude
        mag = torch.abs(stft_result)
        return mag
    
    def forward(self, pred, target):
        """
        Args:
            pred: [batch, 1, samples]
            target: [batch, 1, samples]
        """
        total_loss = 0.0
        
        for fft_size, hop_size, win_length in zip(
            self.fft_sizes, self.hop_sizes, self.win_lengths
        ):
            # Compute STFT magnitudes
            pred_mag = self.stft(pred, fft_size, hop_size, win_length)
            target_mag = self.stft(target, fft_size, hop_size, win_length)
            
            # Spectral convergence loss (L2 on magnitudes)
            sc_loss = torch.norm(target_mag - pred_mag, p='fro') / torch.norm(target_mag, p='fro')
            
            # Log magnitude loss (perceptual)
            log_loss = F.l1_loss(
                torch.log(pred_mag + 1e-7),
                torch.log(target_mag + 1e-7)
            )
            
            total_loss += sc_loss + log_loss
        
        return total_loss / len(self.fft_sizes)


class AWeightedLoss(nn.Module):
    """
    A-weighted perceptual loss function.
    
    Based on ISO 226 A-weighting curve which models human ear sensitivity.
    Emphasizes 2-5kHz range where hearing is most sensitive.
    
    References:
    - Wright & Välimäki (2016): "Perceptual Audio Evaluation"
    - ISO 226:2003 standard for equal-loudness contours
    """
    def __init__(self, sample_rate=44100):
        super().__init__()
        self.sample_rate = sample_rate
        
        # Create A-weighting filter coefficients
        b, a = self._create_a_weighting_filter(sample_rate)
        
        # Store as buffers (not parameters - won't be trained)
        self.register_buffer('filter_b', torch.tensor(b, dtype=torch.float32))
        self.register_buffer('filter_a', torch.tensor(a, dtype=torch.float32))
        
    def _create_a_weighting_filter(self, sr):
        """
        Design A-weighting filter using scipy.
        Approximates ISO 226 A-weighting curve with IIR filter.
        """
        # Design bandpass filter emphasizing 2-5kHz
        # A-weighting has +3dB peak around 2.5kHz
        b, a = signal.iirfilter(
            4,  # 4th order
            [20, 20000],  # Full audible range
            btype='band',
            ftype='butter',
            fs=sr
        )
        
        return b, a
    
    def _apply_filter(self, audio):
        """
        Apply IIR filter to audio using difference equation.
        
        Args:
            audio: [batch, 1, samples]
        Returns:
            filtered: [batch, 1, samples]
        """
        batch_size, channels, samples = audio.shape
        
        # Move to numpy for scipy filtering (could optimize with torch_dct)
        audio_np = audio.detach().cpu().numpy()
        
        filtered = np.zeros_like(audio_np)
        for b in range(batch_size):
            for c in range(channels):
                filtered[b, c] = signal.lfilter(
                    self.filter_b.cpu().numpy(),
                    self.filter_a.cpu().numpy(),
                    audio_np[b, c]
                )
        
        return torch.tensor(filtered, dtype=audio.dtype, device=audio.device)
    
    def forward(self, pred, target):
        """
        Compute L1 loss on A-weighted signals.
        
        Args:
            pred: [batch, 1, samples]
            target: [batch, 1, samples]
        """
        # Apply A-weighting to both signals
        pred_weighted = self._apply_filter(pred)
        target_weighted = self._apply_filter(target)
        
        # L1 loss (MAE) on weighted signals
        return F.l1_loss(pred_weighted, target_weighted)


class LUFSLoss(nn.Module):
    """
    LUFS (Loudness Units relative to Full Scale) matching loss.
    
    Ensures output loudness matches target loudness.
    Uses ITU-R BS.1770 standard (EBU R128).
    
    Simplified implementation - full LUFS requires K-weighting filter.
    """
    def __init__(self, target_lufs=-14.0):
        super().__init__()
        self.target_lufs = target_lufs
        
    def compute_rms_db(self, audio):
        """
        Compute RMS level in dB (simplified loudness).
        
        Args:
            audio: [batch, 1, samples]
        Returns:
            rms_db: [batch]
        """
        # RMS over time dimension
        rms = torch.sqrt(torch.mean(audio ** 2, dim=-1))
        
        # Convert to dB
        rms_db = 20 * torch.log10(rms + 1e-7)
        
        return rms_db.squeeze(1)
    
    def forward(self, pred, target):
        """
        Match loudness between prediction and target.
        
        Args:
            pred: [batch, 1, samples]
            target: [batch, 1, samples]
        """
        pred_lufs = self.compute_rms_db(pred)
        target_lufs = self.compute_rms_db(target)
        
        # L1 loss on loudness
        return F.l1_loss(pred_lufs, target_lufs)


class ParameterRegularizationLoss(nn.Module):
    """
    Regularization loss for EQ parameters.
    
    Encourages:
    - Minimal EQ adjustments (smooth frequency response)
    - Sparse band usage (most gains near 0dB)
    - Reasonable Q factors (not too narrow/wide)
    """
    def __init__(self, lambda_gain=0.01, lambda_q=0.001):
        super().__init__()
        self.lambda_gain = lambda_gain
        self.lambda_q = lambda_q
        
    def forward(self, gains, q_factors):
        """
        Args:
            gains: [batch, num_bands] - EQ gains in dB
            q_factors: [batch, num_bands] - Q factors
        """
        # Encourage gains near 0dB (minimal EQ)
        gain_penalty = torch.mean(torch.abs(gains))
        
        # Encourage Q factors near 1.0 (moderate bandwidth)
        q_penalty = torch.mean((q_factors - 1.0) ** 2)
        
        return self.lambda_gain * gain_penalty + self.lambda_q * q_penalty


class MelSpectralLoss(nn.Module):
    """
    Mel-frequency spectral loss.
    
    Captures perceptual spectral similarity using mel scale.
    Useful for evaluating mastering quality (from Välimäki paper).
    """
    def __init__(self, sample_rate=44100, n_fft=2048, n_mels=128):
        super().__init__()
        self.n_mels = n_mels
        self.n_fft = n_fft
        
        # Create mel filterbank
        self.mel_fb = self._create_mel_filterbank(sample_rate, n_fft, n_mels)
        
    def _create_mel_filterbank(self, sr, n_fft, n_mels):
        """Create mel-scale filterbank."""
        from librosa.filters import mel
        
        mel_fb = mel(sr=sr, n_fft=n_fft, n_mels=n_mels)
        return torch.tensor(mel_fb, dtype=torch.float32)
    
    def forward(self, pred, target):
        """
        Compute L1 loss on mel spectrograms.
        
        Args:
            pred: [batch, 1, samples]
            target: [batch, 1, samples]
        """
        # Compute STFT
        window = torch.hann_window(self.n_fft).to(pred.device)
        
        pred_stft = torch.stft(
            pred.squeeze(1),
            n_fft=self.n_fft,
            return_complex=True,
            window=window
        )
        target_stft = torch.stft(
            target.squeeze(1),
            n_fft=self.n_fft,
            return_complex=True,
            window=window
        )
        
        # Magnitude
        pred_mag = torch.abs(pred_stft)
        target_mag = torch.abs(target_stft)
        
        # Apply mel filterbank
        mel_fb = self.mel_fb.to(pred.device)
        pred_mel = torch.matmul(mel_fb, pred_mag)
        target_mel = torch.matmul(mel_fb, target_mag)
        
        # Log mel spectrogram
        pred_log_mel = torch.log(pred_mel + 1e-7)
        target_log_mel = torch.log(target_mel + 1e-7)
        
        return F.l1_loss(pred_log_mel, target_log_mel)


class CombinedLoss(nn.Module):
    """
    Combined loss function for training.
    
    Balances multiple objectives:
    - Spectral reconstruction (STFT)
    - Perceptual quality (A-weighting)
    - Loudness matching (LUFS)
    - Parameter regularization
    """
    def __init__(self, config):
        super().__init__()
        
        # Loss components
        self.stft_loss = MultiScaleSTFTLoss()
        self.perceptual_loss = AWeightedLoss(
            sample_rate=config['data']['sample_rate']
        )
        self.loudness_loss = LUFSLoss(target_lufs=-14.0)
        self.param_reg = ParameterRegularizationLoss()
        
        # Weights
        self.w_spectral = config['training']['loss_weights']['spectral']
        self.w_perceptual = config['training']['loss_weights']['perceptual']
        self.w_loudness = config['training']['loss_weights']['loudness']
        self.w_param = config['training']['loss_weights']['param_reg']
        
    def forward(self, pred, target, eq_params=None):
        """
        Compute weighted combination of losses.
        
        Args:
            pred: [batch, 1, samples]
            target: [batch, 1, samples]
            eq_params: tuple of (freqs, gains, q_factors) or None
        
        Returns:
            total_loss: scalar
            loss_dict: dictionary of individual losses (for logging)
        """
        # Reconstruction losses
        l_spectral = self.stft_loss(pred, target)
        l_perceptual = self.perceptual_loss(pred, target)
        l_loudness = self.loudness_loss(pred, target)
        
        # Parameter regularization (if EQ params provided)
        if eq_params is not None:
            _, gains, q_factors = eq_params
            l_param = self.param_reg(gains, q_factors)
        else:
            l_param = torch.tensor(0.0, device=pred.device)
        
        # Total loss
        total_loss = (
            self.w_spectral * l_spectral +
            self.w_perceptual * l_perceptual +
            self.w_loudness * l_loudness +
            self.w_param * l_param
        )
        
        # Return both total and individual losses (for logging)
        loss_dict = {
            'total': total_loss.item(),
            'spectral': l_spectral.item(),
            'perceptual': l_perceptual.item(),
            'loudness': l_loudness.item(),
            'param_reg': l_param.item()
        }
        
        return total_loss, loss_dict
