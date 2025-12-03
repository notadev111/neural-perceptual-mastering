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
        """Compute STFT magnitude for stereo audio."""
        # Window
        window = torch.hann_window(win_length).to(x.device)

        # Process each channel separately for stereo
        batch_size, channels, samples = x.shape

        # Compute STFT for each channel
        mags = []
        for ch in range(channels):
            stft_result = torch.stft(
                x[:, ch, :],  # [batch, samples]
                n_fft=fft_size,
                hop_length=hop_size,
                win_length=win_length,
                window=window,
                return_complex=True
            )
            mags.append(torch.abs(stft_result))

        # Average magnitude across channels
        mag = torch.stack(mags, dim=0).mean(dim=0)
        return mag
    
    def forward(self, pred, target):
        """
        Args:
            pred: [batch, channels, samples] - stereo (2 channels)
            target: [batch, channels, samples] - stereo (2 channels)
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

    Uses frequency-domain weighting (more stable than time-domain IIR filter).

    References:
    - ISO 226:2003 standard for equal-loudness contours
    - A-weighting: https://en.wikipedia.org/wiki/A-weighting
    """
    def __init__(self, sample_rate=44100, n_fft=2048):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft

        # Create A-weighting curve in frequency domain
        a_weights = self._create_a_weighting_curve(sample_rate, n_fft)
        self.register_buffer('a_weights', a_weights)

    def _create_a_weighting_curve(self, sr, n_fft):
        """
        Create A-weighting curve in frequency domain.

        A-weighting formula from ISO 226:
        A(f) = 20*log10(Ra(f)) where
        Ra(f) = (12194^2 * f^4) / ((f^2 + 20.6^2) * sqrt((f^2 + 107.7^2)(f^2 + 737.9^2)) * (f^2 + 12194^2))
        """
        # Frequency bins
        freqs = np.fft.rfftfreq(n_fft, 1.0/sr)

        # A-weighting formula (ISO 226 standard)
        f = freqs + 1e-10  # Avoid division by zero at DC

        # Constants
        c1 = 20.6**2
        c2 = 107.7**2
        c3 = 737.9**2
        c4 = 12194.0**2

        # A-weighting transfer function
        numerator = c4 * f**4
        denominator = (f**2 + c1) * np.sqrt((f**2 + c2) * (f**2 + c3)) * (f**2 + c4)

        # Convert to dB
        a_weight_db = 20 * np.log10(numerator / (denominator + 1e-10))

        # Normalize to 0dB at 1kHz
        a_weight_db = a_weight_db - a_weight_db[np.argmin(np.abs(freqs - 1000))]

        # Convert dB to linear scale
        a_weight_linear = 10 ** (a_weight_db / 20.0)

        return torch.tensor(a_weight_linear, dtype=torch.float32)

    def forward(self, pred, target):
        """
        Compute L1 loss on A-weighted signals using frequency-domain weighting.

        Args:
            pred: [batch, channels, samples] - stereo (2 channels)
            target: [batch, channels, samples] - stereo (2 channels)
        """
        batch_size, channels, samples = pred.shape

        # Window for STFT
        window = torch.hann_window(self.n_fft).to(pred.device)

        # Process each channel
        pred_weighted_list = []
        target_weighted_list = []

        for ch in range(channels):
            # Compute STFT
            pred_stft = torch.stft(
                pred[:, ch, :],
                n_fft=self.n_fft,
                hop_length=self.n_fft // 4,
                win_length=self.n_fft,
                window=window,
                return_complex=True
            )
            target_stft = torch.stft(
                target[:, ch, :],
                n_fft=self.n_fft,
                hop_length=self.n_fft // 4,
                win_length=self.n_fft,
                window=window,
                return_complex=True
            )

            # Apply A-weighting in frequency domain
            # a_weights shape: [freq_bins], stft shape: [batch, freq_bins, time_frames]
            a_weights = self.a_weights.to(pred.device).unsqueeze(0).unsqueeze(-1)  # [1, freq_bins, 1]
            pred_stft_weighted = pred_stft * a_weights
            target_stft_weighted = target_stft * a_weights

            # Convert back to time domain
            pred_weighted = torch.istft(
                pred_stft_weighted,
                n_fft=self.n_fft,
                hop_length=self.n_fft // 4,
                win_length=self.n_fft,
                window=window,
                length=samples
            )
            target_weighted = torch.istft(
                target_stft_weighted,
                n_fft=self.n_fft,
                hop_length=self.n_fft // 4,
                win_length=self.n_fft,
                window=window,
                length=samples
            )

            pred_weighted_list.append(pred_weighted)
            target_weighted_list.append(target_weighted)

        # Stack channels back
        pred_weighted_audio = torch.stack(pred_weighted_list, dim=1)
        target_weighted_audio = torch.stack(target_weighted_list, dim=1)

        # L1 loss (MAE) on weighted signals
        return F.l1_loss(pred_weighted_audio, target_weighted_audio)


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
            audio: [batch, channels, samples] - stereo (2 channels)
        Returns:
            rms_db: [batch]
        """
        # RMS over time and channel dimensions (average across stereo channels)
        rms = torch.sqrt(torch.mean(audio ** 2, dim=(1, 2)))

        # Convert to dB
        rms_db = 20 * torch.log10(rms + 1e-7)

        return rms_db
    
    def forward(self, pred, target):
        """
        Match loudness between prediction and target.

        Args:
            pred: [batch, channels, samples] - stereo (2 channels)
            target: [batch, channels, samples] - stereo (2 channels)
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
            pred: [batch, channels, samples] - stereo (2 channels)
            target: [batch, channels, samples] - stereo (2 channels)
        """
        # Compute STFT
        window = torch.hann_window(self.n_fft).to(pred.device)

        # Process each channel and average
        pred_mels = []
        target_mels = []

        for ch in range(pred.shape[1]):
            pred_stft = torch.stft(
                pred[:, ch, :],
                n_fft=self.n_fft,
                return_complex=True,
                window=window
            )
            target_stft = torch.stft(
                target[:, ch, :],
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

            pred_mels.append(pred_mel)
            target_mels.append(target_mel)

        # Average across channels
        pred_mel_avg = torch.stack(pred_mels, dim=0).mean(dim=0)
        target_mel_avg = torch.stack(target_mels, dim=0).mean(dim=0)

        # Log mel spectrogram
        pred_log_mel = torch.log(pred_mel_avg + 1e-7)
        target_log_mel = torch.log(target_mel_avg + 1e-7)

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
            pred: [batch, channels, samples] - stereo (2 channels)
            target: [batch, channels, samples] - stereo (2 channels)
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
