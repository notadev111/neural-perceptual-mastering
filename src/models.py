"""
Neural Perceptual Audio Mastering Models

Architecture:
- AudioEncoder: TCN-based latent representation (Comunità 2025)
- ParametricDecoder: Differentiable biquad EQ (Nercessian 2020)
- ResidualDecoder: Wave-U-Net black-box correction
- AdaptiveParametricDecoder: Novel adaptive band selection

Phases:
- Phase 1A: Parametric EQ only (baseline)
- Phase 1B: EQ + black-box residual (full hybrid)
- Phase 1C: Adaptive band selection + residual (novel)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import math


class AudioEncoder(nn.Module):
    """
    TCN (Temporal Convolutional Network) encoder.

    Extracts latent representation from raw audio.
    Uses strided convolutions + dilated TCN blocks for long-range dependencies.

    Based on Comunità et al. 2025 - TCNs for audio processing.
    """
    def __init__(self, latent_dim=512, input_channels=2):  # Changed to 2 for stereo
        super().__init__()

        # Initial strided convolution (downsampling)
        self.stem = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=15, stride=4, padding=7),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=15, stride=4, padding=7),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        
        # TCN blocks with increasing dilation
        self.tcn_blocks = nn.ModuleList([
            TCNBlock(128, 128, kernel_size=3, dilation=1),
            TCNBlock(128, 256, kernel_size=3, dilation=2),
            TCNBlock(256, 256, kernel_size=3, dilation=4),
            TCNBlock(256, 512, kernel_size=3, dilation=8),
        ])
        
        # Global average pooling + projection to latent space
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(512, latent_dim),
        )
        
    def forward(self, audio):
        """
        Args:
            audio: [batch, 2, samples] - stereo
        Returns:
            z: [batch, latent_dim]
        """
        x = self.stem(audio)
        
        for block in self.tcn_blocks:
            x = block(x)
        
        z = self.head(x)
        return z


class TCNBlock(nn.Module):
    """Single TCN block with dilated convolution and residual connection."""
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1):
        super().__init__()
        
        self.conv1 = nn.Conv1d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=dilation * (kernel_size - 1) // 2
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        
        self.conv2 = nn.Conv1d(
            out_channels, out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=dilation * (kernel_size - 1) // 2
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        # Residual connection
        self.residual = nn.Conv1d(in_channels, out_channels, kernel_size=1) \
            if in_channels != out_channels else nn.Identity()
        
    def forward(self, x):
        residual = self.residual(x)
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        return F.relu(out + residual)


class SimpleBiquadEQ(nn.Module):
    """
    Differentiable biquad EQ using torchaudio.
    
    Implements parametric EQ with learnable center frequencies, gains, and Q factors.
    Based on Nercessian et al. 2020.
    """
    def __init__(self, sample_rate=44100):
        super().__init__()
        self.sample_rate = sample_rate
        
    def forward(self, audio, center_freqs, gains, q_factors):
        """
        Apply biquad EQ filters to stereo audio.

        Args:
            audio: [batch, 2, samples] - stereo
            center_freqs: [batch, num_bands] - frequencies in Hz
            gains: [batch, num_bands] - gains in dB
            q_factors: [batch, num_bands] - Q factors

        Returns:
            filtered: [batch, 2, samples] - stereo
        """
        batch_size, channels, samples = audio.shape
        num_bands = center_freqs.shape[1]

        output = audio.clone()

        # Apply each band sequentially (same EQ to both channels)
        for band_idx in range(num_bands):
            for batch_idx in range(batch_size):
                freq = center_freqs[batch_idx, band_idx]
                gain = gains[batch_idx, band_idx]
                q = q_factors[batch_idx, band_idx]

                # Apply biquad peaking filter using torchaudio.functional
                # Compute biquad coefficients for peaking filter
                import torch
                import math

                # Convert parameters
                w0 = 2 * math.pi * freq.item() / self.sample_rate
                alpha = math.sin(w0) / (2 * q.item())
                A = 10 ** (gain.item() / 40)  # sqrt of gain in linear scale

                # Peaking EQ coefficients
                b0 = 1 + alpha * A
                b1 = -2 * math.cos(w0)
                b2 = 1 - alpha * A
                a0 = 1 + alpha / A
                a1 = -2 * math.cos(w0)
                a2 = 1 - alpha / A

                # Normalize
                b0, b1, b2, a1, a2 = b0/a0, b1/a0, b2/a0, a1/a0, a2/a0

                # Apply biquad filter to both channels
                output[batch_idx] = torchaudio.functional.biquad(
                    output[batch_idx],
                    b0, b1, b2, 1.0, a1, a2
                )

        return output


class ParametricDecoder(nn.Module):
    """
    Parametric decoder with differentiable biquad EQ.
    
    Predicts EQ parameters (frequency, gain, Q) from latent code.
    Uses interpretable DSP processing (white-box).
    """
    def __init__(self, latent_dim=512, num_bands=5, sample_rate=44100,
                 hidden_dim=256, dropout=0.2):
        super().__init__()
        
        self.num_bands = num_bands
        self.sample_rate = sample_rate
        
        # MLP to predict EQ parameters
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        # Separate heads for each parameter type
        self.freq_head = nn.Linear(hidden_dim, num_bands)
        self.gain_head = nn.Linear(hidden_dim, num_bands)
        self.q_head = nn.Linear(hidden_dim, num_bands)
        
        # Differentiable EQ
        self.eq = SimpleBiquadEQ(sample_rate=sample_rate)
        
    def forward(self, z, audio):
        """
        Args:
            z: [batch, latent_dim]
            audio: [batch, 2, samples] - input stereo audio

        Returns:
            eq_audio: [batch, 2, samples] - EQ'd stereo audio
            eq_params: tuple of (freqs, gains, q_factors)
        """
        # Predict parameters
        features = self.mlp(z)
        
        # Frequencies: 20Hz - 20kHz (log scale)
        freqs = torch.sigmoid(self.freq_head(features)) * (20000 - 20) + 20
        
        # Gains: -12dB to +12dB
        gains = torch.tanh(self.gain_head(features)) * 12
        
        # Q factors: 0.5 to 5.0 (wider to narrower)
        q_factors = torch.sigmoid(self.q_head(features)) * 4.5 + 0.5
        
        # Apply EQ
        eq_audio = self.eq(audio, freqs, gains, q_factors)
        
        return eq_audio, (freqs, gains, q_factors)


class AdaptiveParametricDecoder(nn.Module):
    """
    Adaptive band selection decoder (NOVEL CONTRIBUTION).
    
    Learns which EQ bands to activate (5-10 bands).
    Uses soft gating to let model decide band usage.
    """
    def __init__(self, latent_dim=512, max_bands=10, sample_rate=44100,
                 hidden_dim=256, dropout=0.2):
        super().__init__()
        
        self.max_bands = max_bands
        self.sample_rate = sample_rate
        
        # MLP backbone
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        # Band selector (soft gating)
        self.band_selector = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, max_bands),
            nn.Sigmoid()  # [0, 1] per band
        )
        
        # Parameter heads (for all possible bands)
        self.freq_head = nn.Linear(hidden_dim, max_bands)
        self.gain_head = nn.Linear(hidden_dim, max_bands)
        self.q_head = nn.Linear(hidden_dim, max_bands)
        
        # Differentiable EQ
        self.eq = SimpleBiquadEQ(sample_rate=sample_rate)
        
    def forward(self, z, audio):
        """
        Args:
            z: [batch, latent_dim]
            audio: [batch, 2, samples] - stereo

        Returns:
            eq_audio: [batch, 2, samples] - stereo
            params: tuple of (freqs, gains_gated, q_factors, band_weights)
        """
        # Predict band activation weights
        band_weights = self.band_selector(z)  # [batch, max_bands]
        
        # Predict EQ parameters
        features = self.mlp(z)
        
        freqs = torch.sigmoid(self.freq_head(features)) * (20000 - 20) + 20
        gains = torch.tanh(self.gain_head(features)) * 12
        q_factors = torch.sigmoid(self.q_head(features)) * 4.5 + 0.5
        
        # Apply soft gating (inactive bands → 0dB gain)
        gains_gated = gains * band_weights
        
        # Apply EQ with gated gains
        eq_audio = self.eq(audio, freqs, gains_gated, q_factors)
        
        return eq_audio, (freqs, gains_gated, q_factors, band_weights)


class ResidualDecoder(nn.Module):
    """
    Black-box residual decoder using Wave-U-Net.

    Learns non-linear corrections that can't be captured by EQ.
    Uses FiLM conditioning on latent code.
    """
    def __init__(self, latent_dim=512, base_channels=32):
        super().__init__()

        # Downsampling path (starts with 2 channels for stereo)
        self.down1 = WaveUNetBlock(2, base_channels, latent_dim)  # Changed to 2
        self.down2 = WaveUNetBlock(base_channels, base_channels * 2, latent_dim)
        self.down3 = WaveUNetBlock(base_channels * 2, base_channels * 4, latent_dim)

        # Bottleneck
        self.bottleneck = WaveUNetBlock(base_channels * 4, base_channels * 8, latent_dim)

        # Upsampling path with skip connections
        self.up3 = WaveUNetBlock(base_channels * 8 + base_channels * 4,
                                  base_channels * 4, latent_dim)
        self.up2 = WaveUNetBlock(base_channels * 4 + base_channels * 2,
                                  base_channels * 2, latent_dim)
        self.up1 = WaveUNetBlock(base_channels * 2 + base_channels,
                                  base_channels, latent_dim)

        # Output projection (outputs 2 channels for stereo)
        self.output = nn.Conv1d(base_channels, 2, kernel_size=1)  # Changed to 2
        
    def forward(self, z, audio):
        """
        Args:
            z: [batch, latent_dim]
            audio: [batch, 2, samples] - input stereo audio (for skip connection)

        Returns:
            residual: [batch, 2, samples] - residual correction (stereo)
        """
        # Downsampling
        d1 = self.down1(audio, z)
        d1_pool = F.avg_pool1d(d1, 2)
        
        d2 = self.down2(d1_pool, z)
        d2_pool = F.avg_pool1d(d2, 2)
        
        d3 = self.down3(d2_pool, z)
        d3_pool = F.avg_pool1d(d3, 2)
        
        # Bottleneck
        bottleneck = self.bottleneck(d3_pool, z)
        
        # Upsampling with skip connections
        u3 = F.interpolate(bottleneck, size=d3.shape[-1], mode='linear', align_corners=False)
        u3 = torch.cat([u3, d3], dim=1)
        u3 = self.up3(u3, z)
        
        u2 = F.interpolate(u3, size=d2.shape[-1], mode='linear', align_corners=False)
        u2 = torch.cat([u2, d2], dim=1)
        u2 = self.up2(u2, z)
        
        u1 = F.interpolate(u2, size=d1.shape[-1], mode='linear', align_corners=False)
        u1 = torch.cat([u1, d1], dim=1)
        u1 = self.up1(u1, z)
        
        # Output residual
        residual = self.output(u1)
        
        # Ensure same length as input
        if residual.shape[-1] != audio.shape[-1]:
            residual = F.interpolate(residual, size=audio.shape[-1], 
                                      mode='linear', align_corners=False)
        
        return residual


class WaveUNetBlock(nn.Module):
    """Wave-U-Net block with FiLM conditioning."""
    def __init__(self, in_channels, out_channels, latent_dim):
        super().__init__()
        
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=15, padding=7),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
        )
        
        # FiLM layers (Feature-wise Linear Modulation)
        self.film_gamma = nn.Linear(latent_dim, out_channels)
        self.film_beta = nn.Linear(latent_dim, out_channels)
        
    def forward(self, x, z):
        """
        Args:
            x: [batch, in_channels, length]
            z: [batch, latent_dim]
        """
        x = self.conv(x)
        
        # FiLM conditioning
        gamma = self.film_gamma(z).unsqueeze(-1)  # [batch, out_channels, 1]
        beta = self.film_beta(z).unsqueeze(-1)
        
        x = gamma * x + beta
        
        return x


# ============================================================================
# PHASE MODELS
# ============================================================================

class MasteringModel_Phase1A(nn.Module):
    """
    Phase 1A: Baseline parametric EQ only.

    Pure interpretable white-box model.
    Tests if differentiable EQ alone is sufficient.
    """
    def __init__(self, config):
        super().__init__()

        latent_dim = config['model']['encoder']['latent_dim']
        num_bands = config['model']['parametric_decoder']['num_bands']
        sample_rate = config['data']['sample_rate']
        hidden_dim = config['model']['parametric_decoder']['hidden_dim']
        dropout = config['model']['parametric_decoder']['dropout']

        self.encoder = AudioEncoder(latent_dim=latent_dim)
        self.parametric_decoder = ParametricDecoder(
            latent_dim=latent_dim,
            num_bands=num_bands,
            sample_rate=sample_rate,
            hidden_dim=hidden_dim,
            dropout=dropout
        )
    
    def forward(self, audio):
        """
        Args:
            audio: [batch, 2, samples] - stereo

        Returns:
            output: [batch, 2, samples] - stereo
            params: dict of extracted parameters
        """
        z = self.encoder(audio)
        output, eq_params = self.parametric_decoder(z, audio)

        params = {
            'latent': z,
            'eq_frequencies': eq_params[0],
            'eq_gains': eq_params[1],
            'eq_q_factors': eq_params[2]
        }

        return output, params


class MasteringModel_Phase1B(nn.Module):
    """
    Phase 1B: Parametric EQ + Black-box residual.
    
    Full hybrid architecture (white-box + black-box).
    Tests if residual improves over EQ-only.
    """
    def __init__(self, config):
        super().__init__()
        
        latent_dim = config['model']['encoder']['latent_dim']
        num_bands = config['model']['parametric_decoder']['num_bands']
        sample_rate = config['data']['sample_rate']
        
        self.encoder = AudioEncoder(latent_dim=latent_dim)
        self.parametric_decoder = ParametricDecoder(
            latent_dim=latent_dim,
            num_bands=num_bands,
            sample_rate=sample_rate
        )
        self.residual_decoder = ResidualDecoder(latent_dim=latent_dim)
    
    def forward(self, audio):
        """
        Args:
            audio: [batch, 2, samples] - stereo

        Returns:
            output: [batch, 2, samples] - stereo
            params: dict of extracted parameters
        """
        z = self.encoder(audio)

        # Parametric path
        eq_out, eq_params = self.parametric_decoder(z, audio)

        # Residual path
        residual_out = self.residual_decoder(z, audio)

        # Combine
        output = eq_out + residual_out

        params = {
            'latent': z,
            'eq_frequencies': eq_params[0],
            'eq_gains': eq_params[1],
            'eq_q_factors': eq_params[2]
        }

        return output, params


class MasteringModel_Phase1C(nn.Module):
    """
    Phase 1C: Adaptive band selection + Residual.
    
    NOVEL CONTRIBUTION: Model learns which EQ bands to use.
    Tests if adaptive band selection improves over fixed bands.
    """
    def __init__(self, config):
        super().__init__()
        
        latent_dim = config['model']['encoder']['latent_dim']
        max_bands = config['model']['parametric_decoder'].get('max_bands', 10)
        sample_rate = config['data']['sample_rate']
        
        self.encoder = AudioEncoder(latent_dim=latent_dim)
        self.parametric_decoder = AdaptiveParametricDecoder(
            latent_dim=latent_dim,
            max_bands=max_bands,
            sample_rate=sample_rate
        )
        self.residual_decoder = ResidualDecoder(latent_dim=latent_dim)
    
    def forward(self, audio):
        """
        Args:
            audio: [batch, 2, samples] - stereo

        Returns:
            output: [batch, 2, samples] - stereo
            params: dict of extracted parameters
        """
        z = self.encoder(audio)

        # Adaptive parametric path
        eq_out, eq_params = self.parametric_decoder(z, audio)

        # Residual path
        residual_out = self.residual_decoder(z, audio)

        # Combine
        output = eq_out + residual_out

        params = {
            'latent': z,
            'eq_frequencies': eq_params[0],
            'eq_gains': eq_params[1],  # Gated gains
            'eq_q_factors': eq_params[2],
            'band_weights': eq_params[3]  # Soft selection weights
        }

        return output, params
