
import torch
import torch.nn as nn
import dasp_pytorch.functional as daspf

class SimpleParametricEQ(nn.Module):
    """
    Wrapper around dasp-pytorch for easier use.
    """
    def __init__(self, sample_rate=44100):
        super().__init__()
        self.sample_rate = sample_rate
    
    def forward(self, audio, frequencies, gains, q_factors):
        """
        Apply parametric EQ with multiple bands.
        
        Args:
            audio: [batch, channels, samples]
            frequencies: [batch, num_bands] in Hz
            gains: [batch, num_bands] in dB
            q_factors: [batch, num_bands]
        
        Returns:
            output: [batch, channels, samples]
        """
        batch_size, num_bands = frequencies.shape
        
        output = audio.clone()
        
        # Apply each band sequentially
        for b in range(batch_size):
            for band_idx in range(num_bands):
                # Use dasp functional API for single peaking filter
                output[b] = daspf.peaking_filter(
                    output[b].unsqueeze(0),
                    sample_rate=self.sample_rate,
                    cutoff_freq=frequencies[b, band_idx].item(),
                    gain_db=gains[b, band_idx].item(),
                    q_factor=q_factors[b, band_idx].item()
                ).squeeze(0)
        
        return output