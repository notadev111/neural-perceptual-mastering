# Phase 2: Complete Mastering Chain

**Grey-box Implementation of Professional Mastering Pipeline**

---

## 🎯 Current Status (Phase 1)

**What We Have:**
```
Input Audio → Encoder → Parametric EQ (white-box) → Output
                     └→ Residual (black-box) ──────┘
```

**Focus:** Spectral shaping (EQ) only

**What We're Missing:** The rest of the mastering chain!

---

## 🎚️ Professional Mastering Chain

A typical professional mastering session follows this signal flow:

```
1. EQ (Spectral Shaping)          ← Phase 1 (current)
   ↓
2. Compression (Dynamics)         ← Phase 2
   ↓
3. Saturation (Harmonics)         ← Phase 2
   ↓
4. Stereo Imaging (Width)         ← Phase 2
   ↓
5. Limiting (Final Loudness)      ← Phase 2
   ↓
Mastered Audio
```

### Why This Order?

**1. EQ First (Phase 1)**
- Fix spectral problems before dynamics
- Clean up muddy/harsh frequencies
- Shape tonal balance

**2. Compression Second**
- Control dynamics after EQ
- Add punch/sustain
- Glue the mix together

**3. Saturation Third**
- Add harmonic richness
- Analog warmth
- Perceived loudness boost

**4. Stereo Imaging Fourth**
- Widen stereo field
- Enhance spatial depth
- Mono compatibility check

**5. Limiting Last**
- Final loudness maximization
- Peak control
- Streaming/broadcast compliance

---

## 🔬 Phase 2 Architecture: Grey-Box Mastering Chain

### Overview

**Hybrid approach:** Differentiable DSP (interpretable) + Neural (flexible)

```
┌─────────────────────────────────────────────────────────────┐
│                        INPUT AUDIO                           │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
                    ┌───────────────┐
                    │  TCN ENCODER  │
                    │  (unchanged)  │
                    └───────────────┘
                            │
                            ▼
                   Latent z ∈ ℝ^512
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
        ▼                   ▼                   ▼
┌─────────────┐  ┌──────────────────┐  ┌─────────────┐
│ Parametric  │  │   Grey-Box       │  │   Neural    │
│ EQ Path     │  │   Chain Path     │  │ Catch-All   │
│ (Phase 1)   │  │   (Phase 2)      │  │   Path      │
└─────────────┘  └──────────────────┘  └─────────────┘
        │                   │                   │
        └───────────────────┼───────────────────┘
                            │
                            ▼
                   ┌────────────────┐
                   │  Final Mix     │
                   │  (sum all)     │
                   └────────────────┘
                            │
                            ▼
                    OUTPUT AUDIO
```

---

## 🔧 Phase 2 Components

### Component 1: Differentiable Compressor (Grey-Box)

**Purpose:** Dynamic range control, punch, glue

#### Parameters to Learn:
```python
- threshold: -40 to 0 dB        # Where compression starts
- ratio: 1.0 to 20.0           # Amount of compression (1:1 to 20:1)
- attack: 0.1 to 30 ms         # How fast it clamps down
- release: 10 to 500 ms        # How fast it lets go
- knee: 0 to 12 dB             # Hard (0) or soft (12) transition
- makeup_gain: 0 to 24 dB      # Compensate for volume loss
```

#### Implementation Approach:

**Option A: Simplified Differentiable Compressor**
```python
class DifferentiableCompressor(nn.Module):
    """
    Simplified compressor with differentiable gain computer.
    Based on envelope detection + gain reduction.
    """
    def __init__(self, sample_rate=44100):
        super().__init__()
        self.sr = sample_rate
    
    def forward(self, audio, threshold, ratio, attack, release, makeup_gain):
        """
        Args:
            audio: [B, 1, T]
            threshold: [B] in dB
            ratio: [B] compression ratio
            attack: [B] in milliseconds
            release: [B] in milliseconds
            makeup_gain: [B] in dB
        """
        # 1. Compute envelope (RMS with ballistics)
        envelope_db = self._compute_envelope(audio, attack, release)
        
        # 2. Compute gain reduction
        over_threshold = envelope_db - threshold.unsqueeze(-1)
        over_threshold = torch.clamp(over_threshold, min=0)
        
        # Compression curve
        gain_reduction_db = over_threshold * (1 - 1/ratio.unsqueeze(-1))
        
        # Convert to linear
        gain_reduction = 10 ** (-gain_reduction_db / 20)
        
        # 3. Apply gain reduction
        compressed = audio * gain_reduction.unsqueeze(1)
        
        # 4. Makeup gain
        makeup = 10 ** (makeup_gain.unsqueeze(-1).unsqueeze(1) / 20)
        output = compressed * makeup
        
        return output
    
    def _compute_envelope(self, audio, attack_ms, release_ms):
        """
        Envelope follower with attack/release ballistics.
        Differentiable approximation using exponential smoothing.
        """
        # Convert ms to samples
        attack_samples = attack_ms * self.sr / 1000
        release_samples = release_ms * self.sr / 1000
        
        # Attack/release coefficients
        attack_coeff = torch.exp(-1.0 / attack_samples)
        release_coeff = torch.exp(-1.0 / release_samples)
        
        # RMS envelope
        audio_squared = audio ** 2
        
        # Smooth with attack/release (simplified)
        # In practice, need proper peak detection + smoothing
        envelope = torch.sqrt(
            F.avg_pool1d(audio_squared, kernel_size=int(self.sr * 0.01))
        )
        
        # Convert to dB
        envelope_db = 20 * torch.log10(envelope + 1e-7)
        
        return envelope_db
```

**Option B: Use dasp-pytorch (if available)**
```python
from dasp_pytorch import Compressor

class CompressorLayer(nn.Module):
    def __init__(self, sample_rate=44100):
        super().__init__()
        self.compressor = Compressor(sample_rate=sample_rate)
    
    def forward(self, audio, params):
        return self.compressor(
            audio,
            threshold=params['threshold'],
            ratio=params['ratio'],
            attack=params['attack'],
            release=params['release']
        )
```

**Parameter Predictor:**
```python
class CompressorPredictor(nn.Module):
    def __init__(self, latent_dim=512):
        super().__init__()
        
        self.predictor = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 6)  # 6 parameters
        )
    
    def forward(self, z):
        params = self.predictor(z)
        
        return {
            'threshold': torch.sigmoid(params[:, 0]) * (-40) + 0,  # -40 to 0 dB
            'ratio': torch.sigmoid(params[:, 1]) * 19 + 1,          # 1 to 20
            'attack': torch.sigmoid(params[:, 2]) * 29.9 + 0.1,    # 0.1 to 30 ms
            'release': torch.sigmoid(params[:, 3]) * 490 + 10,     # 10 to 500 ms
            'knee': torch.sigmoid(params[:, 4]) * 12,               # 0 to 12 dB
            'makeup_gain': torch.sigmoid(params[:, 5]) * 24         # 0 to 24 dB
        }
```

---

### Component 2: Differentiable Saturation (Grey-Box)

**Purpose:** Harmonic richness, warmth, analog character

#### Parameters to Learn:
```python
- drive: 0 to 24 dB              # Input gain (how hard to push)
- saturation_type: soft/hard     # Waveshaping function
- mix: 0 to 100%                 # Wet/dry blend
- bias: -1 to 1                  # Asymmetry (even harmonics)
```

#### Implementation:

```python
class DifferentiableSaturation(nn.Module):
    """
    Waveshaping saturation with differentiable clipping curves.
    Adds harmonics similar to analog tape/tube saturation.
    """
    def __init__(self):
        super().__init__()
    
    def forward(self, audio, drive_db, saturation_type, mix, bias):
        """
        Args:
            audio: [B, 1, T]
            drive_db: [B] input gain in dB
            saturation_type: [B] 0=soft, 1=hard (learned)
            mix: [B] wet/dry ratio 0-1
            bias: [B] DC bias for asymmetry
        """
        # 1. Apply drive
        drive = 10 ** (drive_db.unsqueeze(-1).unsqueeze(1) / 20)
        driven = audio * drive
        
        # 2. Apply bias (for even harmonics)
        driven = driven + bias.unsqueeze(-1).unsqueeze(1)
        
        # 3. Saturation curve (differentiable tanh)
        # Soft saturation: tanh (smooth)
        soft_sat = torch.tanh(driven * 2) / 2
        
        # Hard saturation: hard clip (differentiable approximation)
        hard_sat = torch.clamp(driven, min=-1.0, max=1.0)
        
        # Blend between soft and hard
        alpha = saturation_type.unsqueeze(-1).unsqueeze(1)
        saturated = alpha * hard_sat + (1 - alpha) * soft_sat
        
        # 4. Remove bias
        saturated = saturated - bias.unsqueeze(-1).unsqueeze(1)
        
        # 5. Wet/dry mix
        mix_factor = mix.unsqueeze(-1).unsqueeze(1)
        output = mix_factor * saturated + (1 - mix_factor) * audio
        
        return output
```

**Parameter Predictor:**
```python
class SaturationPredictor(nn.Module):
    def __init__(self, latent_dim=512):
        super().__init__()
        
        self.predictor = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 4)
        )
    
    def forward(self, z):
        params = self.predictor(z)
        
        return {
            'drive_db': torch.sigmoid(params[:, 0]) * 24,           # 0 to 24 dB
            'saturation_type': torch.sigmoid(params[:, 1]),         # 0 to 1
            'mix': torch.sigmoid(params[:, 2]),                     # 0 to 1
            'bias': torch.tanh(params[:, 3]) * 0.1                  # -0.1 to 0.1
        }
```

---

### Component 3: Stereo Imaging (Grey-Box) **NOVEL!**

**Purpose:** Stereo width control, spatial enhancement

**Challenge:** Our current system is mono! Need to either:
1. Work with stereo input/output
2. Generate stereo from mono (upmixing)

#### Approach A: Mid-Side Processing (If Stereo Available)

```python
class StereoImager(nn.Module):
    """
    Mid-Side stereo processing for width control.
    
    Mid (M) = (L + R) / 2     # Center content
    Side (S) = (L - R) / 2    # Stereo content
    
    Adjust width by scaling Side signal.
    """
    def __init__(self):
        super().__init__()
    
    def forward(self, audio_stereo, width, bass_mono_freq=150):
        """
        Args:
            audio_stereo: [B, 2, T] (left, right)
            width: [B] stereo width factor (0=mono, 1=normal, 2=wide)
            bass_mono_freq: Keep bass centered (mono below this freq)
        """
        left = audio_stereo[:, 0:1, :]   # [B, 1, T]
        right = audio_stereo[:, 1:2, :]  # [B, 1, T]
        
        # Convert to Mid-Side
        mid = (left + right) / 2
        side = (left - right) / 2
        
        # Optional: Keep bass mono (frequency-dependent width)
        if bass_mono_freq > 0:
            # High-pass filter on side signal
            side = self._highpass_filter(side, cutoff=bass_mono_freq)
        
        # Apply width
        width_factor = width.unsqueeze(-1).unsqueeze(1)
        side_adjusted = side * width_factor
        
        # Convert back to Left-Right
        left_out = mid + side_adjusted
        right_out = mid - side_adjusted
        
        return torch.cat([left_out, right_out], dim=1)
    
    def _highpass_filter(self, audio, cutoff=150, sample_rate=44100):
        """Simple first-order highpass (differentiable)."""
        # Simplified: proper implementation would use biquad
        # For now, return as-is (TODO: implement proper filter)
        return audio
```

#### Approach B: Mono-to-Stereo Neural Upmixer (Novel!)

```python
class MonoToStereoUpmixer(nn.Module):
    """
    Neural network to generate stereo from mono.
    
    Learns to:
    - Predict which frequencies should be wide
    - Generate decorrelated left/right channels
    - Maintain mono compatibility
    """
    def __init__(self, latent_dim=512):
        super().__init__()
        
        # Predict stereo width per frequency band
        self.width_predictor = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 10),  # 10 frequency bands
            nn.Sigmoid()         # Width per band (0=mono, 1=wide)
        )
        
        # Neural decorrelator (generates stereo difference)
        self.decorrelator = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=15, padding=7),
            nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=15, padding=7),
            nn.ReLU(),
            nn.Conv1d(32, 1, kernel_size=15, padding=7),
            nn.Tanh()  # Difference signal
        )
    
    def forward(self, audio_mono, z):
        """
        Args:
            audio_mono: [B, 1, T]
            z: [B, latent_dim]
        
        Returns:
            audio_stereo: [B, 2, T]
        """
        # Predict frequency-dependent width
        width_bands = self.width_predictor(z)  # [B, 10]
        
        # Generate decorrelated signal (difference)
        difference = self.decorrelator(audio_mono)  # [B, 1, T]
        
        # Apply width (simplified - would use filterbank in practice)
        # TODO: Implement proper multi-band splitting
        width_avg = width_bands.mean(dim=1, keepdim=True).unsqueeze(-1)
        difference_scaled = difference * width_avg
        
        # Create stereo
        left = audio_mono + difference_scaled * 0.5
        right = audio_mono - difference_scaled * 0.5
        
        return torch.cat([left, right], dim=1)
```

---

### Component 4: Differentiable Limiter (Grey-Box)

**Purpose:** Final loudness maximization, peak control

#### Parameters to Learn:
```python
- threshold: -12 to 0 dB         # Limiting ceiling
- release: 1 to 100 ms           # How fast it recovers
- lookahead: 0 to 5 ms           # Peak anticipation
- ceiling: -0.1 to -3.0 dB       # True peak ceiling
```

#### Implementation:

```python
class DifferentiableLimiter(nn.Module):
    """
    Brickwall limiter with differentiable gain computer.
    
    Similar to compressor but with ∞:1 ratio and lookahead.
    """
    def __init__(self, sample_rate=44100):
        super().__init__()
        self.sr = sample_rate
    
    def forward(self, audio, threshold_db, release_ms, ceiling_db):
        """
        Args:
            audio: [B, 1, T]
            threshold_db: [B] limiting threshold
            release_ms: [B] release time
            ceiling_db: [B] output ceiling
        """
        # 1. Detect peaks
        peaks = self._detect_peaks(audio)
        
        # 2. Compute gain reduction (anything above threshold)
        threshold = 10 ** (threshold_db.unsqueeze(-1).unsqueeze(1) / 20)
        peaks_db = 20 * torch.log10(peaks + 1e-7)
        threshold_db_expanded = threshold_db.unsqueeze(-1).unsqueeze(1)
        
        # Gain reduction needed
        over_threshold = torch.clamp(peaks_db - threshold_db_expanded, min=0)
        gain_reduction_db = over_threshold  # Full reduction (∞:1 ratio)
        
        # Smooth with release
        gain_reduction_smooth = self._smooth_gain(
            gain_reduction_db, release_ms
        )
        
        # Convert to linear
        gain = 10 ** (-gain_reduction_smooth / 20)
        
        # 3. Apply limiting
        limited = audio * gain
        
        # 4. Apply ceiling
        ceiling = 10 ** (ceiling_db.unsqueeze(-1).unsqueeze(1) / 20)
        limited = torch.clamp(limited, min=-ceiling, max=ceiling)
        
        return limited
    
    def _detect_peaks(self, audio, window_size=1024):
        """Peak detection with local maxima."""
        # Simplified: Use max pooling for peak detection
        peaks = F.max_pool1d(
            torch.abs(audio),
            kernel_size=window_size,
            stride=1,
            padding=window_size // 2
        )
        return peaks
    
    def _smooth_gain(self, gain_reduction_db, release_ms):
        """Smooth gain reduction with release time."""
        # Simplified exponential smoothing
        # Proper implementation would use state-based release
        kernel_size = int(release_ms.mean() * self.sr / 1000)
        kernel_size = max(kernel_size, 1)
        
        smoothed = F.avg_pool1d(
            gain_reduction_db,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2
        )
        
        return smoothed
```

---

## 🏗️ Complete Phase 2 Architecture

### GreyBoxMasteringChain Module

```python
class GreyBoxMasteringChain(nn.Module):
    """
    Complete mastering chain with grey-box components.
    
    Signal flow:
    1. EQ (Phase 1, kept)
    2. Compression (grey-box)
    3. Saturation (grey-box)
    4. Stereo Imaging (grey-box/neural)
    5. Limiting (grey-box)
    """
    def __init__(self, latent_dim=512, sample_rate=44100):
        super().__init__()
        
        # Parameter predictors
        self.compressor_predictor = CompressorPredictor(latent_dim)
        self.saturation_predictor = SaturationPredictor(latent_dim)
        self.stereo_predictor = StereoPredictor(latent_dim)
        self.limiter_predictor = LimiterPredictor(latent_dim)
        
        # DSP components
        self.compressor = DifferentiableCompressor(sample_rate)
        self.saturator = DifferentiableSaturation()
        self.stereo_imager = MonoToStereoUpmixer(latent_dim)
        self.limiter = DifferentiableLimiter(sample_rate)
        
        # Optional neural catch-all (for anything grey-box misses)
        self.neural_residual = SimpleWaveUNet(latent_dim)
    
    def forward(self, audio, z):
        """
        Args:
            audio: [B, 1, T] mono input
            z: [B, latent_dim] latent code
        
        Returns:
            output: [B, 2, T] stereo output
            params: dict of all parameters
        """
        # 1. Compression
        comp_params = self.compressor_predictor(z)
        compressed = self.compressor(
            audio,
            comp_params['threshold'],
            comp_params['ratio'],
            comp_params['attack'],
            comp_params['release'],
            comp_params['makeup_gain']
        )
        
        # 2. Saturation
        sat_params = self.saturation_predictor(z)
        saturated = self.saturator(
            compressed,
            sat_params['drive_db'],
            sat_params['saturation_type'],
            sat_params['mix'],
            sat_params['bias']
        )
        
        # 3. Stereo imaging (mono to stereo)
        stereo = self.stereo_imager(saturated, z)  # [B, 2, T]
        
        # 4. Optional: Neural residual (adds detail)
        residual = self.neural_residual(saturated, z)  # [B, 1, T]
        # Add residual to mid signal
        mid = (stereo[:, 0:1] + stereo[:, 1:2]) / 2
        mid_enhanced = mid + residual * 0.3  # Subtle blend
        stereo[:, 0:1] = mid_enhanced
        stereo[:, 1:2] = mid_enhanced
        
        # 5. Limiting (final loudness)
        lim_params = self.limiter_predictor(z)
        limited = self.limiter(
            stereo,
            lim_params['threshold'],
            lim_params['release'],
            lim_params['ceiling']
        )
        
        # Collect all parameters for visualization
        params = {
            'compression': comp_params,
            'saturation': sat_params,
            'stereo': self.stereo_imager.width_predictor(z),
            'limiting': lim_params
        }
        
        return limited, params
```

---

## 📊 Training Strategy for Phase 2

### Approach A: End-to-End Training

**Train all components simultaneously:**

```python
# Model
model = Phase2MasteringModel(
    encoder=AudioEncoder(latent_dim=512),
    eq_decoder=ParametricDecoder(...),      # From Phase 1
    chain=GreyBoxMasteringChain(...)        # New!
)

# Loss (same as Phase 1 but adjusted for stereo)
loss = CombinedLoss(
    stft_loss=MultiScaleSTFTLoss(),
    perceptual_loss=AWeightedLoss(),
    loudness_loss=LUFSLoss(),
    param_reg=ParameterRegularizationLoss()
)

# Training
for unmastered, mastered in dataloader:
    # Forward pass
    z = model.encoder(unmastered)
    eq_out = model.eq_decoder(z, unmastered)
    chain_out, params = model.chain(eq_out, z)
    
    # Loss
    loss_value = loss(chain_out, mastered, params)
    
    # Backward
    loss_value.backward()
    optimizer.step()
```

### Approach B: Progressive Training

**Train components one at a time:**

1. **Phase 1:** EQ + Residual (done)
2. **Phase 2a:** Add Compressor (freeze EQ)
3. **Phase 2b:** Add Saturation (freeze EQ + Compressor)
4. **Phase 2c:** Add Stereo (freeze all previous)
5. **Phase 2d:** Add Limiter (freeze all previous)
6. **Phase 2e:** Fine-tune all together

---

## 🎯 Implementation Priority

### Must-Have (Core Mastering Chain):

1. **✅ EQ** - Phase 1 (done)
2. **🔄 Compression** - Phase 2 priority #1
3. **🔄 Limiting** - Phase 2 priority #2

### Nice-to-Have (Enhancement):

4. **Saturation** - Phase 2 priority #3
5. **Stereo Imaging** - Phase 2 priority #4 (novel!)

### Optional (Advanced):

6. De-essing (tame sibilance)
7. Multiband compression
8. Transient shaping
9. Mid-side EQ

---

## 📝 Implementation Roadmap

### Week 1-4: Phase 1 (Current)
- [x] EQ implementation
- [x] Training infrastructure
- [ ] Phase 1A/1B/1C training
- [ ] Evaluation

### Week 5-6: Phase 2a (Compression)
- [ ] Implement differentiable compressor
- [ ] Parameter predictor
- [ ] Train with EQ + Compression
- [ ] Evaluate improvement

### Week 7: Phase 2b (Limiting)
- [ ] Implement differentiable limiter
- [ ] Train with EQ + Compression + Limiting
- [ ] LUFS compliance testing

### Week 8: Phase 2c (Saturation - Optional)
- [ ] Implement differentiable saturation
- [ ] Train full chain
- [ ] Evaluate harmonic content

### Week 9: Phase 2d (Stereo - Optional/Novel)
- [ ] Implement mono-to-stereo upmixer
- [ ] Train full stereo chain
- [ ] Evaluate spatial characteristics

### Week 10: Final Integration
- [ ] Train all components end-to-end
- [ ] Compare with Phase 1
- [ ] Ablation studies
- [ ] Final evaluation

---

## 📚 Research References for Phase 2

### Compression:
- Giannoulis et al. (2012): "Digital Dynamic Range Compressor Design"
- Reiss & McPherson (2015): "Audio Effects: Theory, Implementation and Application"

### Saturation:
- Parker et al. (2019): "Modelling of Nonlinear State-Space Systems Using Deep Neural Networks"
- Steinmetz et al. (2020): "Neural Waveshaping Synthesis"

### Stereo:
- Jot et al. (1995): "Spatial Enhancement of Audio Recordings"
- *Novel contribution*: Neural mono-to-stereo upmixing

### Limiting:
- Giannoulis et al. (2012): "Digital Dynamic Range Compressor Design"
- ITU-R BS.1770: LUFS measurement for loudness compliance

---

## 🎓 Why Grey-Box > Black-Box for Phase 2

### Advantages:

**1. Interpretability**
- Engineers can see/adjust compression parameters
- Understand what model is doing at each stage
- Easier debugging

**2. Generalization**
- DSP components have known behavior
- Less likely to overfit
- Works on unseen genres

**3. Efficiency**
- Fewer parameters than pure neural
- Faster inference
- Lower computational cost

**4. Professional Trust**
- Engineers understand compressor settings
- Can override neural decisions
- Hybrid human-AI workflow

### Trade-off:

**Black-box (Phase 1 residual):**
- More flexible
- Can learn unknown transformations
- But: harder to interpret

**Grey-box (Phase 2 chain):**
- Less flexible
- Limited to known DSP operations
- But: fully interpretable

**Best approach:** Use both!
- Grey-box for known operations (EQ, compression, limiting)
- Black-box catch-all for everything else

---

## 🔬 Novel Research Contributions (Phase 2)

### 1. Neural Mono-to-Stereo Upmixing
- Learn frequency-dependent stereo width
- Maintain mono compatibility
- Genre-adaptive spatial enhancement

### 2. End-to-End Grey-Box Mastering
- Complete differentiable mastering chain
- Joint optimization of all stages
- Interpretable + expressive

### 3. Adaptive Dynamic Processing
- Learn optimal compression per genre/style
- Context-dependent attack/release times
- Novel use of FiLM conditioning in dynamics

---

## 📊 Expected Improvements

### Phase 1 vs Phase 2:

| Metric | Phase 1 (EQ only) | Phase 2 (Full chain) |
|--------|-------------------|----------------------|
| **STFT Loss** | 3-6 | 1-3 |
| **SNR (dB)** | 20-25 | 25-30 |
| **Loudness** | Variable | Compliant (-14 LUFS) |
| **Dynamics** | No control | Controlled |
| **Stereo Width** | Mono | Stereo/Enhanced |
| **Interpretability** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |

---

**Last Updated:** [Date]  
**Version:** 1.0 (Phase 2 Planning)

**Next Steps:**
1. Complete Phase 1 training
2. Evaluate Phase 1 results
3. Begin Phase 2a (Compression) implementation
4. Iteratively add remaining components
