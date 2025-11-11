# Neural Perceptual Audio Mastering
**ELEC0036 Project UCL EEE**  
**Supervisor:** Prof. Miguel Rio  
**Student:** Daniel Dutulescu

---

## Project Overview

This project explores methods involving a hybrid neural DSP system for automatic audio mastering that combines the 
interpretability of traditional digital signal processing with deep learning.

Unlike purely black-box neural approaches, the system uses:
- **White box parametric EQ** (differentiable biquad filters) for transparent spectral shaping
- **Black box residual path** (Wave-U-Net) for non linear corrections
- **Perceptual loss functions** (A-weighting, multi-scale STFT) aligned with human auditory perception

---

## Why this project?

Audio mastering is the final stage of music production, preparing a mix for distribution and playback consistency 
across systems and formats. Historically, it emerged from the vinyl era, where engineers applied EQ and compression 
to optimise recordings for the physical medium. In the digital age, mastering remains essential to ensure clarity, 
loudness balance, and tonal cohesion across streaming platforms. (iZotope, Sage Audio)

This project explores whether machine learning models, specifically a Temporal Convolutional Network (TCN)
combined with Differentiable Digital Signal Processing (DDSP) can demonstrate effective mastering tools,
starting with equalisation (EQ), using limited training data.

The aim is to democratise professional mastering for digital audio by providing a transparent, interpretable ML toolchain,
rather than a blackbox system. It’s an exploratory proof-of-concept into how ML can assist creative and technical decision
making, specifically as an accessible mastering tool for bedroom producers. 

## Novel Aspects

### 1. **Hybrid Architecture (White-box + Black-box)**
Most neural mastering systems are pure black-boxes. Our system combines:
- **Parametric EQ** (interpretable) for known corrections
- **Residual network** (flexible) for complex non-linearities (saturation, distortion, etc)

### 2. **Adaptive Band Selection (Phase 1C)**
Traditional EQ uses fixed band counts. **Our novel contribution:**
- Model learns which bands to activate (5-10 bands dynamically)
- Soft gating mechanism lets network decide band importance
- Genre-adaptive: different music styles need different EQ profiles

### 3. **Perceptual Loss Design**
We align loss functions with human hearing:
- **A-weighting filter** (ISO 226) emphasizes 2-5kHz where ears are most sensitive
- **Multi-scale STFT** captures both coarse and fine spectral detail
- **LUFS matching** for broadcast standard loudness (note to add specific LUFs formats in e.g spotify, apple music)

---

### Challenges:
Dataset creation: A key challenge is assembling a suitable dataset of pre- and post-mastered audio pairs, as no
open-source dataset of this kind currently exists, likely due to copyright restrictions and the relative rarity 
of released master versions compared to raw mixes. While there are several semantic mixing datasets that explore 
how natural language describes audio processing decisions, these are typically designed for mixing stems rather 
than than mastering or stereo material. As a result, a sub-research question within this project involves investigating 
the use of the Social-FX dataset (from the LLM2FX project) to study semantic effects control in the context of mastering.
More details can be found at GitHub.com/s

Research that enables this project

Our approach builds on:

| Component | Key Papers |
|-----------|-----------|
| **Differentiable DSP** | Nercessian et al. 2020 - "Differentiable IIR Filters" |
| **Multi-scale STFT Loss** | Koo et al. 2022 - "Music Demixing Challenge" |
| **A-weighted Perceptual Loss** | Wright & Välimäki 2016 - "Perceptual Audio Evaluation" |
| **TCN Encoder** | Comunità et al. 2025 - "Temporal Convolutions for Audio" |
| **Wave-U-Net Residual** | Stoller et al. 2018 - "Wave-U-Net for Audio Separation" |
| **Grey-box Systems** | Steinmetz & Reiss 2021 - "auraloss" |

### Key Insights:
1. **Differentiable biquad EQ** (Nercessian 2020): Makes traditional DSP trainable via gradient descent
2. **Perceptual spectral loss** (Koo 2022): Better than L1/L2 for audio quality
3. **A-weighting** (Wright 2016): Models human ear's frequency dependent sensitivity
4. **Hybrid architectures** enable both transparency and expressiveness

---

## Architecture

```
┌────────────────────────────────────────────────────────────┐
│                      Input Audio                            │
│                  (Unmastered track)                         │
└────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────────┐
│                    TCN Encoder                              │
│  • Strided convolutions (downsampling)                      │
│  • Dilated TCN blocks (long-range dependencies)            │
│  • Global average pooling                                   │
│  Output: Latent code z ∈ ℝ^512                             │
└────────────────────────────────────────────────────────────┘
                            │
                            ▼
            ┌───────────────┴───────────────┐
            │                               │
            ▼                               ▼
┌─────────────────────────┐   ┌─────────────────────────┐
│  Parametric Decoder     │   │  Residual Decoder       │
│  (White-box)            │   │  (Black-box)            │
│                         │   │                         │
│ • MLP predicts EQ params│   │ • Wave-U-Net with       │
│   - Frequencies         │   │   FiLM conditioning     │
│   - Gains (dB)          │   │ • Skip connections      │
│   - Q factors           │   │                         │
│ • Differentiable biquad │   │ Output: Residual        │
│   EQ cascade            │   │                         │
│                         │   │                         │
│ Output: EQ'd audio      │   │                         │
└─────────────────────────┘   └─────────────────────────┘
            │                               │
            └───────────────┬───────────────┘
                            │
                            ▼
                ┌───────────────────────┐
                │   Output Audio        │
                │   (Mastered track)    │
                └───────────────────────┘
```

---

## Training Strategy (Phased Approach)

We train **three models sequentially** to understand each component's contribution:

### **Phase 1A: Parametric EQ Only** (Baseline)
- Pure white-box model
- Tests if differentiable EQ alone is sufficient
- **Expected:** Good spectral shaping, limited expressiveness
- **Timeline:** 1 week

### **Phase 1B: EQ + Residual** (Full Hybrid)
- Add black-box neural path
- Tests if residual improves over EQ-only
- **Expected:** Better quality, less interpretability of residual
- **Timeline:** 1 week

### **Phase 1C: Adaptive Bands** (Novel)
- Model learns which bands to use (5-10)
- **Novel contribution** to the field
- **Expected:** Genre-adaptive, more efficient
- **Timeline:** 1 week

---

## Loss Functions

Our training objective combines multiple perceptual criteria:

### 1. **Multi-scale STFT Loss** (w=1.0)
```python
L_spectral = Σ_scales || STFT(y_pred) - STFT(y_target) ||
```
Captures both coarse (low-freq) and fine (high-freq) spectral detail.

### 2. **A-weighted Perceptual Loss** (w=0.1)
```python
L_perceptual = || A_filter(y_pred) - A_filter(y_target) ||_1
```
Uses ISO 226 A-weighting curve (models human ear sensitivity peak at 2-5kHz).

### 3. **LUFS Matching Loss** (w=0.01)
```python
L_loudness = | LUFS(y_pred) - LUFS(y_target) |
```
Ensures broadcast-standard loudness (ITU-R BS.1770).

### 4. **Parameter Regularization** (w=0.001)
```python
L_reg = λ_gain * ||gains||_1 + λ_Q * ||(Q - 1)^2||_2
```
Encourages minimal EQ adjustments and reasonable Q factors.

**Total Loss:**
```
L_total = 1.0·L_spectral + 0.1·L_perceptual + 0.01·L_loudness + 0.001·L_reg
```

---

## Dataset Preparation

**Note:** Dataset remains **private** due to copyright of commercial music.

### Directory Structure:
```
data/
├── unmastered/
│   ├── song001.wav
│   ├── song002.wav
│   └── ...
└── mastered/
    ├── song001.wav  (professionally mastered version)
    ├── song002.wav
    └── ...
```

### Requirements:
- **Pre/post mastering pairs** (same song before and after professional mastering)
- **Format:** WAV, 44.1kHz, stereo 
- **Duration:** Any length (automatically segmented into 5s chunks during training)
- **Recommended:** 20-50 diverse songs across genres

### How Pairs Were Created:
1. Source unmastered stems from production projects
2. Send to professional mastering engineer
3. Align mastered output with original (same length)

---

## Installation

### Requirements:
```bash
Python 3.8+
PyTorch 2.0+
CUDA 11.8+ (for GPU training)
```

### Setup:
```bash
# Clone repository
git clone https://github.com/notadev111/neural-perceptual-mastering
cd neural-mastering

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## Usage

### 1. **Prepare Your Dataset**
```bash
# Create dummy dataset for testing
python src/data_loader.py

# Or organize your own dataset:
mkdir -p data/{unmastered,mastered}
# Place your audio files...
```

### 2. **Train Phase 1A (Parametric Only)**
```bash
python src/train.py --config configs/phase1a_parametric_only.yaml
```
Monitor training:
```bash
tensorboard --logdir runs/
```

### 3. **Train Phase 1B (Hybrid)**
After Phase 1A converges:
```bash
python src/train.py --config configs/phase1b_hybrid.yaml
```

### 4. **Train Phase 1C (Adaptive Bands)**
Test novel contribution:
```bash
python src/train.py --config configs/phase1c_adaptive.yaml
```

### 5. **Evaluate Models**
```bash
python src/evaluate.py \
    --checkpoint checkpoints/phase_1A_best/best_model.pt \
    --config configs/phase1a_parametric_only.yaml \
    --save_dir evaluation_results
```

---

## Evaluation Metrics

We measure performance using perceptually motivated metrics:

### Objective Metrics:
1. **Multi-scale STFT Loss:** Spectral reconstruction accuracy
2. **Mel Spectral Distance:** Perceptual frequency similarity
3. **LUFS Error:** Loudness matching accuracy (dB)
4. **A-weighted Error:** Human-perceived audio difference
5. **SNR (dB):** Signal-to-noise ratio

### Interpretability Analysis:
- **EQ parameter statistics** (mean frequencies, gains, Q factors)
- **Band usage patterns** (Phase 1C: which bands are active?)
- **Frequency response visualization**

---

## Debugging & Analysis

### View EQ Curves:
The evaluation script automatically generates:
- `phase_1A_eq_curve.png` - Average EQ frequency response
- `phase_1A_metrics.png` - Distribution of evaluation metrics

### Inspect Parameters:
```python
import torch
checkpoint = torch.load('checkpoints/phase_1A_best/best_model.pt')
# View learned EQ parameters
```

### Audio Demos:
```bash
# Process a test file through trained model
python src/inference.py \
    --checkpoint checkpoints/phase_1B_best/best_model.pt \
    --input audio/test_unmastered.wav \
    --output audio/test_mastered.wav
```

---

## Phase 2: Grey-box Residual (Future Work)

After Phase 1 completes, we plan to add **interpretable DSP components** to the residual path:

### Proposed Components:
1. **Differentiable Compressor**
   - Parameters: threshold, ratio, attack, release
   - Use `dasp-pytorch` or custom implementation

2. **Differentiable Saturation/Distortion**
   - Soft-clipping for harmonic richness
   - Neural waveshaper

3. **Stereo Imaging (Novel)**
   - Mid-side processing
   - Width control
   - Neural approach to stereo enhancement

4. **Dynamic Limiting**
   - Peak limiting for loudness maximization
   - Lookahead buffer
   - Release time learning

### Architecture:
```
Residual Path = Grey-box DSP + Neural Catch-all
                (Compressor → Saturation → Stereo → Neural)
```

---

## Key Implementation Details

### Differentiable Biquad EQ
We use **torchaudio.transforms.BandBiquad** for differentiable filtering:
```python
eq_filter = torchaudio.transforms.BandBiquad(
    sample_rate=44100,
    central_freq=predicted_freq,
    Q=predicted_q,
    gain=predicted_gain_db
)
output = eq_filter(input_audio)
```
Gradients flow through the filter to update frequency/gain/Q predictions.

### TCN Encoder
**Why TCN over RNN/Transformer?**
- **Parallelizable** (faster than RNN)
- **Long context** via dilated convolutions (receptive field = thousands of samples)
- **Efficient** (fewer parameters than Transformer for audio)

### Wave-U-Net Residual
**Why U-Net over fully convolutional?**
- **Skip connections** preserve high-frequency detail
- **Multi-scale** processing (downsampling → bottleneck → upsampling)
- **FiLM conditioning** injects latent code at each layer

---

1. Nercessian, S. et al. (2020). "Differentiable IIR Filters for Machine Learning Applications"
2. Koo, K. et al. (2022). "Music Demixing Challenge: Multi-scale STFT Loss"
3. Wright, A. & Välimäki, V. (2016). "Perceptual Evaluation of Audio Quality"
4. Comunità, M. et al. (2025). "Temporal Convolutional Networks for Audio Processing"
5. Stoller, D. et al. (2018). "Wave-U-Net: Multi-scale Neural Network for Audio Source Separation"
6. Steinmetz, C. & Reiss, J. (2021). "auraloss: Audio-focused Loss Functions in PyTorch"
7. ISO 226:2003: "Acoustics - Normal Equal-loudness-level Contours"
8. ITU-R BS.1770: "Algorithms to Measure Audio Programme Loudness and True-peak Audio Level"

---

## License

This project is for **academic purposes only** at UCL.  
Dataset remains **private** due to copyright restrictions on commercial music.

---

## Contact
**Email:** zceeddu@ucl.ac.uk


**Project Repository:** https://github.com/notadev111/neural-perceptual-mastering
**Documentation:** README.md, ARCHITECTURE.md, LITERATURE_NOTES.md, PROJECT_STRUCTURE.md

---

**Last Updated:** 10/11/2025
**Version:** 1.0 (Phase 1A-C Implementation)
