# Project Structure

Complete file organization for the Neural Perceptual Audio Mastering system.

```
neural-mastering/
│
├── README.md                          # Comprehensive project documentation
├── QUICKSTART.md                      # Quick start guide for getting started
├── requirements.txt                   # Python dependencies
├── .gitignore                         # Git ignore patterns
│
├── configs/                           # Training configurations
│   ├── phase1a_parametric_only.yaml   # Phase 1A: Parametric EQ only
│   ├── phase1b_hybrid.yaml            # Phase 1B: EQ + Residual
│   └── phase1c_adaptive.yaml          # Phase 1C: Adaptive bands (novel)
│
├── src/                               # Source code
│   ├── __init__.py                    # Package initialization
│   ├── models.py                      # Model architectures
│   │   ├── AudioEncoder               # TCN encoder
│   │   ├── ParametricDecoder          # Differentiable EQ
│   │   ├── AdaptiveParametricDecoder  # Adaptive bands (novel)
│   │   ├── ResidualDecoder            # Wave-U-Net residual
│   │   ├── MasteringModel_Phase1A     # Phase 1A model
│   │   ├── MasteringModel_Phase1B     # Phase 1B model
│   │   └── MasteringModel_Phase1C     # Phase 1C model
│   │
│   ├── losses.py                      # Loss functions
│   │   ├── MultiScaleSTFTLoss         # Multi-resolution spectral loss
│   │   ├── AWeightedLoss              # A-weighted perceptual loss ***YOUR FUNCTION***
│   │   ├── LUFSLoss                   # Loudness matching
│   │   ├── ParameterRegularizationLoss# EQ parameter regularization
│   │   ├── MelSpectralLoss            # Mel-frequency spectral distance
│   │   └── CombinedLoss               # Weighted combination
│   │
│   ├── data_loader.py                 # Dataset and data loading
│   │   ├── MasteringDataset           # Pre/post mastering pairs
│   │   ├── get_dataloaders            # Create train/val/test loaders
│   │   └── create_dummy_dataset       # Generate test data
│   │
│   ├── train.py                       # Training script
│   │   ├── train_epoch()              # Train one epoch
│   │   ├── validate()                 # Validation
│   │   └── save_checkpoint()          # Save model
│   │
│   ├── evaluate.py                    # Evaluation and metrics
│   │   ├── compute_snr()              # Signal-to-noise ratio
│   │   ├── compute_mel_distance()     # Mel spectral distance
│   │   ├── analyze_eq_parameters()    # EQ parameter analysis
│   │   ├── visualize_eq_curve()       # Plot frequency response
│   │   └── evaluate_model()           # Comprehensive evaluation
│   │
│   └── inference.py                   # Process audio through model
│       ├── load_model()               # Load trained model
│       ├── preprocess_audio()         # Load and normalize audio
│       ├── process_audio()            # Process in segments
│       └── postprocess_audio()        # Save output
│
├── data/                              # Dataset (user-provided)
│   ├── unmastered/                    # Pre-mastering audio
│   │   ├── song001.wav
│   │   ├── song002.wav
│   │   └── ...
│   └── mastered/                      # Post-mastering audio (targets)
│       ├── song001.wav
│       ├── song002.wav
│       └── ...
│
├── checkpoints/                       # Saved model checkpoints (generated)
│   ├── phase_1A/
│   │   ├── best_model.pt
│   │   └── phase_1A_epoch_*.pt
│   ├── phase_1B/
│   └── phase_1C/
│
└── runs/                              # Tensorboard logs (generated)
    ├── phase_1A_20250101_120000/
    ├── phase_1B_20250108_120000/
    └── phase_1C_20250115_120000/
```

## 📁 Key File Details

### Configuration Files (`configs/`)
YAML files defining model architecture, hyperparameters, and training settings for each phase.

### Source Code (`src/`)

#### `models.py` (521 lines)
- **AudioEncoder**: TCN-based encoder with dilated convolutions
- **ParametricDecoder**: MLP → differentiable biquad EQ (white-box)
- **AdaptiveParametricDecoder**: Novel adaptive band selection
- **ResidualDecoder**: Wave-U-Net with FiLM conditioning (black-box)
- **Phase Models**: 1A (parametric only), 1B (hybrid), 1C (adaptive)

#### `losses.py` (305 lines) **← YOUR A-WEIGHTED LOSS IS HERE**
- **MultiScaleSTFTLoss**: Captures fine and coarse spectral detail
- **AWeightedLoss**: Perceptual loss using ISO 226 A-weighting filter
- **LUFSLoss**: Loudness matching (ITU-R BS.1770)
- **ParameterRegularizationLoss**: Encourages minimal EQ adjustments
- **MelSpectralLoss**: Mel-frequency perceptual metric
- **CombinedLoss**: Weighted combination of all losses

#### `data_loader.py` (210 lines)
- **MasteringDataset**: Loads pre/post mastering pairs
- Self-supervised segmentation (5s chunks)
- Audio augmentation (random gain, polarity flip)
- Automatic mono conversion and resampling

#### `train.py` (242 lines)
- Phase-aware model selection
- Training loop with validation
- Tensorboard logging
- Checkpoint saving
- Learning rate scheduling

#### `evaluate.py` (329 lines)
- Multiple perceptual metrics (STFT, mel distance, SNR, LUFS, A-weighted)
- EQ parameter analysis
- Frequency response visualization
- Metric distribution plots

#### `inference.py` (248 lines)
- Load trained model
- Process long audio files in segments
- Extract and display EQ parameters
- Save mastered output

## 🎯 Training Workflow

```
Phase 1A (Baseline)
└─> configs/phase1a_parametric_only.yaml
    └─> src/train.py
        └─> checkpoints/phase_1A/best_model.pt
            └─> src/evaluate.py
                └─> results_phase1a/

Phase 1B (Hybrid)
└─> configs/phase1b_hybrid.yaml
    └─> src/train.py
        └─> checkpoints/phase_1B/best_model.pt
            └─> src/evaluate.py
                └─> results_phase1b/

Phase 1C (Novel)
└─> configs/phase1c_adaptive.yaml
    └─> src/train.py
        └─> checkpoints/phase_1C/best_model.pt
            └─> src/evaluate.py
                └─> results_phase1c/
```

## 📊 Generated Outputs

### During Training:
- `runs/` - Tensorboard logs (loss curves, parameter tracking)
- `checkpoints/` - Model weights saved every N epochs

### During Evaluation:
- `evaluation_results/phase_X_metrics.json` - Numerical results
- `evaluation_results/phase_X_eq_curve.png` - Average EQ frequency response
- `evaluation_results/phase_X_metrics.png` - Metric distribution histograms

### During Inference:
- `audio/*_mastered.wav` - Processed audio outputs

## 🔍 Where to Find Key Components

| What | Where |
|------|-------|
| **Your A-weighted loss function** | `src/losses.py` line 66-127 |
| **Differentiable EQ** | `src/models.py` line 69-120 |
| **TCN encoder** | `src/models.py` line 19-67 |
| **Wave-U-Net residual** | `src/models.py` line 320-400 |
| **Phase 1A model** | `src/models.py` line 410-447 |
| **Phase 1B model** | `src/models.py` line 450-495 |
| **Phase 1C model** | `src/models.py` line 498-544 |
| **Training loop** | `src/train.py` line 22-88 |
| **Evaluation metrics** | `src/evaluate.py` line 18-160 |
| **Data loading** | `src/data_loader.py` line 18-190 |

## 🚀 Quick Commands

```bash
# Train Phase 1A
python src/train.py --config configs/phase1a_parametric_only.yaml

# Evaluate Phase 1A
python src/evaluate.py \
    --checkpoint checkpoints/phase_1A/best_model.pt \
    --config configs/phase1a_parametric_only.yaml \
    --save_dir evaluation_results

# Process audio
python src/inference.py \
    --checkpoint checkpoints/phase_1A/best_model.pt \
    --config configs/phase1a_parametric_only.yaml \
    --input audio/unmastered.wav \
    --output audio/mastered.wav

# Monitor training
tensorboard --logdir runs/
```

## 📦 Dependencies

Install via `requirements.txt`:
- `torch>=2.0.0` - Deep learning framework
- `torchaudio>=2.0.0` - Audio processing
- `librosa>=0.10.0` - Audio analysis
- `tensorboard>=2.13.0` - Training visualization
- `matplotlib>=3.7.0` - Plotting
- `scipy>=1.10.0` - Scientific computing
- `pyyaml>=6.0` - Config files

## 🔮 Future Extensions (Phase 2)

The modular structure allows easy addition of:
- Differentiable compressor in `src/models.py`
- Differentiable saturation/distortion
- Stereo imaging module
- Dynamic limiting
- Text-conditioning (CLAP embeddings)

All would be added to `ResidualDecoder` or a new `GreyBoxDecoder` class.

---

**Version:** 1.0 (Phases 1A-C)  
**Last Updated:** [Date]
