# Semantic Mastering System

A complete semantic mastering EQ system based on real engineer data from the SocialFX dataset.

## Overview

This system uses the SocialFX dataset from the research paper 'Can Large Language Models Predict Audio Effects Parameters
from Natural Language?' (Seungheon Doh) which contains actual EQ parameters from 1,595 mixing examples by 1,233 engineers to create semantic mastering profiles. Unlike traditional parametric EQ, you can specify what you want using natural language terms like "warm", "bright", "punchy", etc.

## Key Features

- **Real Data**: Based on actual engineer decisions from SocialFX-original dataset
- **765 Semantic Terms**: Natural language EQ control ("warm", "bright", "aggressive", etc.)
- **Mastering Context**: Automatically scales mixing parameters for mastering (÷2.5)
- **5 Validated Presets**: warm, bright, punchy, smooth, balanced
- **Comprehensive Analysis**: Visualisation, A/B testing, detailed reports

## Quick Start

### 1. Setup
```bash
# From the semantic_mastering_system folder
cd semantic_mastering_system

# Install dependencies (if not already done from main repo)
pip install pandas pyarrow huggingface_hub datasets
pip install -r ../requirements.txt

# Login to HuggingFace (required for dataset access)
huggingface-cli login
```

### 2. Basic Usage
```bash
# Inspect the dataset
python semantic_mastering.py --inspect

# Apply semantic mastering
python semantic_mastering.py --input your_mix.wav --preset warm
python semantic_mastering.py --input your_mix.wav --preset bright

# Test all available presets
python semantic_mastering.py --input your_mix.wav --test-all
```

### 3. Analysis and Visualization
```bash
# Visualize EQ curves
python analyze_eq_profiles.py --profiles warm bright punchy --plot-response

# A/B test two profiles
python test_eq_profiles.py --audio your_mix.wav --compare warm bright

# Full analysis with reports
python analyze_eq_profiles.py --audio your_mix.wav --profiles warm --analyze-all --report
```

### 4. Demo (No Audio Files Needed)
```bash
# Run complete demo with synthetic audio
python demo_analysis.py
```

## Files Description

| File | Purpose |
|------|---------|
| `semantic_mastering.py` | Main semantic mastering system |
| `analyze_eq_profiles.py` | Visualization and frequency analysis tools |
| `test_eq_profiles.py` | A/B testing and comparison system |
| `demo_analysis.py` | Demo with synthetic test signals |
| `test_semantic_mastering.py` | Test suite for validation |

## How It Works

1. **Dataset Loading**: Downloads SocialFX-original from HuggingFace (1,595 examples)
2. **Parameter Processing**: Converts 40-parameter EQ format to 6-band mastering EQ
3. **Semantic Mapping**: Maps natural language terms to EQ curves
4. **Mastering Scaling**: Reduces mixing gains (±6-12dB) to mastering range (±2dB)
5. **Audio Processing**: Applies EQ using dasp-pytorch

## Available Profiles

### Validated Presets
- **warm**: Enhanced low-end, gentle high cut
- **bright**: Enhanced high frequencies, air and detail  
- **punchy**: Enhanced mids, aggressive and forward
- **smooth**: Gentle adjustments, refined sound
- **balanced**: Neutral with subtle enhancements

### Dataset Terms (Examples)
From 765 available terms: aggressive, soft, clear, heavy, crisp, gentle, powerful, etc.

## Technical Details

- **Input Format**: WAV, FLAC, MP3 (auto-resampled to 44.1kHz)
- **EQ Format**: 6-band parametric EQ via dasp-pytorch
- **Scaling Factor**: 2.5x reduction from mixing to mastering context
- **Confidence Scores**: Each profile includes reliability metric (0-100%)

## Analysis Capabilities

### Frequency Response
- Visual EQ curves for any profile
- Per-band analysis (Sub, Bass, Low-mid, Mid, High-mid, Treble)
- Confidence indicators based on source data

### Audio Analysis
- Before/after spectrum comparison
- Loudness analysis (RMS, LUFS, peak)
- Tonal analysis (spectral centroid, rolloff)
- Dynamic range metrics

### A/B Testing
- Matched audio files for blind testing
- Detailed metric comparisons
- Listening test instructions
- Batch testing capabilities

## Integration Notes

This semantic mastering system is **separate** from the main neural perceptual mastering research in the parent directory. It provides:

- **Immediate usability**: Works out-of-the-box with real data
- **Interpretable results**: Clear understanding of what each term does
- **Validated approach**: Based on actual engineer preferences
- **Complementary functionality**: Can be used alongside neural approaches

## Research Context

Based on the LLM2Fx paper methodology but adapted for mastering context:
- Uses SocialFX-original dataset (Steinmetz et al.)
- Implements semantic-to-parameter mapping
- Scales for mastering vs mixing context
- Provides confidence metrics and validation

## Output Examples

### Generated Files
- `your_mix_warm.wav` - Processed audio
- `frequency_response.png` - EQ curve visualization  
- `spectrum_analysis.png` - Before/after spectrum
- `analysis_report.md` - Detailed technical report

### Typical Workflow
1. Visualize available profiles → Choose candidates
2. A/B test top choices → Select best fit
3. Generate detailed analysis → Validate technically
4. Apply to full project → Consistent results