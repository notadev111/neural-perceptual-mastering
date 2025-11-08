# Neural Perceptual Mastering

Neural audio mastering with perceptual losses and limited training data.

## Repository Structure

### 📁 Main Research (Neural Approach)
- `src/` - Neural mastering implementation
- `configs/` - Training configurations  
- `notebooks/` - Research notebooks
- `scripts/` - Training and evaluation scripts

### 📁 Semantic Mastering System
- `semantic_mastering_system/` - Complete semantic mastering system based on real engineer data
  - Uses SocialFX dataset with 1,595 real mixing examples
  - Natural language EQ control ("warm", "bright", "punchy", etc.)
  - Immediate usability with comprehensive analysis tools
  - See [`semantic_mastering_system/README.md`](semantic_mastering_system/README.md) for details

## Quick Start

### Neural Mastering (Research)
```bash
# Setup environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
pip install -r requirements.txt

# Training and evaluation scripts in scripts/
```

### Semantic Mastering (Ready-to-Use)
```bash
# Navigate to semantic system
cd semantic_mastering_system

# Apply semantic mastering
python semantic_mastering.py --input your_mix.wav --preset warm

# Analyze and visualize profiles  
python analyze_eq_profiles.py --profiles warm bright --plot-response
```

Both systems can be used independently or in combination for comprehensive mastering workflows.
