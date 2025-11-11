# Start guide

```bash
# 1. Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Verify installation
python -c "import torch; print(f'PyTorch {torch.__version__} installed')"
```




## Training Workflow

### Phase 1A (Week 1-2)
```bash
# 1. Train parametric only model
python src/train.py --config configs/phase1a_parametric_only.yaml

# 2. Evaluate
python src/evaluate.py \
    --checkpoint checkpoints/phase_1A/best_model.pt \
    --config configs/phase1a_parametric_only.yaml \
    --save_dir results_phase1a
```



### Phase 1B (Week 2-4)
```bash
# 1. Train hybrid model
python src/train.py --config configs/phase1b_hybrid.yaml

# 2. Evaluate
python src/evaluate.py \
    --checkpoint checkpoints/phase_1B/best_model.pt \
    --config configs/phase1b_hybrid.yaml \
    --save_dir results_phase1b
```


### Phase 1C (Week 4-6)
```bash
# 1. Train adaptive bands
python src/train.py --config configs/phase1c_adaptive.yaml

# 2. Evaluate
python src/evaluate.py \
    --checkpoint checkpoints/phase_1C/best_model.pt \
    --config configs/phase1c_adaptive.yaml \
    --save_dir results_phase1c
```
## Process Audio (After Training)

```bash
# Process a test file
python src/inference.py \
    --checkpoint checkpoints/phase_1A_best/best_model.pt \
    --config configs/phase1a_parametric_only.yaml \
    --input audio/my_song_unmastered.wav \## Dataset
    --output audio/my_song_mastered.wav
```

## Evaluate Model

```bash
# Evaluate on test set
python src/evaluate.py \
    --checkpoint checkpoints/phase_1A_best/best_model.pt \
    --config configs/phase1a_parametric_only.yaml \
    --save_dir evaluation_results

# View results
ls evaluation_results/

## Dataset
**Requirements:**
- WAV format, 44.1kHz
- Stereo 
- Matching pre/post mastering pairs
