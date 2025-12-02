# Weights & Biases Setup Guide

## What You Get with W&B

✅ **Real-time monitoring** from anywhere (laptop, phone, etc.)
✅ **Live loss curves** and training metrics
✅ **Audio samples** logged every 5 epochs (listen to model progress!)
✅ **System metrics** (GPU usage, memory, etc.)
✅ **Model checkpointing** in the cloud
✅ **Experiment comparison** across different runs

---

## Setup (One-Time)

### 1. Create W&B Account (Free)

1. Go to [https://wandb.ai/signup](https://wandb.ai/signup)
2. Sign up (free for academics)
3. Get your API key from [https://wandb.ai/authorize](https://wandb.ai/authorize)

### 2. Install on UCL Server

SSH into the server and run:

```bash
# Activate your virtual environment
source ~/audio-training-venv/bin/activate

# Install wandb
pip install wandb

# Login (one-time setup)
wandb login
```

When prompted, paste your API key from step 1.

---

## Running Training with W&B

### Start Training (Default - W&B Enabled)

```bash
cd ~/neural-perceptual-mastering
python src/train.py --config configs/phase1a.yaml
```

### Disable W&B (if needed)

```bash
python src/train.py --config configs/phase1a.yaml --no-wandb
```

### Custom Project Name

```bash
python src/train.py --config configs/phase1a.yaml --wandb-project my-mastering-project
```

---

## Viewing Your Training

### Option 1: Web Dashboard

Once training starts, you'll see a URL like:
```
Weights & Biases initialized: https://wandb.ai/your-username/neural-mastering/runs/abc123
```

Open that URL in any browser to watch training in real-time!

### Option 2: Mobile App

1. Download the W&B app (iOS/Android)
2. Login with your account
3. View training metrics on the go!

---

## What Gets Logged

### Every 10 Batches
- Batch loss
- Spectral loss
- Perceptual loss
- Loudness loss

### Every Epoch
- Epoch train loss
- Epoch validation loss
- Loss components breakdown
- Learning rate

### Every 5 Epochs
- **Audio samples!**
  - Unmastered input
  - Target mastered
  - Model prediction

### Continuous
- System metrics (GPU, RAM, CPU)
- Gradients & parameters

---

## Features to Explore

### 1. Live Charts
- Interactive loss curves
- Zoom, pan, compare runs

### 2. Audio Player
- Listen to model outputs as it trains
- A/B compare with targets

### 3. System Metrics
- GPU memory usage
- GPU utilization
- Watch for OOM warnings

### 4. Model Checkpointing
- Automatically saves checkpoints to W&B cloud
- Download best models from dashboard

### 5. Hyperparameter Tracking
- All config values logged automatically
- Compare different hyperparameter settings

---

## Tips

- **Keep browser open**: W&B dashboard auto-updates in real-time
- **Check GPU**: Look for GPU metrics to ensure it's being used
- **Listen to audio**: Best way to judge if model is improving
- **Compare runs**: Use W&B's comparison tools to find best config

---

## Troubleshooting

### "wandb: ERROR Error uploading"
- Network issue, training continues normally
- Logs will sync when connection restored

### "wandb: WARNING W&B disabled due to login failure"
- Run `wandb login` again with your API key

### Audio not showing up
- Audio logs every 5 epochs (epoch 0, 5, 10, etc.)
- Check "Media" tab in W&B dashboard

---

## Example Dashboard Screenshot

You'll see charts like:
- Train Loss (decreasing over time ↘️)
- Val Loss (tracking train loss)
- Learning Rate (stepped down when plateau)
- GPU Memory (should be ~500 MB - 1 GB)
- Audio waveforms (visual comparison)

Happy training! 🎵🔥
