# src/evaluate.py
import torch

def evaluate_model(model, test_loader, device):
    model.eval()
    
    metrics = {
        'spectral_distance': [],
        'lufs_error': [],
        'pesq': [],
        'sisdr': []  # Scale-Invariant SDR
    }
    
    with torch.no_grad():
        for unmastered, mastered in test_loader:
            unmastered = unmastered.to(device)
            mastered = mastered.to(device)
            
            pred, _ = model(unmastered)
            
            # Spectral distance
            pred_spec = torch.stft(pred, n_fft=2048, return_complex=True)
            target_spec = torch.stft(mastered, n_fft=2048, return_complex=True)
            spec_dist = torch.mean((torch.abs(pred_spec) - torch.abs(target_spec))**2)
            metrics['spectral_distance'].append(spec_dist.item())
            
            # LUFS error
            pred_np = pred.cpu().numpy()
            target_np = mastered.cpu().numpy()
            meter = pyln.Meter(44100)
            for i in range(pred.shape[0]):
                pred_lufs = meter.integrated_loudness(pred_np[i].T)
                target_lufs = meter.integrated_loudness(target_np[i].T)
                metrics['lufs_error'].append(abs(pred_lufs - target_lufs))
    
    return {k: np.mean(v) for k, v in metrics.items()}