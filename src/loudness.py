import librosa
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pyloudnorm as pyln
from scipy import signal
import pandas as pd

class AudioAnalyzer:
    def __init__(self, sample_rate=44100):
        self.sr = sample_rate
        self.meter = pyln.Meter(sample_rate)
    
    def compute_lufs(self, audio):
        """Compute integrated LUFS loudness"""
        if audio.ndim == 1:
            audio = audio.reshape(-1, 1)
        return self.meter.integrated_loudness(audio)
    
    def compute_peak(self, audio):
        """Compute peak level in dB"""
        return 20 * np.log10(np.abs(audio).max() + 1e-10)
    
    def compute_rms(self, audio):
        """Compute RMS level in dB"""
        return 20 * np.log10(np.sqrt(np.mean(audio**2)) + 1e-10)
    
    def compute_dynamic_range(self, audio, percentile=95):
        """Compute dynamic range (difference between loud and quiet parts)"""
        # Use short-term loudness windows
        frame_length = int(0.4 * self.sr)
        hop_length = int(0.1 * self.sr)
        
        rms_values = []
        for i in range(0, len(audio) - frame_length, hop_length):
            frame = audio[i:i+frame_length]
            rms = np.sqrt(np.mean(frame**2))
            if rms > 0:
                rms_values.append(20 * np.log10(rms))
        
        if not rms_values:
            return 0
        
        return np.percentile(rms_values, percentile) - np.percentile(rms_values, 5)
    
    def compute_stereo_width(self, audio):
        """Compute stereo width (correlation between L/R channels)"""
        if audio.ndim == 1:
            return 0.0  # Mono
        
        left = audio[:, 0]
        right = audio[:, 1]
        
        # Correlation coefficient (closer to 1 = more mono, closer to 0 = wider)
        correlation = np.corrcoef(left, right)[0, 1]
        
        # Convert to width metric (0 = mono, 1 = wide)
        width = 1 - abs(correlation)
        return width
    
    def compute_spectral_centroid(self, audio):
        """Compute average spectral centroid (brightness)"""
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        
        centroid = librosa.feature.spectral_centroid(y=audio, sr=self.sr)
        return np.mean(centroid)
    
    def compute_spectral_rolloff(self, audio, roll_percent=0.85):
        """Compute spectral rolloff (high frequency content)"""
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        
        rolloff = librosa.feature.spectral_rolloff(y=audio, sr=self.sr, roll_percent=roll_percent)
        return np.mean(rolloff)
    
    def compute_crest_factor(self, audio):
        """Compute crest factor (peak to RMS ratio) - indicates compression"""
        peak = np.abs(audio).max()
        rms = np.sqrt(np.mean(audio**2))
        if rms == 0:
            return 0
        return 20 * np.log10(peak / rms)
    
    def compute_high_freq_energy(self, audio, cutoff_freq=8000):
        """Compute energy in high frequencies"""
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        
        # Design high-pass filter
        nyquist = self.sr / 2
        cutoff_norm = cutoff_freq / nyquist
        b, a = signal.butter(4, cutoff_norm, btype='high')
        
        # Filter and compute energy
        filtered = signal.filtfilt(b, a, audio)
        high_energy = np.mean(filtered**2)
        total_energy = np.mean(audio**2)
        
        if total_energy == 0:
            return 0
        
        return 10 * np.log10(high_energy / total_energy + 1e-10)
    
    def analyze_file(self, filepath):
        """Analyze a single audio file and return all metrics"""
        audio, sr = librosa.load(filepath, sr=self.sr, mono=False)
        
        if audio.ndim == 1:
            audio = audio.reshape(1, -1)
        audio = audio.T  # Shape: (samples, channels)
        
        metrics = {
            'lufs': self.compute_lufs(audio),
            'peak_db': self.compute_peak(audio),
            'rms_db': self.compute_rms(audio),
            'dynamic_range': self.compute_dynamic_range(audio.flatten()),
            'stereo_width': self.compute_stereo_width(audio),
            'spectral_centroid': self.compute_spectral_centroid(audio),
            'spectral_rolloff': self.compute_spectral_rolloff(audio),
            'crest_factor': self.compute_crest_factor(audio.flatten()),
            'high_freq_energy': self.compute_high_freq_energy(audio)
        }
        
        return metrics


def analyze_dataset(pre_dir, post_dir, limit=None):
    """Analyze pre/post pairs and return DataFrame"""
    analyzer = AudioAnalyzer()
    
    pre_path = Path(pre_dir)
    post_path = Path(post_dir)
    
    # Find matching pairs
    pre_files = sorted(pre_path.glob('*.wav'))
    
    results = []
    
    for i, pre_file in enumerate(pre_files):
        if limit and i >= limit:
            break
            
        post_file = post_path / pre_file.name
        
        if not post_file.exists():
            print(f"Warning: No matching post file for {pre_file.name}")
            continue
        
        print(f"Analyzing {i+1}/{len(pre_files)}: {pre_file.name}")
        
        try:
            pre_metrics = analyzer.analyze_file(pre_file)
            post_metrics = analyzer.analyze_file(post_file)
            
            # Compute deltas
            result = {
                'filename': pre_file.name,
            }
            
            for key in pre_metrics.keys():
                result[f'pre_{key}'] = pre_metrics[key]
                result[f'post_{key}'] = post_metrics[key]
                result[f'delta_{key}'] = post_metrics[key] - pre_metrics[key]
            
            results.append(result)
            
        except Exception as e:
            print(f"Error analyzing {pre_file.name}: {e}")
            continue
    
    return pd.DataFrame(results)


def plot_analysis(df, output_dir='analysis_plots'):
    """Create comprehensive visualization of the analysis"""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (15, 10)
    
    # Metrics to analyze
    metrics = ['lufs', 'peak_db', 'rms_db', 'dynamic_range', 'stereo_width', 
               'spectral_centroid', 'crest_factor', 'high_freq_energy']
    
    metric_labels = {
        'lufs': 'Loudness (LUFS)',
        'peak_db': 'Peak Level (dB)',
        'rms_db': 'RMS Level (dB)',
        'dynamic_range': 'Dynamic Range (dB)',
        'stereo_width': 'Stereo Width',
        'spectral_centroid': 'Spectral Centroid (Hz)',
        'crest_factor': 'Crest Factor (dB)',
        'high_freq_energy': 'High Freq Energy (dB)'
    }
    
    # 1. Distribution comparison plots
    fig, axes = plt.subplots(4, 2, figsize=(15, 16))
    axes = axes.flatten()
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        
        pre_col = f'pre_{metric}'
        post_col = f'post_{metric}'
        
        # Plot distributions
        ax.hist(df[pre_col], alpha=0.5, bins=30, label='Pre-Master', color='blue')
        ax.hist(df[post_col], alpha=0.5, bins=30, label='Post-Master', color='red')
        
        ax.set_xlabel(metric_labels.get(metric, metric))
        ax.set_ylabel('Count')
        ax.set_title(f'{metric_labels.get(metric, metric)} Distribution')
        ax.legend()
        ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path / 'distributions.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Delta (change) plots
    fig, axes = plt.subplots(4, 2, figsize=(15, 16))
    axes = axes.flatten()
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        delta_col = f'delta_{metric}'
        
        # Violin plot of changes
        parts = ax.violinplot([df[delta_col].values], positions=[0], 
                               showmeans=True, showmedians=True)
        
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.set_ylabel(f'Δ {metric_labels.get(metric, metric)}')
        ax.set_title(f'Change in {metric_labels.get(metric, metric)}\n'
                     f'Mean: {df[delta_col].mean():.2f}, Median: {df[delta_col].median():.2f}')
        ax.set_xticks([])
        ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path / 'deltas.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Scatter plots (pre vs post)
    fig, axes = plt.subplots(4, 2, figsize=(15, 16))
    axes = axes.flatten()
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        pre_col = f'pre_{metric}'
        post_col = f'post_{metric}'
        
        ax.scatter(df[pre_col], df[post_col], alpha=0.5)
        
        # Add diagonal line (y=x)
        lims = [
            np.min([ax.get_xlim(), ax.get_ylim()]),
            np.max([ax.get_xlim(), ax.get_ylim()])
        ]
        ax.plot(lims, lims, 'r--', alpha=0.5, label='No Change')
        
        ax.set_xlabel(f'Pre-Master {metric_labels.get(metric, metric)}')
        ax.set_ylabel(f'Post-Master {metric_labels.get(metric, metric)}')
        ax.set_title(f'{metric_labels.get(metric, metric)}: Pre vs Post')
        ax.legend()
        ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path / 'scatter.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    for metric in metrics:
        pre_col = f'pre_{metric}'
        post_col = f'post_{metric}'
        delta_col = f'delta_{metric}'
        
        print(f"\n{metric_labels.get(metric, metric).upper()}")
        print(f"  Pre-Master:  Mean={df[pre_col].mean():8.2f}  Std={df[pre_col].std():8.2f}")
        print(f"  Post-Master: Mean={df[post_col].mean():8.2f}  Std={df[post_col].std():8.2f}")
        print(f"  Change:      Mean={df[delta_col].mean():8.2f}  Std={df[delta_col].std():8.2f}")
        print(f"  Change:      Min={df[delta_col].min():8.2f}  Max={df[delta_col].max():8.2f}")
    
    # Save summary to CSV
    summary_stats = []
    for metric in metrics:
        pre_col = f'pre_{metric}'
        post_col = f'post_{metric}'
        delta_col = f'delta_{metric}'
        
        summary_stats.append({
            'metric': metric,
            'pre_mean': df[pre_col].mean(),
            'pre_std': df[pre_col].std(),
            'post_mean': df[post_col].mean(),
            'post_std': df[post_col].std(),
            'delta_mean': df[delta_col].mean(),
            'delta_std': df[delta_col].std(),
            'delta_min': df[delta_col].min(),
            'delta_max': df[delta_col].max()
        })
    
    summary_df = pd.DataFrame(summary_stats)
    summary_df.to_csv(output_path / 'summary_statistics.csv', index=False)
    print(f"\nSummary statistics saved to {output_path / 'summary_statistics.csv'}")
    
    return summary_df


# Example usage
if __name__ == "__main__":
    # Analyze your dataset
    df = analyze_dataset(
        pre_dir='path/to/pre_master',
        post_dir='path/to/post_master',
        limit=None  # Set to a number to analyze only first N files
    )
    
    # Save full results
    df.to_csv('full_analysis.csv', index=False)
    print(f"Full analysis saved to full_analysis.csv")
    
    # Create visualizations
    summary = plot_analysis(df)
    
    print("\nAnalysis complete! Check the 'analysis_plots' directory for visualizations.")