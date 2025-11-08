"""
EQ Profile Analysis and Visualization Tool
==========================================

Comprehensive analysis system for semantic mastering EQ profiles:
- Visualize frequency response curves
- Compare multiple profiles side-by-side
- Analyze audio spectrum before/after processing
- Generate detailed reports with metrics

Usage:
    # Visualize specific profiles
    python analyze_eq_profiles.py --profiles warm bright punchy --plot-response
    
    # Test profiles on audio with full analysis
    python analyze_eq_profiles.py --audio mix.wav --profiles warm clean --analyze-all
    
    # Compare all available profiles
    python analyze_eq_profiles.py --compare-all --save-plots
    
    # Generate detailed report
    python analyze_eq_profiles.py --audio mix.wav --profiles warm --report
"""

import torch
import torchaudio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
import pandas as pd
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings("ignore")

# Audio analysis
try:
    import librosa
    import pyloudnorm as pyln
    AUDIO_LIBS_AVAILABLE = True
except ImportError:
    print("Warning: librosa and/or pyloudnorm not available. Some features disabled.")
    AUDIO_LIBS_AVAILABLE = False

# Import our semantic mastering system
try:
    from semantic_mastering import SemanticMasteringEQ, EQProfile
    print("Successfully imported semantic mastering system")
except ImportError:
    print("Error: Cannot import semantic_mastering.py")
    exit(1)


class EQAnalyzer:
    """
    Comprehensive EQ profile analysis and visualization
    """
    
    def __init__(self, sample_rate: int = 44100, cache_dir: str = "./cache"):
        self.sr = sample_rate
        self.cache_dir = Path(cache_dir)
        
        # Initialize semantic mastering system
        print("Initializing semantic mastering system...")
        self.mastering_system = SemanticMasteringEQ(sample_rate=sample_rate, cache_dir=cache_dir)
        
        try:
            self.mastering_system.initialize()
            print(f"System ready with {len(self.mastering_system.loader.term_profiles)} terms")
        except Exception as e:
            print(f"Warning: Could not fully initialize system: {e}")
            self.mastering_system = None
        
        # Initialize loudness meter if available
        if AUDIO_LIBS_AVAILABLE:
            self.loudness_meter = pyln.Meter(sample_rate)
        else:
            self.loudness_meter = None
        
        # Setup matplotlib for high-quality plots
        plt.style.use('default')
        plt.rcParams.update({
            'figure.figsize': (12, 8),
            'font.size': 10,
            'axes.grid': True,
            'grid.alpha': 0.3,
            'lines.linewidth': 2,
            'figure.dpi': 100
        })
    
    def get_available_profiles(self) -> Dict[str, List[str]]:
        """Get all available profiles organized by type"""
        if not self.mastering_system:
            return {"presets": [], "dataset_terms": []}
        
        presets = list(self.mastering_system.presets.keys())
        dataset_terms = list(self.mastering_system.loader.term_profiles.keys())
        
        return {
            "presets": presets,
            "dataset_terms": dataset_terms[:20]  # Limit to first 20 for manageable visualization
        }
    
    def compute_frequency_response(self, eq_params: torch.Tensor, 
                                 freq_range: Tuple[float, float] = (20, 20000),
                                 n_points: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute frequency response from EQ parameters
        
        Args:
            eq_params: [1, 18] tensor with normalized EQ parameters
            freq_range: (min_freq, max_freq) in Hz
            n_points: Number of frequency points to compute
            
        Returns:
            (frequencies, magnitude_db)
        """
        
        # Generate frequency points (log-spaced)
        freqs = np.logspace(np.log10(freq_range[0]), np.log10(freq_range[1]), n_points)
        
        # Convert normalized parameters back to physical units
        # eq_params shape: [1, 18] -> 6 bands × 3 params (gain, freq, Q)
        
        magnitude_db = np.zeros_like(freqs)
        
        for band in range(6):
            # Extract parameters for this band
            gain_norm = eq_params[0, band*3].item()
            freq_norm = eq_params[0, band*3 + 1].item()
            q_norm = eq_params[0, band*3 + 2].item()
            
            # Convert normalized to physical units
            # Gain: 0.5 = 0dB, range ±12dB
            gain_db = (gain_norm - 0.5) * 24.0
            
            # Frequency: log scale 20-20000 Hz
            center_freq = 20 * (20000/20) ** freq_norm
            
            # Q: 0.1 - 10 range
            q_factor = 0.1 * (10/0.1) ** q_norm
            
            # Skip if gain is negligible
            if abs(gain_db) < 0.01:
                continue
            
            # Compute bell filter response
            # H(f) = 1 + G * (1 / (1 + Q^2 * ((f/fc) - (fc/f))^2))
            # where G is linear gain
            
            omega = freqs / center_freq
            omega_inv = center_freq / freqs
            
            # Bell filter magnitude response
            h_squared = 1 + (10**(gain_db/20) - 1) / (1 + q_factor**2 * (omega - omega_inv)**2)
            h_squared = np.maximum(h_squared, 1e-10)  # Avoid log(0)
            
            magnitude_db += 10 * np.log10(h_squared)
        
        return freqs, magnitude_db
    
    def plot_frequency_responses(self, profiles: List[str], 
                               save_path: Optional[str] = None,
                               show_plot: bool = True) -> None:
        """
        Plot frequency response curves for multiple profiles
        """
        
        if not self.mastering_system:
            print("Error: Mastering system not available")
            return
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        
        colors = plt.cm.Set1(np.linspace(0, 1, len(profiles)))
        
        # Data for summary table
        profile_data = []
        
        for i, profile_name in enumerate(profiles):
            try:
                # Get profile
                if profile_name in self.mastering_system.presets:
                    profile = self.mastering_system.presets[profile_name]
                else:
                    profile = self.mastering_system.loader.get_profile(profile_name)
                
                # Compute frequency response
                freqs, magnitude = self.compute_frequency_response(profile.params_dasp)
                
                # Plot frequency response
                ax1.semilogx(freqs, magnitude, color=colors[i], 
                           label=f"{profile_name} (conf: {profile.confidence:.1%})",
                           linewidth=2.5)
                
                # Analyze key frequency bands
                bands = {
                    'Sub (20-60Hz)': (20, 60),
                    'Bass (60-200Hz)': (60, 200), 
                    'Low-mid (200-500Hz)': (200, 500),
                    'Mid (500-2kHz)': (500, 2000),
                    'High-mid (2-8kHz)': (2000, 8000),
                    'Treble (8-20kHz)': (8000, 20000)
                }
                
                band_gains = {}
                for band_name, (f_low, f_high) in bands.items():
                    mask = (freqs >= f_low) & (freqs <= f_high)
                    if np.any(mask):
                        avg_gain = np.mean(magnitude[mask])
                        band_gains[band_name] = avg_gain
                    else:
                        band_gains[band_name] = 0.0
                
                # Store profile data
                profile_info = {
                    'Profile': profile_name,
                    'Examples': profile.n_examples,
                    'Confidence': f"{profile.confidence:.1%}",
                    'Max Boost': f"{np.max(magnitude):.1f} dB",
                    'Max Cut': f"{np.min(magnitude):.1f} dB"
                }
                profile_info.update({k: f"{v:.1f}" for k, v in band_gains.items()})
                profile_data.append(profile_info)
                
            except Exception as e:
                print(f"Error processing profile '{profile_name}': {e}")
        
        # Format main plot
        ax1.set_title('EQ Profile Frequency Responses', fontsize=16, fontweight='bold')
        ax1.set_xlabel('Frequency (Hz)')
        ax1.set_ylabel('Magnitude (dB)')
        ax1.grid(True, alpha=0.3)
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.set_xlim(20, 20000)
        ax1.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        # Add frequency band annotations
        band_colors = ['lightblue', 'lightgreen', 'lightyellow', 'lightcoral', 'lightpink', 'lavender']
        band_ranges = [(20, 60), (60, 200), (200, 500), (500, 2000), (2000, 8000), (8000, 20000)]
        band_names = ['Sub', 'Bass', 'Low-mid', 'Mid', 'High-mid', 'Treble']
        
        y_min, y_max = ax1.get_ylim()
        for i, ((f_low, f_high), name, color) in enumerate(zip(band_ranges, band_names, band_colors)):
            ax1.axvspan(f_low, f_high, alpha=0.1, color=color)
            ax1.text((f_low * f_high)**0.5, y_max * 0.9, name, 
                    ha='center', va='center', fontsize=9, 
                    bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.7))
        
        # Create summary table
        if profile_data:
            df = pd.DataFrame(profile_data)
            
            # Hide axes for table
            ax2.axis('tight')
            ax2.axis('off')
            
            # Create table
            table = ax2.table(cellText=df.values, colLabels=df.columns,
                            cellLoc='center', loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(8)
            table.scale(1.2, 1.5)
            
            # Style table
            for i in range(len(df.columns)):
                table[(0, i)].set_facecolor('#4CAF50')
                table[(0, i)].set_text_props(weight='bold', color='white')
            
            ax2.set_title('Profile Analysis Summary', fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {save_path}")
        
        if show_plot:
            plt.show()
        else:
            plt.close()
    
    def analyze_audio_spectrum(self, audio_path: str, profiles: List[str],
                             save_path: Optional[str] = None) -> Dict:
        """
        Analyze audio spectrum before and after EQ processing
        """
        
        if not AUDIO_LIBS_AVAILABLE:
            print("Error: librosa required for audio spectrum analysis")
            return {}
        
        if not self.mastering_system:
            print("Error: Mastering system not available")
            return {}
        
        # Load audio
        try:
            audio, sr = torchaudio.load(audio_path)
            if sr != self.sr:
                audio = torchaudio.functional.resample(audio, sr, self.sr)
            
            if audio.shape[0] == 1:
                audio = audio.repeat(2, 1)
            elif audio.shape[0] > 2:
                audio = audio[:2]
                
            print(f"Loaded audio: {audio.shape} @ {self.sr} Hz")
            
        except Exception as e:
            print(f"Error loading audio: {e}")
            return {}
        
        # Convert to mono for analysis
        audio_mono = torch.mean(audio, dim=0).numpy()
        
        # Compute original spectrum
        n_fft = 2048
        freqs_orig = librosa.fft_frequencies(sr=self.sr, n_fft=n_fft)
        stft_orig = librosa.stft(audio_mono, n_fft=n_fft, hop_length=n_fft//4)
        magnitude_orig = np.mean(np.abs(stft_orig), axis=1)
        magnitude_orig_db = 20 * np.log10(magnitude_orig + 1e-10)
        
        # Set up plot
        n_profiles = len(profiles)
        fig = plt.figure(figsize=(16, 4 + 3*n_profiles))
        gs = GridSpec(n_profiles + 1, 2, height_ratios=[2] + [1]*n_profiles, hspace=0.3)
        
        # Main spectrum plot
        ax_main = fig.add_subplot(gs[0, :])
        
        # Plot original spectrum
        ax_main.semilogx(freqs_orig, magnitude_orig_db, 'k-', linewidth=2, 
                        label='Original', alpha=0.7)
        
        colors = plt.cm.Set1(np.linspace(0, 1, len(profiles)))
        results = {'original': {'spectrum': magnitude_orig_db, 'freqs': freqs_orig}}
        
        for i, profile_name in enumerate(profiles):
            try:
                # Process audio with this profile
                processed_audio, profile = self.mastering_system.apply_mastering(audio, profile_name)
                processed_mono = torch.mean(processed_audio[0], dim=0).numpy()
                
                # Compute processed spectrum
                stft_proc = librosa.stft(processed_mono, n_fft=n_fft, hop_length=n_fft//4)
                magnitude_proc = np.mean(np.abs(stft_proc), axis=1)
                magnitude_proc_db = 20 * np.log10(magnitude_proc + 1e-10)
                
                # Plot processed spectrum
                ax_main.semilogx(freqs_orig, magnitude_proc_db, color=colors[i], 
                               linewidth=2, label=f"{profile_name}")
                
                # Store results
                results[profile_name] = {
                    'spectrum': magnitude_proc_db,
                    'freqs': freqs_orig,
                    'profile': profile
                }
                
                # Plot difference in subplot
                ax_diff = fig.add_subplot(gs[i+1, 0])
                difference = magnitude_proc_db - magnitude_orig_db
                ax_diff.semilogx(freqs_orig, difference, color=colors[i], linewidth=2)
                ax_diff.axhline(y=0, color='black', linestyle='--', alpha=0.5)
                ax_diff.set_title(f'{profile_name} Difference', fontsize=10)
                ax_diff.set_ylabel('Δ dB')
                ax_diff.grid(True, alpha=0.3)
                ax_diff.set_xlim(20, 20000)
                
                # Audio metrics subplot
                ax_metrics = fig.add_subplot(gs[i+1, 1])
                
                # Compute audio metrics
                rms_orig = np.sqrt(np.mean(audio_mono**2))
                rms_proc = np.sqrt(np.mean(processed_mono**2))
                peak_orig = np.max(np.abs(audio_mono))
                peak_proc = np.max(np.abs(processed_mono))
                
                if self.loudness_meter:
                    lufs_orig = self.loudness_meter.integrated_loudness(audio_mono)
                    lufs_proc = self.loudness_meter.integrated_loudness(processed_mono)
                else:
                    lufs_orig = lufs_proc = 0
                
                # Display metrics
                metrics_text = f"""
RMS: {20*np.log10(rms_proc/rms_orig):.1f} dB
Peak: {20*np.log10(peak_proc/peak_orig):.1f} dB
LUFS: {lufs_proc - lufs_orig:.1f} dB
Confidence: {profile.confidence:.1%}
Examples: {profile.n_examples}
                """.strip()
                
                ax_metrics.text(0.05, 0.95, metrics_text, transform=ax_metrics.transAxes,
                              fontsize=9, verticalalignment='top', fontfamily='monospace',
                              bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgray', alpha=0.8))
                ax_metrics.axis('off')
                
            except Exception as e:
                print(f"Error processing profile '{profile_name}': {e}")
        
        # Format main plot
        ax_main.set_title(f'Audio Spectrum Analysis: {Path(audio_path).name}', 
                         fontsize=16, fontweight='bold')
        ax_main.set_xlabel('Frequency (Hz)')
        ax_main.set_ylabel('Magnitude (dB)')
        ax_main.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax_main.grid(True, alpha=0.3)
        ax_main.set_xlim(20, 20000)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Spectrum analysis saved to: {save_path}")
        else:
            plt.show()
        
        return results
    
    def generate_detailed_report(self, audio_path: str, profiles: List[str],
                               output_dir: str = "./analysis_reports") -> str:
        """
        Generate comprehensive analysis report
        """
        
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        audio_name = Path(audio_path).stem
        report_path = output_dir / f"{audio_name}_analysis_report.md"
        
        # Generate plots
        plots_dir = output_dir / "plots"
        plots_dir.mkdir(exist_ok=True)
        
        freq_plot_path = plots_dir / f"{audio_name}_frequency_response.png"
        spectrum_plot_path = plots_dir / f"{audio_name}_spectrum_analysis.png"
        
        # Create visualizations
        self.plot_frequency_responses(profiles, save_path=freq_plot_path, show_plot=False)
        spectrum_results = self.analyze_audio_spectrum(audio_path, profiles, save_path=spectrum_plot_path)
        
        # Generate markdown report
        report_content = f"""# EQ Profile Analysis Report
        
## Audio File: {Path(audio_path).name}
**Analysis Date**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
**Sample Rate**: {self.sr} Hz

## Profiles Analyzed
{', '.join(profiles)}

## Frequency Response Analysis
![Frequency Response](plots/{freq_plot_path.name})

## Spectrum Analysis  
![Spectrum Analysis](plots/{spectrum_plot_path.name})

## Profile Details
"""
        
        for profile_name in profiles:
            if not self.mastering_system:
                continue
                
            try:
                if profile_name in self.mastering_system.presets:
                    profile = self.mastering_system.presets[profile_name]
                else:
                    profile = self.mastering_system.loader.get_profile(profile_name)
                
                report_content += f"""
### {profile_name.upper()}
- **Source**: {profile.n_examples} examples from SocialFX dataset
- **Confidence**: {profile.confidence:.1%}
- **Reasoning**: {profile.reasoning}

"""
            except Exception as e:
                report_content += f"\\n### {profile_name.upper()}\\nError: {e}\\n\\n"
        
        # Add technical analysis
        if spectrum_results:
            report_content += """
## Technical Analysis

| Profile | RMS Change | Peak Change | Spectral Centroid | Notes |
|---------|------------|-------------|-------------------|--------|
"""
            
            orig_spectrum = spectrum_results.get('original', {}).get('spectrum', [])
            
            for profile_name in profiles:
                if profile_name in spectrum_results:
                    proc_spectrum = spectrum_results[profile_name]['spectrum']
                    
                    # Compute spectral centroid
                    freqs = spectrum_results[profile_name]['freqs']
                    orig_centroid = np.sum(freqs * np.exp(orig_spectrum/20)) / np.sum(np.exp(orig_spectrum/20))
                    proc_centroid = np.sum(freqs * np.exp(proc_spectrum/20)) / np.sum(np.exp(proc_spectrum/20))
                    
                    centroid_change = proc_centroid / orig_centroid
                    
                    profile_obj = spectrum_results[profile_name].get('profile')
                    rms_change = "N/A"
                    peak_change = "N/A"
                    
                    if centroid_change > 1.1:
                        note = "Brighter"
                    elif centroid_change < 0.9:
                        note = "Warmer"
                    else:
                        note = "Balanced"
                    
                    report_content += f"| {profile_name} | {rms_change} | {peak_change} | {centroid_change:.2f}x | {note} |\\n"
        
        # Save report
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        print(f"Detailed report saved to: {report_path}")
        return str(report_path)
    
    def compare_all_profiles(self, save_plots: bool = True) -> None:
        """
        Compare all available profiles in a comprehensive visualization
        """
        
        available = self.get_available_profiles()
        all_profiles = available['presets'] + available['dataset_terms'][:10]  # Limit for readability
        
        if not all_profiles:
            print("No profiles available for comparison")
            return
        
        print(f"Comparing {len(all_profiles)} profiles...")
        
        save_path = "./plots/all_profiles_comparison.png" if save_plots else None
        if save_plots:
            Path("./plots").mkdir(exist_ok=True)
        
        self.plot_frequency_responses(all_profiles, save_path=save_path, show_plot=not save_plots)


def main():
    parser = argparse.ArgumentParser(description="EQ Profile Analysis and Visualization")
    parser.add_argument('--profiles', nargs='+', default=['warm', 'bright'], 
                       help='Profiles to analyze')
    parser.add_argument('--audio', help='Audio file to analyze')
    parser.add_argument('--plot-response', action='store_true', 
                       help='Plot frequency response curves')
    parser.add_argument('--analyze-spectrum', action='store_true',
                       help='Analyze audio spectrum before/after')
    parser.add_argument('--analyze-all', action='store_true',
                       help='Perform complete analysis')
    parser.add_argument('--compare-all', action='store_true',
                       help='Compare all available profiles')
    parser.add_argument('--report', action='store_true',
                       help='Generate detailed report')
    parser.add_argument('--save-plots', action='store_true',
                       help='Save plots instead of showing')
    parser.add_argument('--list-profiles', action='store_true',
                       help='List all available profiles')
    
    args = parser.parse_args()
    
    # Initialize analyzer
    try:
        analyzer = EQAnalyzer(sample_rate=44100)
    except Exception as e:
        print(f"Failed to initialize analyzer: {e}")
        return
    
    # List available profiles
    if args.list_profiles:
        available = analyzer.get_available_profiles()
        print("\\nAvailable Profiles:")
        print(f"Presets: {available['presets']}")
        print(f"Dataset terms (sample): {available['dataset_terms']}")
        return
    
    # Compare all profiles
    if args.compare_all:
        analyzer.compare_all_profiles(save_plots=args.save_plots)
        return
    
    # Plot frequency responses
    if args.plot_response or args.analyze_all:
        save_path = "./plots/frequency_response.png" if args.save_plots else None
        if args.save_plots:
            Path("./plots").mkdir(exist_ok=True)
        analyzer.plot_frequency_responses(args.profiles, save_path=save_path, 
                                        show_plot=not args.save_plots)
    
    # Analyze audio spectrum
    if (args.analyze_spectrum or args.analyze_all) and args.audio:
        if not Path(args.audio).exists():
            print(f"Audio file not found: {args.audio}")
            return
        
        save_path = "./plots/spectrum_analysis.png" if args.save_plots else None
        if args.save_plots:
            Path("./plots").mkdir(exist_ok=True)
        analyzer.analyze_audio_spectrum(args.audio, args.profiles, save_path=save_path)
    
    # Generate detailed report
    if args.report and args.audio:
        if not Path(args.audio).exists():
            print(f"Audio file not found: {args.audio}")
            return
        
        analyzer.generate_detailed_report(args.audio, args.profiles)
    
    # Default action if no specific command
    if not any([args.plot_response, args.analyze_spectrum, args.analyze_all, 
               args.compare_all, args.report, args.list_profiles]):
        print(__doc__)
        print("\\nQuick examples:")
        print("python analyze_eq_profiles.py --list-profiles")
        print("python analyze_eq_profiles.py --profiles warm bright --plot-response")
        if args.audio:
            print(f"python analyze_eq_profiles.py --audio {args.audio} --profiles warm --analyze-all")


if __name__ == '__main__':
    main()