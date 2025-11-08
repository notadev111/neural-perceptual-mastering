"""
EQ Profile Testing and A/B Comparison Tool
==========================================

Interactive testing system for semantic mastering profiles:
- A/B compare different profiles on the same audio
- Batch test multiple profiles with metrics
- Generate comparison audio files
- Interactive parameter exploration

Usage:
    # Quick A/B test
    python test_eq_profiles.py --audio mix.wav --compare warm bright
    
    # Batch test all presets
    python test_eq_profiles.py --audio mix.wav --test-presets --metrics
    
    # Test specific terms from dataset
    python test_eq_profiles.py --audio mix.wav --test-terms aggressive smooth calm
    
    # Interactive exploration
    python test_eq_profiles.py --audio mix.wav --interactive
"""

import torch
import torchaudio
import numpy as np
import pandas as pd
import argparse
from pathlib import Path
import json
import time
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings("ignore")

# Audio analysis imports
try:
    import librosa
    import pyloudnorm as pyln
    from scipy import signal
    AUDIO_LIBS_AVAILABLE = True
except ImportError:
    print("Warning: Some audio analysis features disabled (missing librosa/pyloudnorm)")
    AUDIO_LIBS_AVAILABLE = False

# Import our systems
try:
    from semantic_mastering import SemanticMasteringEQ, EQProfile
    print("Semantic mastering system imported successfully")
except ImportError:
    print("Error: Cannot import semantic_mastering.py")
    exit(1)


class EQProfileTester:
    """
    Comprehensive EQ profile testing and comparison system
    """
    
    def __init__(self, sample_rate: int = 44100, cache_dir: str = "./cache"):
        self.sr = sample_rate
        self.cache_dir = Path(cache_dir)
        
        # Initialize mastering system
        print("Initializing semantic mastering system...")
        self.mastering_system = SemanticMasteringEQ(sample_rate=sample_rate, cache_dir=cache_dir)
        
        try:
            self.mastering_system.initialize()
            print(f"System ready with {len(self.mastering_system.loader.term_profiles)} dataset terms")
            print(f"Available presets: {list(self.mastering_system.presets.keys())}")
        except Exception as e:
            print(f"Warning: Could not fully initialize system: {e}")
            self.mastering_system = None
        
        # Initialize audio analysis tools
        if AUDIO_LIBS_AVAILABLE:
            self.loudness_meter = pyln.Meter(sample_rate)
        else:
            self.loudness_meter = None
    
    def load_and_prepare_audio(self, audio_path: str) -> torch.Tensor:
        """Load and prepare audio for processing"""
        try:
            audio, sr = torchaudio.load(audio_path)
            
            # Resample if needed
            if sr != self.sr:
                print(f"Resampling: {sr} -> {self.sr} Hz")
                audio = torchaudio.functional.resample(audio, sr, self.sr)
            
            # Ensure stereo
            if audio.shape[0] == 1:
                audio = audio.repeat(2, 1)
            elif audio.shape[0] > 2:
                audio = audio[:2]
            
            duration = audio.shape[-1] / self.sr
            print(f"Loaded audio: {audio.shape} ({duration:.1f}s)")
            
            return audio
            
        except Exception as e:
            print(f"Error loading audio: {e}")
            return None
    
    def compute_audio_metrics(self, audio: torch.Tensor) -> Dict[str, float]:
        """Compute comprehensive audio metrics"""
        
        # Convert to numpy for analysis
        if isinstance(audio, torch.Tensor):
            if audio.dim() == 3:  # [batch, channels, samples]
                audio_np = audio[0].numpy()
            else:  # [channels, samples]
                audio_np = audio.numpy()
        else:
            audio_np = audio
        
        # Mono version for some analyses
        mono = np.mean(audio_np, axis=0)
        
        metrics = {}
        
        # Basic level metrics
        metrics['rms_db'] = 20 * np.log10(np.sqrt(np.mean(mono**2)) + 1e-10)
        metrics['peak_db'] = 20 * np.log10(np.max(np.abs(mono)) + 1e-10)
        metrics['crest_factor'] = metrics['peak_db'] - metrics['rms_db']
        
        # Dynamic range
        if AUDIO_LIBS_AVAILABLE:
            try:
                # LUFS loudness
                metrics['lufs'] = self.loudness_meter.integrated_loudness(mono)
                
                # Spectral centroid (brightness measure)
                centroid = librosa.feature.spectral_centroid(y=mono, sr=self.sr)[0]
                metrics['spectral_centroid_hz'] = np.mean(centroid)
                
                # Spectral rolloff (high frequency energy)
                rolloff = librosa.feature.spectral_rolloff(y=mono, sr=self.sr)[0]
                metrics['spectral_rolloff_hz'] = np.mean(rolloff)
                
                # Zero crossing rate (roughness indicator)
                zcr = librosa.feature.zero_crossing_rate(mono)[0]
                metrics['zero_crossing_rate'] = np.mean(zcr)
                
                # RMS energy in frequency bands
                stft = librosa.stft(mono, n_fft=2048)
                magnitude = np.abs(stft)
                freqs = librosa.fft_frequencies(sr=self.sr, n_fft=2048)
                
                # Define frequency bands
                bands = {
                    'sub_bass': (20, 60),
                    'bass': (60, 200),
                    'low_mid': (200, 500),
                    'mid': (500, 2000),
                    'high_mid': (2000, 8000),
                    'treble': (8000, 20000)
                }
                
                for band_name, (f_low, f_high) in bands.items():
                    mask = (freqs >= f_low) & (freqs <= f_high)
                    if np.any(mask):
                        band_energy = np.mean(magnitude[mask, :])
                        metrics[f'{band_name}_energy_db'] = 20 * np.log10(band_energy + 1e-10)
                    else:
                        metrics[f'{band_name}_energy_db'] = -80.0
                        
            except Exception as e:
                print(f"Warning: Advanced metrics failed: {e}")
        
        return metrics
    
    def ab_compare_profiles(self, audio_path: str, profile_a: str, profile_b: str,
                          output_dir: str = "./ab_test_output") -> Dict:
        """
        Generate A/B comparison audio files and analysis
        """
        
        if not self.mastering_system:
            print("Error: Mastering system not available")
            return {}
        
        # Load audio
        audio = self.load_and_prepare_audio(audio_path)
        if audio is None:
            return {}
        
        # Create output directory
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        audio_name = Path(audio_path).stem
        
        print(f"\\nA/B Testing: {profile_a} vs {profile_b}")
        print("="*50)
        
        results = {
            'audio_file': audio_path,
            'profile_a': profile_a,
            'profile_b': profile_b,
            'original': {},
            'processed_a': {},
            'processed_b': {}
        }
        
        # Analyze original
        print("Analyzing original audio...")
        original_metrics = self.compute_audio_metrics(audio)
        results['original'] = original_metrics
        
        # Save original for reference
        original_path = output_dir / f"{audio_name}_original.wav"
        torchaudio.save(str(original_path), audio, self.sr)
        
        # Process with profile A
        print(f"Processing with {profile_a}...")
        try:
            processed_a, profile_obj_a = self.mastering_system.apply_mastering(audio, profile_a)
            metrics_a = self.compute_audio_metrics(processed_a)
            results['processed_a'] = {
                'metrics': metrics_a,
                'profile_info': {
                    'confidence': profile_obj_a.confidence,
                    'examples': profile_obj_a.n_examples,
                    'reasoning': profile_obj_a.reasoning
                }
            }
            
            # Save processed A
            processed_a_path = output_dir / f"{audio_name}_{profile_a}.wav"
            torchaudio.save(str(processed_a_path), processed_a[0], self.sr)
            
        except Exception as e:
            print(f"Error processing with {profile_a}: {e}")
            return results
        
        # Process with profile B
        print(f"Processing with {profile_b}...")
        try:
            processed_b, profile_obj_b = self.mastering_system.apply_mastering(audio, profile_b)
            metrics_b = self.compute_audio_metrics(processed_b)
            results['processed_b'] = {
                'metrics': metrics_b,
                'profile_info': {
                    'confidence': profile_obj_b.confidence,
                    'examples': profile_obj_b.n_examples,
                    'reasoning': profile_obj_b.reasoning
                }
            }
            
            # Save processed B
            processed_b_path = output_dir / f"{audio_name}_{profile_b}.wav"
            torchaudio.save(str(processed_b_path), processed_b[0], self.sr)
            
        except Exception as e:
            print(f"Error processing with {profile_b}: {e}")
            return results
        
        # Generate comparison report
        self._generate_ab_comparison_report(results, output_dir, audio_name)
        
        print(f"\\nA/B test files saved to: {output_dir}")
        print(f"Listen to:")
        print(f"  Original: {original_path.name}")
        print(f"  {profile_a}: {processed_a_path.name}")
        print(f"  {profile_b}: {processed_b_path.name}")
        
        return results
    
    def _generate_ab_comparison_report(self, results: Dict, output_dir: Path, audio_name: str):
        """Generate detailed A/B comparison report"""
        
        report_path = output_dir / f"{audio_name}_ab_report.md"
        
        profile_a = results['profile_a']
        profile_b = results['profile_b']
        
        # Calculate differences
        orig_metrics = results['original']
        metrics_a = results['processed_a']['metrics']
        metrics_b = results['processed_b']['metrics']
        
        report_content = f"""# A/B Comparison Report
        
## Audio: {audio_name}
**Comparison**: {profile_a} vs {profile_b}
**Test Date**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

## Profile Information

### {profile_a}
- **Confidence**: {results['processed_a']['profile_info']['confidence']:.1%}
- **Examples**: {results['processed_a']['profile_info']['examples']}
- **Reasoning**: {results['processed_a']['profile_info']['reasoning']}

### {profile_b}  
- **Confidence**: {results['processed_b']['profile_info']['confidence']:.1%}
- **Examples**: {results['processed_b']['profile_info']['examples']}
- **Reasoning**: {results['processed_b']['profile_info']['reasoning']}

## Metric Comparison

| Metric | Original | {profile_a} | {profile_b} | Δ{profile_a} | Δ{profile_b} |
|--------|----------|-------------|-------------|--------------|--------------|
"""
        
        # Add key metrics to table
        key_metrics = ['rms_db', 'peak_db', 'crest_factor']
        if AUDIO_LIBS_AVAILABLE:
            key_metrics.extend(['lufs', 'spectral_centroid_hz', 'bass_energy_db', 'treble_energy_db'])
        
        for metric in key_metrics:
            if metric in orig_metrics and metric in metrics_a and metric in metrics_b:
                orig_val = orig_metrics[metric]
                val_a = metrics_a[metric]
                val_b = metrics_b[metric]
                
                if 'hz' in metric:
                    # Frequency values
                    report_content += f"| {metric} | {orig_val:.0f} | {val_a:.0f} | {val_b:.0f} | {val_a-orig_val:+.0f} | {val_b-orig_val:+.0f} |\\n"
                else:
                    # dB values
                    report_content += f"| {metric} | {orig_val:.1f} | {val_a:.1f} | {val_b:.1f} | {val_a-orig_val:+.1f} | {val_b-orig_val:+.1f} |\\n"
        
        report_content += f"""
## Analysis Summary

### Loudness Changes
- **{profile_a}**: {'Louder' if metrics_a.get('rms_db', 0) > orig_metrics.get('rms_db', 0) else 'Quieter'} by {abs(metrics_a.get('rms_db', 0) - orig_metrics.get('rms_db', 0)):.1f} dB RMS
- **{profile_b}**: {'Louder' if metrics_b.get('rms_db', 0) > orig_metrics.get('rms_db', 0) else 'Quieter'} by {abs(metrics_b.get('rms_db', 0) - orig_metrics.get('rms_db', 0)):.1f} dB RMS

### Tonal Changes
"""
        
        if AUDIO_LIBS_AVAILABLE:
            # Analyze tonal changes
            orig_centroid = orig_metrics.get('spectral_centroid_hz', 1000)
            centroid_a = metrics_a.get('spectral_centroid_hz', 1000)
            centroid_b = metrics_b.get('spectral_centroid_hz', 1000)
            
            brightness_a = "Brighter" if centroid_a > orig_centroid * 1.05 else "Darker" if centroid_a < orig_centroid * 0.95 else "Similar"
            brightness_b = "Brighter" if centroid_b > orig_centroid * 1.05 else "Darker" if centroid_b < orig_centroid * 0.95 else "Similar"
            
            report_content += f"- **{profile_a}**: {brightness_a} (centroid: {centroid_a:.0f} Hz)\\n"
            report_content += f"- **{profile_b}**: {brightness_b} (centroid: {centroid_b:.0f} Hz)\\n"
        
        report_content += """
## Listening Test Instructions

1. **Load all three files in your DAW**:
   - `{audio_name}_original.wav`
   - `{audio_name}_{profile_a}.wav`  
   - `{audio_name}_{profile_b}.wav`

2. **Level match** for fair comparison (use gain to match RMS levels)

3. **A/B switch** between versions while listening to:
   - Bass clarity and weight
   - Midrange presence and clarity
   - High frequency air and detail
   - Overall balance and musicality

4. **Consider the context**:
   - What genre/style is this?
   - What's the intended listening environment?
   - How does it compare to reference tracks?

## Recommendations

Choose **{profile_a}** if you want: [Based on profile characteristics]
Choose **{profile_b}** if you want: [Based on profile characteristics]
""".format(audio_name=audio_name, profile_a=profile_a, profile_b=profile_b)
        
        # Save report
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        print(f"Detailed report saved: {report_path.name}")
    
    def batch_test_profiles(self, audio_path: str, profiles: List[str],
                           output_dir: str = "./batch_test_output") -> pd.DataFrame:
        """
        Batch test multiple profiles and generate comparison table
        """
        
        if not self.mastering_system:
            print("Error: Mastering system not available")
            return pd.DataFrame()
        
        # Load audio
        audio = self.load_and_prepare_audio(audio_path)
        if audio is None:
            return pd.DataFrame()
        
        # Create output directory
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        audio_name = Path(audio_path).stem
        
        print(f"\\nBatch testing {len(profiles)} profiles on {audio_name}")
        print("="*60)
        
        # Analyze original
        print("Analyzing original...")
        original_metrics = self.compute_audio_metrics(audio)
        
        # Save original
        original_path = output_dir / f"{audio_name}_original.wav"
        torchaudio.save(str(original_path), audio, self.sr)
        
        # Test each profile
        results = []
        
        for i, profile_name in enumerate(profiles):
            print(f"Testing {profile_name} ({i+1}/{len(profiles)})...")
            
            try:
                # Process with profile
                processed, profile_obj = self.mastering_system.apply_mastering(audio, profile_name)
                
                # Compute metrics
                processed_metrics = self.compute_audio_metrics(processed)
                
                # Save processed audio
                processed_path = output_dir / f"{audio_name}_{profile_name}.wav"
                torchaudio.save(str(processed_path), processed[0], self.sr)
                
                # Compile results
                result = {
                    'Profile': profile_name,
                    'Confidence': f"{profile_obj.confidence:.1%}",
                    'Examples': profile_obj.n_examples,
                    'File': processed_path.name
                }
                
                # Add metric changes
                for metric_name, orig_value in original_metrics.items():
                    if metric_name in processed_metrics:
                        new_value = processed_metrics[metric_name]
                        change = new_value - orig_value
                        
                        if 'hz' in metric_name:
                            result[f'Δ{metric_name}'] = f"{change:+.0f}"
                        else:
                            result[f'Δ{metric_name}'] = f"{change:+.1f}"
                
                results.append(result)
                
            except Exception as e:
                print(f"  Error: {e}")
                results.append({
                    'Profile': profile_name,
                    'Error': str(e)
                })
        
        # Create results DataFrame
        df = pd.DataFrame(results)
        
        # Save results table
        csv_path = output_dir / f"{audio_name}_batch_results.csv"
        df.to_csv(csv_path, index=False)
        
        # Display results
        print(f"\\nBatch Test Results:")
        print("="*60)
        
        # Show key columns
        display_columns = ['Profile', 'Confidence', 'Examples']
        if 'Δrms_db' in df.columns:
            display_columns.append('Δrms_db')
        if 'Δlufs' in df.columns:
            display_columns.append('Δlufs')
        if 'Δspectral_centroid_hz' in df.columns:
            display_columns.append('Δspectral_centroid_hz')
        
        display_df = df[display_columns] if all(col in df.columns for col in display_columns) else df
        print(display_df.to_string(index=False))
        
        print(f"\\nFiles saved to: {output_dir}")
        print(f"Results table: {csv_path.name}")
        
        return df
    
    def interactive_exploration(self, audio_path: str):
        """
        Interactive profile exploration mode
        """
        
        if not self.mastering_system:
            print("Error: Mastering system not available")
            return
        
        print("\\n" + "="*60)
        print("INTERACTIVE EQ PROFILE EXPLORATION")
        print("="*60)
        
        # Show available profiles
        presets = list(self.mastering_system.presets.keys())
        dataset_terms = list(self.mastering_system.loader.term_profiles.keys())
        
        print(f"\\nAvailable presets: {presets}")
        print(f"Dataset terms (first 10): {dataset_terms[:10]}")
        print("\\nCommands:")
        print("  test <profile>     - Test single profile")
        print("  compare <a> <b>    - A/B compare two profiles")
        print("  batch <p1,p2,...>  - Batch test multiple profiles")
        print("  list               - Show available profiles")
        print("  quit               - Exit")
        
        while True:
            try:
                command = input("\\n> ").strip()
                
                if command == "quit":
                    break
                elif command == "list":
                    print(f"Presets: {presets}")
                    print(f"Dataset terms: {dataset_terms[:20]}")
                elif command.startswith("test "):
                    profile = command[5:].strip()
                    self.batch_test_profiles(audio_path, [profile])
                elif command.startswith("compare "):
                    parts = command[8:].split()
                    if len(parts) >= 2:
                        self.ab_compare_profiles(audio_path, parts[0], parts[1])
                    else:
                        print("Usage: compare <profile_a> <profile_b>")
                elif command.startswith("batch "):
                    profiles_str = command[6:].strip()
                    profiles = [p.strip() for p in profiles_str.split(',')]
                    self.batch_test_profiles(audio_path, profiles)
                else:
                    print("Unknown command. Type 'quit' to exit.")
                    
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: {e}")
        
        print("\\nExiting interactive mode.")


def main():
    parser = argparse.ArgumentParser(description="EQ Profile Testing and A/B Comparison")
    parser.add_argument('--audio', required=True, help='Audio file to test')
    parser.add_argument('--compare', nargs=2, metavar=('A', 'B'), 
                       help='A/B compare two profiles')
    parser.add_argument('--test-presets', action='store_true',
                       help='Test all available presets')
    parser.add_argument('--test-terms', nargs='+',
                       help='Test specific dataset terms')
    parser.add_argument('--batch', nargs='+',
                       help='Batch test specific profiles')
    parser.add_argument('--interactive', action='store_true',
                       help='Interactive exploration mode')
    parser.add_argument('--metrics', action='store_true',
                       help='Show detailed metrics')
    parser.add_argument('--output-dir', default='./test_output',
                       help='Output directory for test files')
    
    args = parser.parse_args()
    
    # Check audio file exists
    if not Path(args.audio).exists():
        print(f"Error: Audio file not found: {args.audio}")
        return
    
    # Initialize tester
    try:
        tester = EQProfileTester(sample_rate=44100)
    except Exception as e:
        print(f"Failed to initialize tester: {e}")
        return
    
    # Execute commands
    if args.compare:
        tester.ab_compare_profiles(args.audio, args.compare[0], args.compare[1], args.output_dir)
    
    elif args.test_presets:
        if tester.mastering_system:
            presets = list(tester.mastering_system.presets.keys())
            tester.batch_test_profiles(args.audio, presets, args.output_dir)
        else:
            print("No presets available")
    
    elif args.test_terms:
        tester.batch_test_profiles(args.audio, args.test_terms, args.output_dir)
    
    elif args.batch:
        tester.batch_test_profiles(args.audio, args.batch, args.output_dir)
    
    elif args.interactive:
        tester.interactive_exploration(args.audio)
    
    else:
        print(__doc__)
        print("\\nQuick examples:")
        print(f"python test_eq_profiles.py --audio {args.audio} --compare warm bright")
        print(f"python test_eq_profiles.py --audio {args.audio} --test-presets")
        print(f"python test_eq_profiles.py --audio {args.audio} --interactive")


if __name__ == '__main__':
    main()