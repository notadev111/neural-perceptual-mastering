"""
Demo Script for EQ Profile Analysis
===================================

Quick demonstration of analysis capabilities without requiring audio files.
Generates synthetic test signals and demonstrates all analysis features.

Usage:
    python demo_analysis.py
"""

import torch
import torchaudio
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

def create_test_audio(duration: float = 5.0, sample_rate: int = 44100) -> torch.Tensor:
    """
    Create synthetic test audio with multiple frequency components
    """
    
    t = torch.linspace(0, duration, int(sample_rate * duration))
    
    # Create complex test signal
    # Fundamental + harmonics + noise
    frequencies = [220, 440, 880, 1320, 1760]  # A3 and harmonics
    amplitudes = [0.3, 0.4, 0.2, 0.1, 0.05]   # Decreasing amplitude
    
    audio = torch.zeros(2, len(t))  # Stereo
    
    # Add harmonic content
    for freq, amp in zip(frequencies, amplitudes):
        sine_wave = amp * torch.sin(2 * np.pi * freq * t)
        audio[0] += sine_wave  # Left channel
        audio[1] += sine_wave  # Right channel
    
    # Add some broadband content (filtered noise)
    noise = 0.02 * torch.randn_like(t)
    # Simple lowpass filter (crude)
    noise_filtered = torch.conv1d(noise.unsqueeze(0).unsqueeze(0), 
                                 torch.ones(1, 1, 50) / 50, padding=25)[0, 0]
    
    audio[0] += noise_filtered[:len(t)]
    audio[1] += noise_filtered[:len(t)]
    
    # Add some stereo separation
    audio[1] *= 0.9  # Slight difference
    
    # Normalize to prevent clipping
    peak = torch.max(torch.abs(audio))
    if peak > 0.8:
        audio = audio * (0.8 / peak)
    
    return audio


def demo_frequency_response_analysis():
    """Demonstrate frequency response visualization"""
    
    print("\\n" + "="*60)
    print("DEMO: FREQUENCY RESPONSE ANALYSIS")
    print("="*60)
    
    try:
        from analyze_eq_profiles import EQAnalyzer
        
        # Initialize analyzer
        analyzer = EQAnalyzer(sample_rate=44100)
        
        # Get available profiles
        available = analyzer.get_available_profiles()
        print(f"Available presets: {available['presets']}")
        
        if available['presets']:
            # Test with first few presets
            test_profiles = available['presets'][:3]
            print(f"\\nTesting profiles: {test_profiles}")
            
            # Create plots directory
            Path("./demo_plots").mkdir(exist_ok=True)
            
            # Generate frequency response plot
            analyzer.plot_frequency_responses(
                test_profiles, 
                save_path="./demo_plots/demo_frequency_response.png",
                show_plot=False
            )
            
            print("✓ Frequency response plot saved to: ./demo_plots/demo_frequency_response.png")
        else:
            print("No presets available for testing")
            
    except Exception as e:
        print(f"Error in frequency response demo: {e}")


def demo_audio_testing():
    """Demonstrate audio testing with synthetic signal"""
    
    print("\\n" + "="*60)
    print("DEMO: AUDIO TESTING WITH SYNTHETIC SIGNAL")
    print("="*60)
    
    try:
        from test_eq_profiles import EQProfileTester
        
        # Create synthetic test audio
        print("Creating synthetic test audio...")
        test_audio = create_test_audio(duration=5.0, sample_rate=44100)
        
        # Save test audio
        test_audio_path = Path("./demo_test_audio.wav")
        torchaudio.save(str(test_audio_path), test_audio, 44100)
        print(f"✓ Test audio saved: {test_audio_path}")
        
        # Initialize tester
        tester = EQProfileTester(sample_rate=44100)
        
        if tester.mastering_system and tester.mastering_system.presets:
            presets = list(tester.mastering_system.presets.keys())
            
            # Test first two presets if available
            if len(presets) >= 2:
                print(f"\\nTesting A/B comparison: {presets[0]} vs {presets[1]}")
                
                # Create output directory
                Path("./demo_output").mkdir(exist_ok=True)
                
                # Run A/B test
                results = tester.ab_compare_profiles(
                    str(test_audio_path), 
                    presets[0], 
                    presets[1],
                    output_dir="./demo_output"
                )
                
                if results:
                    print("✓ A/B comparison completed")
                    print("✓ Check ./demo_output/ for processed audio files")
            
            # Batch test first 3 presets
            if len(presets) >= 3:
                print(f"\\nBatch testing: {presets[:3]}")
                
                df = tester.batch_test_profiles(
                    str(test_audio_path),
                    presets[:3],
                    output_dir="./demo_output"
                )
                
                if not df.empty:
                    print("✓ Batch test completed")
                    print("✓ Results saved to ./demo_output/")
        else:
            print("No mastering system or presets available")
            
        # Clean up test file
        if test_audio_path.exists():
            test_audio_path.unlink()
            
    except Exception as e:
        print(f"Error in audio testing demo: {e}")


def demo_spectrum_analysis():
    """Demonstrate spectrum analysis"""
    
    print("\\n" + "="*60)
    print("DEMO: SPECTRUM ANALYSIS")
    print("="*60)
    
    try:
        from analyze_eq_profiles import EQAnalyzer
        
        # Create test audio
        test_audio = create_test_audio(duration=3.0, sample_rate=44100)
        test_audio_path = Path("./demo_spectrum_audio.wav")
        torchaudio.save(str(test_audio_path), test_audio, 44100)
        
        # Initialize analyzer
        analyzer = EQAnalyzer(sample_rate=44100)
        
        # Get available profiles
        available = analyzer.get_available_profiles()
        
        if available['presets']:
            test_profiles = available['presets'][:2]  # First 2 presets
            
            print(f"Analyzing spectrum with profiles: {test_profiles}")
            
            # Create plots directory
            Path("./demo_plots").mkdir(exist_ok=True)
            
            # Run spectrum analysis
            results = analyzer.analyze_audio_spectrum(
                str(test_audio_path),
                test_profiles,
                save_path="./demo_plots/demo_spectrum_analysis.png"
            )
            
            if results:
                print("✓ Spectrum analysis completed")
                print("✓ Plot saved to: ./demo_plots/demo_spectrum_analysis.png")
        else:
            print("No presets available for spectrum analysis")
        
        # Clean up
        if test_audio_path.exists():
            test_audio_path.unlink()
            
    except Exception as e:
        print(f"Error in spectrum analysis demo: {e}")


def main():
    """Run all demo functions"""
    
    print("EQ PROFILE ANALYSIS DEMONSTRATION")
    print("="*60)
    print("This demo shows the analysis capabilities using synthetic test signals.")
    print("No external audio files required!")
    
    # Run demos
    demo_frequency_response_analysis()
    demo_audio_testing() 
    demo_spectrum_analysis()
    
    print("\\n" + "="*60)
    print("DEMO COMPLETE")
    print("="*60)
    print("\\nGenerated files:")
    print("📁 ./demo_plots/ - Visualization plots")
    print("📁 ./demo_output/ - Processed audio files and reports")
    print("\\nNext steps:")
    print("1. Try with your own audio:")
    print("   python test_eq_profiles.py --audio your_mix.wav --compare warm bright")
    print("2. Explore all profiles:")
    print("   python analyze_eq_profiles.py --compare-all --save-plots")
    print("3. Generate detailed reports:")
    print("   python analyze_eq_profiles.py --audio your_mix.wav --profiles warm --report")


if __name__ == '__main__':
    main()