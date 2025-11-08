#!/usr/bin/env python3
"""
Test script for Semantic Mastering System
==========================================

This script helps you test and verify the semantic mastering implementation
step by step without needing audio files initially.

Usage:
    python test_semantic_mastering.py --step 1  # Test dataset loading
    python test_semantic_mastering.py --step 2  # Test EQ parameter processing  
    python test_semantic_mastering.py --step 3  # Test audio processing (needs audio file)
    python test_semantic_mastering.py --all     # Run all tests
"""

import torch
import torchaudio
import numpy as np
import argparse
from pathlib import Path
import sys

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

try:
    from semantic_mastering import SocialFXDataLoader, SemanticMasteringEQ
    print("✅ Successfully imported semantic_mastering module")
except Exception as e:
    print(f"❌ Failed to import semantic_mastering: {e}")
    print("\nMake sure you have installed the requirements:")
    print("pip install -r requirements.txt")
    sys.exit(1)


def test_dataset_loading():
    """Test Step 1: Dataset Loading"""
    print(f"\n{'='*70}")
    print("TEST 1: DATASET LOADING")
    print(f"{'='*70}")
    
    try:
        # Initialize loader
        loader = SocialFXDataLoader(cache_dir="./test_cache")
        print("✅ Created SocialFXDataLoader")
        
        # Try to load dataset
        print("\n🔄 Attempting to load SocialFX dataset...")
        print("💡 If this fails, you may need to login to HuggingFace:")
        print("   huggingface-cli login")
        
        loader.load_dataset()
        
        if loader.df_eq is not None:
            print(f"✅ Dataset loaded successfully!")
            print(f"   Shape: {loader.df_eq.shape}")
            print(f"   Columns: {list(loader.df_eq.columns)}")
            
            # Show sample
            print(f"\n📊 Sample data:")
            print(loader.df_eq.head())
            
            return True
        else:
            print("❌ Dataset is None")
            return False
            
    except Exception as e:
        print(f"❌ Dataset loading failed: {e}")
        print("\n🔧 Troubleshooting tips:")
        print("1. Check internet connection")
        print("2. Login to HuggingFace: huggingface-cli login")
        print("3. Verify dataset access permissions")
        return False


def test_parameter_processing():
    """Test Step 2: Parameter Processing"""
    print(f"\n{'='*70}")
    print("TEST 2: PARAMETER PROCESSING")
    print(f"{'='*70}")
    
    try:
        # Initialize loader
        loader = SocialFXDataLoader(cache_dir="./test_cache")
        
        # Load dataset (use cache if available)
        print("🔄 Loading dataset...")
        loader.load_dataset()
        
        if loader.df_eq is None:
            print("❌ No dataset available. Run test 1 first.")
            return False
        
        # Inspect dataset structure
        print("\n🔍 Inspecting dataset structure...")
        loader.inspect_dataset()
        
        # Process parameters
        print("\n🔄 Processing EQ parameters...")
        loader.process_parameters()
        
        if loader.term_profiles:
            print(f"\n✅ Successfully created {len(loader.term_profiles)} profiles!")
            
            # Show profile details
            for term, profile in loader.term_profiles.items():
                print(f"\n--- {term} ---")
                print(f"Examples: {profile.n_examples}")
                print(f"Confidence: {profile.confidence:.2f}")
                print(f"EQ shape: {profile.params_dasp.shape}")
                print(f"Reasoning: {profile.reasoning[:100]}...")
            
            return True
        else:
            print("❌ No profiles created")
            return False
            
    except Exception as e:
        print(f"❌ Parameter processing failed: {e}")
        return False


def test_audio_processing():
    """Test Step 3: Audio Processing"""
    print(f"\n{'='*70}")
    print("TEST 3: AUDIO PROCESSING")
    print(f"{'='*70}")
    
    try:
        # Initialize full system
        print("🔄 Initializing semantic mastering system...")
        system = SemanticMasteringEQ(sample_rate=44100, cache_dir="./test_cache")
        system.initialize()
        
        # Create test audio signal (sine wave)
        print("\n🎵 Creating test audio signal...")
        duration = 2.0  # 2 seconds
        sample_rate = 44100
        t = torch.linspace(0, duration, int(sample_rate * duration))
        
        # Multi-tone test signal (fundamental + harmonics)
        frequencies = [440, 880, 1320]  # A4 and harmonics
        test_audio = torch.zeros(2, len(t))  # Stereo
        
        for freq in frequencies:
            tone = 0.1 * torch.sin(2 * np.pi * freq * t)
            test_audio[0] += tone  # Left channel
            test_audio[1] += tone  # Right channel
        
        print(f"✅ Created test signal: {test_audio.shape} @ {sample_rate} Hz")
        print(f"   Duration: {duration}s")
        print(f"   RMS level: {torch.sqrt(torch.mean(test_audio**2)):.3f}")
        
        # Test different terms/presets
        test_terms = ['warm', 'bright', 'punchy']
        available_terms = list(system.loader.term_profiles.keys())
        available_presets = list(system.presets.keys())
        
        print(f"\n📋 Available dataset terms: {available_terms}")
        print(f"📋 Available presets: {available_presets}")
        
        # Use first available term for testing
        if available_presets:
            test_term = available_presets[0]
        elif available_terms:
            test_term = available_terms[0]
        else:
            test_term = 'warm'  # Fallback
        
        print(f"\n🎛️  Testing with term: '{test_term}'")
        
        # Apply mastering
        processed_audio, profile = system.apply_mastering(test_audio, test_term)
        
        print(f"✅ Mastering applied successfully!")
        print(f"   Input shape: {test_audio.shape}")
        print(f"   Output shape: {processed_audio.shape}")
        print(f"   Profile confidence: {profile.confidence:.2f}")
        
        # Audio statistics
        rms_before = torch.sqrt(torch.mean(test_audio ** 2))
        rms_after = torch.sqrt(torch.mean(processed_audio ** 2))
        rms_change_db = 20 * torch.log10(rms_after / rms_before)
        
        print(f"\n📊 Audio statistics:")
        print(f"   RMS change: {rms_change_db:.2f} dB")
        print(f"   Peak before: {torch.max(torch.abs(test_audio)):.3f}")
        print(f"   Peak after: {torch.max(torch.abs(processed_audio)):.3f}")
        
        # Save test output
        output_dir = Path("./test_output")
        output_dir.mkdir(exist_ok=True)
        
        original_path = output_dir / "test_original.wav"
        processed_path = output_dir / f"test_{test_term}.wav"
        
        torchaudio.save(str(original_path), test_audio, sample_rate)
        torchaudio.save(str(processed_path), processed_audio[0], sample_rate)
        
        print(f"\n💾 Test files saved:")
        print(f"   Original: {original_path}")
        print(f"   Processed: {processed_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ Audio processing failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_with_real_audio(audio_path: str):
    """Test with real audio file"""
    print(f"\n{'='*70}")
    print(f"TEST: REAL AUDIO PROCESSING")
    print(f"{'='*70}")
    
    if not Path(audio_path).exists():
        print(f"❌ Audio file not found: {audio_path}")
        return False
    
    try:
        # Initialize system
        system = SemanticMasteringEQ(sample_rate=44100, cache_dir="./test_cache")
        system.initialize()
        
        # Load audio
        audio, sr = torchaudio.load(audio_path)
        print(f"📥 Loaded: {audio.shape} @ {sr} Hz")
        
        # Test with multiple terms
        test_terms = []
        if system.presets:
            test_terms.extend(list(system.presets.keys())[:3])  # First 3 presets
        if system.loader.term_profiles:
            available = list(system.loader.term_profiles.keys())
            test_terms.extend(available[:3])  # First 3 dataset terms
        
        test_terms = list(set(test_terms))  # Remove duplicates
        
        if not test_terms:
            test_terms = ['warm']  # Fallback
        
        print(f"🧪 Testing with terms: {test_terms}")
        
        output_dir = Path("./test_output")
        output_dir.mkdir(exist_ok=True)
        
        for term in test_terms:
            print(f"\n🎛️  Processing with '{term}'...")
            
            # Apply mastering
            processed, profile = system.apply_mastering(audio, term)
            
            # Save output
            input_stem = Path(audio_path).stem
            output_path = output_dir / f"{input_stem}_{term}.wav"
            torchaudio.save(str(output_path), processed[0], system.sr)
            
            print(f"   ✅ Saved: {output_path}")
            print(f"   Confidence: {profile.confidence:.2f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Real audio processing failed: {e}")
        return False


def run_all_tests():
    """Run all tests in sequence"""
    print(f"\n{'='*70}")
    print("🚀 RUNNING ALL TESTS")
    print(f"{'='*70}")
    
    results = []
    
    # Test 1: Dataset Loading
    results.append(("Dataset Loading", test_dataset_loading()))
    
    # Test 2: Parameter Processing
    results.append(("Parameter Processing", test_parameter_processing()))
    
    # Test 3: Audio Processing
    results.append(("Audio Processing", test_audio_processing()))
    
    # Summary
    print(f"\n{'='*70}")
    print("📊 TEST RESULTS SUMMARY")
    print(f"{'='*70}")
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:30} {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("\n🎉 All tests passed! Your semantic mastering system is ready.")
        print("\nNext steps:")
        print("1. Try with your own audio: python semantic_mastering.py --input your_mix.wav --preset warm")
        print("2. Explore dataset: python semantic_mastering.py --inspect")
        print("3. Test all presets: python semantic_mastering.py --input your_mix.wav --test-all")
    else:
        print(f"\n⚠️  {len(results) - passed} test(s) failed. Check the error messages above.")


def main():
    parser = argparse.ArgumentParser(description="Test Semantic Mastering System")
    parser.add_argument('--step', type=int, choices=[1, 2, 3], 
                       help='Run specific test step (1=dataset, 2=processing, 3=audio)')
    parser.add_argument('--all', action='store_true', help='Run all tests')
    parser.add_argument('--audio', help='Test with specific audio file')
    
    args = parser.parse_args()
    
    if args.all:
        run_all_tests()
    elif args.step == 1:
        test_dataset_loading()
    elif args.step == 2:
        test_parameter_processing()
    elif args.step == 3:
        test_audio_processing()
    elif args.audio:
        test_with_real_audio(args.audio)
    else:
        print(__doc__)
        print("\nQuick start:")
        print("python test_semantic_mastering.py --all")


if __name__ == '__main__':
    main()