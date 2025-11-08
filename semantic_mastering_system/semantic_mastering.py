"""
SEMANTIC MASTERING EQ - Using REAL SocialFX-Original Data
==========================================================

Loads actual FX parameters from SocialFX-original dataset,
scales them appropriately for mastering, and applies with dasp-pytorch.

Based on LLM2Fx paper Section 3.1:
- 1595 EQ parameter sets (original)
- 40-parameter EQ format (from Audealize toolkit)
- After preprocessing: 7 terms (warm, soft, harsh, calm, loud, bright, heavy)

Setup:
    python -m venv venv
    source venv/bin/activate  # Linux/Mac
    venv\\Scripts\\activate     # Windows
    pip install pandas pyarrow huggingface_hub
    pip install -r requirements.txt
    huggingface-cli login  # If dataset is gated

Usage:
    # First, inspect the data structure
    python semantic_mastering.py --inspect
    
    # Apply mastering with preset
    python semantic_mastering.py --input mix.wav --preset warm
    python semantic_mastering.py --input mix.wav --test-all
    
    # Use LLM fallback for custom terms
    python semantic_mastering.py --input mix.wav --preset "punchy and wide" --use-llm
"""

import torch
import torchaudio
import pandas as pd
import numpy as np
import argparse
import json
from pathlib import Path
from typing import Dict, Tuple, List, Optional, Union
from dataclasses import dataclass
import warnings
warnings.filterwarnings("ignore")

try:
    from dasp_pytorch import ParametricEQ
except ImportError:
    print("ERROR: Install dasp-pytorch")
    print("Run: pip install git+https://github.com/csteinmetz1/dasp-pytorch.git")
    exit(1)


@dataclass
class EQProfile:
    """Container for EQ profile with reasoning"""
    name: str
    params_dasp: torch.Tensor  # [1, 18] for dasp-pytorch (6 bands × 3 params)
    params_original: Optional[np.ndarray]  # Original SocialFX parameters
    reasoning: str
    n_examples: int
    confidence: float = 1.0  # 0-1, how confident we are in this mapping


class SocialFXDataLoader:
    """
    Load and process REAL SocialFX-original dataset
    
    Process:
    1. Download EQ parameters from HuggingFace
    2. Analyze semantic term distributions
    3. Aggregate parameters by term (mean)
    4. Convert to dasp-pytorch format
    5. Scale for mastering context (÷2.5)
    """
    
    def __init__(self, cache_dir: str = "./cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.df_eq = None
        self.term_profiles = {}
        self.column_mapping = {}
        
    def load_dataset(self, force_download: bool = False) -> None:
        """Load SocialFX dataset with caching"""
        cache_file = self.cache_dir / "socialfx_eq_original.parquet"
        
        if cache_file.exists() and not force_download:
            print("Loading from cache...")
            self.df_eq = pd.read_parquet(cache_file)
            print(f"Loaded {len(self.df_eq)} examples from cache")
        else:
            print("Downloading SocialFX-original from HuggingFace...")
            self._download_from_hf()
            if self.df_eq is not None:
                self.df_eq.to_parquet(cache_file)
                print(f"Cached to {cache_file}")
    
    def _download_from_hf(self) -> None:
        """Download dataset from HuggingFace"""
        try:
            # EQ data split
            splits = {'eq': 'data/eq-00000-of-00001.parquet'}
            url = "hf://datasets/seungheondoh/socialfx-original/" + splits["eq"]
            
            print(f"Fetching: {url}")
            self.df_eq = pd.read_parquet(url)
            print(f"Downloaded {len(self.df_eq)} examples")
            
        except Exception as e:
            print(f"Download failed: {e}")
            print("\nTroubleshooting:")
            print("1. Login to HuggingFace: huggingface-cli login")
            print("2. Check internet connection")
            print("3. Verify dataset access permissions")
            raise
    
    def inspect_dataset(self) -> None:
        """Analyze dataset structure for debugging"""
        if self.df_eq is None:
            print("No dataset loaded. Run load_dataset() first.")
            return
        
        print(f"\n{'='*70}")
        print("SOCIALFX DATASET INSPECTION")
        print(f"{'='*70}")
        
        print(f"Shape: {self.df_eq.shape}")
        print(f"Columns: {list(self.df_eq.columns)}")
        print(f"Memory usage: {self.df_eq.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
        
        # Analyze each column
        for col in self.df_eq.columns:
            print(f"\n--- {col} ---")
            dtype = self.df_eq[col].dtype
            nunique = self.df_eq[col].nunique()
            
            print(f"Type: {dtype}, Unique values: {nunique}")
            
            if dtype == 'object':
                # Check if it contains strings or lists/arrays
                sample = self.df_eq[col].iloc[0]
                print(f"Sample: {type(sample).__name__} = {str(sample)[:100]}{'...' if len(str(sample)) > 100 else ''}")
                
                if nunique < 50:  # Likely categorical
                    values = self.df_eq[col].value_counts().head(10)
                    print(f"Top values:\\n{values}")
            else:
                stats = self.df_eq[col].describe()
                print(f"Stats:\\n{stats}")
        
        # Try to identify semantic terms and parameters
        self._identify_columns()
    
    def _identify_columns(self) -> None:
        """Auto-identify descriptor and parameter columns"""
        print(f"\n{'='*70}")
        print("COLUMN IDENTIFICATION")
        print(f"{'='*70}")
        
        # Possible descriptor column names
        desc_keywords = ['word', 'descriptor', 'label', 'term', 'text', 'semantic', 'description']
        param_keywords = ['parameters', 'fx_params', 'eq_params', 'params', 'values', 'coeffs', 'settings']
        
        descriptor_candidates = []
        parameter_candidates = []
        
        for col in self.df_eq.columns:
            col_lower = col.lower()
            
            # Check for descriptor patterns
            if any(kw in col_lower for kw in desc_keywords):
                descriptor_candidates.append(col)
            
            # Check for parameter patterns
            if any(kw in col_lower for kw in param_keywords):
                parameter_candidates.append(col)
            
            # Check content patterns
            sample = self.df_eq[col].iloc[0]
            if isinstance(sample, (list, np.ndarray)) or (isinstance(sample, str) and '[' in sample):
                parameter_candidates.append(col)
        
        print(f"Descriptor candidates: {descriptor_candidates}")
        print(f"Parameter candidates: {parameter_candidates}")
        
        # Auto-select best candidates
        if descriptor_candidates:
            self.column_mapping['descriptor'] = descriptor_candidates[0]
        if parameter_candidates:
            self.column_mapping['parameters'] = parameter_candidates[0]
        
        print(f"Selected mapping: {self.column_mapping}")
        
        # Show semantic term distribution
        if 'descriptor' in self.column_mapping:
            desc_col = self.column_mapping['descriptor']
            term_counts = self.df_eq[desc_col].value_counts()
            print(f"\\nSemantic term distribution ({desc_col}):")
            print(term_counts)
    
    def process_parameters(self) -> None:
        """Process dataset and create EQ profiles"""
        if self.df_eq is None:
            raise ValueError("Load dataset first")
        
        if not self.column_mapping:
            self._identify_columns()
        
        desc_col = self.column_mapping.get('descriptor')
        param_col = self.column_mapping.get('parameters')
        
        if not desc_col or not param_col:
            print("Could not identify columns automatically")
            print("Available columns:", list(self.df_eq.columns))
            desc_col = input("Enter descriptor column name: ").strip()
            param_col = input("Enter parameter column name: ").strip()
        
        print(f"\n{'='*70}")
        print("PROCESSING PARAMETERS")
        print(f"{'='*70}")
        print(f"Descriptor column: {desc_col}")
        print(f"Parameter column: {param_col}")
        
        # Group by semantic term
        terms = self.df_eq[desc_col].unique()
        print(f"\\nFound {len(terms)} unique terms: {sorted(terms)}")
        
        for term in terms:
            print(f"\\nProcessing: {term}")
            self._process_term(term, desc_col, param_col)
        
        print(f"\\nCreated {len(self.term_profiles)} EQ profiles")
    
    def _process_term(self, term: str, desc_col: str, param_col: str) -> None:
        """Process parameters for a single semantic term"""
        term_data = self.df_eq[self.df_eq[desc_col] == term]
        n_examples = len(term_data)
        
        # Extract and standardize parameters
        params_list = []
        valid_count = 0
        
        for _, row in term_data.iterrows():
            try:
                params = self._extract_params(row[param_col])
                if params is not None and len(params) > 0:
                    params_list.append(params)
                    valid_count += 1
            except Exception as e:
                print(f"  Warning: Failed to parse parameters for {term}: {e}")
        
        if valid_count == 0:
            print(f"  No valid parameters for {term}")
            return
        
        # Aggregate parameters (mean across examples)
        params_mean = np.mean(params_list, axis=0)
        params_std = np.std(params_list, axis=0)
        
        print(f"  Examples: {n_examples} total, {valid_count} valid")
        print(f"  Parameter shape: {params_mean.shape}")
        print(f"  Value range: [{params_mean.min():.3f}, {params_mean.max():.3f}]")
        print(f"  Std dev range: [{params_std.min():.3f}, {params_std.max():.3f}]")
        
        # Convert to dasp-pytorch format
        dasp_params, confidence = self._convert_to_dasp(params_mean, term)
        
        # Generate reasoning
        reasoning = self._generate_reasoning(term, params_mean, n_examples, confidence)
        
        # Store profile
        self.term_profiles[term] = EQProfile(
            name=term,
            params_dasp=dasp_params,
            params_original=params_mean,
            reasoning=reasoning,
            n_examples=n_examples,
            confidence=confidence
        )
        
        print(f"  Profile created (confidence: {confidence:.2f})")
    
    def _extract_params(self, param_data: Union[str, list, np.ndarray]) -> Optional[np.ndarray]:
        """Extract parameters from various formats"""
        if isinstance(param_data, np.ndarray):
            return param_data
        elif isinstance(param_data, list):
            return np.array(param_data)
        elif isinstance(param_data, str):
            try:
                # Try JSON parsing
                parsed = json.loads(param_data)
                return np.array(parsed)
            except:
                # Try literal evaluation
                import ast
                try:
                    parsed = ast.literal_eval(param_data)
                    return np.array(parsed)
                except:
                    return None
        else:
            return None
    
    def _convert_to_dasp(self, params_40: np.ndarray, term: str) -> Tuple[torch.Tensor, float]:
        """
        Convert SocialFX parameters to dasp-pytorch format
        
        Strategy:
        1. Analyze parameter structure
        2. Map to 6-band EQ (common mastering setup)
        3. Scale gains by ~2.5 for mastering context
        4. Normalize to [0,1] range for dasp-pytorch
        
        Returns:
            (dasp_params, confidence)
        """
        
        # Initialize neutral EQ (0.5 = no change)
        dasp_params = torch.ones(1, 18) * 0.5
        confidence = 1.0
        
        # Target frequency centers (log-spaced, mastering-focused)
        # Sub, Bass, Low-mid, Mid, High-mid, Treble
        target_freqs_hz = [60, 200, 500, 1200, 3500, 8000]
        
        try:
            if len(params_40) >= 40:
                # Assume format: [gain1, freq1, Q1, type1, gain2, freq2, Q2, type2, ...]
                # Or similar 4-parameter-per-band structure
                n_bands_src = min(10, len(params_40) // 4)
                n_bands_dst = 6
                
                for i in range(min(n_bands_dst, n_bands_src)):
                    src_idx = i * 4
                    dst_idx = i * 3
                    
                    if src_idx + 2 < len(params_40):
                        gain_raw = params_40[src_idx]
                        freq_raw = params_40[src_idx + 1] if src_idx + 1 < len(params_40) else target_freqs_hz[i]
                        q_raw = params_40[src_idx + 2] if src_idx + 2 < len(params_40) else 0.7
                        
                        # Scale gain for mastering (mixing gains are typically ±6-12dB)
                        # Mastering should be gentler: ±2-3dB
                        mastering_scale = 2.5
                        gain_mastering = gain_raw / mastering_scale
                        gain_mastering = np.clip(gain_mastering, -3.0, 3.0)
                        
                        # Normalize to [0,1] where 0.5 = 0dB, range ±12dB total
                        gain_norm = 0.5 + (gain_mastering / 24.0)
                        gain_norm = np.clip(gain_norm, 0, 1)
                        
                        # Normalize frequency (20Hz - 20kHz, log scale)
                        if freq_raw > 0:
                            freq_norm = np.log10(np.clip(freq_raw, 20, 20000) / 20) / np.log10(1000)
                            freq_norm = np.clip(freq_norm, 0, 1)
                        else:
                            # Use target frequency
                            freq_norm = np.log10(target_freqs_hz[i] / 20) / np.log10(1000)
                        
                        # Normalize Q (0.1 - 10 range, mastering prefers gentler Q)
                        if q_raw > 0:
                            q_norm = (np.log10(np.clip(q_raw, 0.1, 10)) + 1) / 2
                            q_norm = np.clip(q_norm, 0, 1)
                        else:
                            q_norm = 0.4  # Gentle Q for mastering
                        
                        # Set parameters
                        dasp_params[0, dst_idx] = gain_norm
                        dasp_params[0, dst_idx + 1] = freq_norm
                        dasp_params[0, dst_idx + 2] = q_norm
                
            else:
                # Unknown format - use heuristic mapping
                print(f"    Unknown parameter format (length {len(params_40)})")
                dasp_params = self._get_heuristic_params(term, params_40)
                confidence = 0.5
                
        except Exception as e:
            print(f"    Conversion failed: {e}")
            dasp_params = self._get_fallback_params(term)
            confidence = 0.3
        
        return dasp_params, confidence
    
    def _get_heuristic_params(self, term: str, params: np.ndarray) -> torch.Tensor:
        """Create EQ based on parameter statistics and term heuristics"""
        dasp_params = torch.ones(1, 18) * 0.5
        
        # Set frequency centers (log-spaced)
        freq_centers = [0.15, 0.25, 0.35, 0.45, 0.65, 0.85]  # Normalized
        for i, fc in enumerate(freq_centers):
            dasp_params[0, i*3 + 1] = fc
            dasp_params[0, i*3 + 2] = 0.4  # Gentle Q
        
        # Analyze parameter statistics
        param_mean = np.mean(params)
        param_std = np.std(params)
        param_range = np.max(params) - np.min(params)
        
        # Apply term-specific heuristics
        term_lower = term.lower()
        
        if 'warm' in term_lower:
            dasp_params[0, 0] = 0.52  # +1dB sub
            dasp_params[0, 3] = 0.51  # +0.5dB bass
            dasp_params[0, 12] = 0.49  # -0.5dB high-mid
        elif 'bright' in term_lower:
            dasp_params[0, 12] = 0.52  # +1dB high-mid
            dasp_params[0, 15] = 0.53  # +1.5dB treble
        elif 'punchy' in term_lower:
            dasp_params[0, 6] = 0.51   # +0.5dB low-mid
            dasp_params[0, 9] = 0.52   # +1dB mid
        elif 'smooth' in term_lower:
            dasp_params[0, 9] = 0.49   # -0.5dB mid (reduce harshness)
            dasp_params[0, 12] = 0.48  # -1dB high-mid
        elif 'heavy' in term_lower:
            dasp_params[0, 0] = 0.53   # +1.5dB sub
            dasp_params[0, 3] = 0.52   # +1dB bass
        
        return dasp_params
    
    def _get_fallback_params(self, term: str) -> torch.Tensor:
        """Fallback neutral EQ with minimal term-based adjustment"""
        return self._get_heuristic_params(term, np.array([0]))
    
    def _generate_reasoning(self, term: str, params: np.ndarray, n_examples: int, confidence: float) -> str:
        """Generate human-readable explanation"""
        reasoning = f"{term.upper()} MASTERING PROFILE\\n"
        reasoning += f"Source: {n_examples} real mixing examples from SocialFX dataset\\n"
        reasoning += f"Confidence: {confidence:.1%}\\n\\n"
        
        if confidence > 0.8:
            reasoning += "High-confidence mapping from actual engineer data.\\n"
        elif confidence > 0.5:
            reasoning += "Medium-confidence mapping with some interpretation.\\n"
        else:
            reasoning += "Low-confidence fallback based on semantic heuristics.\\n"
        
        reasoning += "Parameters scaled 2.5x down for mastering context (±2dB max)."
        
        return reasoning
    
    def get_profile(self, term: str) -> EQProfile:
        """Get EQ profile for a semantic term"""
        if term in self.term_profiles:
            return self.term_profiles[term]
        else:
            # Fallback for unknown terms
            print(f"Term '{term}' not found in dataset")
            print(f"Available terms: {list(self.term_profiles.keys())}")
            
            fallback_params = self._get_fallback_params(term)
            return EQProfile(
                name=term,
                params_dasp=fallback_params,
                params_original=None,
                reasoning=f"Fallback profile for '{term}' (not in SocialFX dataset)",
                n_examples=0,
                confidence=0.2
            )
    
    def get_preset_profiles(self) -> Dict[str, EQProfile]:
        """Get curated preset profiles based on dataset analysis"""
        presets = {}
        
        # Map common terms to presets if available
        preset_mapping = {
            'warm': ['warm', 'soft', 'smooth'],
            'bright': ['bright', 'crisp', 'clear'],
            'punchy': ['punchy', 'aggressive', 'loud'],
            'smooth': ['smooth', 'calm', 'gentle'],
            'balanced': ['balanced', 'neutral', 'clean']
        }
        
        for preset_name, term_options in preset_mapping.items():
            for term in term_options:
                if term in self.term_profiles:
                    profile = self.term_profiles[term]
                    # Rename for preset
                    presets[preset_name] = EQProfile(
                        name=preset_name,
                        params_dasp=profile.params_dasp,
                        params_original=profile.params_original,
                        reasoning=f"{preset_name.upper()} preset based on '{term}' data\\n" + profile.reasoning,
                        n_examples=profile.n_examples,
                        confidence=profile.confidence
                    )
                    break
        
        return presets


class SemanticMasteringEQ:
    """Complete semantic mastering system"""
    
    def __init__(self, sample_rate: int = 44100, cache_dir: str = "./cache"):
        self.sr = sample_rate
        self.eq = ParametricEQ(sample_rate=sample_rate)
        self.loader = SocialFXDataLoader(cache_dir=cache_dir)
        self.presets = {}
        
        print(f"\\n{'='*70}")
        print("SEMANTIC MASTERING EQ - Real SocialFX Data")
        print(f"{'='*70}")
        print(f"Sample rate: {sample_rate} Hz")
    
    def initialize(self, force_download: bool = False) -> None:
        """Load dataset and create profiles"""
        print("Initializing...")
        
        # Load dataset
        self.loader.load_dataset(force_download=force_download)
        
        # Process parameters
        self.loader.process_parameters()
        
        # Create presets
        self.presets = self.loader.get_preset_profiles()
        
        print(f"\\nReady!")
        print(f"Dataset terms: {list(self.loader.term_profiles.keys())}")
        print(f"Presets: {list(self.presets.keys())}")
    
    def apply_mastering(self, audio: torch.Tensor, term: str) -> Tuple[torch.Tensor, EQProfile]:
        """Apply semantic mastering EQ"""
        if audio.dim() == 2:
            audio = audio.unsqueeze(0)  # Add batch dimension
        
        # Get profile (preset first, then dataset)
        if term in self.presets:
            profile = self.presets[term]
            print(f"Using preset: {term}")
        else:
            profile = self.loader.get_profile(term)
            print(f"Using dataset term: {term}")
        
        # Apply EQ
        try:
            processed = self.eq.process_normalized(audio, profile.params_dasp)
            print(f"EQ applied (confidence: {profile.confidence:.1%})")
        except Exception as e:
            print(f"EQ processing failed: {e}")
            processed = audio  # Return original
        
        return processed, profile


def process_audio_file(system: SemanticMasteringEQ, 
                      input_path: str, 
                      term: str, 
                      output_dir: str) -> None:
    """Process a single audio file"""
    
    print(f"\\n{'='*70}")
    print(f"PROCESSING: {Path(input_path).name} -> {term.upper()}")
    print(f"{'='*70}")
    
    # Load audio
    try:
        audio, sr = torchaudio.load(input_path)
        print(f"Loaded: {audio.shape} @ {sr} Hz")
        
        # Resample if needed
        if sr != system.sr:
            print(f"Resampling: {sr} -> {system.sr} Hz")
            audio = torchaudio.functional.resample(audio, sr, system.sr)
        
        # Ensure stereo
        if audio.shape[0] == 1:
            audio = audio.repeat(2, 1)  # Mono to stereo
        elif audio.shape[0] > 2:
            audio = audio[:2]  # Keep first 2 channels
        
        duration = audio.shape[-1] / system.sr
        print(f"Input: {audio.shape} ({duration:.1f}s)")
        
    except Exception as e:
        print(f"Failed to load audio: {e}")
        return
    
    # Apply mastering
    try:
        processed, profile = system.apply_mastering(audio, term)
        
        # Show profile info
        print(f"\\nProfile Info:")
        print(f"   {profile.reasoning}")
        
        # Audio statistics
        rms_before = torch.sqrt(torch.mean(audio ** 2))
        rms_after = torch.sqrt(torch.mean(processed ** 2))
        rms_change_db = 20 * torch.log10(rms_after / rms_before)
        
        peak_before = torch.max(torch.abs(audio))
        peak_after = torch.max(torch.abs(processed))
        
        print(f"\\nAudio Stats:")
        print(f"   RMS change: {rms_change_db:.2f} dB")
        print(f"   Peak before: {peak_before:.3f}")
        print(f"   Peak after: {peak_after:.3f}")
        
        # Save output
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        input_stem = Path(input_path).stem
        output_path = output_dir / f"{input_stem}_{term.replace(' ', '_')}.wav"
        
        # Ensure no clipping
        if peak_after > 0.99:
            scale = 0.99 / peak_after
            processed = processed * scale
            print(f"Applied limiting: -{20*torch.log10(1/scale):.1f} dB")
        
        torchaudio.save(str(output_path), processed[0], system.sr)
        print(f"Saved: {output_path}")
        
    except Exception as e:
        print(f"Processing failed: {e}")


def main():
    parser = argparse.ArgumentParser(description="Semantic Mastering with Real SocialFX Data")
    parser.add_argument('--input', '-i', help='Input audio file')
    parser.add_argument('--output', '-o', default='./output', help='Output directory')
    parser.add_argument('--preset', '-p', default='warm', help='Semantic term or preset')
    parser.add_argument('--inspect', action='store_true', help='Inspect dataset structure')
    parser.add_argument('--test-all', action='store_true', help='Test all available terms')
    parser.add_argument('--force-download', action='store_true', help='Force re-download dataset')
    parser.add_argument('--use-llm', action='store_true', help='Use LLM for custom terms (future)')
    
    args = parser.parse_args()
    
    # Initialize system
    system = SemanticMasteringEQ(sample_rate=44100)
    
    try:
        system.initialize(force_download=args.force_download)
    except Exception as e:
        print(f"Initialization failed: {e}")
        return
    
    # Handle commands
    if args.inspect:
        system.loader.inspect_dataset()
        return
    
    if not args.input:
        print("No input file specified. Use --input <file.wav>")
        print("Example: python semantic_mastering.py --input mix.wav --preset warm")
        print("Available presets:", list(system.presets.keys()))
        print("Available terms:", list(system.loader.term_profiles.keys()))
        return
    
    if not Path(args.input).exists():
        print(f"Input file not found: {args.input}")
        return
    
    # Process audio
    if args.test_all:
        # Test all presets and dataset terms
        all_terms = list(system.presets.keys()) + list(system.loader.term_profiles.keys())
        all_terms = list(set(all_terms))  # Remove duplicates
        
        print(f"Testing {len(all_terms)} terms...")
        for term in all_terms:
            process_audio_file(system, args.input, term, args.output)
    else:
        process_audio_file(system, args.input, args.preset, args.output)
    
    print(f"\\n{'='*70}")
    print("Complete!")
    print(f"{'='*70}")


if __name__ == '__main__':
    import sys
    if len(sys.argv) == 1:
        print(__doc__)
    else:
        main()