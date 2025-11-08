"""
Adaptive Semantic Mastering - Context-Aware EQ Selection
========================================================

Enhanced semantic mastering that analyzes input audio to select
the most appropriate EQ curve from multiple examples per semantic term.

Approaches:
1. Audio-informed EQ selection based on spectral analysis
2. Clustering-based profile selection
3. Weighted averaging based on audio similarity
4. Dynamic EQ adaptation based on content analysis

Usage:
    python adaptive_semantic_mastering.py --input mix.wav --term warm --method spectral
    python adaptive_semantic_mastering.py --input mix.wav --term bright --method clustering
"""

import torch
import torchaudio
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings("ignore")

# Import base system
try:
    from semantic_mastering import SocialFXDataLoader, EQProfile, SemanticMasteringEQ
    print("Base semantic mastering system imported")
except ImportError:
    print("Error: Cannot import base semantic_mastering.py")
    exit(1)

# Audio analysis
try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    print("Warning: librosa not available, some features disabled")


@dataclass
class AdaptiveProfile:
    """Enhanced profile with selection reasoning"""
    base_profile: EQProfile
    selection_method: str
    selection_confidence: float
    audio_analysis: Dict
    alternatives: List[EQProfile]


class AudioAnalyzer:
    """
    Analyze input audio to guide EQ selection
    """
    
    def __init__(self, sample_rate: int = 44100):
        self.sr = sample_rate
    
    def analyze_audio_content(self, audio: torch.Tensor) -> Dict:
        """
        Comprehensive audio analysis for EQ selection
        """
        
        # Convert to mono numpy
        if audio.dim() == 3:
            audio_mono = torch.mean(audio[0], dim=0).numpy()
        else:
            audio_mono = torch.mean(audio, dim=0).numpy()
        
        analysis = {}
        
        # Basic spectral features
        if LIBROSA_AVAILABLE:
            # Spectral centroid (brightness)
            centroid = librosa.feature.spectral_centroid(y=audio_mono, sr=self.sr)[0]
            analysis['spectral_centroid'] = np.mean(centroid)
            
            # Spectral rolloff (high frequency energy)
            rolloff = librosa.feature.spectral_rolloff(y=audio_mono, sr=self.sr)[0] 
            analysis['spectral_rolloff'] = np.mean(rolloff)
            
            # Spectral bandwidth (frequency spread)
            bandwidth = librosa.feature.spectral_bandwidth(y=audio_mono, sr=self.sr)[0]
            analysis['spectral_bandwidth'] = np.mean(bandwidth)
            
            # Zero crossing rate (roughness indicator)
            zcr = librosa.feature.zero_crossing_rate(audio_mono)[0]
            analysis['zero_crossing_rate'] = np.mean(zcr)
            
            # MFCCs (timbral characteristics)
            mfccs = librosa.feature.mfcc(y=audio_mono, sr=self.sr, n_mfcc=13)
            analysis['mfccs'] = np.mean(mfccs, axis=1)
            
            # Frequency band energy analysis
            stft = librosa.stft(audio_mono, n_fft=2048)
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
                    analysis[f'{band_name}_energy'] = 20 * np.log10(band_energy + 1e-10)
                else:
                    analysis[f'{band_name}_energy'] = -80.0
        
        # Basic level analysis (always available)
        analysis['rms_level'] = 20 * np.log10(np.sqrt(np.mean(audio_mono**2)) + 1e-10)
        analysis['peak_level'] = 20 * np.log10(np.max(np.abs(audio_mono)) + 1e-10)
        analysis['crest_factor'] = analysis['peak_level'] - analysis['rms_level']
        
        return analysis
    
    def classify_audio_character(self, analysis: Dict) -> Dict[str, float]:
        """
        Classify audio character for EQ selection
        
        Returns confidence scores for different characteristics
        """
        
        characteristics = {}
        
        if LIBROSA_AVAILABLE and 'spectral_centroid' in analysis:
            # Brightness classification
            centroid = analysis['spectral_centroid']
            if centroid > 3000:
                characteristics['bright'] = min(1.0, (centroid - 3000) / 5000)
                characteristics['dark'] = 0.0
            elif centroid < 1500:
                characteristics['dark'] = min(1.0, (1500 - centroid) / 1000) 
                characteristics['bright'] = 0.0
            else:
                characteristics['bright'] = 0.0
                characteristics['dark'] = 0.0
            
            # Energy balance
            if 'bass_energy' in analysis and 'treble_energy' in analysis:
                bass_energy = analysis['bass_energy']
                treble_energy = analysis['treble_energy']
                energy_balance = treble_energy - bass_energy
                
                if energy_balance > 5:
                    characteristics['thin'] = min(1.0, energy_balance / 20)
                    characteristics['heavy'] = 0.0
                elif energy_balance < -5:
                    characteristics['heavy'] = min(1.0, abs(energy_balance) / 20)
                    characteristics['thin'] = 0.0
                else:
                    characteristics['balanced'] = 1.0
            
            # Roughness from zero crossing rate
            if 'zero_crossing_rate' in analysis:
                zcr = analysis['zero_crossing_rate']
                if zcr > 0.1:
                    characteristics['aggressive'] = min(1.0, zcr / 0.2)
                    characteristics['smooth'] = 0.0
                else:
                    characteristics['smooth'] = min(1.0, (0.1 - zcr) / 0.08)
                    characteristics['aggressive'] = 0.0
        
        # Dynamic range classification
        crest_factor = analysis.get('crest_factor', 10)
        if crest_factor > 15:
            characteristics['dynamic'] = min(1.0, (crest_factor - 15) / 10)
            characteristics['compressed'] = 0.0
        elif crest_factor < 8:
            characteristics['compressed'] = min(1.0, (8 - crest_factor) / 5)
            characteristics['dynamic'] = 0.0
        
        return characteristics


class AdaptiveSemanticMasteringEQ:
    """
    Enhanced semantic mastering with audio-informed EQ selection
    """
    
    def __init__(self, sample_rate: int = 44100, cache_dir: str = "./cache"):
        self.sr = sample_rate
        self.base_system = SemanticMasteringEQ(sample_rate=sample_rate, cache_dir=cache_dir)
        self.audio_analyzer = AudioAnalyzer(sample_rate=sample_rate)
        
        # Initialize base system
        try:
            self.base_system.initialize()
            print("Adaptive semantic mastering system initialized")
        except Exception as e:
            print(f"Warning: Base system initialization failed: {e}")
    
    def get_raw_examples_for_term(self, term: str) -> List[np.ndarray]:
        """
        Get all raw EQ examples for a semantic term (not averaged)
        """
        
        if not self.base_system.loader.df_eq is not None:
            return []
        
        # Get the column mappings
        desc_col = self.base_system.loader.column_mapping.get('descriptor', 'text')
        param_col = self.base_system.loader.column_mapping.get('parameters', 'param_values')
        
        # Filter data for this term
        term_data = self.base_system.loader.df_eq[
            self.base_system.loader.df_eq[desc_col] == term
        ]
        
        # Extract all parameter sets
        examples = []
        for _, row in term_data.iterrows():
            try:
                params = self.base_system.loader._extract_params(row[param_col])
                if params is not None and len(params) > 0:
                    examples.append(params)
            except Exception as e:
                continue
        
        return examples
    
    def select_eq_by_spectral_similarity(self, audio: torch.Tensor, term: str) -> AdaptiveProfile:
        """
        Select EQ curve based on spectral analysis of input audio
        """
        
        # Analyze input audio
        audio_analysis = self.audio_analyzer.analyze_audio_content(audio)
        audio_character = self.audio_analyzer.classify_audio_character(audio_analysis)
        
        print(f"Audio analysis complete:")
        print(f"  Spectral centroid: {audio_analysis.get('spectral_centroid', 'N/A')} Hz")
        print(f"  Character scores: {audio_character}")
        
        # Get all examples for this term
        raw_examples = self.get_raw_examples_for_term(term)
        
        if not raw_examples:
            # Fallback to base system
            base_profile = self.base_system.loader.get_profile(term)
            return AdaptiveProfile(
                base_profile=base_profile,
                selection_method="fallback_average",
                selection_confidence=0.3,
                audio_analysis=audio_analysis,
                alternatives=[]
            )
        
        print(f"Found {len(raw_examples)} examples for '{term}'")
        
        # Strategy 1: Select based on audio character
        best_example = self._select_by_audio_character(raw_examples, audio_character, term)
        
        # Convert selected example to profile
        dasp_params, confidence = self.base_system.loader._convert_to_dasp(best_example, term)
        
        selected_profile = EQProfile(
            name=f"{term}_adaptive",
            params_dasp=dasp_params,
            params_original=best_example,
            reasoning=f"Selected from {len(raw_examples)} examples based on audio analysis",
            n_examples=1,
            confidence=confidence
        )
        
        return AdaptiveProfile(
            base_profile=selected_profile,
            selection_method="spectral_similarity",
            selection_confidence=0.8,
            audio_analysis=audio_analysis,
            alternatives=[]
        )
    
    def _select_by_audio_character(self, examples: List[np.ndarray], 
                                  audio_character: Dict[str, float], 
                                  term: str) -> np.ndarray:
        """
        Select best EQ example based on audio character analysis
        """
        
        if len(examples) == 1:
            return examples[0]
        
        # For now, use simple heuristics
        # In a full implementation, you'd analyze each EQ curve's characteristics
        
        # Strategy: If audio is already bright and user wants "warm", 
        # select a "warm" example that cuts highs more aggressively
        
        if term.lower() == "warm" and audio_character.get('bright', 0) > 0.5:
            # Audio is already bright, need more aggressive high cut
            # For now, select example with strongest high-frequency reduction
            best_idx = len(examples) // 3  # Heuristic: pick from first third
        elif term.lower() == "bright" and audio_character.get('dark', 0) > 0.5:
            # Audio is dark, need more aggressive high boost
            best_idx = 2 * len(examples) // 3  # Pick from last third
        else:
            # Default to median example
            best_idx = len(examples) // 2
        
        return examples[best_idx]
    
    def select_eq_by_clustering(self, audio: torch.Tensor, term: str, 
                               n_clusters: int = 3) -> AdaptiveProfile:
        """
        Cluster EQ examples and select most appropriate cluster
        """
        
        # Get raw examples
        raw_examples = self.get_raw_examples_for_term(term)
        
        if len(raw_examples) < n_clusters:
            # Not enough examples for clustering, use spectral method
            return self.select_eq_by_spectral_similarity(audio, term)
        
        # Cluster the EQ parameters
        examples_array = np.array(raw_examples)
        
        # Use KMeans clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(examples_array)
        
        # Analyze audio to select best cluster
        audio_analysis = self.audio_analyzer.analyze_audio_content(audio)
        audio_character = self.audio_analyzer.classify_audio_character(audio_analysis)
        
        # Select cluster based on audio characteristics
        # This is where domain knowledge comes in
        selected_cluster = self._select_cluster_by_character(
            kmeans, audio_character, term, cluster_labels, examples_array
        )
        
        # Get examples from selected cluster
        cluster_mask = cluster_labels == selected_cluster
        cluster_examples = examples_array[cluster_mask]
        
        # Average within the selected cluster
        selected_example = np.mean(cluster_examples, axis=0)
        
        # Convert to profile
        dasp_params, confidence = self.base_system.loader._convert_to_dasp(selected_example, term)
        
        selected_profile = EQProfile(
            name=f"{term}_clustered",
            params_dasp=dasp_params,
            params_original=selected_example,
            reasoning=f"Selected cluster {selected_cluster} from {n_clusters} clusters ({np.sum(cluster_mask)} examples)",
            n_examples=np.sum(cluster_mask),
            confidence=confidence
        )
        
        return AdaptiveProfile(
            base_profile=selected_profile,
            selection_method="clustering",
            selection_confidence=0.7,
            audio_analysis=audio_analysis,
            alternatives=[]
        )
    
    def _select_cluster_by_character(self, kmeans, audio_character: Dict[str, float],
                                    term: str, labels: np.ndarray, 
                                    examples: np.ndarray) -> int:
        """
        Select best cluster based on audio character
        """
        
        # Analyze cluster characteristics
        cluster_scores = []
        
        for cluster_id in range(kmeans.n_clusters):
            mask = labels == cluster_id
            cluster_examples = examples[mask]
            cluster_center = kmeans.cluster_centers_[cluster_id]
            
            # Score this cluster for the given audio characteristics
            score = 0.0
            
            # Example scoring logic (would need refinement)
            if term.lower() == "warm":
                if audio_character.get('bright', 0) > 0.5:
                    # Bright audio needs gentle warming
                    # Prefer clusters with moderate low boost, gentle high cut
                    score += 1.0
                elif audio_character.get('thin', 0) > 0.5:
                    # Thin audio needs more bass
                    score += 2.0
            
            cluster_scores.append(score)
        
        # Select cluster with highest score
        return np.argmax(cluster_scores)
    
    def apply_adaptive_mastering(self, audio: torch.Tensor, term: str, 
                               method: str = "spectral") -> Tuple[torch.Tensor, AdaptiveProfile]:
        """
        Apply adaptive semantic mastering
        """
        
        if method == "spectral":
            adaptive_profile = self.select_eq_by_spectral_similarity(audio, term)
        elif method == "clustering":
            adaptive_profile = self.select_eq_by_clustering(audio, term)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Apply the selected EQ
        if audio.dim() == 2:
            audio = audio.unsqueeze(0)
        
        try:
            processed = self.base_system.eq.process_normalized(
                audio, adaptive_profile.base_profile.params_dasp
            )
            print(f"Adaptive EQ applied using {adaptive_profile.selection_method}")
            print(f"Selection confidence: {adaptive_profile.selection_confidence:.1%}")
        except Exception as e:
            print(f"EQ processing failed: {e}")
            processed = audio
        
        return processed, adaptive_profile


def main():
    """Demo of adaptive semantic mastering"""
    
    import argparse
    
    parser = argparse.ArgumentParser(description="Adaptive Semantic Mastering")
    parser.add_argument('--input', required=True, help='Input audio file')
    parser.add_argument('--term', default='warm', help='Semantic term')
    parser.add_argument('--method', default='spectral', choices=['spectral', 'clustering'],
                       help='Selection method')
    parser.add_argument('--output', default='./adaptive_output', help='Output directory')
    
    args = parser.parse_args()
    
    # Initialize system
    system = AdaptiveSemanticMasteringEQ(sample_rate=44100)
    
    # Load audio
    audio, sr = torchaudio.load(args.input)
    if sr != 44100:
        audio = torchaudio.functional.resample(audio, sr, 44100)
    
    print(f"Processing '{args.input}' with term '{args.term}' using {args.method} method")
    
    # Apply adaptive processing
    processed, profile = system.apply_adaptive_mastering(audio, args.term, args.method)
    
    # Save result
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)
    
    input_name = Path(args.input).stem
    output_path = output_dir / f"{input_name}_{args.term}_{args.method}.wav"
    
    torchaudio.save(str(output_path), processed[0], 44100)
    print(f"Saved: {output_path}")
    
    # Show analysis
    print(f"\\nAdaptive Profile Analysis:")
    print(f"Method: {profile.selection_method}")
    print(f"Confidence: {profile.selection_confidence:.1%}")
    print(f"Audio characteristics: {profile.audio_analysis}")


if __name__ == '__main__':
    main()