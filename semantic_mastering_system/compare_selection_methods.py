"""
Compare EQ Selection Methods
============================

Compare different approaches for selecting EQ curves from multiple examples:
1. Simple averaging (current method)
2. Audio-informed selection
3. Clustering-based selection
4. Variance analysis of semantic terms

Usage:
    python compare_selection_methods.py --term warm --analyze-variance
    python compare_selection_methods.py --audio mix.wav --term bright --compare-all
"""

import torch
import torchaudio
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import argparse
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings("ignore")

from semantic_mastering import SocialFXDataLoader
try:
    from adaptive_semantic_mastering import AdaptiveSemanticMasteringEQ
    ADAPTIVE_AVAILABLE = True
except ImportError:
    ADAPTIVE_AVAILABLE = False
    print("Adaptive system not available")


class SelectionMethodComparator:
    """
    Compare different EQ selection approaches
    """
    
    def __init__(self):
        self.loader = SocialFXDataLoader()
        self.loader.load_dataset()
        
        if ADAPTIVE_AVAILABLE:
            self.adaptive_system = AdaptiveSemanticMasteringEQ()
        
        # Setup plotting
        plt.style.use('default')
        plt.rcParams.update({
            'figure.figsize': (15, 10),
            'font.size': 10
        })
    
    def analyze_term_variance(self, term: str) -> Dict:
        """
        Analyze variance in EQ parameters for a semantic term
        """
        
        print(f"\\n{'='*60}")
        print(f"VARIANCE ANALYSIS: {term.upper()}")
        print(f"{'='*60}")
        
        # Get raw examples
        if not hasattr(self.loader, 'df_eq') or self.loader.df_eq is None:
            print("No dataset available")
            return {}
        
        desc_col = 'text'
        param_col = 'param_values'
        
        term_data = self.loader.df_eq[self.loader.df_eq[desc_col] == term]
        
        if len(term_data) == 0:
            print(f"No examples found for '{term}'")
            return {}
        
        # Extract all examples
        examples = []
        for _, row in term_data.iterrows():
            try:
                params = self.loader._extract_params(row[param_col])
                if params is not None and len(params) > 0:
                    examples.append(params)
            except:
                continue
        
        if len(examples) == 0:
            print("No valid examples extracted")
            return {}
        
        examples_array = np.array(examples)
        
        print(f"Found {len(examples)} valid examples")
        print(f"Parameter dimensions: {examples_array.shape}")
        
        # Compute statistics
        mean_params = np.mean(examples_array, axis=0)
        std_params = np.std(examples_array, axis=0)
        min_params = np.min(examples_array, axis=0)
        max_params = np.max(examples_array, axis=0)
        
        # Analyze variance
        variance_ratio = std_params / (np.abs(mean_params) + 1e-6)
        high_variance_indices = np.where(variance_ratio > 1.0)[0]
        
        print(f"\\nVariance Analysis:")
        print(f"  Mean parameter range: [{mean_params.min():.3f}, {mean_params.max():.3f}]")
        print(f"  Std deviation range: [{std_params.min():.3f}, {std_params.max():.3f}]") 
        print(f"  High variance parameters: {len(high_variance_indices)}/{len(mean_params)}")
        
        # Identify most/least consistent parameters
        most_consistent_idx = np.argmin(variance_ratio)
        least_consistent_idx = np.argmax(variance_ratio)
        
        print(f"  Most consistent param {most_consistent_idx}: {mean_params[most_consistent_idx]:.3f} ± {std_params[most_consistent_idx]:.3f}")
        print(f"  Least consistent param {least_consistent_idx}: {mean_params[least_consistent_idx]:.3f} ± {std_params[least_consistent_idx]:.3f}")
        
        # Create visualization
        self._plot_parameter_variance(examples_array, term)
        
        return {
            'term': term,
            'n_examples': len(examples),
            'mean_params': mean_params,
            'std_params': std_params,
            'variance_ratio': variance_ratio,
            'examples_array': examples_array
        }
    
    def _plot_parameter_variance(self, examples_array: np.ndarray, term: str):
        """Plot parameter variance analysis"""
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f"Parameter Variance Analysis: {term.upper()}", fontsize=16, fontweight='bold')
        
        # 1. Parameter distribution (first 20 params)
        ax1 = axes[0, 0]
        n_params_to_show = min(20, examples_array.shape[1])
        param_indices = range(n_params_to_show)
        
        ax1.boxplot([examples_array[:, i] for i in param_indices], labels=param_indices)
        ax1.set_title('Parameter Value Distributions (first 20)')
        ax1.set_xlabel('Parameter Index')
        ax1.set_ylabel('Parameter Value')
        ax1.grid(True, alpha=0.3)
        
        # 2. Variance vs Mean scatter
        ax2 = axes[0, 1]
        means = np.mean(examples_array, axis=0)
        stds = np.std(examples_array, axis=0)
        
        ax2.scatter(means, stds, alpha=0.6)
        ax2.set_title('Standard Deviation vs Mean')
        ax2.set_xlabel('Mean Parameter Value')
        ax2.set_ylabel('Standard Deviation')
        ax2.grid(True, alpha=0.3)
        
        # 3. Example diversity (first few examples)
        ax3 = axes[1, 0]
        n_examples_to_show = min(10, len(examples_array))
        for i in range(n_examples_to_show):
            ax3.plot(examples_array[i, :n_params_to_show], alpha=0.7, label=f'Ex {i+1}')
        
        ax3.plot(means[:n_params_to_show], 'k-', linewidth=3, label='Mean')
        ax3.set_title(f'Example Diversity (first {n_examples_to_show} examples)')
        ax3.set_xlabel('Parameter Index')
        ax3.set_ylabel('Parameter Value')
        ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax3.grid(True, alpha=0.3)
        
        # 4. Variance ratio
        ax4 = axes[1, 1]
        variance_ratio = stds / (np.abs(means) + 1e-6)
        ax4.bar(range(len(variance_ratio)), variance_ratio)
        ax4.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='High variance threshold')
        ax4.set_title('Variance Ratio (std/|mean|)')
        ax4.set_xlabel('Parameter Index')
        ax4.set_ylabel('Variance Ratio')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plots_dir = Path("./variance_analysis_plots")
        plots_dir.mkdir(exist_ok=True)
        plt.savefig(plots_dir / f"{term}_variance_analysis.png", dpi=300, bbox_inches='tight')
        print(f"Variance plot saved: {plots_dir}/{term}_variance_analysis.png")
        plt.show()
    
    def compare_selection_methods_on_audio(self, audio_path: str, term: str) -> Dict:
        """
        Compare different selection methods on actual audio
        """
        
        print(f"\\n{'='*60}")
        print(f"COMPARING SELECTION METHODS: {term.upper()}")
        print(f"{'='*60}")
        
        # Load audio
        audio, sr = torchaudio.load(audio_path)
        if sr != 44100:
            audio = torchaudio.functional.resample(audio, sr, 44100)
        
        results = {}
        
        # Method 1: Simple averaging (current default)
        print("\\n1. Simple Averaging (Current Method)")
        profile_avg = self.loader.get_profile(term)
        results['averaging'] = {
            'method': 'Simple averaging',
            'profile': profile_avg,
            'reasoning': 'Mean of all examples'
        }
        print(f"   Examples used: {profile_avg.n_examples}")
        print(f"   Confidence: {profile_avg.confidence:.1%}")
        
        # Method 2: Audio-informed selection (if available)
        if ADAPTIVE_AVAILABLE:
            print("\\n2. Audio-Informed Selection")
            try:
                _, adaptive_profile = self.adaptive_system.apply_adaptive_mastering(
                    audio, term, method="spectral"
                )
                results['adaptive'] = {
                    'method': 'Audio-informed selection',
                    'profile': adaptive_profile.base_profile,
                    'reasoning': adaptive_profile.base_profile.reasoning,
                    'audio_analysis': adaptive_profile.audio_analysis
                }
                print(f"   Selection confidence: {adaptive_profile.selection_confidence:.1%}")
                print(f"   Audio centroid: {adaptive_profile.audio_analysis.get('spectral_centroid', 'N/A')} Hz")
            except Exception as e:
                print(f"   Error: {e}")
        
        # Method 3: Variance-informed selection
        print("\\n3. Variance-Informed Selection")
        variance_profile = self._select_by_variance_analysis(term)
        if variance_profile:
            results['variance'] = {
                'method': 'Variance-informed selection',
                'profile': variance_profile,
                'reasoning': 'Selected based on parameter consistency'
            }
            print(f"   Selection based on consistency analysis")
        
        # Create comparison visualization
        self._plot_method_comparison(results, term, audio_path)
        
        return results
    
    def _select_by_variance_analysis(self, term: str):
        """
        Select EQ based on variance analysis
        (Prefer examples with consistent characteristics)
        """
        
        variance_data = self.analyze_term_variance(term)
        if not variance_data:
            return None
        
        examples_array = variance_data['examples_array']
        variance_ratio = variance_data['variance_ratio']
        
        # Strategy: Select example closest to median in high-consistency parameters
        # Focus on parameters with low variance (more reliable)
        consistent_params = np.where(variance_ratio < 0.5)[0]
        
        if len(consistent_params) == 0:
            # If no consistent params, use mean
            selected_params = variance_data['mean_params']
        else:
            # Find example closest to mean in consistent parameters
            mean_consistent = variance_data['mean_params'][consistent_params]
            
            distances = []
            for example in examples_array:
                example_consistent = example[consistent_params]
                distance = np.linalg.norm(example_consistent - mean_consistent)
                distances.append(distance)
            
            best_idx = np.argmin(distances)
            selected_params = examples_array[best_idx]
        
        # Convert to profile
        dasp_params, confidence = self.loader._convert_to_dasp(selected_params, term)
        
        from semantic_mastering import EQProfile
        return EQProfile(
            name=f"{term}_variance",
            params_dasp=dasp_params,
            params_original=selected_params,
            reasoning=f"Selected based on {len(consistent_params)} consistent parameters",
            n_examples=1,
            confidence=confidence
        )
    
    def _plot_method_comparison(self, results: Dict, term: str, audio_path: str):
        """Plot comparison of different selection methods"""
        
        fig, axes = plt.subplots(2, 1, figsize=(16, 12))
        fig.suptitle(f"EQ Selection Method Comparison: {term.upper()}\\n{Path(audio_path).name}", 
                    fontsize=16, fontweight='bold')
        
        # Frequency response comparison
        ax1 = axes[0]
        freqs = np.logspace(np.log10(20), np.log10(20000), 1000)
        
        colors = ['blue', 'red', 'green', 'purple']
        
        for i, (method_name, method_data) in enumerate(results.items()):
            profile = method_data['profile']
            
            # Compute frequency response (simplified)
            magnitude = self._compute_frequency_response(profile.params_dasp, freqs)
            
            ax1.semilogx(freqs, magnitude, color=colors[i % len(colors)], 
                        linewidth=2.5, label=f"{method_name} (conf: {profile.confidence:.1%})")
        
        ax1.set_title('Frequency Response Comparison')
        ax1.set_xlabel('Frequency (Hz)')
        ax1.set_ylabel('Magnitude (dB)')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        ax1.set_xlim(20, 20000)
        
        # Method characteristics table
        ax2 = axes[1]
        ax2.axis('off')
        
        # Create comparison table
        table_data = []
        for method_name, method_data in results.items():
            profile = method_data['profile']
            row = [
                method_name,
                f"{profile.confidence:.1%}",
                f"{profile.n_examples}",
                method_data['reasoning'][:60] + "..." if len(method_data['reasoning']) > 60 else method_data['reasoning']
            ]
            table_data.append(row)
        
        if table_data:
            table = ax2.table(
                cellText=table_data,
                colLabels=['Method', 'Confidence', 'Examples', 'Reasoning'],
                cellLoc='left',
                loc='center'
            )
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1.2, 2)
            
            # Style header
            for i in range(len(table_data[0])):
                table[(0, i)].set_facecolor('#4CAF50')
                table[(0, i)].set_text_props(weight='bold', color='white')
        
        ax2.set_title('Method Comparison Summary', fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        
        # Save plot
        plots_dir = Path("./method_comparison_plots")
        plots_dir.mkdir(exist_ok=True)
        audio_name = Path(audio_path).stem
        plt.savefig(plots_dir / f"{audio_name}_{term}_method_comparison.png", 
                   dpi=300, bbox_inches='tight')
        print(f"Comparison plot saved: {plots_dir}/{audio_name}_{term}_method_comparison.png")
        plt.show()
    
    def _compute_frequency_response(self, eq_params: torch.Tensor, freqs: np.ndarray) -> np.ndarray:
        """Simplified frequency response computation"""
        
        magnitude_db = np.zeros_like(freqs)
        
        # Extract 6-band EQ parameters
        for band in range(6):
            gain_norm = eq_params[0, band*3].item()
            freq_norm = eq_params[0, band*3 + 1].item()
            q_norm = eq_params[0, band*3 + 2].item()
            
            # Convert to physical units
            gain_db = (gain_norm - 0.5) * 24.0
            center_freq = 20 * (20000/20) ** freq_norm
            q_factor = 0.1 * (10/0.1) ** q_norm
            
            if abs(gain_db) < 0.01:
                continue
            
            # Bell filter response
            omega = freqs / center_freq
            omega_inv = center_freq / freqs
            h_squared = 1 + (10**(gain_db/20) - 1) / (1 + q_factor**2 * (omega - omega_inv)**2)
            h_squared = np.maximum(h_squared, 1e-10)
            magnitude_db += 10 * np.log10(h_squared)
        
        return magnitude_db


def main():
    parser = argparse.ArgumentParser(description="Compare EQ Selection Methods")
    parser.add_argument('--term', default='warm', help='Semantic term to analyze')
    parser.add_argument('--audio', help='Audio file for method comparison')
    parser.add_argument('--analyze-variance', action='store_true', help='Analyze parameter variance')
    parser.add_argument('--compare-all', action='store_true', help='Compare all methods on audio')
    
    args = parser.parse_args()
    
    comparator = SelectionMethodComparator()
    
    if args.analyze_variance:
        comparator.analyze_term_variance(args.term)
    
    if args.compare_all and args.audio:
        if Path(args.audio).exists():
            comparator.compare_selection_methods_on_audio(args.audio, args.term)
        else:
            print(f"Audio file not found: {args.audio}")
    
    if not any([args.analyze_variance, args.compare_all]):
        print("Use --analyze-variance or --compare-all with --audio")
        print("Example: python compare_selection_methods.py --term warm --analyze-variance")


if __name__ == '__main__':
    main()