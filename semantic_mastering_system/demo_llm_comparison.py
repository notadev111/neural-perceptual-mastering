"""
Demo: LLM vs Dataset vs Adaptive EQ Generation
==============================================

Compare different approaches for generating EQ parameters from semantic terms.
Shows the value of each method and when to use them.

No API keys required - uses rule-based LLM simulation for demo.
"""

import numpy as np
import torch
from pathlib import Path
from typing import Dict, List
import matplotlib.pyplot as plt

try:
    from semantic_mastering import EQProfile
    SEMANTIC_AVAILABLE = True
except ImportError:
    SEMANTIC_AVAILABLE = False
    
    # Fallback EQProfile definition
    from dataclasses import dataclass
    
    @dataclass 
    class EQProfile:
        name: str
        params_dasp: torch.Tensor
        params_original: np.ndarray
        reasoning: str
        n_examples: int
        confidence: float = 1.0


class EQMethodDemo:
    """
    Demo different EQ generation approaches without requiring API keys
    """
    
    def __init__(self):
        # Simulated dataset statistics for common terms
        self.dataset_stats = {
            'warm': {
                'examples': 64,
                'mean_eq': [1.4, 1.4, 0.3, -0.2, -0.8, -1.5],  # Simplified 6-band gains
                'variance': [1.0, 0.8, 0.4, 0.5, 0.7, 0.9],
                'strategies': {
                    'gentle': [0.5, 0.5, 0.5, 0.0, 0.0, -0.5],
                    'aggressive': [2.5, 2.0, 0.0, -1.0, -2.0, -3.0], 
                    'balanced': [1.0, 1.0, 0.0, 0.0, -0.5, -1.0]
                }
            },
            'bright': {
                'examples': 19,
                'mean_eq': [-0.5, 0.0, 0.5, 1.0, 1.5, 2.0],
                'variance': [0.8, 0.6, 0.9, 1.2, 0.8, 1.1],
                'strategies': {
                    'gentle': [0.0, 0.0, 0.0, 0.5, 1.0, 1.0],
                    'aggressive': [-1.0, -0.5, 1.0, 2.0, 3.0, 3.5],
                    'balanced': [-0.5, 0.0, 0.5, 1.0, 1.5, 2.0]
                }
            },
            'heavy': {
                'examples': 15,
                'mean_eq': [2.0, 1.5, 0.5, 0.0, -0.5, -0.5],
                'variance': [1.2, 1.0, 0.7, 0.5, 0.4, 0.3],
                'strategies': {
                    'gentle': [1.0, 0.5, 0.0, 0.0, 0.0, 0.0],
                    'aggressive': [3.5, 2.5, 1.0, 0.0, -1.0, -1.0],
                    'balanced': [2.0, 1.5, 0.5, 0.0, -0.5, -0.5]
                }
            }
        }
        
        # LLM response simulation
        self.llm_rules = {
            'warm': {
                'gpt4': [1.2, 0.8, 0.0, -0.3, -1.0, -1.8],  # Conservative
                'claude': [1.5, 1.0, 0.2, -0.5, -1.2, -2.0],  # Slightly more aggressive
                'reasoning': "Warm audio typically benefits from enhanced low frequencies and gentle high-frequency rolloff"
            },
            'bright': {
                'gpt4': [-0.3, 0.0, 0.8, 1.5, 2.0, 1.8],
                'claude': [-0.5, 0.2, 1.0, 1.8, 2.2, 2.0],
                'reasoning': "Brightness is achieved through high-frequency enhancement while maintaining balance"
            },
            'heavy': {
                'gpt4': [2.2, 1.8, 0.8, 0.0, -0.3, -0.5],
                'claude': [2.5, 2.0, 1.0, 0.2, -0.2, -0.3],
                'reasoning': "Heavy sound requires substantial low-frequency enhancement for weight and power"
            }
        }
    
    def simulate_dataset_method(self, term: str) -> Dict:
        """Simulate current dataset averaging method"""
        
        if term not in self.dataset_stats:
            return {'error': f"Term '{term}' not in dataset"}
        
        stats = self.dataset_stats[term]
        
        profile = EQProfile(
            name=f"{term}_dataset",
            params_dasp=torch.tensor([stats['mean_eq']]),  # Simplified
            params_original=np.array(stats['mean_eq']),
            reasoning=f"Average of {stats['examples']} real engineer examples",
            n_examples=stats['examples'],
            confidence=0.7
        )
        
        return {
            'method': 'Dataset Averaging',
            'profile': profile,
            'variance_info': {
                'high_variance_bands': sum(1 for v in stats['variance'] if v > 0.8),
                'max_variance': max(stats['variance']),
                'avg_variance': np.mean(stats['variance'])
            }
        }
    
    def simulate_adaptive_method(self, term: str, audio_type: str = "balanced") -> Dict:
        """Simulate adaptive audio-informed selection"""
        
        if term not in self.dataset_stats:
            return {'error': f"Term '{term}' not in dataset"}
        
        stats = self.dataset_stats[term]
        
        # Select strategy based on audio analysis
        strategy_mapping = {
            "dark_muddy": "aggressive",
            "bright_harsh": "gentle", 
            "balanced": "balanced",
            "thin": "aggressive",
            "bass_heavy": "gentle"
        }
        
        strategy = strategy_mapping.get(audio_type, "balanced")
        selected_eq = stats['strategies'][strategy]
        
        profile = EQProfile(
            name=f"{term}_adaptive",
            params_dasp=torch.tensor([selected_eq]),
            params_original=np.array(selected_eq),
            reasoning=f"Selected '{strategy}' strategy from {stats['examples']} examples based on audio analysis",
            n_examples=stats['examples'] // 3,  # Approximate examples in this strategy
            confidence=0.85
        )
        
        return {
            'method': f'Adaptive Selection ({strategy})',
            'profile': profile,
            'audio_analysis': {
                'detected_type': audio_type,
                'selected_strategy': strategy,
                'alternatives': list(stats['strategies'].keys())
            }
        }
    
    def simulate_llm_method(self, term: str, model: str = "gpt4", audio_description: str = None) -> Dict:
        """Simulate LLM-generated EQ"""
        
        if term not in self.llm_rules:
            # Generate fallback for unknown terms
            fallback_eq = [0.0] * 6  # Neutral
            reasoning = f"No specific rule for '{term}', using neutral EQ"
            confidence = 0.3
        else:
            rule = self.llm_rules[term]
            fallback_eq = rule.get(model, rule['gpt4'])
            reasoning = rule['reasoning']
            confidence = 0.6
        
        # Simulate some variation if audio description provided
        if audio_description:
            if "bright" in audio_description.lower():
                # Reduce high frequency boost for bright audio
                fallback_eq = [g - 0.3 if i >= 3 else g for i, g in enumerate(fallback_eq)]
                reasoning += f" (Adapted for {audio_description})"
                confidence += 0.1
            elif "dark" in audio_description.lower():
                # Increase high frequency boost for dark audio  
                fallback_eq = [g + 0.3 if i >= 3 else g for i, g in enumerate(fallback_eq)]
                reasoning += f" (Adapted for {audio_description})"
                confidence += 0.1
        
        profile = EQProfile(
            name=f"{term}_llm_{model}",
            params_dasp=torch.tensor([fallback_eq]),
            params_original=np.array(fallback_eq),
            reasoning=reasoning,
            n_examples=1,
            confidence=min(confidence, 1.0)
        )
        
        return {
            'method': f'LLM ({model.upper()})',
            'profile': profile,
            'model_info': {
                'model': model,
                'has_audio_context': audio_description is not None,
                'reasoning': reasoning
            }
        }
    
    def compare_all_methods(self, term: str, audio_type: str = "balanced", 
                          audio_description: str = None) -> Dict:
        """Compare all methods for a given term"""
        
        print(f"\\n{'='*70}")
        print(f"COMPARING EQ GENERATION METHODS: {term.upper()}")
        print(f"{'='*70}")
        print(f"Audio context: {audio_type}")
        if audio_description:
            print(f"Audio description: {audio_description}")
        
        results = {}
        
        # Method 1: Dataset averaging
        print("\\n1. Dataset Averaging (Current)")
        dataset_result = self.simulate_dataset_method(term)
        if 'error' not in dataset_result:
            results['dataset'] = dataset_result
            profile = dataset_result['profile']
            variance = dataset_result['variance_info']
            print(f"   Examples: {profile.n_examples}")
            print(f"   Confidence: {profile.confidence:.1%}")
            print(f"   High variance bands: {variance['high_variance_bands']}/6")
            print(f"   EQ curve: {profile.params_original}")
        
        # Method 2: Adaptive selection
        print("\\n2. Adaptive Audio-Informed Selection")
        adaptive_result = self.simulate_adaptive_method(term, audio_type)
        if 'error' not in adaptive_result:
            results['adaptive'] = adaptive_result
            profile = adaptive_result['profile']
            analysis = adaptive_result['audio_analysis']
            print(f"   Selected strategy: {analysis['selected_strategy']}")
            print(f"   Confidence: {profile.confidence:.1%}")
            print(f"   EQ curve: {profile.params_original}")
        
        # Method 3: LLM (GPT-4)
        print("\\n3. LLM Generation (GPT-4 simulation)")
        llm_gpt_result = self.simulate_llm_method(term, "gpt4", audio_description)
        results['llm_gpt'] = llm_gpt_result
        profile = llm_gpt_result['profile']
        print(f"   Confidence: {profile.confidence:.1%}")
        print(f"   EQ curve: {profile.params_original}")
        
        # Method 4: LLM (Claude)
        print("\\n4. LLM Generation (Claude simulation)")
        llm_claude_result = self.simulate_llm_method(term, "claude", audio_description)
        results['llm_claude'] = llm_claude_result
        profile = llm_claude_result['profile']
        print(f"   Confidence: {profile.confidence:.1%}")
        print(f"   EQ curve: {profile.params_original}")
        
        return results
    
    def visualize_comparison(self, results: Dict, term: str):
        """Create comparison visualization"""
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # EQ curve comparison
        band_names = ['Sub\\n60Hz', 'Bass\\n200Hz', 'LowMid\\n500Hz', 'Mid\\n1kHz', 'HighMid\\n3kHz', 'Treble\\n8kHz']
        x = np.arange(len(band_names))
        
        colors = ['blue', 'red', 'green', 'purple', 'orange']
        i = 0
        
        for method_name, method_data in results.items():
            if 'error' in method_data:
                continue
                
            profile = method_data['profile']
            eq_curve = profile.params_original
            
            ax1.plot(x, eq_curve, 'o-', color=colors[i % len(colors)], 
                    linewidth=2.5, markersize=6, 
                    label=f"{method_data['method']} (conf: {profile.confidence:.1%})")
            i += 1
        
        ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax1.set_title(f'EQ Curve Comparison: {term.upper()}', fontweight='bold')
        ax1.set_xlabel('Frequency Band')
        ax1.set_ylabel('Gain (dB)')
        ax1.set_xticks(x)
        ax1.set_xticklabels(band_names)
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Confidence and method characteristics
        methods = []
        confidences = []
        sources = []
        
        for method_name, method_data in results.items():
            if 'error' in method_data:
                continue
            methods.append(method_data['method'])
            confidences.append(method_data['profile'].confidence)
            
            if 'dataset' in method_name:
                sources.append(f"{method_data['profile'].n_examples} examples")
            elif 'adaptive' in method_name:
                sources.append("Audio-informed")
            else:
                sources.append("LLM generated")
        
        bars = ax2.bar(methods, confidences, color=colors[:len(methods)], alpha=0.7)
        ax2.set_title('Method Confidence Comparison', fontweight='bold')
        ax2.set_ylabel('Confidence')
        ax2.set_ylim(0, 1)
        
        # Add source labels
        for i, (bar, source) in enumerate(zip(bars, sources)):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                    source, ha='center', va='bottom', fontsize=9)
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # Save plot
        plots_dir = Path("./comparison_plots")
        plots_dir.mkdir(exist_ok=True)
        plt.savefig(plots_dir / f"{term}_method_comparison.png", dpi=300, bbox_inches='tight')
        print(f"\\nComparison plot saved: ./comparison_plots/{term}_method_comparison.png")
        plt.show()
    
    def analyze_method_strengths(self):
        """Analyze when each method works best"""
        
        print(f"\\n{'='*70}")
        print("WHEN TO USE EACH METHOD")
        print(f"{'='*70}")
        
        analysis = {
            "Dataset Averaging": {
                "strengths": [
                    "Based on real engineer decisions",
                    "Works for terms with many examples",
                    "Reliable baseline approach",
                    "No external dependencies"
                ],
                "weaknesses": [
                    "Ignores audio context completely",
                    "May average out contradictory strategies",
                    "One-size-fits-all approach",
                    "Loses engineering expertise nuance"
                ],
                "best_for": "Terms with high agreement (low variance) among engineers"
            },
            
            "Adaptive Selection": {
                "strengths": [
                    "Considers input audio characteristics",
                    "Preserves engineering expertise",
                    "Adapts to different audio scenarios", 
                    "Higher confidence in context"
                ],
                "weaknesses": [
                    "Requires audio analysis",
                    "Limited to available examples",
                    "More complex implementation",
                    "Needs good clustering/selection logic"
                ],
                "best_for": "Terms with high variance where context matters"
            },
            
            "LLM Generation": {
                "strengths": [
                    "Can handle any semantic term",
                    "Incorporates broad audio knowledge",
                    "Can adapt to text descriptions",
                    "Works for novel/rare terms"
                ],
                "weaknesses": [
                    "Not based on real engineer data",
                    "May hallucinate parameters",
                    "Requires API access/cost",
                    "Harder to validate results"
                ],
                "best_for": "Novel terms not in dataset, creative exploration"
            }
        }
        
        for method, info in analysis.items():
            print(f"\\n{method}:")
            print("  Strengths:")
            for strength in info['strengths']:
                print(f"    + {strength}")
            print("  Weaknesses:")  
            for weakness in info['weaknesses']:
                print(f"    - {weakness}")
            print(f"  Best for: {info['best_for']}")


def main():
    """Run comparison demo"""
    
    print("LLM vs Dataset vs Adaptive EQ Generation Demo")
    print("=" * 55)
    print("This demo compares different approaches without requiring API keys")
    
    demo = EQMethodDemo()
    
    # Test different scenarios
    scenarios = [
        {
            'term': 'warm',
            'audio_type': 'bright_harsh',
            'description': 'bright and harsh mix needing warmth'
        },
        {
            'term': 'bright', 
            'audio_type': 'dark_muddy',
            'description': 'dark and muddy mix needing brightness'
        },
        {
            'term': 'heavy',
            'audio_type': 'thin',
            'description': 'thin mix lacking weight'
        }
    ]
    
    for scenario in scenarios:
        results = demo.compare_all_methods(
            scenario['term'], 
            scenario['audio_type'],
            scenario['description']
        )
        
        try:
            demo.visualize_comparison(results, scenario['term'])
        except Exception as e:
            print(f"Visualization error: {e}")
    
    # Show method analysis
    demo.analyze_method_strengths()
    
    print(f"\\n{'='*70}")
    print("KEY INSIGHTS")
    print(f"{'='*70}")
    print("1. Dataset averaging works well for consistent terms")
    print("2. Adaptive selection shines when context matters")  
    print("3. LLMs provide flexibility for novel terms")
    print("4. Hybrid approaches could combine strengths")
    print("\\nThe best approach depends on:")
    print("  - Term popularity in dataset")
    print("  - Parameter variance across examples")
    print("  - Available audio context")
    print("  - Use case (reliable vs creative)")


if __name__ == '__main__':
    main()