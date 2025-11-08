"""
Simple Demo: Why Averaging EQ Parameters Can Be Problematic
===========================================================

Shows the key issue with averaging multiple EQ examples for semantic terms.
"""

import numpy as np
import matplotlib.pyplot as plt

def demo_eq_averaging_problem():
    """
    Demonstrate why simple averaging can lose important information
    """
    
    # Simulate 5 different "warm" EQ curves from engineers
    # Each engineer had different audio and used different "warm" strategies
    
    print("DEMO: Why Simple EQ Averaging Can Be Problematic")
    print("="*60)
    
    # Simplified 6-band EQ (gain values in dB)
    # Bands: Sub(60Hz), Bass(200Hz), LowMid(500Hz), Mid(1k), HighMid(3k), Treble(8k)
    
    warm_examples = {
        "Engineer 1 - Dark Rock Mix": [2.0, 1.5, 0.0, -0.5, -1.0, -2.0],  # Boost lows, cut highs
        "Engineer 2 - Thin Vocal":   [1.0, 2.5, 1.0,  0.5, -0.5, -1.0],  # More mid boost
        "Engineer 3 - Bright Pop":   [0.5, 0.5, 0.0, -1.0, -2.0, -3.0],  # Aggressive high cut
        "Engineer 4 - Bass-heavy":   [3.0, 2.0, 0.0,  0.0, -0.5, -1.0],  # Strong low boost
        "Engineer 5 - Gentle":       [0.5, 0.5, 0.5,  0.0,  0.0, -0.5]   # Very subtle
    }
    
    band_names = ['Sub\n60Hz', 'Bass\n200Hz', 'LowMid\n500Hz', 'Mid\n1kHz', 'HighMid\n3kHz', 'Treble\n8kHz']
    
    # Convert to array for analysis
    examples_array = np.array(list(warm_examples.values()))
    
    # Current system: Simple average
    averaged_eq = np.mean(examples_array, axis=0)
    
    # Analysis
    std_dev = np.std(examples_array, axis=0)
    min_vals = np.min(examples_array, axis=0)
    max_vals = np.max(examples_array, axis=0)
    
    print(f"\\nAnalyzing {len(warm_examples)} 'warm' EQ examples:")
    print("-" * 60)
    
    for i, band in enumerate(band_names):
        print(f"{band:12}: Avg={averaged_eq[i]:+5.1f}dB  Range=[{min_vals[i]:+4.1f}, {max_vals[i]:+4.1f}]  Std={std_dev[i]:.1f}")
    
    # Identify problems
    high_variance_bands = np.where(std_dev > 1.0)[0]
    
    print(f"\\nPROBLEMS with simple averaging:")
    print("-" * 40)
    print(f"• {len(high_variance_bands)}/{len(band_names)} bands have high variance (>1dB)")
    
    for band_idx in high_variance_bands:
        band = band_names[band_idx].split('\\n')[0]
        print(f"  - {band}: Engineers disagreed by {max_vals[band_idx] - min_vals[band_idx]:.1f}dB")
    
    print(f"\\n• Averaged EQ might not match ANY real engineer's decision")
    print(f"• Context of original audio is completely lost")
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Individual examples vs average
    x = np.arange(len(band_names))
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    
    for i, (engineer, eq_curve) in enumerate(warm_examples.items()):
        ax1.plot(x, eq_curve, 'o-', color=colors[i], alpha=0.7, 
                label=engineer, linewidth=2, markersize=6)
    
    ax1.plot(x, averaged_eq, 'k-', linewidth=4, marker='s', markersize=8,
            label='AVERAGED (Current System)', alpha=0.8)
    
    ax1.set_title('Individual "Warm" EQ Examples vs Simple Average', fontweight='bold')
    ax1.set_xlabel('Frequency Band')
    ax1.set_ylabel('Gain (dB)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(band_names)
    ax1.grid(True, alpha=0.3)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    # Plot 2: Variance analysis
    ax2.bar(x, std_dev, color='red', alpha=0.7)
    ax2.axhline(y=1.0, color='orange', linestyle='--', linewidth=2, 
               label='High Variance Threshold (1dB)')
    ax2.set_title('Parameter Variance Across "Warm" Examples', fontweight='bold')
    ax2.set_xlabel('Frequency Band')
    ax2.set_ylabel('Standard Deviation (dB)')
    ax2.set_xticks(x)
    ax2.set_xticklabels(band_names)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Add variance labels
    for i, v in enumerate(std_dev):
        if v > 1.0:
            ax2.text(i, v + 0.1, f'{v:.1f}dB\\nHIGH', ha='center', va='bottom', 
                    fontweight='bold', color='red')
        else:
            ax2.text(i, v + 0.05, f'{v:.1f}dB', ha='center', va='bottom')
    
    plt.tight_layout()
    
    # Save plot
    plots_dir = Path("./demo_plots")
    plots_dir.mkdir(exist_ok=True)
    plt.savefig(plots_dir / "eq_averaging_problem_demo.png", dpi=300, bbox_inches='tight')
    print(f"\\nDemo plot saved: ./demo_plots/eq_averaging_problem_demo.png")
    plt.show()
    
    return examples_array, averaged_eq


def demo_better_approaches():
    """
    Show how we could do better than simple averaging
    """
    
    print("\\n" + "="*60)
    print("BETTER APPROACHES")
    print("="*60)
    
    # Same example data
    warm_examples = {
        "Engineer 1 - Dark Rock Mix": [2.0, 1.5, 0.0, -0.5, -1.0, -2.0],
        "Engineer 2 - Thin Vocal":   [1.0, 2.5, 1.0,  0.5, -0.5, -1.0], 
        "Engineer 3 - Bright Pop":   [0.5, 0.5, 0.0, -1.0, -2.0, -3.0],
        "Engineer 4 - Bass-heavy":   [3.0, 2.0, 0.0,  0.0, -0.5, -1.0],
        "Engineer 5 - Gentle":       [0.5, 0.5, 0.5,  0.0,  0.0, -0.5]
    }
    
    examples_array = np.array(list(warm_examples.values()))
    
    # Approach 1: Audio-informed selection
    print("\\n1. AUDIO-INFORMED SELECTION")
    print("-" * 40)
    
    audio_scenarios = {
        "Bright, thin mix": "Engineer 3 - Bright Pop",     # Needs aggressive high cut
        "Dark, muddy mix": "Engineer 2 - Thin Vocal",      # Needs clarity + warmth
        "Well-balanced mix": "Engineer 5 - Gentle",        # Needs subtle enhancement
        "Bass-heavy mix": "Engineer 1 - Dark Rock Mix"     # Needs controlled warmth
    }
    
    for audio_type, best_engineer in audio_scenarios.items():
        eq_curve = warm_examples[best_engineer]
        print(f"  {audio_type:18} → {best_engineer}")
        print(f"    EQ: {eq_curve}")
    
    # Approach 2: Clustering
    print("\\n2. CLUSTERING-BASED SELECTION")
    print("-" * 40)
    
    # Simple clustering by strategy
    clusters = {
        "Aggressive warmth": ["Engineer 1 - Dark Rock Mix", "Engineer 3 - Bright Pop"],
        "Balanced warmth":   ["Engineer 2 - Thin Vocal", "Engineer 4 - Bass-heavy"],
        "Subtle warmth":     ["Engineer 5 - Gentle"]
    }
    
    for cluster_name, engineers in clusters.items():
        cluster_eqs = [warm_examples[eng] for eng in engineers]
        cluster_avg = np.mean(cluster_eqs, axis=0)
        print(f"  {cluster_name:18} → {len(engineers)} examples")
        print(f"    Avg EQ: {cluster_avg}")
    
    # Approach 3: Variance-weighted selection
    print("\\n3. VARIANCE-WEIGHTED SELECTION") 
    print("-" * 40)
    
    std_dev = np.std(examples_array, axis=0)
    
    # Weight by inverse variance (more weight to consistent parameters)
    weights = 1.0 / (std_dev + 0.1)  # Add small constant to avoid division by zero
    weights = weights / np.sum(weights)  # Normalize
    
    # Weighted average
    weighted_avg = np.average(examples_array, axis=0, weights=np.tile(weights, (len(examples_array), 1)).T)
    simple_avg = np.mean(examples_array, axis=0)
    
    print(f"  Parameter weights (higher = more reliable):")
    band_names = ['Sub', 'Bass', 'LowMid', 'Mid', 'HighMid', 'Treble']
    for i, (band, weight) in enumerate(zip(band_names, weights)):
        print(f"    {band:8}: {weight:.3f}")
    
    print(f"\\n  Simple avg:   {simple_avg}")
    print(f"  Weighted avg: {weighted_avg}")
    
    print(f"\\nKEY INSIGHT:")
    print(f"The choice of EQ curve should depend on:")
    print(f"  • Input audio characteristics (bright vs dark, thin vs full)")
    print(f"  • Reliability of parameters (some are more consistent)")
    print(f"  • Musical context and genre expectations")


def main():
    from pathlib import Path
    
    print("Understanding the EQ Selection Problem")
    print("="*60)
    print("This demo shows why simple averaging of EQ parameters")
    print("can lose important engineering decisions and context.")
    print("="*60)
    
    # Run demos
    examples_array, averaged_eq = demo_eq_averaging_problem()
    demo_better_approaches()
    
    print(f"\\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print("Current system: Simple averaging across all examples")
    print("→ Loses context and engineering expertise")
    print("\\nBetter approaches:")
    print("1. Analyze input audio → Select appropriate example")
    print("2. Cluster examples by strategy → Choose best cluster") 
    print("3. Weight by parameter reliability → Smart averaging")
    print("\\nThis is what the adaptive_semantic_mastering.py implements!")


if __name__ == '__main__':
    main()