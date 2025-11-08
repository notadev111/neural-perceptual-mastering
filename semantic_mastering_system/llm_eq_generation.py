"""
LLM-Based EQ Parameter Generation
=================================

Compare LLM-generated EQ parameters with dataset-based approaches.
Implements the LLM2FX methodology from the original paper.

Comparison methods:
1. Dataset averaging (current)
2. Audio-informed selection (adaptive) 
3. LLM generation (GPT/Claude/local models)
4. Hybrid: LLM + dataset validation

Usage:
    python llm_eq_generation.py --term warm --llm gpt-4
    python llm_eq_generation.py --audio mix.wav --term bright --compare-all-methods
"""

import torch
import torchaudio
import numpy as np
import json
import requests
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import warnings
warnings.filterwarnings("ignore")

# Import our existing systems
try:
    from semantic_mastering import SocialFXDataLoader, EQProfile, SemanticMasteringEQ
    from adaptive_semantic_mastering import AdaptiveSemanticMasteringEQ
    SYSTEMS_AVAILABLE = True
except ImportError:
    print("Warning: Base systems not available")
    SYSTEMS_AVAILABLE = False


@dataclass
class LLMEQResponse:
    """Container for LLM-generated EQ parameters"""
    term: str
    llm_model: str
    raw_response: str
    extracted_params: Optional[np.ndarray]
    confidence: float
    reasoning: str
    processing_time: float


class LLMEQGenerator:
    """
    Generate EQ parameters using LLMs following LLM2FX methodology
    """
    
    def __init__(self):
        self.api_keys = self._load_api_keys()
        
    def _load_api_keys(self) -> Dict[str, str]:
        """Load API keys from environment or config"""
        import os
        
        keys = {
            'openai': os.getenv('OPENAI_API_KEY'),
            'anthropic': os.getenv('ANTHROPIC_API_KEY'),
            'huggingface': os.getenv('HUGGINGFACE_API_KEY')
        }
        
        # Check for config file
        config_file = Path('llm_config.json')
        if config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    config = json.load(f)
                    keys.update(config.get('api_keys', {}))
            except:
                pass
        
        return keys
    
    def generate_eq_with_gpt(self, term: str, audio_description: str = None) -> LLMEQResponse:
        """Generate EQ parameters using GPT models"""
        
        if not self.api_keys.get('openai'):
            return self._create_fallback_response(term, "No OpenAI API key")
        
        # Construct prompt following LLM2FX methodology
        prompt = self._create_eq_prompt(term, audio_description)
        
        start_time = time.time()
        
        try:
            import openai
            openai.api_key = self.api_keys['openai']
            
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an expert audio mastering engineer. Generate precise EQ parameters based on semantic descriptions."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.3  # Lower temperature for more consistent results
            )
            
            raw_response = response.choices[0].message.content
            processing_time = time.time() - start_time
            
            # Extract EQ parameters from response
            eq_params, confidence = self._parse_llm_response(raw_response)
            
            return LLMEQResponse(
                term=term,
                llm_model="gpt-4",
                raw_response=raw_response,
                extracted_params=eq_params,
                confidence=confidence,
                reasoning=f"GPT-4 generated EQ for '{term}'",
                processing_time=processing_time
            )
            
        except Exception as e:
            return self._create_fallback_response(term, f"GPT error: {e}")
    
    def generate_eq_with_claude(self, term: str, audio_description: str = None) -> LLMEQResponse:
        """Generate EQ parameters using Claude"""
        
        if not self.api_keys.get('anthropic'):
            return self._create_fallback_response(term, "No Anthropic API key")
        
        prompt = self._create_eq_prompt(term, audio_description)
        
        start_time = time.time()
        
        try:
            import anthropic
            
            client = anthropic.Anthropic(api_key=self.api_keys['anthropic'])
            
            response = client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=500,
                temperature=0.3,
                messages=[{
                    "role": "user", 
                    "content": f"You are an expert audio mastering engineer. {prompt}"
                }]
            )
            
            raw_response = response.content[0].text
            processing_time = time.time() - start_time
            
            eq_params, confidence = self._parse_llm_response(raw_response)
            
            return LLMEQResponse(
                term=term,
                llm_model="claude-3-sonnet",
                raw_response=raw_response,
                extracted_params=eq_params,
                confidence=confidence,
                reasoning=f"Claude generated EQ for '{term}'",
                processing_time=processing_time
            )
            
        except Exception as e:
            return self._create_fallback_response(term, f"Claude error: {e}")
    
    def generate_eq_with_local_model(self, term: str, audio_description: str = None) -> LLMEQResponse:
        """Generate EQ using local/open-source models"""
        
        # This would use models like Mistral, Llama, etc.
        # For now, implement a rule-based fallback that simulates LLM reasoning
        
        start_time = time.time()
        
        eq_params = self._generate_rule_based_eq(term, audio_description)
        processing_time = time.time() - start_time
        
        return LLMEQResponse(
            term=term,
            llm_model="rule-based-fallback",
            raw_response=f"Generated {term} EQ using rule-based approach",
            extracted_params=eq_params,
            confidence=0.6,
            reasoning=f"Rule-based EQ generation for '{term}'",
            processing_time=processing_time
        )
    
    def _create_eq_prompt(self, term: str, audio_description: str = None) -> str:
        """Create LLM prompt following LLM2FX methodology"""
        
        base_prompt = f"""
Generate precise EQ parameters for making audio sound "{term}".

Requirements:
- Provide 6-band parametric EQ settings
- Format: [gain1, freq1, Q1, gain2, freq2, Q2, gain3, freq3, Q3, gain4, freq4, Q4, gain5, freq5, Q5, gain6, freq6, Q6]
- Gains in dB (-12 to +12)
- Frequencies in Hz (20 to 20000)  
- Q values (0.1 to 10.0)
- Focus on mastering context (subtle adjustments, typically ±2-3dB)

Semantic term: "{term}"
"""
        
        if audio_description:
            base_prompt += f"\nInput audio characteristics: {audio_description}"
        
        base_prompt += f"""

Example format:
[+1.5, 200, 0.7, -0.5, 500, 1.2, +2.0, 1000, 0.9, -1.0, 3000, 1.5, +0.8, 8000, 0.6, -0.3, 12000, 0.8]

Explain your reasoning and provide the parameter array.
"""
        
        return base_prompt
    
    def _parse_llm_response(self, response: str) -> Tuple[Optional[np.ndarray], float]:
        """Extract EQ parameters from LLM response"""
        
        import re
        
        # Look for parameter arrays in various formats
        patterns = [
            r'\[([-+0-9.,\s]+)\]',  # [1.5, 200, 0.7, ...]
            r'Parameters?:\s*([-+0-9.,\s]+)',  # Parameters: 1.5, 200, 0.7
            r'EQ:\s*([-+0-9.,\s]+)',  # EQ: 1.5, 200, 0.7
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, response)
            if matches:
                try:
                    # Parse the first match
                    param_str = matches[0]
                    # Clean up the string
                    param_str = re.sub(r'[^\d.,+-]', ' ', param_str)
                    params = [float(x.strip()) for x in param_str.split(',') if x.strip()]
                    
                    if len(params) >= 18:  # 6 bands × 3 params
                        return np.array(params[:18]), 0.8
                    elif len(params) >= 6:  # Just gains
                        # Assume standard frequencies and Q values
                        full_params = []
                        standard_freqs = [60, 200, 500, 1200, 3000, 8000]
                        standard_qs = [0.7, 0.7, 1.0, 1.2, 1.0, 0.8]
                        
                        for i in range(6):
                            if i < len(params):
                                full_params.extend([params[i], standard_freqs[i], standard_qs[i]])
                            else:
                                full_params.extend([0.0, standard_freqs[i], standard_qs[i]])
                        
                        return np.array(full_params), 0.6
                except:
                    continue
        
        # If no valid parameters found, return None
        return None, 0.0
    
    def _generate_rule_based_eq(self, term: str, audio_description: str = None) -> np.ndarray:
        """Generate EQ using rule-based approach (fallback)"""
        
        # Standard 6-band setup
        # [gain1, freq1, Q1, gain2, freq2, Q2, ...]
        standard_freqs = [60, 200, 500, 1200, 3000, 8000]
        standard_qs = [0.7, 0.8, 1.0, 1.2, 1.0, 0.8]
        
        # Rule-based EQ curves for common terms
        eq_rules = {
            'warm': [1.0, 0.5, 0.0, -0.5, -1.0, -0.5],  # Boost lows, cut highs
            'bright': [-0.5, 0.0, 0.5, 1.0, 1.5, 1.0],  # Cut lows, boost highs
            'heavy': [2.0, 1.5, 0.5, 0.0, -0.5, -0.5],  # Strong low boost
            'soft': [0.5, 0.0, -0.5, -1.0, -1.5, -1.0], # Gentle high cut
            'punchy': [0.5, 0.5, 1.5, 0.5, 0.0, -0.5],  # Mid boost
            'smooth': [0.0, 0.0, -0.5, -0.5, -0.5, 0.0], # Gentle smoothing
            'aggressive': [0.0, 1.0, 2.0, 1.0, 0.5, 0.0], # Mid/high boost
            'cool': [-1.0, -0.5, 0.0, 0.5, 1.0, 0.5],   # Opposite of warm
        }
        
        # Get base curve or neutral
        gains = eq_rules.get(term.lower(), [0.0] * 6)
        
        # Build full parameter array
        params = []
        for i in range(6):
            params.extend([gains[i], standard_freqs[i], standard_qs[i]])
        
        return np.array(params)
    
    def _create_fallback_response(self, term: str, error: str) -> LLMEQResponse:
        """Create fallback response when LLM fails"""
        
        fallback_params = self._generate_rule_based_eq(term)
        
        return LLMEQResponse(
            term=term,
            llm_model="fallback",
            raw_response=f"Fallback due to: {error}",
            extracted_params=fallback_params,
            confidence=0.3,
            reasoning=f"Fallback EQ for '{term}' due to LLM error",
            processing_time=0.0
        )


class ComprehensiveEQComparator:
    """
    Compare all EQ generation methods: dataset, adaptive, and LLM
    """
    
    def __init__(self):
        if SYSTEMS_AVAILABLE:
            self.dataset_system = SemanticMasteringEQ()
            self.adaptive_system = AdaptiveSemanticMasteringEQ() 
            self.dataset_system.initialize()
        
        self.llm_generator = LLMEQGenerator()
    
    def compare_all_methods(self, term: str, audio_path: str = None, 
                          audio_description: str = None) -> Dict:
        """
        Compare all EQ generation methods for a semantic term
        """
        
        print(f"\\n{'='*70}")
        print(f"COMPREHENSIVE EQ METHOD COMPARISON: {term.upper()}")
        print(f"{'='*70}")
        
        results = {}
        
        # Method 1: Dataset averaging (current)
        if SYSTEMS_AVAILABLE:
            print("\\n1. Dataset Averaging (Current Method)")
            try:
                dataset_profile = self.dataset_system.loader.get_profile(term)
                results['dataset_avg'] = {
                    'method': 'Dataset Averaging',
                    'profile': dataset_profile,
                    'confidence': dataset_profile.confidence,
                    'reasoning': dataset_profile.reasoning,
                    'source': f"{dataset_profile.n_examples} real examples"
                }
                print(f"   ✓ {dataset_profile.n_examples} examples, confidence: {dataset_profile.confidence:.1%}")
            except Exception as e:
                print(f"   ❌ Error: {e}")
        
        # Method 2: Adaptive selection
        if SYSTEMS_AVAILABLE and audio_path:
            print("\\n2. Audio-Informed Adaptive Selection")
            try:
                audio, _ = torchaudio.load(audio_path)
                _, adaptive_profile = self.adaptive_system.apply_adaptive_mastering(audio, term)
                results['adaptive'] = {
                    'method': 'Adaptive Selection',
                    'profile': adaptive_profile.base_profile,
                    'confidence': adaptive_profile.selection_confidence,
                    'reasoning': adaptive_profile.base_profile.reasoning,
                    'source': f"Selected from dataset based on audio analysis"
                }
                print(f"   ✓ Audio-informed selection, confidence: {adaptive_profile.selection_confidence:.1%}")
            except Exception as e:
                print(f"   ❌ Error: {e}")
        
        # Method 3: LLM Generation (GPT-4)
        print("\\n3. LLM Generation (GPT-4)")
        try:
            llm_response = self.llm_generator.generate_eq_with_gpt(term, audio_description)
            if llm_response.extracted_params is not None:
                # Convert to our EQProfile format
                dasp_params, confidence = self._convert_llm_to_dasp(llm_response.extracted_params, term)
                
                llm_profile = EQProfile(
                    name=f"{term}_llm",
                    params_dasp=dasp_params,
                    params_original=llm_response.extracted_params,
                    reasoning=llm_response.reasoning,
                    n_examples=1,
                    confidence=llm_response.confidence
                )
                
                results['llm_gpt4'] = {
                    'method': 'LLM (GPT-4)',
                    'profile': llm_profile,
                    'confidence': llm_response.confidence,
                    'reasoning': llm_response.reasoning,
                    'source': f"Generated by {llm_response.llm_model}",
                    'processing_time': llm_response.processing_time,
                    'raw_response': llm_response.raw_response[:200] + "..." if len(llm_response.raw_response) > 200 else llm_response.raw_response
                }
                print(f"   ✓ Generated in {llm_response.processing_time:.1f}s, confidence: {llm_response.confidence:.1%}")
            else:
                print(f"   ❌ Failed to extract valid parameters")
        except Exception as e:
            print(f"   ❌ Error: {e}")
        
        # Method 4: LLM Generation (Claude)
        print("\\n4. LLM Generation (Claude)")
        try:
            llm_response = self.llm_generator.generate_eq_with_claude(term, audio_description)
            if llm_response.extracted_params is not None:
                dasp_params, confidence = self._convert_llm_to_dasp(llm_response.extracted_params, term)
                
                llm_profile = EQProfile(
                    name=f"{term}_claude",
                    params_dasp=dasp_params,
                    params_original=llm_response.extracted_params,
                    reasoning=llm_response.reasoning,
                    n_examples=1,
                    confidence=llm_response.confidence
                )
                
                results['llm_claude'] = {
                    'method': 'LLM (Claude)',
                    'profile': llm_profile,
                    'confidence': llm_response.confidence,
                    'reasoning': llm_response.reasoning,
                    'source': f"Generated by {llm_response.llm_model}",
                    'processing_time': llm_response.processing_time
                }
                print(f"   ✓ Generated in {llm_response.processing_time:.1f}s, confidence: {llm_response.confidence:.1%}")
            else:
                print(f"   ❌ Failed to extract valid parameters")
        except Exception as e:
            print(f"   ❌ Error: {e}")
        
        # Method 5: Rule-based (control)
        print("\\n5. Rule-Based (Control)")
        try:
            rule_response = self.llm_generator.generate_eq_with_local_model(term, audio_description)
            dasp_params, confidence = self._convert_llm_to_dasp(rule_response.extracted_params, term)
            
            rule_profile = EQProfile(
                name=f"{term}_rule",
                params_dasp=dasp_params,
                params_original=rule_response.extracted_params,
                reasoning="Rule-based EQ generation",
                n_examples=1,
                confidence=0.5
            )
            
            results['rule_based'] = {
                'method': 'Rule-Based',
                'profile': rule_profile,
                'confidence': 0.5,
                'reasoning': "Hand-crafted rules based on audio engineering knowledge",
                'source': "Pre-defined EQ curves"
            }
            print(f"   ✓ Rule-based generation, confidence: 50%")
        except Exception as e:
            print(f"   ❌ Error: {e}")
        
        return results
    
    def _convert_llm_to_dasp(self, llm_params: np.ndarray, term: str) -> Tuple[torch.Tensor, float]:
        """Convert LLM parameters to dasp-pytorch format"""
        
        if SYSTEMS_AVAILABLE:
            # Use the existing converter from the dataset system
            return self.dataset_system.loader._convert_to_dasp(llm_params, term)
        else:
            # Simple fallback conversion
            # Assume 6 bands with [gain, freq, Q] format
            dasp_params = torch.ones(1, 18) * 0.5
            
            for i in range(6):
                if i*3 + 2 < len(llm_params):
                    gain_db = llm_params[i*3]
                    gain_norm = 0.5 + (gain_db / 24.0)  # Normalize to [0,1]
                    dasp_params[0, i*3] = np.clip(gain_norm, 0, 1)
            
            return dasp_params, 0.7


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="LLM-Based EQ Parameter Generation")
    parser.add_argument('--term', default='warm', help='Semantic term')
    parser.add_argument('--audio', help='Audio file for adaptive comparison')
    parser.add_argument('--audio-desc', help='Text description of audio characteristics')
    parser.add_argument('--llm', choices=['gpt-4', 'claude', 'local'], default='gpt-4', help='LLM model')
    parser.add_argument('--compare-all-methods', action='store_true', help='Compare all methods')
    
    args = parser.parse_args()
    
    if args.compare_all_methods:
        comparator = ComprehensiveEQComparator()
        results = comparator.compare_all_methods(
            args.term, 
            args.audio, 
            args.audio_desc
        )
        
        # Show summary
        print(f"\\n{'='*70}")
        print("COMPARISON SUMMARY")
        print(f"{'='*70}")
        
        for method_name, method_data in results.items():
            print(f"\\n{method_data['method']:25}: {method_data['confidence']:.1%} confidence")
            print(f"{'':25}  {method_data['source']}")
    
    else:
        generator = LLMEQGenerator()
        
        if args.llm == 'gpt-4':
            response = generator.generate_eq_with_gpt(args.term, args.audio_desc)
        elif args.llm == 'claude':
            response = generator.generate_eq_with_claude(args.term, args.audio_desc)
        else:
            response = generator.generate_eq_with_local_model(args.term, args.audio_desc)
        
        print(f"\\nLLM EQ Generation for '{args.term}':")
        print(f"Model: {response.llm_model}")
        print(f"Confidence: {response.confidence:.1%}")
        print(f"Processing time: {response.processing_time:.2f}s")
        print(f"\\nReasoning: {response.reasoning}")
        print(f"\\nParameters: {response.extracted_params}")


if __name__ == '__main__':
    main()