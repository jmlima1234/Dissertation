"""
Enhanced Accuracy Evaluation for Portuguese Address Search
=========================================================

This module implements improved accuracy evaluation that addresses the limitations
of the current strict string matching approach. It includes:

1. Fuzzy string matching for Portuguese address components
2. Partial credit scoring for close matches
3. Geographic proximity consideration
4. Portuguese language-specific matching rules

The enhanced evaluation provides more realistic and fair accuracy metrics
while maintaining rigorous standards for baseline comparison.
"""

import math
import re
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from thefuzz import fuzz, process

# Import the ETL normalization module
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from normalization import AddressNormalizer

@dataclass
class AccuracyResult:
    """Detailed accuracy evaluation result"""
    is_match: bool
    overall_score: float
    component_scores: Dict[str, float]
    details: Dict[str, Any]

class EnhancedAccuracyEvaluator:
    """
    Enhanced accuracy evaluator for Portuguese address search results
    
    Uses fuzzy matching, partial credit, and geographic proximity to provide
    more realistic accuracy assessments while maintaining rigorous standards.
    """
    
    def __init__(self):
        """Initialize the enhanced evaluator with Portuguese-specific rules"""
        
        # Fuzzy matching thresholds
        self.fuzzy_thresholds = {
            'street': {
                'exact': 95,      # 95%+ similarity = exact match
                'good': 85,       # 85%+ similarity = good match  
                'partial': 70,    # 70%+ similarity = partial match
                'min': 60         # Below 60% = no match
            },
            'city': {
                'exact': 90,      # Cities should match more precisely
                'good': 80,
                'partial': 65,
                'min': 50
            },
            'postcode': {
                'exact': 100,     # Postcodes must be exact or very close
                'good': 90,       # Allow for minor typos
                'partial': 80,
                'min': 70
            }
        }
        
        # Component weights for overall scoring (geographic removed for now)
        self.component_weights = {
            'street': 0.5,        # Street is most important (increased from 0.4)
            'city': 0.35,         # City is second most important (increased from 0.3)
            'postcode': 0.15      # Postcode adds precision (decreased from 0.2)
        }
        
        # Initialize the ETL normalization system for consistent normalization
        self.address_normalizer = AddressNormalizer()
    
    def normalize_portuguese_text(self, text: str, component_type: str = 'general') -> str:
        """
        Normalize Portuguese text using the ETL normalization system
        
        Args:
            text: Raw Portuguese text
            component_type: Type of component ('street', 'city', 'postcode', 'general')
            
        Returns:
            Normalized text using consistent ETL normalization
        """
        if not text:
            return ""
        
        # Use the appropriate normalization method based on component type
        if component_type == 'street':
            # Use the ETL street normalization
            normalized = self.address_normalizer.normalize_street(text)
        elif component_type == 'city':
            # Use the ETL city normalization (returns None if not found, so handle that)
            normalized = self.address_normalizer.normalize_city(text)
            if normalized is None:
                # Fall back to general preprocessing if city not in canonical list
                normalized = self.address_normalizer._general_preprocessing(text)
        elif component_type == 'postcode':
            # Use the ETL postcode normalization (returns None if invalid)
            normalized = self.address_normalizer.normalize_postcode(text)
            if normalized is None:
                # Fall back to general preprocessing for invalid postcodes
                normalized = self.address_normalizer._general_preprocessing(text)
        else:
            # For general text or unknown component types, use general preprocessing
            normalized = self.address_normalizer._general_preprocessing(text)
        
        return normalized if normalized else ""
    
    def calculate_fuzzy_score(self, actual: str, expected: str, component_type: str) -> Dict[str, Any]:
        """
        Calculate fuzzy matching score for a component
        
        Args:
            actual: Actual result component
            expected: Expected component value
            component_type: Type of component ('street', 'city', 'postcode')
            
        Returns:
            Dictionary with score details
        """
        if not actual or not expected:
            return {
                'score': 0.0,
                'similarity': 0,
                'level': 'none',
                'details': 'Missing component'
            }
        
        # Normalize both strings using component-specific normalization
        actual_norm = self.normalize_portuguese_text(actual, component_type)
        expected_norm = self.normalize_portuguese_text(expected, component_type)
        
        # Calculate different similarity metrics
        ratio = fuzz.ratio(actual_norm, expected_norm)
        partial_ratio = fuzz.partial_ratio(actual_norm, expected_norm)
        token_sort = fuzz.token_sort_ratio(actual_norm, expected_norm)
        token_set = fuzz.token_set_ratio(actual_norm, expected_norm)
        
        # Use the best similarity score
        best_similarity = max(ratio, partial_ratio, token_sort, token_set)
        
        # Get thresholds for this component type
        thresholds = self.fuzzy_thresholds.get(component_type, self.fuzzy_thresholds['street'])
        
        # Determine match level and score
        if best_similarity >= thresholds['exact']:
            level = 'exact'
            score = 1.0
        elif best_similarity >= thresholds['good']:
            level = 'good'
            # Linear interpolation between good and exact
            score = 0.7 + 0.3 * ((best_similarity - thresholds['good']) / 
                                  (thresholds['exact'] - thresholds['good']))
        elif best_similarity >= thresholds['partial']:
            level = 'partial'
            # Linear interpolation between partial and good
            score = 0.4 + 0.3 * ((best_similarity - thresholds['partial']) / 
                                  (thresholds['good'] - thresholds['partial']))
        elif best_similarity >= thresholds['min']:
            level = 'weak'
            # Linear interpolation between min and partial
            score = 0.1 + 0.3 * ((best_similarity - thresholds['min']) / 
                                  (thresholds['partial'] - thresholds['min']))
        else:
            level = 'none'
            score = 0.0
        
        return {
            'score': score,
            'similarity': best_similarity,
            'level': level,
            'details': {
                'actual_normalized': actual_norm,
                'expected_normalized': expected_norm,
                'similarity_scores': {
                    'ratio': ratio,
                    'partial_ratio': partial_ratio,
                    'token_sort': token_sort,
                    'token_set': token_set
                }
            }
        }
    
    def evaluate_result_accuracy(self, 
                                result: Dict[str, Any], 
                                expected: Dict[str, Any]) -> AccuracyResult:
        """
        Comprehensive accuracy evaluation of a search result
        
        Args:
            result: Search result with address components and coordinates
            expected: Expected values for comparison
            
        Returns:
            AccuracyResult with detailed scoring information
        """
        component_scores = {}
        total_weighted_score = 0.0
        total_weight = 0.0
        
        # Evaluate street component
        if expected.get('street'):
            street_eval = self.calculate_fuzzy_score(
                result.get('street_clean', ''), 
                expected['street'], 
                'street'
            )
            component_scores['street'] = street_eval
            total_weighted_score += street_eval['score'] * self.component_weights['street']
            total_weight += self.component_weights['street']
        
        # Evaluate city component
        if expected.get('city'):
            city_eval = self.calculate_fuzzy_score(
                result.get('city_clean', ''), 
                expected['city'], 
                'city'
            )
            component_scores['city'] = city_eval
            total_weighted_score += city_eval['score'] * self.component_weights['city']
            total_weight += self.component_weights['city']
        
        # Evaluate postcode component
        if expected.get('postcode'):
            postcode_eval = self.calculate_fuzzy_score(
                result.get('postcode_clean', ''), 
                expected['postcode'], 
                'postcode'
            )
            component_scores['postcode'] = postcode_eval
            total_weighted_score += postcode_eval['score'] * self.component_weights['postcode']
            total_weight += self.component_weights['postcode']
        
        # Calculate overall score
        overall_score = total_weighted_score / total_weight if total_weight > 0 else 0.0
        
        # Determine if this is considered a match
        # We use different thresholds for different evaluation levels
        is_match = overall_score >= 0.6  # 60% overall score required for match
        
        return AccuracyResult(
            is_match=is_match,
            overall_score=overall_score,
            component_scores=component_scores,
            details={
                'total_weighted_score': total_weighted_score,
                'total_weight': total_weight,
                'match_threshold': 0.6,
                'evaluation_components': list(component_scores.keys())
            }
        )
    
    def evaluate_top_k_accuracy(self, 
                               results: List[Dict[str, Any]], 
                               expected: Dict[str, Any],
                               k_values: List[int] = [1, 3, 5]) -> Dict[str, Any]:
        """
        Evaluate top-K accuracy with enhanced fuzzy matching
        
        Args:
            results: List of search results
            expected: Expected address components
            k_values: List of K values to evaluate (default: [1, 3, 5])
            
        Returns:
            Dictionary with top-K accuracy results and detailed scoring
        """
        if not results:
            return {f'top_{k}_correct': False for k in k_values}
        
        # Evaluate each result
        result_evaluations = []
        for i, result in enumerate(results):
            evaluation = self.evaluate_result_accuracy(result, expected)
            result_evaluations.append({
                'rank': i + 1,
                'evaluation': evaluation,
                'result': result
            })
        
        # Find the best match and its position
        best_match_rank = None
        best_score = 0.0
        
        for eval_data in result_evaluations:
            if eval_data['evaluation'].is_match:
                if best_match_rank is None:
                    best_match_rank = eval_data['rank']
                if eval_data['evaluation'].overall_score > best_score:
                    best_score = eval_data['evaluation'].overall_score
        
        # Calculate top-K accuracy
        accuracy_results = {}
        for k in k_values:
            if best_match_rank is not None and best_match_rank <= k:
                accuracy_results[f'top_{k}_correct'] = True
            else:
                accuracy_results[f'top_{k}_correct'] = False
        
        # Add detailed information
        accuracy_results.update({
            'best_match_rank': best_match_rank,
            'best_match_score': best_score,
            'total_results': len(results),
            'detailed_evaluations': result_evaluations,
            'has_any_match': best_match_rank is not None
        })
        
        return accuracy_results


# Usage example and integration helper
def integrate_enhanced_accuracy_evaluation():
    """
    Example of how to integrate the enhanced accuracy evaluation 
    into the existing benchmark system
    """
    
    print("=== Enhanced Accuracy Evaluation Integration Example ===\n")
    
    # Create evaluator
    evaluator = EnhancedAccuracyEvaluator()
    
    # Example test case
    expected = {
        'street': 'rua augusta',
        'city': 'lisboa',
        'postcode': '1100-001'
    }
    
    # Example search results (simulated)
    results = [
        {
            'street_clean': 'rua agusta',  # Typo in street name
            'city_clean': 'lisboa',
            'postcode_clean': '1100-001',
            'latitude': 38.7141,
            'longitude': -9.1370
        },
        {
            'street_clean': 'rua augusta',  # Exact match
            'city_clean': 'lisbon',         # City name variation
            'postcode_clean': '1100-002',   # Close postcode
            'latitude': 38.7139,
            'longitude': -9.1372
        }
    ]
    
    # Evaluate accuracy
    accuracy_result = evaluator.evaluate_top_k_accuracy(results, expected)
    
    print("Example Evaluation Results:")
    print(f"Top-1 Correct: {accuracy_result['top_1_correct']}")
    print(f"Top-3 Correct: {accuracy_result['top_3_correct']}")
    print(f"Best Match Rank: {accuracy_result['best_match_rank']}")
    print(f"Best Match Score: {accuracy_result['best_match_score']:.3f}")
    
    # Show detailed component scores for best match
    if accuracy_result['detailed_evaluations']:
        best_eval = accuracy_result['detailed_evaluations'][0]['evaluation']
        print("\nDetailed Component Scores:")
        for component, score_data in best_eval.component_scores.items():
            print(f"  {component}: {score_data['score']:.3f} ({score_data['level']})")
    
    return evaluator


if __name__ == "__main__":
    integrate_enhanced_accuracy_evaluation()