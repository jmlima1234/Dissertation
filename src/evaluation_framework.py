"""
Evaluation Framework for Phase 2: Fuzzy Matching Algorithms

This module provides common evaluation functions and metrics for testing
different fuzzy matching algorithms on Portuguese address data.
"""

import time
import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Any, Callable
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


class FuzzyMatchingEvaluator:
    """
    Comprehensive evaluation framework for fuzzy matching algorithms.
    """
    
    def __init__(self, gold_standard_path: str = None):
        """
        Initialize the evaluator with gold standard data.
        
        Args:
            gold_standard_path: Path to the gold standard dataset
        """
        self.gold_standard_path = gold_standard_path
        self.gold_standard_df = None
        self.results_history = []
        
        if gold_standard_path:
            self.load_gold_standard()
    
    def load_gold_standard(self):
        """Load the gold standard dataset for evaluation."""
        try:
            self.gold_standard_df = pd.read_csv(self.gold_standard_path)
            print(f"Loaded gold standard with {len(self.gold_standard_df)} pairs")
        except Exception as e:
            print(f"Error loading gold standard: {e}")
    
    def create_test_pairs(self, addresses: List[str], n_positive: int = 100, 
                         n_negative: int = 100, seed: int = 42) -> Tuple[List[Tuple[str, str, int]], pd.DataFrame]:
        """
        Create test pairs for evaluation when no gold standard is available.
        
        Args:
            addresses: List of addresses to create pairs from
            n_positive: Number of positive pairs to create
            n_negative: Number of negative pairs to create
            seed: Random seed for reproducibility
            
        Returns:
            Tuple of (test_pairs, test_df)
        """
        np.random.seed(seed)
        
        positive_pairs = []
        negative_pairs = []
        
        # Create positive pairs (similar addresses)
        addresses_array = np.array(addresses)
        for _ in range(n_positive):
            # Select a random address and create a variation
            base_addr = np.random.choice(addresses_array)
            # Simple variation: add/remove spaces, change case
            variations = [
                base_addr.upper(),
                base_addr.lower(),
                base_addr.replace(' ', '  '),  # Double spaces
                base_addr.replace(',', ' ,'),   # Space before comma
                base_addr.strip()
            ]
            varied_addr = np.random.choice([v for v in variations if v != base_addr])
            positive_pairs.append((base_addr, varied_addr, 1))
        
        # Create negative pairs (different addresses)
        for _ in range(n_negative):
            addr1, addr2 = np.random.choice(addresses_array, 2, replace=False)
            negative_pairs.append((addr1, addr2, 0))
        
        all_pairs = positive_pairs + negative_pairs
        test_df = pd.DataFrame(all_pairs, columns=['address1', 'address2', 'is_match'])
        
        return all_pairs, test_df
    
    def evaluate_algorithm(self, matching_function: Callable, test_pairs: List[Tuple[str, str, int]], 
                          algorithm_name: str, threshold: float = 0.8, **kwargs) -> Dict[str, Any]:
        """
        Evaluate a fuzzy matching algorithm.
        
        Args:
            matching_function: Function that takes two addresses and returns similarity score
            test_pairs: List of (addr1, addr2, true_label) tuples
            algorithm_name: Name of the algorithm being tested
            threshold: Similarity threshold for classification
            **kwargs: Additional arguments for the matching function
            
        Returns:
            Dictionary containing evaluation metrics
        """
        print(f"Evaluating {algorithm_name}...")
        
        predictions = []
        similarities = []
        true_labels = []
        processing_times = []
        
        for addr1, addr2, true_label in test_pairs:
            start_time = time.time()
            
            try:
                similarity = matching_function(addr1, addr2, **kwargs)
                prediction = 1 if similarity >= threshold else 0
            except Exception as e:
                print(f"Error processing pair ({addr1}, {addr2}): {e}")
                similarity = 0
                prediction = 0
            
            end_time = time.time()
            
            predictions.append(prediction)
            similarities.append(similarity)
            true_labels.append(true_label)
            processing_times.append(end_time - start_time)
        
        # Calculate metrics
        metrics = self.calculate_metrics(true_labels, predictions, similarities)
        metrics['algorithm_name'] = algorithm_name
        metrics['threshold'] = threshold
        metrics['avg_processing_time'] = np.mean(processing_times)
        metrics['total_processing_time'] = sum(processing_times)
        
        # Handle zero processing time (very fast algorithms)
        total_time = sum(processing_times)
        if total_time > 0:
            metrics['pairs_per_second'] = len(test_pairs) / total_time
        else:
            # If total time is 0, estimate a very high processing rate
            metrics['pairs_per_second'] = len(test_pairs) * 1000000  # Assume microsecond precision
        
        # Store results
        self.results_history.append(metrics)
        
        print(f"Results for {algorithm_name}:")
        print(f"  F1-Score: {metrics['f1_score']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  Avg Processing Time: {metrics['avg_processing_time']:.6f}s")
        
        # Format pairs per second nicely
        pps = metrics['pairs_per_second']
        if pps >= 1000000:
            print(f"  Pairs/Second: {pps/1000000:.1f}M")
        elif pps >= 1000:
            print(f"  Pairs/Second: {pps/1000:.1f}K")
        else:
            print(f"  Pairs/Second: {pps:.2f}")
        
        return metrics
    
    def calculate_metrics(self, true_labels: List[int], predictions: List[int], 
                         similarities: List[float]) -> Dict[str, float]:
        """
        Calculate comprehensive evaluation metrics.
        
        Args:
            true_labels: Ground truth labels
            predictions: Predicted labels
            similarities: Raw similarity scores
            
        Returns:
            Dictionary of metrics
        """
        true_labels = np.array(true_labels)
        predictions = np.array(predictions)
        similarities = np.array(similarities)
        
        # Basic classification metrics
        tp = np.sum((predictions == 1) & (true_labels == 1))
        fp = np.sum((predictions == 1) & (true_labels == 0))
        fn = np.sum((predictions == 0) & (true_labels == 1))
        tn = np.sum((predictions == 0) & (true_labels == 0))
        
        # Calculate metrics with zero-division handling
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (tp + tn) / len(true_labels)
        
        # Additional metrics
        mean_similarity_positive = np.mean(similarities[true_labels == 1]) if np.any(true_labels == 1) else 0
        mean_similarity_negative = np.mean(similarities[true_labels == 0]) if np.any(true_labels == 0) else 0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'accuracy': accuracy,
            'true_positives': int(tp),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'true_negatives': int(tn),
            'mean_similarity_positive': mean_similarity_positive,
            'mean_similarity_negative': mean_similarity_negative,
            'similarity_separation': mean_similarity_positive - mean_similarity_negative
        }
    
    def threshold_analysis(self, matching_function: Callable, test_pairs: List[Tuple[str, str, int]], 
                          algorithm_name: str, thresholds: List[float] = None, **kwargs) -> pd.DataFrame:
        """
        Analyze performance across different similarity thresholds.
        
        Args:
            matching_function: Function that takes two addresses and returns similarity score
            test_pairs: List of (addr1, addr2, true_label) tuples
            algorithm_name: Name of the algorithm being tested
            thresholds: List of thresholds to test
            **kwargs: Additional arguments for the matching function
            
        Returns:
            DataFrame with metrics for each threshold
        """
        if thresholds is None:
            thresholds = np.arange(0.1, 1.0, 0.1)
        
        results = []
        
        # Pre-calculate similarities to avoid recomputation
        similarities = []
        true_labels = []
        
        for addr1, addr2, true_label in test_pairs:
            try:
                similarity = matching_function(addr1, addr2, **kwargs)
            except:
                similarity = 0
            similarities.append(similarity)
            true_labels.append(true_label)
        
        # Test each threshold
        for threshold in thresholds:
            predictions = [1 if sim >= threshold else 0 for sim in similarities]
            metrics = self.calculate_metrics(true_labels, predictions, similarities)
            metrics['threshold'] = threshold
            metrics['algorithm_name'] = algorithm_name
            results.append(metrics)
        
        return pd.DataFrame(results)
    
    def compare_algorithms(self, results_list: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Compare multiple algorithm results.
        
        Args:
            results_list: List of results dictionaries from evaluate_algorithm
            
        Returns:
            DataFrame comparing all algorithms
        """
        comparison_df = pd.DataFrame(results_list)
        
        # Sort by F1-score descending
        comparison_df = comparison_df.sort_values('f1_score', ascending=False)
        
        return comparison_df
    
    def plot_threshold_analysis(self, threshold_results: pd.DataFrame, save_path: str = None):
        """
        Plot threshold analysis results.
        
        Args:
            threshold_results: DataFrame from threshold_analysis
            save_path: Path to save the plot
        """
        plt.figure(figsize=(12, 8))
        
        # Plot metrics vs threshold
        plt.subplot(2, 2, 1)
        plt.plot(threshold_results['threshold'], threshold_results['precision'], 'b-o', label='Precision')
        plt.plot(threshold_results['threshold'], threshold_results['recall'], 'r-o', label='Recall')
        plt.plot(threshold_results['threshold'], threshold_results['f1_score'], 'g-o', label='F1-Score')
        plt.xlabel('Threshold')
        plt.ylabel('Score')
        plt.title('Metrics vs Threshold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot accuracy
        plt.subplot(2, 2, 2)
        plt.plot(threshold_results['threshold'], threshold_results['accuracy'], 'purple', marker='o')
        plt.xlabel('Threshold')
        plt.ylabel('Accuracy')
        plt.title('Accuracy vs Threshold')
        plt.grid(True, alpha=0.3)
        
        # Plot confusion matrix metrics
        plt.subplot(2, 2, 3)
        plt.plot(threshold_results['threshold'], threshold_results['true_positives'], 'g-o', label='True Positives')
        plt.plot(threshold_results['threshold'], threshold_results['false_positives'], 'r-o', label='False Positives')
        plt.plot(threshold_results['threshold'], threshold_results['false_negatives'], 'orange', marker='o', label='False Negatives')
        plt.xlabel('Threshold')
        plt.ylabel('Count')
        plt.title('Classification Counts vs Threshold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot similarity separation
        plt.subplot(2, 2, 4)
        plt.plot(threshold_results['threshold'], threshold_results['similarity_separation'], 'brown', marker='o')
        plt.xlabel('Threshold')
        plt.ylabel('Separation')
        plt.title('Similarity Separation (Positive - Negative)')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Threshold analysis plot saved to: {save_path}")
        
        plt.show()
    
    def plot_algorithm_comparison(self, comparison_df: pd.DataFrame, save_path: str = None):
        """
        Plot comparison of different algorithms.
        
        Args:
            comparison_df: DataFrame from compare_algorithms
            save_path: Path to save the plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        algorithms = comparison_df['algorithm_name']
        
        # F1-Score comparison
        axes[0, 0].barh(algorithms, comparison_df['f1_score'])
        axes[0, 0].set_xlabel('F1-Score')
        axes[0, 0].set_title('F1-Score Comparison')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Precision vs Recall
        axes[0, 1].scatter(comparison_df['recall'], comparison_df['precision'], s=100)
        for i, alg in enumerate(algorithms):
            axes[0, 1].annotate(alg, (comparison_df['recall'].iloc[i], comparison_df['precision'].iloc[i]),
                              xytext=(5, 5), textcoords='offset points', fontsize=9)
        axes[0, 1].set_xlabel('Recall')
        axes[0, 1].set_ylabel('Precision')
        axes[0, 1].set_title('Precision vs Recall')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Processing Speed
        axes[1, 0].barh(algorithms, comparison_df['pairs_per_second'])
        axes[1, 0].set_xlabel('Pairs per Second')
        axes[1, 0].set_title('Processing Speed Comparison')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Accuracy comparison
        axes[1, 1].barh(algorithms, comparison_df['accuracy'])
        axes[1, 1].set_xlabel('Accuracy')
        axes[1, 1].set_title('Accuracy Comparison')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Algorithm comparison plot saved to: {save_path}")
        
        plt.show()
    
    def save_results(self, results: Dict[str, Any], save_path: str):
        """
        Save evaluation results to JSON file.
        
        Args:
            results: Results dictionary to save
            save_path: Path to save the results
        """
        # Convert numpy types to native Python types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        # Convert all numpy types
        json_results = {}
        for key, value in results.items():
            json_results[key] = convert_numpy(value)
        
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(json_results, f, indent=2, ensure_ascii=False)
        
        print(f"Results saved to: {save_path}")
    
    def load_results(self, load_path: str) -> Dict[str, Any]:
        """
        Load evaluation results from JSON file.
        
        Args:
            load_path: Path to load the results from
            
        Returns:
            Results dictionary
        """
        with open(load_path, 'r', encoding='utf-8') as f:
            results = json.load(f)
        
        print(f"Results loaded from: {load_path}")
        return results


def create_baseline_similarity_function():
    """
    Create a simple baseline similarity function for comparison.
    This is a basic exact match function.
    """
    def baseline_similarity(addr1: str, addr2: str) -> float:
        """Simple baseline: exact match after basic cleaning."""
        clean1 = addr1.lower().strip()
        clean2 = addr2.lower().strip()
        return 1.0 if clean1 == clean2 else 0.0
    
    return baseline_similarity
