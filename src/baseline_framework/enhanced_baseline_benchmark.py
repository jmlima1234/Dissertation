"""
Enhanced Baseline Benchmarking Script with Improved Accuracy Evaluation
======================================================================

This script integrates the enhanced accuracy evaluation system that uses:
- Fuzzy string matching for Portuguese addresses
- Partial credit scoring for close matches  
- Portuguese language-specific normalization

This provides more realistic and fair accuracy metrics while maintaining
rigorous standards for baseline comparison.
"""

import time
import json
import csv
import statistics
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, asdict
import logging
from pathlib import Path

# Import our modules
from search_api import HybridAddressSearch, SearchResult
from baseline_test_suite import BaselineTestSuite, TestQuery, QueryCategory, DifficultyLevel
from enhanced_accuracy_evaluator import EnhancedAccuracyEvaluator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('enhanced_baseline_benchmark.log')
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class EnhancedQueryResult:
    """Enhanced results for a single test query execution"""
    query: str
    category: str
    difficulty: str
    expected_street: Optional[str]
    expected_city: Optional[str] 
    expected_postcode: Optional[str]
    expected_municipality: Optional[str]
    
    # Performance metrics
    response_time_ms: float
    success: bool
    error_message: Optional[str]
    
    # Results returned
    num_results: int
    results: List[Dict[str, Any]]
    
    # Enhanced accuracy metrics
    top1_correct: bool
    top3_correct: bool 
    top5_correct: bool
    
    # Detailed accuracy scores
    best_match_rank: Optional[int]
    best_match_score: float
    best_overall_score: float
    
    # Component accuracy breakdown
    street_accuracy: float
    city_accuracy: float
    postcode_accuracy: float
    
    # Quality metrics
    avg_confidence_score: float
    best_confidence_score: float
    avg_elasticsearch_score: float

@dataclass
class EnhancedBenchmarkSummary:
    """Enhanced summary of benchmark results"""
    total_queries: int
    successful_queries: int
    failed_queries: int
    
    # Performance metrics  
    avg_response_time_ms: float
    median_response_time_ms: float
    p95_response_time_ms: float
    p99_response_time_ms: float
    min_response_time_ms: float
    max_response_time_ms: float
    
    # Enhanced accuracy metrics
    overall_top1_accuracy: float
    overall_top3_accuracy: float
    overall_top5_accuracy: float
    
    # Component accuracy averages  
    avg_street_accuracy: float
    avg_city_accuracy: float
    avg_postcode_accuracy: float
    avg_overall_match_score: float
    
    # Quality metrics
    avg_confidence_score: float
    avg_results_per_query: float
    
    # Enhanced analysis
    accuracy_improvement_vs_strict: Dict[str, float]
    
    # By category breakdown
    category_results: Dict[str, Dict[str, Any]]
    
    # By difficulty breakdown  
    difficulty_results: Dict[str, Dict[str, Any]]

class EnhancedBaselineBenchmark:
    """Enhanced benchmarking system with improved accuracy evaluation"""
    
    def __init__(self, results_dir: str = "enhanced_benchmark_results"):
        """
        Initialize the enhanced benchmark system
        
        Args:
            results_dir: Directory to save benchmark results
        """
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        
        self.search_system = None
        self.test_suite = BaselineTestSuite()
        self.accuracy_evaluator = EnhancedAccuracyEvaluator()
        self.query_results: List[EnhancedQueryResult] = []
        
        # Benchmark configuration
        self.warmup_queries = 5
        self.max_results_per_query = 10
        self.timeout_seconds = 5.0
        
    def initialize_search_system(self):
        """Initialize the search system and perform warmup"""
        logger.info("Initializing enhanced search system...")
        try:
            self.search_system = HybridAddressSearch()
            logger.info("Search system initialized successfully")
            
            # Perform warmup queries
            logger.info(f"Performing {self.warmup_queries} warmup queries...")
            warmup_queries = ["lisboa", "porto", "rua augusta", "1000-001", "coimbra"][:self.warmup_queries]
            
            for query in warmup_queries:
                try:
                    self.search_system.search(query, max_results=3)
                    time.sleep(0.1)
                except Exception as e:
                    logger.warning(f"Warmup query '{query}' failed: {e}")
            
            logger.info("Warmup completed")
            
        except Exception as e:
            logger.error(f"Failed to initialize search system: {e}")
            raise
    
    def prepare_expected_data(self, query: TestQuery) -> Dict[str, Any]:
        """
        Prepare expected data dictionary for enhanced evaluation
        
        Args:
            query: Test query object
            
        Returns:
            Dictionary with expected components for evaluation
        """
        expected = {}
        
        if query.expected_street:
            expected['street'] = query.expected_street
        if query.expected_city:
            expected['city'] = query.expected_city
        if query.expected_postcode:
            expected['postcode'] = query.expected_postcode
        if query.expected_municipality:
            expected['municipality'] = query.expected_municipality
            
        return expected
    
    def convert_search_results(self, search_results: List[SearchResult]) -> List[Dict[str, Any]]:
        """
        Convert SearchResult objects to dictionaries for evaluation
        
        Args:
            search_results: List of SearchResult objects
            
        Returns:
            List of dictionaries suitable for enhanced evaluation
        """
        converted_results = []
        
        for result in search_results:
            converted_results.append({
                'street_clean': result.street_clean,
                'city_clean': result.city_clean,
                'postcode_clean': result.postcode_clean,
                'municipality': result.municipality,
                'district': result.district,
                'latitude': result.latitude,
                'longitude': result.longitude,
                'confidence_score': result.confidence_score,
                'elasticsearch_score': result.elasticsearch_score,
                'address_full': result.address_full
            })
        
        return converted_results
    
    def execute_query(self, test_query: TestQuery) -> EnhancedQueryResult:
        """Execute a single test query and collect enhanced metrics"""
        logger.debug(f"Executing query: '{test_query.query}'")
        
        start_time = time.perf_counter()
        success = False
        error_message = None
        results = []
        
        try:
            # Execute the search
            search_results = self.search_system.search(
                query=test_query.query,
                max_results=self.max_results_per_query,
                min_score=0.1,
                include_raw=False
            )
            
            success = True
            results = search_results
            
        except Exception as e:
            error_message = str(e)
            logger.error(f"Query '{test_query.query}' failed: {error_message}")
        
        end_time = time.perf_counter()
        response_time_ms = (end_time - start_time) * 1000
        
        # Enhanced accuracy evaluation
        if success and results:
            # Prepare data for enhanced evaluation
            expected_data = self.prepare_expected_data(test_query)
            converted_results = self.convert_search_results(results)
            
            # Use enhanced accuracy evaluation
            accuracy_result = self.accuracy_evaluator.evaluate_top_k_accuracy(
                converted_results, expected_data, k_values=[1, 3, 5]
            )
            
            # Extract enhanced metrics
            top1_correct = accuracy_result['top_1_correct']
            top3_correct = accuracy_result['top_3_correct'] 
            top5_correct = accuracy_result['top_5_correct']
            best_match_rank = accuracy_result['best_match_rank']
            best_match_score = accuracy_result['best_match_score']
            
            # Extract component accuracy scores from best match
            component_accuracies = {'street': 0.0, 'city': 0.0, 'postcode': 0.0}
            best_overall_score = 0.0
            
            if accuracy_result['detailed_evaluations']:
                best_evaluation = accuracy_result['detailed_evaluations'][0]['evaluation']
                best_overall_score = best_evaluation.overall_score
                
                for component, score_data in best_evaluation.component_scores.items():
                    if component in component_accuracies:
                        component_accuracies[component] = score_data['score']
        else:
            # Default values for failed queries
            top1_correct = top3_correct = top5_correct = False
            best_match_rank = None
            best_match_score = 0.0
            best_overall_score = 0.0
            component_accuracies = {'street': 0.0, 'city': 0.0, 'postcode': 0.0}
        
        # Calculate quality metrics
        if results:
            confidence_scores = [r.confidence_score for r in results if r.confidence_score is not None]
            es_scores = [r.elasticsearch_score for r in results if r.elasticsearch_score is not None]
            
            avg_confidence = statistics.mean(confidence_scores) if confidence_scores else 0.0
            best_confidence = max(confidence_scores) if confidence_scores else 0.0
            avg_es_score = statistics.mean(es_scores) if es_scores else 0.0
        else:
            avg_confidence = best_confidence = avg_es_score = 0.0
        
        # Convert results to serializable format
        results_data = self.convert_search_results(results) if results else []
        
        return EnhancedQueryResult(
            query=test_query.query,
            category=test_query.category.value,
            difficulty=test_query.difficulty.value,
            expected_street=test_query.expected_street,
            expected_city=test_query.expected_city,
            expected_postcode=test_query.expected_postcode,
            expected_municipality=test_query.expected_municipality,
            response_time_ms=response_time_ms,
            success=success,
            error_message=error_message,
            num_results=len(results),
            results=results_data,
            top1_correct=top1_correct,
            top3_correct=top3_correct,
            top5_correct=top5_correct,
            best_match_rank=best_match_rank,
            best_match_score=best_match_score,
            best_overall_score=best_overall_score,
            street_accuracy=component_accuracies['street'],
            city_accuracy=component_accuracies['city'],
            postcode_accuracy=component_accuracies['postcode'],
            avg_confidence_score=avg_confidence,
            best_confidence_score=best_confidence,
            avg_elasticsearch_score=avg_es_score
        )
    
    def run_enhanced_benchmark(self) -> EnhancedBenchmarkSummary:
        """Run the complete enhanced benchmark suite"""
        logger.info("Starting enhanced baseline benchmark execution...")
        
        # Initialize search system
        self.initialize_search_system()
        
        # Get all test queries
        test_queries = self.test_suite.get_all_queries()
        total_queries = len(test_queries)
        
        logger.info(f"Executing {total_queries} test queries with enhanced accuracy evaluation...")
        
        # Execute all queries
        self.query_results = []
        for i, test_query in enumerate(test_queries, 1):
            logger.info(f"Progress: {i}/{total_queries} - '{test_query.query}'")
            
            result = self.execute_query(test_query)
            self.query_results.append(result)
            
            # Small delay to avoid overwhelming the system
            time.sleep(0.05)
        
        # Generate enhanced summary
        summary = self.generate_enhanced_summary()
        
        # Save results
        self.save_enhanced_results(summary)
        
        logger.info("Enhanced benchmark execution completed!")
        return summary
    
    def generate_enhanced_summary(self) -> EnhancedBenchmarkSummary:
        """Generate comprehensive enhanced summary of benchmark results"""
        successful_results = [r for r in self.query_results if r.success]
        failed_results = [r for r in self.query_results if not r.success]
        
        # Performance metrics (only successful queries)
        if successful_results:
            response_times = [r.response_time_ms for r in successful_results]
            avg_response_time = statistics.mean(response_times)
            median_response_time = statistics.median(response_times)
            p95_response_time = statistics.quantiles(response_times, n=20)[18] if len(response_times) >= 20 else max(response_times)
            p99_response_time = statistics.quantiles(response_times, n=100)[98] if len(response_times) >= 100 else max(response_times)
            min_response_time = min(response_times)
            max_response_time = max(response_times)
        else:
            avg_response_time = median_response_time = p95_response_time = p99_response_time = 0.0
            min_response_time = max_response_time = 0.0
        
        # Enhanced accuracy metrics
        if successful_results:
            top1_accuracy = sum(1 for r in successful_results if r.top1_correct) / len(successful_results)
            top3_accuracy = sum(1 for r in successful_results if r.top3_correct) / len(successful_results)
            top5_accuracy = sum(1 for r in successful_results if r.top5_correct) / len(successful_results)
            
            # Component accuracy averages
            avg_street_accuracy = statistics.mean([r.street_accuracy for r in successful_results])
            avg_city_accuracy = statistics.mean([r.city_accuracy for r in successful_results])
            avg_postcode_accuracy = statistics.mean([r.postcode_accuracy for r in successful_results])
            avg_overall_match_score = statistics.mean([r.best_overall_score for r in successful_results])
        else:
            top1_accuracy = top3_accuracy = top5_accuracy = 0.0
            avg_street_accuracy = avg_city_accuracy = avg_postcode_accuracy = 0.0
            avg_overall_match_score = 0.0
        
        # Quality metrics
        if successful_results:
            avg_confidence = statistics.mean([r.avg_confidence_score for r in successful_results])
            avg_results_per_query = statistics.mean([r.num_results for r in successful_results])
        else:
            avg_confidence = avg_results_per_query = 0.0
        
        # Calculate improvement vs strict evaluation (simulated comparison)
        # This would be calculated by comparing with original benchmark results
        accuracy_improvement = {
            'top1_improvement': 0.0,  # Will be calculated when comparing with original
            'top3_improvement': 0.0,
            'top5_improvement': 0.0
        }
        
        # Category breakdown
        category_results = {}
        for category in QueryCategory:
            cat_results = [r for r in self.query_results if r.category == category.value]
            if cat_results:
                cat_successful = [r for r in cat_results if r.success]
                category_results[category.value] = {
                    'total_queries': len(cat_results),
                    'successful': len(cat_successful),
                    'avg_response_time_ms': statistics.mean([r.response_time_ms for r in cat_successful]) if cat_successful else 0.0,
                    'top1_accuracy': sum(1 for r in cat_successful if r.top1_correct) / len(cat_successful) if cat_successful else 0.0,
                    'avg_overall_score': statistics.mean([r.best_overall_score for r in cat_successful]) if cat_successful else 0.0,
                    'avg_confidence': statistics.mean([r.avg_confidence_score for r in cat_successful]) if cat_successful else 0.0
                }
        
        # Difficulty breakdown
        difficulty_results = {}
        for difficulty in DifficultyLevel:
            diff_results = [r for r in self.query_results if r.difficulty == difficulty.value]
            if diff_results:
                diff_successful = [r for r in diff_results if r.success]
                difficulty_results[difficulty.value] = {
                    'total_queries': len(diff_results),
                    'successful': len(diff_successful), 
                    'avg_response_time_ms': statistics.mean([r.response_time_ms for r in diff_successful]) if diff_successful else 0.0,
                    'top1_accuracy': sum(1 for r in diff_successful if r.top1_correct) / len(diff_successful) if diff_successful else 0.0,
                    'avg_overall_score': statistics.mean([r.best_overall_score for r in diff_successful]) if diff_successful else 0.0,
                    'avg_confidence': statistics.mean([r.avg_confidence_score for r in diff_successful]) if diff_successful else 0.0
                }
        
        return EnhancedBenchmarkSummary(
            total_queries=len(self.query_results),
            successful_queries=len(successful_results),
            failed_queries=len(failed_results),
            avg_response_time_ms=avg_response_time,
            median_response_time_ms=median_response_time,
            p95_response_time_ms=p95_response_time,
            p99_response_time_ms=p99_response_time,
            min_response_time_ms=min_response_time,
            max_response_time_ms=max_response_time,
            overall_top1_accuracy=top1_accuracy,
            overall_top3_accuracy=top3_accuracy,
            overall_top5_accuracy=top5_accuracy,
            avg_street_accuracy=avg_street_accuracy,
            avg_city_accuracy=avg_city_accuracy,
            avg_postcode_accuracy=avg_postcode_accuracy,
            avg_overall_match_score=avg_overall_match_score,
            avg_confidence_score=avg_confidence,
            avg_results_per_query=avg_results_per_query,
            accuracy_improvement_vs_strict=accuracy_improvement,
            category_results=category_results,
            difficulty_results=difficulty_results
        )
    
    def save_enhanced_results(self, summary: EnhancedBenchmarkSummary):
        """Save enhanced benchmark results in multiple formats"""
        
        # Save detailed results as JSON
        detailed_results = {
            'summary': asdict(summary),
            'detailed_results': [asdict(r) for r in self.query_results],
            'evaluation_methodology': {
                'fuzzy_matching': True,
                'partial_credit': True,
                'component_weights': self.accuracy_evaluator.component_weights,
                'fuzzy_thresholds': self.accuracy_evaluator.fuzzy_thresholds
            }
        }

        json_path = self.results_dir / f"enhanced_benchmark_detailed.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(detailed_results, f, indent=2, ensure_ascii=False)
        logger.info(f"Enhanced detailed results saved to {json_path}")
        
        # Save summary as JSON
        summary_path = self.results_dir / f"enhanced_benchmark_summary.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(asdict(summary), f, indent=2, ensure_ascii=False)
        logger.info(f"Enhanced summary saved to {summary_path}")
        
        # Save results as CSV
        csv_path = self.results_dir / f"enhanced_benchmark_results.csv"
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Enhanced header
            writer.writerow([
                'Query', 'Category', 'Difficulty', 'Expected_Street', 'Expected_City', 
                'Response_Time_ms', 'Success', 'Num_Results', 'Top1_Correct', 
                'Top3_Correct', 'Top5_Correct', 'Best_Match_Rank', 'Best_Match_Score',
                'Overall_Score', 'Street_Accuracy', 'City_Accuracy', 'Postcode_Accuracy',
                'Avg_Confidence', 'Best_Confidence'
            ])
            
            # Enhanced data rows
            for result in self.query_results:
                writer.writerow([
                    result.query, result.category, result.difficulty, 
                    result.expected_street or '', result.expected_city or '',
                    f"{result.response_time_ms:.2f}", result.success, result.num_results,
                    result.top1_correct, result.top3_correct, result.top5_correct,
                    result.best_match_rank or '', f"{result.best_match_score:.3f}",
                    f"{result.best_overall_score:.3f}", f"{result.street_accuracy:.3f}",
                    f"{result.city_accuracy:.3f}", f"{result.postcode_accuracy:.3f}",
                    f"{result.avg_confidence_score:.3f}",
                    f"{result.best_confidence_score:.3f}"
                ])
        
        logger.info(f"Enhanced CSV results saved to {csv_path}")
        
        # Generate enhanced report
        self.generate_enhanced_report(summary)
    
    def generate_enhanced_report(self, summary: EnhancedBenchmarkSummary):
        """Generate enhanced human-readable benchmark report"""
        report_path = self.results_dir / f"enhanced_benchmark_report.md"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# Portuguese Address Search - Enhanced Baseline Benchmark Report\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"**Test Suite:** {summary.total_queries} queries with enhanced fuzzy matching evaluation\n\n")
            
            # Executive Summary
            f.write("## Executive Summary\n\n")
            f.write(f"- **Success Rate:** {summary.successful_queries}/{summary.total_queries} ({summary.successful_queries/summary.total_queries*100:.1f}%)\n")
            f.write(f"- **Average Response Time:** {summary.avg_response_time_ms:.1f}ms\n")
            f.write(f"- **95th Percentile Response Time:** {summary.p95_response_time_ms:.1f}ms\n")
            f.write(f"- **Enhanced Top-1 Accuracy:** {summary.overall_top1_accuracy*100:.1f}%\n")
            f.write(f"- **Enhanced Top-3 Accuracy:** {summary.overall_top3_accuracy*100:.1f}%\n")
            f.write(f"- **Average Overall Match Score:** {summary.avg_overall_match_score:.3f}\n\n")
            
            # Enhanced Accuracy Metrics
            f.write("## Enhanced Accuracy Metrics\n\n")
            f.write("| Metric | Value | Description |\n")
            f.write("|--------|-------|-------------|\n")
            f.write(f"| Top-1 Accuracy | {summary.overall_top1_accuracy*100:.1f}% | Best result matches expected (fuzzy) |\n")
            f.write(f"| Top-3 Accuracy | {summary.overall_top3_accuracy*100:.1f}% | Match found in top 3 results |\n")
            f.write(f"| Top-5 Accuracy | {summary.overall_top5_accuracy*100:.1f}% | Match found in top 5 results |\n")
            f.write(f"| Avg Overall Score | {summary.avg_overall_match_score:.3f} | Weighted component accuracy |\n")
            f.write(f"| Street Accuracy | {summary.avg_street_accuracy:.3f} | Street name fuzzy matching |\n")
            f.write(f"| City Accuracy | {summary.avg_city_accuracy:.3f} | City name fuzzy matching |\n")
            f.write(f"| Postcode Accuracy | {summary.avg_postcode_accuracy:.3f} | Postcode exact/near matching |\n\n")
            
            # Performance Metrics
            f.write("## Performance Metrics\n\n")
            f.write("| Metric | Value |\n")
            f.write("|--------|-------|\n")
            f.write(f"| Average Response Time | {summary.avg_response_time_ms:.1f}ms |\n")
            f.write(f"| Median Response Time | {summary.median_response_time_ms:.1f}ms |\n")
            f.write(f"| 95th Percentile | {summary.p95_response_time_ms:.1f}ms |\n")
            f.write(f"| 99th Percentile | {summary.p99_response_time_ms:.1f}ms |\n")
            f.write(f"| Min Response Time | {summary.min_response_time_ms:.1f}ms |\n")
            f.write(f"| Max Response Time | {summary.max_response_time_ms:.1f}ms |\n\n")
            
            # Enhanced category breakdown
            f.write("## Enhanced Performance by Query Category\n\n")
            f.write("| Category | Queries | Success Rate | Avg Time (ms) | Top-1 Acc | Overall Score | Confidence |\n")
            f.write("|----------|---------|--------------|---------------|-----------|---------------|------------|\n")
            for category, stats in summary.category_results.items():
                success_rate = stats['successful'] / stats['total_queries'] * 100
                f.write(f"| {category.replace('_', ' ').title()} | {stats['total_queries']} | {success_rate:.1f}% | {stats['avg_response_time_ms']:.1f} | {stats['top1_accuracy']*100:.1f}% | {stats['avg_overall_score']:.3f} | {stats['avg_confidence']:.2f} |\n")
            f.write("\n")
            
            # Enhanced difficulty breakdown
            f.write("## Enhanced Performance by Query Difficulty\n\n")
            f.write("| Difficulty | Queries | Success Rate | Avg Time (ms) | Top-1 Acc | Overall Score | Confidence |\n")
            f.write("|------------|---------|--------------|---------------|-----------|---------------|------------|\n")
            for difficulty, stats in summary.difficulty_results.items():
                success_rate = stats['successful'] / stats['total_queries'] * 100
                f.write(f"| {difficulty.title()} | {stats['total_queries']} | {success_rate:.1f}% | {stats['avg_response_time_ms']:.1f} | {stats['top1_accuracy']*100:.1f}% | {stats['avg_overall_score']:.3f} | {stats['avg_confidence']:.2f} |\n")
            f.write("\n")
            
            # Enhanced insights
            f.write("## Enhanced Evaluation Insights\n\n")
            
            best_category = max(summary.category_results.items(), key=lambda x: x[1]['top1_accuracy'])
            worst_category = min(summary.category_results.items(), key=lambda x: x[1]['top1_accuracy'])
            
            f.write(f"- **Best Category:** {best_category[0].replace('_', ' ').title()} with {best_category[1]['top1_accuracy']*100:.1f}% enhanced accuracy\n")
            f.write(f"- **Most Challenging:** {worst_category[0].replace('_', ' ').title()} with {worst_category[1]['top1_accuracy']*100:.1f}% enhanced accuracy\n")
            f.write(f"- **Component Performance:** Street ({summary.avg_street_accuracy:.3f}), City ({summary.avg_city_accuracy:.3f}), Postcode ({summary.avg_postcode_accuracy:.3f})\n")
            
            f.write("\n## Enhanced Evaluation Methodology\n\n")
            f.write("This enhanced benchmark uses:\n\n")
            f.write("- **Fuzzy String Matching:** Portuguese-aware text similarity with thresholds\n")
            f.write("- **Partial Credit Scoring:** Graduated scoring based on similarity levels\n")
            f.write("- **Component Weighting:** Street (50%), City (35%), Postcode (15%)\n")
            f.write("- **Portuguese Normalization:** Accent removal, abbreviation expansion, article handling\n\n")
            
            f.write("## Baseline Establishment\n\n")
            f.write("This enhanced benchmark provides more realistic accuracy metrics while maintaining rigorous evaluation standards. ")
            f.write("The fuzzy matching approach better reflects real-world address search scenarios where exact string matches are rare but semantically equivalent results should be considered correct.\n")
        
        logger.info(f"Enhanced human-readable report saved to {report_path}")


# Main execution
if __name__ == "__main__":
    print("=== Portuguese Address Search - Enhanced Baseline Benchmark ===\n")
    
    benchmark = EnhancedBaselineBenchmark()
    
    try:
        # Run the enhanced benchmark
        summary = benchmark.run_enhanced_benchmark()
        
        # Print key results
        print("\n=== ENHANCED BENCHMARK RESULTS ===")
        print(f"Total Queries: {summary.total_queries}")
        print(f"Success Rate: {summary.successful_queries}/{summary.total_queries} ({summary.successful_queries/summary.total_queries*100:.1f}%)")
        print(f"Average Response Time: {summary.avg_response_time_ms:.1f}ms")
        print(f"95th Percentile Response Time: {summary.p95_response_time_ms:.1f}ms")
        print(f"Enhanced Top-1 Accuracy: {summary.overall_top1_accuracy*100:.1f}%")
        print(f"Enhanced Top-3 Accuracy: {summary.overall_top3_accuracy*100:.1f}%")
        print(f"Average Overall Match Score: {summary.avg_overall_match_score:.3f}")
        
        print("\n=== COMPONENT ACCURACY SCORES ===")
        print(f"Street Accuracy: {summary.avg_street_accuracy:.3f}")
        print(f"City Accuracy: {summary.avg_city_accuracy:.3f}")
        print(f"Postcode Accuracy: {summary.avg_postcode_accuracy:.3f}")
        
        print("\n=== TOP PERFORMING CATEGORIES (ENHANCED) ===")
        sorted_categories = sorted(summary.category_results.items(), 
                                 key=lambda x: x[1]['top1_accuracy'], reverse=True)
        for i, (category, stats) in enumerate(sorted_categories[:3], 1):
            print(f"{i}. {category.replace('_', ' ').title()}: {stats['top1_accuracy']*100:.1f}% accuracy, {stats['avg_overall_score']:.3f} overall score")
        
        print(f"\n✅ Enhanced benchmark completed! Results saved to 'enhanced_benchmark_results/' directory.")
        
    except Exception as e:
        print(f"❌ Enhanced benchmark failed: {e}")
        logger.error(f"Enhanced benchmark execution failed: {e}", exc_info=True)