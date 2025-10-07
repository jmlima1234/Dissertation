"""
Baseline Benchmarking Script for Portuguese Address Search
=========================================================

This script systematically evaluates the baseline search system performance
and accuracy using the comprehensive test suite. It measures key metrics 
that can be used for future comparisons and improvements.

Metrics Collected:
- Performance: Response times, percentiles, throughput
- Accuracy: Top-1, Top-3, Top-5 matching rates  
- Quality: Confidence scores, result relevance
- Reliability: Success rates, error handling

The results are exported in multiple formats for analysis and documentation.
"""

import time
import json
import csv
import statistics
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, asdict
import logging
from pathlib import Path

# Import our modules
from src.baseline_framework.search_api import HybridAddressSearch, SearchResult
from baseline_test_suite import BaselineTestSuite, TestQuery, QueryCategory, DifficultyLevel
from src.baseline_framework.enhanced_accuracy_evaluator import EnhancedAccuracyEvaluator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f'baseline_benchmark.log')
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class QueryResult:
    """Results for a single test query execution"""
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
    
    # Accuracy metrics
    top1_correct: bool
    top3_correct: bool 
    top5_correct: bool
    
    # Quality metrics
    avg_confidence_score: float
    best_confidence_score: float
    avg_elasticsearch_score: float

@dataclass
class BenchmarkSummary:
    """Summary of benchmark results"""
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
    
    # Accuracy metrics
    overall_top1_accuracy: float
    overall_top3_accuracy: float
    overall_top5_accuracy: float
    
    # Quality metrics
    avg_confidence_score: float
    avg_results_per_query: float
    
    # By category breakdown
    category_results: Dict[str, Dict[str, Any]]
    
    # By difficulty breakdown  
    difficulty_results: Dict[str, Dict[str, Any]]

class BaselineBenchmark:
    """Main benchmarking system for baseline evaluation"""
    
    def __init__(self, results_dir: str = "benchmark_results"):
        """
        Initialize the benchmark system
        
        Args:
            results_dir: Directory to save benchmark results
        """
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        
        self.search_system = None
        self.test_suite = BaselineTestSuite()
        self.query_results: List[QueryResult] = []
        
        # Benchmark configuration
        self.warmup_queries = 5  # Number of warmup queries
        self.max_results_per_query = 10
        self.timeout_seconds = 5.0
        
    def initialize_search_system(self):
        """Initialize the search system and perform warmup"""
        logger.info("Initializing search system...")
        try:
            self.search_system = HybridAddressSearch()
            logger.info("Search system initialized successfully")
            
            # Perform warmup queries
            logger.info(f"Performing {self.warmup_queries} warmup queries...")
            warmup_queries = ["lisboa", "porto", "rua augusta", "1000-001", "coimbra"][:self.warmup_queries]
            
            for query in warmup_queries:
                try:
                    self.search_system.search(query, max_results=3)
                    time.sleep(0.1)  # Small delay between warmup queries
                except Exception as e:
                    logger.warning(f"Warmup query '{query}' failed: {e}")
            
            logger.info("Warmup completed")
            
        except Exception as e:
            logger.error(f"Failed to initialize search system: {e}")
            raise
    
    def evaluate_accuracy(self, query: TestQuery, results: List[SearchResult]) -> Tuple[bool, bool, bool]:
        """
        Evaluate accuracy of results against expected values
        
        Returns:
            Tuple of (top1_correct, top3_correct, top5_correct)
        """
        if not results:
            return False, False, False
        
        def matches_expected(result: SearchResult, query: TestQuery) -> bool:
            """Check if a result matches the expected criteria"""
            match_score = 0
            total_criteria = 0
            
            # Street matching
            if query.expected_street:
                total_criteria += 1
                if result.street_clean and query.expected_street.lower() in result.street_clean.lower():
                    match_score += 1
            
            # City matching  
            if query.expected_city:
                total_criteria += 1
                if result.city_clean and query.expected_city.lower() in result.city_clean.lower():
                    match_score += 1
            
            # Postcode matching
            if query.expected_postcode:
                total_criteria += 1
                if result.postcode_clean and query.expected_postcode == result.postcode_clean:
                    match_score += 1
            
            # Municipality matching
            if query.expected_municipality:
                total_criteria += 1
                if result.municipality and query.expected_municipality.lower() in result.municipality.lower():
                    match_score += 1
            
            # If no specific criteria, consider it a match if we got results (for city-only, etc.)
            if total_criteria == 0:
                return True
            
            # Require at least 70% of criteria to match
            return (match_score / total_criteria) >= 0.7
        
        # Check top-k accuracy
        top1_correct = matches_expected(results[0], query)
        top3_correct = any(matches_expected(result, query) for result in results[:3])
        top5_correct = any(matches_expected(result, query) for result in results[:5])
        
        return top1_correct, top3_correct, top5_correct
    
    def execute_query(self, test_query: TestQuery) -> QueryResult:
        """Execute a single test query and collect metrics"""
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
                min_score=0.1,  # Low threshold to get more results for analysis
                include_raw=False
            )
            
            success = True
            results = search_results
            
        except Exception as e:
            error_message = str(e)
            logger.error(f"Query '{test_query.query}' failed: {error_message}")
        
        end_time = time.perf_counter()
        response_time_ms = (end_time - start_time) * 1000
        
        # Evaluate accuracy
        top1_correct, top3_correct, top5_correct = self.evaluate_accuracy(test_query, results) if success else (False, False, False)
        
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
        results_data = []
        for result in results:
            results_data.append({
                'address_full': result.address_full,
                'street_clean': result.street_clean,
                'city_clean': result.city_clean,
                'postcode_clean': result.postcode_clean,
                'municipality': result.municipality,
                'district': result.district,
                'latitude': result.latitude,
                'longitude': result.longitude,
                'confidence_score': result.confidence_score,
                'elasticsearch_score': result.elasticsearch_score
            })
        
        return QueryResult(
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
            avg_confidence_score=avg_confidence,
            best_confidence_score=best_confidence,
            avg_elasticsearch_score=avg_es_score
        )
    
    def run_benchmark(self) -> BenchmarkSummary:
        """Run the complete benchmark suite"""
        logger.info("Starting baseline benchmark execution...")
        
        # Initialize search system
        self.initialize_search_system()
        
        # Get all test queries
        test_queries = self.test_suite.get_all_queries()
        total_queries = len(test_queries)
        
        logger.info(f"Executing {total_queries} test queries...")
        
        # Execute all queries
        self.query_results = []
        for i, test_query in enumerate(test_queries, 1):
            logger.info(f"Progress: {i}/{total_queries} - '{test_query.query}'")
            
            result = self.execute_query(test_query)
            self.query_results.append(result)
            
            # Small delay to avoid overwhelming the system
            time.sleep(0.05)
        
        # Generate summary
        summary = self.generate_summary()
        
        # Save results
        self.save_results(summary)
        
        logger.info("Benchmark execution completed!")
        return summary
    
    def generate_summary(self) -> BenchmarkSummary:
        """Generate comprehensive summary of benchmark results"""
        successful_results = [r for r in self.query_results if r.success]
        failed_results = [r for r in self.query_results if not r.success]
        
        # Performance metrics (only successful queries)
        if successful_results:
            response_times = [r.response_time_ms for r in successful_results]
            avg_response_time = statistics.mean(response_times)
            median_response_time = statistics.median(response_times)
            p95_response_time = statistics.quantiles(response_times, n=20)[18]  # 95th percentile
            p99_response_time = statistics.quantiles(response_times, n=100)[98]  # 99th percentile
            min_response_time = min(response_times)
            max_response_time = max(response_times)
        else:
            avg_response_time = median_response_time = p95_response_time = p99_response_time = 0.0
            min_response_time = max_response_time = 0.0
        
        # Accuracy metrics
        if successful_results:
            top1_accuracy = sum(1 for r in successful_results if r.top1_correct) / len(successful_results)
            top3_accuracy = sum(1 for r in successful_results if r.top3_correct) / len(successful_results)
            top5_accuracy = sum(1 for r in successful_results if r.top5_correct) / len(successful_results)
        else:
            top1_accuracy = top3_accuracy = top5_accuracy = 0.0
        
        # Quality metrics
        if successful_results:
            avg_confidence = statistics.mean([r.avg_confidence_score for r in successful_results])
            avg_results_per_query = statistics.mean([r.num_results for r in successful_results])
        else:
            avg_confidence = avg_results_per_query = 0.0
        
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
                    'avg_confidence': statistics.mean([r.avg_confidence_score for r in diff_successful]) if diff_successful else 0.0
                }
        
        return BenchmarkSummary(
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
            avg_confidence_score=avg_confidence,
            avg_results_per_query=avg_results_per_query,
            category_results=category_results,
            difficulty_results=difficulty_results
        )
    
    def save_results(self, summary: BenchmarkSummary):
        """Save benchmark results in multiple formats"""
        
        # Save detailed results as JSON
        detailed_results = {
            'summary': asdict(summary),
            'detailed_results': [asdict(r) for r in self.query_results]
        }
        
        json_path = self.results_dir / f"baseline_benchmark_detailed.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(detailed_results, f, indent=2, ensure_ascii=False)
        logger.info(f"Detailed results saved to {json_path}")
        
        # Save summary as JSON
        summary_path = self.results_dir / f"baseline_benchmark_summary.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(asdict(summary), f, indent=2, ensure_ascii=False)
        logger.info(f"Summary saved to {summary_path}")
        
        # Save results as CSV
        csv_path = self.results_dir / f"baseline_benchmark_results.csv"
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Header
            writer.writerow([
                'Query', 'Category', 'Difficulty', 'Expected_Street', 'Expected_City', 
                'Response_Time_ms', 'Success', 'Num_Results', 'Top1_Correct', 
                'Top3_Correct', 'Top5_Correct', 'Avg_Confidence', 'Best_Confidence'
            ])
            
            # Data rows
            for result in self.query_results:
                writer.writerow([
                    result.query, result.category, result.difficulty, 
                    result.expected_street or '', result.expected_city or '',
                    f"{result.response_time_ms:.2f}", result.success, result.num_results,
                    result.top1_correct, result.top3_correct, result.top5_correct,
                    f"{result.avg_confidence_score:.3f}", f"{result.best_confidence_score:.3f}"
                ])
        
        logger.info(f"CSV results saved to {csv_path}")
        
        # Generate and save human-readable report
        self.generate_report(summary)
    
    def generate_report(self, summary: BenchmarkSummary):
        """Generate a human-readable benchmark report"""
        report_path = self.results_dir / f"baseline_benchmark_report.md"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# Portuguese Address Search - Baseline Benchmark Report\n\n")
            f.write(f"**Test Suite:** {summary.total_queries} queries across 8 categories\n\n")
            
            # Executive Summary
            f.write("## Executive Summary\n\n")
            f.write(f"- **Success Rate:** {summary.successful_queries}/{summary.total_queries} ({summary.successful_queries/summary.total_queries*100:.1f}%)\n")
            f.write(f"- **Average Response Time:** {summary.avg_response_time_ms:.1f}ms\n")
            f.write(f"- **95th Percentile Response Time:** {summary.p95_response_time_ms:.1f}ms\n")
            f.write(f"- **Top-1 Accuracy:** {summary.overall_top1_accuracy*100:.1f}%\n")
            f.write(f"- **Top-3 Accuracy:** {summary.overall_top3_accuracy*100:.1f}%\n")
            f.write(f"- **Average Confidence Score:** {summary.avg_confidence_score:.2f}\n\n")
            
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
            
            # Accuracy Metrics
            f.write("## Accuracy Metrics\n\n")
            f.write("| Metric | Value |\n")
            f.write("|--------|-------|\n")
            f.write(f"| Top-1 Accuracy | {summary.overall_top1_accuracy*100:.1f}% |\n")
            f.write(f"| Top-3 Accuracy | {summary.overall_top3_accuracy*100:.1f}% |\n")
            f.write(f"| Top-5 Accuracy | {summary.overall_top5_accuracy*100:.1f}% |\n")
            f.write(f"| Avg Results per Query | {summary.avg_results_per_query:.1f} |\n")
            f.write(f"| Avg Confidence Score | {summary.avg_confidence_score:.2f} |\n\n")
            
            # Category Breakdown
            f.write("## Performance by Query Category\n\n")
            f.write("| Category | Queries | Success Rate | Avg Time (ms) | Top-1 Acc | Avg Confidence |\n")
            f.write("|----------|---------|--------------|---------------|-----------|----------------|\n")
            for category, stats in summary.category_results.items():
                success_rate = stats['successful'] / stats['total_queries'] * 100
                f.write(f"| {category.replace('_', ' ').title()} | {stats['total_queries']} | {success_rate:.1f}% | {stats['avg_response_time_ms']:.1f} | {stats['top1_accuracy']*100:.1f}% | {stats['avg_confidence']:.2f} |\n")
            f.write("\n")
            
            # Difficulty Breakdown
            f.write("## Performance by Query Difficulty\n\n")
            f.write("| Difficulty | Queries | Success Rate | Avg Time (ms) | Top-1 Acc | Avg Confidence |\n")
            f.write("|------------|---------|--------------|---------------|-----------|----------------|\n")
            for difficulty, stats in summary.difficulty_results.items():
                success_rate = stats['successful'] / stats['total_queries'] * 100
                f.write(f"| {difficulty.title()} | {stats['total_queries']} | {success_rate:.1f}% | {stats['avg_response_time_ms']:.1f} | {stats['top1_accuracy']*100:.1f}% | {stats['avg_confidence']:.2f} |\n")
            f.write("\n")
            
            # Key Insights
            f.write("## Key Insights\n\n")
            
            # Find best and worst performing categories
            best_category = max(summary.category_results.items(), key=lambda x: x[1]['top1_accuracy'])
            worst_category = min(summary.category_results.items(), key=lambda x: x[1]['top1_accuracy'])
            
            f.write(f"- **Best Category:** {best_category[0].replace('_', ' ').title()} with {best_category[1]['top1_accuracy']*100:.1f}% top-1 accuracy\n")
            f.write(f"- **Most Challenging:** {worst_category[0].replace('_', ' ').title()} with {worst_category[1]['top1_accuracy']*100:.1f}% top-1 accuracy\n")
            
            if summary.p95_response_time_ms < 500:
                f.write("- **Performance:** Excellent response times with 95% of queries under 500ms\n")
            elif summary.p95_response_time_ms < 1000:
                f.write("- **Performance:** Good response times with 95% of queries under 1 second\n")
            else:
                f.write("- **Performance:** Some queries exceed 1 second - consider optimization\n")
            
            f.write("\n## Baseline Establishment\n\n")
            f.write("This benchmark establishes the baseline performance characteristics for the Portuguese Address Search system using the hybrid PostGIS + Elasticsearch architecture. ")
            f.write("These metrics should be used for comparison when evaluating future improvements, optimizations, or alternative approaches.\n")
        
        logger.info(f"Human-readable report saved to {report_path}")


# Main execution
if __name__ == "__main__":
    print("=== Portuguese Address Search - Baseline Benchmark ===\n")
    
    benchmark = BaselineBenchmark()
    
    try:
        # Run the complete benchmark
        summary = benchmark.run_benchmark()
        
        # Print key results
        print("\n=== BENCHMARK RESULTS ===")
        print(f"Total Queries: {summary.total_queries}")
        print(f"Success Rate: {summary.successful_queries}/{summary.total_queries} ({summary.successful_queries/summary.total_queries*100:.1f}%)")
        print(f"Average Response Time: {summary.avg_response_time_ms:.1f}ms")
        print(f"95th Percentile Response Time: {summary.p95_response_time_ms:.1f}ms")
        print(f"Top-1 Accuracy: {summary.overall_top1_accuracy*100:.1f}%")
        print(f"Top-3 Accuracy: {summary.overall_top3_accuracy*100:.1f}%")
        print(f"Average Confidence Score: {summary.avg_confidence_score:.2f}")
        
        print("\n=== TOP PERFORMING CATEGORIES ===")
        sorted_categories = sorted(summary.category_results.items(), 
                                 key=lambda x: x[1]['top1_accuracy'], reverse=True)
        for i, (category, stats) in enumerate(sorted_categories[:3], 1):
            print(f"{i}. {category.replace('_', ' ').title()}: {stats['top1_accuracy']*100:.1f}% accuracy, {stats['avg_response_time_ms']:.1f}ms avg time")
        
        print(f"\n✅ Benchmark completed! Results saved to 'benchmark_results/' directory.")
        
    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
        logger.error(f"Benchmark execution failed: {e}", exc_info=True)