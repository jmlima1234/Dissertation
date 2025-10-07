# Portuguese Address Search - Enhanced Baseline Benchmark Report

**Generated:** 2025-09-30 12:11:45

**Test Suite:** 44 queries with enhanced fuzzy matching evaluation

## Executive Summary

- **Success Rate:** 44/44 (100.0%)
- **Average Response Time:** 258.2ms
- **95th Percentile Response Time:** 457.4ms
- **Enhanced Top-1 Accuracy:** 43.2%
- **Enhanced Top-3 Accuracy:** 47.7%
- **Average Overall Match Score:** 0.511

## Enhanced Accuracy Metrics

| Metric | Value | Description |
|--------|-------|-------------|
| Top-1 Accuracy | 43.2% | Best result matches expected (fuzzy) |
| Top-3 Accuracy | 47.7% | Match found in top 3 results |
| Top-5 Accuracy | 47.7% | Match found in top 5 results |
| Avg Overall Score | 0.511 | Weighted component accuracy |
| Street Accuracy | 0.456 | Street name fuzzy matching |
| City Accuracy | 0.449 | City name fuzzy matching |
| Postcode Accuracy | 0.045 | Postcode exact/near matching |

## Performance Metrics

| Metric | Value |
|--------|-------|
| Average Response Time | 258.2ms |
| Median Response Time | 240.0ms |
| 95th Percentile | 457.4ms |
| 99th Percentile | 465.6ms |
| Min Response Time | 49.8ms |
| Max Response Time | 465.6ms |

## Enhanced Performance by Query Category

| Category | Queries | Success Rate | Avg Time (ms) | Top-1 Acc | Overall Score | Confidence |
|----------|---------|--------------|---------------|-----------|---------------|------------|
| Exact Match | 5 | 100.0% | 328.8 | 100.0% | 1.000 | 59.55 |
| Abbreviation | 5 | 100.0% | 284.0 | 80.0% | 0.816 | 54.39 |
| Typo | 5 | 100.0% | 302.6 | 40.0% | 0.596 | 31.61 |
| Partial | 5 | 100.0% | 250.4 | 40.0% | 0.728 | 83.56 |
| Postcode Only | 5 | 100.0% | 235.1 | 40.0% | 0.400 | 39.31 |
| City Only | 5 | 100.0% | 315.7 | 20.0% | 0.200 | 107.98 |
| Complex | 5 | 100.0% | 245.8 | 0.0% | 0.153 | 37.97 |
| Edge Case | 9 | 100.0% | 172.1 | 33.3% | 0.333 | 31.57 |

## Enhanced Performance by Query Difficulty

| Difficulty | Queries | Success Rate | Avg Time (ms) | Top-1 Acc | Overall Score | Confidence |
|------------|---------|--------------|---------------|-----------|---------------|------------|
| Easy | 15 | 100.0% | 293.2 | 53.3% | 0.533 | 68.95 |
| Medium | 15 | 100.0% | 281.2 | 60.0% | 0.656 | 52.98 |
| Hard | 14 | 100.0% | 196.1 | 14.3% | 0.330 | 37.65 |

## Enhanced Evaluation Insights

- **Best Category:** Exact Match with 100.0% enhanced accuracy
- **Most Challenging:** Complex with 0.0% enhanced accuracy
- **Component Performance:** Street (0.456), City (0.449), Postcode (0.045)

## Enhanced Evaluation Methodology

This enhanced benchmark uses:

- **Fuzzy String Matching:** Portuguese-aware text similarity with thresholds
- **Partial Credit Scoring:** Graduated scoring based on similarity levels
- **Component Weighting:** Street (50%), City (35%), Postcode (15%)
- **Portuguese Normalization:** Accent removal, abbreviation expansion, article handling

## Baseline Establishment

This enhanced benchmark provides more realistic accuracy metrics while maintaining rigorous evaluation standards. The fuzzy matching approach better reflects real-world address search scenarios where exact string matches are rare but semantically equivalent results should be considered correct.
