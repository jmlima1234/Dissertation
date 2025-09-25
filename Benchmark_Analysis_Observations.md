# Portuguese Address Search System — Benchmark Analysis Observations

Generated: 2025-09-22  
Scope: Spatial and Textual search benchmarks over 3,254,526 address records  
Artifacts: results under `benchmark_results/` (PNG, CSV, TEX, JSON, TXT)

---

## Executive Summary

The experiments provide strong, data-backed evidence for two key conclusions:

- Spatial indexing substantially accelerates the right classes of geographic queries, with extreme gains for KNN proximity search and consistent wins for medium/large-area filters.
- A hybrid architecture (Elasticsearch + PostGIS) materially improves hard search cases (typos, abbreviations, complex and postcode-only queries), while exact city-only lookups regress due to orchestration overhead.

Top-line results (from exported summaries):
- Spatial: mean speedup 44.63x, median 1.04x, max 913.78x; 6/21 tests significant (MW). Average time reduction 12.3%.
- Hybrid: mean speed ratio (Naive/Hybrid) 3.63x; 12/19 faster; average relevance 51.90; accuracy +0.121; 7/19 significant (MW).

---

## Data, Methods, and Reproducibility

- Data: 3,254,526 normalized Portuguese addresses with valid geometries (`enderecos_normalizados`).
- Spatial tests: 21 queries spanning point-in-radius, bounding boxes, nearest-neighbor, distance comparisons, and large-area filters; regions include Lisboa, Porto, nationwide.
- Indexes evaluated: GiST, SP-GiST, BRIN, versus baseline (no index) where relevant.
- Hybrid search: 19 text queries across exact, partial, typo, incomplete, abbreviation, postcode-only, partial-postcode, complex. Evaluated on latency, accuracy, relevance.
- Statistics: Mann–Whitney U, t-test, Cohen’s d, 95% CIs. Significance tracked per test and summarized.
- Exports: see `benchmark_results/dissertation_exports/*.csv|.json|.tex|.txt` and figures in `benchmark_results/*.png`.

---

## Experiment A — Spatial Index Performance

Key findings (across all index types):
- Max speedup: 913.78x on KNN nearest-neighbor with GiST (Lisboa center).
- Median speedup: 1.04x (distribution is right-skewed by KNN outlier).
- Significant improvements: 6 of 21 spatial tests (MW).

By query complexity (speedup mean ± sd):
- Complex (n=9): 102.47 ± 304.24 — massive gains dominated by KNN; distance-comparisons show little change.
- Medium (n=6): 1.58 ± 0.55 — consistent, practical improvements on metro-area bounding boxes.
- Simple (n=6): 0.91 ± 0.11 — slight regressions for small-radius point queries due to overhead.

Index-specific observations (representative tests):
- GiST: 
   - KNN nearest-neighbor: 913.78x speedup; significant (MW, t); very large effect (d≈20.23).
   - Large area (nationwide envelope): 1.12x; significant (MW), modest effect; consistent improvement.
   - Distance-to-city comparison: ~1.00x (no material change), not significant.
- SP-GiST:
   - Lisboa bbox (medium): 2.45x; significant (MW, t); large effect (d≈8.00).
   - Porto bbox (medium): 1.55x; not significant but directionally positive.
   - KNN: ~1.03x (minimal impact) — SP-GiST does not excel at KNN here.
- BRIN:
   - Nearest-neighbor: 1.13x; significant (MW, t); useful but smaller than GiST.
   - Medium/large area: low single-digit gains; often not significant but safe/cheap.

Practical takeaways:
- Use GiST for KNN/proximity search; the benefit is transformative when the operator `<->` is used.
- Prefer SP-GiST for metro-area bounding boxes where partitioning aligns with data distribution.
- BRIN offers lightweight gains at minimal storage/cost; suitable for scans over broad extents but not a replacement for GiST on KNN.

---

## Experiment B — Hybrid Search Architecture

Summary metrics:
- Average speed ratio (Naive/Hybrid): 3.63x (hybrid faster on ratio basis across tests).
- Average time improvement: -78.6% (skewed negative by very fast naive city-only queries where hybrid overhead dominates).
- Wins: 12/19 faster; Significant (MW): 7/19. Accuracy +0.121; average relevance 51.90.

Performance by query type (mean time improvement):
- Abbreviation (n=2): +87.4%
- Complex (n=3): +79.4%
- Typo (n=1): +79.1%
- Postcode-only (n=1): +93.8%
- Partial (n=3): +27.2% (mixed)
- Incomplete (n=1): -44.9%
- Partial postcode (n=1): -116.5%
- Exact (n=7): -285.7% (city-only queries are the main regressions)

Representative cases:
- Fuzzy “rua augusta, lisbon” (typo): 79.05% faster; significant; hybrid achieves accuracy 0.5 from 0.0.
- Abbreviations (“r. da liberdade”, “av liberdade 123”): 92.61% and 82.26% faster; significant; major usability gains.
- Postcode-only (“1000-001”): 93.82% faster; significant; indexing and tokenization shine.
- Complex multi-term (“rua augusta 100 lisboa”): 74.60% faster; significant; hybrid returns more and better-ranked hits.
- City-only (“lisboa”, “porto”, “coimbra”, “braga”): large negative percentages — naive is already near-instant, so hybrid orchestration overhead dominates and appears “slower” by percentage despite small absolute latencies.

Interpretation:
- The hybrid system should be the default for ambiguous, noisy, or multi-signal queries (typos, abbreviations, composite phrases, postcode-only).
- For exact city-only or very broad terms, route to a lightweight SQL path (or an Elasticsearch-only shortcut) to avoid overhead.

---

## Statistical Rigor and Effect Sizes

- Spatial tests: 6/21 significant (MW). Average Cohen’s d ≈ 1.92; 9 large effects. KNN with GiST shows exceptional magnitude (d≈20.23).
- Hybrid tests: 7/19 significant (MW). Average Cohen’s d ≈ 13.30; 10 large effects. Several fuzzy/complex tasks show very strong, practically meaningful deltas.
- Confidence intervals and dual-testing (MW and t-test) were reported per test; results are exported in `spatial_statistical_analysis.csv` and embedded in `comprehensive_benchmark_results.json`.

---

## Systems Insights and Design Implications

- Query–architecture matching: No single stack wins universally. Optimal performance comes from routing by intent/complexity.
- Spatial index strategy: 
   - GiST for proximity; SP-GiST for structured bounding-box workloads; BRIN as low-cost helper for broad scans.
- Hybrid orchestration: Introduce a fast path for exact/city-only queries to avoid overhead; keep hybrid for fuzzy/complex.
- Quality vs. speed: Hybrid improved accuracy and relevance in hard cases, justifying use when user intent is uncertain.

---

## Limitations and Threats to Validity

- City-only regressions heavily skew percent-improvement metrics; absolute latency remains low but percent changes look extreme.
- Data distribution effects: Urban vs. rural density can change index selectivity; results may vary across regions and time.
- Environment sensitivity: Cache warmup, I/O, and concurrency were controlled but can affect absolute timings in production.

---

## Recommendations

- Implement an adaptive router:
   - If query matches city-only or very-short exact terms → route to simple SQL/ES shortcut.
   - If query includes typos, abbreviations, postcode fragments, or multi-field signals → route to hybrid pipeline.
- Spatial defaults:
   - Use GiST for KNN operators and proximity search.
   - Use SP-GiST for metro-area bounding boxes; consider BRIN for broad-range scans.
- Monitoring & evaluation:
   - Track per-query-type latency, accuracy, and click-through; use online significance testing to validate lab gains.
   - Keep exporting CSV/TEX tables and PNG dashboards for dissertation artifacts.

---

## Artifacts and How to Read Them

- `benchmark_results/performance_summary_dashboard.png`: high-level overview across experiments.
- `benchmark_results/search_architecture_comparison.png`: hybrid vs. naive by query type.
- `benchmark_results/comprehensive_spatial_index_analysis.png`: speedups by index and complexity.
- `benchmark_results/dissertation_exports/*.csv`: machine-readable results per test; `*_statistical_analysis.csv` for significance.
- `benchmark_results/dissertation_exports/*.tex`: LaTeX-ready tables; `executive_summary.txt` for concise highlights.

---

## Bottom Line

The hybrid architecture and spatial indexing both deliver substantial, statistically supported benefits when applied to the right problems. The clearest wins are:
- KNN/proximity search with GiST (order-of-magnitude+ speedups).
- Fuzzy, abbreviated, postcode-only, and complex textual queries with the hybrid stack (large, significant improvements and better result quality).

Use simple, low-overhead paths for exact/city-only lookups; use the full hybrid pipeline for ambiguous or rich-intent queries. This adaptive approach captures the best of both worlds and is well supported by the evidence in this benchmark suite.
