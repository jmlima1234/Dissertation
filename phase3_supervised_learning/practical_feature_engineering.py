#!/usr/bin/env python3
"""
Phase 3.3: Practical Feature Engineering Framework
==================================================

A streamlined implementation focusing on the most impactful features:
1. Core Phase 2 algorithm scores
2. Component-wise similarities (using parsed components)
3. NER features (already computed)
4. Statistical and interaction features

This creates a production-ready feature engineering pipeline.
"""

import pandas as pd
import numpy as np
import json
import time
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import Levenshtein for quick similarity computation
try:
    import Levenshtein
    LEVENSHTEIN_AVAILABLE = True
except ImportError:
    LEVENSHTEIN_AVAILABLE = False
    print("Warning: python-Levenshtein not available, using basic implementation")

class PracticalFeatureEngineer:
    """
    Practical feature engineering pipeline that creates ML-ready features
    from address matching data by combining multiple similarity approaches.
    """
    
    def __init__(self, data_path="ner_enhanced_gold_standard.csv"):
        """Initialize the feature engineering pipeline."""
        self.data_path = data_path
        print("🔧 Practical Feature Engineering Framework Initialized")
    
    def load_data(self):
        """Load the enhanced dataset with existing features."""
        print(f"\n📊 Loading Enhanced Dataset from {self.data_path}...")
        
        df = pd.read_csv(self.data_path, encoding='utf-8')
        print(f"   ✅ Loaded {len(df):,} address pairs")
        print(f"   ✅ Found {len(df.columns)} existing columns")
        
        return df
    
    def quick_similarity(self, s1, s2, method='jaro_winkler'):
        """Quick similarity computation with fallback implementations."""
        if pd.isna(s1) or pd.isna(s2):
            return 0.0
        
        s1, s2 = str(s1).strip().lower(), str(s2).strip().lower()
        
        if not s1 or not s2:
            return 0.0
        
        if s1 == s2:
            return 1.0
        
        try:
            if method == 'jaro_winkler' and LEVENSHTEIN_AVAILABLE:
                return Levenshtein.jaro_winkler(s1, s2)
            elif method == 'levenshtein' and LEVENSHTEIN_AVAILABLE:
                return 1.0 - (Levenshtein.distance(s1, s2) / max(len(s1), len(s2)))
            elif method == 'jaccard':
                # Simple Jaccard similarity on character bigrams
                bigrams1 = set(s1[i:i+2] for i in range(len(s1)-1))
                bigrams2 = set(s2[i:i+2] for i in range(len(s2)-1))
                if not bigrams1 and not bigrams2:
                    return 1.0
                if not bigrams1 or not bigrams2:
                    return 0.0
                return len(bigrams1.intersection(bigrams2)) / len(bigrams1.union(bigrams2))
            else:
                # Fallback: simple character overlap
                chars1, chars2 = set(s1), set(s2)
                if not chars1 and not chars2:
                    return 1.0
                if not chars1 or not chars2:
                    return 0.0
                return len(chars1.intersection(chars2)) / len(chars1.union(chars2))
                
        except Exception as e:
            return 0.0
    
    def compute_phase2_scores(self, df, sample_size=None):
        """Compute essential Phase 2 algorithm scores."""
        print("\n🔢 Computing Phase 2 Algorithm Scores...")
        
        if sample_size:
            df_work = df.sample(n=min(sample_size, len(df)), random_state=42).copy()
            print(f"   📋 Processing sample of {len(df_work):,} pairs")
        else:
            df_work = df.copy()
            print(f"   📋 Processing full dataset of {len(df_work):,} pairs")
        
        start_time = time.time()
        
        # Core algorithms to compute
        algorithms = ['jaro_winkler', 'levenshtein', 'jaccard']
        
        for alg in algorithms:
            print(f"   🔍 Computing {alg} similarities...")
            scores = []
            
            for idx, row in df_work.iterrows():
                addr1 = str(row['address_1'])
                addr2 = str(row['address_2'])
                score = self.quick_similarity(addr1, addr2, method=alg)
                scores.append(score)
                
                if len(scores) % 2000 == 0:
                    progress = len(scores) / len(df_work) * 100
                    print(f"     Progress: {len(scores):,}/{len(df_work):,} ({progress:.1f}%)")
            
            df_work[f'phase2_{alg}_score'] = scores
        
        # Add hybrid scores
        df_work['phase2_hybrid_avg'] = (df_work['phase2_jaro_winkler_score'] + 
                                       df_work['phase2_levenshtein_score'] + 
                                       df_work['phase2_jaccard_score']) / 3
        
        df_work['phase2_weighted_combo'] = (0.5 * df_work['phase2_jaro_winkler_score'] + 
                                           0.3 * df_work['phase2_jaccard_score'] + 
                                           0.2 * df_work['phase2_levenshtein_score'])
        
        elapsed = time.time() - start_time
        print(f"   ✅ Phase 2 scoring completed in {elapsed:.1f}s")
        print(f"   ✅ Added {len(algorithms) + 2} algorithm features")
        
        return df_work
    
    def compute_component_features(self, df):
        """Compute component-wise similarity features."""
        print("\n🏗️  Computing Component-wise Features...")
        
        # Components to analyze
        components = ['road', 'house_number', 'city', 'postcode', 'normalized']
        
        start_time = time.time()
        
        for component in components:
            print(f"   🔍 Processing {component} component...")
            
            col1 = f'address_1_{component}'
            col2 = f'address_2_{component}'
            
            if col1 in df.columns and col2 in df.columns:
                # Compute Jaro-Winkler similarity for this component
                similarities = []
                
                for idx, row in df.iterrows():
                    comp1 = row.get(col1, '')
                    comp2 = row.get(col2, '')
                    sim = self.quick_similarity(comp1, comp2, 'jaro_winkler')
                    similarities.append(sim)
                
                df[f'comp_{component}_similarity'] = similarities
                
                # Add exact match indicator
                exact_matches = []
                for idx, row in df.iterrows():
                    comp1 = str(row.get(col1, '')).strip().lower()
                    comp2 = str(row.get(col2, '')).strip().lower()
                    exact_matches.append(comp1 == comp2 and comp1 != '' and comp1 != 'nan')
                
                df[f'comp_{component}_exact_match'] = exact_matches
            else:
                print(f"     ⚠️  Component {component} columns not found")
        
        # Add component-level aggregate features
        component_cols = [col for col in df.columns if col.startswith('comp_') and col.endswith('_similarity')]
        if component_cols:
            df['comp_avg_similarity'] = df[component_cols].mean(axis=1)
            df['comp_max_similarity'] = df[component_cols].max(axis=1)
            df['comp_min_similarity'] = df[component_cols].min(axis=1)
        
        elapsed = time.time() - start_time
        print(f"   ✅ Component features completed in {elapsed:.1f}s")
        
        return df
    
    def compute_statistical_features(self, df):
        """Compute statistical features about the addresses."""
        print("\n📊 Computing Statistical Features...")
        
        # Address length features
        df['stat_addr1_length'] = df['address_1'].astype(str).str.len()
        df['stat_addr2_length'] = df['address_2'].astype(str).str.len()
        df['stat_length_diff'] = abs(df['stat_addr1_length'] - df['stat_addr2_length'])
        df['stat_length_ratio'] = np.minimum(df['stat_addr1_length'], df['stat_addr2_length']) / np.maximum(df['stat_addr1_length'], df['stat_addr2_length'])
        df['stat_length_ratio'] = df['stat_length_ratio'].fillna(0)
        
        # Word count features
        df['stat_addr1_words'] = df['address_1'].astype(str).str.split().str.len()
        df['stat_addr2_words'] = df['address_2'].astype(str).str.split().str.len()
        df['stat_word_diff'] = abs(df['stat_addr1_words'] - df['stat_addr2_words'])
        
        # Digit analysis
        df['stat_addr1_digits'] = df['address_1'].astype(str).str.count(r'\d')
        df['stat_addr2_digits'] = df['address_2'].astype(str).str.count(r'\d')
        df['stat_digits_match'] = (df['stat_addr1_digits'] == df['stat_addr2_digits']).astype(int)
        
        print("   ✅ Statistical features completed")
        
        return df
    
    def compute_interaction_features(self, df):
        """Compute interaction features between different similarity types."""
        print("\n🔗 Computing Interaction Features...")
        
        # Get representative scores from each category
        phase2_score = df.get('phase2_jaro_winkler_score', pd.Series([0]*len(df)))
        comp_score = df.get('comp_avg_similarity', pd.Series([0]*len(df)))
        ner_score = df.get('ner_overall_entity_similarity', pd.Series([0]*len(df)))
        
        # Multiplicative interactions
        df['interact_phase2_comp'] = phase2_score * comp_score
        df['interact_phase2_ner'] = phase2_score * ner_score
        df['interact_comp_ner'] = comp_score * ner_score
        df['interact_all_three'] = phase2_score * comp_score * ner_score
        
        # Agreement features
        high_threshold = 0.7
        df['interact_phase2_high'] = (phase2_score > high_threshold).astype(int)
        df['interact_comp_high'] = (comp_score > high_threshold).astype(int)
        df['interact_ner_high'] = (ner_score > high_threshold).astype(int)
        
        df['interact_all_high'] = ((df['interact_phase2_high'] == 1) & 
                                  (df['interact_comp_high'] == 1) & 
                                  (df['interact_ner_high'] == 1)).astype(int)
        
        df['interact_majority_high'] = ((df['interact_phase2_high'] + 
                                       df['interact_comp_high'] + 
                                       df['interact_ner_high']) >= 2).astype(int)
        
        # Score variance (consensus indicator)
        score_matrix = pd.DataFrame({
            'phase2': phase2_score,
            'comp': comp_score,
            'ner': ner_score
        })
        df['interact_score_variance'] = score_matrix.var(axis=1)
        df['interact_score_mean'] = score_matrix.mean(axis=1)
        
        print("   ✅ Interaction features completed")
        
        return df
    
    def engineer_features(self, sample_size=None, include_phase2=True):
        """Main feature engineering pipeline."""
        print("\n" + "="*70)
        print("🚀 PRACTICAL FEATURE ENGINEERING PIPELINE")
        print("="*70)
        
        start_time = time.time()
        
        # Load data
        df = self.load_data()
        
        # Compute Phase 2 scores if requested
        if include_phase2:
            df = self.compute_phase2_scores(df, sample_size)
        else:
            print("\n⏭️  Skipping Phase 2 algorithm computation")
        
        # Add component features
        df = self.compute_component_features(df)
        
        # Add statistical features
        df = self.compute_statistical_features(df)
        
        # Add interaction features
        df = self.compute_interaction_features(df)
        
        # Generate summary
        summary = self._generate_summary(df)
        
        total_time = time.time() - start_time
        print("\n" + "="*70)
        print("✅ FEATURE ENGINEERING COMPLETE")
        print(f"   💫 Total Features: {len(df.columns):,}")
        print(f"   📊 Total Pairs: {len(df):,}")
        print(f"   ⏱️  Processing Time: {total_time:.1f}s")
        print(f"   🚀 Speed: {len(df)/total_time:.0f} pairs/s")
        print("="*70)
        
        return df, summary
    
    def _generate_summary(self, df):
        """Generate feature summary."""
        feature_categories = {
            'Original': [col for col in df.columns if not any(prefix in col for prefix in ['phase2_', 'comp_', 'ner_', 'stat_', 'interact_'])],
            'Phase 2': [col for col in df.columns if col.startswith('phase2_')],
            'Component': [col for col in df.columns if col.startswith('comp_')],
            'NER': [col for col in df.columns if col.startswith('ner_')],
            'Statistical': [col for col in df.columns if col.startswith('stat_')],
            'Interaction': [col for col in df.columns if col.startswith('interact_')]
        }
        
        print("\n📋 FEATURE SUMMARY:")
        for category, features in feature_categories.items():
            if features:
                print(f"   {category}: {len(features)} features")
        
        if 'label' in df.columns:
            print(f"\n📊 LABEL DISTRIBUTION:")
            for label, count in df['label'].value_counts().items():
                print(f"   Label {label}: {count:,} pairs ({count/len(df)*100:.1f}%)")
        
        return {
            'feature_categories': {cat: len(feats) for cat, feats in feature_categories.items()},
            'total_features': len(df.columns),
            'total_pairs': len(df)
        }
    
    def save_results(self, df, summary, prefix="engineered_features"):
        """Save results to files."""
        
        # Save dataset
        csv_file = f"{prefix}.csv"
        df.to_csv(csv_file, index=False, encoding='utf-8')
        print(f"\n💾 Saved dataset: {csv_file}")
        
        # Save summary
        json_file = f"{prefix}_summary.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
        print(f"💾 Saved summary: {json_file}")
        
        return csv_file, json_file

def main():
    """Main execution."""
    print("🚀 Phase 3.3: Practical Feature Engineering")
    print("=" * 50)
    
    # Initialize
    engineer = PracticalFeatureEngineer()
    
    # Test with sample
    print("\n🧪 TESTING WITH SAMPLE (500 pairs)")
    sample_df, sample_summary = engineer.engineer_features(
        sample_size=500, 
        include_phase2=True
    )
    
    # Save sample results
    sample_file, summary_file = engineer.save_results(
        sample_df, sample_summary, "sample_features"
    )
    
    print(f"\n✅ Sample completed! Features: {len(sample_df.columns)}")
    
    # Check if user wants to process full dataset
    print("\n" + "="*50)
    print("🤔 Ready to process full dataset (40,000 pairs)?")
    print("   This will take several minutes...")
    
    proceed = input("Continue with full dataset? (y/N): ").strip().lower()
    
    if proceed == 'y':
        print("\n🚀 PROCESSING FULL DATASET...")
        full_df, full_summary = engineer.engineer_features(
            sample_size=None,
            include_phase2=True
        )
        
        full_file, full_summary_file = engineer.save_results(
            full_df, full_summary, "full_features"
        )
        
        print(f"\n🎉 COMPLETE! Full dataset ready for ML training!")
        print(f"   📁 File: {full_file}")
        print(f"   📊 Features: {len(full_df.columns)}")
    else:
        print("\n⏭️  Use sample for testing Phase 3.4 ML training")

if __name__ == "__main__":
    main()