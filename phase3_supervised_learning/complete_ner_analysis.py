#!/usr/bin/env python3
"""
Final NER Analysis - Complete Phase 3.2 Implementation
Efficient processing for the complete dataset.
"""

import pandas as pd
import spacy
import numpy as np
import json
import time
from pathlib import Path

def run_complete_ner_analysis():
    """Run complete NER analysis efficiently."""
    
    print("PHASE 3.2: NAMED ENTITY RECOGNITION (NER) ANALYSIS - FULL DATASET")
    print("="*70)
    
    # Load dataset
    print("Loading parsed address dataset...")
    df = pd.read_csv("parsed_gold_standard_normalized.csv")
    print(f"Dataset loaded: {len(df):,} address pairs")
    
    # Load spaCy model
    print("Loading Portuguese spaCy model...")
    nlp = spacy.load("pt_core_news_sm")
    print("spaCy model loaded successfully")
    
    # Prepare for batch processing
    batch_size = 5000
    total_batches = (len(df) + batch_size - 1) // batch_size
    
    print(f"\nProcessing in {total_batches} batches of {batch_size} pairs each...")
    
    # Initialize feature storage
    all_ner_features = []
    start_time = time.time()
    
    # Fast entity extraction function
    def fast_extract_entities(text):
        if not text or pd.isna(text):
            return []
        try:
            doc = nlp(str(text))
            return [ent.text.strip().lower() for ent in doc.ents 
                   if ent.label_ in ["LOC", "GPE"] and len(ent.text.strip()) > 1]
        except:
            return []
    
    # Process each batch
    for batch_idx in range(total_batches):
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, len(df))
        
        print(f"Processing batch {batch_idx + 1}/{total_batches} (rows {batch_start+1}-{batch_end})")
        
        batch_features = []
        for idx in range(batch_start, batch_end):
            row = df.iloc[idx]
            
            try:
                # Extract location entities from addresses
                entities1 = set(fast_extract_entities(row['address_1']))
                entities2 = set(fast_extract_entities(row['address_2']))
                
                # Extract from parsed components
                road1_entities = set(fast_extract_entities(row.get('address_1_road', '')))
                road2_entities = set(fast_extract_entities(row.get('address_2_road', '')))
                city1_entities = set(fast_extract_entities(row.get('address_1_city', '')))
                city2_entities = set(fast_extract_entities(row.get('address_2_city', '')))
                
                # Calculate Jaccard similarities
                def jaccard(s1, s2):
                    if not s1 and not s2:
                        return 1.0
                    if not s1 or not s2:
                        return 0.0
                    return len(s1.intersection(s2)) / len(s1.union(s2))
                
                # Create feature vector
                features = {
                    'ner_loc_similarity': jaccard(entities1, entities2),
                    'ner_road_loc_similarity': jaccard(road1_entities, road2_entities),
                    'ner_city_loc_similarity': jaccard(city1_entities, city2_entities),
                    'ner_address1_loc_count': len(entities1),
                    'ner_address2_loc_count': len(entities2),
                    'ner_entity_count_diff': abs(len(entities1) - len(entities2)),
                    'ner_has_location_entities': len(entities1) > 0 or len(entities2) > 0,
                    'ner_both_have_locations': len(entities1) > 0 and len(entities2) > 0,
                    'ner_shared_entities': len(entities1.intersection(entities2)),
                    'ner_total_unique_entities': len(entities1.union(entities2))
                }
                
                batch_features.append(features)
                
            except Exception as e:
                # Add empty features on error
                empty_features = {
                    'ner_loc_similarity': 0.0, 'ner_road_loc_similarity': 0.0,
                    'ner_city_loc_similarity': 0.0, 'ner_address1_loc_count': 0,
                    'ner_address2_loc_count': 0, 'ner_entity_count_diff': 0,
                    'ner_has_location_entities': False, 'ner_both_have_locations': False,
                    'ner_shared_entities': 0, 'ner_total_unique_entities': 0
                }
                batch_features.append(empty_features)
        
        all_ner_features.extend(batch_features)
        
        # Progress update
        elapsed = time.time() - start_time
        processed = len(all_ner_features)
        progress = processed / len(df) * 100
        speed = processed / elapsed if elapsed > 0 else 0
        eta = (len(df) - processed) / speed if speed > 0 else 0
        
        print(f"   Completed: {processed:,}/{len(df):,} ({progress:.1f}%) | "
              f"Speed: {speed:.0f} pairs/s | ETA: {eta:.0f}s")
    
    # Convert to DataFrame and merge
    print(f"\nFinalizing results...")
    ner_df = pd.DataFrame(all_ner_features)
    
    # Add NER features to original dataset
    for col in ner_df.columns:
        df[col] = ner_df[col]
    
    # Calculate final statistics
    end_time = time.time()
    total_time = end_time - start_time
    
    print(f"\nNER ANALYSIS COMPLETED!")
    print(f"   Total processing time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
    print(f"   Processing speed: {len(df)/total_time:.0f} pairs/second")
    print(f"   Features added: {len(ner_df.columns)}")
    
    # Generate comprehensive statistics
    ner_cols = [col for col in df.columns if col.startswith('ner_')]
    
    print(f"\nFEATURE ANALYSIS:")
    print(f"   Total NER features: {len(ner_cols)}")
    
    # Location similarity statistics
    loc_sim = df['ner_loc_similarity']
    high_sim = (loc_sim > 0.5).sum()
    print(f"   High location similarity (>0.5): {high_sim:,}/{len(df):,} ({high_sim/len(df)*100:.1f}%)")
    
    # Entity coverage
    has_entities = df['ner_has_location_entities'].sum()
    both_have = df['ner_both_have_locations'].sum()
    print(f"   Entity coverage: {has_entities:,}/{len(df):,} ({has_entities/len(df)*100:.1f}%)")
    print(f"   Both addresses have entities: {both_have:,}/{len(df):,} ({both_have/len(df)*100:.1f}%)")
    
    # Performance by label
    print(f"   Average similarity by label:")
    for label in sorted(df['label'].unique()):
        label_mean = df[df['label'] == label]['ner_loc_similarity'].mean()
        label_count = (df['label'] == label).sum()
        print(f"     Label {label}: {label_mean:.3f} (n={label_count:,})")
    
    # Save results
    print(f"\nSaving results...")
    output_file = "ner_enhanced_gold_standard.csv"
    df.to_csv(output_file, index=False, encoding='utf-8')
    print(f"   Enhanced dataset: {output_file}")
    
    # Save statistics
    stats = {
        'processing_summary': {
            'total_pairs': len(df),
            'processing_time_seconds': total_time,
            'processing_speed_pairs_per_second': len(df) / total_time,
            'total_ner_features': len(ner_cols)
        },
        'feature_coverage': {
            'pairs_with_entities': int(has_entities),
            'pairs_with_entities_percentage': float(has_entities / len(df) * 100),
            'both_have_entities': int(both_have),
            'both_have_entities_percentage': float(both_have / len(df) * 100)
        },
        'similarity_distribution': {
            'high_similarity_pairs': int((df['ner_loc_similarity'] > 0.5).sum()),
            'medium_similarity_pairs': int(((df['ner_loc_similarity'] >= 0.2) & (df['ner_loc_similarity'] <= 0.5)).sum()),
            'low_similarity_pairs': int((df['ner_loc_similarity'] < 0.2).sum()),
            'mean_location_similarity': float(df['ner_loc_similarity'].mean())
        },
        'feature_statistics': {
            col: {
                'mean': float(df[col].mean()) if df[col].dtype in ['float64', 'int64'] else None,
                'std': float(df[col].std()) if df[col].dtype in ['float64', 'int64'] else None,
                'non_zero_count': int((df[col] != 0).sum()) if df[col].dtype in ['float64', 'int64'] else int(df[col].sum())
            } for col in ner_cols
        }
    }
    
    stats_file = "ner_analysis_statistics.json"
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"   Statistics: {stats_file}")
    
    # Show sample results
    print(f"\nSample results (first 5 pairs):")
    sample_cols = ['label', 'ner_loc_similarity', 'ner_address1_loc_count', 'ner_address2_loc_count', 'ner_both_have_locations']
    print(df[sample_cols].head().to_string(index=False))
    
    print(f"\n" + "="*70)
    print(f"🎯 PHASE 3.2 (NER ANALYSIS) SUCCESSFULLY COMPLETED!")
    print(f"   ✓ Enhanced {len(df):,} address pairs with {len(ner_cols)} NER features")
    print(f"   ✓ Processing speed: {len(df)/total_time:.0f} pairs/second")
    print(f"   ✓ Entity detection rate: {has_entities/len(df)*100:.1f}%")
    print(f"   ✓ Dataset ready for Phase 3.3: Feature Engineering")
    print(f"\n🚀 NEXT: Combine Phase 2 algorithm scores + parsed components + NER features!")
    
    return df, stats

if __name__ == "__main__":
    try:
        result_df, statistics = run_complete_ner_analysis()
        print("\n✅ Phase 3.2 completed successfully!")
    except Exception as e:
        print(f"\n❌ Error in NER analysis: {e}")
        import traceback
        traceback.print_exc()