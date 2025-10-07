#!/usr/bin/env python3
"""
Run Full Normalized Address Parsing
Execute the complete parsing with error handling for Unicode issues.
"""

import pandas as pd
import re
import json
import time
import sys
from pathlib import Path

# Add src directory to path to import normalization module
current_dir = Path(__file__).parent
src_dir = current_dir.parent / 'src'
sys.path.append(str(src_dir))

from normalization import AddressNormalizer

def parse_normalized_address_robust(address: str, normalizer: AddressNormalizer) -> dict:
    """
    Parse an address using normalization preprocessing - robust version.
    """
    
    components = {
        'road': '',
        'house_number': '',
        'city': '',
        'postcode': '',
        'original': address,
        'normalized': ''
    }
    
    try:
        # Step 1: Apply normalization
        normalized_address = normalizer._general_preprocessing(address)
        components['normalized'] = normalized_address
        
        # Step 2: Extract postcode (from original to preserve format)
        postcode_match = re.search(r'\b(\d{4}-\d{3})\b', address)
        if postcode_match:
            components['postcode'] = postcode_match.group(1)
        
        # Step 3: Extract house number
        house_patterns = [
            r',\s*(\d+\s*[a-zA-Z]?)\s*,',
            r',\s*(\d+\s*[a-zA-Z]?)\s*$',
            r'\s(\d+\s*[a-zA-Z]?)\s*,',
            r'\s(\d+\s*[a-zA-Z]?)\s+[a-zA-Z]',
        ]
        
        for text in [address, normalized_address]:
            for pattern in house_patterns:
                match = re.search(pattern, text)
                if match:
                    components['house_number'] = match.group(1).strip()
                    break
            if components['house_number']:
                break
        
        # Step 4: Extract street using normalizer
        street_part = address.split(',')[0] if ',' in address else address
        
        # Remove postcode and house number from street part
        if components['postcode']:
            street_part = street_part.replace(components['postcode'], '')
        if components['house_number']:
            street_part = re.sub(rf"{re.escape(components['house_number'])}", '', street_part)
        
        # Clean and normalize street
        street_part = re.sub(r'\s*,.*$', '', street_part).strip()
        if street_part:
            normalized_street = normalizer.normalize_street(street_part)
            components['road'] = normalized_street
        
        # Step 5: Extract city
        address_parts = [part.strip() for part in address.split(',') if part.strip()]
        
        if len(address_parts) >= 2:
            for i in range(len(address_parts) - 1, -1, -1):
                part = address_parts[i].strip()
                
                # Skip postcode
                if re.match(r'^\d{4}-\d{3}$', part):
                    continue
                
                # Skip house number
                if re.match(r'^\d+\s*[A-Za-z]?$', part):
                    continue
                
                # Skip street prefixes
                if any(part.lower().startswith(prefix.lower()) for prefix in 
                      ['rua', 'avenida', 'travessa', 'estrada', 'largo', 'praca', 'beco', 'tv', 'av']):
                    continue
                
                # Clean and use as city
                part = re.sub(r'\b\d+\s*[A-Za-z]?\b', '', part).strip()
                if part:
                    normalized_city = normalizer._general_preprocessing(part)
                    components['city'] = normalized_city.title() if normalized_city else ''
                    break
        
        return components
        
    except Exception as e:
        print(f"Error parsing address: {e}")
        return components

def run_full_normalized_parsing():
    """Run the full normalized parsing pipeline."""
    
    print("Full Normalized Address Parsing - Gold Standard Dataset")
    print("=" * 60)
    
    # Initialize normalizer
    print("Initializing normalizer...")
    normalizer = AddressNormalizer()
    
    # Load dataset
    print("Loading gold standard dataset...")
    df = pd.read_csv('gold_standard_dataset.csv')
    print(f"Loaded {len(df)} address pairs")
    
    # Extract unique addresses
    print("Extracting unique addresses...")
    addresses_1 = df['address_1'].unique().tolist()
    addresses_2 = df['address_2'].unique().tolist()
    all_addresses = list(set(addresses_1 + addresses_2))
    
    print(f"Total unique addresses to parse: {len(all_addresses)}")
    
    # Parse addresses
    print("\nStarting normalized parsing...")
    start_time = time.time()
    
    parsed_results = []
    for i, address in enumerate(all_addresses):
        if i % 10000 == 0:
            print(f"Progress: {i+1}/{len(all_addresses)} ({(i+1)/len(all_addresses)*100:.1f}%)")
        
        try:
            parsed = parse_normalized_address_robust(address, normalizer)
            parsed_results.append(parsed)
        except Exception as e:
            print(f"Error with address {i}: {e}")
            # Add empty result to maintain alignment
            parsed_results.append({
                'road': '', 'house_number': '', 'city': '', 'postcode': '',
                'original': address, 'normalized': ''
            })
    
    end_time = time.time()
    processing_time = end_time - start_time
    
    print(f"\nCompleted parsing in {processing_time:.2f} seconds")
    print(f"Speed: {len(all_addresses)/processing_time:.0f} addresses/second")
    
    # Create mapping and add to dataframe
    print("\nAdding parsed components to dataset...")
    address_to_parsed = {addr: parsed for addr, parsed in zip(all_addresses, parsed_results)}
    
    # Add columns for address_1
    df['address_1_road'] = df['address_1'].apply(lambda x: address_to_parsed[x]['road'])
    df['address_1_house_number'] = df['address_1'].apply(lambda x: address_to_parsed[x]['house_number'])
    df['address_1_city'] = df['address_1'].apply(lambda x: address_to_parsed[x]['city'])
    df['address_1_postcode'] = df['address_1'].apply(lambda x: address_to_parsed[x]['postcode'])
    df['address_1_normalized'] = df['address_1'].apply(lambda x: address_to_parsed[x]['normalized'])
    
    # Add columns for address_2
    df['address_2_road'] = df['address_2'].apply(lambda x: address_to_parsed[x]['road'])
    df['address_2_house_number'] = df['address_2'].apply(lambda x: address_to_parsed[x]['house_number'])
    df['address_2_city'] = df['address_2'].apply(lambda x: address_to_parsed[x]['city'])
    df['address_2_postcode'] = df['address_2'].apply(lambda x: address_to_parsed[x]['postcode'])
    df['address_2_normalized'] = df['address_2'].apply(lambda x: address_to_parsed[x]['normalized'])
    
    # Calculate quality metrics
    print("\nCalculating quality metrics...")
    
    metrics = {
        'total_addresses': len(parsed_results),
        'postcode_extraction_rate': sum(1 for r in parsed_results if r['postcode']) / len(parsed_results) * 100,
        'house_number_extraction_rate': sum(1 for r in parsed_results if r['house_number']) / len(parsed_results) * 100,
        'road_extraction_rate': sum(1 for r in parsed_results if r['road']) / len(parsed_results) * 100,
        'city_extraction_rate': sum(1 for r in parsed_results if r['city']) / len(parsed_results) * 100,
        'complete_parsing_rate': sum(1 for r in parsed_results if r['road'] and r['city']) / len(parsed_results) * 100,
        'processing_speed': len(all_addresses) / processing_time
    }
    
    print("\n" + "=" * 60)
    print("PARSING QUALITY METRICS (NORMALIZED)")
    print("=" * 60)
    
    for metric, value in metrics.items():
        if metric == 'total_addresses':
            print(f"{metric}: {value}")
        elif metric == 'processing_speed':
            print(f"{metric}: {value:.0f} addresses/second")
        else:
            print(f"{metric}: {value:.1f}%")
    
    # Show sample results
    print("\n" + "=" * 60)
    print("SAMPLE RESULTS")
    print("=" * 60)
    
    for idx in range(min(5, len(df))):
        row = df.iloc[idx]
        print(f"\nExample {idx + 1} (Label: {row['label']}):")
        print(f"Address 1: {row['address_1']}")
        print(f"  Road='{row['address_1_road']}', House='{row['address_1_house_number']}', City='{row['address_1_city']}', Post='{row['address_1_postcode']}'")
        print(f"Address 2: {row['address_2']}")
        print(f"  Road='{row['address_2_road']}', House='{row['address_2_house_number']}', City='{row['address_2_city']}', Post='{row['address_2_postcode']}'")
    
    # Save results
    print("\nSaving results...")
    
    output_path = Path('parsed_gold_standard_normalized.csv')
    df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"Dataset saved to: {output_path}")
    
    metrics_path = Path('normalized_parsing_metrics.json')
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"Metrics saved to: {metrics_path}")
    
    # Component coverage statistics
    coverage_stats = {
        'dataset_info': {
            'total_pairs': len(df),
            'unique_addresses': len(all_addresses),
            'processing_time_seconds': processing_time,
            'addresses_per_second': len(all_addresses) / processing_time
        },
        'extraction_rates': metrics,
        'component_coverage_by_address': {
            'address_1_road_coverage': (df['address_1_road'] != '').sum() / len(df) * 100,
            'address_1_city_coverage': (df['address_1_city'] != '').sum() / len(df) * 100,
            'address_1_postcode_coverage': (df['address_1_postcode'] != '').sum() / len(df) * 100,
            'address_2_road_coverage': (df['address_2_road'] != '').sum() / len(df) * 100,
            'address_2_city_coverage': (df['address_2_city'] != '').sum() / len(df) * 100,
            'address_2_postcode_coverage': (df['address_2_postcode'] != '').sum() / len(df) * 100,
        }
    }
    
    coverage_path = Path('parsing_coverage_analysis.json')
    with open(coverage_path, 'w', encoding='utf-8') as f:
        json.dump(coverage_stats, f, indent=2, ensure_ascii=False)
    print(f"Coverage analysis saved to: {coverage_path}")
    
    # Get normalizer statistics
    norm_stats = normalizer.get_improvement_stats()
    norm_stats_path = Path('normalization_improvement_stats.json')
    with open(norm_stats_path, 'w', encoding='utf-8') as f:
        json.dump(norm_stats, f, indent=2, ensure_ascii=False)
    print(f"Normalization stats saved to: {norm_stats_path}")
    
    print("\n" + "=" * 60)
    print("PHASE 3.1: NORMALIZED ADDRESS PARSING COMPLETE!")
    print("=" * 60)
    print(f"✅ Successfully parsed {len(all_addresses)} unique addresses")
    print(f"✅ Enhanced {len(df)} address pairs with structured components") 
    print(f"✅ Processing speed: {len(all_addresses)/processing_time:.0f} addresses/second")
    print(f"✅ Road extraction rate: {metrics['road_extraction_rate']:.1f}%")
    print(f"✅ City extraction rate: {metrics['city_extraction_rate']:.1f}%")
    print(f"✅ Complete parsing rate: {metrics['complete_parsing_rate']:.1f}%")
    print(f"✅ Applied normalization improvements to {norm_stats['preprocessing_applied']} addresses")
    print("\n🔄 Ready for Phase 3.2: NER Analysis!")
    
    return df, metrics, coverage_stats, norm_stats

if __name__ == "__main__":
    try:
        result = run_full_normalized_parsing()
        print("\n🎉 Phase 3.1 successfully completed!")
    except Exception as e:
        print(f"\n❌ Error during processing: {e}")
        import traceback
        traceback.print_exc()