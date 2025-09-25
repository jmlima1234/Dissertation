"""
Portuguese Address Search API - Hybrid Architecture
==================================================

This module implements the core search functionality for the Portuguese address
baseline system using a hybrid PostGIS + Elasticsearch architecture.

The search process follows this optimized pipeline:
1. Normalize user query using AddressNormalizer
2. Query Elasticsearch for ranked address IDs based on relevance
3. Use these IDs for precise PostGIS lookup to get full geometrical data
4. Return structured results with confidence scores and spatial data

Based on benchmark analysis showing hybrid architecture provides:
- Mean speed ratio of 3.63x for complex queries
- +12.1% accuracy improvement for fuzzy/typo queries  
- +51.90 average relevance score for multi-field searches

Usage:
    from search_api import HybridAddressSearch
    
    search = HybridAddressSearch()
    results = search.search("rua augusta 100 lisboa")
"""

import os
import sys
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
import time
from datetime import datetime
from dataclasses import dataclass
from dotenv import load_dotenv

# Database imports
import psycopg2
import psycopg2.extras
from elasticsearch import Elasticsearch

# Add src directory to path for normalization
sys.path.append(str(Path(__file__).resolve().parent / "src"))
from normalization import AddressNormalizer

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('search_api.log')
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Structured search result with all relevant address information"""
    osm_id: int
    osm_type: str
    address_full: str
    street_clean: str
    city_clean: str
    postcode_clean: Optional[str] = None
    housenumber_primary: Optional[int] = None
    housenumber_specifier: Optional[str] = None
    municipality: Optional[str] = None
    district: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    elasticsearch_score: Optional[float] = None
    confidence_score: Optional[float] = None
    raw_data: Optional[Dict] = None


class DatabaseConfig:
    """Configuration for database connections - matches ETL configuration"""
    
    # PostGIS Configuration
    POSTGIS_HOST = os.getenv('POSTGIS_HOST', 'localhost')
    POSTGIS_PORT = os.getenv('POSTGIS_PORT', '5432')
    POSTGIS_DB = os.getenv('POSTGIS_DB', 'portugal_addresses')
    POSTGIS_USER = os.getenv('POSTGIS_USER', 'postgres')
    POSTGIS_PASSWORD = os.getenv('POSTGIS_PASSWORD')
    
    # Elasticsearch Configuration
    ELASTICSEARCH_HOST = os.getenv('ELASTICSEARCH_HOST', 'localhost')
    ELASTICSEARCH_PORT = os.getenv('ELASTICSEARCH_PORT', '9200')
    ELASTICSEARCH_INDEX = os.getenv('ELASTICSEARCH_INDEX', 'portugal_addresses')


class HybridAddressSearch:
    """
    Hybrid Address Search System
    
    Implements the optimized search architecture combining Elasticsearch's 
    fuzzy text matching capabilities with PostGIS's precise spatial queries.
    """
    
    def __init__(self, config: Optional[DatabaseConfig] = None):
        """
        Initialize the hybrid search system
        
        Args:
            config: Database configuration. Uses default if None.
        """
        self.config = config or DatabaseConfig()
        self.postgis_conn = None
        self.es_client = None
        self.normalizer = None
        self.search_stats = {
            'total_searches': 0,
            'avg_elasticsearch_time': 0.0,
            'avg_postgis_time': 0.0,
            'avg_total_time': 0.0
        }
        
        # Initialize connections
        self._initialize()
    
    def _initialize(self):
        """Initialize all components and connections"""
        logger.info("[INIT] Initializing Hybrid Address Search system...")
        
        # Initialize normalizer
        self.normalizer = AddressNormalizer()
        logger.info("[INIT] Address normalizer loaded")
        
        # Connect to databases
        self._connect_postgis()
        self._connect_elasticsearch()
        
        # Verify setup
        self._verify_setup()
        
        logger.info("[SUCCESS] Hybrid search system ready!")
    
    def _connect_postgis(self):
        """Establish PostGIS connection"""
        try:
            connection_string = (
                f"host={self.config.POSTGIS_HOST} "
                f"port={self.config.POSTGIS_PORT} "
                f"dbname={self.config.POSTGIS_DB} "
                f"user={self.config.POSTGIS_USER} "
                f"password={self.config.POSTGIS_PASSWORD}"
            )
            
            self.postgis_conn = psycopg2.connect(connection_string)
            self.postgis_conn.autocommit = True  # Enable autocommit to avoid transaction issues
            
            with self.postgis_conn.cursor() as cursor:
                cursor.execute("SELECT version();")
                version = cursor.fetchone()[0]
                logger.info(f"[SUCCESS] PostGIS connected: {version[:50]}...")
                
        except Exception as e:
            logger.error(f"[ERROR] PostGIS connection failed: {e}")
            raise
    
    def _connect_elasticsearch(self):
        """Establish Elasticsearch connection"""
        try:
            self.es_client = Elasticsearch([{
                'host': self.config.ELASTICSEARCH_HOST,
                'port': int(self.config.ELASTICSEARCH_PORT),
                'scheme': 'http'
            }])
            
            # Test connection
            info = self.es_client.info()
            logger.info(f"[SUCCESS] Elasticsearch connected: {info['version']['number']}")
                
        except Exception as e:
            logger.error(f"[ERROR] Elasticsearch connection failed: {e}")
            raise
    
    def _verify_setup(self):
        """Verify that both databases have data"""
        try:
            # Check PostGIS table
            with self.postgis_conn.cursor() as cursor:
                cursor.execute("SELECT COUNT(*) FROM enderecos_normalizados;")
                postgis_count = cursor.fetchone()[0]
                logger.info(f"[VERIFY] PostGIS table has {postgis_count:,} records")
            
            # Check Elasticsearch index
            es_count = self.es_client.count(index=self.config.ELASTICSEARCH_INDEX)['count']
            logger.info(f"[VERIFY] Elasticsearch index has {es_count:,} documents")
            
            if postgis_count == 0 or es_count == 0:
                logger.warning("[WARNING] One or both databases appear to be empty!")
                
        except Exception as e:
            logger.error(f"[ERROR] Setup verification failed: {e}")
            raise
    
    def search(self, 
               query: str, 
               max_results: int = 10,
               min_score: float = 0.1,
               include_raw: bool = False) -> List[SearchResult]:
        """
        Main search function implementing the hybrid architecture
        
        Args:
            query: Raw user query string
            max_results: Maximum number of results to return
            min_score: Minimum Elasticsearch relevance score threshold
            include_raw: Whether to include raw database records in results
            
        Returns:
            List of SearchResult objects ordered by relevance
        """
        start_time = time.time()
        
        logger.info(f"[SEARCH] Processing query: '{query}'")
        
        # Step 1: Normalize the query
        normalized_query = self._normalize_query(query)
        logger.info(f"[SEARCH] Normalized query: '{normalized_query}'")
        
        # Step 2: Query Elasticsearch for ranked address IDs
        es_start = time.time()
        elasticsearch_results = self._elasticsearch_search(normalized_query, max_results * 2, min_score)
        es_time = time.time() - es_start
        
        if not elasticsearch_results:
            logger.info("[SEARCH] No results from Elasticsearch")
            return []
        
        logger.info(f"[SEARCH] Elasticsearch returned {len(elasticsearch_results)} candidates in {es_time:.3f}s")
        
        # Step 3: Get precise records from PostGIS
        postgis_start = time.time()
        final_results = self._postgis_lookup(elasticsearch_results, include_raw)
        postgis_time = time.time() - postgis_start
        
        # Update statistics
        total_time = time.time() - start_time
        self._update_search_stats(es_time, postgis_time, total_time)
        
        logger.info(f"[SEARCH] Completed search in {total_time:.3f}s, returning {len(final_results)} results")
        
        return final_results[:max_results]
    
    def _normalize_query(self, query: str) -> str:
        """
        Normalize the user query using AddressNormalizer
        
        This handles the common issues found in Portuguese addresses:
        - Case normalization
        - Accent removal
        - Street prefix standardization
        - Common abbreviations
        """
        if not query or not query.strip():
            return ""
        
        # Use the normalizer's general preprocessing
        normalized = self.normalizer._general_preprocessing(query)
        
        # Apply street normalization techniques for better matching
        tokens = normalized.split()
        processed_tokens = []
        
        for token in tokens:
            # Check for street prefixes
            token_clean = token.rstrip('.')
            if token_clean in self.normalizer.PREFIX_MAP_EXPANDED:
                processed_tokens.append(self.normalizer.PREFIX_MAP_EXPANDED[token_clean])
            elif token_clean not in self.normalizer.STOP_WORDS:
                processed_tokens.append(token)
        
        return ' '.join(processed_tokens)
    
    def _elasticsearch_search(self, 
                            normalized_query: str, 
                            size: int, 
                            min_score: float) -> List[Dict]:
        """
        Search Elasticsearch for relevant address IDs with confidence scores
        
        Uses multi-field search across street, city, postcode, and full address
        with Portuguese-specific analyzer and fuzzy matching capabilities.
        """
        try:
            # Build multi-field query with fuzzy matching
            search_body = {
                "query": {
                    "bool": {
                        "should": [
                            # Exact matches get highest boost
                            {
                                "multi_match": {
                                    "query": normalized_query,
                                    "fields": ["street_clean^3", "city_clean^2", "full_address^1"],
                                    "type": "phrase",
                                    "boost": 5.0
                                }
                            },
                            # Fuzzy matches for typo tolerance
                            {
                                "multi_match": {
                                    "query": normalized_query,
                                    "fields": ["street_clean^2", "city_clean^1.5", "full_address^1"],
                                    "type": "best_fields",
                                    "fuzziness": "AUTO",
                                    "prefix_length": 2,
                                    "boost": 2.0
                                }
                            },
                            # Cross-field matching for complex queries
                            {
                                "multi_match": {
                                    "query": normalized_query,
                                    "fields": ["street_clean", "city_clean", "postcode_clean", "full_address"],
                                    "type": "cross_fields",
                                    "operator": "and",
                                    "boost": 1.0
                                }
                            },
                            # Postcode-only queries
                            {
                                "term": {
                                    "postcode_clean": {
                                        "value": normalized_query,
                                        "boost": 3.0
                                    }
                                }
                            }
                        ],
                        "minimum_should_match": 1
                    }
                },
                "size": size,
                "min_score": min_score,
                "_source": ["osm_id", "osm_type", "street_clean", "city_clean", 
                           "postcode_clean", "housenumber_primary", "housenumber_specifier"]
            }
            
            response = self.es_client.search(
                index=self.config.ELASTICSEARCH_INDEX,
                body=search_body
            )
            
            results = []
            for hit in response['hits']['hits']:
                results.append({
                    'osm_id': hit['_source']['osm_id'],
                    'osm_type': hit['_source']['osm_type'],
                    'score': hit['_score'],
                    'source': hit['_source']
                })
            
            return results
            
        except Exception as e:
            logger.error(f"[ERROR] Elasticsearch search failed: {e}")
            return []
    
    def _postgis_lookup(self, 
                       elasticsearch_results: List[Dict], 
                       include_raw: bool = False) -> List[SearchResult]:
        """
        Perform precise PostGIS lookup using Elasticsearch result IDs
        
        Gets complete address records with spatial data for the relevant IDs
        identified by Elasticsearch ranking.
        """
        if not elasticsearch_results:
            return []
        
        try:
            # Extract OSM IDs and types for lookup
            lookup_pairs = [(r['osm_id'], r['osm_type']) for r in elasticsearch_results]
            id_to_score = {(r['osm_id'], r['osm_type']): r['score'] for r in elasticsearch_results}
            
            # Build parameterized query for security
            placeholders = ','.join(['(%s,%s)'] * len(lookup_pairs))
            query = f"""
                SELECT 
                    osm_id, osm_type, street_clean, city_clean, postcode_clean,
                    housenumber_primary, housenumber_specifier,
                    municipality, district,
                    ST_Y(geometry) as latitude, ST_X(geometry) as longitude,
                    original_completeness, improved_completeness, normalization_applied
                FROM enderecos_normalizados 
                WHERE (osm_id, osm_type) IN ({placeholders})
                ORDER BY osm_id;
            """
            
            # Flatten lookup pairs for parameter substitution
            params = [item for pair in lookup_pairs for item in pair]
            
            with self.postgis_conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
                cursor.execute(query, params)
                rows = cursor.fetchall()
            
            # Convert to SearchResult objects
            results = []
            for row in rows:
                lookup_key = (row['osm_id'], row['osm_type'])
                es_score = id_to_score.get(lookup_key, 0.0)
                
                # Build full address string
                address_parts = []
                if row['street_clean']:
                    if row['housenumber_primary']:
                        address_parts.append(f"{row['street_clean']} {row['housenumber_primary']}")
                        if row['housenumber_specifier']:
                            address_parts[-1] += f" {row['housenumber_specifier']}"
                    else:
                        address_parts.append(row['street_clean'])
                
                if row['city_clean']:
                    address_parts.append(row['city_clean'])
                
                if row['postcode_clean']:
                    address_parts.append(row['postcode_clean'])
                
                full_address = ', '.join(address_parts)
                
                # Calculate confidence score (weighted combination of ES score and completeness)
                completeness = float(row['improved_completeness']) if row['improved_completeness'] else 0.5
                confidence = (es_score * 0.7) + (completeness * 0.3)
                
                result = SearchResult(
                    osm_id=row['osm_id'],
                    osm_type=row['osm_type'],
                    address_full=full_address,
                    street_clean=row['street_clean'],
                    city_clean=row['city_clean'],
                    postcode_clean=row['postcode_clean'],
                    housenumber_primary=row['housenumber_primary'],
                    housenumber_specifier=row['housenumber_specifier'],
                    municipality=row['municipality'],
                    district=row['district'],
                    latitude=row['latitude'],
                    longitude=row['longitude'],
                    elasticsearch_score=es_score,
                    confidence_score=confidence,
                    raw_data=dict(row) if include_raw else None
                )
                
                results.append(result)
            
            # Sort by confidence score (Elasticsearch score prioritized)
            results.sort(key=lambda x: x.elasticsearch_score, reverse=True)
            
            return results
            
        except Exception as e:
            logger.error(f"[ERROR] PostGIS lookup failed: {e}")
            return []
    
    def _update_search_stats(self, es_time: float, postgis_time: float, total_time: float):
        """Update search performance statistics"""
        self.search_stats['total_searches'] += 1
        
        # Running average calculation
        n = self.search_stats['total_searches']
        self.search_stats['avg_elasticsearch_time'] = (
            (self.search_stats['avg_elasticsearch_time'] * (n-1) + es_time) / n
        )
        self.search_stats['avg_postgis_time'] = (
            (self.search_stats['avg_postgis_time'] * (n-1) + postgis_time) / n
        )
        self.search_stats['avg_total_time'] = (
            (self.search_stats['avg_total_time'] * (n-1) + total_time) / n
        )
    
    def get_search_stats(self) -> Dict:
        """Get search performance statistics"""
        return self.search_stats.copy()
    
    def close(self):
        """Close database connections"""
        if self.postgis_conn:
            self.postgis_conn.close()
            logger.info("[CLEANUP] PostGIS connection closed")
        
        if self.es_client:
            # Elasticsearch client doesn't need explicit closing
            logger.info("[CLEANUP] Elasticsearch client cleaned up")


# Example usage and testing
if __name__ == '__main__':
    """
    Example usage demonstrating the hybrid search capabilities
    """
    print("=== Portuguese Address Search API - Test Suite ===\n")
    
    # Initialize search system
    try:
        search_system = HybridAddressSearch()
        
        # Test queries based on benchmark analysis findings
        test_queries = [
            "rua augusta lisboa",           # Exact street + city
            "r augusta 100 lisboa",        # Abbreviation + number
            "avenida liberdade porto",      # Common street name
            "1000-001",                     # Postcode only
            "rua agusta, lisbon",          # Typo + variant spelling
            "av liberdade 123",            # Abbreviation query
            "coimbra",                      # City only
            "rua da misericordia faro",     # Complete address
        ]
        
        for i, query in enumerate(test_queries, 1):
            print(f"--- Test {i}: '{query}' ---")
            
            start_time = time.time()
            results = search_system.search(query, max_results=3)
            search_time = time.time() - start_time
            
            print(f"Results: {len(results)} addresses found in {search_time:.3f}s")
            
            for j, result in enumerate(results, 1):
                print(f"  {j}. {result.address_full}")
                print(f"     Confidence: {result.confidence_score:.3f} | ES Score: {result.elasticsearch_score:.3f}")
                if result.latitude and result.longitude:
                    print(f"     Location: {result.latitude:.6f}, {result.longitude:.6f}")
                print()
            
            if not results:
                print("  No results found.\n")
        
        # Print performance statistics
        stats = search_system.get_search_stats()
        print("=== Performance Statistics ===")
        print(f"Total searches: {stats['total_searches']}")
        print(f"Avg Elasticsearch time: {stats['avg_elasticsearch_time']:.3f}s")
        print(f"Avg PostGIS time: {stats['avg_postgis_time']:.3f}s")
        print(f"Avg total time: {stats['avg_total_time']:.3f}s")
        
        # Cleanup
        search_system.close()
        
    except Exception as e:
        print(f"Error during testing: {e}")
        logger.error(f"Test suite failed: {e}")