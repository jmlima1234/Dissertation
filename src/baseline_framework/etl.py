"""
ETL Script for Portuguese Address Data Processing
==================================================

This script implements the complete ETL pipeline for processing OpenStreetMap
address data through normalization and loading into PostGIS and Elasticsearch.

Based on the dissertation research by João Macedo Lima.
Data source: 97MB filtered_osm_buildings.parquet (847K address records)
Target infrastructure: PostGIS (spatial queries) + Elasticsearch (fuzzy search)

Pipeline Steps:
1. Connect to PostGIS and Elasticsearch
2. Create tables/indexes if they don't exist
3. Read data in chunks (10K records at a time)
4. Apply AddressNormalizer to each chunk
5. Batch insert into both systems
6. Track progress and performance metrics

Usage:
    python etl.py [--test] [--chunk-size 10000] [--max-records 1000]
    
    --test: Run with first 1000 records only
    --chunk-size: Records per batch (default: 10000)
    --max-records: Maximum records to process (default: all)
"""

import os
import sys
import time
import argparse
import logging
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
from dotenv import load_dotenv
import re as re

load_dotenv()

import psycopg2
import psycopg2.extras
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk

sys.path.append(str(Path(__file__).resolve().parent / "src"))
from normalization import AddressNormalizer


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('etl_process.log', encoding='utf-8'), 
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class DatabaseConfig:
    """Configuration for database connections"""
    
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


class ETLProcessor:
    """Main ETL processor for Portuguese address data"""
    
    def __init__(self, config: DatabaseConfig):
        self.config = config
        self.postgis_conn = None
        self.es_client = None
        self.normalizer = None
        self.stats = {
            'start_time': None,
            'records_processed': 0,
            'records_inserted_postgis': 0,
            'records_inserted_elasticsearch': 0,
            'chunks_processed': 0,
            'errors': 0,
            'normalization_improvements': 0
        }
    
    def initialize(self) -> bool:
        """Initialize all connections and components"""
        logger.info("[INIT] Initializing ETL processor...")
        
        try:
            # Check for password
            if not self.config.POSTGIS_PASSWORD:
                logger.error("[ERROR] POSTGIS_PASSWORD not set. Please create a .env file or set the environment variable.")
                return False

            # Initialize AddressNormalizer
            logger.info("[INIT] Loading AddressNormalizer...")
            self.normalizer = AddressNormalizer()
            logger.info(f"[SUCCESS] Normalizer loaded with {len(self.normalizer.CANONICAL_CITIES)} canonical cities")
            
            # Connect to PostGIS
            logger.info("[DB] Connecting to PostGIS...")
            self._connect_postgis()
            
            # Connect to Elasticsearch
            logger.info("[SEARCH] Connecting to Elasticsearch...")
            self._connect_elasticsearch()
            
            # Prepare infrastructure
            logger.info("[SETUP] Preparing database infrastructure...")
            self._prepare_infrastructure()
            
            logger.info("[SUCCESS] ETL processor initialized successfully!")
            return True
            
        except Exception as e:
            logger.error(f"[ERROR] Failed to initialize ETL processor: {e}")
            return False
    
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
            self.postgis_conn.autocommit = True
            
            with self.postgis_conn.cursor() as cursor:
                cursor.execute("SELECT version();")
                version = cursor.fetchone()[0]
                logger.info(f"[SUCCESS] PostGIS connected: {version}")
                
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
            
            # Test connection using info() instead of ping() for compatibility
            info = self.es_client.info()
            logger.info(f"[SUCCESS] Elasticsearch connected: {info['version']['number']}")
                
        except Exception as e:
            logger.error(f"[ERROR] Elasticsearch connection failed: {e}")
            raise
    
    def _prepare_infrastructure(self):
        """Create tables and indexes if they don't exist"""
        self._create_postgis_table()
        self._create_elasticsearch_index()
    
    def _create_postgis_table(self):
        """Create the enderecos_normalizados table in PostGIS"""
        try:
            with self.postgis_conn.cursor() as cursor:
                cursor.execute("CREATE EXTENSION IF NOT EXISTS postgis;")
                
                # Check if table exists and drop it if it has the wrong schema
                cursor.execute("""
                    SELECT column_name, data_type 
                    FROM information_schema.columns 
                    WHERE table_name = 'enderecos_normalizados' 
                    AND column_name = 'housenumber_primary'
                """)
                result = cursor.fetchone()
                
                if result and result[1] == 'integer':
                    logger.info("[SETUP] Dropping existing table with incorrect schema...")
                    cursor.execute("DROP TABLE IF EXISTS enderecos_normalizados CASCADE;")
                
                create_table_sql = """
                CREATE TABLE IF NOT EXISTS enderecos_normalizados (
                    id SERIAL PRIMARY KEY,
                    osm_id BIGINT,
                    osm_type VARCHAR(10),
                    geometry GEOMETRY(POINT, 4326),
                    raw_street TEXT,
                    raw_housenumber TEXT,
                    raw_postcode TEXT,
                    raw_city TEXT,
                    raw_country TEXT,
                    street_clean TEXT,
                    housenumber_primary BIGINT,
                    housenumber_specifier TEXT,
                    postcode_clean CHAR(8),
                    city_clean TEXT,
                    original_completeness NUMERIC(3,2),
                    improved_completeness NUMERIC(3,2),
                    normalization_applied BOOLEAN,
                    municipality TEXT,
                    district TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                CREATE INDEX IF NOT EXISTS idx_enderecos_geometry ON enderecos_normalizados USING GIST (geometry);
                CREATE INDEX IF NOT EXISTS idx_enderecos_street_clean ON enderecos_normalizados (street_clean);
                CREATE INDEX IF NOT EXISTS idx_enderecos_city_clean ON enderecos_normalizados (city_clean);
                CREATE INDEX IF NOT EXISTS idx_enderecos_postcode_clean ON enderecos_normalizados (postcode_clean);
                CREATE INDEX IF NOT EXISTS idx_enderecos_municipality ON enderecos_normalizados (municipality);
                """
                cursor.execute(create_table_sql)
                logger.info("[SUCCESS] PostGIS table 'enderecos_normalizados' is ready.")
        except Exception as e:
            logger.error(f"[ERROR] Failed to create PostGIS table: {e}")
            raise
    
    def _create_elasticsearch_index(self):
        """Create the Elasticsearch index with proper mapping"""
        mapping = {
            "settings": {
                "number_of_shards": 1,
                "number_of_replicas": 0,
                "analysis": {
                    "analyzer": {
                        "portuguese_address_analyzer": {
                            "type": "custom", "tokenizer": "standard",
                            "filter": ["lowercase", "asciifolding", "portuguese_stop", "portuguese_stemmer"]
                        }
                    },
                    "filter": {
                        "portuguese_stop": {"type": "stop", "stopwords": ["de", "da", "do", "dos", "das", "e", "a", "o", "as", "os"]},
                        "portuguese_stemmer": {"type": "stemmer", "language": "portuguese"}
                    }
                }
            },
            "mappings": {
                "properties": {
                    "osm_id": {"type": "long"}, "osm_type": {"type": "keyword"},
                    "street_clean": {"type": "text", "analyzer": "portuguese_address_analyzer", "fields": {"keyword": {"type": "keyword"}, "suggest": {"type": "completion", "analyzer": "portuguese_address_analyzer"}}},
                    "city_clean": {"type": "text", "analyzer": "portuguese_address_analyzer", "fields": {"keyword": {"type": "keyword"}}},
                    "postcode_clean": {"type": "keyword"}, "housenumber_primary": {"type": "long"}, "housenumber_specifier": {"type": "keyword"},
                    "raw_street": {"type": "text"}, "raw_housenumber": {"type": "text"}, "raw_postcode": {"type": "text"}, "raw_city": {"type": "text"},
                    "location": {"type": "geo_point"}, "municipality": {"type": "keyword"}, "district": {"type": "keyword"},
                    "original_completeness": {"type": "float"}, "improved_completeness": {"type": "float"}, "normalization_applied": {"type": "boolean"},
                    "created_at": {"type": "date"},
                    "full_address": {"type": "text", "analyzer": "portuguese_address_analyzer"}
                }
            }
        }
        try:
            if self.es_client.indices.exists(index=self.config.ELASTICSEARCH_INDEX):
                logger.info(f"[SETUP] Deleting existing index '{self.config.ELASTICSEARCH_INDEX}'")
                self.es_client.indices.delete(index=self.config.ELASTICSEARCH_INDEX)
            
            self.es_client.indices.create(index=self.config.ELASTICSEARCH_INDEX, body=mapping)
            logger.info(f"[SUCCESS] Elasticsearch index '{self.config.ELASTICSEARCH_INDEX}' is ready.")
        except Exception as e:
            logger.error(f"[ERROR] Failed to create Elasticsearch index: {e}")
            raise
    
    def process_data(self, chunk_size: int = 10000, max_records: Optional[int] = None):
        """Main data processing loop"""
        
        logger.info("[ETL] Starting data processing...")
        self.stats['start_time'] = time.time()
        
        data_file = Path(__file__).resolve().parent / "data" / "filtered_osm_buildings.parquet"
        
        if not data_file.exists():
            raise FileNotFoundError(f"Data file not found: {data_file}")
        
        try:
            # --- FIX: Load the DataFrame and prepare for chunked iteration ---
            logger.info(f"[ETL] Reading data from: {data_file}")
            full_df = pd.read_parquet(data_file, engine='pyarrow')
            logger.info(f"[ETL] Loaded {len(full_df):,} total records from parquet file")

            if max_records:
                logger.info(f"[ETL] Max records set to: {max_records:,}")
                full_df = full_df.head(max_records)

            total_records_to_process = len(full_df)
            
            # --- FIX: Iterate in chunks using a more standard Python approach ---
            for i in range(0, total_records_to_process, chunk_size):
                chunk_df = full_df.iloc[i:i + chunk_size]
                chunk_num = (i // chunk_size) + 1
                
                logger.info(f"[ETL] Processing chunk {chunk_num}: {len(chunk_df):,} records (rows {i}-{i+len(chunk_df)-1})")
                
                self._process_chunk(chunk_df, chunk_num)
                
                self.stats['records_processed'] += len(chunk_df)
                self.stats['chunks_processed'] = chunk_num
                
                elapsed = time.time() - self.stats['start_time']
                rate = self.stats['records_processed'] / elapsed if elapsed > 0 else 0
                logger.info(f"[PROGRESS] {self.stats['records_processed']:,} records in {elapsed:.1f}s ({rate:.0f} recs/sec)")
        
        except Exception as e:
            logger.error(f"[ERROR] Error during data processing: {e}")
            raise
        
        finally:
            self._log_final_stats()

    
    def _process_chunk(self, chunk_df: pd.DataFrame, chunk_num: int):
        """Process a single chunk of data"""
        chunk_start_time = time.time()
        postgis_records, elasticsearch_docs = [], []
        
        self.normalizer.reset_stats()
        
        try:
            for _, row in chunk_df.iterrows():
                try:
                    raw_record = self._extract_address_components(row)
                    if not raw_record: continue
                    
                    normalized_record = self.normalizer.normalize_address_record(raw_record)
                    if not normalized_record: 
                        logger.debug(f"[DEBUG] Normalization returned None for raw_record: {raw_record}")
                        continue  # Skip if normalization returns None
                    
                    postgis_record = self._prepare_postgis_record(row, raw_record, normalized_record)
                    elasticsearch_doc = self._prepare_elasticsearch_doc(row, raw_record, normalized_record)
                    
                    if postgis_record: postgis_records.append(postgis_record)
                    if elasticsearch_doc: elasticsearch_docs.append(elasticsearch_doc)
                
                except Exception as e:
                    logger.warning(f"[WARN] Error processing record {row.get('osm_id', 'N/A')}: {e}")
                    self.stats['errors'] += 1
                    continue
            
            if postgis_records: self._batch_insert_postgis(postgis_records)
            if elasticsearch_docs: self._batch_insert_elasticsearch(elasticsearch_docs)
            
            chunk_time = time.time() - chunk_start_time
            logger.info(f"[SUCCESS] Chunk {chunk_num} completed in {chunk_time:.2f}s: {len(postgis_records)} to PostGIS, {len(elasticsearch_docs)} to Elasticsearch")
        
        except Exception as e:
            logger.error(f"[ERROR] Error processing chunk {chunk_num}: {e}")
            raise
    
    def _extract_address_components(self, row: pd.Series) -> Optional[Dict[str, str]]:
        """
        Extracts address components from an OSM data row using a robust parsing strategy.
        """
        tags_column = None
        if 'tags' in row and pd.notna(row['tags']):
            tags_column = 'tags'
        elif 'parsed_tags' in row and pd.notna(row['parsed_tags']):
            tags_column = 'parsed_tags'
        
        if tags_column is None:
            return None
        
        tags = {}
        tags_value = row[tags_column]
        
        if isinstance(tags_value, dict):
            tags = tags_value
        elif isinstance(tags_value, str):
            try:
                pattern = re.compile(r"'([^']+)':\s*'((?:[^']|'')*)'")
                matches = pattern.findall(tags_value)
                tags = {key: value.replace("''", "'") for key, value in matches}
            except Exception:
                logger.warning(f"[WARN] Could not parse tags string: {tags_value[:100]}...")
                return None

        # Extract only the keys that start with 'addr:' and have a non-empty value
        raw_record = {key: str(value) for key, value in tags.items() if key.startswith('addr:') and value}
        
        return raw_record if raw_record else None
    
    def _prepare_postgis_record(self, row: pd.Series, raw_record: Dict, normalized_record: Dict) -> Optional[Tuple]:
        """Prepare a record for PostGIS insertion"""
        try:
            if not normalized_record:
                return None
                
            nc = normalized_record
            hn_data = nc.get('housenumber_clean', {}) or {}
            quality = nc.get('quality_metrics', {}) or {}
            
            geom_wkt = None
            if 'geometry' in row and pd.notna(row['geometry']):
                try:
                    geom = row['geometry']
                    if hasattr(geom, 'wkt'): geom_wkt = geom.wkt
                    else: geom_wkt = str(geom)
                except: pass

            # Validate housenumber_primary to prevent overflow issues
            housenumber_primary = hn_data.get('numero_primario')
            if housenumber_primary is not None:
                # More reasonable validation: Portuguese house numbers typically range 1-9999
                # Allow up to 999999 to accommodate edge cases but reject obvious parsing errors
                if housenumber_primary > 999999 or housenumber_primary < 1:
                    # Only log occasionally to reduce spam
                    if housenumber_primary > 99999 or housenumber_primary == 0:
                        logger.debug(f"[DEBUG] Filtered unrealistic house number {housenumber_primary} from '{raw_record.get('addr:housenumber', 'N/A')}'")
                    housenumber_primary = None

            return (
                row.get('id', None), row.get('type', 'unknown'), geom_wkt,
                raw_record.get('addr:street'), raw_record.get('addr:housenumber'), raw_record.get('addr:postcode'), raw_record.get('addr:city'), raw_record.get('addr:country'),
                nc.get('street_clean'), housenumber_primary, hn_data.get('especificador'), nc.get('postcode_clean'), nc.get('city_clean'),
                quality.get('original_completeness'), quality.get('improved_completeness'), quality.get('preprocessing_applied'),
                None, None # municipality, district
            )
        except Exception as e:
            logger.warning(f"[WARN] Error preparing PostGIS record: {e}")
            return None
    
    def _prepare_elasticsearch_doc(self, row: pd.Series, raw_record: Dict, normalized_record: Dict) -> Optional[Dict]:
        """Prepare a document for Elasticsearch indexing"""
        try:
            if not normalized_record:
                return None
                
            nc = normalized_record
            hn_data = nc.get('housenumber_clean', {}) or {}
            quality = nc.get('quality_metrics', {}) or {}
            
            location = None
            if 'geometry' in row and pd.notna(row['geometry']):
                try:
                    geom = row['geometry']
                    if hasattr(geom, 'x') and hasattr(geom, 'y'): 
                        # Validate latitude and longitude ranges
                        if -90 <= geom.y <= 90 and -180 <= geom.x <= 180:
                            location = {"lat": geom.y, "lon": geom.x}
                        else:
                            logger.warning(f"[WARN] Invalid coordinates: lat={geom.y}, lon={geom.x}")
                except: pass
            
            # Validate housenumber_primary for Elasticsearch
            housenumber_primary = hn_data.get('numero_primario')
            if housenumber_primary is not None:
                # More reasonable validation: Portuguese house numbers typically range 1-9999
                # Allow up to 999999 to accommodate edge cases but reject obvious parsing errors
                if housenumber_primary > 999999 or housenumber_primary < 1:
                    # Only log occasionally to reduce spam
                    if housenumber_primary > 99999 or housenumber_primary == 0:
                        logger.debug(f"[DEBUG] Filtered unrealistic house number {housenumber_primary} from '{raw_record.get('addr:housenumber', 'N/A')}'")
                    housenumber_primary = None
            
            # Clean and validate text fields to prevent indexing issues
            def clean_text_field(value):
                if value is None:
                    return None
                text = str(value).strip()
                if not text:
                    return None
                text = text.replace('\x00', '')
                # Truncate very long strings to prevent issues
                if len(text) > 32766:  # Elasticsearch text field limit
                    text = text[:32766]
                    logger.warning(f"[WARN] Truncated long text field: {text[:50]}...")
                return text
            
            parts = [
                clean_text_field(nc.get('street_clean')), 
                str(housenumber_primary) if housenumber_primary else '', 
                clean_text_field(nc.get('postcode_clean')), 
                clean_text_field(nc.get('city_clean'))
            ]
            full_address = " ".join(filter(None, parts))

            # Validate numeric fields to prevent NaN/None issues
            def clean_numeric_field(value):
                if value is None or pd.isna(value):
                    return None
                try:
                    return float(value) if not pd.isna(float(value)) else None
                except (ValueError, TypeError):
                    return None

            doc = {
                "osm_id": row.get('id'), 
                "osm_type": clean_text_field(row.get('type', 'unknown')),
                "street_clean": clean_text_field(nc.get('street_clean')), 
                "city_clean": clean_text_field(nc.get('city_clean')), 
                "postcode_clean": clean_text_field(nc.get('postcode_clean')),
                "housenumber_primary": housenumber_primary, 
                "housenumber_specifier": clean_text_field(hn_data.get('especificador')),
                "raw_street": clean_text_field(raw_record.get('addr:street')), 
                "raw_housenumber": clean_text_field(raw_record.get('addr:housenumber')),
                "raw_postcode": clean_text_field(raw_record.get('addr:postcode')), 
                "raw_city": clean_text_field(raw_record.get('addr:city')),
                "location": location,
                "original_completeness": clean_numeric_field(quality.get('original_completeness')), 
                "improved_completeness": clean_numeric_field(quality.get('improved_completeness')),
                "normalization_applied": bool(quality.get('preprocessing_applied')) if quality.get('preprocessing_applied') is not None else None,
                "full_address": clean_text_field(full_address),
                "created_at": datetime.now().isoformat()
            }
            
            return doc
        except Exception as e:
            logger.warning(f"[WARN] Error preparing Elasticsearch document: {e}")
            return None
    
    def _batch_insert_postgis(self, records: List[Tuple]):
        """Batch insert records into PostGIS"""
        insert_sql = """
        INSERT INTO enderecos_normalizados (
            osm_id, osm_type, geometry,
            raw_street, raw_housenumber, raw_postcode, raw_city, raw_country,
            street_clean, housenumber_primary, housenumber_specifier, 
            postcode_clean, city_clean,
            original_completeness, improved_completeness, normalization_applied,
            municipality, district
        ) VALUES %s
        """
        try:
            with self.postgis_conn.cursor() as cursor:
                psycopg2.extras.execute_values(cursor, insert_sql, records, page_size=1000)
                self.stats['records_inserted_postgis'] += len(records)
        except Exception as e:
            logger.error(f"[ERROR] PostGIS batch insert failed: {e}")
            raise
    
    def _batch_insert_elasticsearch(self, docs: List[Dict]):
        """Batch insert documents into Elasticsearch"""
        try:
            actions = [{"_index": self.config.ELASTICSEARCH_INDEX, "_source": doc} for doc in docs]
            
            from elasticsearch.helpers import BulkIndexError
            try:
                es_client_with_options = self.es_client.options(request_timeout=60)
                success, failed = bulk(es_client_with_options, actions, chunk_size=1000)
                self.stats['records_inserted_elasticsearch'] += success
                if failed:
                    logger.warning(f"[WARN] Elasticsearch: {len(failed)} documents failed to insert")
            except BulkIndexError as e:
                logger.error(f"[ERROR] Elasticsearch bulk indexing failed with {len(e.errors)} errors:")
                for i, error in enumerate(e.errors[:5]): 
                    logger.error(f"  Error {i+1}: {error}")
                if len(e.errors) > 5:
                    logger.error(f"  ... and {len(e.errors) - 5} more errors")
                raise
                
        except Exception as e:
            logger.error(f"[ERROR] Elasticsearch batch insert failed: {e}")
            raise
    
    def _log_final_stats(self):
        """Log final processing statistics"""
        elapsed = time.time() - self.stats['start_time'] if self.stats['start_time'] else 0
        logger.info("=" * 60)
        logger.info("ETL PROCESSING COMPLETED")
        logger.info("=" * 60)
        logger.info(f"Total time: {elapsed:.1f} seconds")
        logger.info(f"Chunks processed: {self.stats['chunks_processed']:,}")
        logger.info(f"Records processed: {self.stats['records_processed']:,}")
        logger.info(f"PostGIS inserts: {self.stats['records_inserted_postgis']:,}")
        logger.info(f"Elasticsearch inserts: {self.stats['records_inserted_elasticsearch']:,}")
        logger.info(f"Errors: {self.stats['errors']:,}")
        
        if elapsed > 0:
            rate = self.stats['records_processed'] / elapsed
            logger.info(f"Average rate: {rate:.0f} records/second")
        
        if self.normalizer:
            norm_stats = self.normalizer.get_improvement_stats()
            logger.info(f"Normalization improvements: {norm_stats.get('preprocessing_applied', 0):,}")
        
        logger.info("=" * 60)
    
    def close(self):
        """Close all database connections"""
        if self.postgis_conn:
            self.postgis_conn.close()
            logger.info("[DB] PostGIS connection closed")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='ETL processor for Portuguese address data')
    parser.add_argument('--test', action='store_true', help='Test mode: process only first 1000 records')
    parser.add_argument('--chunk-size', type=int, default=10000, help='Records per chunk (default: 10000)')
    parser.add_argument('--max-records', type=int, default=None, help='Maximum records to process (default: all)')
    
    args = parser.parse_args()
    
    if args.test:
        args.max_records = 1000
        args.chunk_size = min(args.chunk_size, 1000)
        logger.info("[INFO] Running in TEST mode")
    
    config = DatabaseConfig()
    processor = ETLProcessor(config)
    
    try:
        if not processor.initialize():
            logger.error("[ERROR] Failed to initialize ETL processor")
            return 1
        
        processor.process_data(chunk_size=args.chunk_size, max_records=args.max_records)
        
        logger.info("[SUCCESS] ETL process completed successfully!")
        return 0
        
    except KeyboardInterrupt:
        logger.info("[INFO] Process interrupted by user")
        return 1
        
    except Exception as e:
        logger.error(f"[ERROR] ETL process failed during execution: {e}", exc_info=True)
        return 1
        
    finally:
        processor.close()


if __name__ == "__main__":
    sys.exit(main())
