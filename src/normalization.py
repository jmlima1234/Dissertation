import re
import string
import unicodedata
import pandas as pd
from thefuzz import process
from typing import Dict, Optional, Union, List
from collections import Counter
from pathlib import Path
import os

class AddressNormalizer:
    """
    Main Class for Portuguese Address Normalization.
    Implements a normalization pipeline based on literature and EDA.
    """
    
    def __init__(self):
        """Initialize the normalizer with EDA-identified patterns."""

        self.eda_stats = {
            'total_addresses': 847000,
            'completeness_rate': 0.30,
            'prefix_variations': 42,
            'capitalization_issues_rate': 0.67,
            'encoding_issues_rate': 0.23,
            'invalid_postal_rate': 0.22
        }
        
        self.improvement_stats = {
            'preprocessing_applied': 0,
            'case_fixes': 0,
            'punctuation_removed': 0,
            'whitespace_normalized': 0,
            'encoding_fixes': 0
        }

        self.STREET_PREFIX_MAP = {
            "avenida": ["av", "avn", "avend", "avnda"], "rua": ["r"], "praca": ["prc", "pca"],
            "largo": ["lg", "lgo"], "travessa": ["tv", "trav"], "alameda": ["al"],
            "calcada": ["calc"], "estrada": ["estr"], "beco": ["bc"], "jardim": ["jd"],
            "rodovia": ["rod"], "via": ["v"], "quadra": ["qd", "quad"], "lote": ["lt"],
            "conjunto": ["cj", "conj"], "setor": ["st", "set"], "zona": ["zn"],
            "distrito": ["dt", "dist"], "vila": ["vl"], "bairro": ["br"],
            "residencial": ["res", "resid"], "comercial": ["com", "comerc"],
            "industrial": ["ind", "indust"], "urbano": ["urb"], "rural": ["rur"],
            "centro": ["ctr", "cent"], "nucleo": ["nc, nuc"], "area": ["ar"],
            "complexo": ["cplx, comp"], "parque": ["pq", "prq"], "sitio": ["st"],
            "fazenda": ["faz", "fzd"], "chacara": ["ch", "chac"], "colonia": ["col"],
            "povoado": ["pov", "pvd"], "distrito": ["dst"], "municipio": ["mun"],
            "regiao": ["reg"], "territorio": ["terr"], "localidade": ["loc"]
        }

        self.PREFIX_MAP_EXPANDED = {abbr: full for full, abbrs in self.STREET_PREFIX_MAP.items() for abbr in abbrs}
        for full in self.STREET_PREFIX_MAP: self.PREFIX_MAP_EXPANDED[full] = full
        
        self.STOP_WORDS = {"de", "da", "do", "dos", "das", "e", "a", "o", "as", "os"}
        
        possible_paths = [
            "data/municipios_pt.csv",  # From main project directory
            "../data/municipios_pt.csv",  # From EDA or src subdirectory
            "../../data/municipios_pt.csv",  # From deeper subdirectory
            Path(__file__).parent.parent / "data" / "municipios_pt.csv"  # Relative to this file
        ]
        
        data_file_path = None
        for path in possible_paths:
            if Path(path).exists():
                data_file_path = str(path)
                break
                
        if data_file_path is None:
            print("WARNING: Could not find municipios_pt.csv in any expected location")
            data_file_path = "data/municipios_pt.csv"  # Fallback to original
            
        self.CANONICAL_CITIES = self._load_canonical_cities(data_file_path)
    
        
    def _load_canonical_cities(self, filepath: str) -> set:
        """
        Loads a list of canonical municipality names from a CSV file.
        """
        try:
            encodings_to_try = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
            
            df = None
            for encoding in encodings_to_try:
                try:
                    df = pd.read_csv(filepath, skiprows=2, encoding=encoding)
                    print(f"✅ Successfully loaded file with {encoding} encoding")
                    break
                except UnicodeDecodeError:
                    continue
            
            if df is None:
                print(f"ERROR: Could not load file '{filepath}' with any supported encoding")
                return set()
            
            column_name = df.columns[1] if len(df.columns) > 1 else df.columns[0]
            
            if column_name not in df.columns:
                print(f"WARNING: Column '{column_name}' was not found in {filepath}")
                print(f"Available columns: {list(df.columns)}")
                return set()

            cities = {self._general_preprocessing(name) for name in df[column_name].dropna() if isinstance(name, str)}
            
            print(f"✅ {len(cities)} canonical municipality names successfully loaded from '{filepath}'.")
            return cities

        except FileNotFoundError:
            print(f"WARNING: Municipality file '{filepath}' was not found.")
            return set()
        except Exception as e:
            print(f"ERROR loading municipalities from '{filepath}': {e}")
            return set()

    def _general_preprocessing(self, text: str) -> str:
        """
        Step 1: Preprocessing and General Simplification
        Applies: Lowercasing, Unicode Normalization, Punctuation Removal, Whitespace Normalization, Common Encoding Fixes

        """
        if not text or not isinstance(text, str):
            return ""
            
        self.improvement_stats['preprocessing_applied'] += 1
        original_text = text
        
        # Lowercasing
        text = text.lower()
        if original_text != text:
            self.improvement_stats['case_fixes'] += 1

        # Unicode Normalization to remove accents (NFD)
        try:
            nfkd_form = unicodedata.normalize('NFD', text)
            text = "".join([c for c in nfkd_form if not unicodedata.combining(c)])
        except TypeError:
            pass 
        
        # Punctuation Removal (except hyphens and periods)
        punctuation_to_remove = string.punctuation.replace('-', '').replace('.', '')
        translator = str.maketrans('', '', punctuation_to_remove)
        cleaned = text.translate(translator)
        
        if len(cleaned) != len(text):
            self.improvement_stats['punctuation_removed'] += 1
        text = cleaned

        # Whitespace Normalization
        original_spacing = text
        text = ' '.join(text.split())
        text = text.strip()
        
        if original_spacing != text:
            self.improvement_stats['whitespace_normalized'] += 1

        # Common Encoding Fixes
        encoding_fixes = {
            'a': ['a~', 'a`', 'a´', 'a\'', 'a^'],
            'e': ['e´', 'e\'', 'e^'],
            'i': ['i´', 'i\''],
            'o': ['o´', 'o\'', 'o^', 'o~'],
            'u': ['u´', 'u\''],
            'c': ['c,', 'c`']
        }
        
        for correct_char, wrong_variants in encoding_fixes.items():
            for wrong_char in wrong_variants:
                if wrong_char in text:
                    text = text.replace(wrong_char, correct_char)
                    self.improvement_stats['encoding_fixes'] += 1
        
        return text
    
    def normalize_street(self, street_raw: str) -> str:
        """
        Normalizes street names based on patterns identified in EDA.
        
        Pipeline:
        1. General preprocessing (lowercase, accent removal, etc.)
        2. Street string tokenization.
        3. Prefix standardization (e.g., "av" -> "avenida").
        4. Stop word removal (e.g., "de", "da", "do").
        5. Final string recomposition.
        """
        if not street_raw:
            return ""
            
        street_clean = self._general_preprocessing(street_raw)
        
        tokens = street_clean.split()
        
        if not tokens:
            return ""
            
        processed_tokens = []
        
        first_token = tokens[0]
        # Remove trailing period from first token for prefix matching
        first_token_clean = first_token.rstrip('.')
        
        if first_token_clean in self.PREFIX_MAP_EXPANDED:
            processed_tokens.append(self.PREFIX_MAP_EXPANDED[first_token_clean])
            tokens = tokens[1:]
        else:
            processed_tokens.append(first_token)
            tokens = tokens[1:]

        for token in tokens:
            if token not in self.STOP_WORDS:
                processed_tokens.append(token)
        
        return ' '.join(processed_tokens)
    
    def normalize_postcode(self, postcode_raw: str) -> Optional[str]:
        """
        Normalizes postal codes to the canonical XXXX-XXX format.

        Technique:
        - Removes all non-numeric characters.
        - Validates if the result contains exactly 7 digits.
        - Formats the string to the correct format with hyphen.
        - Returns None if the postal code is invalid.

        Args:
            postcode_raw (str): Raw postal code.

        Returns:
            Optional[str]: Normalized postal code or None if invalid.
        """
        if not postcode_raw or not isinstance(postcode_raw, str):
            return None

        # Extract digits only
        digits = re.sub(r'\D', '', postcode_raw)
        
        # Validate length
        if len(digits) == 7:
            # Format to XXXX-XXX
            formatted_postcode = f"{digits[:4]}-{digits[4:]}"
            return formatted_postcode
            
        # If not valid, return None
        return None
    
    def normalize_housenumber(self, housenumber_raw: str) -> Optional[Dict[str, Union[int, str]]]:
        """
        Normalizes and structures house/door numbers.

        Technique:
        - Extracts the first number BEFORE preprocessing to avoid concatenation of ranges
        - Handles ranges like "120/122", "45;47", "22,24,26" by taking the first number
        - Uses general preprocessing only for the specifier part
        - Returns a structured dictionary with the original, primary number and specifier.

        Args:
            housenumber_raw (str): Raw house number.

        Returns:
            Optional[Dict[str, Union[int, str]]]: Structured dictionary or None if invalid.
        """
        if not housenumber_raw or not isinstance(housenumber_raw, str):
            return None

        original = housenumber_raw
        housenumber_work = housenumber_raw.strip()
        
        # Extract first number BEFORE preprocessing to avoid concatenation
        # This handles ranges like "120/122" correctly by taking "120"
        # Try multiple patterns to be more robust
        first_number_match = re.search(r'^\D*(\d+)', housenumber_work)
        if not first_number_match:
            # Fallback: try to find any number in the string
            first_number_match = re.search(r'(\d+)', housenumber_work)
        
        if not first_number_match:
            # Last resort: apply old method with preprocessing
            housenumber_clean = self._general_preprocessing(housenumber_raw)
            if housenumber_clean:
                match = re.search(r'\d+', housenumber_clean)
                if match:
                    return {
                        "original": original,
                        "numero_primario": int(match.group(0)),
                        "especificador": housenumber_clean.replace(match.group(0), '', 1).strip() or None
                    }
            return None
            
        numero_primario = int(first_number_match.group(1))
        
        # For specifier, apply preprocessing to the remainder
        # Remove the matched number from the original and clean up
        remainder = housenumber_work[first_number_match.end():]
        if remainder:
            especificador = self._general_preprocessing(remainder).strip()
            especificador = especificador if especificador else None
        else:
            especificador = None

        return {
            "original": original,
            "numero_primario": numero_primario,
            "especificador": especificador
        }
    
    def normalize_city(self, city_raw: str) -> Optional[str]:
        """
        Normaliza nomes de cidades portuguesas usando uma lista canónica
        e fuzzy matching para corrigir erros ortográficos.
        """
        if not city_raw or not isinstance(city_raw, str):
            return None

        city_clean = self._general_preprocessing(city_raw)

        if city_clean in self.CANONICAL_CITIES:
            return city_clean

        if self.CANONICAL_CITIES:
            best_match, score = process.extractOne(city_clean, self.CANONICAL_CITIES)
            
            if score >= 85:
                return best_match
            
        return None
    
    def normalize_address_record(self, raw_record: Dict[str, str]) -> Dict[str, Union[str, Dict]]:
        """
        Função principal que recebe um registo de morada (dicionário de tags)
        e devolve um dicionário com os campos limpos e estruturados.
        
        Esta é a função de entrada principal do pipeline de normalização.
        Aplica todas as técnicas de normalização desenvolvidas baseadas na EDA.
        
        Args:
            raw_record (Dict[str, str]): Dicionário com tags OSM brutas
                Formato esperado: {'addr:street': '...', 'addr:postcode': '...', etc.}
        
        Returns:
            Dict[str, Union[str, Dict]]: Registo normalizado com:
                - Campos limpos ('street_clean', 'postcode_clean', etc.)
                - Tags originais preservadas ('raw_tags')
                - Métricas de qualidade ('quality_metrics')
                
        Baseado nos descobrimentos da EDA:
        - Melhoria da taxa de completude de 30% para >70%
        - Padronização dos 42 tipos de prefixos
        - Correção dos problemas de capitalização e encoding
        """
        if not raw_record or not isinstance(raw_record, dict):
            return {
                'raw_tags': raw_record,
                'quality_metrics': {'error': 'Invalid input record'}
            }
        
        clean_record = {}
        quality_metrics = {
            'original_completeness': 0,
            'improved_completeness': 0,
            'fields_processed': 0,
            'preprocessing_applied': False
        }
        
        # Campos essenciais identificados na EDA
        essential_fields = ['addr:street', 'addr:housenumber', 'addr:city', 'addr:postcode']
        
        # Calcular completude original
        original_fields = sum(1 for field in essential_fields if field in raw_record and raw_record[field])
        quality_metrics['original_completeness'] = original_fields / len(essential_fields)
        
        # Aplicar normalização a cada campo
        if 'addr:street' in raw_record:
            clean_record['street_clean'] = self.normalize_street(raw_record['addr:street'])
            quality_metrics['fields_processed'] += 1
            
        if 'addr:postcode' in raw_record:
            clean_record['postcode_clean'] = self.normalize_postcode(raw_record['addr:postcode'])
            quality_metrics['fields_processed'] += 1
            
        if 'addr:housenumber' in raw_record:
            clean_record['housenumber_clean'] = self.normalize_housenumber(raw_record['addr:housenumber'])
            quality_metrics['fields_processed'] += 1
            
        if 'addr:city' in raw_record:
            clean_record['city_clean'] = self.normalize_city(raw_record['addr:city'])
            quality_metrics['fields_processed'] += 1
        
        # Calcular completude melhorada
        improved_fields = sum(1 for field_clean in ['street_clean', 'housenumber_clean', 
                                                   'city_clean', 'postcode_clean'] 
                             if field_clean in clean_record and clean_record[field_clean])
        quality_metrics['improved_completeness'] = improved_fields / len(essential_fields)
        quality_metrics['preprocessing_applied'] = quality_metrics['fields_processed'] > 0
        
        # Preservar tags originais para análise
        clean_record['raw_tags'] = raw_record
        clean_record['quality_metrics'] = quality_metrics
        
        return clean_record
    
    def get_improvement_stats(self) -> Dict[str, Union[int, float]]:
        """
        Retorna estatísticas de melhorias aplicadas pelo normalizador.
        
        Returns:
            Dict: Estatísticas de melhorias e comparações com a EDA
        """
        stats = self.improvement_stats.copy()
        
        # Adicionar métricas de comparação com a EDA
        stats['eda_baseline'] = self.eda_stats
        
        # Calcular taxas de melhoria se houver processamentos
        if stats['preprocessing_applied'] > 0:
            stats['case_fix_rate'] = stats['case_fixes'] / stats['preprocessing_applied']
            stats['punctuation_removal_rate'] = stats['punctuation_removed'] / stats['preprocessing_applied']
            stats['encoding_fix_rate'] = stats['encoding_fixes'] / stats['preprocessing_applied']
        
        return stats
    
    def reset_stats(self):
        """Reinicia os contadores de estatísticas."""
        for key in self.improvement_stats:
            self.improvement_stats[key] = 0


# Este bloco só é executado quando o ficheiro é corrido diretamente
if __name__ == '__main__':
    print("--- A testar o carregamento de cidades canónicas ---")
    
    # Criar uma instância para testar o __init__
    test_normalizer = AddressNormalizer()
    
    # Imprimir o número de cidades
    print(f"Total de cidades carregadas: {len(test_normalizer.CANONICAL_CITIES)}")
    
    # Imprimir uma amostra
    print(f"Amostra: {sorted(list(test_normalizer.CANONICAL_CITIES))}")
    
    # Testar a presença de uma cidade chave
    if "angra do heroismo" in test_normalizer.CANONICAL_CITIES:
        print("Teste de presença ('angra do heroismo'): OK")
    else:
        print("Teste de presença ('angra do heroismo'): FALHOU")