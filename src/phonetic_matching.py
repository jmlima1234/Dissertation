"""
Phonetic matching algorithms for Portuguese address comparison.

This module implements phonetic algorithms adapted for Portuguese addresses:
- Soundex (original and Portuguese-adapted)
- Metaphone (with Portuguese considerations)
- NYSIIS (New York State Identification and Intelligence System)
- Double Metaphone

While phonetic algorithms are primarily designed for names and may be less
effective for structured addresses, they can help with:
- Handling pronunciation-based misspellings
- Matching phonetically similar street names
- Dealing with transcription errors from audio sources
"""

import re
import unicodedata
from typing import List, Tuple, Optional
import jellyfish  # For Soundex, Metaphone, NYSIIS implementations
from functools import lru_cache


class PhoneticMatcher:
    """
    Phonetic matching for Portuguese addresses using various algorithms.
    """
    
    def __init__(self, 
                 normalize_input: bool = True,
                 algorithm: str = 'soundex',
                 portuguese_adaptations: bool = True):
        """
        Initialize the phonetic matcher.
        
        Args:
            normalize_input: Whether to normalize text before phonetic encoding
            algorithm: Phonetic algorithm to use ('soundex', 'metaphone', 'nysiis', 'dmetaphone')
            portuguese_adaptations: Whether to apply Portuguese-specific adaptations
        """
        self.normalize_input = normalize_input
        self.algorithm = algorithm.lower()
        self.portuguese_adaptations = portuguese_adaptations
        
        # Portuguese-specific phonetic mappings
        self.portuguese_replacements = {
            'ph': 'f',
            'th': 't',
            'ch': 'x',
            'lh': 'ly',
            'nh': 'ny',
            'rr': 'r',
            'ss': 's',
            'ç': 'c',
            'qu': 'k',
            'gu': 'g',
        }
        
        # Algorithm function mapping
        self.algorithms = {
            'soundex': self._soundex_similarity,
            'metaphone': self._metaphone_similarity,
            'nysiis': self._nysiis_similarity,
            'dmetaphone': self._dmetaphone_similarity
        }
        
        if self.algorithm not in self.algorithms:
            raise ValueError(f"Algorithm '{algorithm}' not supported. "
                           f"Available: {list(self.algorithms.keys())}")
    
    def normalize_portuguese_text(self, text: str) -> str:
        """
        Normalize Portuguese text for phonetic processing.
        
        Args:
            text: Input text to normalize
            
        Returns:
            Normalized text
        """
        if not text:
            return ""
        
        # Convert to lowercase
        text = text.lower().strip()
        
        # Remove accents and diacritics
        text = unicodedata.normalize('NFD', text)
        text = ''.join(c for c in text if unicodedata.category(c) != 'Mn')
        
        # Apply Portuguese-specific replacements
        if self.portuguese_adaptations:
            for old, new in self.portuguese_replacements.items():
                text = text.replace(old, new)
        
        # Remove non-alphabetic characters except spaces
        text = re.sub(r'[^a-z\s]', '', text)
        
        return text
    
    def extract_phonetic_components(self, address: str) -> List[str]:
        """
        Extract components from address that are suitable for phonetic matching.
        
        Args:
            address: Full address string
            
        Returns:
            List of address components suitable for phonetic comparison
        """
        if not address:
            return []
        
        # Normalize if requested
        if self.normalize_input:
            address = self.normalize_portuguese_text(address)
        
        # Split into words
        words = address.split()
        
        # Filter out common address elements that don't benefit from phonetic matching
        address_stopwords = {
            'rua', 'r', 'avenida', 'av', 'praca', 'pc', 'largo', 'lg',
            'travessa', 'tv', 'beco', 'estrada', 'est', 'alameda', 'al',
            'numero', 'num', 'n', 'andar', 'dto', 'esq', 'frente', 'tras',
            'de', 'da', 'do', 'das', 'dos', 'e', 'em', 'na', 'no', 'com',
            'lisboa', 'porto', 'coimbra', 'braga', 'aveiro', 'faro',
            'setubal', 'leiria', 'santarem', 'evora', 'beja', 'portalegre',
            'vila', 'real', 'viana', 'castelo', 'guarda', 'viseu'
        }
        
        # Keep words that are likely street names or meaningful address components
        phonetic_components = []
        for word in words:
            # Skip very short words (likely abbreviations or numbers)
            if len(word) < 3:
                continue
            
            # Skip common address stopwords
            if word in address_stopwords:
                continue
            
            # Skip if it's all digits
            if word.isdigit():
                continue
            
            # Skip postal codes (pattern: 4 digits - 3 digits)
            if re.match(r'\d{4}-?\d{3}', word):
                continue
            
            phonetic_components.append(word)
        
        return phonetic_components
    
    @lru_cache(maxsize=1000)
    def get_phonetic_code(self, word: str, algorithm: str = None) -> str:
        """
        Get phonetic code for a word using specified algorithm.
        
        Args:
            word: Word to encode
            algorithm: Algorithm to use (defaults to instance algorithm)
            
        Returns:
            Phonetic code string
        """
        if not word:
            return ""
        
        alg = algorithm or self.algorithm
        
        try:
            if alg == 'soundex':
                return jellyfish.soundex(word) or ""
            elif alg == 'metaphone':
                return jellyfish.metaphone(word) or ""
            elif alg == 'nysiis':
                return jellyfish.nysiis(word) or ""
            elif alg == 'dmetaphone':
                # Double Metaphone returns tuple, use first value
                result = jellyfish.dmetaphone(word)
                return result[0] if result and result[0] else ""
            else:
                return ""
        except Exception:
            return ""
    
    def _soundex_similarity(self, addr1: str, addr2: str) -> float:
        """Calculate similarity using Soundex algorithm."""
        components1 = self.extract_phonetic_components(addr1)
        components2 = self.extract_phonetic_components(addr2)
        
        if not components1 or not components2:
            return 0.0
        
        # Get Soundex codes for all components
        codes1 = [self.get_phonetic_code(comp, 'soundex') for comp in components1]
        codes2 = [self.get_phonetic_code(comp, 'soundex') for comp in components2]
        
        # Remove empty codes
        codes1 = [code for code in codes1 if code]
        codes2 = [code for code in codes2 if code]
        
        if not codes1 or not codes2:
            return 0.0
        
        # Calculate Jaccard similarity of phonetic codes
        set1 = set(codes1)
        set2 = set(codes2)
        
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        
        return intersection / union if union > 0 else 0.0
    
    def _metaphone_similarity(self, addr1: str, addr2: str) -> float:
        """Calculate similarity using Metaphone algorithm."""
        components1 = self.extract_phonetic_components(addr1)
        components2 = self.extract_phonetic_components(addr2)
        
        if not components1 or not components2:
            return 0.0
        
        codes1 = [self.get_phonetic_code(comp, 'metaphone') for comp in components1]
        codes2 = [self.get_phonetic_code(comp, 'metaphone') for comp in components2]
        
        codes1 = [code for code in codes1 if code]
        codes2 = [code for code in codes2 if code]
        
        if not codes1 or not codes2:
            return 0.0
        
        set1 = set(codes1)
        set2 = set(codes2)
        
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        
        return intersection / union if union > 0 else 0.0
    
    def _nysiis_similarity(self, addr1: str, addr2: str) -> float:
        """Calculate similarity using NYSIIS algorithm."""
        components1 = self.extract_phonetic_components(addr1)
        components2 = self.extract_phonetic_components(addr2)
        
        if not components1 or not components2:
            return 0.0
        
        codes1 = [self.get_phonetic_code(comp, 'nysiis') for comp in components1]
        codes2 = [self.get_phonetic_code(comp, 'nysiis') for comp in components2]
        
        codes1 = [code for code in codes1 if code]
        codes2 = [code for code in codes2 if code]
        
        if not codes1 or not codes2:
            return 0.0
        
        set1 = set(codes1)
        set2 = set(codes2)
        
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        
        return intersection / union if union > 0 else 0.0
    
    def _dmetaphone_similarity(self, addr1: str, addr2: str) -> float:
        """Calculate similarity using Double Metaphone algorithm."""
        components1 = self.extract_phonetic_components(addr1)
        components2 = self.extract_phonetic_components(addr2)
        
        if not components1 or not components2:
            return 0.0
        
        codes1 = [self.get_phonetic_code(comp, 'dmetaphone') for comp in components1]
        codes2 = [self.get_phonetic_code(comp, 'dmetaphone') for comp in components2]
        
        codes1 = [code for code in codes1 if code]
        codes2 = [code for code in codes2 if code]
        
        if not codes1 or not codes2:
            return 0.0
        
        set1 = set(codes1)
        set2 = set(codes2)
        
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        
        return intersection / union if union > 0 else 0.0
    
    def similarity(self, addr1: str, addr2: str) -> float:
        """
        Calculate phonetic similarity between two addresses.
        
        Args:
            addr1: First address
            addr2: Second address
            
        Returns:
            Similarity score between 0 and 1
        """
        return self.algorithms[self.algorithm](addr1, addr2)
    
    def analyze_phonetic_codes(self, address: str) -> dict:
        """
        Analyze phonetic codes for debugging and understanding.
        
        Args:
            address: Address to analyze
            
        Returns:
            Dictionary with analysis results
        """
        components = self.extract_phonetic_components(address)
        
        analysis = {
            'original': address,
            'normalized': self.normalize_portuguese_text(address) if self.normalize_input else address,
            'components': components,
            'phonetic_codes': {}
        }
        
        for alg in ['soundex', 'metaphone', 'nysiis', 'dmetaphone']:
            analysis['phonetic_codes'][alg] = [
                self.get_phonetic_code(comp, alg) for comp in components
            ]
        
        return analysis


# Factory functions for evaluation framework compatibility
def create_soundex_function() -> callable:
    """Create a Soundex similarity function for evaluation."""
    matcher = PhoneticMatcher(algorithm='soundex')
    return matcher.similarity


def create_metaphone_function() -> callable:
    """Create a Metaphone similarity function for evaluation."""
    matcher = PhoneticMatcher(algorithm='metaphone')
    return matcher.similarity


def create_nysiis_function() -> callable:
    """Create a NYSIIS similarity function for evaluation."""
    matcher = PhoneticMatcher(algorithm='nysiis')
    return matcher.similarity


def create_dmetaphone_function() -> callable:
    """Create a Double Metaphone similarity function for evaluation."""
    matcher = PhoneticMatcher(algorithm='dmetaphone')
    return matcher.similarity


# Quick similarity functions for convenience
def quick_soundex_similarity(addr1: str, addr2: str) -> float:
    """Quick Soundex similarity calculation."""
    matcher = PhoneticMatcher(algorithm='soundex')
    return matcher.similarity(addr1, addr2)


def quick_metaphone_similarity(addr1: str, addr2: str) -> float:
    """Quick Metaphone similarity calculation."""
    matcher = PhoneticMatcher(algorithm='metaphone')
    return matcher.similarity(addr1, addr2)


def quick_nysiis_similarity(addr1: str, addr2: str) -> float:
    """Quick NYSIIS similarity calculation."""
    matcher = PhoneticMatcher(algorithm='nysiis')
    return matcher.similarity(addr1, addr2)


def quick_dmetaphone_similarity(addr1: str, addr2: str) -> float:
    """Quick Double Metaphone similarity calculation."""
    matcher = PhoneticMatcher(algorithm='dmetaphone')
    return matcher.similarity(addr1, addr2)


if __name__ == "__main__":
    # Test the phonetic matching
    print("Testing Phonetic Matching for Portuguese Addresses")
    print("=" * 60)
    
    test_pairs = [
        ("Rua das Flores, 123, Lisboa", "Rua das Flôres, 123, Lisboa"),
        ("Avenida da República, 45", "Av. da Republica, 45"),
        ("Praça do Comércio", "Prassa do Comercio"),
        ("Rua Dr. António José de Almeida", "Rua Doutor Antonio Jose de Almeida"),
        ("Rua Silva", "Rua Cilva"),  # Phonetic similarity test
    ]
    
    matcher = PhoneticMatcher(algorithm='soundex')
    
    for addr1, addr2 in test_pairs:
        similarity = matcher.similarity(addr1, addr2)
        print(f"\nAddresses:")
        print(f"  1: {addr1}")
        print(f"  2: {addr2}")
        print(f"  Soundex Similarity: {similarity:.4f}")
        
        # Show analysis
        analysis1 = matcher.analyze_phonetic_codes(addr1)
        analysis2 = matcher.analyze_phonetic_codes(addr2)
        print(f"  Components 1: {analysis1['components']}")
        print(f"  Components 2: {analysis2['components']}")
        print(f"  Soundex codes 1: {analysis1['phonetic_codes']['soundex']}")
        print(f"  Soundex codes 2: {analysis2['phonetic_codes']['soundex']}")
