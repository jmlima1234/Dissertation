"""
Token-Based Fuzzy Matching Algorithms

This module implements token-level similarity algorithms for address matching,
including Jaccard similarity and TF-IDF with cosine similarity.

Experiment 2.2: Token-Based Metrics
- Jaccard Similarity
- TF-IDF with Cosine Similarity
"""

import re
import unicodedata
from typing import List, Set, Optional, Tuple, Dict
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd


# Portuguese stopwords for address matching
PORTUGUESE_STOPWORDS = {
    'de', 'da', 'do', 'dos', 'das', 'na', 'no', 'nos', 'nas',
    'em', 'ao', 'aos', 'à', 'às', 'para', 'por', 'com', 'sem',
    'e', 'ou', 'a', 'o', 'os', 'as', 'um', 'uma', 'uns', 'umas'
}

# Portuguese address type abbreviations
ADDRESS_ABBREVIATIONS = {
    'r': 'rua', 'rua': 'rua',
    'av': 'avenida', 'avenida': 'avenida',
    'pç': 'praça', 'praça': 'praça', 'pc': 'praça',
    'lg': 'largo', 'largo': 'largo',
    'tv': 'travessa', 'travessa': 'travessa',
    'est': 'estrada', 'estrada': 'estrada',
    'al': 'alameda', 'alameda': 'alameda',
    'bco': 'beco', 'beco': 'beco',
    'cal': 'calçada', 'calçada': 'calçada',
    'rot': 'rotunda', 'rotunda': 'rotunda',
    'qta': 'quinta', 'quinta': 'quinta',
    'urb': 'urbanização', 'urbanização': 'urbanização'
}


def normalize_text(text: str) -> str:
    """
    Normalize text for consistent tokenization.
    
    Args:
        text: Input text
        
    Returns:
        Normalized text
    """
    if not text:
        return ""
    
    # Convert to lowercase
    normalized = text.lower()
    
    # Remove accents and diacritics
    normalized = ''.join(c for c in unicodedata.normalize('NFD', normalized)
                        if unicodedata.category(c) != 'Mn')
    
    # Replace punctuation with spaces
    normalized = re.sub(r'[^\w\s]', ' ', normalized)
    
    # Replace multiple spaces with single space
    normalized = ' '.join(normalized.split())
    
    return normalized.strip()


def expand_abbreviations(text: str) -> str:
    """
    Expand Portuguese address abbreviations.
    
    Args:
        text: Input text with potential abbreviations
        
    Returns:
        Text with expanded abbreviations
    """
    words = text.split()
    expanded_words = []
    
    for word in words:
        # Check if word is an abbreviation
        clean_word = word.rstrip('.')
        if clean_word in ADDRESS_ABBREVIATIONS:
            expanded_words.append(ADDRESS_ABBREVIATIONS[clean_word])
        else:
            expanded_words.append(word)
    
    return ' '.join(expanded_words)


def tokenize_address(address: str, remove_stopwords: bool = True, 
                    expand_abbrev: bool = True, min_token_length: int = 2) -> List[str]:
    """
    Tokenize an address into meaningful tokens.
    
    Args:
        address: Address string to tokenize
        remove_stopwords: Whether to remove Portuguese stopwords
        expand_abbrev: Whether to expand abbreviations
        min_token_length: Minimum length for tokens
        
    Returns:
        List of tokens
    """
    if not address:
        return []
    
    # Normalize text
    normalized = normalize_text(address)
    
    # Expand abbreviations if requested
    if expand_abbrev:
        normalized = expand_abbreviations(normalized)
    
    # Split into tokens
    tokens = normalized.split()
    
    # Filter tokens
    filtered_tokens = []
    for token in tokens:
        # Skip short tokens
        if len(token) < min_token_length:
            continue
        
        # Skip stopwords if requested
        if remove_stopwords and token in PORTUGUESE_STOPWORDS:
            continue
        
        # Skip pure numbers (optional - might want to keep house numbers)
        # if token.isdigit():
        #     continue
        
        filtered_tokens.append(token)
    
    return filtered_tokens


def jaccard_similarity(set1: Set[str], set2: Set[str]) -> float:
    """
    Calculate Jaccard similarity between two sets.
    
    The Jaccard similarity is the size of the intersection divided by 
    the size of the union of the two sets.
    
    Args:
        set1: First set of tokens
        set2: Second set of tokens
        
    Returns:
        Jaccard similarity score between 0 and 1
    """
    if not set1 and not set2:
        return 1.0
    
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    
    if union == 0:
        return 0.0
    
    return intersection / union


def jaccard_address_similarity(addr1: str, addr2: str, **tokenize_kwargs) -> float:
    """
    Calculate Jaccard similarity between two addresses.
    
    Args:
        addr1: First address
        addr2: Second address
        **tokenize_kwargs: Arguments passed to tokenize_address
        
    Returns:
        Jaccard similarity score between 0 and 1
    """
    tokens1 = set(tokenize_address(addr1, **tokenize_kwargs))
    tokens2 = set(tokenize_address(addr2, **tokenize_kwargs))
    
    return jaccard_similarity(tokens1, tokens2)


class TfIdfMatcher:
    """
    TF-IDF based address matcher using cosine similarity.
    """
    
    def __init__(self, max_features: int = 1000, ngram_range: Tuple[int, int] = (1, 2),
                 min_df: int = 1, max_df: float = 0.95):
        """
        Initialize TF-IDF matcher.
        
        Args:
            max_features: Maximum number of features
            ngram_range: Range of n-grams to use
            min_df: Minimum document frequency
            max_df: Maximum document frequency
        """
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df,
            token_pattern=r'\b\w+\b',
            lowercase=True,
            stop_words=None  # We'll handle stopwords in preprocessing
        )
        self.is_fitted = False
        self.vocabulary_ = None
    
    def preprocess_address(self, address: str) -> str:
        """
        Preprocess address for TF-IDF vectorization.
        
        Args:
            address: Input address
            
        Returns:
            Preprocessed address
        """
        # Tokenize and rejoin
        tokens = tokenize_address(address, remove_stopwords=True, expand_abbrev=True)
        return ' '.join(tokens)
    
    def fit(self, addresses: List[str]) -> 'TfIdfMatcher':
        """
        Fit the TF-IDF vectorizer on a corpus of addresses.
        
        Args:
            addresses: List of addresses for training
            
        Returns:
            Self for method chaining
        """
        # Preprocess addresses
        preprocessed = [self.preprocess_address(addr) for addr in addresses]
        
        # Filter out empty addresses
        preprocessed = [addr for addr in preprocessed if addr.strip()]
        
        if not preprocessed:
            raise ValueError("No valid addresses to fit on")
        
        # Fit vectorizer
        self.vectorizer.fit(preprocessed)
        self.is_fitted = True
        self.vocabulary_ = self.vectorizer.vocabulary_
        
        return self
    
    def similarity(self, addr1: str, addr2: str) -> float:
        """
        Calculate TF-IDF cosine similarity between two addresses.
        
        Args:
            addr1: First address
            addr2: Second address
            
        Returns:
            Cosine similarity score between 0 and 1
        """
        if not self.is_fitted:
            # Fit on the two addresses if not fitted
            self.fit([addr1, addr2])
        
        # Preprocess addresses
        prep1 = self.preprocess_address(addr1)
        prep2 = self.preprocess_address(addr2)
        
        # Handle empty addresses
        if not prep1.strip() and not prep2.strip():
            return 1.0
        if not prep1.strip() or not prep2.strip():
            return 0.0
        
        try:
            # Vectorize
            vectors = self.vectorizer.transform([prep1, prep2])
            
            # Calculate cosine similarity
            sim_matrix = cosine_similarity(vectors)
            similarity = sim_matrix[0, 1]
            
            # Ensure result is between 0 and 1
            return max(0.0, min(1.0, similarity))
            
        except ValueError:
            # Handle case where addresses have no valid tokens
            return 0.0
    
    def batch_similarity(self, addresses1: List[str], addresses2: List[str]) -> np.ndarray:
        """
        Calculate similarities for batch of address pairs.
        
        Args:
            addresses1: First set of addresses
            addresses2: Second set of addresses
            
        Returns:
            Array of similarity scores
        """
        if len(addresses1) != len(addresses2):
            raise ValueError("Address lists must have the same length")
        
        # Fit on all addresses if not fitted
        if not self.is_fitted:
            all_addresses = addresses1 + addresses2
            self.fit(all_addresses)
        
        similarities = []
        for addr1, addr2 in zip(addresses1, addresses2):
            sim = self.similarity(addr1, addr2)
            similarities.append(sim)
        
        return np.array(similarities)


class TokenBasedMatcher:
    """
    A class that implements various token-based matching algorithms.
    """
    
    def __init__(self, remove_stopwords: bool = True, expand_abbreviations: bool = True,
                 min_token_length: int = 2):
        """
        Initialize the token-based matcher.
        
        Args:
            remove_stopwords: Whether to remove Portuguese stopwords
            expand_abbreviations: Whether to expand address abbreviations
            min_token_length: Minimum length for tokens
        """
        self.remove_stopwords = remove_stopwords
        self.expand_abbreviations = expand_abbreviations
        self.min_token_length = min_token_length
        self.tfidf_matcher = None
    
    def get_tokenize_kwargs(self) -> Dict:
        """Get tokenization parameters as dictionary."""
        return {
            'remove_stopwords': self.remove_stopwords,
            'expand_abbrev': self.expand_abbreviations,
            'min_token_length': self.min_token_length
        }
    
    def jaccard_match(self, addr1: str, addr2: str) -> float:
        """
        Match addresses using Jaccard similarity.
        
        Args:
            addr1: First address
            addr2: Second address
            
        Returns:
            Jaccard similarity score between 0 and 1
        """
        return jaccard_address_similarity(addr1, addr2, **self.get_tokenize_kwargs())
    
    def tfidf_match(self, addr1: str, addr2: str, corpus: Optional[List[str]] = None) -> float:
        """
        Match addresses using TF-IDF cosine similarity.
        
        Args:
            addr1: First address
            addr2: Second address
            corpus: Optional corpus for TF-IDF fitting
            
        Returns:
            Cosine similarity score between 0 and 1
        """
        if self.tfidf_matcher is None:
            self.tfidf_matcher = TfIdfMatcher()
        
        if corpus and not self.tfidf_matcher.is_fitted:
            self.tfidf_matcher.fit(corpus)
        
        return self.tfidf_matcher.similarity(addr1, addr2)
    
    def combined_token_similarity(self, addr1: str, addr2: str,
                                jaccard_weight: float = 0.5,
                                tfidf_weight: float = 0.5,
                                corpus: Optional[List[str]] = None) -> float:
        """
        Calculate a weighted combination of token-based similarities.
        
        Args:
            addr1: First address
            addr2: Second address
            jaccard_weight: Weight for Jaccard similarity
            tfidf_weight: Weight for TF-IDF similarity
            corpus: Optional corpus for TF-IDF fitting
            
        Returns:
            Combined similarity score between 0 and 1
        """
        if abs(jaccard_weight + tfidf_weight - 1.0) > 1e-6:
            raise ValueError("Weights must sum to 1.0")
        
        jaccard_sim = self.jaccard_match(addr1, addr2)
        tfidf_sim = self.tfidf_match(addr1, addr2, corpus)
        
        combined = (jaccard_sim * jaccard_weight + tfidf_sim * tfidf_weight)
        
        return combined
    
    def analyze_tokens(self, address: str) -> Dict:
        """
        Analyze tokenization of an address for debugging.
        
        Args:
            address: Address to analyze
            
        Returns:
            Dictionary with tokenization details
        """
        original = address
        normalized = normalize_text(address)
        expanded = expand_abbreviations(normalized) if self.expand_abbreviations else normalized
        tokens = tokenize_address(address, **self.get_tokenize_kwargs())
        
        return {
            'original': original,
            'normalized': normalized,
            'expanded': expanded,
            'tokens': tokens,
            'token_count': len(tokens)
        }


# Convenience functions for direct use
def quick_jaccard_similarity(addr1: str, addr2: str) -> float:
    """Quick Jaccard similarity with default parameters."""
    return jaccard_address_similarity(addr1, addr2)


def quick_tfidf_similarity(addr1: str, addr2: str, corpus: Optional[List[str]] = None) -> float:
    """Quick TF-IDF similarity with default parameters."""
    matcher = TfIdfMatcher()
    if corpus:
        matcher.fit(corpus)
    return matcher.similarity(addr1, addr2)


# For compatibility with evaluation framework
def create_jaccard_function():
    """Create a Jaccard matching function compatible with evaluation framework."""
    return quick_jaccard_similarity


def create_tfidf_function(corpus: Optional[List[str]] = None):
    """Create a TF-IDF matching function compatible with evaluation framework."""
    if corpus:
        # Pre-fitted TF-IDF matcher
        matcher = TfIdfMatcher()
        matcher.fit(corpus)
        return lambda addr1, addr2: matcher.similarity(addr1, addr2)
    else:
        # Dynamic TF-IDF matcher
        return quick_tfidf_similarity
