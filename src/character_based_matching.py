"""
Character-Based Fuzzy Matching Algorithms

This module implements character-level similarity algorithms for address matching,
including Levenshtein distance and Jaro-Winkler similarity.

Experiment 2.1: Character-Based Metrics
- Levenshtein Distance
- Jaro-Winkler Distance
"""

import unicodedata
from typing import Optional, Tuple
import numpy as np


def normalize_address(address: str) -> str:
    """
    Basic address normalization for consistent comparison.
    
    Args:
        address: Raw address string
        
    Returns:
        Normalized address string
    """
    if not address:
        return ""
    
    # Convert to lowercase
    normalized = address.lower()
    
    # Remove accents and diacritics
    normalized = ''.join(c for c in unicodedata.normalize('NFD', normalized)
                        if unicodedata.category(c) != 'Mn')
    
    # Replace multiple spaces with single space
    normalized = ' '.join(normalized.split())
    
    # Strip leading/trailing whitespace
    normalized = normalized.strip()
    
    return normalized


def levenshtein_distance(s1: str, s2: str) -> int:
    """
    Calculate the Levenshtein distance between two strings.
    
    The Levenshtein distance is the minimum number of single-character edits
    (insertions, deletions, or substitutions) required to change one word
    into the other.
    
    Args:
        s1: First string
        s2: Second string
        
    Returns:
        Levenshtein distance as integer
    """
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            # Cost of insertions, deletions and substitutions
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]


def levenshtein_similarity(s1: str, s2: str, normalize: bool = True) -> float:
    """
    Calculate normalized Levenshtein similarity (0 to 1 scale).
    
    Args:
        s1: First string
        s2: Second string
        normalize: Whether to normalize addresses before comparison
        
    Returns:
        Similarity score between 0 and 1
    """
    if normalize:
        s1 = normalize_address(s1)
        s2 = normalize_address(s2)
    
    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    
    max_len = max(len(s1), len(s2))
    if max_len == 0:
        return 1.0
    
    distance = levenshtein_distance(s1, s2)
    similarity = 1 - (distance / max_len)
    
    return max(0.0, similarity)


def jaro_similarity(s1: str, s2: str) -> float:
    """
    Calculate Jaro similarity between two strings.
    
    The Jaro similarity is a measure of similarity between two strings.
    It takes into account the number and order of common characters.
    
    Args:
        s1: First string
        s2: Second string
        
    Returns:
        Jaro similarity score between 0 and 1
    """
    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    
    len1, len2 = len(s1), len(s2)
    
    # Maximum allowed distance
    match_distance = max(len1, len2) // 2 - 1
    if match_distance < 0:
        match_distance = 0
    
    # Arrays to track matches
    s1_matches = [False] * len1
    s2_matches = [False] * len2
    
    matches = 0
    
    # Find matches
    for i in range(len1):
        start = max(0, i - match_distance)
        end = min(i + match_distance + 1, len2)
        
        for j in range(start, end):
            if s2_matches[j] or s1[i] != s2[j]:
                continue
            s1_matches[i] = s2_matches[j] = True
            matches += 1
            break
    
    if matches == 0:
        return 0.0
    
    # Count transpositions
    transpositions = 0
    k = 0
    for i in range(len1):
        if not s1_matches[i]:
            continue
        while not s2_matches[k]:
            k += 1
        if s1[i] != s2[k]:
            transpositions += 1
        k += 1
    
    # Calculate Jaro similarity
    jaro = (matches / len1 + matches / len2 + (matches - transpositions / 2) / matches) / 3
    
    return jaro


def jaro_winkler_similarity(s1: str, s2: str, prefix_scale: float = 0.1, normalize: bool = True) -> float:
    """
    Calculate Jaro-Winkler similarity between two strings.
    
    The Jaro-Winkler similarity is an extension of Jaro similarity that gives
    more weight to strings that have a common prefix.
    
    Args:
        s1: First string
        s2: Second string
        prefix_scale: Scaling factor for common prefix (default 0.1)
        normalize: Whether to normalize addresses before comparison
        
    Returns:
        Jaro-Winkler similarity score between 0 and 1
    """
    if normalize:
        s1 = normalize_address(s1)
        s2 = normalize_address(s2)
    
    # Calculate base Jaro similarity
    jaro = jaro_similarity(s1, s2)
    
    # If Jaro similarity is below threshold, return as-is
    if jaro < 0.7:
        return jaro
    
    # Calculate common prefix length (up to 4 characters)
    prefix_length = 0
    max_prefix = min(4, min(len(s1), len(s2)))
    
    for i in range(max_prefix):
        if s1[i] == s2[i]:
            prefix_length += 1
        else:
            break
    
    # Calculate Jaro-Winkler similarity
    jaro_winkler = jaro + (prefix_length * prefix_scale * (1 - jaro))
    
    return min(1.0, jaro_winkler)


class CharacterBasedMatcher:
    """
    A class that implements various character-based matching algorithms.
    """
    
    def __init__(self, normalize_input: bool = True):
        """
        Initialize the matcher.
        
        Args:
            normalize_input: Whether to normalize input addresses
        """
        self.normalize_input = normalize_input
    
    def levenshtein_match(self, addr1: str, addr2: str) -> float:
        """
        Match addresses using Levenshtein similarity.
        
        Args:
            addr1: First address
            addr2: Second address
            
        Returns:
            Similarity score between 0 and 1
        """
        return levenshtein_similarity(addr1, addr2, normalize=self.normalize_input)
    
    def jaro_winkler_match(self, addr1: str, addr2: str, prefix_scale: float = 0.1) -> float:
        """
        Match addresses using Jaro-Winkler similarity.
        
        Args:
            addr1: First address
            addr2: Second address
            prefix_scale: Scaling factor for common prefix
            
        Returns:
            Similarity score between 0 and 1
        """
        return jaro_winkler_similarity(addr1, addr2, prefix_scale=prefix_scale, 
                                     normalize=self.normalize_input)
    
    def jaro_match(self, addr1: str, addr2: str) -> float:
        """
        Match addresses using Jaro similarity.
        
        Args:
            addr1: First address
            addr2: Second address
            
        Returns:
            Similarity score between 0 and 1
        """
        if self.normalize_input:
            addr1 = normalize_address(addr1)
            addr2 = normalize_address(addr2)
        
        return jaro_similarity(addr1, addr2)
    
    def combined_character_similarity(self, addr1: str, addr2: str, 
                                    levenshtein_weight: float = 0.5,
                                    jaro_winkler_weight: float = 0.5) -> float:
        """
        Calculate a weighted combination of character-based similarities.
        
        Args:
            addr1: First address
            addr2: Second address
            levenshtein_weight: Weight for Levenshtein similarity
            jaro_winkler_weight: Weight for Jaro-Winkler similarity
            
        Returns:
            Combined similarity score between 0 and 1
        """
        if abs(levenshtein_weight + jaro_winkler_weight - 1.0) > 1e-6:
            raise ValueError("Weights must sum to 1.0")
        
        lev_sim = self.levenshtein_match(addr1, addr2)
        jw_sim = self.jaro_winkler_match(addr1, addr2)
        
        combined = (lev_sim * levenshtein_weight + jw_sim * jaro_winkler_weight)
        
        return combined
    
    def batch_similarity(self, addresses1: list, addresses2: list, 
                        algorithm: str = 'jaro_winkler') -> np.ndarray:
        """
        Calculate similarities for batch of address pairs.
        
        Args:
            addresses1: First set of addresses
            addresses2: Second set of addresses (same length as addresses1)
            algorithm: Algorithm to use ('levenshtein', 'jaro_winkler', 'jaro')
            
        Returns:
            Array of similarity scores
        """
        if len(addresses1) != len(addresses2):
            raise ValueError("Address lists must have the same length")
        
        similarities = []
        
        for addr1, addr2 in zip(addresses1, addresses2):
            if algorithm == 'levenshtein':
                sim = self.levenshtein_match(addr1, addr2)
            elif algorithm == 'jaro_winkler':
                sim = self.jaro_winkler_match(addr1, addr2)
            elif algorithm == 'jaro':
                sim = self.jaro_match(addr1, addr2)
            else:
                raise ValueError(f"Unknown algorithm: {algorithm}")
            
            similarities.append(sim)
        
        return np.array(similarities)


# Convenience functions for direct use
def quick_levenshtein_similarity(addr1: str, addr2: str) -> float:
    """Quick Levenshtein similarity with normalization."""
    return levenshtein_similarity(addr1, addr2, normalize=True)


def quick_jaro_winkler_similarity(addr1: str, addr2: str) -> float:
    """Quick Jaro-Winkler similarity with normalization."""
    return jaro_winkler_similarity(addr1, addr2, normalize=True)


# For compatibility with evaluation framework
def create_levenshtein_function():
    """Create a Levenshtein matching function compatible with evaluation framework."""
    return quick_levenshtein_similarity


def create_jaro_winkler_function():
    """Create a Jaro-Winkler matching function compatible with evaluation framework."""
    return quick_jaro_winkler_similarity
