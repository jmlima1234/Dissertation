"""
Baseline Test Suite for Portuguese Address Search
===============================================

This module provides a comprehensive test suite designed to systematically
evaluate the baseline address search system. It includes diverse queries
that test different aspects of the search functionality.

Test Categories:
- EXACT_MATCH: Perfect matches to test basic functionality
- ABBREVIATION: Common abbreviations (R., Av., etc.)
- TYPO: Common typing errors and misspellings
- PARTIAL: Incomplete addresses or partial information
- POSTCODE_ONLY: Searches using only postal codes
- CITY_ONLY: Searches using only city names
- COMPLEX: Multi-component complex queries
- EDGE_CASE: Boundary conditions and unusual cases
"""

from enum import Enum
from dataclasses import dataclass
from typing import List, Optional

class QueryCategory(Enum):
    """Categories of test queries for systematic evaluation"""
    EXACT_MATCH = "exact_match"
    ABBREVIATION = "abbreviation"
    TYPO = "typo"
    PARTIAL = "partial"
    POSTCODE_ONLY = "postcode_only"
    CITY_ONLY = "city_only"
    COMPLEX = "complex"
    EDGE_CASE = "edge_case"

class DifficultyLevel(Enum):
    """Difficulty levels for search queries"""
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"

@dataclass
class TestQuery:
    """Individual test query with expected results"""
    query: str
    category: QueryCategory
    difficulty: DifficultyLevel
    description: str
    expected_street: Optional[str] = None
    expected_city: Optional[str] = None
    expected_postcode: Optional[str] = None
    expected_municipality: Optional[str] = None
    expected_latitude: Optional[float] = None
    expected_longitude: Optional[float] = None

class BaselineTestSuite:
    """Comprehensive test suite for baseline evaluation"""
    
    def __init__(self):
        """Initialize the test suite with predefined queries"""
        self.queries = self._create_test_queries()
    
    def _create_test_queries(self) -> List[TestQuery]:
        """Create the comprehensive set of test queries"""
        queries = []
        
        # EXACT_MATCH queries - Perfect matches to test basic functionality
        queries.extend([
            TestQuery(
                query="Rua Augusta, Lisboa",
                category=QueryCategory.EXACT_MATCH,
                difficulty=DifficultyLevel.EASY,
                description="Famous street in Lisbon",
                expected_street="Rua Augusta",
                expected_city="Lisboa",
                expected_municipality="Lisboa"
            ),
            TestQuery(
                query="Avenida da Liberdade, Lisboa",
                category=QueryCategory.EXACT_MATCH,
                difficulty=DifficultyLevel.EASY,
                description="Major avenue in Lisbon",
                expected_street="Avenida da Liberdade",
                expected_city="Lisboa",
                expected_municipality="Lisboa"
            ),
            TestQuery(
                query="Rua Santa Catarina, Porto",
                category=QueryCategory.EXACT_MATCH,
                difficulty=DifficultyLevel.EASY,
                description="Shopping street in Porto",
                expected_street="Rua Santa Catarina",
                expected_city="Porto",
                expected_municipality="Porto"
            ),
            TestQuery(
                query="Praça do Comércio, Lisboa",
                category=QueryCategory.EXACT_MATCH,
                difficulty=DifficultyLevel.EASY,
                description="Historic square in Lisbon",
                expected_street="Praça do Comércio",
                expected_city="Lisboa",
                expected_municipality="Lisboa"
            ),
            TestQuery(
                query="Rua das Flores, Porto",
                category=QueryCategory.EXACT_MATCH,
                difficulty=DifficultyLevel.EASY,
                description="Historic street in Porto",
                expected_street="Rua das Flores",
                expected_city="Porto",
                expected_municipality="Porto"
            )
        ])
        
        # ABBREVIATION queries - Common abbreviations used in Portuguese addresses
        queries.extend([
            TestQuery(
                query="R. Augusta, Lisboa",
                category=QueryCategory.ABBREVIATION,
                difficulty=DifficultyLevel.MEDIUM,
                description="Abbreviated Rua Augusta",
                expected_street="Rua Augusta",
                expected_city="Lisboa",
                expected_municipality="Lisboa"
            ),
            TestQuery(
                query="Av. da Liberdade, Lisboa",
                category=QueryCategory.ABBREVIATION,
                difficulty=DifficultyLevel.MEDIUM,
                description="Abbreviated Avenida da Liberdade",
                expected_street="Avenida da Liberdade",
                expected_city="Lisboa",
                expected_municipality="Lisboa"
            ),
            TestQuery(
                query="Pr. do Comércio, Lisboa",
                category=QueryCategory.ABBREVIATION,
                difficulty=DifficultyLevel.MEDIUM,
                description="Abbreviated Praça do Comércio",
                expected_street="Praça do Comércio",
                expected_city="Lisboa",
                expected_municipality="Lisboa"
            ),
            TestQuery(
                query="R. Santa Catarina, Porto",
                category=QueryCategory.ABBREVIATION,
                difficulty=DifficultyLevel.MEDIUM,
                description="Abbreviated Rua Santa Catarina",
                expected_street="Rua Santa Catarina",
                expected_city="Porto",
                expected_municipality="Porto"
            ),
            TestQuery(
                query="Av. dos Aliados, Porto",
                category=QueryCategory.ABBREVIATION,
                difficulty=DifficultyLevel.MEDIUM,
                description="Abbreviated Avenida dos Aliados",
                expected_street="Avenida dos Aliados",
                expected_city="Porto",
                expected_municipality="Porto"
            )
        ])
        
        # TYPO queries - Common typing errors and misspellings
        queries.extend([
            TestQuery(
                query="Rua Agusta, Lisboa",
                category=QueryCategory.TYPO,
                difficulty=DifficultyLevel.HARD,
                description="Misspelled Rua Augusta (missing 'u')",
                expected_street="Rua Augusta",
                expected_city="Lisboa",
                expected_municipality="Lisboa"
            ),
            TestQuery(
                query="Avenida da Liberdde, Lisboa",
                category=QueryCategory.TYPO,
                difficulty=DifficultyLevel.HARD,
                description="Misspelled Liberdade (missing 'a')",
                expected_street="Avenida da Liberdade",
                expected_city="Lisboa",
                expected_municipality="Lisboa"
            ),
            TestQuery(
                query="Rua Santa Catrian, Porto",
                category=QueryCategory.TYPO,
                difficulty=DifficultyLevel.HARD,
                description="Misspelled Catarina",
                expected_street="Rua Santa Catarina",
                expected_city="Porto",
                expected_municipality="Porto"
            ),
            TestQuery(
                query="Praca do Comercio, Lisboa",
                category=QueryCategory.TYPO,
                difficulty=DifficultyLevel.HARD,
                description="Missing accent on Praça and ç in Comércio",
                expected_street="Praça do Comércio",
                expected_city="Lisboa",
                expected_municipality="Lisboa"
            ),
            TestQuery(
                query="Avenida da Republica, Lisboa",
                category=QueryCategory.TYPO,
                difficulty=DifficultyLevel.HARD,
                description="Missing accent on República",
                expected_street="Avenida da República",
                expected_city="Lisboa",
                expected_municipality="Lisboa"
            )
        ])
        
        # PARTIAL queries - Incomplete addresses or partial information
        queries.extend([
            TestQuery(
                query="Augusta Lisboa",
                category=QueryCategory.PARTIAL,
                difficulty=DifficultyLevel.MEDIUM,
                description="Partial query without 'Rua'",
                expected_street="Rua Augusta",
                expected_city="Lisboa",
                expected_municipality="Lisboa"
            ),
            TestQuery(
                query="Liberdade",
                category=QueryCategory.PARTIAL,
                difficulty=DifficultyLevel.HARD,
                description="Single word query",
                expected_street="Avenida da Liberdade",
                expected_city="Lisboa",
                expected_municipality="Lisboa"
            ),
            TestQuery(
                query="Santa Catarina Porto",
                category=QueryCategory.PARTIAL,
                difficulty=DifficultyLevel.MEDIUM,
                description="Partial query without street type",
                expected_street="Rua Santa Catarina",
                expected_city="Porto",
                expected_municipality="Porto"
            ),
            TestQuery(
                query="Comércio",
                category=QueryCategory.PARTIAL,
                difficulty=DifficultyLevel.HARD,
                description="Single word plaza name",
                expected_street="Praça do Comércio",
                expected_city="Lisboa",
                expected_municipality="Lisboa"
            ),
            TestQuery(
                query="Aliados",
                category=QueryCategory.PARTIAL,
                difficulty=DifficultyLevel.HARD,
                description="Single word avenue name",
                expected_street="Avenida dos Aliados",
                expected_city="Porto",
                expected_municipality="Porto"
            )
        ])
        
        # POSTCODE_ONLY queries - Searches using only postal codes
        queries.extend([
            TestQuery(
                query="1100-001",
                category=QueryCategory.POSTCODE_ONLY,
                difficulty=DifficultyLevel.EASY,
                description="Central Lisbon postcode",
                expected_postcode="1100-001",
                expected_city="Lisboa",
                expected_municipality="Lisboa"
            ),
            TestQuery(
                query="4000-001",
                category=QueryCategory.POSTCODE_ONLY,
                difficulty=DifficultyLevel.EASY,
                description="Central Porto postcode",
                expected_postcode="4000-001",
                expected_city="Porto",
                expected_municipality="Porto"
            ),
            TestQuery(
                query="3000-001",
                category=QueryCategory.POSTCODE_ONLY,
                difficulty=DifficultyLevel.EASY,
                description="Central Coimbra postcode",
                expected_postcode="3000-001",
                expected_city="Coimbra",
                expected_municipality="Coimbra"
            ),
            TestQuery(
                query="2700-001",
                category=QueryCategory.POSTCODE_ONLY,
                difficulty=DifficultyLevel.EASY,
                description="Amadora postcode",
                expected_postcode="2700-001",
                expected_city="Amadora",
                expected_municipality="Amadora"
            ),
            TestQuery(
                query="1000-001",
                category=QueryCategory.POSTCODE_ONLY,
                difficulty=DifficultyLevel.EASY,
                description="Historic Lisbon center",
                expected_postcode="1000-001",
                expected_city="Lisboa",
                expected_municipality="Lisboa"
            )
        ])
        
        # CITY_ONLY queries - Searches using only city names
        queries.extend([
            TestQuery(
                query="Lisboa",
                category=QueryCategory.CITY_ONLY,
                difficulty=DifficultyLevel.EASY,
                description="Capital city",
                expected_city="Lisboa",
                expected_municipality="Lisboa"
            ),
            TestQuery(
                query="Porto",
                category=QueryCategory.CITY_ONLY,
                difficulty=DifficultyLevel.EASY,
                description="Second largest city",
                expected_city="Porto",
                expected_municipality="Porto"
            ),
            TestQuery(
                query="Coimbra",
                category=QueryCategory.CITY_ONLY,
                difficulty=DifficultyLevel.EASY,
                description="Historic university city",
                expected_city="Coimbra",
                expected_municipality="Coimbra"
            ),
            TestQuery(
                query="Braga",
                category=QueryCategory.CITY_ONLY,
                difficulty=DifficultyLevel.EASY,
                description="Northern Portuguese city",
                expected_city="Braga",
                expected_municipality="Braga"
            ),
            TestQuery(
                query="Faro",
                category=QueryCategory.CITY_ONLY,
                difficulty=DifficultyLevel.EASY,
                description="Southern coastal city",
                expected_city="Faro",
                expected_municipality="Faro"
            )
        ])
        
        # COMPLEX queries - Multi-component complex queries
        queries.extend([
            TestQuery(
                query="Rua Augusta 100 1100-048 Lisboa",
                category=QueryCategory.COMPLEX,
                difficulty=DifficultyLevel.MEDIUM,
                description="Full address with number and postcode",
                expected_street="Rua Augusta",
                expected_city="Lisboa",
                expected_postcode="1100-048",
                expected_municipality="Lisboa"
            ),
            TestQuery(
                query="Avenida da Liberdade número 200, Lisboa",
                category=QueryCategory.COMPLEX,
                difficulty=DifficultyLevel.MEDIUM,
                description="Address with written number reference",
                expected_street="Avenida da Liberdade",
                expected_city="Lisboa",
                expected_municipality="Lisboa"
            ),
            TestQuery(
                query="Centro Comercial Colombo, Lisboa",
                category=QueryCategory.COMPLEX,
                difficulty=DifficultyLevel.HARD,
                description="Shopping center reference",
                expected_city="Lisboa",
                expected_municipality="Lisboa"
            ),
            TestQuery(
                query="Estação de São Bento, Porto",
                category=QueryCategory.COMPLEX,
                difficulty=DifficultyLevel.MEDIUM,
                description="Train station reference",
                expected_city="Porto",
                expected_municipality="Porto"
            ),
            TestQuery(
                query="Universidade de Coimbra",
                category=QueryCategory.COMPLEX,
                difficulty=DifficultyLevel.MEDIUM,
                description="University reference",
                expected_city="Coimbra",
                expected_municipality="Coimbra"
            )
        ])
        
        # EDGE_CASE queries - Boundary conditions and unusual cases
        queries.extend([
            TestQuery(
                query="",
                category=QueryCategory.EDGE_CASE,
                difficulty=DifficultyLevel.HARD,
                description="Empty query"
            ),
            TestQuery(
                query="   ",
                category=QueryCategory.EDGE_CASE,
                difficulty=DifficultyLevel.HARD,
                description="Whitespace only query"
            ),
            TestQuery(
                query="123456789",
                category=QueryCategory.EDGE_CASE,
                difficulty=DifficultyLevel.HARD,
                description="Numeric only query"
            ),
            TestQuery(
                query="XYZABC",
                category=QueryCategory.EDGE_CASE,
                difficulty=DifficultyLevel.HARD,
                description="Random letters"
            ),
            TestQuery(
                query="Rua Muito Longa Nome de Uma Rua Que Não Existe Em Portugal",
                category=QueryCategory.EDGE_CASE,
                difficulty=DifficultyLevel.HARD,
                description="Very long non-existent street name"
            ),
            TestQuery(
                query="São João da Madeira",
                category=QueryCategory.EDGE_CASE,
                difficulty=DifficultyLevel.MEDIUM,
                description="City with special characters and multiple words",
                expected_city="São João da Madeira",
                expected_municipality="São João da Madeira"
            ),
            TestQuery(
                query="Vila Nova de Gaia",
                category=QueryCategory.EDGE_CASE,
                difficulty=DifficultyLevel.MEDIUM,
                description="Multi-word city name",
                expected_city="Vila Nova de Gaia",
                expected_municipality="Vila Nova de Gaia"
            ),
            TestQuery(
                query="rua augusta lisboa",
                category=QueryCategory.EDGE_CASE,
                difficulty=DifficultyLevel.MEDIUM,
                description="All lowercase query",
                expected_street="Rua Augusta",
                expected_city="Lisboa",
                expected_municipality="Lisboa"
            ),
            TestQuery(
                query="RUA AUGUSTA LISBOA",
                category=QueryCategory.EDGE_CASE,
                difficulty=DifficultyLevel.MEDIUM,
                description="All uppercase query",
                expected_street="Rua Augusta",
                expected_city="Lisboa",
                expected_municipality="Lisboa"
            )
        ])
        
        return queries
    
    def get_all_queries(self) -> List[TestQuery]:
        """Get all test queries"""
        return self.queries
    
    def get_queries_by_category(self, category: QueryCategory) -> List[TestQuery]:
        """Get queries filtered by category"""
        return [q for q in self.queries if q.category == category]
    
    def get_queries_by_difficulty(self, difficulty: DifficultyLevel) -> List[TestQuery]:
        """Get queries filtered by difficulty"""
        return [q for q in self.queries if q.difficulty == difficulty]
    
    def get_query_stats(self) -> dict:
        """Get statistics about the test suite"""
        total = len(self.queries)
        
        by_category = {}
        for category in QueryCategory:
            by_category[category.value] = len(self.get_queries_by_category(category))
        
        by_difficulty = {}
        for difficulty in DifficultyLevel:
            by_difficulty[difficulty.value] = len(self.get_queries_by_difficulty(difficulty))
        
        return {
            'total_queries': total,
            'by_category': by_category,
            'by_difficulty': by_difficulty
        }

# Usage example
if __name__ == "__main__":
    test_suite = BaselineTestSuite()
    stats = test_suite.get_query_stats()
    
    print("=== Baseline Test Suite Statistics ===")
    print(f"Total Queries: {stats['total_queries']}")
    
    print("\nBy Category:")
    for category, count in stats['by_category'].items():
        print(f"  {category.replace('_', ' ').title()}: {count} queries")
    
    print("\nBy Difficulty:")
    for difficulty, count in stats['by_difficulty'].items():
        print(f"  {difficulty.title()}: {count} queries")
    
    print("\nSample Queries:")
    for i, query in enumerate(test_suite.get_all_queries()[:3]):
        print(f"  {i+1}. '{query.query}' ({query.category.value}, {query.difficulty.value})")
        print(f"     {query.description}")