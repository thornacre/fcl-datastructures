"""
Bloom Filters for Probabilistic Membership Testing
==================================================

This module implements Bloom filters for efficient probabilistic membership testing
in Find Case Law (FCL). Provides memory-efficient storage for large sets of legal
citations, case names, and other identifiers with fast lookup capabilities.

Key FCL Use Cases:
- Fast citation existence checking before expensive database queries
- Duplicate judgment detection during bulk imports
- Efficient set operations on large collections of case identifiers
- Cache miss optimization for frequently accessed legal documents
- Real-time filtering of search results by known citation patterns
"""

import hashlib
import math
from typing import List, Set, Optional, Any, Union
import mmh3  # MurmurHash3 - can fall back to hashlib if not available
from dataclasses import dataclass
from bitarray import bitarray  # Can fall back to list of booleans
import json


@dataclass
class BloomFilterStats:
    """Statistics about a Bloom filter's state and performance"""
    capacity: int
    num_hash_functions: int
    bit_array_size: int
    items_added: int
    estimated_false_positive_rate: float
    memory_usage_bytes: int


class BloomFilter:
    """
    Space-efficient probabilistic data structure for membership testing.
    Optimized for legal citation and case name filtering in FCL.
    """

    def __init__(self, capacity: int = 10000, false_positive_rate: float = 0.01):
        """
        Initialize Bloom filter with specified capacity and error rate.

        Args:
            capacity: Expected number of items to store
            false_positive_rate: Desired false positive probability (0.0 to 1.0)
        """
        self.capacity = capacity
        self.false_positive_rate = false_positive_rate

        # Calculate optimal bit array size and number of hash functions
        self.bit_array_size = self._calculate_bit_array_size(capacity, false_positive_rate)
        self.num_hash_functions = self._calculate_num_hash_functions(
            self.bit_array_size, capacity
        )

        # Initialize bit array
        try:
            self.bit_array = bitarray(self.bit_array_size)
            self.bit_array.setall(0)
        except ImportError:
            # Fallback to list of booleans if bitarray not available
            self.bit_array = [False] * self.bit_array_size

        self.items_added = 0

    def add(self, item: Union[str, bytes]) -> None:
        """
        Add an item to the Bloom filter.

        Args:
            item: Item to add (string or bytes)
        """
        if isinstance(item, str):
            item = item.encode('utf-8')

        # Generate hash values and set corresponding bits
        for i in range(self.num_hash_functions):
            hash_value = self._hash(item, i)
            bit_index = hash_value % self.bit_array_size
            self.bit_array[bit_index] = True

        self.items_added += 1

    def contains(self, item: Union[str, bytes]) -> bool:
        """
        Test if an item might be in the set.

        Args:
            item: Item to test

        Returns:
            True if item might be in set (with small false positive probability)
            False if item is definitely not in set
        """
        if isinstance(item, str):
            item = item.encode('utf-8')

        # Check all hash positions
        for i in range(self.num_hash_functions):
            hash_value = self._hash(item, i)
            bit_index = hash_value % self.bit_array_size
            if not self.bit_array[bit_index]:
                return False

        return True

    def add_multiple(self, items: List[Union[str, bytes]]) -> None:
        """Add multiple items efficiently"""
        for item in items:
            self.add(item)

    def union(self, other: 'BloomFilter') -> 'BloomFilter':
        """
        Create union of two Bloom filters (logical OR operation).
        Both filters must have same parameters.
        """
        if (self.bit_array_size != other.bit_array_size or
            self.num_hash_functions != other.num_hash_functions):
            raise ValueError("Cannot union Bloom filters with different parameters")

        # Create new filter with same parameters
        result = BloomFilter(self.capacity, self.false_positive_rate)

        # Perform bitwise OR
        for i in range(self.bit_array_size):
            result.bit_array[i] = self.bit_array[i] or other.bit_array[i]

        # Estimate items in union (approximate)
        result.items_added = max(self.items_added, other.items_added)

        return result

    def intersection_estimate(self, other: 'BloomFilter') -> float:
        """
        Estimate the size of intersection with another Bloom filter.
        Returns approximate number of common elements.
        """
        if (self.bit_array_size != other.bit_array_size or
            self.num_hash_functions != other.num_hash_functions):
            raise ValueError("Cannot intersect Bloom filters with different parameters")

        # Count bits set in both filters
        common_bits = sum(1 for i in range(self.bit_array_size)
                         if self.bit_array[i] and other.bit_array[i])

        # Estimate intersection size using formula
        if common_bits == 0:
            return 0.0

        # Simplified estimation (can be improved with more sophisticated formulas)
        estimated_intersection = (
            common_bits * self.capacity * other.capacity /
            (self.bit_array_size * max(self.items_added, other.items_added))
        )

        return max(0.0, estimated_intersection)

    def get_statistics(self) -> BloomFilterStats:
        """Get comprehensive statistics about the filter"""
        # Calculate current false positive rate
        current_fpr = self._calculate_current_false_positive_rate()

        # Estimate memory usage
        try:
            memory_usage = self.bit_array.buffer_info()[1] * self.bit_array.itemsize
        except AttributeError:
            # Fallback for list implementation
            memory_usage = len(self.bit_array) * 1  # Rough estimate

        return BloomFilterStats(
            capacity=self.capacity,
            num_hash_functions=self.num_hash_functions,
            bit_array_size=self.bit_array_size,
            items_added=self.items_added,
            estimated_false_positive_rate=current_fpr,
            memory_usage_bytes=memory_usage
        )

    def save_to_file(self, filename: str) -> None:
        """Save Bloom filter to file"""
        data = {
            'capacity': self.capacity,
            'false_positive_rate': self.false_positive_rate,
            'bit_array_size': self.bit_array_size,
            'num_hash_functions': self.num_hash_functions,
            'items_added': self.items_added,
            'bit_array': self._bit_array_to_bytes()
        }

        with open(filename, 'w') as f:
            json.dump(data, f)

    @classmethod
    def load_from_file(cls, filename: str) -> 'BloomFilter':
        """Load Bloom filter from file"""
        with open(filename, 'r') as f:
            data = json.load(f)

        filter_obj = cls(data['capacity'], data['false_positive_rate'])
        filter_obj.items_added = data['items_added']
        filter_obj._bit_array_from_bytes(data['bit_array'])

        return filter_obj

    def _hash(self, item: bytes, seed: int) -> int:
        """Generate hash value for item with given seed"""
        try:
            # Use MurmurHash3 if available (faster and better distribution)
            return mmh3.hash(item, seed) & 0x7FFFFFFF  # Ensure positive
        except (ImportError, NameError):
            # Fallback to hashlib
            hasher = hashlib.md5()
            hasher.update(item)
            hasher.update(seed.to_bytes(4, 'big'))
            return int.from_bytes(hasher.digest()[:4], 'big')

    def _calculate_bit_array_size(self, capacity: int, fp_rate: float) -> int:
        """Calculate optimal bit array size"""
        # Formula: m = -(n * ln(p)) / (ln(2)^2)
        return int(-capacity * math.log(fp_rate) / (math.log(2) ** 2))

    def _calculate_num_hash_functions(self, bit_array_size: int, capacity: int) -> int:
        """Calculate optimal number of hash functions"""
        # Formula: k = (m/n) * ln(2)
        return max(1, int((bit_array_size / capacity) * math.log(2)))

    def _calculate_current_false_positive_rate(self) -> float:
        """Calculate current false positive rate based on items added"""
        if self.items_added == 0:
            return 0.0

        # Formula: (1 - e^(-k*n/m))^k
        try:
            exponent = -self.num_hash_functions * self.items_added / self.bit_array_size
            return (1 - math.exp(exponent)) ** self.num_hash_functions
        except (OverflowError, ZeroDivisionError):
            return 1.0

    def _bit_array_to_bytes(self) -> str:
        """Convert bit array to base64 string for serialization"""
        try:
            return self.bit_array.tobytes().hex()
        except AttributeError:
            # Fallback for list implementation
            bit_string = ''.join('1' if bit else '0' for bit in self.bit_array)
            # Pad to byte boundary
            while len(bit_string) % 8 != 0:
                bit_string += '0'
            # Convert to bytes and then hex
            bytes_data = bytes(int(bit_string[i:i+8], 2) for i in range(0, len(bit_string), 8))
            return bytes_data.hex()

    def _bit_array_from_bytes(self, hex_string: str) -> None:
        """Restore bit array from hex string"""
        bytes_data = bytes.fromhex(hex_string)
        try:
            self.bit_array = bitarray()
            self.bit_array.frombytes(bytes_data)
            # Trim to correct size
            self.bit_array = self.bit_array[:self.bit_array_size]
        except (ImportError, AttributeError):
            # Fallback for list implementation
            bit_string = ''.join(format(byte, '08b') for byte in bytes_data)
            self.bit_array = [bit == '1' for bit in bit_string[:self.bit_array_size]]


class CitationBloomFilter:
    """
    Specialized Bloom filter for UK legal citations with optimized
    hashing and validation for legal document identifiers.
    """

    def __init__(self, capacity: int = 50000, false_positive_rate: float = 0.001):
        """
        Initialize citation filter with legal-specific optimizations.

        Args:
            capacity: Expected number of citations
            false_positive_rate: Lower error rate for critical legal lookups
        """
        self.filter = BloomFilter(capacity, false_positive_rate)

        # Legal citation patterns for validation
        self.citation_patterns = [
            r'\[(\d{4})\]\s+([A-Z]{2,})\s+(\d+)',  # [2023] UKSC 15
            r'\[(\d{4})\]\s+([A-Z]{2,})\s+([A-Za-z]+)\s+(\d+)',  # [2023] EWCA Civ 892
            r'\((\d{4})\)\s+([A-Z]{2,})\s+(\d+)',  # (2023) AC 123
        ]

        # Court hierarchies for weighting
        self.court_weights = {
            'UKSC': 1.0,    # Supreme Court - highest priority
            'UKHL': 0.95,   # House of Lords
            'EWCA': 0.9,    # Court of Appeal
            'EWHC': 0.85,   # High Court
            'UKUT': 0.8,    # Upper Tribunal
            'UKFTT': 0.75,  # First-tier Tribunal
        }

    def add_citation(self, citation: str) -> bool:
        """
        Add a legal citation to the filter with validation.

        Args:
            citation: Legal citation string

        Returns:
            True if citation was added, False if invalid format
        """
        normalized = self._normalize_citation(citation)
        if normalized:
            self.filter.add(normalized)
            return True
        return False

    def contains_citation(self, citation: str) -> bool:
        """
        Check if citation might exist in the filter.

        Args:
            citation: Legal citation to check

        Returns:
            True if citation might exist, False if definitely not
        """
        normalized = self._normalize_citation(citation)
        if not normalized:
            return False
        return self.filter.contains(normalized)

    def add_citations_bulk(self, citations: List[str]) -> int:
        """
        Add multiple citations efficiently.

        Args:
            citations: List of legal citations

        Returns:
            Number of successfully added citations
        """
        added_count = 0
        normalized_citations = []

        for citation in citations:
            normalized = self._normalize_citation(citation)
            if normalized:
                normalized_citations.append(normalized)
                added_count += 1

        self.filter.add_multiple(normalized_citations)
        return added_count

    def get_court_coverage(self, citations: List[str]) -> dict:
        """
        Analyze court coverage in the filter.

        Args:
            citations: Sample citations to analyze

        Returns:
            Dictionary of court -> count mappings
        """
        court_counts = {}

        for citation in citations:
            court = self._extract_court(citation)
            if court and self.contains_citation(citation):
                court_counts[court] = court_counts.get(court, 0) + 1

        return court_counts

    def _normalize_citation(self, citation: str) -> Optional[str]:
        """
        Normalize legal citation for consistent storage.

        Args:
            citation: Raw citation string

        Returns:
            Normalized citation or None if invalid
        """
        # Remove extra whitespace and standardize format
        citation = ' '.join(citation.split())

        # Basic validation - must contain year and court
        if not any(char.isdigit() for char in citation):
            return None

        if not any(char.isupper() for char in citation):
            return None

        # Convert to uppercase for consistency
        return citation.upper().strip()

    def _extract_court(self, citation: str) -> Optional[str]:
        """Extract court identifier from citation"""
        import re
        for pattern in self.citation_patterns:
            match = re.search(pattern, citation)
            if match:
                # Return the court identifier (second capture group)
                return match.group(2) if len(match.groups()) >= 2 else None
        return None


class ScalableBloomFilter:
    """
    Scalable Bloom filter that automatically grows when capacity is exceeded.
    Useful for long-running FCL systems with unpredictable data volumes.
    """

    def __init__(self, initial_capacity: int = 10000,
                 false_positive_rate: float = 0.01,
                 growth_factor: int = 2):
        """
        Initialize scalable filter.

        Args:
            initial_capacity: Starting capacity
            false_positive_rate: Target false positive rate
            growth_factor: Capacity multiplier when scaling
        """
        self.initial_capacity = initial_capacity
        self.false_positive_rate = false_positive_rate
        self.growth_factor = growth_factor

        # List of Bloom filters (each with increasing capacity)
        self.filters = [BloomFilter(initial_capacity, false_positive_rate)]
        self.current_filter_index = 0

    def add(self, item: Union[str, bytes]) -> None:
        """Add item to the appropriate filter"""
        current_filter = self.filters[self.current_filter_index]

        # Check if current filter is full
        if current_filter.items_added >= current_filter.capacity:
            self._create_new_filter()
            current_filter = self.filters[self.current_filter_index]

        current_filter.add(item)

    def contains(self, item: Union[str, bytes]) -> bool:
        """Check if item exists in any of the filters"""
        # Check all filters (most recent first for better performance)
        for filter_obj in reversed(self.filters):
            if filter_obj.contains(item):
                return True
        return False

    def get_total_capacity(self) -> int:
        """Get total capacity across all filters"""
        return sum(f.capacity for f in self.filters)

    def get_total_items(self) -> int:
        """Get total items across all filters"""
        return sum(f.items_added for f in self.filters)

    def get_overall_statistics(self) -> dict:
        """Get comprehensive statistics for all filters"""
        return {
            'num_filters': len(self.filters),
            'total_capacity': self.get_total_capacity(),
            'total_items': self.get_total_items(),
            'memory_usage_bytes': sum(f.get_statistics().memory_usage_bytes for f in self.filters),
            'average_false_positive_rate': sum(f.get_statistics().estimated_false_positive_rate
                                             for f in self.filters) / len(self.filters)
        }

    def _create_new_filter(self) -> None:
        """Create new filter with increased capacity"""
        new_capacity = int(self.initial_capacity * (self.growth_factor ** len(self.filters)))
        new_filter = BloomFilter(new_capacity, self.false_positive_rate)
        self.filters.append(new_filter)
        self.current_filter_index = len(self.filters) - 1


def demonstrate_bloom_filters():
    """Demonstrate Bloom filter implementations with UK legal data."""

    print("=== Bloom Filters for Legal Citation Testing Demo ===\n")

    # Sample UK legal citations
    sample_citations = [
        "[2023] UKSC 15",
        "[2023] EWCA Civ 892",
        "[2023] EWHC 1456 (Admin)",
        "[2022] UKHL 7",
        "[2023] UKFTT 234 (TC)",
        "[2022] EWCA Crim 567",
        "[2023] EWHC 789 (QB)",
        "[2021] UKSC 42",
        "(2023) AC 123",
        "[2023] UKUT 456 (AAC)"
    ]

    # Additional citations for testing false positives
    test_citations = [
        "[2024] UKSC 99",   # Future citation (should not exist)
        "[2023] FAKE 123",  # Invalid court
        "[2023] UKSC 999",  # High number (might not exist)
        "[2020] EWCA Civ 1",  # Old citation
    ]

    # 1. Basic Bloom Filter Demo
    print("1. BASIC BLOOM FILTER:")
    basic_filter = BloomFilter(capacity=1000, false_positive_rate=0.01)

    print("   Adding sample citations:")
    for citation in sample_citations:
        basic_filter.add(citation)
        print(f"   Added: {citation}")

    print(f"\n   Filter statistics:")
    stats = basic_filter.get_statistics()
    print(f"     Items added: {stats.items_added}")
    print(f"     Bit array size: {stats.bit_array_size}")
    print(f"     Hash functions: {stats.num_hash_functions}")
    print(f"     Memory usage: {stats.memory_usage_bytes} bytes")
    print(f"     Est. false positive rate: {stats.estimated_false_positive_rate:.4f}")

    # 2. Membership Testing
    print(f"\n2. MEMBERSHIP TESTING:")
    print("   Testing known citations (should all return True):")
    for citation in sample_citations[:5]:
        result = basic_filter.contains(citation)
        print(f"     {citation}: {result}")

    print("\n   Testing unknown citations (may have false positives):")
    for citation in test_citations:
        result = basic_filter.contains(citation)
        print(f"     {citation}: {result}")

    # 3. Citation-Specific Bloom Filter
    print(f"\n3. CITATION-SPECIFIC BLOOM FILTER:")
    citation_filter = CitationBloomFilter(capacity=5000, false_positive_rate=0.001)

    added_count = citation_filter.add_citations_bulk(sample_citations)
    print(f"   Successfully added {added_count}/{len(sample_citations)} citations")

    print("   Citation validation and lookup:")
    test_cases = sample_citations[:3] + ["INVALID CITATION", "[2024] UNKNOWN 1"]

    for citation in test_cases:
        exists = citation_filter.contains_citation(citation)
        normalized = citation_filter._normalize_citation(citation)
        print(f"     {citation}: exists={exists}, normalized='{normalized}'")

    # 4. Court Coverage Analysis
    print(f"\n4. COURT COVERAGE ANALYSIS:")
    court_coverage = citation_filter.get_court_coverage(sample_citations)
    print("   Citations by court:")
    for court, count in court_coverage.items():
        print(f"     {court}: {count} citations")

    # 5. Scalable Bloom Filter Demo
    print(f"\n5. SCALABLE BLOOM FILTER:")
    scalable_filter = ScalableBloomFilter(
        initial_capacity=5,  # Small capacity to force scaling
        false_positive_rate=0.01,
        growth_factor=2
    )

    print("   Adding citations to trigger scaling:")
    for i, citation in enumerate(sample_citations):
        scalable_filter.add(citation)
        total_items = scalable_filter.get_total_items()
        num_filters = len(scalable_filter.filters)
        print(f"     Added {citation}: {total_items} items in {num_filters} filter(s)")

    overall_stats = scalable_filter.get_overall_statistics()
    print(f"\n   Scalable filter statistics:")
    for key, value in overall_stats.items():
        print(f"     {key}: {value}")

    # 6. Filter Union Operations
    print(f"\n6. BLOOM FILTER UNION:")
    filter1 = BloomFilter(capacity=1000, false_positive_rate=0.01)
    filter2 = BloomFilter(capacity=1000, false_positive_rate=0.01)

    # Add different sets to each filter
    for citation in sample_citations[:5]:
        filter1.add(citation)

    for citation in sample_citations[3:8]:  # Some overlap
        filter2.add(citation)

    print(f"   Filter 1 items: {filter1.items_added}")
    print(f"   Filter 2 items: {filter2.items_added}")

    # Create union
    union_filter = filter1.union(filter2)
    print(f"   Union filter created")

    # Test union membership
    print("   Testing union membership:")
    test_items = sample_citations[:8]
    for citation in test_items:
        in_union = union_filter.contains(citation)
        in_filter1 = filter1.contains(citation)
        in_filter2 = filter2.contains(citation)
        print(f"     {citation}: union={in_union}, f1={in_filter1}, f2={in_filter2}")

    # 7. Intersection Estimation
    print(f"\n7. INTERSECTION ESTIMATION:")
    estimated_intersection = filter1.intersection_estimate(filter2)
    print(f"   Estimated intersection size: {estimated_intersection:.2f}")

    # 8. Performance Comparison
    print(f"\n8. PERFORMANCE AND MEMORY EFFICIENCY:")

    # Compare with set-based approach
    citation_set = set(sample_citations)
    print(f"   Python set memory (approx): {len(str(citation_set))} bytes")
    print(f"   Bloom filter memory: {basic_filter.get_statistics().memory_usage_bytes} bytes")

    # Memory efficiency ratio
    if basic_filter.get_statistics().memory_usage_bytes > 0:
        efficiency_ratio = len(str(citation_set)) / basic_filter.get_statistics().memory_usage_bytes
        print(f"   Memory efficiency ratio: {efficiency_ratio:.2f}x")

    return {
        'basic_filter': basic_filter,
        'citation_filter': citation_filter,
        'scalable_filter': scalable_filter,
        'union_filter': union_filter,
        'statistics': {
            'basic_stats': stats,
            'overall_stats': overall_stats,
            'court_coverage': court_coverage
        }
    }


if __name__ == "__main__":
    demonstrate_bloom_filters()