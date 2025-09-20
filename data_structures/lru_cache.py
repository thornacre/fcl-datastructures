"""
LRU Cache for Frequently Accessed Judgments
===========================================

This module implements an LRU (Least Recently Used) cache optimized for storing
and retrieving frequently accessed legal judgments in Find Case Law (FCL).
The cache provides fast access to popular cases while managing memory efficiently.

Key FCL Use Cases:
- Cache parsed judgment XML documents
- Store processed citation networks
- Cache search result metadata
- Store user session data and preferences
- Cache expensive API responses
- Store preprocessed text embeddings
"""

from typing import Optional, Any, Dict, List, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import hashlib
import json
import threading
from collections import OrderedDict
import time


@dataclass
class CacheEntry:
    """Represents a single cache entry with metadata"""
    key: str
    value: Any
    access_count: int = 0
    created_at: datetime = field(default_factory=datetime.now)
    last_accessed: datetime = field(default_factory=datetime.now)
    size_bytes: int = 0
    ttl_seconds: Optional[int] = None

    def is_expired(self) -> bool:
        """Check if cache entry has expired based on TTL"""
        if self.ttl_seconds is None:
            return False
        return (datetime.now() - self.created_at).total_seconds() > self.ttl_seconds

    def touch(self):
        """Update access metadata"""
        self.last_accessed = datetime.now()
        self.access_count += 1


class LRUCache:
    """
    Thread-safe LRU Cache implementation for legal document storage.

    Features:
    - Configurable capacity with automatic eviction
    - TTL (Time To Live) support for entries
    - Access statistics and cache hit metrics
    - Memory usage tracking
    - Thread-safe operations
    """

    def __init__(self, capacity: int = 1000, default_ttl: Optional[int] = None):
        self.capacity = capacity
        self.default_ttl = default_ttl
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = threading.RLock()

        # Statistics
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._total_size_bytes = 0

    def get(self, key: str) -> Optional[Any]:
        """Retrieve value from cache, updating access statistics"""
        with self._lock:
            if key not in self._cache:
                self._misses += 1
                return None

            entry = self._cache[key]

            # Check if entry has expired
            if entry.is_expired():
                self._remove_entry(key)
                self._misses += 1
                return None

            # Move to end (most recently used)
            self._cache.move_to_end(key)
            entry.touch()
            self._hits += 1

            return entry.value

    def put(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Store value in cache with optional TTL"""
        with self._lock:
            # Calculate size estimate
            size_bytes = self._estimate_size(value)

            # Remove existing entry if present
            if key in self._cache:
                self._remove_entry(key)

            # Create new entry
            entry_ttl = ttl if ttl is not None else self.default_ttl
            entry = CacheEntry(
                key=key,
                value=value,
                size_bytes=size_bytes,
                ttl_seconds=entry_ttl
            )

            # Add to cache
            self._cache[key] = entry
            self._total_size_bytes += size_bytes

            # Evict if over capacity
            self._evict_if_needed()

    def delete(self, key: str) -> bool:
        """Remove entry from cache"""
        with self._lock:
            if key in self._cache:
                self._remove_entry(key)
                return True
            return False

    def clear(self) -> None:
        """Remove all entries from cache"""
        with self._lock:
            self._cache.clear()
            self._total_size_bytes = 0

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self._lock:
            total_requests = self._hits + self._misses
            hit_rate = self._hits / total_requests if total_requests > 0 else 0

            return {
                'capacity': self.capacity,
                'size': len(self._cache),
                'hits': self._hits,
                'misses': self._misses,
                'hit_rate': hit_rate,
                'evictions': self._evictions,
                'total_size_bytes': self._total_size_bytes,
                'avg_size_per_entry': self._total_size_bytes / len(self._cache) if self._cache else 0
            }

    def get_entries_by_access_pattern(self, limit: int = 10) -> List[Tuple[str, int, datetime]]:
        """Get most/least accessed entries for analysis"""
        with self._lock:
            entries = [(key, entry.access_count, entry.last_accessed)
                      for key, entry in self._cache.items()]
            return sorted(entries, key=lambda x: x[1], reverse=True)[:limit]

    def cleanup_expired(self) -> int:
        """Remove all expired entries and return count removed"""
        with self._lock:
            expired_keys = [key for key, entry in self._cache.items() if entry.is_expired()]
            for key in expired_keys:
                self._remove_entry(key)
            return len(expired_keys)

    def _remove_entry(self, key: str) -> None:
        """Internal method to remove entry and update statistics"""
        if key in self._cache:
            entry = self._cache[key]
            self._total_size_bytes -= entry.size_bytes
            del self._cache[key]

    def _evict_if_needed(self) -> None:
        """Evict least recently used entries if over capacity"""
        while len(self._cache) > self.capacity:
            # Remove least recently used (first item)
            oldest_key = next(iter(self._cache))
            self._remove_entry(oldest_key)
            self._evictions += 1

    def _estimate_size(self, value: Any) -> int:
        """Estimate size of value in bytes"""
        try:
            if isinstance(value, str):
                return len(value.encode('utf-8'))
            elif isinstance(value, (int, float)):
                return 8
            elif isinstance(value, dict):
                return len(json.dumps(value, default=str).encode('utf-8'))
            elif hasattr(value, '__len__'):
                return len(str(value).encode('utf-8'))
            else:
                return len(str(value).encode('utf-8'))
        except:
            return 1024  # Default estimate


class JudgmentCache:
    """
    Specialized cache for legal judgment documents with domain-specific features.

    Features:
    - Separate caches for different data types
    - Citation-based key generation
    - Automatic metadata extraction
    - Court hierarchy awareness
    """

    def __init__(self,
                 judgment_capacity: int = 500,
                 citation_capacity: int = 1000,
                 search_capacity: int = 200):

        # Separate caches for different data types
        self.judgment_cache = LRUCache(capacity=judgment_capacity, default_ttl=3600)  # 1 hour
        self.citation_cache = LRUCache(capacity=citation_capacity, default_ttl=7200)  # 2 hours
        self.search_cache = LRUCache(capacity=search_capacity, default_ttl=1800)    # 30 minutes
        self.metadata_cache = LRUCache(capacity=judgment_capacity, default_ttl=3600)

        # Court priority for cache retention
        self.court_priorities = {
            'UKSC': 10,    # Supreme Court - highest priority
            'UKHL': 9,     # House of Lords
            'EWCA': 8,     # Court of Appeal
            'EWHC': 7,     # High Court
            'UKUT': 6,     # Upper Tribunal
            'UKFTT': 5,    # First-tier Tribunal
        }

    def cache_judgment(self, citation: str, judgment_data: Dict[str, Any],
                      court: str = None) -> None:
        """Cache a complete judgment document"""
        key = self._normalize_citation(citation)

        # Add court priority for better retention
        priority_ttl = self._get_priority_ttl(court)

        self.judgment_cache.put(key, judgment_data, ttl=priority_ttl)

        # Cache metadata separately for quick access
        metadata = self._extract_metadata(judgment_data)
        self.metadata_cache.put(f"meta_{key}", metadata, ttl=priority_ttl)

    def get_judgment(self, citation: str) -> Optional[Dict[str, Any]]:
        """Retrieve cached judgment by citation"""
        key = self._normalize_citation(citation)
        return self.judgment_cache.get(key)

    def cache_citation_network(self, base_citation: str,
                             citations: List[Dict[str, Any]]) -> None:
        """Cache citation network for a judgment"""
        key = f"citations_{self._normalize_citation(base_citation)}"
        self.citation_cache.put(key, citations)

    def get_citation_network(self, citation: str) -> Optional[List[Dict[str, Any]]]:
        """Retrieve cached citation network"""
        key = f"citations_{self._normalize_citation(citation)}"
        return self.citation_cache.get(key)

    def cache_search_results(self, query_hash: str, results: List[Dict[str, Any]],
                           query_metadata: Dict[str, Any] = None) -> None:
        """Cache search results with query hash"""
        cache_data = {
            'results': results,
            'metadata': query_metadata or {},
            'cached_at': datetime.now().isoformat()
        }
        self.search_cache.put(query_hash, cache_data)

    def get_search_results(self, query_hash: str) -> Optional[Dict[str, Any]]:
        """Retrieve cached search results"""
        return self.search_cache.get(query_hash)

    def generate_query_hash(self, query: str, filters: Dict[str, Any] = None,
                          page: int = 1, per_page: int = 20) -> str:
        """Generate consistent hash for search queries"""
        query_data = {
            'query': query.lower().strip(),
            'filters': filters or {},
            'page': page,
            'per_page': per_page
        }
        query_string = json.dumps(query_data, sort_keys=True)
        return hashlib.md5(query_string.encode()).hexdigest()

    def get_cache_statistics(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics"""
        return {
            'judgment_cache': self.judgment_cache.get_stats(),
            'citation_cache': self.citation_cache.get_stats(),
            'search_cache': self.search_cache.get_stats(),
            'metadata_cache': self.metadata_cache.get_stats(),
        }

    def cleanup_all_expired(self) -> Dict[str, int]:
        """Clean up expired entries across all caches"""
        return {
            'judgments': self.judgment_cache.cleanup_expired(),
            'citations': self.citation_cache.cleanup_expired(),
            'searches': self.search_cache.cleanup_expired(),
            'metadata': self.metadata_cache.cleanup_expired(),
        }

    def preload_important_judgments(self, important_citations: List[str]) -> None:
        """Preload frequently accessed judgments into cache"""
        for citation in important_citations:
            # In a real implementation, this would fetch from database
            # For demo, we'll create placeholder data
            placeholder_data = {
                'citation': citation,
                'court': self._extract_court_from_citation(citation),
                'text': f'Placeholder judgment text for {citation}',
                'preloaded': True,
                'preload_time': datetime.now().isoformat()
            }
            self.cache_judgment(citation, placeholder_data)

    def _normalize_citation(self, citation: str) -> str:
        """Normalize citation format for consistent keys"""
        # Remove extra whitespace and standardize format
        return citation.strip().replace('  ', ' ').upper()

    def _extract_court_from_citation(self, citation: str) -> str:
        """Extract court code from citation"""
        import re
        # Look for common UK court patterns
        patterns = {
            r'\bUKSC\b': 'UKSC',
            r'\bUKHL\b': 'UKHL',
            r'\bEWCA\b': 'EWCA',
            r'\bEWHC\b': 'EWHC',
            r'\bUKUT\b': 'UKUT',
            r'\bUKFTT\b': 'UKFTT',
        }

        for pattern, court in patterns.items():
            if re.search(pattern, citation, re.IGNORECASE):
                return court
        return 'UNKNOWN'

    def _get_priority_ttl(self, court: str) -> int:
        """Get TTL based on court priority"""
        if not court:
            return 3600  # Default 1 hour

        priority = self.court_priorities.get(court, 1)
        # Higher priority courts get longer TTL
        return 3600 + (priority * 600)  # Base 1 hour + up to 6 hours

    def _extract_metadata(self, judgment_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract lightweight metadata from judgment data"""
        return {
            'citation': judgment_data.get('citation', ''),
            'court': judgment_data.get('court', ''),
            'date': judgment_data.get('date', ''),
            'judges': judgment_data.get('judges', [])[:3],  # First 3 judges only
            'subject_matter': judgment_data.get('subject_matter', [])[:5],  # First 5 subjects
            'text_length': len(judgment_data.get('text', '')),
            'citation_count': len(judgment_data.get('citations', [])),
            'cached_at': datetime.now().isoformat()
        }


def demonstrate_lru_cache():
    """Demonstrate LRU cache with UK legal data"""

    print("=== LRU Cache for Legal Judgments Demo ===")

    # Sample UK legal data
    sample_judgments = [
        {
            'citation': '[2023] UKSC 15',
            'court': 'UKSC',
            'date': '2023-05-15',
            'judges': ['Lord Reed', 'Lady Hale', 'Lord Kerr'],
            'text': 'This case concerns constitutional law and parliamentary sovereignty...',
            'subject_matter': ['Constitutional Law', 'Parliamentary Sovereignty'],
            'citations': ['[2019] UKSC 41', '[2017] UKSC 5']
        },
        {
            'citation': '[2023] EWCA Civ 892',
            'court': 'EWCA',
            'date': '2023-08-22',
            'judges': ['Sir Geoffrey Vos MR', 'Lord Justice Singh'],
            'text': 'The appellant challenges the decision on grounds of procedural fairness...',
            'subject_matter': ['Administrative Law', 'Judicial Review'],
            'citations': ['[2019] UKSC 41', '[2018] EWCA Civ 234']
        },
        {
            'citation': '[2023] EWHC 1456 (Admin)',
            'court': 'EWHC',
            'date': '2023-06-30',
            'judges': ['Mr Justice Swift'],
            'text': 'This judicial review concerns the lawfulness of the decision...',
            'subject_matter': ['Administrative Law', 'Statutory Interpretation'],
            'citations': ['[2023] UKSC 15', '[2018] EWCA Civ 234']
        }
    ]

    # 1. Basic LRU Cache Demo
    print("\n1. BASIC LRU CACHE OPERATIONS:")
    cache = LRUCache(capacity=3)

    # Add some judgments
    for judgment in sample_judgments:
        cache.put(judgment['citation'], judgment)
        print(f"   Cached: {judgment['citation']}")

    # Test retrieval
    print("\n   Cache retrievals:")
    for citation in ['[2023] UKSC 15', '[2023] EWCA Civ 892', '[2023] EWHC 1456 (Admin)']:
        result = cache.get(citation)
        print(f"   {citation}: {'HIT' if result else 'MISS'}")

    # Add one more to trigger eviction
    new_judgment = {
        'citation': '[2023] UKSC 28',
        'court': 'UKSC',
        'text': 'New Supreme Court judgment...'
    }
    cache.put(new_judgment['citation'], new_judgment)
    print(f"\n   Added: {new_judgment['citation']} (should evict least recently used)")

    # Check what got evicted
    print("\n   After eviction:")
    for citation in ['[2023] UKSC 15', '[2023] EWCA Civ 892', '[2023] EWHC 1456 (Admin)', '[2023] UKSC 28']:
        result = cache.get(citation)
        print(f"   {citation}: {'HIT' if result else 'MISS (evicted)'}")

    # Show statistics
    stats = cache.get_stats()
    print(f"\n   Cache Statistics:")
    print(f"   - Hit rate: {stats['hit_rate']:.1%}")
    print(f"   - Total hits: {stats['hits']}")
    print(f"   - Total misses: {stats['misses']}")
    print(f"   - Evictions: {stats['evictions']}")

    # 2. Specialized Judgment Cache Demo
    print("\n2. SPECIALIZED JUDGMENT CACHE:")
    judgment_cache = JudgmentCache(judgment_capacity=5, citation_capacity=10, search_capacity=3)

    # Cache judgments with court priorities
    for judgment in sample_judgments:
        judgment_cache.cache_judgment(
            judgment['citation'],
            judgment,
            court=judgment['court']
        )
        print(f"   Cached judgment: {judgment['citation']} (Court: {judgment['court']})")

    # Test retrieval
    print("\n   Judgment retrievals:")
    for citation in ['[2023] UKSC 15', '[2023] EWCA Civ 892']:
        result = judgment_cache.get_judgment(citation)
        print(f"   {citation}: {'FOUND' if result else 'NOT FOUND'}")

    # Cache citation networks
    print("\n   Caching citation networks:")
    for judgment in sample_judgments[:2]:
        citations = judgment['citations']
        judgment_cache.cache_citation_network(judgment['citation'], citations)
        print(f"   Cached {len(citations)} citations for {judgment['citation']}")

    # Cache search results
    print("\n   Caching search results:")
    search_queries = [
        ('constitutional law', {'court': 'UKSC', 'year': 2023}),
        ('administrative law', {'court': 'EWCA'}),
        ('judicial review', {})
    ]

    for query, filters in search_queries:
        query_hash = judgment_cache.generate_query_hash(query, filters)
        results = [j for j in sample_judgments if any(subject.lower() in query.lower()
                                                     for subject in j.get('subject_matter', []))]
        judgment_cache.cache_search_results(query_hash, results, {'query': query, 'filters': filters})
        print(f"   Cached {len(results)} results for query: '{query}'")

    # Test search cache retrieval
    print("\n   Search cache retrievals:")
    for query, filters in search_queries[:2]:
        query_hash = judgment_cache.generate_query_hash(query, filters)
        cached_results = judgment_cache.get_search_results(query_hash)
        if cached_results:
            print(f"   Query '{query}': {len(cached_results['results'])} cached results found")

    # 3. Cache Statistics and Analysis
    print("\n3. COMPREHENSIVE CACHE STATISTICS:")
    all_stats = judgment_cache.get_cache_statistics()
    for cache_name, stats in all_stats.items():
        print(f"\n   {cache_name.replace('_', ' ').title()}:")
        print(f"     - Size: {stats['size']}/{stats['capacity']}")
        print(f"     - Hit rate: {stats['hit_rate']:.1%}")
        print(f"     - Memory usage: {stats['total_size_bytes']:,} bytes")
        print(f"     - Avg entry size: {stats['avg_size_per_entry']:.0f} bytes")

    # 4. Preloading Important Judgments
    print("\n4. PRELOADING IMPORTANT JUDGMENTS:")
    important_cases = [
        '[2019] UKSC 41',  # Key constitutional case
        '[2017] UKSC 5',   # Important precedent
        '[2020] UKSC 8',   # Recent significant case
    ]

    judgment_cache.preload_important_judgments(important_cases)
    print(f"   Preloaded {len(important_cases)} important judgments")

    # Show access patterns
    print("\n   Most accessed entries:")
    judgment_entries = judgment_cache.judgment_cache.get_entries_by_access_pattern(5)
    for i, (key, access_count, last_accessed) in enumerate(judgment_entries, 1):
        print(f"   {i}. {key}: {access_count} accesses (last: {last_accessed.strftime('%H:%M:%S')})")

    # 5. TTL and Expiration Demo
    print("\n5. TTL AND EXPIRATION HANDLING:")

    # Create cache with short TTL for demo
    short_ttl_cache = LRUCache(capacity=5, default_ttl=2)  # 2 seconds TTL

    # Add entry
    short_ttl_cache.put('test_key', 'test_value')
    print("   Added entry with 2-second TTL")

    # Immediate retrieval
    result = short_ttl_cache.get('test_key')
    print(f"   Immediate retrieval: {'SUCCESS' if result else 'FAILED'}")

    # Wait and retry (simulated with manual expiration check)
    print("   Simulating expiration...")
    # In real usage, you would wait: time.sleep(3)

    # Manual cleanup demonstration
    expired_count = judgment_cache.cleanup_all_expired()
    print(f"   Cleaned up expired entries: {sum(expired_count.values())} total")

    return {
        'basic_cache': cache,
        'judgment_cache': judgment_cache,
        'statistics': all_stats
    }


if __name__ == "__main__":
    demonstrate_lru_cache()