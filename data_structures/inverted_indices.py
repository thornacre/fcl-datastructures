"""
Inverted Indices for Full-Text Search
=====================================

This module implements inverted indices for full-text search functionality
in Find Case Law (FCL). Provides efficient text search capabilities for
large collections of legal documents.

Key FCL Use Cases:
- Fast full-text search across judgment collections
- Boolean query processing (AND, OR, NOT)
- Phrase search and proximity queries
- Relevance scoring and ranking
- Auto-completion and suggestion features
"""

from typing import Dict, List, Set, Tuple, Optional, Any
from collections import defaultdict, Counter
from dataclasses import dataclass
import re
import math
from datetime import datetime


@dataclass
class PostingListEntry:
    """Represents a single entry in a posting list"""
    document_id: str
    term_frequency: int
    positions: List[int]
    normalized_frequency: float = 0.0


@dataclass
class SearchResult:
    """Represents a search result with scoring information"""
    document_id: str
    score: float
    matched_terms: List[str]
    snippet: str


class InvertedIndex:
    """
    High-performance inverted index implementation for legal document search.
    """

    def __init__(self):
        # Core index structures
        self.term_index: Dict[str, List[PostingListEntry]] = defaultdict(list)
        self.document_frequencies: Dict[str, int] = defaultdict(int)
        self.documents: Dict[str, Dict[str, Any]] = {}
        self.total_documents = 0

        # Search optimization
        self.stopwords = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to',
            'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be',
            'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
            'will', 'would', 'could', 'should', 'may', 'might', 'can'
        }

        # Legal-specific boosts
        self.court_weights = {
            'UKSC': 2.0,    # Supreme Court
            'UKHL': 1.8,    # House of Lords
            'EWCA': 1.6,    # Court of Appeal
            'EWHC': 1.4,    # High Court
            'UKUT': 1.2,    # Upper Tribunal
            'UKFTT': 1.0,   # First-tier Tribunal
        }

    def add_document(self, doc_id: str, content: str, metadata: Dict[str, Any] = None) -> None:
        """Add a document to the index."""
        if doc_id in self.documents:
            self.remove_document(doc_id)

        # Store document metadata
        self.documents[doc_id] = {
            'content': content,
            'metadata': metadata or {},
            'indexed_at': datetime.now().isoformat(),
            'length': len(content.split())
        }

        # Tokenize and index
        tokens = self._tokenize(content)
        term_positions = defaultdict(list)
        term_frequencies = Counter()

        # Build term positions and frequencies
        for position, token in enumerate(tokens):
            if token not in self.stopwords and len(token) > 2:
                term_positions[token].append(position)
                term_frequencies[token] += 1

        # Add to inverted index
        for term, frequency in term_frequencies.items():
            # Calculate normalized frequency (TF component)
            normalized_freq = frequency / len(tokens) if tokens else 0

            posting = PostingListEntry(
                document_id=doc_id,
                term_frequency=frequency,
                positions=term_positions[term],
                normalized_frequency=normalized_freq
            )

            self.term_index[term].append(posting)
            self.document_frequencies[term] += 1

        self.total_documents += 1

    def remove_document(self, doc_id: str) -> bool:
        """Remove a document from the index."""
        if doc_id not in self.documents:
            return False

        # Remove from posting lists
        terms_to_remove = []
        for term, postings in self.term_index.items():
            # Remove postings for this document
            original_length = len(postings)
            self.term_index[term] = [p for p in postings if p.document_id != doc_id]

            # Update document frequency
            if len(self.term_index[term]) < original_length:
                self.document_frequencies[term] -= 1

            # Mark empty terms for removal
            if not self.term_index[term]:
                terms_to_remove.append(term)

        # Remove empty terms
        for term in terms_to_remove:
            del self.term_index[term]
            del self.document_frequencies[term]

        # Remove document metadata
        del self.documents[doc_id]
        self.total_documents -= 1
        return True

    def search(self, query: str, max_results: int = 20,
              filters: Dict[str, Any] = None) -> List[SearchResult]:
        """Perform full-text search with TF-IDF scoring."""
        # Parse and tokenize query
        query_terms = self._tokenize(query.lower())
        query_terms = [term for term in query_terms if term not in self.stopwords and len(term) > 2]

        if not query_terms:
            return []

        # Get candidate documents
        candidates = self._get_candidate_documents(query_terms)

        # Apply filters if specified
        if filters:
            candidates = self._apply_filters(candidates, filters)

        if not candidates:
            return []

        # Calculate TF-IDF scores
        results = []
        for doc_id in candidates:
            score = self._calculate_tfidf_score(doc_id, query_terms)

            # Apply legal-specific boosts
            score = self._apply_legal_boosts(doc_id, score)

            # Generate snippet
            snippet = self._generate_snippet(doc_id, query_terms)

            result = SearchResult(
                document_id=doc_id,
                score=score,
                matched_terms=query_terms,
                snippet=snippet
            )
            results.append(result)

        # Sort by score and return top results
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:max_results]

    def boolean_search(self, query: str) -> Set[str]:
        """Perform boolean search with AND, OR, NOT operators."""
        # Simple boolean query parser
        if ' AND ' in query:
            terms = [term.strip().lower() for term in query.split(' AND ')]
            result_sets = [self._get_documents_for_term(term) for term in terms]
            return set.intersection(*result_sets) if result_sets else set()

        elif ' OR ' in query:
            terms = [term.strip().lower() for term in query.split(' OR ')]
            result_set = set()
            for term in terms:
                result_set.update(self._get_documents_for_term(term))
            return result_set

        elif ' NOT ' in query:
            parts = query.split(' NOT ')
            if len(parts) == 2:
                positive_docs = self._get_documents_for_term(parts[0].strip().lower())
                negative_docs = self._get_documents_for_term(parts[1].strip().lower())
                return positive_docs - negative_docs

        # Single term
        return self._get_documents_for_term(query.lower())

    def phrase_search(self, phrase: str) -> List[str]:
        """Search for exact phrases."""
        phrase_terms = self._tokenize(phrase.lower())
        phrase_terms = [term for term in phrase_terms if term not in self.stopwords]

        if len(phrase_terms) < 2:
            return list(self._get_documents_for_term(phrase_terms[0]) if phrase_terms else set())

        # Get documents containing all terms
        candidate_docs = self._get_documents_for_term(phrase_terms[0])
        for term in phrase_terms[1:]:
            term_docs = self._get_documents_for_term(term)
            candidate_docs &= term_docs

        # Check phrase positions in candidates
        matching_docs = []
        for doc_id in candidate_docs:
            if self._contains_phrase(doc_id, phrase_terms):
                matching_docs.append(doc_id)

        return matching_docs

    def get_suggestions(self, partial_term: str, max_suggestions: int = 10) -> List[str]:
        """Get term suggestions for auto-completion."""
        partial_lower = partial_term.lower()
        suggestions = []

        for term in self.term_index.keys():
            if term.startswith(partial_lower):
                # Score by frequency
                doc_freq = self.document_frequencies[term]
                suggestions.append((term, doc_freq))

        # Sort by frequency and return top suggestions
        suggestions.sort(key=lambda x: x[1], reverse=True)
        return [term for term, _ in suggestions[:max_suggestions]]

    def get_term_statistics(self, term: str) -> Dict[str, Any]:
        """Get detailed statistics for a specific term."""
        if term not in self.term_index:
            return {}

        postings = self.term_index[term]
        total_freq = sum(p.term_frequency for p in postings)
        doc_freq = len(postings)

        return {
            'term': term,
            'document_frequency': doc_freq,
            'total_frequency': total_freq,
            'idf': math.log(self.total_documents / doc_freq) if doc_freq > 0 else 0,
            'average_tf': total_freq / doc_freq if doc_freq > 0 else 0,
            'documents': [p.document_id for p in postings]
        }

    def get_index_statistics(self) -> Dict[str, Any]:
        """Get comprehensive index statistics."""
        return {
            'documents': self.total_documents,
            'unique_terms': len(self.term_index),
            'average_doc_length': sum(doc['length'] for doc in self.documents.values()) / self.total_documents if self.total_documents > 0 else 0,
            'most_frequent_terms': self._get_most_frequent_terms(10)
        }

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into terms."""
        # Basic tokenization for legal text
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        tokens = text.split()
        return [token for token in tokens if len(token) > 0]

    def _get_candidate_documents(self, terms: List[str]) -> Set[str]:
        """Get documents containing any of the query terms."""
        candidates = set()
        for term in terms:
            if term in self.term_index:
                for posting in self.term_index[term]:
                    candidates.add(posting.document_id)
        return candidates

    def _apply_filters(self, candidates: Set[str], filters: Dict[str, Any]) -> Set[str]:
        """Apply metadata filters to candidate documents."""
        filtered = set()
        for doc_id in candidates:
            doc_metadata = self.documents[doc_id]['metadata']

            # Apply court filter
            if 'court' in filters and doc_metadata.get('court') != filters['court']:
                continue

            # Apply date range filter
            if 'date_from' in filters or 'date_to' in filters:
                doc_date = doc_metadata.get('date')
                if not doc_date:
                    continue

                if 'date_from' in filters and doc_date < filters['date_from']:
                    continue
                if 'date_to' in filters and doc_date > filters['date_to']:
                    continue

            filtered.add(doc_id)

        return filtered

    def _calculate_tfidf_score(self, doc_id: str, query_terms: List[str]) -> float:
        """Calculate TF-IDF score for a document given query terms."""
        score = 0.0
        doc_length = self.documents[doc_id]['length']

        for term in query_terms:
            if term in self.term_index:
                # Find posting for this document
                posting = None
                for p in self.term_index[term]:
                    if p.document_id == doc_id:
                        posting = p
                        break

                if posting:
                    # TF component (normalized)
                    tf = posting.term_frequency / doc_length if doc_length > 0 else 0

                    # IDF component
                    df = self.document_frequencies[term]
                    idf = math.log(self.total_documents / df) if df > 0 else 0

                    # TF-IDF score
                    score += tf * idf

        return score

    def _apply_legal_boosts(self, doc_id: str, base_score: float) -> float:
        """Apply legal-specific scoring boosts."""
        metadata = self.documents[doc_id]['metadata']
        boost = 1.0

        # Court hierarchy boost
        court = metadata.get('court', '')
        if court in self.court_weights:
            boost *= self.court_weights[court]

        return base_score * boost

    def _generate_snippet(self, doc_id: str, query_terms: List[str],
                         snippet_length: int = 200) -> str:
        """Generate search snippet."""
        content = self.documents[doc_id]['content']

        # Simple snippet generation - find first occurrence of query term
        for term in query_terms:
            start = content.lower().find(term)
            if start != -1:
                snippet_start = max(0, start - 50)
                snippet_end = min(len(content), start + snippet_length)
                return content[snippet_start:snippet_end].strip()

        # Fallback to beginning of document
        return content[:snippet_length].strip()

    def _get_documents_for_term(self, term: str) -> Set[str]:
        """Get set of document IDs containing a term."""
        if term in self.term_index:
            return {posting.document_id for posting in self.term_index[term]}
        return set()

    def _contains_phrase(self, doc_id: str, phrase_terms: List[str]) -> bool:
        """Check if document contains phrase in order."""
        # Get positions for all terms
        term_positions = {}
        for term in phrase_terms:
            if term in self.term_index:
                for posting in self.term_index[term]:
                    if posting.document_id == doc_id:
                        term_positions[term] = posting.positions
                        break

        # Check if all terms found
        if len(term_positions) != len(phrase_terms):
            return False

        # Check if terms appear in sequence
        first_term_positions = term_positions[phrase_terms[0]]

        for start_pos in first_term_positions:
            valid_phrase = True
            current_pos = start_pos

            for i, term in enumerate(phrase_terms[1:], 1):
                term_pos_list = term_positions[term]
                expected_pos = current_pos + 1

                if expected_pos in term_pos_list:
                    current_pos = expected_pos
                else:
                    valid_phrase = False
                    break

            if valid_phrase:
                return True

        return False

    def _get_most_frequent_terms(self, n: int) -> List[Tuple[str, int]]:
        """Get the n most frequent terms in the index."""
        term_freqs = [(term, self.document_frequencies[term])
                     for term in self.term_index.keys()]
        term_freqs.sort(key=lambda x: x[1], reverse=True)
        return term_freqs[:n]


def demonstrate_inverted_index():
    """Demonstrate inverted index with UK legal documents."""

    print("=== Inverted Index for Legal Documents Demo ===\n")

    # Sample UK legal documents
    documents = {
        'uksc_2023_15': {
            'content': 'This Supreme Court case concerns constitutional law and parliamentary sovereignty. The principles established in previous Supreme Court decisions are fundamental to our constitutional framework.',
            'metadata': {
                'citation': '[2023] UKSC 15',
                'court': 'UKSC',
                'date': '2023-05-15',
                'case_name': 'R (Miller) v Prime Minister'
            }
        },
        'ewca_2023_892': {
            'content': 'The Court of Appeal considered administrative law principles and judicial review procedures. The Wednesbury test applies to decisions of public bodies.',
            'metadata': {
                'citation': '[2023] EWCA Civ 892',
                'court': 'EWCA',
                'date': '2023-08-22',
                'case_name': 'Smith v Secretary of State'
            }
        },
        'ewhc_2023_1456': {
            'content': 'This High Court case examined statutory interpretation principles and parliamentary intent. Administrative law requires compliance with procedural fairness.',
            'metadata': {
                'citation': '[2023] EWHC 1456 (Admin)',
                'court': 'EWHC',
                'date': '2023-06-30',
                'case_name': 'Jones v Local Authority'
            }
        }
    }

    # 1. Build Index
    print("1. BUILDING INVERTED INDEX:")
    index = InvertedIndex()

    for doc_id, doc_data in documents.items():
        index.add_document(doc_id, doc_data['content'], doc_data['metadata'])
        print(f"   Indexed: {doc_data['metadata']['case_name']}")

    # 2. Basic Search
    print(f"\n2. BASIC SEARCH QUERIES:")
    queries = ["constitutional law", "administrative law", "Supreme Court"]

    for query in queries:
        results = index.search(query, max_results=3)
        print(f"\n   Query: '{query}' ({len(results)} results)")
        for i, result in enumerate(results, 1):
            doc_meta = documents[result.document_id]['metadata']
            print(f"     {i}. {doc_meta['case_name']} (Score: {result.score:.3f})")
            print(f"        Snippet: {result.snippet[:60]}...")

    # 3. Boolean Search
    print(f"\n3. BOOLEAN SEARCH:")
    boolean_queries = ["constitutional AND law", "administrative OR Supreme"]

    for query in boolean_queries:
        results = index.boolean_search(query)
        print(f"\n   Boolean Query: '{query}' ({len(results)} documents)")
        for doc_id in list(results)[:2]:
            case_name = documents[doc_id]['metadata']['case_name']
            print(f"     - {case_name}")

    # 4. Phrase Search
    print(f"\n4. PHRASE SEARCH:")
    phrase_results = index.phrase_search("constitutional law")
    print(f"   Phrase: 'constitutional law' ({len(phrase_results)} documents)")
    for doc_id in phrase_results:
        case_name = documents[doc_id]['metadata']['case_name']
        print(f"     - {case_name}")

    # 5. Auto-completion
    print(f"\n5. AUTO-COMPLETION:")
    suggestions = index.get_suggestions("const", 3)
    print(f"   'const*': {', '.join(suggestions)}")

    # 6. Term Statistics
    print(f"\n6. TERM STATISTICS:")
    stats = index.get_term_statistics("law")
    if stats:
        print(f"   'law':")
        print(f"     Document frequency: {stats['document_frequency']}")
        print(f"     IDF score: {stats['idf']:.3f}")

    # 7. Index Statistics
    print(f"\n7. INDEX STATISTICS:")
    index_stats = index.get_index_statistics()
    print(f"   Total documents: {index_stats['documents']}")
    print(f"   Unique terms: {index_stats['unique_terms']}")
    print(f"   Most frequent terms: {index_stats['most_frequent_terms'][:3]}")

    return {'index': index, 'documents': documents, 'statistics': index_stats}


if __name__ == "__main__":
    demonstrate_inverted_index()