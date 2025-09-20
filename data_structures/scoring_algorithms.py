"""
Scoring Algorithms for Search Relevance Ranking
===============================================

This module implements various scoring algorithms for ranking search results
in Find Case Law (FCL). Provides sophisticated relevance scoring that considers
legal document characteristics and user context.

Key FCL Use Cases:
- Rank search results by relevance and authority
- Boost important legal precedents
- Consider court hierarchy in scoring
- Apply temporal relevance factors
- Personalize results based on user context
"""

from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import math
from datetime import datetime
from collections import defaultdict, Counter


@dataclass
class DocumentMetadata:
    """Metadata for legal documents used in scoring"""
    document_id: str
    citation: str
    court: str
    date: datetime
    judges: List[str]
    subject_areas: List[str]
    case_name: str
    citation_count: int = 0
    view_count: int = 0


@dataclass
class ScoredResult:
    """Represents a scored search result with detailed breakdown"""
    document_id: str
    final_score: float
    base_relevance: float
    authority_boost: float
    temporal_boost: float
    explanation: Dict[str, Any]


class TFIDFScorer:
    """Traditional TF-IDF scoring with legal document enhancements."""

    def __init__(self, documents: Dict[str, str], vocab: set):
        self.documents = documents
        self.vocab = vocab
        self.doc_freq = self._calculate_document_frequencies()
        self.total_docs = len(documents)

    def score(self, doc_id: str, query_terms: List[str]) -> float:
        """Calculate TF-IDF score for document given query terms."""
        if doc_id not in self.documents:
            return 0.0

        doc_text = self.documents[doc_id].lower()
        doc_tokens = doc_text.split()
        doc_length = len(doc_tokens)

        if doc_length == 0:
            return 0.0

        score = 0.0
        for term in query_terms:
            term = term.lower()
            if term in self.vocab:
                # Term frequency
                tf = doc_tokens.count(term) / doc_length

                # Inverse document frequency
                df = self.doc_freq.get(term, 0)
                idf = math.log(self.total_docs / (1 + df))

                # TF-IDF component
                score += tf * idf

        return score

    def _calculate_document_frequencies(self) -> Dict[str, int]:
        """Calculate document frequency for each term."""
        doc_freq = defaultdict(int)

        for doc_text in self.documents.values():
            doc_tokens = set(doc_text.lower().split())
            for token in doc_tokens:
                if token in self.vocab:
                    doc_freq[token] += 1

        return dict(doc_freq)


class BM25Scorer:
    """BM25 scoring algorithm optimized for legal documents."""

    def __init__(self, documents: Dict[str, str], vocab: set, k1: float = 1.5, b: float = 0.75):
        self.documents = documents
        self.vocab = vocab
        self.k1 = k1  # Term frequency saturation parameter
        self.b = b    # Length normalization parameter

        self.doc_freq = self._calculate_document_frequencies()
        self.doc_lengths = {doc_id: len(text.split()) for doc_id, text in documents.items()}
        self.avg_doc_length = sum(self.doc_lengths.values()) / len(documents) if documents else 0
        self.total_docs = len(documents)

    def score(self, doc_id: str, query_terms: List[str]) -> float:
        """Calculate BM25 score for document given query terms."""
        if doc_id not in self.documents:
            return 0.0

        doc_text = self.documents[doc_id].lower()
        doc_tokens = doc_text.split()
        doc_length = len(doc_tokens)

        if doc_length == 0:
            return 0.0

        score = 0.0
        term_freqs = Counter(doc_tokens)

        for term in query_terms:
            term = term.lower()
            if term in self.vocab:
                # Term frequency in document
                tf = term_freqs.get(term, 0)

                # Document frequency
                df = self.doc_freq.get(term, 0)

                # IDF component
                idf = math.log((self.total_docs - df + 0.5) / (df + 0.5))

                # BM25 formula
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * (doc_length / self.avg_doc_length))

                score += idf * (numerator / denominator)

        return score

    def _calculate_document_frequencies(self) -> Dict[str, int]:
        """Calculate document frequency for each term."""
        doc_freq = defaultdict(int)

        for doc_text in self.documents.values():
            doc_tokens = set(doc_text.lower().split())
            for token in doc_tokens:
                if token in self.vocab:
                    doc_freq[token] += 1

        return dict(doc_freq)


class LegalAuthorityScorer:
    """Scoring algorithm that considers legal authority and precedent value."""

    def __init__(self):
        # Court hierarchy weights
        self.court_weights = {
            'UKSC': 2.0,    # Supreme Court
            'UKHL': 1.9,    # House of Lords
            'UKPC': 1.8,    # Privy Council
            'EWCA': 1.6,    # Court of Appeal
            'CSIH': 1.6,    # Court of Session Inner House
            'EWHC': 1.4,    # High Court
            'CSOH': 1.4,    # Court of Session Outer House
            'UKUT': 1.2,    # Upper Tribunal
            'UKFTT': 1.0,   # First-tier Tribunal
        }

    def score(self, metadata: DocumentMetadata) -> float:
        """Calculate authority score based on court, judges, and citations."""
        score = 0.0

        # Court hierarchy boost
        court_boost = self.court_weights.get(metadata.court, 1.0)
        score += court_boost

        # Citation count boost (logarithmic to avoid extreme skew)
        citation_boost = 1.0 + 0.1 * math.log(1 + metadata.citation_count)
        score *= citation_boost

        return score


class TemporalScorer:
    """Scoring algorithm that considers temporal relevance factors."""

    def __init__(self, decay_factor: float = 0.1):
        self.decay_factor = decay_factor

    def score(self, metadata: DocumentMetadata) -> float:
        """Calculate temporal relevance score."""
        current_date = datetime.now()
        doc_date = metadata.date

        # Calculate age in years
        age_years = (current_date - doc_date).days / 365.25

        # Base temporal score (more recent = higher score)
        base_score = math.exp(-self.decay_factor * age_years)

        # Boost for landmark cases (these remain relevant longer)
        if metadata.citation_count > 50:  # Frequently cited cases
            base_score *= 1.5

        return base_score


class CompositeScorer:
    """Composite scoring algorithm that combines multiple scoring methods."""

    def __init__(self, documents: Dict[str, str], vocab: set):
        self.documents = documents
        self.vocab = vocab

        # Initialize component scorers
        self.tfidf_scorer = TFIDFScorer(documents, vocab)
        self.bm25_scorer = BM25Scorer(documents, vocab)
        self.authority_scorer = LegalAuthorityScorer()
        self.temporal_scorer = TemporalScorer()

        # Scoring weights
        self.weights = {
            'base_relevance': 0.5,      # TF-IDF or BM25
            'authority': 0.3,           # Court hierarchy, citations
            'temporal': 0.2,            # Recency and temporal relevance
        }

    def score_document(self, doc_id: str, query_terms: List[str],
                      metadata: DocumentMetadata) -> ScoredResult:
        """Calculate comprehensive score for a document."""
        # Base relevance score (BM25 preferred over TF-IDF)
        base_relevance = self.bm25_scorer.score(doc_id, query_terms)

        # Authority score
        authority_boost = self.authority_scorer.score(metadata)

        # Temporal relevance score
        temporal_boost = self.temporal_scorer.score(metadata)

        # Composite score calculation
        final_score = (
            self.weights['base_relevance'] * base_relevance +
            self.weights['authority'] * authority_boost +
            self.weights['temporal'] * temporal_boost
        )

        # Additional boosts for exact matches
        if query_terms:
            case_name_lower = metadata.case_name.lower()
            query_lower = ' '.join(query_terms).lower()
            if query_lower in case_name_lower:
                final_score *= 1.3  # Boost for case name matches

        # Prepare explanation
        explanation = {
            'base_relevance_contribution': self.weights['base_relevance'] * base_relevance,
            'authority_contribution': self.weights['authority'] * authority_boost,
            'temporal_contribution': self.weights['temporal'] * temporal_boost,
            'court': metadata.court,
            'date': metadata.date.isoformat(),
            'citation_count': metadata.citation_count,
        }

        return ScoredResult(
            document_id=doc_id,
            final_score=final_score,
            base_relevance=base_relevance,
            authority_boost=authority_boost,
            temporal_boost=temporal_boost,
            explanation=explanation
        )

    def rank_documents(self, candidate_docs: List[str], query_terms: List[str],
                      documents_metadata: Dict[str, DocumentMetadata],
                      max_results: int = 20) -> List[ScoredResult]:
        """Rank a list of candidate documents by relevance."""
        scored_results = []

        for doc_id in candidate_docs:
            if doc_id in documents_metadata:
                metadata = documents_metadata[doc_id]
                scored_result = self.score_document(doc_id, query_terms, metadata)
                scored_results.append(scored_result)

        # Sort by final score (descending)
        scored_results.sort(key=lambda x: x.final_score, reverse=True)

        return scored_results[:max_results]


def demonstrate_scoring_algorithms():
    """Demonstrate scoring algorithms with UK legal data."""

    print("=== Legal Document Scoring Algorithms Demo ===\n")

    # Sample legal documents
    documents = {
        'uksc_2023_15': 'This Supreme Court case concerns constitutional law and parliamentary sovereignty. The principles established in previous Supreme Court decisions are fundamental.',
        'ewca_2023_892': 'The Court of Appeal considered administrative law principles and judicial review procedures. The Wednesbury test applies to decisions of public bodies.',
        'ewhc_2023_1456': 'This High Court case examined statutory interpretation principles and parliamentary intent. Administrative law requires compliance with procedural fairness.'
    }

    # Sample metadata
    documents_metadata = {
        'uksc_2023_15': DocumentMetadata(
            document_id='uksc_2023_15',
            citation='[2023] UKSC 15',
            court='UKSC',
            date=datetime(2023, 5, 15),
            judges=['Lord Reed', 'Lady Hale', 'Lord Kerr'],
            subject_areas=['Constitutional Law', 'Parliamentary Sovereignty'],
            case_name='R (Miller) v Prime Minister',
            citation_count=25,
            view_count=1500
        ),
        'ewca_2023_892': DocumentMetadata(
            document_id='ewca_2023_892',
            citation='[2023] EWCA Civ 892',
            court='EWCA',
            date=datetime(2023, 8, 22),
            judges=['Sir Geoffrey Vos MR', 'Lord Justice Singh'],
            subject_areas=['Administrative Law', 'Judicial Review'],
            case_name='Smith v Secretary of State',
            citation_count=12,
            view_count=800
        ),
        'ewhc_2023_1456': DocumentMetadata(
            document_id='ewhc_2023_1456',
            citation='[2023] EWHC 1456 (Admin)',
            court='EWHC',
            date=datetime(2023, 6, 30),
            judges=['Mr Justice Swift'],
            subject_areas=['Administrative Law', 'Statutory Interpretation'],
            case_name='Jones v Local Authority',
            citation_count=5,
            view_count=300
        )
    }

    # Build vocabulary
    vocab = set()
    for text in documents.values():
        vocab.update(text.lower().split())

    # Initialize composite scorer
    scorer = CompositeScorer(documents, vocab)

    # 1. Basic TF-IDF Scoring
    print("1. TF-IDF SCORING:")
    query_terms = ['constitutional', 'law']
    tfidf_scorer = TFIDFScorer(documents, vocab)

    for doc_id in documents:
        score = tfidf_scorer.score(doc_id, query_terms)
        case_name = documents_metadata[doc_id].case_name
        print(f"   {case_name}: {score:.4f}")

    # 2. BM25 Scoring
    print(f"\n2. BM25 SCORING:")
    bm25_scorer = BM25Scorer(documents, vocab)

    for doc_id in documents:
        score = bm25_scorer.score(doc_id, query_terms)
        case_name = documents_metadata[doc_id].case_name
        print(f"   {case_name}: {score:.4f}")

    # 3. Authority Scoring
    print(f"\n3. AUTHORITY SCORING:")
    authority_scorer = LegalAuthorityScorer()

    for doc_id in documents:
        metadata = documents_metadata[doc_id]
        score = authority_scorer.score(metadata)
        print(f"   {metadata.case_name}: {score:.4f}")
        print(f"     Court: {metadata.court}, Citations: {metadata.citation_count}")

    # 4. Temporal Scoring
    print(f"\n4. TEMPORAL SCORING:")
    temporal_scorer = TemporalScorer()

    for doc_id in documents:
        metadata = documents_metadata[doc_id]
        score = temporal_scorer.score(metadata)
        print(f"   {metadata.case_name}: {score:.4f}")
        print(f"     Date: {metadata.date.strftime('%Y-%m-%d')}")

    # 5. Composite Scoring
    print(f"\n5. COMPOSITE SCORING:")
    query_terms = ['constitutional', 'administrative', 'law']

    results = scorer.rank_documents(
        list(documents.keys()),
        query_terms,
        documents_metadata
    )

    print(f"\n   Query: {' '.join(query_terms)}")
    print(f"   Results ranked by composite score:")

    for i, result in enumerate(results, 1):
        metadata = documents_metadata[result.document_id]
        print(f"\n   {i}. {metadata.case_name}")
        print(f"      Final Score: {result.final_score:.4f}")
        print(f"      Base Relevance: {result.base_relevance:.4f}")
        print(f"      Authority Boost: {result.authority_boost:.4f}")
        print(f"      Temporal Boost: {result.temporal_boost:.4f}")
        print(f"      Court: {metadata.court}")

    return {
        'scorer': scorer,
        'results': results,
        'documents_metadata': documents_metadata
    }


if __name__ == "__main__":
    demonstrate_scoring_algorithms()