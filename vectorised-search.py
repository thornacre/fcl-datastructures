"""
Vectorised Search Query Processor

A Python class for preparing search queries for vectorized search in NoSQL databases.
Handles text preprocessing, tokenization, and vector embedding preparation.

Author: FCL
Version: 1.0.0
"""

import re
import hashlib
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field


@dataclass
class SearchConfig:
    """Configuration for search query processing."""
    max_tokens: int = 512
    min_token_length: int = 2
    lowercase: bool = True
    remove_stopwords: bool = True
    stem_tokens: bool = False
    vector_dimensions: int = 384
    similarity_metric: str = "cosine"  # cosine, euclidean, dot_product
    # N-gram settings
    ngram_range: tuple = (1, 1)  # (min_n, max_n) - (1,1) for unigrams, (1,2) for uni+bigrams
    # Hybrid search settings
    hybrid_alpha: float = 0.7  # Weight for vector search (1-alpha for keyword search)


class VectorisedSearchQuery:
    """
    Prepares search queries for vectorised search in NoSQL databases.

    Handles text preprocessing, tokenization, and formats queries
    for vector similarity search operations.

    Example:
        >>> processor = VectorisedSearchQuery()
        >>> query = processor.prepare("Find documents about machine learning")
        >>> print(query.tokens)
        ['find', 'documents', 'machine', 'learning']
    """

    DEFAULT_STOPWORDS = frozenset([
        'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were', 'be', 'been',
        'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
        'could', 'should', 'may', 'might', 'must', 'shall', 'can', 'need',
        'it', 'its', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she',
        'we', 'they', 'what', 'which', 'who', 'whom', 'when', 'where', 'why',
        'how', 'all', 'each', 'every', 'both', 'few', 'more', 'most', 'other',
        'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than',
        'too', 'very', 'just', 'about', 'into', 'through', 'during', 'before',
        'after', 'above', 'below', 'between', 'under', 'again', 'further', 'then'
    ])

    # Common English suffixes for stemming
    SUFFIX_RULES = [
        ('ational', 'ate'), ('tional', 'tion'), ('enci', 'ence'), ('anci', 'ance'),
        ('izer', 'ize'), ('isation', 'ize'), ('ization', 'ize'), ('ation', 'ate'),
        ('ator', 'ate'), ('alism', 'al'), ('iveness', 'ive'), ('fulness', 'ful'),
        ('ousness', 'ous'), ('aliti', 'al'), ('iviti', 'ive'), ('biliti', 'ble'),
        ('ling', 'l'), ('ement', ''), ('ment', ''), ('ent', ''), ('ness', ''),
        ('ible', ''), ('able', ''), ('ing', ''), ('ies', 'i'), ('es', ''),
        ('ed', ''), ('ly', ''), ('s', '')
    ]

    def __init__(
        self,
        config: Optional[SearchConfig] = None,
        embedding_fn: Optional[Callable[[str], List[float]]] = None,
        custom_stopwords: Optional[set] = None
    ):
        """
        Initialize the search query processor.

        Args:
            config: Search configuration options
            embedding_fn: Custom function to generate embeddings from text
            custom_stopwords: Additional stopwords to filter out
        """
        self.config = config or SearchConfig()
        self.embedding_fn = embedding_fn
        self.stopwords = self.DEFAULT_STOPWORDS
        if custom_stopwords:
            self.stopwords = self.stopwords | custom_stopwords

    def preprocess(self, text: str) -> str:
        """
        Clean and normalize input text.

        Args:
            text: Raw input text

        Returns:
            Cleaned and normalized text
        """
        # Remove URLs
        text = re.sub(r'https?://\S+|www\.\S+', '', text)

        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)

        # Remove special characters but keep alphanumeric and spaces
        text = re.sub(r'[^\w\s]', ' ', text)

        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        # Convert to lowercase if configured
        if self.config.lowercase:
            text = text.lower()

        return text

    def tokenize(self, text: str) -> List[str]:
        """
        Split text into tokens with filtering, stemming, and n-gram generation.

        Args:
            text: Preprocessed text

        Returns:
            List of filtered tokens (including n-grams if configured)
        """
        tokens = text.split()

        # Filter by minimum length
        tokens = [t for t in tokens if len(t) >= self.config.min_token_length]

        # Remove stopwords if configured
        if self.config.remove_stopwords:
            tokens = [t for t in tokens if t.lower() not in self.stopwords]

        # Apply stemming if configured
        if self.config.stem_tokens:
            tokens = [self._stem(t) for t in tokens]

        # Truncate to max tokens before n-gram generation
        tokens = tokens[:self.config.max_tokens]

        # Generate n-grams if configured
        min_n, max_n = self.config.ngram_range
        if max_n > 1:
            tokens = self._generate_ngrams(tokens, min_n, max_n)

        return tokens

    def _stem(self, word: str) -> str:
        """
        Apply simple suffix-stripping stemming to a word.

        Args:
            word: Word to stem

        Returns:
            Stemmed word
        """
        if len(word) <= 3:
            return word

        for suffix, replacement in self.SUFFIX_RULES:
            if word.endswith(suffix):
                stemmed = word[:-len(suffix)] + replacement
                # Ensure we don't over-stem
                if len(stemmed) >= 2:
                    return stemmed
        return word

    def _generate_ngrams(self, tokens: List[str], min_n: int, max_n: int) -> List[str]:
        """
        Generate n-grams from a list of tokens.

        Args:
            tokens: List of tokens
            min_n: Minimum n-gram size
            max_n: Maximum n-gram size

        Returns:
            List containing original tokens and n-grams
        """
        result = []

        for n in range(min_n, max_n + 1):
            if n == 1:
                result.extend(tokens)
            else:
                for i in range(len(tokens) - n + 1):
                    ngram = '_'.join(tokens[i:i + n])
                    result.append(ngram)

        return result

    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate vector embedding for the text.

        Uses custom embedding function if provided, otherwise generates
        a deterministic hash-based embedding (for demonstration).

        Args:
            text: Text to embed

        Returns:
            Vector embedding as list of floats
        """
        if self.embedding_fn:
            return self.embedding_fn(text)

        # Default: deterministic hash-based embedding (placeholder)
        # In production, replace with actual embedding model (e.g., sentence-transformers)
        return self._hash_embedding(text)

    def _hash_embedding(self, text: str) -> List[float]:
        """
        Generate a deterministic hash-based embedding.

        Note: This is a placeholder. Use a proper embedding model in production.

        Args:
            text: Text to embed

        Returns:
            Normalized vector of configured dimensions
        """
        dimensions = self.config.vector_dimensions
        embedding = []

        for i in range(dimensions):
            hash_input = f"{text}_{i}".encode('utf-8')
            hash_val = int(hashlib.sha256(hash_input).hexdigest(), 16)
            # Normalize to [-1, 1]
            normalized = (hash_val % 10000) / 5000 - 1
            embedding.append(normalized)

        # L2 normalize the vector
        magnitude = sum(x * x for x in embedding) ** 0.5
        if magnitude > 0:
            embedding = [x / magnitude for x in embedding]

        return embedding

    def prepare(self, query: str) -> 'PreparedQuery':
        """
        Prepare a search query for vectorised search.

        Args:
            query: Raw search query string

        Returns:
            PreparedQuery object ready for NoSQL database search
        """
        preprocessed = self.preprocess(query)
        tokens = self.tokenize(preprocessed)
        token_text = ' '.join(tokens)
        embedding = self.generate_embedding(token_text)

        return PreparedQuery(
            original=query,
            preprocessed=preprocessed,
            tokens=tokens,
            embedding=embedding,
            config=self.config
        )

    def to_nosql_query(
        self,
        prepared: 'PreparedQuery',
        collection: str,
        vector_field: str = "embedding",
        limit: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Format prepared query for NoSQL database vector search.

        Generates a query structure compatible with common NoSQL databases
        like MongoDB Atlas, Elasticsearch, or Pinecone.

        Args:
            prepared: PreparedQuery object
            collection: Target collection/index name
            vector_field: Name of the vector field in documents
            limit: Maximum number of results to return
            filters: Additional filter conditions

        Returns:
            Dictionary representing the NoSQL query
        """
        query = {
            "collection": collection,
            "vector_search": {
                "field": vector_field,
                "vector": prepared.embedding,
                "similarity": self.config.similarity_metric,
                "limit": limit
            },
            "metadata": {
                "original_query": prepared.original,
                "tokens": prepared.tokens,
                "token_count": len(prepared.tokens)
            }
        }

        if filters:
            query["filters"] = filters

        return query

    def to_mongodb_query(
        self,
        prepared: 'PreparedQuery',
        index_name: str,
        vector_field: str = "embedding",
        limit: int = 10,
        num_candidates: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Generate MongoDB Atlas Vector Search aggregation pipeline.

        Args:
            prepared: PreparedQuery object
            index_name: Name of the vector search index
            vector_field: Path to the vector field
            limit: Number of results to return
            num_candidates: Number of candidates to consider

        Returns:
            MongoDB aggregation pipeline
        """
        return [
            {
                "$vectorSearch": {
                    "index": index_name,
                    "path": vector_field,
                    "queryVector": prepared.embedding,
                    "numCandidates": num_candidates,
                    "limit": limit
                }
            },
            {
                "$project": {
                    "_id": 1,
                    "score": {"$meta": "vectorSearchScore"}
                }
            }
        ]

    def to_elasticsearch_query(
        self,
        prepared: 'PreparedQuery',
        vector_field: str = "embedding",
        limit: int = 10
    ) -> Dict[str, Any]:
        """
        Generate Elasticsearch kNN query.

        Args:
            prepared: PreparedQuery object
            vector_field: Name of the dense vector field
            limit: Number of results to return

        Returns:
            Elasticsearch query dictionary
        """
        return {
            "knn": {
                "field": vector_field,
                "query_vector": prepared.embedding,
                "k": limit,
                "num_candidates": limit * 10
            },
            "_source": ["_id", "_score"]
        }

    def to_hybrid_mongodb_query(
        self,
        prepared: 'PreparedQuery',
        index_name: str,
        text_field: str = "content",
        vector_field: str = "embedding",
        limit: int = 10,
        num_candidates: int = 100
    ) -> Dict[str, Any]:
        """
        Generate MongoDB hybrid search combining vector and text search.

        Uses $vectorSearch with a text search filter or returns both
        pipelines for client-side score fusion.

        Args:
            prepared: PreparedQuery object
            index_name: Name of the vector search index
            text_field: Field to search for keywords
            vector_field: Path to the vector field
            limit: Number of results to return
            num_candidates: Number of candidates for vector search

        Returns:
            Dictionary with vector and text search pipelines
        """
        alpha = self.config.hybrid_alpha

        # Vector search pipeline
        vector_pipeline = [
            {
                "$vectorSearch": {
                    "index": index_name,
                    "path": vector_field,
                    "queryVector": prepared.embedding,
                    "numCandidates": num_candidates,
                    "limit": limit * 2  # Get more for fusion
                }
            },
            {
                "$addFields": {
                    "vector_score": {"$meta": "vectorSearchScore"},
                    "search_type": "vector"
                }
            }
        ]

        # Text search pipeline (requires text index)
        keyword_query = ' '.join(prepared.tokens)
        text_pipeline = [
            {
                "$match": {
                    "$text": {"$search": keyword_query}
                }
            },
            {
                "$addFields": {
                    "text_score": {"$meta": "textScore"},
                    "search_type": "text"
                }
            },
            {"$limit": limit * 2}
        ]

        return {
            "vector_pipeline": vector_pipeline,
            "text_pipeline": text_pipeline,
            "fusion": {
                "method": "weighted_sum",
                "vector_weight": alpha,
                "text_weight": 1 - alpha
            },
            "tokens": prepared.tokens,
            "limit": limit
        }

    def to_hybrid_elasticsearch_query(
        self,
        prepared: 'PreparedQuery',
        text_fields: List[str] = None,
        vector_field: str = "embedding",
        limit: int = 10
    ) -> Dict[str, Any]:
        """
        Generate Elasticsearch hybrid query combining kNN and BM25.

        Args:
            prepared: PreparedQuery object
            text_fields: Fields to search with BM25 (default: ["content", "title"])
            vector_field: Name of the dense vector field
            limit: Number of results to return

        Returns:
            Elasticsearch hybrid query dictionary
        """
        if text_fields is None:
            text_fields = ["content", "title"]

        alpha = self.config.hybrid_alpha
        keyword_query = ' '.join(prepared.tokens)

        return {
            "size": limit,
            "query": {
                "bool": {
                    "should": [
                        # BM25 text search
                        {
                            "multi_match": {
                                "query": keyword_query,
                                "fields": text_fields,
                                "type": "best_fields",
                                "boost": 1 - alpha
                            }
                        }
                    ]
                }
            },
            "knn": {
                "field": vector_field,
                "query_vector": prepared.embedding,
                "k": limit,
                "num_candidates": limit * 10,
                "boost": alpha
            },
            "_source": True
        }

    def to_hybrid_nosql_query(
        self,
        prepared: 'PreparedQuery',
        collection: str,
        text_field: str = "content",
        vector_field: str = "embedding",
        limit: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate a generic hybrid search query for NoSQL databases.

        Combines vector similarity search with keyword matching.

        Args:
            prepared: PreparedQuery object
            collection: Target collection/index name
            text_field: Field for keyword search
            vector_field: Name of the vector field
            limit: Maximum number of results
            filters: Additional filter conditions

        Returns:
            Dictionary representing the hybrid NoSQL query
        """
        alpha = self.config.hybrid_alpha

        query = {
            "collection": collection,
            "hybrid_search": {
                "vector": {
                    "field": vector_field,
                    "query_vector": prepared.embedding,
                    "similarity": self.config.similarity_metric,
                    "weight": alpha
                },
                "keyword": {
                    "field": text_field,
                    "tokens": prepared.tokens,
                    "match_type": "any",  # any or all
                    "weight": 1 - alpha
                },
                "fusion_method": "weighted_sum",
                "limit": limit
            },
            "metadata": {
                "original_query": prepared.original,
                "tokens": prepared.tokens,
                "ngrams": [t for t in prepared.tokens if '_' in t]
            }
        }

        if filters:
            query["filters"] = filters

        return query


@dataclass
class PreparedQuery:
    """
    Represents a prepared search query ready for vector search.

    Attributes:
        original: The original query string
        preprocessed: Cleaned and normalized text
        tokens: List of processed tokens
        embedding: Vector embedding of the query
        config: Configuration used for processing
    """
    original: str
    preprocessed: str
    tokens: List[str]
    embedding: List[float]
    config: SearchConfig = field(default_factory=SearchConfig)

    @property
    def token_count(self) -> int:
        """Return the number of tokens."""
        return len(self.tokens)

    @property
    def vector_dimensions(self) -> int:
        """Return the embedding dimensions."""
        return len(self.embedding)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "original": self.original,
            "preprocessed": self.preprocessed,
            "tokens": self.tokens,
            "embedding": self.embedding,
            "token_count": self.token_count,
            "vector_dimensions": self.vector_dimensions
        }


# Example usage
if __name__ == "__main__":
    # Example 1: Basic usage with default config
    print("=" * 60)
    print("Example 1: Basic Usage")
    print("=" * 60)
    processor = VectorisedSearchQuery()
    query = "What are the best practices for machine learning in production?"
    prepared = processor.prepare(query)

    print(f"Original: {prepared.original}")
    print(f"Preprocessed: {prepared.preprocessed}")
    print(f"Tokens: {prepared.tokens}")
    print(f"Token count: {prepared.token_count}")

    # Example 2: With stemming enabled
    print("\n" + "=" * 60)
    print("Example 2: With Stemming")
    print("=" * 60)
    config_stemmed = SearchConfig(stem_tokens=True)
    processor_stemmed = VectorisedSearchQuery(config=config_stemmed)
    prepared_stemmed = processor_stemmed.prepare(query)
    print(f"Stemmed tokens: {prepared_stemmed.tokens}")

    # Example 3: With n-grams (bigrams)
    print("\n" + "=" * 60)
    print("Example 3: With Bigrams")
    print("=" * 60)
    config_ngram = SearchConfig(ngram_range=(1, 2))
    processor_ngram = VectorisedSearchQuery(config=config_ngram)
    prepared_ngram = processor_ngram.prepare(query)
    print(f"Tokens with bigrams: {prepared_ngram.tokens}")

    # Example 4: Stemming + Trigrams
    print("\n" + "=" * 60)
    print("Example 4: Stemming + Trigrams")
    print("=" * 60)
    config_full = SearchConfig(stem_tokens=True, ngram_range=(1, 3))
    processor_full = VectorisedSearchQuery(config=config_full)
    prepared_full = processor_full.prepare(query)
    print(f"Stemmed with trigrams: {prepared_full.tokens}")

    # Example 5: Hybrid search query
    print("\n" + "=" * 60)
    print("Example 5: Hybrid Search (MongoDB)")
    print("=" * 60)
    hybrid_query = processor.to_hybrid_mongodb_query(
        prepared,
        index_name="vector_index"
    )
    print(f"Fusion method: {hybrid_query['fusion']}")
    print(f"Vector weight: {hybrid_query['fusion']['vector_weight']}")
    print(f"Text weight: {hybrid_query['fusion']['text_weight']}")

    # Example 6: Hybrid Elasticsearch query
    print("\n" + "=" * 60)
    print("Example 6: Hybrid Search (Elasticsearch)")
    print("=" * 60)
    es_hybrid = processor.to_hybrid_elasticsearch_query(
        prepared,
        text_fields=["content", "title", "summary"]
    )
    print(f"Elasticsearch hybrid query keys: {list(es_hybrid.keys())}")
    print(f"kNN boost: {es_hybrid['knn']['boost']}")
