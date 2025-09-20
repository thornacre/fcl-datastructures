# FCL Data Structures

Python implementations and tests of data structures critical for the Free Case Law (FCL) system - a national legal archive managing 70,000+ judgments scaling to millions.

## Why Data Structures Matter for FCL

FCL faces unique challenges that make proper data structure selection critical:

### 1. **Hierarchical Legal Documents**
- LegalDocML's nested XML structure (judgment → sections → paragraphs → citations)
- Requires efficient tree traversal algorithms
- Must extract metadata without loading 500-page documents into memory

### 2. **Graph Networks of Citations**
- Judgments form directed graphs through citations
- Finding "most influential" cases requires handling cycles
- Must traverse chains 10+ levels deep without stack overflow

### 3. **Search Index Optimization**
- Instant search across millions of words requires:
  - Inverted indices for full-text search
  - Tries for autocomplete
  - B-trees for date ranges
  - Hash tables for O(1) metadata lookups

### 4. **Memory-Efficient Processing**
- Streaming XML parsers for large documents
- Memory-mapped files to avoid server crashes
- Django querysets with `select_related()` and `prefetch_related()` to prevent N+1 queries

### 5. **Complex Query Requirements**
- Boolean searches with proximity operators
- Position-aware data structures
- Bitmap indices for faceted search
- Must intersect millions of documents in milliseconds

## Performance Impact

Without proper data structures:
- Search would take minutes, not milliseconds
- Citation networks would cause stack overflow
- Memory usage would crash servers at scale
- Editorial interfaces would freeze on large documents

**The difference between O(n²) and O(log n) algorithms determines whether FCL remains usable at millions of judgments or becomes impossibly slow.**

## Repository Contents

This repository contains:
- Python implementations of essential data structures
- Comprehensive test suites demonstrating usage
- Performance benchmarks for FCL use cases
- Real-world examples from legal document processing

## Data Structures Included

| Data Structure | File | FCL Use Case |
|----------------|------|--------------|
| Trees | [trees.py](data_structures/trees.py) | Document hierarchy |
| Graphs | [graphs.py](data_structures/graphs.py) | Citation networks |
| Hash Tables | [hash_tables.py](data_structures/hash_tables.py) | Metadata lookups |
| Tries | [tries.py](data_structures/tries.py) | Autocomplete |
| B-Trees | [b_trees.py](data_structures/b_trees.py) | Range queries |
| Inverted Indices | [inverted_indices.py](data_structures/inverted_indices.py) | Full-text search |
| pandas DataFrames | [dataframes.py](data_structures/dataframes.py) | Tabular legal data analysis |
| Sets with Intersection | [sets_intersection.py](data_structures/sets_intersection.py) | Document filtering |
| Regular Expressions | [regex_patterns.py](data_structures/regex_patterns.py) | Pattern matching in legal text |
| DOM Trees/BeautifulSoup | [dom_trees.py](data_structures/dom_trees.py) | HTML/XML parsing |
| Django ORM QuerySets | [django_querysets.py](data_structures/django_querysets.py) | Database operations |
| NumPy Arrays | [numpy_arrays.py](data_structures/numpy_arrays.py) | Numerical analysis |
| Bloom Filters | [bloom_filters.py](data_structures/bloom_filters.py) | Probabilistic membership testing |
| LRU Cache | [lru_cache.py](data_structures/lru_cache.py) | Caching frequent queries |
| Queue Structures | [queues.py](data_structures/queues.py) | Task processing |
| Segment Trees | [segment_trees.py](data_structures/segment_trees.py) | Range query optimization |
| Scoring Algorithms | [scoring_algorithms.py](data_structures/scoring_algorithms.py) | Relevance ranking |
| Red-Black Trees | [red_black_trees.py](data_structures/red_black_trees.py) | Balanced search trees |
| NLP Token Trees | [nlp_token_trees.py](data_structures/nlp_token_trees.py) | Natural language processing |

## Getting Started

```bash
# Clone the repository
git clone https://github.com/yourusername/fcl-datastructures.git
cd fcl-datastructures

# Install dependencies
pip install -r requirements.txt

# Run tests
pytest
```
