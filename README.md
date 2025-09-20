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

## Data Structures

| Data Structure | Python | JavaScript | Description | FCL Use Case |
|----------------|--------|------------|-------------|--------------|
| Trees | [trees.py](data_structures/trees.py) | [trees.js](data_structures/trees.js) | Hierarchical data organization | Document hierarchy |
| Graphs | [graphs.py](data_structures/graphs.py) | [graphs.js](data_structures/graphs.js) | Network/relationship storage | Citation networks |
| Hash Tables | [hash_tables.py](data_structures/hash_tables.py) | [hash_tables.js](data_structures/hash_tables.js) | Key-value storage with O(1) access | Metadata lookups |
| Tries | [tries.py](data_structures/tries.py) | [tries.js](data_structures/tries.js) | Prefix tree for string storage | Autocomplete |
| B-Trees | [b_trees.py](data_structures/b_trees.py) | [b_trees.js](data_structures/b_trees.js) | Self-balancing tree for sorted data | Range queries |
| pandas DataFrames | [dataframes.py](data_structures/dataframes.py) | [dataframes.js](data_structures/dataframes.js) | Tabular data structure | Tabular legal data analysis |
| Sets | [sets_intersection.py](data_structures/sets_intersection.py) | [sets_intersection.js](data_structures/sets_intersection.js) | Unique element collection | Document filtering |
| DOM Trees | [dom_trees.py](data_structures/dom_trees.py) | [dom_trees.js](data_structures/dom_trees.js) | Document object model structure | HTML/XML parsing |
| NumPy Arrays | [numpy_arrays.py](data_structures/numpy_arrays.py) | [numpy_arrays.js](data_structures/numpy_arrays.js) | Multi-dimensional array structure | Numerical analysis |
| LRU Cache | [lru_cache.py](data_structures/lru_cache.py) | [lru_cache.js](data_structures/lru_cache.js) | Limited-size cache with eviction policy | Caching frequent queries |
| Queue Structures | [queues.py](data_structures/queues.py) | [queues.js](data_structures/queues.js) | FIFO data structure | Task processing |
| Red-Black Trees | [red_black_trees.py](data_structures/red_black_trees.py) | [red_black_trees.js](data_structures/red_black_trees.js) | Self-balancing binary search tree | Balanced search trees |
| NLP Token Trees | [nlp_token_trees.py](data_structures/nlp_token_trees.py) | [nlp_token_trees.js](data_structures/nlp_token_trees.js) | Parsed language structure | Natural language processing |

## Algorithms for Data Manipulation/Traversal

| Algorithm | Python | JavaScript | Description | FCL Use Case |
|-----------|--------|------------|-------------|--------------|
| Regular Expressions | [regex_patterns.py](data_structures/regex_patterns.py) | [regex_patterns.js](data_structures/regex_patterns.js) | Pattern matching and extraction algorithm | Pattern matching in legal text |
| Scoring Algorithms | [scoring_algorithms.py](data_structures/scoring_algorithms.py) | [scoring_algorithms.js](data_structures/scoring_algorithms.js) | Relevance calculation and ranking | Relevance ranking |
| Set Intersection Operations | [sets_intersection.py](data_structures/sets_intersection.py) | [sets_intersection.js](data_structures/sets_intersection.js) | Algorithm for finding common elements | Document filtering |
| BeautifulSoup (parsing/traversal) | [dom_trees.py](data_structures/dom_trees.py) | [dom_trees.js](data_structures/dom_trees.js) | HTML/XML traversal algorithms | HTML/XML parsing |

## Hybrid (Structure + Built-in Algorithms)

| Hybrid | Python | JavaScript | Description | FCL Use Case |
|--------|--------|------------|-------------|--------------|
| Inverted Indices | [inverted_indices.py](data_structures/inverted_indices.py) | [inverted_indices.js](data_structures/inverted_indices.js) | Include both structure and search algorithms | Full-text search |
| Bloom Filters | [bloom_filters.py](data_structures/bloom_filters.py) | [bloom_filters.js](data_structures/bloom_filters.js) | Include hash algorithms for membership testing | Probabilistic membership testing |
| Django ORM QuerySets | [django_querysets.py](data_structures/django_querysets.py) | [django_querysets.js](data_structures/django_querysets.js) | Include SQL generation algorithms | Database operations |
| Segment Trees | [segment_trees.py](data_structures/segment_trees.py) | [segment_trees.js](data_structures/segment_trees.js) | Include range query algorithms | Range query optimization |

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
