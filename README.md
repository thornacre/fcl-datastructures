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

| Data Structure | File | Description | FCL Use Case |
|----------------|------|-------------|--------------|
| Trees | [trees.py](data_structures/trees.py) | Hierarchical data organization | Document hierarchy |
| Graphs | [graphs.py](data_structures/graphs.py) | Network/relationship storage | Citation networks |
| Hash Tables | [hash_tables.py](data_structures/hash_tables.py) | Key-value storage with O(1) access | Metadata lookups |
| Tries | [tries.py](data_structures/tries.py) | Prefix tree for string storage | Autocomplete |
| B-Trees | [b_trees.py](data_structures/b_trees.py) | Self-balancing tree for sorted data | Range queries |
| pandas DataFrames | [dataframes.py](data_structures/dataframes.py) | Tabular data structure | Tabular legal data analysis |
| Sets | [sets_intersection.py](data_structures/sets_intersection.py) | Unique element collection | Document filtering |
| DOM Trees | [dom_trees.py](data_structures/dom_trees.py) | Document object model structure | HTML/XML parsing |
| NumPy Arrays | [numpy_arrays.py](data_structures/numpy_arrays.py) | Multi-dimensional array structure | Numerical analysis |
| LRU Cache | [lru_cache.py](data_structures/lru_cache.py) | Limited-size cache with eviction policy | Caching frequent queries |
| Queue Structures | [queues.py](data_structures/queues.py) | FIFO data structure | Task processing |
| Red-Black Trees | [red_black_trees.py](data_structures/red_black_trees.py) | Self-balancing binary search tree | Balanced search trees |
| NLP Token Trees | [nlp_token_trees.py](data_structures/nlp_token_trees.py) | Parsed language structure | Natural language processing |

## Algorithms for Data Manipulation/Traversal

| Algorithm | File | Description | FCL Use Case |
|-----------|------|-------------|--------------|
| Regular Expressions | [regex_patterns.py](data_structures/regex_patterns.py) | Pattern matching and extraction algorithm | Pattern matching in legal text |
| Scoring Algorithms | [scoring_algorithms.py](data_structures/scoring_algorithms.py) | Relevance calculation and ranking | Relevance ranking |
| Set Intersection Operations | [sets_intersection.py](data_structures/sets_intersection.py) | Algorithm for finding common elements | Document filtering |
| BeautifulSoup (parsing/traversal) | [dom_trees.py](data_structures/dom_trees.py) | HTML/XML traversal algorithms | HTML/XML parsing |

## Hybrid (Structure + Built-in Algorithms)

| Hybrid | File | Description | FCL Use Case |
|--------|------|-------------|--------------|
| Inverted Indices | [inverted_indices.py](data_structures/inverted_indices.py) | Include both structure and search algorithms | Full-text search |
| Bloom Filters | [bloom_filters.py](data_structures/bloom_filters.py) | Include hash algorithms for membership testing | Probabilistic membership testing |
| Django ORM QuerySets | [django_querysets.py](data_structures/django_querysets.py) | Include SQL generation algorithms | Database operations |
| Segment Trees | [segment_trees.py](data_structures/segment_trees.py) | Include range query algorithms | Range query optimization |

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
