"""
Trie Data Structure for FCL Legal Case Autocomplete

This implementation provides autocomplete functionality for legal case names,
neutral citations, and legal terms commonly used in UK legal databases.

Use cases:
- Autocomplete for case name searches
- Legal term suggestions in search interfaces
- Citation format validation and completion
- Quick lookup of case references
"""

class TrieNode:
    """Node in the trie structure for storing characters and metadata."""

    def __init__(self):
        self.children = {}  # Dictionary mapping character to TrieNode
        self.is_end_word = False  # True if this node marks the end of a word
        self.metadata = {}  # Store additional info like case details, citation format
        self.frequency = 0  # Track how often this term is searched

class LegalTrie:
    """
    Trie optimized for legal case names and terms with metadata support.
    Handles UK-specific legal formatting and neutral citations.
    """

    def __init__(self):
        self.root = TrieNode()
        self.case_count = 0

    def insert(self, word, metadata=None, frequency=1):
        """
        Insert a legal term or case name into the trie.

        Args:
            word (str): Case name or legal term
            metadata (dict): Additional info (court, year, parties, etc.)
            frequency (int): Search frequency for ranking suggestions
        """
        node = self.root
        word = word.lower().strip()  # Normalize input

        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]

        node.is_end_word = True
        node.frequency += frequency
        if metadata:
            node.metadata.update(metadata)

        if metadata and not hasattr(node, 'case_inserted'):
            self.case_count += 1
            node.case_inserted = True

    def search(self, word):
        """
        Search for exact match of a legal term.

        Args:
            word (str): Term to search for

        Returns:
            dict: Metadata if found, None otherwise
        """
        node = self.root
        word = word.lower().strip()

        for char in word:
            if char not in node.children:
                return None
            node = node.children[char]

        return node.metadata if node.is_end_word else None

    def autocomplete(self, prefix, max_suggestions=10):
        """
        Get autocomplete suggestions for legal terms and case names.

        Args:
            prefix (str): Partial case name or legal term
            max_suggestions (int): Maximum number of suggestions to return

        Returns:
            list: Tuples of (suggestion, metadata, frequency) sorted by relevance
        """
        prefix = prefix.lower().strip()
        suggestions = []

        # Navigate to the prefix node
        node = self.root
        for char in prefix:
            if char not in node.children:
                return []
            node = node.children[char]

        # Collect all completions from this point
        self._collect_suggestions(node, prefix, suggestions)

        # Sort by frequency (most searched first) and limit results
        suggestions.sort(key=lambda x: x[2], reverse=True)
        return suggestions[:max_suggestions]

    def _collect_suggestions(self, node, current_word, suggestions):
        """Recursively collect all valid completions from a node."""
        if node.is_end_word:
            suggestions.append((current_word, node.metadata, node.frequency))

        for char, child_node in node.children.items():
            self._collect_suggestions(child_node, current_word + char, suggestions)

    def get_citation_suggestions(self, partial_citation):
        """
        Specialized autocomplete for UK neutral citations.

        Args:
            partial_citation (str): Partial citation like "[2023] EW"

        Returns:
            list: Valid citation format completions
        """
        # Common UK neutral citation patterns
        citation_patterns = {
            "[2023] ewhc": "[2023] EWHC (High Court)",
            "[2023] ewca": "[2023] EWCA (Court of Appeal)",
            "[2023] uksc": "[2023] UKSC (Supreme Court)",
            "[2023] ukut": "[2023] UKUT (Upper Tribunal)",
            "[2023] ewcop": "[2023] EWCOP (Court of Protection)"
        }

        partial = partial_citation.lower().strip()
        matches = []

        for pattern, description in citation_patterns.items():
            if pattern.startswith(partial):
                matches.append((pattern.upper(), {"type": "citation", "description": description}, 100))

        return matches


def demo_fcl_legal_autocomplete():
    """Demonstration of the trie with realistic FCL legal data."""

    # Initialize the legal trie
    legal_trie = LegalTrie()

    # Sample UK legal cases and terms
    legal_data = [
        # Famous UK cases with metadata
        ("donoghue v stevenson", {
            "citation": "[1932] AC 562",
            "court": "House of Lords",
            "year": 1932,
            "area": "Tort Law",
            "significance": "Established modern negligence law"
        }, 150),

        ("carlill v carbolic smoke ball company", {
            "citation": "[1893] 1 QB 256",
            "court": "Court of Appeal",
            "year": 1893,
            "area": "Contract Law",
            "significance": "Unilateral contract formation"
        }, 120),

        ("rylands v fletcher", {
            "citation": "(1868) LR 3 HL 330",
            "court": "House of Lords",
            "year": 1868,
            "area": "Tort Law",
            "significance": "Strict liability for dangerous activities"
        }, 95),

        ("r v brown", {
            "citation": "[1994] 1 AC 212",
            "court": "House of Lords",
            "year": 1994,
            "area": "Criminal Law",
            "significance": "Consent in assault cases"
        }, 80),

        ("pepper v hart", {
            "citation": "[1993] AC 593",
            "court": "House of Lords",
            "year": 1993,
            "area": "Constitutional Law",
            "significance": "Use of Hansard in statutory interpretation"
        }, 110),

        # Common legal terms
        ("negligence", {"type": "legal_concept", "area": "Tort Law"}, 200),
        ("consideration", {"type": "legal_concept", "area": "Contract Law"}, 180),
        ("judicial review", {"type": "legal_concept", "area": "Administrative Law"}, 160),
        ("natural justice", {"type": "legal_concept", "area": "Administrative Law"}, 140),
        ("estoppel", {"type": "legal_concept", "area": "Contract Law"}, 130),
        ("ultra vires", {"type": "legal_concept", "area": "Administrative Law"}, 120),
        ("burden of proof", {"type": "legal_concept", "area": "Evidence Law"}, 115),
        ("reasonable doubt", {"type": "legal_concept", "area": "Criminal Law"}, 105),
    ]

    # Insert all legal data into the trie
    print("Inserting FCL legal database entries...")
    for term, metadata, frequency in legal_data:
        legal_trie.insert(term, metadata, frequency)

    print(f"Loaded {legal_trie.case_count} legal cases and terms\n")

    # Demonstrate autocomplete functionality
    test_queries = [
        "don",      # Should suggest "donoghue v stevenson"
        "car",      # Should suggest "carlill v carbolic smoke ball company"
        "neg",      # Should suggest "negligence"
        "r v",      # Should suggest "r v brown"
        "jud",      # Should suggest "judicial review"
        "pep",      # Should suggest "pepper v hart"
    ]

    print("=== Legal Term Autocomplete Demo ===")
    for query in test_queries:
        suggestions = legal_trie.autocomplete(query, max_suggestions=3)
        print(f"\nQuery: '{query}'")

        if suggestions:
            for i, (suggestion, metadata, freq) in enumerate(suggestions, 1):
                print(f"  {i}. {suggestion.title()}")
                if 'citation' in metadata:
                    print(f"     Citation: {metadata['citation']}")
                if 'area' in metadata:
                    print(f"     Area: {metadata['area']}")
                if 'significance' in metadata:
                    print(f"     Significance: {metadata['significance']}")
                print(f"     Search frequency: {freq}")
        else:
            print("  No suggestions found")

    # Demonstrate citation autocomplete
    print("\n=== Citation Format Autocomplete Demo ===")
    citation_queries = ["[2023] ew", "[2023] uk", "[2023] ewc"]

    for query in citation_queries:
        suggestions = legal_trie.get_citation_suggestions(query)
        print(f"\nCitation query: '{query}'")

        if suggestions:
            for i, (citation, metadata, _) in enumerate(suggestions, 1):
                print(f"  {i}. {citation} - {metadata['description']}")
        else:
            print("  No citation patterns found")

    # Search for specific cases
    print("\n=== Exact Case Lookup Demo ===")
    exact_searches = ["donoghue v stevenson", "pepper v hart", "negligence"]

    for search_term in exact_searches:
        result = legal_trie.search(search_term)
        print(f"\nSearching for: '{search_term}'")

        if result:
            print(f"  Found: {search_term.title()}")
            if 'citation' in result:
                print(f"  Citation: {result['citation']}")
            if 'court' in result:
                print(f"  Court: {result['court']}")
            if 'year' in result:
                print(f"  Year: {result['year']}")
            if 'area' in result:
                print(f"  Legal Area: {result['area']}")
        else:
            print("  Case not found in database")


if __name__ == "__main__":
    demo_fcl_legal_autocomplete()