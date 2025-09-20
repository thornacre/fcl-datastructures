/**
 * Trie Data Structure for FCL Legal Case Autocomplete
 *
 * This implementation provides autocomplete functionality for legal case names,
 * neutral citations, and legal terms commonly used in UK legal databases.
 *
 * Use cases:
 * - Autocomplete for case name searches
 * - Legal term suggestions in search interfaces
 * - Citation format validation and completion
 * - Quick lookup of case references
 */

class TrieNode {
    /**
     * Node in the trie structure for storing characters and metadata.
     */
    constructor() {
        this.children = new Map(); // Map character to TrieNode
        this.isEndWord = false;    // True if this node marks the end of a word
        this.metadata = {};        // Store additional info like case details, citation format
        this.frequency = 0;        // Track how often this term is searched
    }
}

class LegalTrie {
    /**
     * Trie optimized for legal case names and terms with metadata support.
     * Handles UK-specific legal formatting and neutral citations.
     */
    constructor() {
        this.root = new TrieNode();
        this.caseCount = 0;
    }

    /**
     * Insert a legal term or case name into the trie.
     *
     * @param {string} word - Case name or legal term
     * @param {Object} metadata - Additional info (court, year, parties, etc.)
     * @param {number} frequency - Search frequency for ranking suggestions
     */
    insert(word, metadata = {}, frequency = 1) {
        let node = this.root;
        word = word.toLowerCase().trim(); // Normalize input

        for (const char of word) {
            if (!node.children.has(char)) {
                node.children.set(char, new TrieNode());
            }
            node = node.children.get(char);
        }

        node.isEndWord = true;
        node.frequency += frequency;
        Object.assign(node.metadata, metadata);

        if (metadata && !node.caseInserted) {
            this.caseCount++;
            node.caseInserted = true;
        }
    }

    /**
     * Search for exact match of a legal term.
     *
     * @param {string} word - Term to search for
     * @returns {Object|null} Metadata if found, null otherwise
     */
    search(word) {
        let node = this.root;
        word = word.toLowerCase().trim();

        for (const char of word) {
            if (!node.children.has(char)) {
                return null;
            }
            node = node.children.get(char);
        }

        return node.isEndWord ? node.metadata : null;
    }

    /**
     * Get autocomplete suggestions for legal terms and case names.
     *
     * @param {string} prefix - Partial case name or legal term
     * @param {number} maxSuggestions - Maximum number of suggestions to return
     * @returns {Array} Tuples of [suggestion, metadata, frequency] sorted by relevance
     */
    autocomplete(prefix, maxSuggestions = 10) {
        prefix = prefix.toLowerCase().trim();
        const suggestions = [];

        // Navigate to the prefix node
        let node = this.root;
        for (const char of prefix) {
            if (!node.children.has(char)) {
                return [];
            }
            node = node.children.get(char);
        }

        // Collect all completions from this point
        this._collectSuggestions(node, prefix, suggestions);

        // Sort by frequency (most searched first) and limit results
        suggestions.sort((a, b) => b[2] - a[2]);
        return suggestions.slice(0, maxSuggestions);
    }

    /**
     * Recursively collect all valid completions from a node.
     * @private
     */
    _collectSuggestions(node, currentWord, suggestions) {
        if (node.isEndWord) {
            suggestions.push([currentWord, node.metadata, node.frequency]);
        }

        for (const [char, childNode] of node.children) {
            this._collectSuggestions(childNode, currentWord + char, suggestions);
        }
    }

    /**
     * Specialized autocomplete for UK neutral citations.
     *
     * @param {string} partialCitation - Partial citation like "[2023] EW"
     * @returns {Array} Valid citation format completions
     */
    getCitationSuggestions(partialCitation) {
        // Common UK neutral citation patterns
        const citationPatterns = {
            "[2023] ewhc": "[2023] EWHC (High Court)",
            "[2023] ewca": "[2023] EWCA (Court of Appeal)",
            "[2023] uksc": "[2023] UKSC (Supreme Court)",
            "[2023] ukut": "[2023] UKUT (Upper Tribunal)",
            "[2023] ewcop": "[2023] EWCOP (Court of Protection)"
        };

        const partial = partialCitation.toLowerCase().trim();
        const matches = [];

        for (const [pattern, description] of Object.entries(citationPatterns)) {
            if (pattern.startsWith(partial)) {
                matches.push([
                    pattern.toUpperCase(),
                    { type: "citation", description: description },
                    100
                ]);
            }
        }

        return matches;
    }
}

/**
 * Demonstration of the trie with realistic FCL legal data.
 */
function demoFCLLegalAutocomplete() {
    // Initialize the legal trie
    const legalTrie = new LegalTrie();

    // Sample UK legal cases and terms
    const legalData = [
        // Famous UK cases with metadata
        ["donoghue v stevenson", {
            citation: "[1932] AC 562",
            court: "House of Lords",
            year: 1932,
            area: "Tort Law",
            significance: "Established modern negligence law"
        }, 150],

        ["carlill v carbolic smoke ball company", {
            citation: "[1893] 1 QB 256",
            court: "Court of Appeal",
            year: 1893,
            area: "Contract Law",
            significance: "Unilateral contract formation"
        }, 120],

        ["rylands v fletcher", {
            citation: "(1868) LR 3 HL 330",
            court: "House of Lords",
            year: 1868,
            area: "Tort Law",
            significance: "Strict liability for dangerous activities"
        }, 95],

        ["r v brown", {
            citation: "[1994] 1 AC 212",
            court: "House of Lords",
            year: 1994,
            area: "Criminal Law",
            significance: "Consent in assault cases"
        }, 80],

        ["pepper v hart", {
            citation: "[1993] AC 593",
            court: "House of Lords",
            year: 1993,
            area: "Constitutional Law",
            significance: "Use of Hansard in statutory interpretation"
        }, 110],

        // Common legal terms
        ["negligence", { type: "legal_concept", area: "Tort Law" }, 200],
        ["consideration", { type: "legal_concept", area: "Contract Law" }, 180],
        ["judicial review", { type: "legal_concept", area: "Administrative Law" }, 160],
        ["natural justice", { type: "legal_concept", area: "Administrative Law" }, 140],
        ["estoppel", { type: "legal_concept", area: "Contract Law" }, 130],
        ["ultra vires", { type: "legal_concept", area: "Administrative Law" }, 120],
        ["burden of proof", { type: "legal_concept", area: "Evidence Law" }, 115],
        ["reasonable doubt", { type: "legal_concept", area: "Criminal Law" }, 105]
    ];

    // Insert all legal data into the trie
    console.log("Inserting FCL legal database entries...");
    for (const [term, metadata, frequency] of legalData) {
        legalTrie.insert(term, metadata, frequency);
    }

    console.log(`Loaded ${legalTrie.caseCount} legal cases and terms\n`);

    // Demonstrate autocomplete functionality
    const testQueries = [
        "don",      // Should suggest "donoghue v stevenson"
        "car",      // Should suggest "carlill v carbolic smoke ball company"
        "neg",      // Should suggest "negligence"
        "r v",      // Should suggest "r v brown"
        "jud",      // Should suggest "judicial review"
        "pep"       // Should suggest "pepper v hart"
    ];

    console.log("=== Legal Term Autocomplete Demo ===");
    for (const query of testQueries) {
        const suggestions = legalTrie.autocomplete(query, 3);
        console.log(`\nQuery: '${query}'`);

        if (suggestions.length > 0) {
            suggestions.forEach(([suggestion, metadata, freq], i) => {
                console.log(`  ${i + 1}. ${toTitleCase(suggestion)}`);
                if (metadata.citation) {
                    console.log(`     Citation: ${metadata.citation}`);
                }
                if (metadata.area) {
                    console.log(`     Area: ${metadata.area}`);
                }
                if (metadata.significance) {
                    console.log(`     Significance: ${metadata.significance}`);
                }
                console.log(`     Search frequency: ${freq}`);
            });
        } else {
            console.log("  No suggestions found");
        }
    }

    // Demonstrate citation autocomplete
    console.log("\n=== Citation Format Autocomplete Demo ===");
    const citationQueries = ["[2023] ew", "[2023] uk", "[2023] ewc"];

    for (const query of citationQueries) {
        const suggestions = legalTrie.getCitationSuggestions(query);
        console.log(`\nCitation query: '${query}'`);

        if (suggestions.length > 0) {
            suggestions.forEach(([citation, metadata], i) => {
                console.log(`  ${i + 1}. ${citation} - ${metadata.description}`);
            });
        } else {
            console.log("  No citation patterns found");
        }
    }

    // Search for specific cases
    console.log("\n=== Exact Case Lookup Demo ===");
    const exactSearches = ["donoghue v stevenson", "pepper v hart", "negligence"];

    for (const searchTerm of exactSearches) {
        const result = legalTrie.search(searchTerm);
        console.log(`\nSearching for: '${searchTerm}'`);

        if (result) {
            console.log(`  Found: ${toTitleCase(searchTerm)}`);
            if (result.citation) {
                console.log(`  Citation: ${result.citation}`);
            }
            if (result.court) {
                console.log(`  Court: ${result.court}`);
            }
            if (result.year) {
                console.log(`  Year: ${result.year}`);
            }
            if (result.area) {
                console.log(`  Legal Area: ${result.area}`);
            }
        } else {
            console.log("  Case not found in database");
        }
    }
}

/**
 * Helper function to convert string to title case.
 */
function toTitleCase(str) {
    return str.replace(/\w\S*/g, (txt) =>
        txt.charAt(0).toUpperCase() + txt.substr(1).toLowerCase()
    );
}

// Export for Node.js environment
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { LegalTrie, TrieNode, demoFCLLegalAutocomplete };
}

// Run demo if called directly
if (typeof require !== 'undefined' && require.main === module) {
    demoFCLLegalAutocomplete();
}