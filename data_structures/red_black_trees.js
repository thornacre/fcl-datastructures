/**
 * Red-Black Trees for Sorted Judgment Collections
 * ===============================================
 *
 * This module implements self-balancing binary search trees (Red-Black Trees)
 * for maintaining sorted collections of judgments in Find Case Law (FCL).
 * Provides efficient insertion, deletion, and range query operations.
 *
 * Key FCL Use Cases:
 * - Maintaining chronologically sorted judgment collections
 * - Efficient range queries by date, citation, or relevance score
 * - Balanced tree structure for consistent O(log n) performance
 * - Supporting sorted iteration through large judgment datasets
 * - Timeline views and date-based filtering operations
 */

// Node colors for Red-Black Tree
const Color = {
    RED: 'RED',
    BLACK: 'BLACK'
};

/**
 * Represents a UK legal judgment with sorting criteria
 */
class Judgment {
    constructor(neutralCitation, caseName, court, judgmentDate, uri, relevanceScore = 0.0) {
        this.neutralCitation = neutralCitation;
        this.caseName = caseName;
        this.court = court;
        this.judgmentDate = new Date(judgmentDate);
        this.uri = uri;
        this.relevanceScore = relevanceScore;
    }

    toString() {
        return `${this.neutralCitation}: ${this.caseName}`;
    }
}

/**
 * Red-Black Tree node containing judgment data
 */
class RBNode {
    constructor(key, value, color = Color.RED) {
        this.key = key;
        this.value = value;
        this.color = color;
        this.left = null;
        this.right = null;
        this.parent = null;
    }

    toString() {
        const colorStr = this.color === Color.RED ? 'R' : 'B';
        return `[${colorStr}] ${this.key}: ${this.value.caseName}`;
    }
}

/**
 * Self-balancing binary search tree for sorted judgment collections.
 * Maintains Red-Black Tree properties for guaranteed O(log n) operations.
 */
class RedBlackTree {
    /**
     * Initialize Red-Black Tree with optional custom key function.
     * @param {Function} keyFunction - Function to extract sort key from Judgment objects.
     *                                Defaults to sorting by judgment date.
     */
    constructor(keyFunction = null) {
        this.nil = new RBNode(null, null, Color.BLACK); // Sentinel node
        this.root = this.nil;
        this.size = 0;

        // Default to sorting by judgment date
        this.keyFunction = keyFunction || ((j) => j.judgmentDate.getTime());
    }

    /**
     * Insert a judgment into the tree
     */
    insert(judgment) {
        const key = this.keyFunction(judgment);
        const newNode = new RBNode(key, judgment, Color.RED);
        newNode.left = this.nil;
        newNode.right = this.nil;

        // Standard BST insertion
        let parent = null;
        let current = this.root;

        while (current !== this.nil) {
            parent = current;
            if (key < current.key) {
                current = current.left;
            } else if (key > current.key) {
                current = current.right;
            } else {
                // Key already exists, update the value
                current.value = judgment;
                return;
            }
        }

        newNode.parent = parent;

        if (parent === null) {
            this.root = newNode;
        } else if (key < parent.key) {
            parent.left = newNode;
        } else {
            parent.right = newNode;
        }

        this.size++;

        // Fix Red-Black Tree properties
        this._insertFixup(newNode);
    }

    /**
     * Delete a judgment by key
     */
    delete(key) {
        const node = this._searchNode(key);
        if (node === this.nil) {
            return false;
        }

        this._deleteNode(node);
        this.size--;
        return true;
    }

    /**
     * Search for a judgment by key
     */
    search(key) {
        const node = this._searchNode(key);
        return node !== this.nil ? node.value : null;
    }

    /**
     * Find all judgments with keys in the range [startKey, endKey].
     * Useful for date range queries or score range filtering.
     */
    rangeQuery(startKey, endKey) {
        const result = [];
        this._rangeQueryHelper(this.root, startKey, endKey, result);
        return result;
    }

    /**
     * Get judgments within a specific date range
     */
    getByDateRange(startDate, endDate) {
        const startTime = new Date(startDate).getTime();
        const endTime = new Date(endDate).getTime();

        if (this.keyFunction.toString().includes('judgmentDate')) {
            return this.rangeQuery(startTime, endTime);
        } else {
            // If not sorting by date, need to scan all nodes
            const result = [];
            for (const judgment of this.inorderTraversal()) {
                const judgmentTime = judgment.judgmentDate.getTime();
                if (startTime <= judgmentTime && judgmentTime <= endTime) {
                    result.push(judgment);
                }
            }
            return result;
        }
    }

    /**
     * Get all judgments from a specific court
     */
    getByCourt(court) {
        const result = [];
        for (const judgment of this.inorderTraversal()) {
            if (judgment.court === court) {
                result.push(judgment);
            }
        }
        return result;
    }

    /**
     * Iterate through judgments in sorted order
     */
    *inorderTraversal() {
        yield* this._inorderHelper(this.root);
    }

    /**
     * Get judgment with minimum key
     */
    getMinimum() {
        const node = this._minimum(this.root);
        return node !== this.nil ? node.value : null;
    }

    /**
     * Get judgment with maximum key
     */
    getMaximum() {
        const node = this._maximum(this.root);
        return node !== this.nil ? node.value : null;
    }

    /**
     * Get all judgments as a sorted array
     */
    getSortedArray(reverse = false) {
        const judgments = Array.from(this.inorderTraversal());
        return reverse ? judgments.reverse() : judgments;
    }

    /**
     * Get tree statistics
     */
    getStatistics() {
        const height = this._height(this.root);
        return {
            size: this.size,
            height: height,
            isBalanced: this.size > 0 ? height <= 2 * this._log2(this.size + 1) : true,
            blackHeight: this._blackHeight(this.root)
        };
    }

    /**
     * Fix Red-Black Tree properties after insertion
     */
    _insertFixup(node) {
        while (node.parent && node.parent.color === Color.RED) {
            if (node.parent === node.parent.parent.left) {
                const uncle = node.parent.parent.right;

                if (uncle.color === Color.RED) {
                    // Case 1: Uncle is red
                    node.parent.color = Color.BLACK;
                    uncle.color = Color.BLACK;
                    node.parent.parent.color = Color.RED;
                    node = node.parent.parent;
                } else {
                    if (node === node.parent.right) {
                        // Case 2: Node is right child
                        node = node.parent;
                        this._leftRotate(node);
                    }

                    // Case 3: Node is left child
                    node.parent.color = Color.BLACK;
                    node.parent.parent.color = Color.RED;
                    this._rightRotate(node.parent.parent);
                }
            } else {
                // Symmetric cases (parent is right child)
                const uncle = node.parent.parent.left;

                if (uncle.color === Color.RED) {
                    node.parent.color = Color.BLACK;
                    uncle.color = Color.BLACK;
                    node.parent.parent.color = Color.RED;
                    node = node.parent.parent;
                } else {
                    if (node === node.parent.left) {
                        node = node.parent;
                        this._rightRotate(node);
                    }

                    node.parent.color = Color.BLACK;
                    node.parent.parent.color = Color.RED;
                    this._leftRotate(node.parent.parent);
                }
            }
        }

        this.root.color = Color.BLACK;
    }

    /**
     * Delete a node and maintain Red-Black properties
     */
    _deleteNode(node) {
        let y = node;
        let yOriginalColor = y.color;
        let x;

        if (node.left === this.nil) {
            x = node.right;
            this._transplant(node, node.right);
        } else if (node.right === this.nil) {
            x = node.left;
            this._transplant(node, node.left);
        } else {
            y = this._minimum(node.right);
            yOriginalColor = y.color;
            x = y.right;

            if (y.parent === node) {
                x.parent = y;
            } else {
                this._transplant(y, y.right);
                y.right = node.right;
                y.right.parent = y;
            }

            this._transplant(node, y);
            y.left = node.left;
            y.left.parent = y;
            y.color = node.color;
        }

        if (yOriginalColor === Color.BLACK) {
            this._deleteFixup(x);
        }
    }

    /**
     * Fix Red-Black Tree properties after deletion
     */
    _deleteFixup(node) {
        while (node !== this.root && node.color === Color.BLACK) {
            if (node === node.parent.left) {
                let sibling = node.parent.right;

                if (sibling.color === Color.RED) {
                    sibling.color = Color.BLACK;
                    node.parent.color = Color.RED;
                    this._leftRotate(node.parent);
                    sibling = node.parent.right;
                }

                if (sibling.left.color === Color.BLACK && sibling.right.color === Color.BLACK) {
                    sibling.color = Color.RED;
                    node = node.parent;
                } else {
                    if (sibling.right.color === Color.BLACK) {
                        sibling.left.color = Color.BLACK;
                        sibling.color = Color.RED;
                        this._rightRotate(sibling);
                        sibling = node.parent.right;
                    }

                    sibling.color = node.parent.color;
                    node.parent.color = Color.BLACK;
                    sibling.right.color = Color.BLACK;
                    this._leftRotate(node.parent);
                    node = this.root;
                }
            } else {
                // Symmetric cases
                let sibling = node.parent.left;

                if (sibling.color === Color.RED) {
                    sibling.color = Color.BLACK;
                    node.parent.color = Color.RED;
                    this._rightRotate(node.parent);
                    sibling = node.parent.left;
                }

                if (sibling.right.color === Color.BLACK && sibling.left.color === Color.BLACK) {
                    sibling.color = Color.RED;
                    node = node.parent;
                } else {
                    if (sibling.left.color === Color.BLACK) {
                        sibling.right.color = Color.BLACK;
                        sibling.color = Color.RED;
                        this._leftRotate(sibling);
                        sibling = node.parent.left;
                    }

                    sibling.color = node.parent.color;
                    node.parent.color = Color.BLACK;
                    sibling.left.color = Color.BLACK;
                    this._rightRotate(node.parent);
                    node = this.root;
                }
            }
        }

        node.color = Color.BLACK;
    }

    /**
     * Perform left rotation
     */
    _leftRotate(x) {
        const y = x.right;
        x.right = y.left;

        if (y.left !== this.nil) {
            y.left.parent = x;
        }

        y.parent = x.parent;

        if (x.parent === null) {
            this.root = y;
        } else if (x === x.parent.left) {
            x.parent.left = y;
        } else {
            x.parent.right = y;
        }

        y.left = x;
        x.parent = y;
    }

    /**
     * Perform right rotation
     */
    _rightRotate(y) {
        const x = y.left;
        y.left = x.right;

        if (x.right !== this.nil) {
            x.right.parent = y;
        }

        x.parent = y.parent;

        if (y.parent === null) {
            this.root = x;
        } else if (y === y.parent.right) {
            y.parent.right = x;
        } else {
            y.parent.left = x;
        }

        x.right = y;
        y.parent = x;
    }

    /**
     * Replace subtree rooted at u with subtree rooted at v
     */
    _transplant(u, v) {
        if (u.parent === null) {
            this.root = v;
        } else if (u === u.parent.left) {
            u.parent.left = v;
        } else {
            u.parent.right = v;
        }
        v.parent = u.parent;
    }

    /**
     * Search for a node by key
     */
    _searchNode(key) {
        let current = this.root;
        while (current !== this.nil && key !== current.key) {
            if (key < current.key) {
                current = current.left;
            } else {
                current = current.right;
            }
        }
        return current;
    }

    /**
     * Find minimum node in subtree
     */
    _minimum(node) {
        while (node.left !== this.nil) {
            node = node.left;
        }
        return node;
    }

    /**
     * Find maximum node in subtree
     */
    _maximum(node) {
        while (node.right !== this.nil) {
            node = node.right;
        }
        return node;
    }

    /**
     * Helper method for range queries
     */
    _rangeQueryHelper(node, startKey, endKey, result) {
        if (node === this.nil) {
            return;
        }

        // If current key is in range, add to result
        if (startKey <= node.key && node.key <= endKey) {
            result.push(node.value);
        }

        // Recursively search left subtree if needed
        if (startKey <= node.key) {
            this._rangeQueryHelper(node.left, startKey, endKey, result);
        }

        // Recursively search right subtree if needed
        if (node.key <= endKey) {
            this._rangeQueryHelper(node.right, startKey, endKey, result);
        }
    }

    /**
     * Helper for inorder traversal
     */
    *_inorderHelper(node) {
        if (node !== this.nil) {
            yield* this._inorderHelper(node.left);
            yield node.value;
            yield* this._inorderHelper(node.right);
        }
    }

    /**
     * Calculate height of subtree
     */
    _height(node) {
        if (node === this.nil) {
            return 0;
        }
        return 1 + Math.max(this._height(node.left), this._height(node.right));
    }

    /**
     * Calculate black height of subtree
     */
    _blackHeight(node) {
        if (node === this.nil) {
            return 0;
        }

        const leftHeight = this._blackHeight(node.left);
        const rightHeight = this._blackHeight(node.right);

        // Add 1 if current node is black
        const heightIncrement = node.color === Color.BLACK ? 1 : 0;
        return leftHeight + heightIncrement;
    }

    /**
     * Calculate log base 2
     */
    _log2(n) {
        if (n <= 1) {
            return 0;
        }
        return 1 + this._log2(Math.floor(n / 2));
    }
}

/**
 * High-level collection manager using Red-Black Trees for different sorting criteria.
 * Maintains multiple sorted views of the same judgment data.
 */
class JudgmentCollection {
    constructor() {
        // Multiple trees for different sorting criteria
        this.byDate = new RedBlackTree(j => j.judgmentDate.getTime());
        this.byCitation = new RedBlackTree(j => j.neutralCitation);
        this.byRelevance = new RedBlackTree(j => -j.relevanceScore); // Descending order

        // Keep track of all judgments for cross-referencing
        this.judgments = new Map(); // citation -> judgment
    }

    /**
     * Add a judgment to all sorted collections
     */
    addJudgment(judgment) {
        // Remove existing judgment if present
        if (this.judgments.has(judgment.neutralCitation)) {
            this.removeJudgment(judgment.neutralCitation);
        }

        // Add to all trees
        this.byDate.insert(judgment);
        this.byCitation.insert(judgment);
        this.byRelevance.insert(judgment);

        // Store in main collection
        this.judgments.set(judgment.neutralCitation, judgment);
    }

    /**
     * Remove a judgment from all collections
     */
    removeJudgment(citation) {
        if (!this.judgments.has(citation)) {
            return false;
        }

        const judgment = this.judgments.get(citation);

        // Remove from all trees
        this.byDate.delete(judgment.judgmentDate.getTime());
        this.byCitation.delete(judgment.neutralCitation);
        this.byRelevance.delete(-judgment.relevanceScore);

        // Remove from main collection
        this.judgments.delete(citation);
        return true;
    }

    /**
     * Get judgment by neutral citation
     */
    getByCitation(citation) {
        return this.judgments.get(citation) || null;
    }

    /**
     * Get most recent judgments
     */
    getRecentJudgments(limit = 10) {
        const judgments = this.byDate.getSortedArray(true);
        return judgments.slice(0, limit);
    }

    /**
     * Get judgments within date range
     */
    getByDateRange(startDate, endDate) {
        return this.byDate.getByDateRange(startDate, endDate);
    }

    /**
     * Get highest scoring judgments
     */
    getByRelevance(limit = 10) {
        const judgments = this.byRelevance.getSortedArray();
        return judgments.slice(0, limit);
    }

    /**
     * Get all judgments from specific court
     */
    getByCourt(court) {
        return this.byDate.getByCourt(court);
    }

    /**
     * Search for judgments with citations starting with prefix
     */
    searchCitations(prefix) {
        const result = [];
        for (const judgment of this.byCitation.inorderTraversal()) {
            if (judgment.neutralCitation.startsWith(prefix)) {
                result.push(judgment);
            }
        }
        return result;
    }

    /**
     * Get comprehensive collection statistics
     */
    getCollectionStats() {
        const courts = {};
        const years = {};

        for (const judgment of this.judgments.values()) {
            // Court statistics
            courts[judgment.court] = (courts[judgment.court] || 0) + 1;

            // Year statistics
            const year = judgment.judgmentDate.getFullYear();
            years[year] = (years[year] || 0) + 1;
        }

        return {
            totalJudgments: this.judgments.size,
            dateTreeStats: this.byDate.getStatistics(),
            citationTreeStats: this.byCitation.getStatistics(),
            relevanceTreeStats: this.byRelevance.getStatistics(),
            courts: courts,
            years: years
        };
    }
}

/**
 * Demonstrate Red-Black Tree implementation with UK legal data
 */
function demonstrateRedBlackTrees() {
    console.log('=== Red-Black Trees for Sorted Judgments Demo ===\n');

    // Sample UK judgments
    const judgmentsData = [
        {
            citation: '[2023] UKSC 15',
            caseName: 'R (Miller) v Prime Minister',
            court: 'UKSC',
            date: '2023-05-15',
            uri: 'https://caselaw.nationalarchives.gov.uk/uksc/2023/15',
            score: 0.95
        },
        {
            citation: '[2023] EWCA Civ 892',
            caseName: 'Smith v Secretary of State for Work and Pensions',
            court: 'EWCA',
            date: '2023-08-22',
            uri: 'https://caselaw.nationalarchives.gov.uk/ewca/civ/2023/892',
            score: 0.87
        },
        {
            citation: '[2023] EWHC 1456 (Admin)',
            caseName: 'Jones v Local Authority',
            court: 'EWHC',
            date: '2023-06-30',
            uri: 'https://caselaw.nationalarchives.gov.uk/ewhc/admin/2023/1456',
            score: 0.73
        },
        {
            citation: '[2023] UKHL 7',
            caseName: 'Williams v Crown Prosecution Service',
            court: 'UKHL',
            date: '2023-03-10',
            uri: 'https://caselaw.nationalarchives.gov.uk/ukhl/2023/7',
            score: 0.91
        },
        {
            citation: '[2023] UKFTT 234 (TC)',
            caseName: 'Brown v HMRC',
            court: 'UKFTT',
            date: '2023-09-05',
            uri: 'https://caselaw.nationalarchives.gov.uk/ukftt/tc/2023/234',
            score: 0.62
        }
    ];

    // Create judgment objects
    const judgments = judgmentsData.map(data => new Judgment(
        data.citation,
        data.caseName,
        data.court,
        data.date,
        data.uri,
        data.score
    ));

    // 1. Basic Red-Black Tree Operations
    console.log('1. BASIC RED-BLACK TREE (sorted by date):');
    const dateTree = new RedBlackTree(j => j.judgmentDate.getTime());

    console.log('   Inserting judgments:');
    judgments.forEach(judgment => {
        dateTree.insert(judgment);
        console.log(`   Added: ${judgment.neutralCitation} (${judgment.judgmentDate.toDateString()})`);
    });

    console.log(`\n   Tree statistics: ${JSON.stringify(dateTree.getStatistics())}`);

    // 2. Sorted Traversal
    console.log(`\n2. CHRONOLOGICAL ORDER (in-order traversal):`);
    let i = 1;
    for (const judgment of dateTree.inorderTraversal()) {
        console.log(`   ${i++}. ${judgment.judgmentDate.toDateString()}: ${judgment.caseName}`);
    }

    // 3. Range Queries
    console.log(`\n3. DATE RANGE QUERIES:`);
    const startDate = new Date('2023-05-01');
    const endDate = new Date('2023-08-31');

    const rangeResults = dateTree.getByDateRange(startDate, endDate);
    console.log(`   Judgments between ${startDate.toDateString()} and ${endDate.toDateString()}:`);
    rangeResults.forEach(judgment => {
        console.log(`     - ${judgment.judgmentDate.toDateString()}: ${judgment.caseName}`);
    });

    // 4. Judgment Collection with Multiple Sort Orders
    console.log(`\n4. JUDGMENT COLLECTION (multiple sorted views):`);
    const collection = new JudgmentCollection();

    judgments.forEach(judgment => {
        collection.addJudgment(judgment);
    });

    console.log(`   Collection statistics:`);
    const stats = collection.getCollectionStats();
    console.log(`     Total judgments: ${stats.totalJudgments}`);
    console.log(`     Courts: ${JSON.stringify(stats.courts)}`);
    console.log(`     Years: ${JSON.stringify(stats.years)}`);

    // 5. Different Sorting Criteria
    console.log(`\n5. SORTING BY DIFFERENT CRITERIA:`);

    // Most recent judgments
    console.log(`   Most recent judgments:`);
    const recent = collection.getRecentJudgments(3);
    recent.forEach(judgment => {
        console.log(`     - ${judgment.judgmentDate.toDateString()}: ${judgment.caseName}`);
    });

    // Highest relevance
    console.log(`\n   Highest relevance scores:`);
    const relevant = collection.getByRelevance(3);
    relevant.forEach(judgment => {
        console.log(`     - ${judgment.relevanceScore.toFixed(2)}: ${judgment.caseName}`);
    });

    // By court
    console.log(`\n   Supreme Court judgments:`);
    const ukscJudgments = collection.getByCourt('UKSC');
    ukscJudgments.forEach(judgment => {
        console.log(`     - ${judgment.neutralCitation}: ${judgment.caseName}`);
    });

    // 6. Search Operations
    console.log(`\n6. SEARCH OPERATIONS:`);

    // Citation search
    const citationSearch = collection.searchCitations('[2023] EW');
    console.log(`   Citations starting with '[2023] EW':`);
    citationSearch.forEach(judgment => {
        console.log(`     - ${judgment.neutralCitation}: ${judgment.caseName}`);
    });

    // Specific judgment lookup
    const specific = collection.getByCitation('[2023] UKSC 15');
    if (specific) {
        console.log(`\n   Found judgment: ${specific.caseName}`);
        console.log(`     Court: ${specific.court}`);
        console.log(`     Date: ${specific.judgmentDate.toDateString()}`);
        console.log(`     Relevance: ${specific.relevanceScore}`);
    }

    // 7. Tree Balance Verification
    console.log(`\n7. TREE BALANCE VERIFICATION:`);
    const trees = [
        ['Date', collection.byDate],
        ['Citation', collection.byCitation],
        ['Relevance', collection.byRelevance]
    ];

    trees.forEach(([treeName, tree]) => {
        const treeStats = tree.getStatistics();
        console.log(`   ${treeName} tree:`);
        console.log(`     Size: ${treeStats.size}`);
        console.log(`     Height: ${treeStats.height}`);
        console.log(`     Balanced: ${treeStats.isBalanced}`);
        console.log(`     Black height: ${treeStats.blackHeight}`);
    });

    return {
        dateTree,
        collection,
        statistics: stats,
        sampleQueries: {
            rangeResults,
            recentJudgments: recent,
            relevantJudgments: relevant,
            citationSearch
        }
    };
}

// Export classes and functions for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        Color,
        Judgment,
        RBNode,
        RedBlackTree,
        JudgmentCollection,
        demonstrateRedBlackTrees
    };
}

// Run demonstration if this file is executed directly
if (typeof require !== 'undefined' && require.main === module) {
    demonstrateRedBlackTrees();
}