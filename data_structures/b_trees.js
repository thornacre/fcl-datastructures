/**
 * B-Tree Data Structure for FCL Legal Judgment Date Range Queries
 *
 * This implementation provides efficient date-based querying for legal judgments,
 * optimized for the frequent date range searches in legal databases.
 *
 * Use cases:
 * - Find all judgments within a specific date range
 * - Chronological browsing of legal decisions
 * - Statistical analysis of case frequency over time
 * - Efficient temporal filtering for large legal databases
 */

class LegalJudgment {
    /**
     * Represents a legal judgment with essential metadata.
     */
    constructor(caseName, citation, judgmentDate, court, neutralCitation = "", judges = []) {
        this.caseName = caseName;
        this.citation = citation;
        this.judgmentDate = new Date(judgmentDate);
        this.court = court;
        this.neutralCitation = neutralCitation;
        this.judges = judges;
    }

    toString() {
        return `${this.caseName} ${this.citation} (${this.judgmentDate.toISOString().split('T')[0]})`;
    }
}

class BTreeNode {
    /**
     * Node in the B-tree structure optimized for date-based legal data.
     */
    constructor(isLeaf = false) {
        this.keys = [];         // Array of dates (sorted)
        this.values = [];       // Array of judgment arrays for each date
        this.children = [];     // Array of child nodes
        this.isLeaf = isLeaf;
    }

    /**
     * Check if node has maximum number of keys.
     */
    isFull(maxDegree) {
        return this.keys.length >= maxDegree - 1;
    }
}

class LegalJudgmentBTree {
    /**
     * B-tree optimized for legal judgment date range queries.
     * Supports efficient insertion, search, and range queries by date.
     */
    constructor(maxDegree = 5) {
        this.maxDegree = maxDegree;
        this.minDegree = Math.floor(maxDegree / 2);
        this.root = new BTreeNode(true);
        this.totalJudgments = 0;
    }

    /**
     * Insert a legal judgment into the B-tree, indexed by date.
     */
    insert(judgment) {
        if (this.root.isFull(this.maxDegree)) {
            // Split root if full
            const newRoot = new BTreeNode(false);
            newRoot.children.push(this.root);
            this._splitChild(newRoot, 0);
            this.root = newRoot;
        }

        this._insertNonFull(this.root, judgment);
        this.totalJudgments++;
    }

    /**
     * Insert judgment into a node that is not full.
     * @private
     */
    _insertNonFull(node, judgment) {
        const judgmentDate = judgment.judgmentDate.getTime();

        if (node.isLeaf) {
            // Find insertion position using binary search
            let pos = 0;
            while (pos < node.keys.length && node.keys[pos].getTime() < judgmentDate) {
                pos++;
            }

            if (pos < node.keys.length && node.keys[pos].getTime() === judgmentDate) {
                // Date already exists, add to existing array
                node.values[pos].push(judgment);
            } else {
                // Insert new date
                node.keys.splice(pos, 0, new Date(judgment.judgmentDate));
                node.values.splice(pos, 0, [judgment]);
            }
        } else {
            // Find child to insert into
            let i = node.keys.length - 1;
            while (i >= 0 && judgmentDate < node.keys[i].getTime()) {
                i--;
            }
            i++;

            if (node.children[i].isFull(this.maxDegree)) {
                this._splitChild(node, i);
                if (judgmentDate > node.keys[i].getTime()) {
                    i++;
                }
            }

            this._insertNonFull(node.children[i], judgment);
        }
    }

    /**
     * Split a full child node.
     * @private
     */
    _splitChild(parent, index) {
        const fullChild = parent.children[index];
        const newChild = new BTreeNode(fullChild.isLeaf);

        const midIndex = this.minDegree - 1;

        // Move half the keys and values to new node
        newChild.keys = fullChild.keys.splice(midIndex + 1);
        newChild.values = fullChild.values.splice(midIndex + 1);

        if (!fullChild.isLeaf) {
            newChild.children = fullChild.children.splice(midIndex + 1);
        }

        // Move middle key up to parent
        parent.children.splice(index + 1, 0, newChild);
        parent.keys.splice(index, 0, fullChild.keys[midIndex]);
        parent.values.splice(index, 0, fullChild.values[midIndex]);

        // Remove middle key from child
        fullChild.keys.splice(midIndex, 1);
        fullChild.values.splice(midIndex, 1);
    }

    /**
     * Find all judgments on a specific date.
     */
    searchByDate(targetDate) {
        const target = new Date(targetDate);
        return this._searchNode(this.root, target);
    }

    /**
     * Search for date in a specific node.
     * @private
     */
    _searchNode(node, targetDate) {
        const targetTime = targetDate.getTime();
        let i = 0;

        while (i < node.keys.length && targetTime > node.keys[i].getTime()) {
            i++;
        }

        if (i < node.keys.length && targetTime === node.keys[i].getTime()) {
            return [...node.values[i]]; // Return copy of judgment array
        }

        if (node.isLeaf) {
            return [];
        }

        return this._searchNode(node.children[i], targetDate);
    }

    /**
     * Find all judgments within a date range (inclusive).
     */
    rangeQuery(startDate, endDate) {
        const start = new Date(startDate);
        const end = new Date(endDate);

        if (start > end) {
            return [];
        }

        const results = [];
        this._rangeQueryNode(this.root, start, end, results);

        // Sort results by date, then by case name
        results.sort((a, b) => {
            const dateDiff = a.judgmentDate - b.judgmentDate;
            return dateDiff !== 0 ? dateDiff : a.caseName.localeCompare(b.caseName);
        });

        return results;
    }

    /**
     * Recursively collect judgments in date range.
     * @private
     */
    _rangeQueryNode(node, startDate, endDate, results) {
        const startTime = startDate.getTime();
        const endTime = endDate.getTime();
        let i = 0;

        // Process each key in the node
        while (i < node.keys.length) {
            // If not leaf, check left child first
            if (!node.isLeaf) {
                this._rangeQueryNode(node.children[i], startDate, endDate, results);
            }

            // Check if current key is in range
            const keyTime = node.keys[i].getTime();
            if (startTime <= keyTime && keyTime <= endTime) {
                results.push(...node.values[i]);
            } else if (keyTime > endTime) {
                // No need to check further keys or right children
                return;
            }

            i++;
        }

        // Check rightmost child if not leaf
        if (!node.isLeaf) {
            this._rangeQueryNode(node.children[i], startDate, endDate, results);
        }
    }

    /**
     * Get all cases in chronological order.
     */
    getChronologicalCases(limit = null) {
        const results = [];
        this._collectAllCases(this.root, results);

        // Sort by date
        results.sort((a, b) => a.judgmentDate - b.judgmentDate);

        return limit ? results.slice(0, limit) : results;
    }

    /**
     * Recursively collect all cases from the tree.
     * @private
     */
    _collectAllCases(node, results) {
        if (node.isLeaf) {
            for (const valueArray of node.values) {
                results.push(...valueArray);
            }
        } else {
            for (let i = 0; i < node.keys.length; i++) {
                this._collectAllCases(node.children[i], results);
                results.push(...node.values[i]);
            }
            // Don't forget the rightmost child
            this._collectAllCases(node.children[node.children.length - 1], results);
        }
    }

    /**
     * Get statistics about the legal judgment database.
     */
    getStatistics() {
        const allCases = this.getChronologicalCases();

        if (allCases.length === 0) {
            return { totalCases: 0 };
        }

        // Calculate date range
        const dates = allCases.map(c => c.judgmentDate);
        const earliest = new Date(Math.min(...dates));
        const latest = new Date(Math.max(...dates));

        // Count by court
        const courtCounts = {};
        for (const case_ of allCases) {
            courtCounts[case_.court] = (courtCounts[case_.court] || 0) + 1;
        }

        // Count by year
        const yearCounts = {};
        for (const case_ of allCases) {
            const year = case_.judgmentDate.getFullYear();
            yearCounts[year] = (yearCounts[year] || 0) + 1;
        }

        return {
            totalCases: allCases.length,
            dateRange: `${earliest.toISOString().split('T')[0]} to ${latest.toISOString().split('T')[0]}`,
            earliestCase: earliest,
            latestCase: latest,
            courtDistribution: courtCounts,
            yearlyDistribution: yearCounts
        };
    }
}

/**
 * Demonstration of B-tree with realistic FCL legal judgment data.
 */
function demoFCLLegalBTree() {
    // Initialize B-tree for legal judgments
    const legalBTree = new LegalJudgmentBTree(5);

    // Sample UK legal judgments with realistic data
    const sampleJudgments = [
        new LegalJudgment(
            "Donoghue v Stevenson",
            "[1932] AC 562",
            "1932-05-26",
            "House of Lords",
            "[1932] UKHL 100",
            ["Lord Atkin", "Lord Thankerton", "Lord Macmillan"]
        ),
        new LegalJudgment(
            "Carlill v Carbolic Smoke Ball Company",
            "[1893] 1 QB 256",
            "1892-12-07",
            "Court of Appeal",
            "[1892] EWCA Civ 1",
            ["Lindley LJ", "Bowen LJ", "A.L. Smith LJ"]
        ),
        new LegalJudgment(
            "Rylands v Fletcher",
            "(1868) LR 3 HL 330",
            "1868-07-17",
            "House of Lords",
            "[1868] UKHL 1",
            ["Lord Cairns", "Lord Cranworth"]
        ),
        new LegalJudgment(
            "R v Brown",
            "[1994] 1 AC 212",
            "1993-03-11",
            "House of Lords",
            "[1993] UKHL 19",
            ["Lord Templeman", "Lord Jauncey", "Lord Lowry"]
        ),
        new LegalJudgment(
            "Pepper v Hart",
            "[1993] AC 593",
            "1992-11-26",
            "House of Lords",
            "[1992] UKHL 3",
            ["Lord Griffiths", "Lord Ackner", "Lord Oliver"]
        ),
        new LegalJudgment(
            "Caparo Industries plc v Dickman",
            "[1990] 2 AC 605",
            "1990-02-08",
            "House of Lords",
            "[1990] UKHL 2",
            ["Lord Bridge", "Lord Roskill", "Lord Ackner"]
        ),
        new LegalJudgment(
            "R v R (Rape: Marital Exemption)",
            "[1992] 1 AC 599",
            "1991-10-23",
            "House of Lords",
            "[1991] UKHL 12",
            ["Lord Keith", "Lord Jauncey", "Lord Ackner"]
        ),
        new LegalJudgment(
            "Miller v Jackson",
            "[1977] QB 966",
            "1977-05-03",
            "Court of Appeal",
            "[1977] EWCA Civ 6",
            ["Lord Denning MR", "Geoffrey Lane LJ", "Cumming-Bruce LJ"]
        ),
        new LegalJudgment(
            "Central London Property Trust Ltd v High Trees House Ltd",
            "[1947] KB 130",
            "1946-11-02",
            "King's Bench Division",
            "[1946] EWHC KB 1",
            ["Denning J"]
        ),
        new LegalJudgment(
            "Hedley Byrne & Co Ltd v Heller & Partners Ltd",
            "[1964] AC 465",
            "1963-05-28",
            "House of Lords",
            "[1963] UKHL 4",
            ["Lord Reid", "Lord Morris", "Lord Hodson"]
        ),
        // Add some contemporary cases
        new LegalJudgment(
            "R (Miller) v The Prime Minister",
            "[2019] UKSC 41",
            "2019-09-24",
            "Supreme Court",
            "[2019] UKSC 41",
            ["Lady Hale", "Lord Reed", "Lord Kerr"]
        ),
        new LegalJudgment(
            "Test Claimants in the FII Group Litigation v Revenue and Customs",
            "[2020] UKSC 47",
            "2020-11-04",
            "Supreme Court",
            "[2020] UKSC 47",
            ["Lord Reed", "Lord Hodge", "Lord Lloyd-Jones"]
        )
    ];

    console.log("=== FCL Legal Judgment B-Tree Demo ===\n");

    // Insert all judgments
    console.log("Inserting legal judgments into B-tree...");
    for (const judgment of sampleJudgments) {
        legalBTree.insert(judgment);
    }

    console.log(`Successfully inserted ${legalBTree.totalJudgments} judgments\n`);

    // Display database statistics
    console.log("=== Database Statistics ===");
    const stats = legalBTree.getStatistics();
    console.log(`Total cases: ${stats.totalCases}`);
    console.log(`Date range: ${stats.dateRange}`);
    console.log(`Earliest case: ${stats.earliestCase.toISOString().split('T')[0]}`);
    console.log(`Latest case: ${stats.latestCase.toISOString().split('T')[0]}`);

    console.log("\nCourt distribution:");
    for (const [court, count] of Object.entries(stats.courtDistribution).sort()) {
        console.log(`  ${court}: ${count} cases`);
    }

    console.log("\nYearly distribution (top 5):");
    const yearlySorted = Object.entries(stats.yearlyDistribution)
        .sort((a, b) => b[1] - a[1])
        .slice(0, 5);
    for (const [year, count] of yearlySorted) {
        console.log(`  ${year}: ${count} cases`);
    }

    // Demonstrate specific date search
    console.log("\n=== Specific Date Search ===");
    const searchDate = "1932-05-26";
    const results = legalBTree.searchByDate(searchDate);
    console.log(`Cases on ${searchDate}:`);
    for (const judgment of results) {
        console.log(`  - ${judgment.caseName} (${judgment.citation})`);
    }

    // Demonstrate date range queries
    console.log("\n=== Date Range Queries ===");

    // 20th century landmark cases
    console.log("\nLandmark cases from 1900-1999:");
    const centuryCases = legalBTree.rangeQuery("1900-01-01", "1999-12-31");
    for (const judgment of centuryCases.slice(0, 5)) { // Show first 5
        const dateStr = judgment.judgmentDate.toISOString().split('T')[0];
        console.log(`  - ${dateStr}: ${judgment.caseName}`);
        console.log(`    ${judgment.citation} (${judgment.court})`);
    }

    // Recent Supreme Court cases
    console.log("\nRecent Supreme Court cases (2010 onwards):");
    const recentCases = legalBTree.rangeQuery("2010-01-01", "2025-12-31");
    for (const judgment of recentCases) {
        const dateStr = judgment.judgmentDate.toISOString().split('T')[0];
        console.log(`  - ${dateStr}: ${judgment.caseName}`);
        console.log(`    ${judgment.citation} (${judgment.court})`);
    }

    // House of Lords cases in specific decade
    console.log("\nHouse of Lords cases from 1990-1999:");
    const hl90s = legalBTree.rangeQuery("1990-01-01", "1999-12-31");
    const hlCases = hl90s.filter(j => j.court === "House of Lords");
    for (const judgment of hlCases) {
        const dateStr = judgment.judgmentDate.toISOString().split('T')[0];
        console.log(`  - ${dateStr}: ${judgment.caseName}`);
        const judgeList = judgment.judges.slice(0, 2).join(", ");
        const moreJudges = judgment.judges.length > 2 ? "..." : "";
        console.log(`    Judges: ${judgeList}${moreJudges}`);
    }

    // Chronological listing
    console.log("\n=== Chronological Case Listing (First 8) ===");
    const chronological = legalBTree.getChronologicalCases(8);
    for (let i = 0; i < chronological.length; i++) {
        const judgment = chronological[i];
        const dateStr = judgment.judgmentDate.toISOString().split('T')[0];
        console.log(`${(i + 1).toString().padStart(2)}. ${dateStr}: ${judgment.caseName}`);
        console.log(`     ${judgment.citation} - ${judgment.court}`);
        if (judgment.neutralCitation) {
            console.log(`     Neutral citation: ${judgment.neutralCitation}`);
        }
    }
}

// Export for Node.js environment
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { LegalJudgmentBTree, BTreeNode, LegalJudgment, demoFCLLegalBTree };
}

// Run demo if called directly
if (typeof require !== 'undefined' && require.main === module) {
    demoFCLLegalBTree();
}