/**
 * Set Operations for FCL Document Filtering
 *
 * JavaScript implementation of set operations for filtering legal documents.
 * Used for complex search queries combining multiple criteria.
 */

class JudgmentFilterSet {
    /**
     * Set-based filtering system for UK legal judgments.
     * Supports complex boolean queries and faceted search.
     */
    constructor(name = '') {
        this.name = name;
        this.judgments = new Set();
    }

    add(judgmentId) {
        this.judgments.add(judgmentId);
        return this;
    }

    addAll(judgmentIds) {
        judgmentIds.forEach(id => this.judgments.add(id));
        return this;
    }

    size() {
        return this.judgments.size;
    }

    contains(judgmentId) {
        return this.judgments.has(judgmentId);
    }

    // Set intersection - judgments matching ALL criteria
    intersect(otherSet) {
        const result = new JudgmentFilterSet(`${this.name} AND ${otherSet.name}`);
        for (const judgment of this.judgments) {
            if (otherSet.contains(judgment)) {
                result.add(judgment);
            }
        }
        return result;
    }

    // Set union - judgments matching ANY criteria
    union(otherSet) {
        const result = new JudgmentFilterSet(`${this.name} OR ${otherSet.name}`);
        this.judgments.forEach(j => result.add(j));
        otherSet.judgments.forEach(j => result.add(j));
        return result;
    }

    // Set difference - judgments in this set but NOT in other
    difference(otherSet) {
        const result = new JudgmentFilterSet(`${this.name} NOT ${otherSet.name}`);
        for (const judgment of this.judgments) {
            if (!otherSet.contains(judgment)) {
                result.add(judgment);
            }
        }
        return result;
    }

    // Symmetric difference - judgments in either set but not both
    symmetricDifference(otherSet) {
        const result = new JudgmentFilterSet(`${this.name} XOR ${otherSet.name}`);
        // Add from this set not in other
        for (const judgment of this.judgments) {
            if (!otherSet.contains(judgment)) {
                result.add(judgment);
            }
        }
        // Add from other set not in this
        for (const judgment of otherSet.judgments) {
            if (!this.contains(judgment)) {
                result.add(judgment);
            }
        }
        return result;
    }

    toArray() {
        return Array.from(this.judgments);
    }
}

// Example: FCL judgment database with filtering
class FCLJudgmentDatabase {
    constructor() {
        this.judgments = new Map();
        this.indices = {
            byCourt: new Map(),
            byYear: new Map(),
            byJudge: new Map(),
            byKeyword: new Map(),
            byArea: new Map()
        };
    }

    addJudgment(id, metadata) {
        this.judgments.set(id, metadata);

        // Update indices
        this._addToIndex(this.indices.byCourt, metadata.court, id);
        this._addToIndex(this.indices.byYear, metadata.year, id);
        this._addToIndex(this.indices.byJudge, metadata.judge, id);
        metadata.keywords?.forEach(keyword => {
            this._addToIndex(this.indices.byKeyword, keyword, id);
        });
        this._addToIndex(this.indices.byArea, metadata.area, id);
    }

    _addToIndex(index, key, value) {
        if (!key) return;
        if (!index.has(key)) {
            index.set(key, new Set());
        }
        index.get(key).add(value);
    }

    // Get set of judgments matching a criterion
    getSet(indexName, value) {
        const set = new JudgmentFilterSet(`${indexName}:${value}`);
        const index = this.indices[indexName];
        if (index && index.has(value)) {
            set.addAll(Array.from(index.get(value)));
        }
        return set;
    }

    // Complex query using set operations
    complexQuery(query) {
        let result = null;

        for (const [operation, criterion] of query) {
            const currentSet = this.getSet(criterion.index, criterion.value);

            if (!result) {
                result = currentSet;
            } else {
                switch (operation) {
                    case 'AND':
                        result = result.intersect(currentSet);
                        break;
                    case 'OR':
                        result = result.union(currentSet);
                        break;
                    case 'NOT':
                        result = result.difference(currentSet);
                        break;
                }
            }
        }

        return result;
    }

    getJudgmentDetails(ids) {
        return ids.map(id => this.judgments.get(id)).filter(j => j);
    }
}

// Demo with FCL data
function demonstrateFCLSetOperations() {
    console.log("=== FCL Set-Based Document Filtering Demo ===\n");

    const db = new FCLJudgmentDatabase();

    // Add sample UK judgments
    const sampleJudgments = [
        {
            id: '[2023] UKSC 42',
            court: 'UK Supreme Court',
            year: 2023,
            judge: 'Lord Reed',
            keywords: ['criminal', 'appeal', 'sentencing'],
            area: 'Criminal Law',
            name: 'R v Smith'
        },
        {
            id: '[2023] UKSC 15',
            court: 'UK Supreme Court',
            year: 2023,
            judge: 'Lady Rose',
            keywords: ['contract', 'breach', 'damages'],
            area: 'Contract Law',
            name: 'ABC Ltd v XYZ Corp'
        },
        {
            id: '[2022] EWCA Civ 789',
            court: 'Court of Appeal',
            year: 2022,
            judge: 'Lord Justice Bean',
            keywords: ['negligence', 'damages', 'tort'],
            area: 'Tort Law',
            name: 'Johnson v Brown'
        },
        {
            id: '[2023] EWCA Crim 456',
            court: 'Court of Appeal',
            year: 2023,
            judge: 'Lord Justice Holroyde',
            keywords: ['criminal', 'fraud', 'conspiracy'],
            area: 'Criminal Law',
            name: 'R v Jones'
        },
        {
            id: '[2021] UKSC 8',
            court: 'UK Supreme Court',
            year: 2021,
            judge: 'Lord Reed',
            keywords: ['constitutional', 'judicial review'],
            area: 'Constitutional Law',
            name: 'Miller v Prime Minister'
        }
    ];

    // Populate database
    sampleJudgments.forEach(j => db.addJudgment(j.id, j));

    // Example 1: Simple intersection (AND)
    console.log("1. Supreme Court cases from 2023:");
    const supremeCourt = db.getSet('byCourt', 'UK Supreme Court');
    const year2023 = db.getSet('byYear', 2023);
    const supremeCourt2023 = supremeCourt.intersect(year2023);
    console.log(`   Found ${supremeCourt2023.size()} judgments:`, supremeCourt2023.toArray());

    // Example 2: Union (OR)
    console.log("\n2. Criminal OR Constitutional cases:");
    const criminal = db.getSet('byArea', 'Criminal Law');
    const constitutional = db.getSet('byArea', 'Constitutional Law');
    const criminalOrConstitutional = criminal.union(constitutional);
    console.log(`   Found ${criminalOrConstitutional.size()} judgments:`, criminalOrConstitutional.toArray());

    // Example 3: Difference (NOT)
    console.log("\n3. Supreme Court cases NOT from 2023:");
    const supremeCourtNot2023 = supremeCourt.difference(year2023);
    console.log(`   Found ${supremeCourtNot2023.size()} judgments:`, supremeCourtNot2023.toArray());

    // Example 4: Complex query
    console.log("\n4. Complex query: (Supreme Court AND 2023) OR (Criminal Law):");
    const complexResult = db.complexQuery([
        [null, { index: 'byCourt', value: 'UK Supreme Court' }],
        ['AND', { index: 'byYear', value: 2023 }],
        ['OR', { index: 'byArea', value: 'Criminal Law' }]
    ]);
    console.log(`   Found ${complexResult.size()} judgments:`, complexResult.toArray());

    // Example 5: Keyword-based filtering
    console.log("\n5. Cases involving 'criminal' AND 'appeal':");
    const criminalKeyword = db.getSet('byKeyword', 'criminal');
    const appealKeyword = db.getSet('byKeyword', 'appeal');
    const criminalAppeal = criminalKeyword.intersect(appealKeyword);
    console.log(`   Found ${criminalAppeal.size()} judgments:`, criminalAppeal.toArray());

    // Show details of filtered results
    console.log("\n6. Detailed results for Criminal Law cases:");
    const criminalDetails = db.getJudgmentDetails(criminal.toArray());
    criminalDetails.forEach(j => {
        console.log(`   - ${j.id}: ${j.name}`);
        console.log(`     Court: ${j.court}, Judge: ${j.judge}`);
    });

    // Performance metrics
    console.log("\n=== Performance Metrics ===");
    console.log(`Total judgments indexed: ${db.judgments.size}`);
    console.log(`Courts indexed: ${db.indices.byCourt.size}`);
    console.log(`Keywords indexed: ${db.indices.byKeyword.size}`);
    console.log(`Legal areas covered: ${db.indices.byArea.size}`);
}

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        JudgmentFilterSet,
        FCLJudgmentDatabase
    };
}

// Run demo if executed directly
if (typeof require !== 'undefined' && require.main === module) {
    demonstrateFCLSetOperations();
}