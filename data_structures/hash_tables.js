class JudgmentMetadataIndex {
    /**
     * Hash table for O(1) access to judgment metadata.
     * Used for quick lookups by neutral citation.
     */
    constructor() {
        this.index = {};  // Hash table for metadata
        this.courtIndex = {};  // Secondary index by court
    }

    addJudgment(neutralCitation, metadata) {
        // Add judgment with O(1) insertion
        this.index[neutralCitation] = metadata;

        // Update court index
        const court = metadata.court;
        if (court) {
            if (!(court in this.courtIndex)) {
                this.courtIndex[court] = [];
            }
            this.courtIndex[court].push(neutralCitation);
        }
    }

    getJudgment(neutralCitation) {
        // O(1) lookup by neutral citation
        return this.index[neutralCitation];
    }

    getByCourt(court) {
        // Get all judgments from a specific court
        const citations = this.courtIndex[court] || [];
        return citations.map(citation => this.index[citation]);
    }

    searchByYear(year) {
        // Linear search - demonstrates when hash tables aren't optimal
        const results = [];
        for (const [citation, metadata] of Object.entries(this.index)) {
            if (metadata.date && metadata.date.startsWith(String(year))) {
                results.push([citation, metadata]);
            }
        }
        return results;
    }
}

// Example: FCL judgment metadata index
const index = new JudgmentMetadataIndex();

// Add UK judgments with metadata
const judgments = [
    ['[2023] UKSC 42', {
        name: 'R v Smith',
        court: 'UK Supreme Court',
        date: '2023-11-15',
        keywords: ['criminal', 'appeal', 'sentencing'],
        judge: 'Lord Reed',
        fileUrl: '/judgments/uksc/2023/42.xml'
    }],
    ['[2023] EWCA Crim 789', {
        name: 'R v Johnson',
        court: 'Court of Appeal Criminal Division',
        date: '2023-06-20',
        keywords: ['fraud', 'conspiracy'],
        judge: 'Lord Justice Holroyde',
        fileUrl: '/judgments/ewca/crim/2023/789.xml'
    }],
    ['[2023] EWHC 1234 (Ch)', {
        name: 'Re XYZ Company Ltd',
        court: 'High Court Chancery Division',
        date: '2023-04-10',
        keywords: ['insolvency', 'directors duties'],
        judge: 'Mr Justice Zacaroli',
        fileUrl: '/judgments/ewhc/ch/2023/1234.xml'
    }]
];

// Populate the index
for (const [citation, metadata] of judgments) {
    index.addJudgment(citation, metadata);
}

// O(1) lookup example
console.log("Direct lookup [2023] UKSC 42:");
const judgment = index.getJudgment('[2023] UKSC 42');
if (judgment) {
    console.log(`  Name: ${judgment.name}`);
    console.log(`  Court: ${judgment.court}`);
    console.log(`  Keywords: ${judgment.keywords.join(', ')}`);
}

// Get all Supreme Court cases
console.log("\nAll UK Supreme Court cases:");
for (const judgment of index.getByCourt('UK Supreme Court')) {
    console.log(`  - ${judgment.name} (${judgment.date})`);
}

// Search by year (demonstrates linear search)
console.log("\nAll 2023 judgments:");
for (const [citation, metadata] of index.searchByYear(2023)) {
    console.log(`  ${citation}: ${metadata.name}`);
}