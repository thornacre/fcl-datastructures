class CitationGraph {
    /**
     * Graph structure for tracking case law citations.
     * Each node is a judgment, edges represent citations.
     */
    constructor() {
        this.adjacencyList = {};  // {caseId: [citedCaseIds]}
        this.metadata = {};  // {caseId: {court, date, name}}
    }

    addJudgment(caseId, name, court, date) {
        if (!(caseId in this.adjacencyList)) {
            this.adjacencyList[caseId] = [];
        }
        this.metadata[caseId] = {
            name: name,
            court: court,
            date: date
        };
    }

    addCitation(citingCase, citedCase) {
        if (citingCase in this.adjacencyList) {
            this.adjacencyList[citingCase].push(citedCase);
        }
    }

    findMostCited(topN = 5) {
        // Find the most influential cases by citation count
        const citationCounts = {};
        for (const citations of Object.values(this.adjacencyList)) {
            for (const cited of citations) {
                citationCounts[cited] = (citationCounts[cited] || 0) + 1;
            }
        }

        return Object.entries(citationCounts)
            .sort((a, b) => b[1] - a[1])
            .slice(0, topN);
    }

    findCitationChain(startCase, targetCase, visited = new Set()) {
        // Find citation path between two cases
        if (startCase === targetCase) {
            return [startCase];
        }

        visited.add(startCase);

        for (const cited of (this.adjacencyList[startCase] || [])) {
            if (!visited.has(cited)) {
                const path = this.findCitationChain(cited, targetCase, visited);
                if (path) {
                    return [startCase, ...path];
                }
            }
        }

        return null;
    }
}

// Example: UK case law citation network
const graph = new CitationGraph();

// Add landmark UK cases
graph.addJudgment('[2023] UKSC 42', 'R v Smith', 'UK Supreme Court', '2023-11-15');
graph.addJudgment('[2016] UKSC 8', 'R v Jogee', 'UK Supreme Court', '2016-02-18');
graph.addJudgment('[1932] AC 562', 'Donoghue v Stevenson', 'House of Lords', '1932-05-26');
graph.addJudgment('[2020] EWCA Civ 123', 'Jones v Brown', 'Court of Appeal', '2020-03-10');
graph.addJudgment('[2019] EWHC 456', 'Re ABC Ltd', 'High Court', '2019-07-22');

// Add citations (newer cases cite older ones)
graph.addCitation('[2023] UKSC 42', '[2016] UKSC 8');
graph.addCitation('[2023] UKSC 42', '[1932] AC 562');
graph.addCitation('[2020] EWCA Civ 123', '[1932] AC 562');
graph.addCitation('[2019] EWHC 456', '[1932] AC 562');
graph.addCitation('[2016] UKSC 8', '[1932] AC 562');

// Find most influential cases
console.log("Most cited cases:");
for (const [caseId, count] of graph.findMostCited(3)) {
    const metadata = graph.metadata[caseId] || {};
    console.log(`  ${caseId}: ${metadata.name || 'Unknown'} - ${count} citations`);
}

// Find citation path
const path = graph.findCitationChain('[2023] UKSC 42', '[1932] AC 562');
if (path) {
    console.log("\nCitation path from R v Smith to Donoghue v Stevenson:");
    console.log(path.join(" -> "));
}