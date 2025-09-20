/**
 * Regex Patterns for Legal Reference Extraction (JavaScript)
 * ==========================================================
 */

class LegalTextExtractor {
    constructor() {
        this.neutralPattern = /\[(\d{4})\]\s+(UKSC|UKHL|EWCA|EWHC|UKUT|UKFTT)\s+(\d+)/gi;
        this.lawReportPattern = /\[(\d{4})\]\s+(\d+)\s+(WLR|All\s*ER|AC|Ch|QB)\s+(\d+)/gi;
        this.caseNamePattern = /(R\s*(?:\([^)]+\))?|[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+v\.?\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)/gi;
        this.judgePattern = /(Lord|Lady|Sir|Dame|Mr|Mrs|Ms)\s+(Justice\s+)?([A-Z][a-z]+)/gi;
        this.statutePattern = /(Section|s\.)\s+(\d+[A-Z]?)\s+of\s+the\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+Act\s+\d{4})/gi;
    }

    extractCitations(text) {
        const citations = [];
        let match;

        // Extract neutral citations
        while ((match = this.neutralPattern.exec(text)) !== null) {
            citations.push({
                type: 'neutral',
                fullText: match[0],
                year: parseInt(match[1]),
                court: match[2],
                number: match[3],
                position: match.index
            });
        }

        // Extract law reports
        this.lawReportPattern.lastIndex = 0;
        while ((match = this.lawReportPattern.exec(text)) !== null) {
            citations.push({
                type: 'law_report',
                fullText: match[0],
                year: parseInt(match[1]),
                volume: match[2],
                series: match[3],
                page: match[4],
                position: match.index
            });
        }

        return citations.sort((a, b) => a.position - b.position);
    }

    extractCaseNames(text) {
        const names = [];
        let match;

        this.caseNamePattern.lastIndex = 0;
        while ((match = this.caseNamePattern.exec(text)) !== null) {
            names.push({
                fullText: match[0],
                applicant: match[1],
                respondent: match[2],
                position: match.index
            });
        }

        return names;
    }

    extractJudges(text) {
        const judges = [];
        let match;

        this.judgePattern.lastIndex = 0;
        while ((match = this.judgePattern.exec(text)) !== null) {
            judges.push({
                fullText: match[0],
                title: match[1],
                name: match[3],
                position: match.index
            });
        }

        return judges;
    }

    validateCitation(citation) {
        const neutralMatch = citation.match(/\[(\d{4})\]\s+(UKSC|UKHL|EWCA|EWHC)\s+(\d+)/);

        if (neutralMatch) {
            const year = parseInt(neutralMatch[1]);
            const court = neutralMatch[2];
            const number = neutralMatch[3];

            const errors = [];
            if (year < 1990 || year > 2030) {
                errors.push(`Year ${year} outside valid range`);
            }

            const validCourts = ['UKSC', 'UKHL', 'EWCA', 'EWHC'];
            if (!validCourts.includes(court)) {
                errors.push(`Invalid court: ${court}`);
            }

            return {
                valid: errors.length === 0,
                normalized: `[${year}] ${court} ${number}`,
                errors: errors
            };
        }

        return {
            valid: false,
            normalized: citation,
            errors: ['Unrecognized citation format']
        };
    }
}

function demonstrateRegexPatterns() {
    console.log("=== Legal Regex Demo (JavaScript) ===");

    const text = `
    In R v Smith [2023] UKSC 15, Lord Reed considered the principles from
    [2019] UKSC 41. The case established that Section 4 of the Human Rights Act 1998
    applies. Lady Hale delivered the judgment in Miller v Secretary of State [2022] EWCA Civ 234.
    `;

    const extractor = new LegalTextExtractor();

    console.log("\n1. Citations found:");
    const citations = extractor.extractCitations(text);
    citations.forEach((cite, i) => {
        console.log(`   ${i+1}. ${cite.fullText} (${cite.type})`);
    });

    console.log("\n2. Case names found:");
    const names = extractor.extractCaseNames(text);
    names.forEach((name, i) => {
        console.log(`   ${i+1}. ${name.fullText}`);
    });

    console.log("\n3. Judges found:");
    const judges = extractor.extractJudges(text);
    judges.forEach((judge, i) => {
        console.log(`   ${i+1}. ${judge.title} ${judge.name}`);
    });

    console.log("\n4. Citation validation:");
    const testCites = ["[2023] UKSC 15", "[2023] BADCOURT 99"];
    testCites.forEach(cite => {
        const result = extractor.validateCitation(cite);
        console.log(`   ${cite}: ${result.valid ? 'Valid' : 'Invalid'}`);
        if (result.errors.length > 0) {
            console.log(`      Errors: ${result.errors.join(', ')}`);
        }
    });

    return { extractor, citations, names, judges };
}

if (typeof module !== 'undefined' && module.exports) {
    module.exports = { LegalTextExtractor, demonstrateRegexPatterns };
} else if (typeof window === 'undefined') {
    demonstrateRegexPatterns();
}