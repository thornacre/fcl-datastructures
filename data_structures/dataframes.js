/**
 * DataFrame-like Data Structure for FCL Legal Data Analysis
 *
 * This implementation provides tabular data analysis capabilities for legal case data,
 * optimized for UK legal database operations and statistical analysis.
 *
 * Use cases:
 * - Statistical analysis of legal judgments
 * - Citation pattern analysis
 * - Court performance metrics
 * - Legal trend analysis over time
 * - Comparative case law studies
 */

class LegalDataFrameAnalyzer {
    /**
     * DataFrame-like wrapper specialized for legal case data analysis.
     * Provides FCL-specific analytical methods for UK legal databases.
     */
    constructor(data = []) {
        this.data = [...data];
        if (this.data.length > 0) {
            this._preprocessLegalData();
        }
    }

    /**
     * Preprocess legal data for analysis.
     * @private
     */
    _preprocessLegalData() {
        this.data = this.data.map(row => {
            const processedRow = { ...row };

            // Convert date strings to Date objects
            if (processedRow.judgment_date) {
                processedRow.judgment_date = new Date(processedRow.judgment_date);
                processedRow.year = processedRow.judgment_date.getFullYear();
                processedRow.month = processedRow.judgment_date.getMonth() + 1;
                processedRow.decade = Math.floor(processedRow.year / 10) * 10;
            }

            // Create citation type
            if (processedRow.neutral_citation) {
                processedRow.citation_type = this._extractCitationType(processedRow.neutral_citation);
            }

            // Calculate case name length (proxy for case complexity)
            if (processedRow.case_name) {
                processedRow.case_name_length = processedRow.case_name.length;
            }

            // Count number of judges
            if (processedRow.judges && Array.isArray(processedRow.judges)) {
                processedRow.judge_count = processedRow.judges.length;
            } else {
                processedRow.judge_count = 0;
            }

            return processedRow;
        });
    }

    /**
     * Extract court type from neutral citation.
     * @private
     */
    _extractCitationType(citation) {
        if (!citation) return "Unknown";

        const citationUpper = citation.toUpperCase();
        if (citationUpper.includes("UKSC")) return "Supreme Court";
        if (citationUpper.includes("UKHL")) return "House of Lords";
        if (citationUpper.includes("EWCA")) return "Court of Appeal";
        if (citationUpper.includes("EWHC")) return "High Court";
        if (citationUpper.includes("UKUT")) return "Upper Tribunal";
        if (citationUpper.includes("EWCOP")) return "Court of Protection";
        return "Other";
    }

    /**
     * Add new cases to the analyzer.
     */
    addCases(cases) {
        this.data.push(...cases);
        this._preprocessLegalData();
    }

    /**
     * Get basic statistics about the legal dataset.
     */
    getBasicStatistics() {
        if (this.data.length === 0) {
            return { error: "No data available" };
        }

        // Calculate date range
        const dates = this.data
            .filter(row => row.judgment_date)
            .map(row => row.judgment_date);

        const earliest = dates.length > 0 ? new Date(Math.min(...dates)) : null;
        const latest = dates.length > 0 ? new Date(Math.max(...dates)) : null;

        // Court distribution
        const courtCounts = this._countValues('court');

        // Cases by decade
        const decadeCounts = this._countValues('decade');

        // Average judges per case
        const judgeCountSum = this.data.reduce((sum, row) => sum + (row.judge_count || 0), 0);
        const avgJudges = this.data.length > 0 ? judgeCountSum / this.data.length : 0;

        return {
            totalCases: this.data.length,
            dateRange: {
                earliest: earliest ? earliest.toISOString().split('T')[0] : null,
                latest: latest ? latest.toISOString().split('T')[0] : null
            },
            courtDistribution: courtCounts,
            casesByDecade: decadeCounts,
            averageJudgesPerCase: avgJudges
        };
    }

    /**
     * Analyze temporal trends in legal judgments.
     */
    analyzeTemporalTrends() {
        const hasTemporalData = this.data.some(row => row.year);
        if (!hasTemporalData) {
            return { error: "No temporal data available" };
        }

        // Cases per year
        const yearlyCounts = this._countValues('year');

        // Cases per month (seasonal patterns)
        const monthlyCounts = this._countValues('month');

        // Court activity over time
        const courtTrends = {};
        const courts = [...new Set(this.data.map(row => row.court).filter(Boolean))];

        for (const court of courts) {
            const courtData = this.data.filter(row => row.court === court);
            courtTrends[court] = this._countValues('year', courtData);
        }

        // Find busiest and quietest years
        const yearEntries = Object.entries(yearlyCounts);
        const busiestYear = yearEntries.length > 0 ?
            yearEntries.reduce((a, b) => a[1] > b[1] ? a : b)[0] : null;
        const quietestYear = yearEntries.length > 0 ?
            yearEntries.reduce((a, b) => a[1] < b[1] ? a : b)[0] : null;

        return {
            yearlyDistribution: yearlyCounts,
            monthlyDistribution: monthlyCounts,
            courtTrendsByYear: courtTrends,
            busiestYear: busiestYear ? parseInt(busiestYear) : null,
            quietestYear: quietestYear ? parseInt(quietestYear) : null
        };
    }

    /**
     * Analyze citation patterns and court hierarchies.
     */
    analyzeCitationPatterns() {
        const hasCitationData = this.data.some(row => row.citation_type);
        if (!hasCitationData) {
            return { error: "No citation data available" };
        }

        const citationTypeDistribution = this._countValues('citation_type');

        // Citation trends over time
        const citationTrendsByYear = {};
        const citationTypes = [...new Set(this.data.map(row => row.citation_type).filter(Boolean))];

        for (const citationType of citationTypes) {
            const typeData = this.data.filter(row => row.citation_type === citationType);
            citationTrendsByYear[citationType] = this._countValues('year', typeData);
        }

        return {
            citationTypeDistribution,
            citationTrendsByYear
        };
    }

    /**
     * Analyze patterns in judicial participation.
     */
    findJudicialPatterns() {
        const hasJudgeCounts = this.data.some(row => row.judge_count !== undefined);
        if (!hasJudgeCounts) {
            return { error: "No judicial data available" };
        }

        // Flatten judges list and count appearances
        const allJudges = [];
        for (const row of this.data) {
            if (Array.isArray(row.judges)) {
                allJudges.push(...row.judges);
            }
        }

        const judgeCounter = this._countArrayValues(allJudges);
        const mostActiveJudges = Object.entries(judgeCounter)
            .sort((a, b) => b[1] - a[1])
            .slice(0, 10)
            .reduce((obj, [judge, count]) => {
                obj[judge] = count;
                return obj;
            }, {});

        // Panel size analysis
        const judgeCounts = this.data.map(row => row.judge_count || 0);
        const avgPanelSize = judgeCounts.length > 0 ?
            judgeCounts.reduce((sum, count) => sum + count, 0) / judgeCounts.length : 0;

        const panelSizeDistribution = this._countValues('judge_count');

        return {
            mostActiveJudges,
            averagePanelSize: avgPanelSize,
            panelSizeDistribution
        };
    }

    /**
     * Compare different courts' characteristics.
     */
    compareCourts() {
        const hasCourts = this.data.some(row => row.court);
        if (!hasCourts) {
            return { error: "No court data available" };
        }

        const courtComparison = {};
        const courts = [...new Set(this.data.map(row => row.court).filter(Boolean))];

        for (const court of courts) {
            const courtData = this.data.filter(row => row.court === court);

            // Calculate date range for this court
            const courtDates = courtData
                .filter(row => row.judgment_date)
                .map(row => row.judgment_date);

            const earliest = courtDates.length > 0 ? new Date(Math.min(...courtDates)) : null;
            const latest = courtDates.length > 0 ? new Date(Math.max(...courtDates)) : null;

            // Calculate averages
            const avgPanelSize = courtData.length > 0 ?
                courtData.reduce((sum, row) => sum + (row.judge_count || 0), 0) / courtData.length : 0;

            const avgComplexity = courtData.length > 0 ?
                courtData.reduce((sum, row) => sum + (row.case_name_length || 0), 0) / courtData.length : 0;

            courtComparison[court] = {
                totalCases: courtData.length,
                dateRange: {
                    earliest: earliest ? earliest.toISOString().split('T')[0] : null,
                    latest: latest ? latest.toISOString().split('T')[0] : null
                },
                averagePanelSize: avgPanelSize,
                caseComplexityProxy: avgComplexity
            };
        }

        return courtComparison;
    }

    /**
     * Search cases based on multiple criteria.
     */
    searchCases(criteria = {}) {
        let filteredData = [...this.data];

        if (criteria.court) {
            filteredData = filteredData.filter(row => row.court === criteria.court);
        }

        if (criteria.yearFrom && typeof criteria.yearFrom === 'number') {
            filteredData = filteredData.filter(row => row.year >= criteria.yearFrom);
        }

        if (criteria.yearTo && typeof criteria.yearTo === 'number') {
            filteredData = filteredData.filter(row => row.year <= criteria.yearTo);
        }

        if (criteria.caseNameContains) {
            const searchTerm = criteria.caseNameContains.toLowerCase();
            filteredData = filteredData.filter(row =>
                row.case_name && row.case_name.toLowerCase().includes(searchTerm)
            );
        }

        if (criteria.judgeName) {
            filteredData = filteredData.filter(row =>
                Array.isArray(row.judges) && row.judges.includes(criteria.judgeName)
            );
        }

        return filteredData;
    }

    /**
     * Export comprehensive analysis report to HTML.
     */
    exportAnalysisReport(filename = "fcl_legal_analysis.html") {
        if (this.data.length === 0) {
            console.log("No data to analyze");
            return null;
        }

        const basicStats = this.getBasicStatistics();
        const temporalTrends = this.analyzeTemporalTrends();
        const citationPatterns = this.analyzeCitationPatterns();
        const judicialPatterns = this.findJudicialPatterns();
        const courtComparison = this.compareCourts();

        const htmlContent = `
<!DOCTYPE html>
<html>
<head>
    <title>FCL Legal Database Analysis Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        h1, h2 { color: #2c3e50; }
        .stat-section { margin: 20px 0; padding: 15px; border-left: 4px solid #3498db; }
        .court { background-color: #f8f9fa; padding: 10px; margin: 10px 0; }
        table { border-collapse: collapse; width: 100%; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #f2f2f2; }
    </style>
</head>
<body>
    <h1>FCL Legal Database Analysis Report</h1>
    <p>Generated on: ${new Date().toISOString().split('T')[0]}</p>

    <div class="stat-section">
        <h2>Basic Statistics</h2>
        <p><strong>Total Cases:</strong> ${basicStats.totalCases}</p>
        <p><strong>Date Range:</strong> ${basicStats.dateRange.earliest} to ${basicStats.dateRange.latest}</p>
        <p><strong>Average Judges per Case:</strong> ${basicStats.averageJudgesPerCase.toFixed(2)}</p>
    </div>

    <div class="stat-section">
        <h2>Court Distribution</h2>
        <table>
            <tr><th>Court</th><th>Number of Cases</th></tr>
            ${Object.entries(basicStats.courtDistribution)
                .sort()
                .map(([court, count]) => `<tr><td>${court}</td><td>${count}</td></tr>`)
                .join('')}
        </table>
    </div>
</body>
</html>`;

        // In a browser environment, this would trigger a download
        // In Node.js, you'd write to filesystem
        console.log(`Analysis report generated (${htmlContent.length} characters)`);
        return htmlContent;
    }

    /**
     * Helper method to count values for a specific field.
     * @private
     */
    _countValues(field, data = null) {
        const targetData = data || this.data;
        const counts = {};

        for (const row of targetData) {
            const value = row[field];
            if (value !== undefined && value !== null) {
                counts[value] = (counts[value] || 0) + 1;
            }
        }

        return counts;
    }

    /**
     * Helper method to count values in an array.
     * @private
     */
    _countArrayValues(array) {
        const counts = {};
        for (const item of array) {
            counts[item] = (counts[item] || 0) + 1;
        }
        return counts;
    }

    /**
     * Get the raw data array.
     */
    getData() {
        return [...this.data];
    }

    /**
     * Get a subset of data with specific columns.
     */
    selectColumns(columns) {
        return this.data.map(row => {
            const selectedRow = {};
            for (const col of columns) {
                if (row.hasOwnProperty(col)) {
                    selectedRow[col] = row[col];
                }
            }
            return selectedRow;
        });
    }

    /**
     * Get basic DataFrame-like info.
     */
    info() {
        if (this.data.length === 0) {
            return { shape: [0, 0], columns: [] };
        }

        const columns = Object.keys(this.data[0]);
        return {
            shape: [this.data.length, columns.length],
            columns: columns
        };
    }
}

/**
 * Demonstration of DataFrame analysis with realistic FCL legal data.
 */
function demoFCLLegalDataFrames() {
    // Sample UK legal cases data
    const sampleLegalData = [
        {
            case_name: "Donoghue v Stevenson",
            citation: "[1932] AC 562",
            neutral_citation: "[1932] UKHL 100",
            judgment_date: "1932-05-26",
            court: "House of Lords",
            judges: ["Lord Atkin", "Lord Thankerton", "Lord Macmillan"],
            legal_area: "Tort Law",
            significance: "Established modern negligence law"
        },
        {
            case_name: "Carlill v Carbolic Smoke Ball Company",
            citation: "[1893] 1 QB 256",
            neutral_citation: "[1892] EWCA Civ 1",
            judgment_date: "1892-12-07",
            court: "Court of Appeal",
            judges: ["Lindley LJ", "Bowen LJ", "A.L. Smith LJ"],
            legal_area: "Contract Law",
            significance: "Unilateral contract formation"
        },
        {
            case_name: "Rylands v Fletcher",
            citation: "(1868) LR 3 HL 330",
            neutral_citation: "[1868] UKHL 1",
            judgment_date: "1868-07-17",
            court: "House of Lords",
            judges: ["Lord Cairns", "Lord Cranworth"],
            legal_area: "Tort Law",
            significance: "Strict liability for dangerous activities"
        },
        {
            case_name: "R v Brown",
            citation: "[1994] 1 AC 212",
            neutral_citation: "[1993] UKHL 19",
            judgment_date: "1993-03-11",
            court: "House of Lords",
            judges: ["Lord Templeman", "Lord Jauncey", "Lord Lowry"],
            legal_area: "Criminal Law",
            significance: "Consent in assault cases"
        },
        {
            case_name: "Pepper v Hart",
            citation: "[1993] AC 593",
            neutral_citation: "[1992] UKHL 3",
            judgment_date: "1992-11-26",
            court: "House of Lords",
            judges: ["Lord Griffiths", "Lord Ackner", "Lord Oliver"],
            legal_area: "Constitutional Law",
            significance: "Use of Hansard in statutory interpretation"
        },
        {
            case_name: "R (Miller) v The Prime Minister",
            citation: "[2019] UKSC 41",
            neutral_citation: "[2019] UKSC 41",
            judgment_date: "2019-09-24",
            court: "Supreme Court",
            judges: ["Lady Hale", "Lord Reed", "Lord Kerr", "Lord Wilson", "Lord Carnwath"],
            legal_area: "Constitutional Law",
            significance: "Limits on executive power"
        }
    ];

    console.log("=== FCL Legal DataFrame Analysis Demo ===\n");

    // Initialize analyzer with sample data
    const analyzer = new LegalDataFrameAnalyzer(sampleLegalData);

    console.log("Loaded legal case database with realistic UK cases\n");

    // Basic statistics
    console.log("=== Basic Database Statistics ===");
    const basicStats = analyzer.getBasicStatistics();
    console.log(`Total cases: ${basicStats.totalCases}`);
    console.log(`Date range: ${basicStats.dateRange.earliest} to ${basicStats.dateRange.latest}`);
    console.log(`Average judges per case: ${basicStats.averageJudgesPerCase.toFixed(2)}`);

    console.log("\nCases by decade:");
    for (const [decade, count] of Object.entries(basicStats.casesByDecade).sort()) {
        console.log(`  ${decade}s: ${count} cases`);
    }

    console.log("\nCourt distribution:");
    for (const [court, count] of Object.entries(basicStats.courtDistribution).sort()) {
        console.log(`  ${court}: ${count} cases`);
    }

    // Temporal analysis
    console.log("\n=== Temporal Trends Analysis ===");
    const temporalAnalysis = analyzer.analyzeTemporalTrends();
    console.log(`Busiest year: ${temporalAnalysis.busiestYear}`);
    console.log(`Quietest year: ${temporalAnalysis.quietestYear}`);

    console.log("\nMonthly distribution (seasonal patterns):");
    const monthNames = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                       'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
    for (const [month, count] of Object.entries(temporalAnalysis.monthlyDistribution).sort()) {
        const monthIndex = parseInt(month) - 1;
        console.log(`  ${monthNames[monthIndex]}: ${count} cases`);
    }

    // Citation pattern analysis
    console.log("\n=== Citation Pattern Analysis ===");
    const citationAnalysis = analyzer.analyzeCitationPatterns();
    console.log("Citation type distribution:");
    for (const [ctype, count] of Object.entries(citationAnalysis.citationTypeDistribution).sort()) {
        console.log(`  ${ctype}: ${count} cases`);
    }

    // Judicial patterns
    console.log("\n=== Judicial Participation Analysis ===");
    const judicialAnalysis = analyzer.findJudicialPatterns();
    console.log(`Average panel size: ${judicialAnalysis.averagePanelSize.toFixed(2)} judges`);

    console.log("\nMost active judges (top 5):");
    const judgeEntries = Object.entries(judicialAnalysis.mostActiveJudges).slice(0, 5);
    for (const [judge, count] of judgeEntries) {
        console.log(`  ${judge}: ${count} cases`);
    }

    console.log("\nPanel size distribution:");
    for (const [size, count] of Object.entries(judicialAnalysis.panelSizeDistribution).sort()) {
        const judgeWord = size === "1" ? "judge" : "judges";
        console.log(`  ${size} ${judgeWord}: ${count} cases`);
    }

    // Court comparison
    console.log("\n=== Court Comparison Analysis ===");
    const courtComparison = analyzer.compareCourts();
    for (const [court, stats] of Object.entries(courtComparison)) {
        console.log(`\n${court}:`);
        console.log(`  Total cases: ${stats.totalCases}`);
        console.log(`  Date range: ${stats.dateRange.earliest} to ${stats.dateRange.latest}`);
        console.log(`  Average panel size: ${stats.averagePanelSize.toFixed(2)}`);
        console.log(`  Case complexity proxy: ${stats.caseComplexityProxy.toFixed(1)} chars`);
    }

    // Search demonstrations
    console.log("\n=== Search and Filter Examples ===");

    // Search by court
    const houseOfLordsCases = analyzer.searchCases({ court: "House of Lords" });
    console.log(`\nHouse of Lords cases: ${houseOfLordsCases.length}`);
    for (const case_ of houseOfLordsCases.slice(0, 3)) {
        console.log(`  - ${case_.case_name} (${case_.year})`);
    }

    // Search by date range
    const modernCases = analyzer.searchCases({ yearFrom: 1990 });
    console.log(`\nCases from 1990 onwards: ${modernCases.length}`);
    for (const case_ of modernCases.slice(0, 3)) {
        console.log(`  - ${case_.case_name} (${case_.year})`);
    }

    // Search by judge
    const lordAckerCases = analyzer.searchCases({ judgeName: "Lord Ackner" });
    console.log(`\nCases involving Lord Ackner: ${lordAckerCases.length}`);
    for (const case_ of lordAckerCases) {
        console.log(`  - ${case_.case_name} (${case_.year})`);
    }

    console.log("\n=== DataFrame Operations Demo ===");
    const info = analyzer.info();
    console.log(`Raw data shape: [${info.shape[0]}, ${info.shape[1]}]`);
    console.log(`Columns: ${info.columns.join(', ')}`);

    console.log("\nFirst few rows (selected columns):");
    const selectedData = analyzer.selectColumns(['case_name', 'court', 'year', 'judge_count']);
    for (const row of selectedData.slice(0, 3)) {
        console.log(`  ${row.case_name} | ${row.court} | ${row.year} | ${row.judge_count} judges`);
    }
}

// Export for Node.js environment
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { LegalDataFrameAnalyzer, demoFCLLegalDataFrames };
}

// Run demo if called directly
if (typeof require !== 'undefined' && require.main === module) {
    demoFCLLegalDataFrames();
}