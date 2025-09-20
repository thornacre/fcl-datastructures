/**
 * JavaScript Arrays for Legal Data Numerical Analysis
 * ==================================================
 *
 * This module demonstrates numerical analysis using JavaScript arrays for legal
 * data processing in Find Case Law (FCL). Provides client-side mathematical
 * operations for statistical analysis and data visualization.
 *
 * Key FCL Use Cases:
 * - Case similarity scoring using vector operations
 * - Statistical analysis of judgment trends
 * - Court performance metrics calculation
 * - Citation network analysis
 * - Search relevance scoring
 * - Document clustering and classification
 */

/**
 * Represents similarity between two legal cases
 */
class CaseSimilarity {
    constructor({
        case1Citation = '',
        case2Citation = '',
        similarityScore = 0.0,
        matchingFeatures = [],
        vectorDistance = 0.0
    } = {}) {
        this.case1Citation = case1Citation;
        this.case2Citation = case2Citation;
        this.similarityScore = similarityScore;
        this.matchingFeatures = matchingFeatures;
        this.vectorDistance = vectorDistance;
    }
}

/**
 * Statistical data for a court
 */
class CourtStatistics {
    constructor({
        courtName = '',
        totalCases = 0,
        avgCaseLength = 0.0,
        citationFrequency = 0.0,
        subjectAreas = {},
        temporalTrends = []
    } = {}) {
        this.courtName = courtName;
        this.totalCases = totalCases;
        this.avgCaseLength = avgCaseLength;
        this.citationFrequency = citationFrequency;
        this.subjectAreas = subjectAreas;
        this.temporalTrends = temporalTrends;
    }
}

/**
 * Mathematical utility functions for array operations
 */
class MathUtils {
    /**
     * Calculate dot product of two vectors
     */
    static dotProduct(a, b) {
        if (a.length !== b.length) return 0;
        return a.reduce((sum, val, i) => sum + val * b[i], 0);
    }

    /**
     * Calculate vector magnitude
     */
    static magnitude(vector) {
        return Math.sqrt(vector.reduce((sum, val) => sum + val * val, 0));
    }

    /**
     * Calculate cosine similarity between two vectors
     */
    static cosineSimilarity(a, b) {
        const dot = this.dotProduct(a, b);
        const magA = this.magnitude(a);
        const magB = this.magnitude(b);
        return magA && magB ? dot / (magA * magB) : 0;
    }

    /**
     * Calculate Euclidean distance between two vectors
     */
    static euclideanDistance(a, b) {
        if (a.length !== b.length) return Infinity;
        return Math.sqrt(a.reduce((sum, val, i) => sum + Math.pow(val - b[i], 2), 0));
    }

    /**
     * Calculate mean of array
     */
    static mean(arr) {
        return arr.length > 0 ? arr.reduce((sum, val) => sum + val, 0) / arr.length : 0;
    }

    /**
     * Calculate standard deviation
     */
    static standardDeviation(arr) {
        const avg = this.mean(arr);
        const squareDiffs = arr.map(val => Math.pow(val - avg, 2));
        return Math.sqrt(this.mean(squareDiffs));
    }

    /**
     * Calculate correlation coefficient
     */
    static correlation(x, y) {
        if (x.length !== y.length || x.length === 0) return 0;

        const meanX = this.mean(x);
        const meanY = this.mean(y);

        const numerator = x.reduce((sum, val, i) => sum + (val - meanX) * (y[i] - meanY), 0);
        const denomX = Math.sqrt(x.reduce((sum, val) => sum + Math.pow(val - meanX, 2), 0));
        const denomY = Math.sqrt(y.reduce((sum, val) => sum + Math.pow(val - meanY, 2), 0));

        return denomX && denomY ? numerator / (denomX * denomY) : 0;
    }

    /**
     * Calculate linear regression slope and intercept
     */
    static linearRegression(x, y) {
        if (x.length !== y.length || x.length === 0) return { slope: 0, intercept: 0 };

        const n = x.length;
        const sumX = x.reduce((sum, val) => sum + val, 0);
        const sumY = y.reduce((sum, val) => sum + val, 0);
        const sumXY = x.reduce((sum, val, i) => sum + val * y[i], 0);
        const sumXX = x.reduce((sum, val) => sum + val * val, 0);

        const slope = (n * sumXY - sumX * sumY) / (n * sumXX - sumX * sumX);
        const intercept = (sumY - slope * sumX) / n;

        return { slope, intercept };
    }
}

/**
 * TF-IDF Vectorizer for text similarity analysis
 */
class TFIDFVectorizer {
    constructor(options = {}) {
        this.maxFeatures = options.maxFeatures || 1000;
        this.vocabulary = new Map();
        this.idfScores = new Map();
        this.stopWords = new Set([
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those'
        ]);
    }

    /**
     * Tokenize and clean text
     */
    tokenize(text) {
        return text.toLowerCase()
            .replace(/[^\w\s]/g, ' ')
            .split(/\s+/)
            .filter(word => word.length > 2 && !this.stopWords.has(word));
    }

    /**
     * Build vocabulary from documents
     */
    fitTransform(documents) {
        // Build vocabulary
        const allTerms = new Set();
        const docTerms = documents.map(doc => {
            const terms = this.tokenize(doc);
            terms.forEach(term => allTerms.add(term));
            return terms;
        });

        // Limit vocabulary size
        const sortedTerms = Array.from(allTerms).slice(0, this.maxFeatures);
        this.vocabulary = new Map(sortedTerms.map((term, i) => [term, i]));

        // Calculate IDF scores
        const docCount = documents.length;
        for (const term of sortedTerms) {
            const docFreq = docTerms.filter(terms => terms.includes(term)).length;
            this.idfScores.set(term, Math.log(docCount / (1 + docFreq)));
        }

        // Transform documents to TF-IDF vectors
        return docTerms.map(terms => this.transformDocument(terms));
    }

    /**
     * Transform a single document to TF-IDF vector
     */
    transformDocument(terms) {
        const vector = new Array(this.vocabulary.size).fill(0);
        const termFreq = new Map();

        // Calculate term frequencies
        terms.forEach(term => {
            if (this.vocabulary.has(term)) {
                termFreq.set(term, (termFreq.get(term) || 0) + 1);
            }
        });

        // Calculate TF-IDF scores
        for (const [term, freq] of termFreq) {
            if (this.vocabulary.has(term)) {
                const index = this.vocabulary.get(term);
                const tf = freq / terms.length;
                const idf = this.idfScores.get(term) || 0;
                vector[index] = tf * idf;
            }
        }

        return vector;
    }
}

/**
 * Numerical analysis tools for legal document processing
 */
class LegalDataAnalyzer {
    constructor() {
        this.vectorizer = new TFIDFVectorizer({ maxFeatures: 1000 });
        this.caseEmbeddings = new Map();
        this.courtStats = new Map();
    }

    /**
     * Calculate pairwise similarities between legal cases using TF-IDF vectors
     */
    calculateCaseSimilarities(cases) {
        if (cases.length < 2) return [];

        // Extract text content for vectorization
        const caseTexts = cases.map(case => case.text || '');
        const citations = cases.map((case, i) => case.citation || `Case_${i}`);

        // Create TF-IDF vectors
        const tfidfVectors = this.vectorizer.fitTransform(caseTexts);

        // Store embeddings for later use
        citations.forEach((citation, i) => {
            this.caseEmbeddings.set(citation, tfidfVectors[i]);
        });

        // Calculate similarity matrix
        const similarityMatrix = [];
        for (let i = 0; i < tfidfVectors.length; i++) {
            const row = [];
            for (let j = 0; j < tfidfVectors.length; j++) {
                if (i === j) {
                    row.push(1.0);
                } else {
                    const similarity = MathUtils.cosineSimilarity(tfidfVectors[i], tfidfVectors[j]);
                    row.push(similarity);
                }
            }
            similarityMatrix.push(row);
        }

        console.log(`Calculated similarities for ${cases.length} cases`);
        console.log(`Similarity matrix shape: ${similarityMatrix.length}x${similarityMatrix[0]?.length || 0}`);

        // Calculate average similarity (excluding diagonal)
        const allSimilarities = [];
        for (let i = 0; i < similarityMatrix.length; i++) {
            for (let j = i + 1; j < similarityMatrix[i].length; j++) {
                allSimilarities.push(similarityMatrix[i][j]);
            }
        }
        const avgSimilarity = MathUtils.mean(allSimilarities);
        console.log(`Average similarity: ${avgSimilarity.toFixed(3)}`);

        return similarityMatrix;
    }

    /**
     * Find the most similar cases to a target case
     */
    findMostSimilarCases(cases, targetCaseIndex, nSimilar = 5) {
        const similarityMatrix = this.calculateCaseSimilarities(cases);

        if (similarityMatrix.length === 0 || targetCaseIndex >= cases.length) {
            return [];
        }

        const targetCitation = cases[targetCaseIndex].citation || `Case_${targetCaseIndex}`;
        const similarities = similarityMatrix[targetCaseIndex];

        // Get indices of most similar cases (excluding self)
        const indexedSimilarities = similarities
            .map((sim, idx) => ({ index: idx, similarity: sim }))
            .filter(item => item.index !== targetCaseIndex)
            .sort((a, b) => b.similarity - a.similarity)
            .slice(0, nSimilar);

        const similarCases = indexedSimilarities.map(item => {
            const caseIndex = item.index;
            const caseCitation = cases[caseIndex].citation || `Case_${caseIndex}`;
            const similarityScore = item.similarity;

            // Identify matching features
            const matchingFeatures = this._identifyMatchingFeatures(
                cases[targetCaseIndex],
                cases[caseIndex]
            );

            // Calculate vector distance
            const targetVector = this.caseEmbeddings.get(targetCitation) || [];
            const caseVector = this.caseEmbeddings.get(caseCitation) || [];
            const vectorDistance = MathUtils.euclideanDistance(targetVector, caseVector);

            return new CaseSimilarity({
                case1Citation: targetCitation,
                case2Citation: caseCitation,
                similarityScore: similarityScore,
                matchingFeatures: matchingFeatures,
                vectorDistance: vectorDistance
            });
        });

        return similarCases;
    }

    /**
     * Perform statistical analysis of court performance metrics
     */
    analyzeCourtPerformance(casesByCourt) {
        const courtStats = new Map();

        for (const [courtName, cases] of Object.entries(casesByCourt)) {
            if (!cases || cases.length === 0) continue;

            // Calculate basic statistics
            const caseLengths = cases.map(case => (case.text || '').length);
            const citationCounts = cases.map(case => (case.citations || []).length);

            // Temporal analysis
            const dates = cases.map(case => {
                try {
                    return new Date(case.date || '2023-01-01');
                } catch {
                    return new Date('2023-01-01');
                }
            });

            // Create temporal trend (cases per month)
            let trend = [];
            if (dates.length > 0) {
                const minDate = new Date(Math.min(...dates));
                const maxDate = new Date(Math.max(...dates));
                const monthsDiff = (maxDate.getFullYear() - minDate.getFullYear()) * 12 +
                                 (maxDate.getMonth() - minDate.getMonth()) + 1;

                trend = new Array(monthsDiff).fill(0);
                dates.forEach(date => {
                    const monthIndex = (date.getFullYear() - minDate.getFullYear()) * 12 +
                                     (date.getMonth() - minDate.getMonth());
                    if (monthIndex >= 0 && monthIndex < trend.length) {
                        trend[monthIndex]++;
                    }
                });
            }

            // Subject area analysis
            const subjectAreas = {};
            cases.forEach(case => {
                const subjects = case.subjectMatter || [];
                subjects.forEach(subject => {
                    subjectAreas[subject] = (subjectAreas[subject] || 0) + 1;
                });
            });

            const stats = new CourtStatistics({
                courtName: courtName,
                totalCases: cases.length,
                avgCaseLength: MathUtils.mean(caseLengths),
                citationFrequency: MathUtils.mean(citationCounts),
                subjectAreas: subjectAreas,
                temporalTrends: trend
            });

            courtStats.set(courtName, stats);
        }

        this.courtStats = courtStats;
        return Object.fromEntries(courtStats);
    }

    /**
     * Analyze citation networks between cases using graph metrics
     */
    calculateCitationNetworkMetrics(cases) {
        const nCases = cases.length;
        if (nCases === 0) return {};

        // Build citation adjacency matrix
        const citationMatrix = Array(nCases).fill(null).map(() => Array(nCases).fill(0));
        const caseCitations = new Map();

        cases.forEach((case, i) => {
            const citation = case.citation || `Case_${i}`;
            caseCitations.set(citation, i);
        });

        // Fill adjacency matrix
        cases.forEach((case, i) => {
            const citedCases = case.citations || [];
            citedCases.forEach(cited => {
                if (caseCitations.has(cited)) {
                    const j = caseCitations.get(cited);
                    citationMatrix[i][j] = 1;
                }
            });
        });

        // Calculate network metrics
        const inDegree = citationMatrix[0].map((_, colIndex) =>
            citationMatrix.reduce((sum, row) => sum + row[colIndex], 0)
        );
        const outDegree = citationMatrix.map(row =>
            row.reduce((sum, val) => sum + val, 0)
        );

        // Authority score (simplified PageRank)
        const authorityScores = this._calculateAuthorityScores(citationMatrix);

        // Clustering coefficient
        const clusteringCoeff = this._calculateClusteringCoefficient(citationMatrix);

        // Network density
        const possibleEdges = nCases * (nCases - 1);
        const actualEdges = citationMatrix.reduce((sum, row) =>
            sum + row.reduce((rowSum, val) => rowSum + val, 0), 0
        );
        const density = possibleEdges > 0 ? actualEdges / possibleEdges : 0;

        return {
            citationMatrix: citationMatrix,
            inDegreeCentrality: inDegree,
            outDegreeCentrality: outDegree,
            authorityScores: authorityScores,
            clusteringCoefficient: clusteringCoeff,
            networkDensity: density,
            mostCitedCases: this._getMostCitedCases(cases, inDegree),
            mostCitingCases: this._getMostCitingCases(cases, outDegree)
        };
    }

    /**
     * Perform time series analysis on legal data trends
     */
    performTrendAnalysis(temporalData) {
        const trendResults = {};

        for (const [metricName, dataPoints] of Object.entries(temporalData)) {
            if (!dataPoints || dataPoints.length === 0) continue;

            // Sort by date
            const sortedData = dataPoints.sort((a, b) => a[0] - b[0]);
            const dates = sortedData.map(point => point[0]);
            const values = sortedData.map(point => point[1]);

            // Convert dates to numeric values for analysis
            const baseDate = dates[0];
            const dateNums = dates.map(date => (date - baseDate) / (1000 * 60 * 60 * 24)); // days

            // Linear trend analysis
            if (values.length > 1) {
                const { slope, intercept } = MathUtils.linearRegression(dateNums, values);
                const trendLine = dateNums.map(x => slope * x + intercept);

                // Calculate trend metrics
                const trendStrength = MathUtils.correlation(dateNums, values);
                const volatility = MathUtils.standardDeviation(values);
                const meanValue = MathUtils.mean(values);
                const growthRate = meanValue !== 0 ? slope / meanValue : 0;

                // Seasonal analysis (if enough data points)
                const seasonalPattern = this._detectSeasonalPattern(dates, values);

                trendResults[metricName] = {
                    slope: slope,
                    intercept: intercept,
                    trendLine: trendLine,
                    trendStrength: trendStrength,
                    volatility: volatility,
                    growthRate: growthRate,
                    seasonalPattern: seasonalPattern,
                    dataPoints: values.length,
                    dateRange: [dates[0], dates[dates.length - 1]]
                };
            }
        }

        return trendResults;
    }

    /**
     * Calculate relevance scores for search results using numerical methods
     */
    calculateSearchRelevanceScores(queryTerms, documents) {
        if (!documents || documents.length === 0) return [];

        // Create query vector
        const queryText = queryTerms.join(' ');
        const allTexts = [queryText, ...documents.map(doc => doc.text || '')];

        // Vectorize all texts
        const tfidfVectors = this.vectorizer.fitTransform(allTexts);
        const queryVector = tfidfVectors[0];
        const docVectors = tfidfVectors.slice(1);

        // Calculate cosine similarity with query
        const similarities = docVectors.map(docVector =>
            MathUtils.cosineSimilarity(queryVector, docVector)
        );

        // Boost scores based on additional factors
        const boostedScores = similarities.map((similarity, i) => {
            const doc = documents[i];
            let score = similarity;

            // Court authority boost
            const court = (doc.court || '').toLowerCase();
            if (court.includes('supreme')) {
                score *= 1.5;
            } else if (court.includes('appeal')) {
                score *= 1.3;
            } else if (court.includes('high')) {
                score *= 1.2;
            }

            // Recency boost
            try {
                const docDate = new Date(doc.date || '2020-01-01');
                const now = new Date();
                const daysOld = (now - docDate) / (1000 * 60 * 60 * 24);
                const recencyFactor = Math.exp(-daysOld / 365.0); // Exponential decay
                score *= (1.0 + 0.2 * recencyFactor);
            } catch {
                // Use default if date parsing fails
            }

            // Citation count boost
            const citationCount = (doc.citations || []).length;
            const citationBoost = 1.0 + 0.1 * Math.log(1 + citationCount);
            score *= citationBoost;

            return score;
        });

        return boostedScores;
    }

    /**
     * Identify common features between two cases
     */
    _identifyMatchingFeatures(case1, case2) {
        const features = [];

        // Same court
        if (case1.court === case2.court) {
            features.push('same_court');
        }

        // Similar subject matter
        const subjects1 = new Set(case1.subjectMatter || []);
        const subjects2 = new Set(case2.subjectMatter || []);
        const sharedSubjects = [...subjects1].filter(subject => subjects2.has(subject));
        if (sharedSubjects.length > 0) {
            features.push('shared_subjects');
        }

        // Same year
        try {
            const date1 = new Date(case1.date || '2020-01-01');
            const date2 = new Date(case2.date || '2020-01-01');
            if (date1.getFullYear() === date2.getFullYear()) {
                features.push('same_year');
            }
        } catch {
            // Ignore date parsing errors
        }

        return features;
    }

    /**
     * Calculate authority scores using simplified PageRank algorithm
     */
    _calculateAuthorityScores(citationMatrix, iterations = 50, damping = 0.85) {
        const n = citationMatrix.length;
        if (n === 0) return [];

        // Initialize scores
        let scores = new Array(n).fill(1 / n);

        // Calculate out-degrees for normalization
        const outDegrees = citationMatrix.map(row =>
            row.reduce((sum, val) => sum + val, 0)
        );

        // PageRank iterations
        for (let iter = 0; iter < iterations; iter++) {
            const newScores = new Array(n).fill((1 - damping) / n);

            for (let i = 0; i < n; i++) {
                for (let j = 0; j < n; j++) {
                    if (citationMatrix[j][i] === 1) {
                        const contribution = outDegrees[j] > 0 ?
                            damping * scores[j] / outDegrees[j] :
                            damping * scores[j] / n;
                        newScores[i] += contribution;
                    }
                }
            }

            scores = newScores;
        }

        return scores;
    }

    /**
     * Calculate clustering coefficient for the citation network
     */
    _calculateClusteringCoefficient(adjMatrix) {
        const n = adjMatrix.length;
        if (n < 3) return 0.0;

        let clusteringSum = 0;
        let validNodes = 0;

        for (let i = 0; i < n; i++) {
            const neighbors = [];
            for (let j = 0; j < n; j++) {
                if (adjMatrix[i][j] === 1) {
                    neighbors.push(j);
                }
            }

            if (neighbors.length < 2) continue;

            // Count triangles
            let triangles = 0;
            const possibleTriangles = (neighbors.length * (neighbors.length - 1)) / 2;

            for (let j = 0; j < neighbors.length; j++) {
                for (let k = j + 1; k < neighbors.length; k++) {
                    if (adjMatrix[neighbors[j]][neighbors[k]] === 1) {
                        triangles++;
                    }
                }
            }

            if (possibleTriangles > 0) {
                clusteringSum += triangles / possibleTriangles;
                validNodes++;
            }
        }

        return validNodes > 0 ? clusteringSum / validNodes : 0.0;
    }

    /**
     * Get the most cited cases
     */
    _getMostCitedCases(cases, inDegrees, n = 5) {
        const indexed = inDegrees.map((degree, index) => ({ index, degree }));
        indexed.sort((a, b) => b.degree - a.degree);

        return indexed.slice(0, n).map(item => ({
            case: cases[item.index],
            citations: item.degree
        }));
    }

    /**
     * Get cases that cite the most other cases
     */
    _getMostCitingCases(cases, outDegrees, n = 5) {
        const indexed = outDegrees.map((degree, index) => ({ index, degree }));
        indexed.sort((a, b) => b.degree - a.degree);

        return indexed.slice(0, n).map(item => ({
            case: cases[item.index],
            citingCount: item.degree
        }));
    }

    /**
     * Detect seasonal patterns in time series data
     */
    _detectSeasonalPattern(dates, values) {
        if (dates.length < 12) return {};

        // Group by month
        const monthlyValues = {};
        dates.forEach((date, index) => {
            const month = date.getMonth() + 1; // 1-12
            if (!monthlyValues[month]) {
                monthlyValues[month] = [];
            }
            monthlyValues[month].push(values[index]);
        });

        // Calculate monthly averages
        const monthlyAverages = {};
        for (const [month, monthValues] of Object.entries(monthlyValues)) {
            monthlyAverages[month] = MathUtils.mean(monthValues);
        }

        return monthlyAverages;
    }
}

/**
 * Demonstrate numerical analysis with sample UK legal data
 */
function demonstrateLegalDataAnalysis() {
    console.log("=== Legal Data Numerical Analysis Demo (JavaScript) ===\n");

    // Sample UK legal cases data
    const sampleCases = [
        {
            citation: '[2023] UKSC 15',
            court: 'Supreme Court',
            date: '2023-05-15',
            text: 'This case concerns constitutional law and the relationship between Parliament and the Executive. The principles of parliamentary sovereignty are fundamental to our constitutional framework. We must consider the precedents established in previous constitutional cases.',
            subjectMatter: ['Constitutional Law', 'Parliamentary Sovereignty'],
            citations: ['[2019] UKSC 41', '[2017] UKSC 5']
        },
        {
            citation: '[2023] EWCA Civ 892',
            court: 'Court of Appeal',
            date: '2023-08-22',
            text: 'The appellant challenges the decision on grounds of procedural fairness and natural justice. Administrative law principles require that decisions are made fairly and with proper consideration of all relevant factors. The Wednesbury principles apply to this review.',
            subjectMatter: ['Administrative Law', 'Judicial Review'],
            citations: ['[2019] UKSC 41', '[2018] EWCA Civ 234']
        },
        {
            citation: '[2023] EWHC 1456 (Admin)',
            court: 'High Court',
            date: '2023-06-30',
            text: 'This judicial review concerns the lawfulness of the defendant\'s decision-making process. Administrative law requires compliance with statutory procedures and consideration of relevant factors. The principles established in constitutional cases provide guidance.',
            subjectMatter: ['Administrative Law', 'Statutory Interpretation'],
            citations: ['[2023] UKSC 15', '[2018] EWCA Civ 234']
        },
        {
            citation: '[2023] UKSC 28',
            court: 'Supreme Court',
            date: '2023-09-10',
            text: 'The constitutional principles governing the exercise of prerogative powers are well-established. Parliamentary sovereignty remains the cornerstone of our constitutional system. This case builds upon previous Supreme Court decisions regarding Executive power.',
            subjectMatter: ['Constitutional Law', 'Prerogative Powers'],
            citations: ['[2023] UKSC 15', '[2019] UKSC 41']
        },
        {
            citation: '[2023] EWCA Crim 567',
            court: 'Court of Appeal Criminal Division',
            date: '2023-07-18',
            text: 'The appellant argues that the conviction is unsafe due to procedural irregularities during the trial. Criminal procedure requires strict adherence to statutory provisions and common law principles. The evidence must be evaluated according to established precedents.',
            subjectMatter: ['Criminal Law', 'Criminal Procedure'],
            citations: ['[2022] UKSC 32', '[2021] EWCA Crim 123']
        }
    ];

    // Initialize analyzer
    const analyzer = new LegalDataAnalyzer();

    // 1. Case Similarity Analysis
    console.log("1. CASE SIMILARITY ANALYSIS:");
    const similarityMatrix = analyzer.calculateCaseSimilarities(sampleCases);
    console.log(`   Similarity matrix shape: ${similarityMatrix.length}x${similarityMatrix[0]?.length || 0}`);

    // Find similar cases to the first case
    const similarCases = analyzer.findMostSimilarCases(sampleCases, 0, 3);
    console.log(`   Cases similar to ${sampleCases[0].citation}:`);
    similarCases.forEach(simCase => {
        console.log(`     - ${simCase.case2Citation}: ${simCase.similarityScore.toFixed(3)} similarity`);
        console.log(`       Features: ${simCase.matchingFeatures.join(', ')}`);
    });

    // 2. Court Performance Analysis
    console.log(`\n2. COURT PERFORMANCE ANALYSIS:`);
    const casesByCourt = {};
    sampleCases.forEach(case => {
        const court = case.court;
        if (!casesByCourt[court]) {
            casesByCourt[court] = [];
        }
        casesByCourt[court].push(case);
    });

    const courtStats = analyzer.analyzeCourtPerformance(casesByCourt);
    Object.values(courtStats).forEach(stats => {
        console.log(`   ${stats.courtName}:`);
        console.log(`     Total cases: ${stats.totalCases}`);
        console.log(`     Avg case length: ${Math.round(stats.avgCaseLength)} characters`);
        console.log(`     Avg citations per case: ${stats.citationFrequency.toFixed(1)}`);
        console.log(`     Subject areas: ${Object.keys(stats.subjectAreas).join(', ')}`);
    });

    // 3. Citation Network Analysis
    console.log(`\n3. CITATION NETWORK ANALYSIS:`);
    const networkMetrics = analyzer.calculateCitationNetworkMetrics(sampleCases);
    console.log(`   Network density: ${networkMetrics.networkDensity.toFixed(3)}`);
    console.log(`   Clustering coefficient: ${networkMetrics.clusteringCoefficient.toFixed(3)}`);

    console.log("   Most cited cases:");
    networkMetrics.mostCitedCases.slice(0, 3).forEach(caseInfo => {
        const citation = caseInfo.case.citation || 'Unknown';
        const citations = caseInfo.citations;
        console.log(`     - ${citation}: ${citations} citations`);
    });

    console.log("   Authority scores:");
    networkMetrics.authorityScores.forEach((score, i) => {
        const citation = sampleCases[i].citation;
        console.log(`     - ${citation}: ${score.toFixed(3)}`);
    });

    // 4. Search Relevance Scoring
    console.log(`\n4. SEARCH RELEVANCE SCORING:`);
    const queryTerms = ['constitutional', 'parliament', 'sovereignty'];
    const relevanceScores = analyzer.calculateSearchRelevanceScores(queryTerms, sampleCases);

    console.log(`   Query: ${queryTerms.join(' ')}`);
    console.log("   Relevance scores:");
    relevanceScores.forEach((score, i) => {
        const citation = sampleCases[i].citation;
        const court = sampleCases[i].court;
        console.log(`     - ${citation} (${court}): ${score.toFixed(3)}`);
    });

    // 5. Trend Analysis
    console.log(`\n5. TREND ANALYSIS:`);

    // Create sample temporal data
    const baseDate = new Date('2023-01-01');
    const temporalData = {
        caseVolume: Array.from({ length: 12 }, (_, i) => [
            new Date(baseDate.getTime() + i * 30 * 24 * 60 * 60 * 1000),
            10 + 5 * Math.sin(i * 0.5) + (Math.random() - 0.5) * 4
        ]),
        avgCaseLength: Array.from({ length: 12 }, (_, i) => [
            new Date(baseDate.getTime() + i * 30 * 24 * 60 * 60 * 1000),
            5000 + 1000 * Math.sin(i * 0.3) + (Math.random() - 0.5) * 1000
        ])
    };

    const trendResults = analyzer.performTrendAnalysis(temporalData);
    Object.entries(trendResults).forEach(([metricName, results]) => {
        console.log(`   ${metricName}:`);
        console.log(`     Trend strength: ${results.trendStrength.toFixed(3)}`);
        console.log(`     Growth rate: ${results.growthRate.toFixed(3)}`);
        console.log(`     Volatility: ${results.volatility.toFixed(1)}`);
    });

    // 6. Statistical Summary
    console.log(`\n6. STATISTICAL SUMMARY:`);
    const allSimilarities = [];
    for (let i = 0; i < similarityMatrix.length; i++) {
        for (let j = i + 1; j < similarityMatrix[i].length; j++) {
            allSimilarities.push(similarityMatrix[i][j]);
        }
    }

    console.log(`   Average case similarity: ${MathUtils.mean(allSimilarities).toFixed(3)}`);
    console.log(`   Similarity std deviation: ${MathUtils.standardDeviation(allSimilarities).toFixed(3)}`);
    console.log(`   Max similarity: ${Math.max(...allSimilarities).toFixed(3)}`);
    console.log(`   Min similarity: ${Math.min(...allSimilarities).toFixed(3)}`);

    // Case length statistics
    const caseLengths = sampleCases.map(case => case.text.length);
    console.log(`   Average case length: ${Math.round(MathUtils.mean(caseLengths))} characters`);
    console.log(`   Case length std dev: ${Math.round(MathUtils.standardDeviation(caseLengths))} characters`);

    // Citation statistics
    const citationCounts = sampleCases.map(case => case.citations.length);
    console.log(`   Average citations per case: ${MathUtils.mean(citationCounts).toFixed(1)}`);

    const allCitations = new Set();
    sampleCases.forEach(case => case.citations.forEach(citation => allCitations.add(citation)));
    console.log(`   Total unique citations: ${allCitations.size}`);

    return {
        similarityMatrix: similarityMatrix,
        courtStats: courtStats,
        networkMetrics: networkMetrics,
        relevanceScores: relevanceScores,
        trendResults: trendResults
    };
}

// Export for use in Node.js or browser environments
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        LegalDataAnalyzer,
        CaseSimilarity,
        CourtStatistics,
        MathUtils,
        TFIDFVectorizer,
        demonstrateLegalDataAnalysis
    };
}

// Auto-run demo if script is executed directly
if (typeof window !== 'undefined') {
    // Browser environment
    document.addEventListener('DOMContentLoaded', () => {
        console.log('Legal Data Analyzer loaded in browser');
        // Uncomment to run demo: demonstrateLegalDataAnalysis();
    });
} else if (typeof require !== 'undefined' && require.main === module) {
    // Node.js environment - run demo
    demonstrateLegalDataAnalysis();
}