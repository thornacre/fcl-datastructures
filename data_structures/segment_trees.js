/**
 * Segment Trees for Range Query Optimization
 * ==========================================
 *
 * This module implements segment trees for efficient range query operations
 * in Find Case Law (FCL). Provides fast aggregation queries over date ranges,
 * score ranges, and other numerical attributes of legal documents.
 *
 * Key FCL Use Cases:
 * - Fast date range queries for judgment filtering by time periods
 * - Efficient aggregation of relevance scores across document collections
 * - Range-based statistics computation (min, max, sum, average)
 * - Timeline visualization data preparation
 * - Performance optimization for complex search filters
 */

/**
 * Represents metrics for a legal judgment
 */
class JudgmentMetrics {
    constructor(judgmentId, date, relevanceScore, wordCount, citationCount, courtLevel) {
        this.judgmentId = judgmentId;
        this.date = new Date(date);
        this.relevanceScore = relevanceScore;
        this.wordCount = wordCount;
        this.citationCount = citationCount;
        this.courtLevel = courtLevel; // Hierarchy level (1=Supreme, 2=Appeal, etc.)
    }

    toString() {
        return `${this.judgmentId}: score=${this.relevanceScore}, date=${this.date.toDateString()}`;
    }
}

/**
 * Node in a segment tree
 */
class SegmentTreeNode {
    constructor(start, end, value = null) {
        this.start = start;
        this.end = end;
        this.value = value;
        this.left = null;
        this.right = null;
        this.lazy = null; // For lazy propagation
    }
}

/**
 * Generic segment tree implementation for range queries and updates.
 * Supports various aggregation functions (sum, min, max, custom).
 */
class SegmentTree {
    /**
     * Initialize segment tree with data and aggregation operation.
     * @param {Array<number>} data - List of numerical values
     * @param {Function} operation - Binary function for combining values (default: sum)
     * @param {*} identity - Identity element for the operation (default: 0)
     */
    constructor(data, operation = null, identity = null) {
        this.n = data.length;
        this.data = [...data];

        // Default to sum operation
        this.operation = operation || ((x, y) => x + y);
        this.identity = identity !== null ? identity : 0;

        // Build the tree
        if (this.n > 0) {
            this.root = this._build(0, this.n - 1);
        } else {
            this.root = null;
        }
    }

    /**
     * Query the aggregate value in range [left, right].
     * @param {number} left - Left boundary (inclusive)
     * @param {number} right - Right boundary (inclusive)
     * @returns {*} Aggregated value over the range
     */
    query(left, right) {
        if (!this.root || left > right) {
            return this.identity;
        }
        return this._query(this.root, 0, this.n - 1, left, right);
    }

    /**
     * Update value at specific index.
     * @param {number} index - Index to update
     * @param {number} value - New value
     */
    update(index, value) {
        if (index >= 0 && index < this.n) {
            this.data[index] = value;
            if (this.root) {
                this._update(this.root, 0, this.n - 1, index, value);
            }
        }
    }

    /**
     * Update all values in range [left, right] by adding delta.
     * Uses lazy propagation for efficiency.
     * @param {number} left - Left boundary (inclusive)
     * @param {number} right - Right boundary (inclusive)
     * @param {number} delta - Value to add to all elements in range
     */
    rangeUpdate(left, right, delta) {
        if (this.root && left <= right) {
            this._rangeUpdate(this.root, 0, this.n - 1, left, right, delta);
        }
    }

    /**
     * Build segment tree recursively
     */
    _build(start, end) {
        if (start === end) {
            // Leaf node
            return new SegmentTreeNode(start, end, this.data[start]);
        }

        const mid = Math.floor((start + end) / 2);
        const node = new SegmentTreeNode(start, end);
        node.left = this._build(start, mid);
        node.right = this._build(mid + 1, end);

        // Combine values from children
        node.value = this.operation(node.left.value, node.right.value);
        return node;
    }

    /**
     * Query range recursively
     */
    _query(node, start, end, left, right) {
        // Push down lazy updates
        this._pushLazy(node, start, end);

        // No overlap
        if (right < start || left > end) {
            return this.identity;
        }

        // Complete overlap
        if (left <= start && end <= right) {
            return node.value;
        }

        // Partial overlap
        const mid = Math.floor((start + end) / 2);
        const leftResult = this._query(node.left, start, mid, left, right);
        const rightResult = this._query(node.right, mid + 1, end, left, right);

        return this.operation(leftResult, rightResult);
    }

    /**
     * Update single element recursively
     */
    _update(node, start, end, index, value) {
        if (start === end) {
            // Leaf node
            node.value = value;
            return;
        }

        const mid = Math.floor((start + end) / 2);
        if (index <= mid) {
            this._update(node.left, start, mid, index, value);
        } else {
            this._update(node.right, mid + 1, end, index, value);
        }

        // Update internal node
        node.value = this.operation(node.left.value, node.right.value);
    }

    /**
     * Range update with lazy propagation
     */
    _rangeUpdate(node, start, end, left, right, delta) {
        // Push existing lazy updates
        this._pushLazy(node, start, end);

        // No overlap
        if (right < start || left > end) {
            return;
        }

        // Complete overlap
        if (left <= start && end <= right) {
            node.lazy = (node.lazy || 0) + delta;
            this._pushLazy(node, start, end);
            return;
        }

        // Partial overlap
        const mid = Math.floor((start + end) / 2);
        this._rangeUpdate(node.left, start, mid, left, right, delta);
        this._rangeUpdate(node.right, mid + 1, end, left, right, delta);

        // Update current node
        this._pushLazy(node.left, start, mid);
        this._pushLazy(node.right, mid + 1, end);
        node.value = this.operation(node.left.value, node.right.value);
    }

    /**
     * Apply lazy updates to node
     */
    _pushLazy(node, start, end) {
        if (node.lazy !== null && node.lazy !== 0) {
            // Apply lazy update to current node
            if (this.operation.toString().includes('x + y')) { // Sum operation
                node.value += node.lazy * (end - start + 1);
            } else {
                // For other operations, this needs to be customized
                node.value += node.lazy;
            }

            // Propagate to children
            if (start !== end) { // Not a leaf
                if (node.left) {
                    node.left.lazy = (node.left.lazy || 0) + node.lazy;
                }
                if (node.right) {
                    node.right.lazy = (node.right.lazy || 0) + node.lazy;
                }
            }

            node.lazy = null;
        }
    }
}

/**
 * Specialized segment tree for date-based range queries on legal judgments.
 * Maps dates to indices for efficient range operations.
 */
class DateRangeSegmentTree {
    /**
     * Initialize with judgment data.
     * @param {Array<JudgmentMetrics>} judgments - List of judgment metrics
     */
    constructor(judgments) {
        this.judgments = [...judgments].sort((a, b) => a.date - b.date);

        // Create date to index mapping
        this.dates = this.judgments.map(j => j.date);
        this.dateToIndex = new Map();
        this.dates.forEach((date, index) => {
            this.dateToIndex.set(date.getTime(), index);
        });

        // Build multiple trees for different metrics
        this.relevanceTree = new SegmentTree(
            this.judgments.map(j => j.relevanceScore),
            (x, y) => x + y // Sum
        );

        this.countTree = new SegmentTree(
            this.judgments.map(() => 1), // Count of judgments
            (x, y) => x + y
        );

        this.maxScoreTree = new SegmentTree(
            this.judgments.map(j => j.relevanceScore),
            Math.max,
            -Infinity
        );

        this.minScoreTree = new SegmentTree(
            this.judgments.map(j => j.relevanceScore),
            Math.min,
            Infinity
        );
    }

    /**
     * Query statistics for judgments in date range.
     * @param {Date} startDate - Start of date range
     * @param {Date} endDate - End of date range
     * @returns {Object} Dictionary with aggregated statistics
     */
    queryDateRange(startDate, endDate) {
        // Find index range
        const startIdx = this._findDateIndex(startDate, true);
        const endIdx = this._findDateIndex(endDate, false);

        if (startIdx > endIdx || startIdx >= this.judgments.length) {
            return {
                count: 0,
                totalRelevance: 0.0,
                averageRelevance: 0.0,
                maxRelevance: null,
                minRelevance: null,
                judgments: []
            };
        }

        // Query all trees
        const count = this.countTree.query(startIdx, endIdx);
        const totalRelevance = this.relevanceTree.query(startIdx, endIdx);
        const maxRelevance = this.maxScoreTree.query(startIdx, endIdx);
        const minRelevance = this.minScoreTree.query(startIdx, endIdx);

        // Get actual judgments in range
        const judgmentsInRange = this.judgments.slice(startIdx, endIdx + 1);

        return {
            count,
            totalRelevance,
            averageRelevance: count > 0 ? totalRelevance / count : 0.0,
            maxRelevance: maxRelevance !== -Infinity ? maxRelevance : null,
            minRelevance: minRelevance !== Infinity ? minRelevance : null,
            judgments: judgmentsInRange,
            dateRange: [this.dates[startIdx], this.dates[endIdx]]
        };
    }

    /**
     * Update relevance score for a specific judgment.
     * @param {string} judgmentId - ID of judgment to update
     * @param {number} newScore - New relevance score
     * @returns {boolean} True if judgment was found and updated
     */
    updateJudgmentScore(judgmentId, newScore) {
        for (let i = 0; i < this.judgments.length; i++) {
            if (this.judgments[i].judgmentId === judgmentId) {
                const oldScore = this.judgments[i].relevanceScore;
                this.judgments[i].relevanceScore = newScore;

                // Update trees
                this.relevanceTree.update(i, newScore);
                this.maxScoreTree.update(i, newScore);
                this.minScoreTree.update(i, newScore);

                return true;
            }
        }
        return false;
    }

    /**
     * Get timeline data for visualization, grouped into buckets.
     * @param {number} numBuckets - Number of time buckets to create
     * @returns {Array<Object>} List of dictionaries with timeline statistics
     */
    getTimelineData(numBuckets = 12) {
        if (this.judgments.length === 0) {
            return [];
        }

        const startDate = this.dates[0];
        const endDate = this.dates[this.dates.length - 1];

        // Calculate bucket size
        const totalDays = Math.floor((endDate - startDate) / (1000 * 60 * 60 * 24));
        const bucketDays = Math.max(1, Math.floor(totalDays / numBuckets));

        const timeline = [];
        let currentDate = new Date(startDate);

        for (let i = 0; i < numBuckets; i++) {
            const bucketEnd = new Date(currentDate.getTime() + bucketDays * 24 * 60 * 60 * 1000);
            if (i === numBuckets - 1) { // Last bucket
                bucketEnd.setTime(endDate.getTime());
            }

            const bucketStats = this.queryDateRange(currentDate, bucketEnd);

            timeline.push({
                bucketIndex: i,
                startDate: new Date(currentDate),
                endDate: new Date(bucketEnd),
                judgmentCount: bucketStats.count,
                averageRelevance: bucketStats.averageRelevance,
                maxRelevance: bucketStats.maxRelevance,
                totalRelevance: bucketStats.totalRelevance
            });

            currentDate = new Date(bucketEnd.getTime() + 24 * 60 * 60 * 1000);
        }

        return timeline;
    }

    /**
     * Binary search to find date index.
     * @param {Date} targetDate - Date to find
     * @param {boolean} leftBound - If true, find leftmost position; if false, rightmost
     * @returns {number} Index of date position
     */
    _findDateIndex(targetDate, leftBound = true) {
        let left = 0;
        let right = this.dates.length - 1;
        let result = this.dates.length; // Default to beyond range

        while (left <= right) {
            const mid = Math.floor((left + right) / 2);

            if (leftBound) {
                if (this.dates[mid] >= targetDate) {
                    result = mid;
                    right = mid - 1;
                } else {
                    left = mid + 1;
                }
            } else {
                if (this.dates[mid] <= targetDate) {
                    result = mid;
                    left = mid + 1;
                } else {
                    right = mid - 1;
                }
            }
        }

        return result;
    }
}

/**
 * Segment tree specialized for relevance score range queries.
 * Enables efficient filtering by score thresholds.
 */
class ScoreRangeTree {
    /**
     * Initialize score range tree.
     * @param {Array<JudgmentMetrics>} judgments - List of judgment metrics
     * @param {number} scorePrecision - Number of score buckets (0.0 to 1.0 mapped to 0 to precision-1)
     */
    constructor(judgments, scorePrecision = 100) {
        this.judgments = judgments;
        this.scorePrecision = scorePrecision;

        // Create score buckets
        this.scoreBuckets = Array.from({ length: scorePrecision }, () => []);

        // Distribute judgments into buckets
        for (const judgment of judgments) {
            const bucketIdx = Math.min(scorePrecision - 1,
                Math.floor(judgment.relevanceScore * scorePrecision));
            this.scoreBuckets[bucketIdx].push(judgment);
        }

        // Build segment tree with counts per bucket
        const bucketCounts = this.scoreBuckets.map(bucket => bucket.length);
        this.countTree = new SegmentTree(bucketCounts, (x, y) => x + y);

        // Build trees for other aggregations
        const bucketMaxScores = [];
        const bucketMinScores = [];

        for (const bucket of this.scoreBuckets) {
            if (bucket.length > 0) {
                bucketMaxScores.push(Math.max(...bucket.map(j => j.relevanceScore)));
                bucketMinScores.push(Math.min(...bucket.map(j => j.relevanceScore)));
            } else {
                bucketMaxScores.push(0.0);
                bucketMinScores.push(0.0);
            }
        }

        this.maxScoreTree = new SegmentTree(bucketMaxScores, Math.max, 0.0);
        this.minScoreTree = new SegmentTree(bucketMinScores, Math.min, 1.0);
    }

    /**
     * Query judgments within score range.
     * @param {number} minScore - Minimum score (inclusive)
     * @param {number} maxScore - Maximum score (inclusive)
     * @returns {Object} Dictionary with statistics and matching judgments
     */
    queryScoreRange(minScore, maxScore) {
        // Convert scores to bucket indices
        const minBucket = Math.max(0, Math.floor(minScore * this.scorePrecision));
        const maxBucket = Math.min(this.scorePrecision - 1, Math.floor(maxScore * this.scorePrecision));

        if (minBucket > maxBucket) {
            return {
                count: 0,
                judgments: [],
                scoreRange: [minScore, maxScore]
            };
        }

        // Query segment tree
        const count = this.countTree.query(minBucket, maxBucket);

        // Collect matching judgments
        const matchingJudgments = [];
        for (let bucketIdx = minBucket; bucketIdx <= maxBucket; bucketIdx++) {
            for (const judgment of this.scoreBuckets[bucketIdx]) {
                if (minScore <= judgment.relevanceScore && judgment.relevanceScore <= maxScore) {
                    matchingJudgments.push(judgment);
                }
            }
        }

        // Sort by score descending
        matchingJudgments.sort((a, b) => b.relevanceScore - a.relevanceScore);

        return {
            count: matchingJudgments.length,
            judgments: matchingJudgments,
            scoreRange: [minScore, maxScore],
            bucketRange: [minBucket, maxBucket]
        };
    }

    /**
     * Get distribution of judgments across score buckets.
     * @returns {Array<Object>} List of bucket statistics
     */
    getScoreDistribution() {
        const distribution = [];

        for (let i = 0; i < this.scoreBuckets.length; i++) {
            const bucket = this.scoreBuckets[i];
            const bucketMin = i / this.scorePrecision;
            const bucketMax = (i + 1) / this.scorePrecision;

            let avgScore = 0;
            let maxScore = 0;
            let minScore = 0;

            if (bucket.length > 0) {
                avgScore = bucket.reduce((sum, j) => sum + j.relevanceScore, 0) / bucket.length;
                maxScore = Math.max(...bucket.map(j => j.relevanceScore));
                minScore = Math.min(...bucket.map(j => j.relevanceScore));
            }

            distribution.push({
                bucketIndex: i,
                scoreRange: [bucketMin, bucketMax],
                count: bucket.length,
                averageScore: avgScore,
                maxScore: maxScore,
                minScore: minScore
            });
        }

        return distribution;
    }
}

/**
 * Demonstrate segment tree implementations with UK legal judgment data
 */
function demonstrateSegmentTrees() {
    console.log('=== Segment Trees for Range Query Optimization Demo ===\n');

    // Sample UK legal judgment metrics
    const sampleJudgments = [
        new JudgmentMetrics(
            'uksc_2023_15',
            '2023-05-15',
            0.95,
            12500,
            45,
            1
        ),
        new JudgmentMetrics(
            'ewca_2023_892',
            '2023-08-22',
            0.87,
            8900,
            32,
            2
        ),
        new JudgmentMetrics(
            'ewhc_2023_1456',
            '2023-06-30',
            0.73,
            6700,
            18,
            3
        ),
        new JudgmentMetrics(
            'ukhl_2023_7',
            '2023-03-10',
            0.91,
            15200,
            67,
            1
        ),
        new JudgmentMetrics(
            'ukftt_2023_234',
            '2023-09-05',
            0.62,
            4500,
            12,
            5
        ),
        new JudgmentMetrics(
            'ewca_2023_445',
            '2023-07-18',
            0.84,
            9800,
            28,
            2
        ),
        new JudgmentMetrics(
            'ewhc_2023_789',
            '2023-04-25',
            0.78,
            7200,
            22,
            3
        ),
    ];

    // 1. Basic Segment Tree Demo
    console.log('1. BASIC SEGMENT TREE OPERATIONS:');
    const values = sampleJudgments.map(j => j.relevanceScore);
    console.log(`   Relevance scores: ${values.map(v => v.toFixed(2))}`);

    const sumTree = new SegmentTree(values, (x, y) => x + y);
    const maxTree = new SegmentTree(values, Math.max, 0.0);
    const minTree = new SegmentTree(values, Math.min, 1.0);

    console.log(`   Sum of all scores: ${sumTree.query(0, values.length - 1).toFixed(2)}`);
    console.log(`   Max score: ${maxTree.query(0, values.length - 1).toFixed(2)}`);
    console.log(`   Min score: ${minTree.query(0, values.length - 1).toFixed(2)}`);

    // Range queries
    console.log(`\n   Range queries (indices 1-4):`);
    console.log(`     Sum: ${sumTree.query(1, 4).toFixed(2)}`);
    console.log(`     Max: ${maxTree.query(1, 4).toFixed(2)}`);
    console.log(`     Min: ${minTree.query(1, 4).toFixed(2)}`);

    // 2. Date Range Segment Tree
    console.log(`\n2. DATE RANGE QUERIES:`);
    const dateTree = new DateRangeSegmentTree(sampleJudgments);

    // Query specific date ranges
    const dateRanges = [
        [new Date('2023-05-01'), new Date('2023-07-31')], // Spring/Summer 2023
        [new Date('2023-08-01'), new Date('2023-09-30')], // Late 2023
        [new Date('2023-01-01'), new Date('2023-12-31')], // All 2023
    ];

    for (const [startDate, endDate] of dateRanges) {
        const result = dateTree.queryDateRange(startDate, endDate);
        console.log(`\n   Date range ${startDate.toDateString()} to ${endDate.toDateString()}:`);
        console.log(`     Count: ${result.count}`);
        console.log(`     Average relevance: ${result.averageRelevance.toFixed(3)}`);
        console.log(`     Max relevance: ${result.maxRelevance?.toFixed(3) || 'N/A'}`);
        console.log(`     Min relevance: ${result.minRelevance?.toFixed(3) || 'N/A'}`);
        console.log(`     Judgments: ${result.judgments.slice(0, 3).map(j => j.judgmentId).join(', ')}...`);
    }

    // 3. Timeline Analysis
    console.log(`\n3. TIMELINE ANALYSIS:`);
    const timeline = dateTree.getTimelineData(6);

    console.log(`   Timeline data (6 buckets):`);
    for (const bucket of timeline) {
        console.log(`     Bucket ${bucket.bucketIndex}: ${bucket.startDate.toDateString()} to ${bucket.endDate.toDateString()}`);
        console.log(`       Judgments: ${bucket.judgmentCount}`);
        console.log(`       Avg relevance: ${bucket.averageRelevance.toFixed(3)}`);
    }

    // 4. Score Range Queries
    console.log(`\n4. SCORE RANGE QUERIES:`);
    const scoreTree = new ScoreRangeTree(sampleJudgments, 100);

    const scoreRanges = [
        [0.9, 1.0], // High relevance
        [0.7, 0.9], // Medium relevance
        [0.0, 0.7], // Lower relevance
    ];

    for (const [minScore, maxScore] of scoreRanges) {
        const result = scoreTree.queryScoreRange(minScore, maxScore);
        console.log(`\n   Score range ${minScore} to ${maxScore}:`);
        console.log(`     Count: ${result.count}`);
        console.log(`     Judgments:`);
        for (const judgment of result.judgments.slice(0, 3)) {
            console.log(`       ${judgment.judgmentId}: ${judgment.relevanceScore.toFixed(3)}`);
        }
    }

    // 5. Score Distribution Analysis
    console.log(`\n5. SCORE DISTRIBUTION ANALYSIS:`);
    const distribution = scoreTree.getScoreDistribution();

    // Show non-empty buckets
    console.log(`   Score distribution (non-empty buckets):`);
    const nonEmptyBuckets = distribution.filter(b => b.count > 0);
    for (const bucket of nonEmptyBuckets.slice(0, 5)) { // Show first 5
        const [min, max] = bucket.scoreRange;
        console.log(`     ${min.toFixed(2)}-${max.toFixed(2)}: ${bucket.count} judgments, avg=${bucket.averageScore.toFixed(3)}`);
    }

    // 6. Dynamic Updates
    console.log(`\n6. DYNAMIC UPDATES:`);
    const originalScore = sampleJudgments.find(j => j.judgmentId === 'uksc_2023_15').relevanceScore;
    console.log(`   Original uksc_2023_15 score: ${originalScore}`);

    // Update score
    const updated = dateTree.updateJudgmentScore('uksc_2023_15', 0.99);
    console.log(`   Update successful: ${updated}`);

    // Query again to show update effect
    const result = dateTree.queryDateRange(new Date('2023-05-01'), new Date('2023-05-31'));
    console.log(`   Updated May 2023 stats:`);
    console.log(`     Average relevance: ${result.averageRelevance.toFixed(3)}`);
    console.log(`     Max relevance: ${result.maxRelevance?.toFixed(3) || 'N/A'}`);

    // 7. Range Updates Demo
    console.log(`\n7. RANGE UPDATES (Lazy Propagation):`);
    const updateTree = new SegmentTree([1, 2, 3, 4, 5], (x, y) => x + y);

    console.log(`   Original array: ${updateTree.data}`);
    console.log(`   Sum of range [1,3]: ${updateTree.query(1, 3)}`);

    // Range update: add 10 to indices 1-3
    updateTree.rangeUpdate(1, 3, 10);
    console.log(`   After adding 10 to range [1,3]:`);
    console.log(`   Sum of range [1,3]: ${updateTree.query(1, 3)}`);
    console.log(`   Sum of entire array: ${updateTree.query(0, 4)}`);

    // 8. Performance Comparison
    console.log(`\n8. PERFORMANCE BENEFITS:`);
    console.log(`   Segment tree advantages:`);
    console.log(`     - Range queries: O(log n) vs O(n) linear scan`);
    console.log(`     - Range updates: O(log n) with lazy propagation`);
    console.log(`     - Multiple query types on same data structure`);
    console.log(`     - Memory efficient for large datasets`);

    // Example with larger dataset
    const largeJudgments = [];
    for (let i = 0; i < 1000; i++) {
        largeJudgments.push(...sampleJudgments);
    }
    const largeDateTree = new DateRangeSegmentTree(largeJudgments);

    console.log(`\n   Large dataset example (${largeJudgments.length} judgments):`);
    const largeResult = largeDateTree.queryDateRange(new Date('2023-01-01'), new Date('2023-12-31'));
    console.log(`     Query result: ${largeResult.count} judgments`);
    console.log(`     Average relevance: ${largeResult.averageRelevance.toFixed(3)}`);

    return {
        dateTree,
        scoreTree,
        basicTrees: {
            sumTree,
            maxTree,
            minTree
        },
        timeline,
        scoreDistribution: distribution,
        sampleJudgments
    };
}

// Export classes and functions for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        JudgmentMetrics,
        SegmentTreeNode,
        SegmentTree,
        DateRangeSegmentTree,
        ScoreRangeTree,
        demonstrateSegmentTrees
    };
}

// Run demonstration if this file is executed directly
if (typeof require !== 'undefined' && require.main === module) {
    demonstrateSegmentTrees();
}