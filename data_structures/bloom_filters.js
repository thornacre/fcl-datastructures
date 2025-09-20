/**
 * Bloom Filters for Probabilistic Membership Testing
 * ==================================================
 *
 * This module implements Bloom filters for efficient probabilistic membership testing
 * in Find Case Law (FCL). Provides memory-efficient storage for large sets of legal
 * citations, case names, and other identifiers with fast lookup capabilities.
 *
 * Key FCL Use Cases:
 * - Fast citation existence checking before expensive database queries
 * - Duplicate judgment detection during bulk imports
 * - Efficient set operations on large collections of case identifiers
 * - Cache miss optimization for frequently accessed legal documents
 * - Real-time filtering of search results by known citation patterns
 */

/**
 * Statistics about a Bloom filter's state and performance
 */
class BloomFilterStats {
    constructor(capacity, numHashFunctions, bitArraySize, itemsAdded, estimatedFalsePositiveRate, memoryUsageBytes) {
        this.capacity = capacity;
        this.numHashFunctions = numHashFunctions;
        this.bitArraySize = bitArraySize;
        this.itemsAdded = itemsAdded;
        this.estimatedFalsePositiveRate = estimatedFalsePositiveRate;
        this.memoryUsageBytes = memoryUsageBytes;
    }
}

/**
 * Space-efficient probabilistic data structure for membership testing.
 * Optimized for legal citation and case name filtering in FCL.
 */
class BloomFilter {
    /**
     * Initialize Bloom filter with specified capacity and error rate.
     * @param {number} capacity - Expected number of items to store
     * @param {number} falsePositiveRate - Desired false positive probability (0.0 to 1.0)
     */
    constructor(capacity = 10000, falsePositiveRate = 0.01) {
        this.capacity = capacity;
        this.falsePositiveRate = falsePositiveRate;

        // Calculate optimal bit array size and number of hash functions
        this.bitArraySize = this._calculateBitArraySize(capacity, falsePositiveRate);
        this.numHashFunctions = this._calculateNumHashFunctions(this.bitArraySize, capacity);

        // Initialize bit array
        this.bitArray = new Array(this.bitArraySize).fill(false);
        this.itemsAdded = 0;
    }

    /**
     * Add an item to the Bloom filter.
     * @param {string|Uint8Array} item - Item to add
     */
    add(item) {
        if (typeof item === 'string') {
            item = new TextEncoder().encode(item);
        }

        // Generate hash values and set corresponding bits
        for (let i = 0; i < this.numHashFunctions; i++) {
            const hashValue = this._hash(item, i);
            const bitIndex = hashValue % this.bitArraySize;
            this.bitArray[bitIndex] = true;
        }

        this.itemsAdded++;
    }

    /**
     * Test if an item might be in the set.
     * @param {string|Uint8Array} item - Item to test
     * @returns {boolean} True if item might be in set, false if definitely not
     */
    contains(item) {
        if (typeof item === 'string') {
            item = new TextEncoder().encode(item);
        }

        // Check all hash positions
        for (let i = 0; i < this.numHashFunctions; i++) {
            const hashValue = this._hash(item, i);
            const bitIndex = hashValue % this.bitArraySize;
            if (!this.bitArray[bitIndex]) {
                return false;
            }
        }

        return true;
    }

    /**
     * Add multiple items efficiently
     * @param {Array} items - Array of items to add
     */
    addMultiple(items) {
        for (const item of items) {
            this.add(item);
        }
    }

    /**
     * Create union of two Bloom filters (logical OR operation).
     * Both filters must have same parameters.
     * @param {BloomFilter} other - Other Bloom filter
     * @returns {BloomFilter} New filter containing union
     */
    union(other) {
        if (this.bitArraySize !== other.bitArraySize ||
            this.numHashFunctions !== other.numHashFunctions) {
            throw new Error('Cannot union Bloom filters with different parameters');
        }

        // Create new filter with same parameters
        const result = new BloomFilter(this.capacity, this.falsePositiveRate);

        // Perform bitwise OR
        for (let i = 0; i < this.bitArraySize; i++) {
            result.bitArray[i] = this.bitArray[i] || other.bitArray[i];
        }

        // Estimate items in union (approximate)
        result.itemsAdded = Math.max(this.itemsAdded, other.itemsAdded);

        return result;
    }

    /**
     * Estimate the size of intersection with another Bloom filter.
     * @param {BloomFilter} other - Other Bloom filter
     * @returns {number} Approximate number of common elements
     */
    intersectionEstimate(other) {
        if (this.bitArraySize !== other.bitArraySize ||
            this.numHashFunctions !== other.numHashFunctions) {
            throw new Error('Cannot intersect Bloom filters with different parameters');
        }

        // Count bits set in both filters
        let commonBits = 0;
        for (let i = 0; i < this.bitArraySize; i++) {
            if (this.bitArray[i] && other.bitArray[i]) {
                commonBits++;
            }
        }

        // Estimate intersection size using formula
        if (commonBits === 0) {
            return 0.0;
        }

        // Simplified estimation (can be improved with more sophisticated formulas)
        const estimatedIntersection = (
            commonBits * this.capacity * other.capacity /
            (this.bitArraySize * Math.max(this.itemsAdded, other.itemsAdded))
        );

        return Math.max(0.0, estimatedIntersection);
    }

    /**
     * Get comprehensive statistics about the filter
     * @returns {BloomFilterStats} Filter statistics
     */
    getStatistics() {
        // Calculate current false positive rate
        const currentFpr = this._calculateCurrentFalsePositiveRate();

        // Estimate memory usage (rough estimate for bit array)
        const memoryUsage = Math.ceil(this.bitArraySize / 8); // bits to bytes

        return new BloomFilterStats(
            this.capacity,
            this.numHashFunctions,
            this.bitArraySize,
            this.itemsAdded,
            currentFpr,
            memoryUsage
        );
    }

    /**
     * Save Bloom filter to JSON string
     * @returns {string} JSON representation
     */
    saveToJSON() {
        return JSON.stringify({
            capacity: this.capacity,
            falsePositiveRate: this.falsePositiveRate,
            bitArraySize: this.bitArraySize,
            numHashFunctions: this.numHashFunctions,
            itemsAdded: this.itemsAdded,
            bitArray: this._bitArrayToHex()
        });
    }

    /**
     * Load Bloom filter from JSON string
     * @param {string} jsonString - JSON representation
     * @returns {BloomFilter} Restored filter
     */
    static loadFromJSON(jsonString) {
        const data = JSON.parse(jsonString);
        const filter = new BloomFilter(data.capacity, data.falsePositiveRate);
        filter.itemsAdded = data.itemsAdded;
        filter._bitArrayFromHex(data.bitArray);
        return filter;
    }

    /**
     * Generate hash value for item with given seed
     * @param {Uint8Array} item - Item to hash
     * @param {number} seed - Hash seed
     * @returns {number} Hash value
     */
    _hash(item, seed) {
        // Simple hash function (djb2 variant with seed)
        let hash = 5381 + seed;
        for (let i = 0; i < item.length; i++) {
            hash = ((hash << 5) + hash) + item[i];
            hash = hash & 0x7FFFFFFF; // Ensure positive 32-bit integer
        }
        return hash;
    }

    /**
     * Calculate optimal bit array size
     * @param {number} capacity - Expected capacity
     * @param {number} fpRate - False positive rate
     * @returns {number} Optimal bit array size
     */
    _calculateBitArraySize(capacity, fpRate) {
        // Formula: m = -(n * ln(p)) / (ln(2)^2)
        return Math.ceil(-capacity * Math.log(fpRate) / (Math.log(2) ** 2));
    }

    /**
     * Calculate optimal number of hash functions
     * @param {number} bitArraySize - Size of bit array
     * @param {number} capacity - Expected capacity
     * @returns {number} Optimal number of hash functions
     */
    _calculateNumHashFunctions(bitArraySize, capacity) {
        // Formula: k = (m/n) * ln(2)
        return Math.max(1, Math.round((bitArraySize / capacity) * Math.log(2)));
    }

    /**
     * Calculate current false positive rate based on items added
     * @returns {number} Current false positive rate
     */
    _calculateCurrentFalsePositiveRate() {
        if (this.itemsAdded === 0) {
            return 0.0;
        }

        // Formula: (1 - e^(-k*n/m))^k
        try {
            const exponent = -this.numHashFunctions * this.itemsAdded / this.bitArraySize;
            return Math.pow(1 - Math.exp(exponent), this.numHashFunctions);
        } catch (error) {
            return 1.0;
        }
    }

    /**
     * Convert bit array to hex string for serialization
     * @returns {string} Hex representation
     */
    _bitArrayToHex() {
        const bytes = [];
        for (let i = 0; i < this.bitArraySize; i += 8) {
            let byte = 0;
            for (let j = 0; j < 8 && i + j < this.bitArraySize; j++) {
                if (this.bitArray[i + j]) {
                    byte |= (1 << j);
                }
            }
            bytes.push(byte);
        }
        return bytes.map(b => b.toString(16).padStart(2, '0')).join('');
    }

    /**
     * Restore bit array from hex string
     * @param {string} hexString - Hex representation
     */
    _bitArrayFromHex(hexString) {
        const bytes = [];
        for (let i = 0; i < hexString.length; i += 2) {
            bytes.push(parseInt(hexString.substr(i, 2), 16));
        }

        this.bitArray = new Array(this.bitArraySize).fill(false);
        for (let i = 0; i < bytes.length; i++) {
            for (let j = 0; j < 8 && i * 8 + j < this.bitArraySize; j++) {
                this.bitArray[i * 8 + j] = !!(bytes[i] & (1 << j));
            }
        }
    }
}

/**
 * Specialized Bloom filter for UK legal citations with optimized
 * hashing and validation for legal document identifiers.
 */
class CitationBloomFilter {
    /**
     * Initialize citation filter with legal-specific optimizations.
     * @param {number} capacity - Expected number of citations
     * @param {number} falsePositiveRate - Lower error rate for critical legal lookups
     */
    constructor(capacity = 50000, falsePositiveRate = 0.001) {
        this.filter = new BloomFilter(capacity, falsePositiveRate);

        // Legal citation patterns for validation
        this.citationPatterns = [
            /\[(\d{4})\]\s+([A-Z]{2,})\s+(\d+)/g, // [2023] UKSC 15
            /\[(\d{4})\]\s+([A-Z]{2,})\s+([A-Za-z]+)\s+(\d+)/g, // [2023] EWCA Civ 892
            /\((\d{4})\)\s+([A-Z]{2,})\s+(\d+)/g, // (2023) AC 123
        ];

        // Court hierarchies for weighting
        this.courtWeights = {
            'UKSC': 1.0,    // Supreme Court - highest priority
            'UKHL': 0.95,   // House of Lords
            'EWCA': 0.9,    // Court of Appeal
            'EWHC': 0.85,   // High Court
            'UKUT': 0.8,    // Upper Tribunal
            'UKFTT': 0.75,  // First-tier Tribunal
        };
    }

    /**
     * Add a legal citation to the filter with validation.
     * @param {string} citation - Legal citation string
     * @returns {boolean} True if citation was added, false if invalid format
     */
    addCitation(citation) {
        const normalized = this._normalizeCitation(citation);
        if (normalized) {
            this.filter.add(normalized);
            return true;
        }
        return false;
    }

    /**
     * Check if citation might exist in the filter.
     * @param {string} citation - Legal citation to check
     * @returns {boolean} True if citation might exist, false if definitely not
     */
    containsCitation(citation) {
        const normalized = this._normalizeCitation(citation);
        if (!normalized) {
            return false;
        }
        return this.filter.contains(normalized);
    }

    /**
     * Add multiple citations efficiently.
     * @param {Array<string>} citations - List of legal citations
     * @returns {number} Number of successfully added citations
     */
    addCitationsBulk(citations) {
        let addedCount = 0;
        const normalizedCitations = [];

        for (const citation of citations) {
            const normalized = this._normalizeCitation(citation);
            if (normalized) {
                normalizedCitations.push(normalized);
                addedCount++;
            }
        }

        this.filter.addMultiple(normalizedCitations);
        return addedCount;
    }

    /**
     * Analyze court coverage in the filter.
     * @param {Array<string>} citations - Sample citations to analyze
     * @returns {Object} Dictionary of court -> count mappings
     */
    getCourtCoverage(citations) {
        const courtCounts = {};

        for (const citation of citations) {
            const court = this._extractCourt(citation);
            if (court && this.containsCitation(citation)) {
                courtCounts[court] = (courtCounts[court] || 0) + 1;
            }
        }

        return courtCounts;
    }

    /**
     * Normalize legal citation for consistent storage.
     * @param {string} citation - Raw citation string
     * @returns {string|null} Normalized citation or null if invalid
     */
    _normalizeCitation(citation) {
        // Remove extra whitespace and standardize format
        citation = citation.trim().replace(/\s+/g, ' ');

        // Basic validation - must contain year and court
        if (!/\d/.test(citation)) {
            return null;
        }

        if (!/[A-Z]/.test(citation)) {
            return null;
        }

        // Convert to uppercase for consistency
        return citation.toUpperCase().trim();
    }

    /**
     * Extract court identifier from citation
     * @param {string} citation - Citation string
     * @returns {string|null} Court identifier or null
     */
    _extractCourt(citation) {
        for (const pattern of this.citationPatterns) {
            pattern.lastIndex = 0; // Reset regex
            const match = pattern.exec(citation);
            if (match && match.length >= 3) {
                // Return the court identifier (second capture group)
                return match[2];
            }
        }
        return null;
    }
}

/**
 * Scalable Bloom filter that automatically grows when capacity is exceeded.
 * Useful for long-running FCL systems with unpredictable data volumes.
 */
class ScalableBloomFilter {
    /**
     * Initialize scalable filter.
     * @param {number} initialCapacity - Starting capacity
     * @param {number} falsePositiveRate - Target false positive rate
     * @param {number} growthFactor - Capacity multiplier when scaling
     */
    constructor(initialCapacity = 10000, falsePositiveRate = 0.01, growthFactor = 2) {
        this.initialCapacity = initialCapacity;
        this.falsePositiveRate = falsePositiveRate;
        this.growthFactor = growthFactor;

        // List of Bloom filters (each with increasing capacity)
        this.filters = [new BloomFilter(initialCapacity, falsePositiveRate)];
        this.currentFilterIndex = 0;
    }

    /**
     * Add item to the appropriate filter
     * @param {string|Uint8Array} item - Item to add
     */
    add(item) {
        let currentFilter = this.filters[this.currentFilterIndex];

        // Check if current filter is full
        if (currentFilter.itemsAdded >= currentFilter.capacity) {
            this._createNewFilter();
            currentFilter = this.filters[this.currentFilterIndex];
        }

        currentFilter.add(item);
    }

    /**
     * Check if item exists in any of the filters
     * @param {string|Uint8Array} item - Item to check
     * @returns {boolean} True if item might exist
     */
    contains(item) {
        // Check all filters (most recent first for better performance)
        for (let i = this.filters.length - 1; i >= 0; i--) {
            if (this.filters[i].contains(item)) {
                return true;
            }
        }
        return false;
    }

    /**
     * Get total capacity across all filters
     * @returns {number} Total capacity
     */
    getTotalCapacity() {
        return this.filters.reduce((sum, filter) => sum + filter.capacity, 0);
    }

    /**
     * Get total items across all filters
     * @returns {number} Total items
     */
    getTotalItems() {
        return this.filters.reduce((sum, filter) => sum + filter.itemsAdded, 0);
    }

    /**
     * Get comprehensive statistics for all filters
     * @returns {Object} Overall statistics
     */
    getOverallStatistics() {
        const totalMemory = this.filters.reduce((sum, filter) =>
            sum + filter.getStatistics().memoryUsageBytes, 0);

        const avgFpr = this.filters.reduce((sum, filter) =>
            sum + filter.getStatistics().estimatedFalsePositiveRate, 0) / this.filters.length;

        return {
            numFilters: this.filters.length,
            totalCapacity: this.getTotalCapacity(),
            totalItems: this.getTotalItems(),
            memoryUsageBytes: totalMemory,
            averageFalsePositiveRate: avgFpr
        };
    }

    /**
     * Create new filter with increased capacity
     */
    _createNewFilter() {
        const newCapacity = Math.floor(this.initialCapacity * Math.pow(this.growthFactor, this.filters.length));
        const newFilter = new BloomFilter(newCapacity, this.falsePositiveRate);
        this.filters.push(newFilter);
        this.currentFilterIndex = this.filters.length - 1;
    }
}

/**
 * Demonstrate Bloom filter implementations with UK legal data
 */
function demonstrateBloomFilters() {
    console.log('=== Bloom Filters for Legal Citation Testing Demo ===\n');

    // Sample UK legal citations
    const sampleCitations = [
        '[2023] UKSC 15',
        '[2023] EWCA Civ 892',
        '[2023] EWHC 1456 (Admin)',
        '[2022] UKHL 7',
        '[2023] UKFTT 234 (TC)',
        '[2022] EWCA Crim 567',
        '[2023] EWHC 789 (QB)',
        '[2021] UKSC 42',
        '(2023) AC 123',
        '[2023] UKUT 456 (AAC)'
    ];

    // Additional citations for testing false positives
    const testCitations = [
        '[2024] UKSC 99',   // Future citation (should not exist)
        '[2023] FAKE 123',  // Invalid court
        '[2023] UKSC 999',  // High number (might not exist)
        '[2020] EWCA Civ 1',  // Old citation
    ];

    // 1. Basic Bloom Filter Demo
    console.log('1. BASIC BLOOM FILTER:');
    const basicFilter = new BloomFilter(1000, 0.01);

    console.log('   Adding sample citations:');
    for (const citation of sampleCitations) {
        basicFilter.add(citation);
        console.log(`   Added: ${citation}`);
    }

    console.log(`\n   Filter statistics:`);
    const stats = basicFilter.getStatistics();
    console.log(`     Items added: ${stats.itemsAdded}`);
    console.log(`     Bit array size: ${stats.bitArraySize}`);
    console.log(`     Hash functions: ${stats.numHashFunctions}`);
    console.log(`     Memory usage: ${stats.memoryUsageBytes} bytes`);
    console.log(`     Est. false positive rate: ${stats.estimatedFalsePositiveRate.toFixed(4)}`);

    // 2. Membership Testing
    console.log(`\n2. MEMBERSHIP TESTING:`);
    console.log('   Testing known citations (should all return True):');
    for (const citation of sampleCitations.slice(0, 5)) {
        const result = basicFilter.contains(citation);
        console.log(`     ${citation}: ${result}`);
    }

    console.log('\n   Testing unknown citations (may have false positives):');
    for (const citation of testCitations) {
        const result = basicFilter.contains(citation);
        console.log(`     ${citation}: ${result}`);
    }

    // 3. Citation-Specific Bloom Filter
    console.log(`\n3. CITATION-SPECIFIC BLOOM FILTER:`);
    const citationFilter = new CitationBloomFilter(5000, 0.001);

    const addedCount = citationFilter.addCitationsBulk(sampleCitations);
    console.log(`   Successfully added ${addedCount}/${sampleCitations.length} citations`);

    console.log('   Citation validation and lookup:');
    const testCases = [...sampleCitations.slice(0, 3), 'INVALID CITATION', '[2024] UNKNOWN 1'];

    for (const citation of testCases) {
        const exists = citationFilter.containsCitation(citation);
        const normalized = citationFilter._normalizeCitation(citation);
        console.log(`     ${citation}: exists=${exists}, normalized='${normalized}'`);
    }

    // 4. Court Coverage Analysis
    console.log(`\n4. COURT COVERAGE ANALYSIS:`);
    const courtCoverage = citationFilter.getCourtCoverage(sampleCitations);
    console.log('   Citations by court:');
    for (const [court, count] of Object.entries(courtCoverage)) {
        console.log(`     ${court}: ${count} citations`);
    }

    // 5. Scalable Bloom Filter Demo
    console.log(`\n5. SCALABLE BLOOM FILTER:`);
    const scalableFilter = new ScalableBloomFilter(
        5,  // Small capacity to force scaling
        0.01,
        2
    );

    console.log('   Adding citations to trigger scaling:');
    for (const [i, citation] of sampleCitations.entries()) {
        scalableFilter.add(citation);
        const totalItems = scalableFilter.getTotalItems();
        const numFilters = scalableFilter.filters.length;
        console.log(`     Added ${citation}: ${totalItems} items in ${numFilters} filter(s)`);
    }

    const overallStats = scalableFilter.getOverallStatistics();
    console.log(`\n   Scalable filter statistics:`);
    for (const [key, value] of Object.entries(overallStats)) {
        console.log(`     ${key}: ${value}`);
    }

    // 6. Filter Union Operations
    console.log(`\n6. BLOOM FILTER UNION:`);
    const filter1 = new BloomFilter(1000, 0.01);
    const filter2 = new BloomFilter(1000, 0.01);

    // Add different sets to each filter
    for (const citation of sampleCitations.slice(0, 5)) {
        filter1.add(citation);
    }

    for (const citation of sampleCitations.slice(3, 8)) { // Some overlap
        filter2.add(citation);
    }

    console.log(`   Filter 1 items: ${filter1.itemsAdded}`);
    console.log(`   Filter 2 items: ${filter2.itemsAdded}`);

    // Create union
    const unionFilter = filter1.union(filter2);
    console.log(`   Union filter created`);

    // Test union membership
    console.log('   Testing union membership:');
    const testItems = sampleCitations.slice(0, 8);
    for (const citation of testItems) {
        const inUnion = unionFilter.contains(citation);
        const inFilter1 = filter1.contains(citation);
        const inFilter2 = filter2.contains(citation);
        console.log(`     ${citation}: union=${inUnion}, f1=${inFilter1}, f2=${inFilter2}`);
    }

    // 7. Intersection Estimation
    console.log(`\n7. INTERSECTION ESTIMATION:`);
    const estimatedIntersection = filter1.intersectionEstimate(filter2);
    console.log(`   Estimated intersection size: ${estimatedIntersection.toFixed(2)}`);

    // 8. Performance Comparison
    console.log(`\n8. PERFORMANCE AND MEMORY EFFICIENCY:`);

    // Compare with set-based approach
    const citationSet = new Set(sampleCitations);
    const setMemoryEstimate = JSON.stringify([...citationSet]).length;
    console.log(`   JavaScript Set memory (approx): ${setMemoryEstimate} bytes`);
    console.log(`   Bloom filter memory: ${basicFilter.getStatistics().memoryUsageBytes} bytes`);

    // Memory efficiency ratio
    if (basicFilter.getStatistics().memoryUsageBytes > 0) {
        const efficiencyRatio = setMemoryEstimate / basicFilter.getStatistics().memoryUsageBytes;
        console.log(`   Memory efficiency ratio: ${efficiencyRatio.toFixed(2)}x`);
    }

    return {
        basicFilter,
        citationFilter,
        scalableFilter,
        unionFilter,
        statistics: {
            basicStats: stats,
            overallStats,
            courtCoverage
        }
    };
}

// Export classes and functions for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        BloomFilterStats,
        BloomFilter,
        CitationBloomFilter,
        ScalableBloomFilter,
        demonstrateBloomFilters
    };
}

// Run demonstration if this file is executed directly
if (typeof require !== 'undefined' && require.main === module) {
    demonstrateBloomFilters();
}