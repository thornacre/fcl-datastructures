/**
 * LRU Cache for Frequently Accessed Judgments (JavaScript)
 * ========================================================
 *
 * This module implements an LRU (Least Recently Used) cache optimized for storing
 * and retrieving frequently accessed legal judgments in Find Case Law (FCL).
 * Provides client-side caching for improved performance.
 */

class LRUCache {
    constructor(capacity = 1000, defaultTtl = null) {
        this.capacity = capacity;
        this.defaultTtl = defaultTtl;
        this._cache = new Map();
        this._hits = 0;
        this._misses = 0;
        this._evictions = 0;
    }

    get(key) {
        if (!this._cache.has(key)) {
            this._misses++;
            return null;
        }

        const entry = this._cache.get(key);

        // Check if expired
        if (entry.ttl && Date.now() > entry.expires) {
            this._cache.delete(key);
            this._misses++;
            return null;
        }

        // Move to end (most recently used)
        this._cache.delete(key);
        this._cache.set(key, {
            ...entry,
            lastAccessed: Date.now(),
            accessCount: entry.accessCount + 1
        });

        this._hits++;
        return entry.value;
    }

    put(key, value, ttl = null) {
        const entryTtl = ttl || this.defaultTtl;
        const entry = {
            value,
            created: Date.now(),
            lastAccessed: Date.now(),
            accessCount: 1,
            ttl: entryTtl,
            expires: entryTtl ? Date.now() + (entryTtl * 1000) : null
        };

        if (this._cache.has(key)) {
            this._cache.delete(key);
        }

        this._cache.set(key, entry);

        // Evict if over capacity
        if (this._cache.size > this.capacity) {
            const firstKey = this._cache.keys().next().value;
            this._cache.delete(firstKey);
            this._evictions++;
        }
    }

    getStats() {
        const total = this._hits + this._misses;
        return {
            capacity: this.capacity,
            size: this._cache.size,
            hits: this._hits,
            misses: this._misses,
            hitRate: total > 0 ? this._hits / total : 0,
            evictions: this._evictions
        };
    }
}

class JudgmentCache {
    constructor(options = {}) {
        this.judgmentCache = new LRUCache(options.judgmentCapacity || 500, 3600);
        this.searchCache = new LRUCache(options.searchCapacity || 200, 1800);
        this.citationCache = new LRUCache(options.citationCapacity || 1000, 7200);
    }

    cacheJudgment(citation, data) {
        const key = citation.toUpperCase().trim();
        this.judgmentCache.put(key, data);
    }

    getJudgment(citation) {
        const key = citation.toUpperCase().trim();
        return this.judgmentCache.get(key);
    }

    cacheSearchResults(query, results) {
        const hash = this._hashQuery(query);
        this.searchCache.put(hash, results);
    }

    getSearchResults(query) {
        const hash = this._hashQuery(query);
        return this.searchCache.get(hash);
    }

    _hashQuery(query) {
        return btoa(JSON.stringify(query)).replace(/[+/=]/g, '');
    }
}

function demonstrateLRUCache() {
    console.log("=== LRU Cache Demo (JavaScript) ===");

    const cache = new LRUCache(3);
    const cases = [
        { citation: '[2023] UKSC 15', court: 'UKSC' },
        { citation: '[2023] EWCA Civ 892', court: 'EWCA' },
        { citation: '[2023] EWHC 1456', court: 'EWHC' }
    ];

    cases.forEach(c => {
        cache.put(c.citation, c);
        console.log(`Cached: ${c.citation}`);
    });

    console.log(`\nStats: ${JSON.stringify(cache.getStats(), null, 2)}`);
    return cache;
}

if (typeof module !== 'undefined' && module.exports) {
    module.exports = { LRUCache, JudgmentCache, demonstrateLRUCache };
} else if (typeof window === 'undefined') {
    demonstrateLRUCache();
}