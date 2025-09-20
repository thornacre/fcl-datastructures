/**
 * Mock ORM Query Examples for Legal Data
 * ======================================
 *
 * This module demonstrates ORM-like query patterns and database operations
 * for Find Case Law (FCL) using JavaScript. Provides examples of how to structure
 * efficient database queries and data access patterns in a Node.js environment.
 *
 * Key FCL Use Cases:
 * - Complex judgment filtering with multiple criteria
 * - Efficient relationship queries across legal entities
 * - Database-level aggregations for statistical analysis
 * - Query optimization for large legal document collections
 * - Advanced search patterns with full-text search integration
 */

/**
 * Mock model definitions representing legal entities
 */

class Court {
    constructor(code, name, level, jurisdiction, parentCourt = null) {
        this.code = code;
        this.name = name;
        this.level = level;
        this.jurisdiction = jurisdiction;
        this.parentCourt = parentCourt;
    }

    toString() {
        return `${this.code}: ${this.name}`;
    }
}

class Judge {
    constructor(name, title, appointedDate, court, active = true) {
        this.name = name;
        this.title = title;
        this.appointedDate = new Date(appointedDate);
        this.court = court;
        this.active = active;
    }

    toString() {
        return `${this.title} ${this.name}`;
    }
}

class LegalArea {
    constructor(name, description, parentArea = null) {
        this.name = name;
        this.description = description;
        this.parentArea = parentArea;
    }

    toString() {
        return this.name;
    }
}

class Judgment {
    constructor(neutralCitation, caseName, uri, court, judgmentDate) {
        this.neutralCitation = neutralCitation;
        this.caseName = caseName;
        this.uri = uri;
        this.court = court;
        this.judgmentDate = new Date(judgmentDate);
        this.summary = '';
        this.fullText = '';
        this.wordCount = 0;
        this.createdAt = new Date();
        this.updatedAt = new Date();
        this.published = false;
        this.relevanceScore = 0.0;
        this.judges = [];
        this.legalAreas = [];
        this.citationsMade = [];
        this.citationsReceived = [];
    }

    toString() {
        return `${this.neutralCitation}: ${this.caseName}`;
    }
}

class Citation {
    constructor(citingJudgment, citedJudgment, context, citationType, paragraphNumber = null) {
        this.citingJudgment = citingJudgment;
        this.citedJudgment = citedJudgment;
        this.context = context;
        this.citationType = citationType;
        this.paragraphNumber = paragraphNumber;
    }
}

/**
 * Mock ORM Query Builder
 * Simulates database query patterns with method chaining
 */
class QueryBuilder {
    constructor(model, data = []) {
        this.model = model;
        this.data = [...data];
        this.conditions = [];
        this.orderFields = [];
        this.limitValue = null;
        this.offsetValue = 0;
        this.selectedFields = null;
        this.joinedData = new Map();
    }

    /**
     * Add filter condition
     */
    filter(condition) {
        this.conditions.push(condition);
        return this;
    }

    /**
     * Add multiple filter conditions (AND)
     */
    where(conditions) {
        Object.entries(conditions).forEach(([field, value]) => {
            if (value !== null && value !== undefined) {
                this.conditions.push(item => {
                    const fieldValue = this._getNestedField(item, field);
                    if (Array.isArray(value)) {
                        return value.includes(fieldValue);
                    }
                    if (typeof value === 'object' && value.gte !== undefined) {
                        return fieldValue >= value.gte;
                    }
                    if (typeof value === 'object' && value.lte !== undefined) {
                        return fieldValue <= value.lte;
                    }
                    if (typeof value === 'object' && value.contains !== undefined) {
                        return fieldValue && fieldValue.toLowerCase().includes(value.contains.toLowerCase());
                    }
                    if (typeof value === 'object' && value.in !== undefined) {
                        return value.in.includes(fieldValue);
                    }
                    return fieldValue === value;
                });
            }
        });
        return this;
    }

    /**
     * Add OR condition
     */
    orWhere(conditions) {
        const orConditions = Object.entries(conditions).map(([field, value]) => {
            return item => {
                const fieldValue = this._getNestedField(item, field);
                if (Array.isArray(value)) {
                    return value.includes(fieldValue);
                }
                return fieldValue === value;
            };
        });

        this.conditions.push(item => orConditions.some(condition => condition(item)));
        return this;
    }

    /**
     * Order results
     */
    orderBy(...fields) {
        this.orderFields = fields;
        return this;
    }

    /**
     * Limit results
     */
    limit(count) {
        this.limitValue = count;
        return this;
    }

    /**
     * Skip results
     */
    offset(count) {
        this.offsetValue = count;
        return this;
    }

    /**
     * Select specific fields
     */
    select(...fields) {
        this.selectedFields = fields;
        return this;
    }

    /**
     * Join related data (simulate SQL joins)
     */
    join(relationName, relatedData) {
        this.joinedData.set(relationName, relatedData);
        return this;
    }

    /**
     * Execute query and return results
     */
    execute() {
        let results = [...this.data];

        // Apply filters
        results = results.filter(item =>
            this.conditions.every(condition => condition(item))
        );

        // Apply ordering
        if (this.orderFields.length > 0) {
            results.sort((a, b) => {
                for (const field of this.orderFields) {
                    const isDescending = field.startsWith('-');
                    const fieldName = isDescending ? field.substring(1) : field;

                    const aValue = this._getNestedField(a, fieldName);
                    const bValue = this._getNestedField(b, fieldName);

                    let comparison = 0;
                    if (aValue < bValue) comparison = -1;
                    else if (aValue > bValue) comparison = 1;

                    if (isDescending) comparison *= -1;

                    if (comparison !== 0) return comparison;
                }
                return 0;
            });
        }

        // Apply offset and limit
        const start = this.offsetValue;
        const end = this.limitValue ? start + this.limitValue : undefined;
        results = results.slice(start, end);

        // Apply field selection
        if (this.selectedFields) {
            results = results.map(item => {
                const selected = {};
                this.selectedFields.forEach(field => {
                    selected[field] = this._getNestedField(item, field);
                });
                return selected;
            });
        }

        return results;
    }

    /**
     * Get count of matching records
     */
    count() {
        return this.data.filter(item =>
            this.conditions.every(condition => condition(item))
        ).length;
    }

    /**
     * Get aggregations
     */
    aggregate(operations) {
        const results = this.data.filter(item =>
            this.conditions.every(condition => condition(item))
        );

        const aggregated = {};

        Object.entries(operations).forEach(([key, operation]) => {
            if (operation.count) {
                aggregated[key] = results.length;
            } else if (operation.sum) {
                aggregated[key] = results.reduce((sum, item) =>
                    sum + (this._getNestedField(item, operation.sum) || 0), 0);
            } else if (operation.avg) {
                const values = results.map(item => this._getNestedField(item, operation.avg) || 0);
                aggregated[key] = values.length > 0 ? values.reduce((a, b) => a + b) / values.length : 0;
            } else if (operation.max) {
                const values = results.map(item => this._getNestedField(item, operation.max));
                aggregated[key] = values.length > 0 ? Math.max(...values) : null;
            } else if (operation.min) {
                const values = results.map(item => this._getNestedField(item, operation.min));
                aggregated[key] = values.length > 0 ? Math.min(...values) : null;
            }
        });

        return aggregated;
    }

    /**
     * Helper to get nested field value
     */
    _getNestedField(obj, field) {
        return field.split('.').reduce((current, key) =>
            current && current[key] !== undefined ? current[key] : null, obj);
    }
}

/**
 * Mock ORM Manager for database-like operations
 */
class ORMManager {
    constructor() {
        this.courts = [];
        this.judges = [];
        this.legalAreas = [];
        this.judgments = [];
        this.citations = [];

        this._seedData();
    }

    /**
     * Create query builder for judgments
     */
    queryJudgments() {
        return new QueryBuilder(Judgment, this.judgments);
    }

    /**
     * Create query builder for courts
     */
    queryCourts() {
        return new QueryBuilder(Court, this.courts);
    }

    /**
     * Create query builder for judges
     */
    queryJudges() {
        return new QueryBuilder(Judge, this.judges);
    }

    /**
     * Create query builder for legal areas
     */
    queryLegalAreas() {
        return new QueryBuilder(LegalArea, this.legalAreas);
    }

    /**
     * Seed with sample data
     */
    _seedData() {
        // Create courts
        const uksc = new Court('UKSC', 'Supreme Court', 1, 'UK');
        const ukhl = new Court('UKHL', 'House of Lords', 1, 'UK');
        const ewca = new Court('EWCA', 'Court of Appeal', 2, 'England and Wales');
        const ewhc = new Court('EWHC', 'High Court', 3, 'England and Wales');

        this.courts = [uksc, ukhl, ewca, ewhc];

        // Create legal areas
        const constitutional = new LegalArea('Constitutional Law', 'Constitutional and administrative law');
        const humanRights = new LegalArea('Human Rights', 'Human rights and civil liberties');
        const contract = new LegalArea('Contract Law', 'Contract and commercial law');
        const tort = new LegalArea('Tort Law', 'Tort and personal injury law');

        this.legalAreas = [constitutional, humanRights, contract, tort];

        // Create judges
        const lordReed = new Judge('Reed', 'Lord', '2012-02-06', uksc);
        const lordHodge = new Judge('Hodge', 'Lord', '2013-10-01', uksc);
        const lordJusticeUnderhill = new Judge('Underhill', 'Lord Justice', '2013-10-01', ewca);

        this.judges = [lordReed, lordHodge, lordJusticeUnderhill];

        // Create sample judgments
        const judgment1 = new Judgment(
            '[2023] UKSC 15',
            'R (Miller) v Prime Minister',
            'https://caselaw.nationalarchives.gov.uk/uksc/2023/15',
            uksc,
            '2023-05-15'
        );
        judgment1.summary = 'Constitutional law case about parliamentary sovereignty';
        judgment1.wordCount = 12500;
        judgment1.relevanceScore = 0.95;
        judgment1.published = true;
        judgment1.judges = [lordReed, lordHodge];
        judgment1.legalAreas = [constitutional, humanRights];

        const judgment2 = new Judgment(
            '[2023] EWCA Civ 892',
            'Smith v Secretary of State',
            'https://caselaw.nationalarchives.gov.uk/ewca/civ/2023/892',
            ewca,
            '2023-08-22'
        );
        judgment2.summary = 'Administrative law and judicial review';
        judgment2.wordCount = 8900;
        judgment2.relevanceScore = 0.87;
        judgment2.published = true;
        judgment2.judges = [lordJusticeUnderhill];
        judgment2.legalAreas = [constitutional];

        const judgment3 = new Judgment(
            '[2023] EWHC 1456 (Admin)',
            'Jones v Local Authority',
            'https://caselaw.nationalarchives.gov.uk/ewhc/admin/2023/1456',
            ewhc,
            '2023-06-30'
        );
        judgment3.summary = 'Human rights and procedural fairness';
        judgment3.wordCount = 6700;
        judgment3.relevanceScore = 0.73;
        judgment3.published = true;
        judgment3.legalAreas = [humanRights];

        this.judgments = [judgment1, judgment2, judgment3];

        // Create citations
        const citation1 = new Citation(judgment2, judgment1, 'Following the principles in Miller', 'followed', 45);
        this.citations = [citation1];

        // Link citations
        judgment1.citationsReceived = [citation1];
        judgment2.citationsMade = [citation1];
    }
}

/**
 * Legal Document Query Service
 * High-level service for common legal document queries
 */
class LegalDocumentQueryService {
    constructor(orm) {
        this.orm = orm;
    }

    /**
     * Get recent judgments with filters
     */
    getRecentJudgments(options = {}) {
        const {
            days = 30,
            courtCodes = null,
            minRelevance = 0.0,
            limit = 20
        } = options;

        const cutoffDate = new Date();
        cutoffDate.setDate(cutoffDate.getDate() - days);

        let query = this.orm.queryJudgments()
            .where({
                published: true,
                'judgmentDate': { gte: cutoffDate },
                'relevanceScore': { gte: minRelevance }
            });

        if (courtCodes) {
            query = query.where({ 'court.code': { in: courtCodes } });
        }

        return query
            .orderBy('-judgmentDate', '-relevanceScore')
            .limit(limit)
            .execute();
    }

    /**
     * Search judgments with text query
     */
    searchJudgments(searchText, options = {}) {
        const {
            legalAreas = null,
            courtLevels = null,
            minRelevance = 0.5,
            limit = 20
        } = options;

        let query = this.orm.queryJudgments()
            .where({
                published: true,
                'relevanceScore': { gte: minRelevance }
            });

        // Text search across multiple fields
        if (searchText) {
            query = query.filter(judgment => {\n                const searchLower = searchText.toLowerCase();\n                return (\n                    judgment.caseName.toLowerCase().includes(searchLower) ||\n                    judgment.summary.toLowerCase().includes(searchLower) ||\n                    judgment.neutralCitation.toLowerCase().includes(searchLower)\n                );\n            });\n        }\n\n        // Filter by legal areas\n        if (legalAreas) {\n            query = query.filter(judgment => \n                judgment.legalAreas.some(area => legalAreas.includes(area.name))\n            );\n        }\n\n        // Filter by court levels\n        if (courtLevels) {\n            query = query.where({ 'court.level': { in: courtLevels } });\n        }\n\n        return query\n            .orderBy('-relevanceScore', '-judgmentDate')\n            .limit(limit)\n            .execute();\n    }\n\n    /**\n     * Get court statistics\n     */\n    getCourtStatistics(startDate, endDate) {\n        const judgments = this.orm.queryJudgments()\n            .where({\n                published: true,\n                'judgmentDate': { gte: startDate },\n                'judgmentDate': { lte: endDate }\n            })\n            .execute();\n\n        // Group by court\n        const courtStats = new Map();\n\n        judgments.forEach(judgment => {\n            const courtCode = judgment.court.code;\n            if (!courtStats.has(courtCode)) {\n                courtStats.set(courtCode, {\n                    courtCode,\n                    courtName: judgment.court.name,\n                    courtLevel: judgment.court.level,\n                    judgmentCount: 0,\n                    totalWordCount: 0,\n                    totalRelevance: 0,\n                    maxRelevance: 0,\n                    citationsMade: 0,\n                    citationsReceived: 0,\n                    uniqueJudges: new Set(),\n                    uniqueLegalAreas: new Set()\n                });\n            }\n\n            const stats = courtStats.get(courtCode);\n            stats.judgmentCount++;\n            stats.totalWordCount += judgment.wordCount;\n            stats.totalRelevance += judgment.relevanceScore;\n            stats.maxRelevance = Math.max(stats.maxRelevance, judgment.relevanceScore);\n            stats.citationsMade += judgment.citationsMade.length;\n            stats.citationsReceived += judgment.citationsReceived.length;\n            \n            judgment.judges.forEach(judge => stats.uniqueJudges.add(judge.name));\n            judgment.legalAreas.forEach(area => stats.uniqueLegalAreas.add(area.name));\n        });\n\n        // Convert to final format\n        return Array.from(courtStats.values()).map(stats => ({\n            ...stats,\n            avgWordCount: stats.totalWordCount / stats.judgmentCount,\n            avgRelevance: stats.totalRelevance / stats.judgmentCount,\n            uniqueJudgeCount: stats.uniqueJudges.size,\n            uniqueLegalAreaCount: stats.uniqueLegalAreas.size,\n            uniqueJudges: undefined,\n            uniqueLegalAreas: undefined\n        })).sort((a, b) => a.courtLevel - b.courtLevel);\n    }\n\n    /**\n     * Get citation network for a judgment\n     */\n    getCitationNetwork(judgmentId, depth = 2) {\n        const targetJudgment = this.orm.queryJudgments()\n            .where({ neutralCitation: judgmentId })\n            .execute()[0];\n\n        if (!targetJudgment) {\n            return null;\n        }\n\n        return {\n            centerJudgment: {\n                id: targetJudgment.neutralCitation,\n                caseName: targetJudgment.caseName,\n                court: targetJudgment.court.code,\n                date: targetJudgment.judgmentDate\n            },\n            citingJudgments: targetJudgment.citationsReceived.map(citation => ({\n                id: citation.citingJudgment.neutralCitation,\n                caseName: citation.citingJudgment.caseName,\n                court: citation.citingJudgment.court.code,\n                citationType: citation.citationType,\n                context: citation.context\n            })),\n            citedJudgments: targetJudgment.citationsMade.map(citation => ({\n                id: citation.citedJudgment.neutralCitation,\n                caseName: citation.citedJudgment.caseName,\n                court: citation.citedJudgment.court.code,\n                citationType: citation.citationType,\n                context: citation.context\n            })),\n            depth\n        };\n    }\n\n    /**\n     * Get aggregated metrics by different dimensions\n     */\n    getAggregatedMetrics(groupBy = 'court') {\n        const judgments = this.orm.queryJudgments()\n            .where({ published: true })\n            .execute();\n\n        if (groupBy === 'court') {\n            return this._groupByCourt(judgments);\n        } else if (groupBy === 'legalArea') {\n            return this._groupByLegalArea(judgments);\n        } else if (groupBy === 'year') {\n            return this._groupByYear(judgments);\n        } else if (groupBy === 'judge') {\n            return this._groupByJudge(judgments);\n        } else {\n            throw new Error(`Unsupported groupBy value: ${groupBy}`);\n        }\n    }\n\n    /**\n     * Complex query with multiple filters\n     */\n    complexQuery(filters) {\n        let query = this.orm.queryJudgments().where({ published: true });\n\n        // Date range\n        if (filters.startDate && filters.endDate) {\n            query = query.where({\n                'judgmentDate': { gte: filters.startDate },\n                'judgmentDate': { lte: filters.endDate }\n            });\n        }\n\n        // Court levels\n        if (filters.courtLevels) {\n            query = query.where({ 'court.level': { in: filters.courtLevels } });\n        }\n\n        // Text search\n        if (filters.searchText) {\n            query = query.filter(judgment => {\n                const searchLower = filters.searchText.toLowerCase();\n                return (\n                    judgment.caseName.toLowerCase().includes(searchLower) ||\n                    judgment.summary.toLowerCase().includes(searchLower)\n                );\n            });\n        }\n\n        // Legal areas\n        if (filters.legalAreas) {\n            query = query.filter(judgment => \n                judgment.legalAreas.some(area => filters.legalAreas.includes(area.name))\n            );\n        }\n\n        // Word count range\n        if (filters.minWords) {\n            query = query.where({ 'wordCount': { gte: filters.minWords } });\n        }\n        if (filters.maxWords) {\n            query = query.where({ 'wordCount': { lte: filters.maxWords } });\n        }\n\n        // Relevance threshold\n        if (filters.minRelevance) {\n            query = query.where({ 'relevanceScore': { gte: filters.minRelevance } });\n        }\n\n        // Apply ordering\n        const orderBy = filters.orderBy || ['-judgmentDate', '-relevanceScore'];\n        query = query.orderBy(...orderBy);\n\n        return query.execute();\n    }\n\n    /**\n     * Group judgments by court\n     */\n    _groupByCourt(judgments) {\n        const groups = new Map();\n\n        judgments.forEach(judgment => {\n            const key = judgment.court.code;\n            if (!groups.has(key)) {\n                groups.set(key, {\n                    courtCode: judgment.court.code,\n                    courtName: judgment.court.name,\n                    courtLevel: judgment.court.level,\n                    judgments: []\n                });\n            }\n            groups.get(key).judgments.push(judgment);\n        });\n\n        return Array.from(groups.values()).map(group => ({\n            ...group,\n            judgmentCount: group.judgments.length,\n            avgWordCount: group.judgments.reduce((sum, j) => sum + j.wordCount, 0) / group.judgments.length,\n            avgRelevance: group.judgments.reduce((sum, j) => sum + j.relevanceScore, 0) / group.judgments.length,\n            judgments: undefined\n        }));\n    }\n\n    /**\n     * Group judgments by legal area\n     */\n    _groupByLegalArea(judgments) {\n        const groups = new Map();\n\n        judgments.forEach(judgment => {\n            judgment.legalAreas.forEach(area => {\n                if (!groups.has(area.name)) {\n                    groups.set(area.name, {\n                        legalAreaName: area.name,\n                        judgments: []\n                    });\n                }\n                groups.get(area.name).judgments.push(judgment);\n            });\n        });\n\n        return Array.from(groups.values()).map(group => ({\n            ...group,\n            judgmentCount: group.judgments.length,\n            avgWordCount: group.judgments.reduce((sum, j) => sum + j.wordCount, 0) / group.judgments.length,\n            avgRelevance: group.judgments.reduce((sum, j) => sum + j.relevanceScore, 0) / group.judgments.length,\n            judgments: undefined\n        }));\n    }\n\n    /**\n     * Group judgments by year\n     */\n    _groupByYear(judgments) {\n        const groups = new Map();\n\n        judgments.forEach(judgment => {\n            const year = judgment.judgmentDate.getFullYear();\n            if (!groups.has(year)) {\n                groups.set(year, {\n                    year,\n                    judgments: []\n                });\n            }\n            groups.get(year).judgments.push(judgment);\n        });\n\n        return Array.from(groups.values()).map(group => ({\n            ...group,\n            judgmentCount: group.judgments.length,\n            avgWordCount: group.judgments.reduce((sum, j) => sum + j.wordCount, 0) / group.judgments.length,\n            avgRelevance: group.judgments.reduce((sum, j) => sum + j.relevanceScore, 0) / group.judgments.length,\n            judgments: undefined\n        })).sort((a, b) => a.year - b.year);\n    }\n\n    /**\n     * Group judgments by judge\n     */\n    _groupByJudge(judgments) {\n        const groups = new Map();\n\n        judgments.forEach(judgment => {\n            judgment.judges.forEach(judge => {\n                const key = judge.name;\n                if (!groups.has(key)) {\n                    groups.set(key, {\n                        judgeName: judge.name,\n                        judgeTitle: judge.title,\n                        courtName: judge.court.name,\n                        judgments: []\n                    });\n                }\n                groups.get(key).judgments.push(judgment);\n            });\n        });\n\n        return Array.from(groups.values()).map(group => ({\n            ...group,\n            judgmentCount: group.judgments.length,\n            totalWordCount: group.judgments.reduce((sum, j) => sum + j.wordCount, 0),\n            avgWordCount: group.judgments.reduce((sum, j) => sum + j.wordCount, 0) / group.judgments.length,\n            recentJudgment: Math.max(...group.judgments.map(j => j.judgmentDate.getTime())),\n            judgments: undefined\n        })).sort((a, b) => b.judgmentCount - a.judgmentCount);\n    }\n}\n\n/**\n * Query optimization examples\n */\nclass QueryOptimizationExamples {\n    constructor(orm) {\n        this.orm = orm;\n    }\n\n    /**\n     * Demonstrate efficient data loading patterns\n     */\n    efficientDataLoading() {\n        // Bad: Multiple queries in loop\n        console.log('L Inefficient: N+1 query pattern');\n        const judgments = this.orm.queryJudgments().execute();\n        judgments.forEach(judgment => {\n            // This would cause additional queries for each judgment\n            console.log(`${judgment.caseName} - ${judgment.court.name}`);\n        });\n\n        // Good: Eager loading with joins\n        console.log(' Efficient: Eager loading with joins');\n        const efficientQuery = this.orm.queryJudgments()\n            .join('court', this.orm.courts)\n            .join('judges', this.orm.judges)\n            .execute();\n\n        return efficientQuery;\n    }\n\n    /**\n     * Database-level vs application-level aggregations\n     */\n    aggregationComparison() {\n        // Bad: Application-level aggregation\n        console.log('L Inefficient: Application-level aggregation');\n        const allJudgments = this.orm.queryJudgments().execute();\n        const totalWords = allJudgments.reduce((sum, j) => sum + j.wordCount, 0);\n        const avgWords = totalWords / allJudgments.length;\n\n        // Good: Database-level aggregation\n        console.log(' Efficient: Database-level aggregation');\n        const stats = this.orm.queryJudgments().aggregate({\n            totalJudgments: { count: true },\n            totalWords: { sum: 'wordCount' },\n            avgWords: { avg: 'wordCount' },\n            maxRelevance: { max: 'relevanceScore' }\n        });\n\n        return { applicationLevel: { totalWords, avgWords }, databaseLevel: stats };\n    }\n\n    /**\n     * Efficient filtering with indexes\n     */\n    efficientFiltering() {\n        // These patterns would benefit from database indexes\n        const recentHighRelevance = this.orm.queryJudgments()\n            .where({\n                'judgmentDate': { gte: new Date('2023-01-01') },\n                'relevanceScore': { gte: 0.8 },\n                published: true\n            })\n            .orderBy('-judgmentDate')\n            .execute();\n\n        return recentHighRelevance;\n    }\n\n    /**\n     * Pagination best practices\n     */\n    efficientPagination(page = 1, pageSize = 10) {\n        const offset = (page - 1) * pageSize;\n\n        const results = this.orm.queryJudgments()\n            .where({ published: true })\n            .orderBy('-judgmentDate', 'neutralCitation')\n            .offset(offset)\n            .limit(pageSize)\n            .execute();\n\n        const totalCount = this.orm.queryJudgments()\n            .where({ published: true })\n            .count();\n\n        return {\n            results,\n            pagination: {\n                page,\n                pageSize,\n                totalCount,\n                totalPages: Math.ceil(totalCount / pageSize),\n                hasNextPage: page * pageSize < totalCount,\n                hasPreviousPage: page > 1\n            }\n        };\n    }\n}\n\n/**\n * Demonstrate mock ORM patterns with legal data\n */\nfunction demonstrateMockORM() {\n    console.log('=== Mock ORM Query Examples for Legal Data ===\\n');\n\n    // Initialize ORM and service\n    const orm = new ORMManager();\n    const queryService = new LegalDocumentQueryService(orm);\n    const optimization = new QueryOptimizationExamples(orm);\n\n    console.log('1. RECENT JUDGMENTS QUERY:');\n    const recentJudgments = queryService.getRecentJudgments({\n        days: 365,\n        courtCodes: ['UKSC', 'EWCA'],\n        minRelevance: 0.8,\n        limit: 10\n    });\n    console.log(`   Found ${recentJudgments.length} recent high-relevance judgments`);\n    recentJudgments.forEach(judgment => {\n        console.log(`     ${judgment.neutralCitation}: ${judgment.caseName}`);\n    });\n\n    console.log('\\n2. FULL-TEXT SEARCH:');\n    const searchResults = queryService.searchJudgments('constitutional', {\n        legalAreas: ['Constitutional Law'],\n        courtLevels: [1, 2],\n        minRelevance: 0.7\n    });\n    console.log(`   Found ${searchResults.length} judgments matching 'constitutional'`);\n    searchResults.forEach(judgment => {\n        console.log(`     ${judgment.neutralCitation}: relevance ${judgment.relevanceScore}`);\n    });\n\n    console.log('\\n3. COURT STATISTICS:');\n    const courtStats = queryService.getCourtStatistics(\n        new Date('2023-01-01'),\n        new Date('2023-12-31')\n    );\n    console.log('   Court statistics for 2023:');\n    courtStats.forEach(stat => {\n        console.log(`     ${stat.courtCode}: ${stat.judgmentCount} judgments, ` +\n                   `avg ${Math.round(stat.avgWordCount)} words`);\n    });\n\n    console.log('\\n4. CITATION NETWORK:');\n    const citationNetwork = queryService.getCitationNetwork('[2023] UKSC 15');\n    if (citationNetwork) {\n        console.log(`   Citation network for ${citationNetwork.centerJudgment.id}:`);\n        console.log(`     Citing judgments: ${citationNetwork.citingJudgments.length}`);\n        console.log(`     Cited judgments: ${citationNetwork.citedJudgments.length}`);\n    }\n\n    console.log('\\n5. AGGREGATED METRICS BY COURT:');\n    const courtMetrics = queryService.getAggregatedMetrics('court');\n    console.log('   Metrics grouped by court:');\n    courtMetrics.forEach(metric => {\n        console.log(`     ${metric.courtCode}: ${metric.judgmentCount} judgments, ` +\n                   `avg relevance ${metric.avgRelevance.toFixed(3)}`);\n    });\n\n    console.log('\\n6. COMPLEX QUERY EXAMPLE:');\n    const complexResults = queryService.complexQuery({\n        startDate: new Date('2023-01-01'),\n        endDate: new Date('2023-12-31'),\n        courtLevels: [1, 2],\n        searchText: 'constitutional',\n        legalAreas: ['Constitutional Law', 'Human Rights'],\n        minWords: 5000,\n        minRelevance: 0.8,\n        orderBy: ['-relevanceScore', '-judgmentDate']\n    });\n    console.log(`   Complex query returned ${complexResults.length} results`);\n\n    console.log('\\n7. QUERY OPTIMIZATION EXAMPLES:');\n    console.log('   Efficient data loading:');\n    const efficientResults = optimization.efficientDataLoading();\n    console.log(`     Loaded ${efficientResults.length} judgments with related data`);\n\n    console.log('\\n   Aggregation comparison:');\n    const aggComparison = optimization.aggregationComparison();\n    console.log(`     Database aggregation: ${JSON.stringify(aggComparison.databaseLevel)}`);\n\n    console.log('\\n   Efficient pagination:');\n    const paginatedResults = optimization.efficientPagination(1, 2);\n    console.log(`     Page 1 of ${paginatedResults.pagination.totalPages} ` +\n               `(${paginatedResults.pagination.totalCount} total)`);\n\n    console.log('\\n8. BEST PRACTICES SUMMARY:');\n    console.log('    Use proper indexing on frequently queried fields');\n    console.log('    Implement eager loading to avoid N+1 queries');\n    console.log('    Perform aggregations at database level');\n    console.log('    Use efficient pagination with offset/limit');\n    console.log('    Implement query result caching for expensive operations');\n    console.log('    Consider database connection pooling');\n    console.log('    Monitor query performance and optimize slow queries');\n\n    return {\n        orm,\n        queryService,\n        results: {\n            recentJudgments,\n            searchResults,\n            courtStats,\n            citationNetwork,\n            courtMetrics,\n            complexResults\n        }\n    };\n}\n\n// Export classes and functions for use in other modules\nif (typeof module !== 'undefined' && module.exports) {\n    module.exports = {\n        Court,\n        Judge,\n        LegalArea,\n        Judgment,\n        Citation,\n        QueryBuilder,\n        ORMManager,\n        LegalDocumentQueryService,\n        QueryOptimizationExamples,\n        demonstrateMockORM\n    };\n}\n\n// Run demonstration if this file is executed directly\nif (typeof require !== 'undefined' && require.main === module) {\n    demonstrateMockORM();\n}"