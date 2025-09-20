/**
 * NLP Token Trees for Legal Text Analysis
 * ======================================
 *
 * This module implements parse trees and token trees for natural language processing
 * of legal documents in Find Case Law (FCL). Provides structured representation
 * of legal text for advanced analysis, citation extraction, and semantic understanding.
 *
 * Key FCL Use Cases:
 * - Parsing legal document structure (paragraphs, sections, citations)
 * - Extracting entity relationships and legal concepts
 * - Building dependency trees for complex legal sentences
 * - Supporting advanced search with syntactic understanding
 * - Enabling structured data extraction from judgment text
 */

// Types of tokens in legal text
const TokenType = {
    WORD: 'word',
    PUNCTUATION: 'punctuation',
    CITATION: 'citation',
    STATUTE: 'statute',
    CASE_NAME: 'case_name',
    COURT: 'court',
    DATE: 'date',
    PARAGRAPH_REF: 'paragraph_ref',
    SECTION_REF: 'section_ref',
    LEGAL_CONCEPT: 'legal_concept',
    ENTITY: 'entity'
};

// Types of parse tree nodes
const TreeNodeType = {
    ROOT: 'root',
    SENTENCE: 'sentence',
    PHRASE: 'phrase',
    CLAUSE: 'clause',
    NOUN_PHRASE: 'noun_phrase',
    VERB_PHRASE: 'verb_phrase',
    PREPOSITIONAL_PHRASE: 'prepositional_phrase',
    CITATION_PHRASE: 'citation_phrase',
    LEGAL_PRINCIPLE: 'legal_principle'
};

/**
 * Represents a single token in legal text
 */
class Token {
    constructor(text, tokenType, position, lemma = null, posTag = null) {
        this.text = text;
        this.tokenType = tokenType;
        this.position = position;
        this.lemma = lemma;
        this.posTag = posTag;
        this.dependencies = [];
        this.confidence = 1.0;
        this.metadata = {};
    }

    toString() {
        return `${this.text} (${this.tokenType})`;
    }
}

/**
 * Node in a parse tree for legal text structure
 */
class ParseTreeNode {
    constructor(nodeType, content) {
        this.nodeType = nodeType;
        this.content = content;
        this.tokens = [];
        this.children = [];
        this.parent = null;
        this.attributes = {};
        this.span = [0, 0]; // Start and end positions
    }

    /**
     * Add a child node
     */
    addChild(child) {
        child.parent = this;
        this.children.push(child);
    }

    /**
     * Get the text content of this node and its children
     */
    getText() {
        if (this.tokens.length > 0) {
            return this.tokens.map(token => token.text).join(' ');
        } else if (this.children.length > 0) {
            return this.children.map(child => child.getText()).join(' ');
        } else {
            return this.content;
        }
    }

    /**
     * Find all descendant nodes of a specific type
     */
    findNodesByType(nodeType) {
        const result = [];
        if (this.nodeType === nodeType) {
            result.push(this);
        }

        for (const child of this.children) {
            result.push(...child.findNodesByType(nodeType));
        }

        return result;
    }

    /**
     * Extract all citation tokens from this node
     */
    getCitations() {
        const citations = [];

        // Check direct tokens
        for (const token of this.tokens) {
            if (token.tokenType === TokenType.CITATION) {
                citations.push(token);
            }
        }

        // Check children
        for (const child of this.children) {
            citations.push(...child.getCitations());
        }

        return citations;
    }

    toString() {
        return `${this.nodeType}: ${this.content.substring(0, 50)}...`;
    }
}

/**
 * Specialized tokenizer for legal text that recognizes legal-specific patterns.
 */
class LegalTokenizer {
    constructor() {
        // Legal citation patterns
        this.citationPatterns = [
            /\[(\d{4})\]\s+([A-Z]{2,})\s+(\d+)/g, // [2023] UKSC 15
            /\[(\d{4})\]\s+([A-Z]{2,})\s+([A-Za-z]+)\s+(\d+)/g, // [2023] EWCA Civ 892
            /\((\d{4})\)\s+([A-Z]{2,})\s+(\d+)/g, // (2023) AC 123
            /(\d+)\s+([A-Z]{2,})\s+(\d+)/g // 2023 AC 123
        ];

        // Court patterns
        this.courtPatterns = [
            /\b(Supreme Court|House of Lords|Court of Appeal|High Court)\b/gi,
            /\b(UKSC|UKHL|EWCA|EWHC|UKUT|UKFTT)\b/gi,
            /\b(Divisional Court|Crown Court|Magistrates[']?\s*Court)\b/gi
        ];

        // Legal concept patterns
        this.legalConcepts = new Set([
            'negligence', 'tort', 'contract', 'breach', 'damages', 'liability',
            'jurisdiction', 'statutory', 'common law', 'precedent', 'ratio',
            'obiter', 'judgment', 'appeal', 'cross-appeal', 'permission',
            'judicial review', 'human rights', 'constitutional', 'administrative',
            'procedural', 'substantive', 'reasonable', 'proportionate'
        ]);

        // Paragraph and section reference patterns
        this.refPatterns = [
            /\[(\d+)\]/g, // [123]
            /para(?:graph)?\s*(\d+)/gi, // paragraph 123
            /section\s*(\d+)/gi, // section 123
            /s\.?\s*(\d+)/gi // s. 123 or s 123
        ];
    }

    /**
     * Tokenize legal text into structured tokens
     */
    tokenize(text) {
        const tokens = [];
        let position = 0;

        // Simple word-based tokenization with legal pattern recognition
        const words = text.match(/\S+/g) || [];

        for (const word of words) {
            const tokenType = this._classifyToken(word, text, position);

            // Create base token
            const token = new Token(
                word,
                tokenType,
                position,
                this._getLemma(word),
                this._getPosTag(word)
            );

            // Add metadata for specific token types
            if (tokenType === TokenType.CITATION) {
                token.metadata.citationType = this._getCitationType(word);
            } else if (tokenType === TokenType.COURT) {
                token.metadata.courtLevel = this._getCourtLevel(word);
            }

            tokens.push(token);
            position += 1;
        }

        return tokens;
    }

    /**
     * Classify a token based on legal text patterns
     */
    _classifyToken(word, fullText, position) {
        // Remove punctuation for classification
        const cleanWord = word.replace(/[^\w\s]/g, '').toLowerCase();

        // Check for citations
        for (const pattern of this.citationPatterns) {
            pattern.lastIndex = 0; // Reset regex
            if (pattern.test(word)) {
                return TokenType.CITATION;
            }
        }

        // Check for courts
        for (const pattern of this.courtPatterns) {
            pattern.lastIndex = 0; // Reset regex
            if (pattern.test(word)) {
                return TokenType.COURT;
            }
        }

        // Check for legal concepts
        if (this.legalConcepts.has(cleanWord)) {
            return TokenType.LEGAL_CONCEPT;
        }

        // Check for references
        for (const pattern of this.refPatterns) {
            pattern.lastIndex = 0; // Reset regex
            if (pattern.test(word)) {
                return TokenType.PARAGRAPH_REF;
            }
        }

        // Check for dates
        if (/\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}/.test(word)) {
            return TokenType.DATE;
        }

        // Check for punctuation
        if (/^[^\w\s]+$/.test(word)) {
            return TokenType.PUNCTUATION;
        }

        // Default to word
        return TokenType.WORD;
    }

    /**
     * Get lemmatized form of word (simplified)
     */
    _getLemma(word) {
        // Simple lemmatization for common legal terms
        const lemmaMap = {
            'courts': 'court',
            'judgments': 'judgment',
            'appeals': 'appeal',
            'applications': 'application',
            'decisions': 'decision',
            'orders': 'order',
            'proceedings': 'proceeding'
        };
        return lemmaMap[word.toLowerCase()] || word.toLowerCase();
    }

    /**
     * Get part-of-speech tag (simplified)
     */
    _getPosTag(word) {
        // Simple POS tagging based on patterns
        if (/^[A-Z][a-z]+$/.test(word)) {
            return 'NOUN';
        } else if (['the', 'a', 'an'].includes(word.toLowerCase())) {
            return 'DET';
        } else if (['and', 'or', 'but'].includes(word.toLowerCase())) {
            return 'CONJ';
        } else if (/^[^\w\s]+$/.test(word)) {
            return 'PUNCT';
        } else {
            return 'WORD';
        }
    }

    /**
     * Determine the type of citation
     */
    _getCitationType(word) {
        if (word.includes('UKSC') || word.includes('UKHL')) {
            return 'supreme_court';
        } else if (word.includes('EWCA')) {
            return 'court_of_appeal';
        } else if (word.includes('EWHC')) {
            return 'high_court';
        } else if (word.includes('UKUT') || word.includes('UKFTT')) {
            return 'tribunal';
        } else {
            return 'other';
        }
    }

    /**
     * Determine court hierarchy level
     */
    _getCourtLevel(word) {
        const wordUpper = word.toUpperCase();
        if (['UKSC', 'SUPREME'].some(court => wordUpper.includes(court))) {
            return 'supreme';
        } else if (['UKHL', 'LORDS'].some(court => wordUpper.includes(court))) {
            return 'lords';
        } else if (['EWCA', 'APPEAL'].some(court => wordUpper.includes(court))) {
            return 'appeal';
        } else if (['EWHC', 'HIGH'].some(court => wordUpper.includes(court))) {
            return 'high';
        } else {
            return 'lower';
        }
    }
}

/**
 * Parse tree for legal document structure analysis.
 * Builds hierarchical representation of legal text.
 */
class LegalParseTree {
    constructor() {
        this.root = new ParseTreeNode(TreeNodeType.ROOT, 'Legal Document');
        this.tokenizer = new LegalTokenizer();
    }

    /**
     * Parse a legal document into a structured tree
     */
    parseDocument(text) {
        // Reset root
        this.root = new ParseTreeNode(TreeNodeType.ROOT, 'Legal Document');

        // Split into sentences
        const sentences = this._splitSentences(text);

        sentences.forEach((sentenceText, i) => {
            const sentenceNode = this._parseSentence(sentenceText, i);
            this.root.addChild(sentenceNode);
        });

        return this.root;
    }

    /**
     * Split text into sentences, handling legal text patterns
     */
    _splitSentences(text) {
        // Enhanced sentence splitting for legal text
        // Handle abbreviations and citations properly
        const sentences = [];

        // Simple sentence splitting (can be enhanced with more sophisticated rules)
        const parts = text.split(/(?<=[.!?])\s+(?=[A-Z])/);

        for (const part of parts) {
            const trimmed = part.trim();
            if (trimmed) {
                sentences.push(trimmed);
            }
        }

        return sentences;
    }

    /**
     * Parse a single sentence into structured components
     */
    _parseSentence(sentence, sentenceId) {
        const sentenceNode = new ParseTreeNode(
            TreeNodeType.SENTENCE,
            sentence
        );
        sentenceNode.span = [0, sentence.length];

        // Tokenize the sentence
        const tokens = this.tokenizer.tokenize(sentence);
        sentenceNode.tokens = tokens;

        // Extract structured components
        this._extractPhrases(sentenceNode, tokens);

        return sentenceNode;
    }

    /**
     * Extract phrases and legal constructs from tokens
     */
    _extractPhrases(sentenceNode, tokens) {
        let i = 0;
        while (i < tokens.length) {
            // Try to identify citation phrases
            const citationPhrase = this._extractCitationPhrase(tokens, i);
            if (citationPhrase) {
                sentenceNode.addChild(citationPhrase);
                i += citationPhrase.tokens.length;
                continue;
            }

            // Try to identify legal principle phrases
            const legalPhrase = this._extractLegalPhrase(tokens, i);
            if (legalPhrase) {
                sentenceNode.addChild(legalPhrase);
                i += legalPhrase.tokens.length;
                continue;
            }

            // Default: create word-level nodes
            if (tokens[i].tokenType !== TokenType.PUNCTUATION) {
                const wordNode = new ParseTreeNode(
                    TreeNodeType.PHRASE,
                    tokens[i].text
                );
                wordNode.tokens = [tokens[i]];
                sentenceNode.addChild(wordNode);
            }

            i += 1;
        }
    }

    /**
     * Extract citation phrases from token sequence
     */
    _extractCitationPhrase(tokens, start) {
        const citationTokens = [];
        let i = start;

        // Look for citation patterns
        while (i < tokens.length && i < start + 10) { // Limit search window
            const token = tokens[i];

            if (token.tokenType === TokenType.CITATION) {
                citationTokens.push(token);
                i += 1;

                // Check for additional citation components
                while (i < tokens.length && i < start + 10) {
                    const nextToken = tokens[i];
                    if ([TokenType.WORD, TokenType.PUNCTUATION].includes(nextToken.tokenType) &&
                        ['v', 'vs', '(', ')'].some(pattern => nextToken.text.includes(pattern))) {
                        citationTokens.push(nextToken);
                        i += 1;
                    } else {
                        break;
                    }
                }

                if (citationTokens.length > 0) {
                    const content = citationTokens.map(t => t.text).join(' ');
                    const node = new ParseTreeNode(
                        TreeNodeType.CITATION_PHRASE,
                        content
                    );
                    node.tokens = citationTokens;
                    return node;
                }
            }
            i += 1;
        }

        return null;
    }

    /**
     * Extract legal principle or concept phrases
     */
    _extractLegalPhrase(tokens, start) {
        const legalTokens = [];
        let i = start;

        // Look for sequences of legal concepts
        while (i < tokens.length && i < start + 8) {
            const token = tokens[i];

            if (token.tokenType === TokenType.LEGAL_CONCEPT) {
                legalTokens.push(token);

                // Include surrounding context words
                let j = i + 1;
                while (j < tokens.length && j < start + 8) {
                    const nextToken = tokens[j];
                    if ([TokenType.WORD, TokenType.LEGAL_CONCEPT].includes(nextToken.tokenType) &&
                        !['the', 'a', 'an'].includes(nextToken.text.toLowerCase())) {
                        legalTokens.push(nextToken);
                        j += 1;
                    } else {
                        break;
                    }
                }

                if (legalTokens.length >= 2) { // Require at least 2 tokens for a phrase
                    const content = legalTokens.map(t => t.text).join(' ');
                    const node = new ParseTreeNode(
                        TreeNodeType.LEGAL_PRINCIPLE,
                        content
                    );
                    node.tokens = legalTokens;
                    return node;
                }
            }

            i += 1;
        }

        return null;
    }

    /**
     * Extract all citations from the parse tree
     */
    extractCitations() {
        const citationNodes = this.root.findNodesByType(TreeNodeType.CITATION_PHRASE);
        return citationNodes.map(node => node.getText());
    }

    /**
     * Extract all legal concepts and principles
     */
    extractLegalConcepts() {
        const legalNodes = this.root.findNodesByType(TreeNodeType.LEGAL_PRINCIPLE);
        return legalNodes.map(node => node.getText());
    }

    /**
     * Get parse tree statistics
     */
    getStatistics() {
        const totalNodes = this._countNodes(this.root);
        const sentences = this.root.findNodesByType(TreeNodeType.SENTENCE);
        const citations = this.root.findNodesByType(TreeNodeType.CITATION_PHRASE);
        const legalPrinciples = this.root.findNodesByType(TreeNodeType.LEGAL_PRINCIPLE);

        // Token type distribution
        const allTokens = [];
        for (const sentence of sentences) {
            allTokens.push(...sentence.tokens);
        }

        const tokenTypes = {};
        for (const token of allTokens) {
            tokenTypes[token.tokenType] = (tokenTypes[token.tokenType] || 0) + 1;
        }

        return {
            totalNodes,
            sentences: sentences.length,
            citations: citations.length,
            legalPrinciples: legalPrinciples.length,
            totalTokens: allTokens.length,
            tokenTypeDistribution: tokenTypes
        };
    }

    /**
     * Recursively count all nodes in tree
     */
    _countNodes(node) {
        let count = 1;
        for (const child of node.children) {
            count += this._countNodes(child);
        }
        return count;
    }

    /**
     * Convert parse tree to dictionary representation
     */
    toDict() {
        return this._nodeToDict(this.root);
    }

    /**
     * Convert a node to dictionary format
     */
    _nodeToDict(node) {
        return {
            type: node.nodeType,
            content: node.content,
            tokens: node.tokens.map(token => ({
                text: token.text,
                type: token.tokenType,
                position: token.position,
                lemma: token.lemma,
                posTag: token.posTag
            })),
            children: node.children.map(child => this._nodeToDict(child)),
            attributes: node.attributes,
            span: node.span
        };
    }
}

/**
 * Specialized entity extractor for legal documents.
 * Identifies and classifies legal entities, relationships, and concepts.
 */
class LegalEntityExtractor {
    constructor() {
        this.caseNamePatterns = [
            /([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+v\.?\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)/g,
            /R\s+v\.?\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)/g,
            /Re\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)/g
        ];

        this.statutePatterns = [
            /([A-Z][a-z\s]+Act)\s+(\d{4})/g,
            /(Human Rights Act|Data Protection Act|Companies Act)/g,
            /section\s+(\d+)\s+of\s+the\s+([A-Z][a-z\s]+Act\s+\d{4})/g
        ];
    }

    /**
     * Extract legal entities from parsed document
     */
    extractEntities(parseTree) {
        const entities = {
            caseNames: [],
            statutes: [],
            courts: [],
            citations: [],
            parties: [],
            judges: []
        };

        // Extract from each sentence
        const sentences = parseTree.root.findNodesByType(TreeNodeType.SENTENCE);

        for (const sentence of sentences) {
            const sentenceText = sentence.getText();

            // Extract case names
            for (const pattern of this.caseNamePatterns) {
                pattern.lastIndex = 0; // Reset regex
                let match;
                while ((match = pattern.exec(sentenceText)) !== null) {
                    entities.caseNames.push({
                        text: match[0],
                        parties: Array.from(match).slice(1),
                        span: [match.index, match.index + match[0].length]
                    });
                }
            }

            // Extract statutes
            for (const pattern of this.statutePatterns) {
                pattern.lastIndex = 0; // Reset regex
                let match;
                while ((match = pattern.exec(sentenceText)) !== null) {
                    entities.statutes.push({
                        text: match[0],
                        components: Array.from(match).slice(1),
                        span: [match.index, match.index + match[0].length]
                    });
                }
            }

            // Extract court mentions
            const courtTokens = sentence.tokens.filter(token => token.tokenType === TokenType.COURT);
            for (const token of courtTokens) {
                entities.courts.push({
                    text: token.text,
                    courtLevel: token.metadata.courtLevel || 'unknown',
                    position: token.position
                });
            }

            // Extract citations
            const citationTokens = sentence.tokens.filter(token => token.tokenType === TokenType.CITATION);
            for (const token of citationTokens) {
                entities.citations.push({
                    text: token.text,
                    citationType: token.metadata.citationType || 'unknown',
                    position: token.position
                });
            }
        }

        return entities;
    }
}

/**
 * Demonstrate NLP token trees with UK legal text
 */
function demonstrateNlpTokenTrees() {
    console.log('=== NLP Token Trees for Legal Text Analysis Demo ===\n');

    // Sample UK legal judgment text
    const legalText = `
    In the case of Smith v Secretary of State for Work and Pensions [2023] EWCA Civ 892,
    the Court of Appeal considered the application of judicial review principles to
    administrative decisions. The appellant argued that the respondent had failed to
    follow proper procedural requirements under section 12 of the Social Security Act 1998.

    Lord Justice Brown held that the test for negligence established in Donoghue v Stevenson
    [1932] AC 562 remains the cornerstone of tort law. The court found that reasonable
    care had not been exercised in this case.

    The Supreme Court in R (Miller) v Prime Minister [2023] UKSC 15 established important
    constitutional principles regarding parliamentary sovereignty and the separation of powers.
    `;

    // 1. Tokenization Demo
    console.log('1. LEGAL TEXT TOKENIZATION:');
    const tokenizer = new LegalTokenizer();

    const sampleSentence = 'Smith v Secretary of State [2023] EWCA Civ 892 applied negligence principles.';
    const tokens = tokenizer.tokenize(sampleSentence);

    console.log(`   Sample sentence: ${sampleSentence}`);
    console.log(`   Tokens:`);
    for (const token of tokens) {
        console.log(`     ${token.text.padEnd(15)} | ${token.tokenType.padEnd(12)} | ${token.lemma}`);
    }

    // 2. Parse Tree Construction
    console.log(`\n2. PARSE TREE CONSTRUCTION:`);
    const parser = new LegalParseTree();
    const parseTree = parser.parseDocument(legalText);

    console.log(`   Document parsed successfully`);
    const stats = parser.getStatistics();
    console.log(`   Statistics:`);
    for (const [key, value] of Object.entries(stats)) {
        if (key !== 'tokenTypeDistribution') {
            console.log(`     ${key}: ${value}`);
        }
    }

    console.log(`   Token type distribution:`);
    for (const [tokenType, count] of Object.entries(stats.tokenTypeDistribution)) {
        console.log(`     ${tokenType}: ${count}`);
    }

    // 3. Citation Extraction
    console.log(`\n3. CITATION EXTRACTION:`);
    const citations = parser.extractCitations();
    console.log(`   Found ${citations.length} citations:`);
    citations.forEach((citation, i) => {
        console.log(`     ${i + 1}. ${citation}`);
    });

    // 4. Legal Concept Extraction
    console.log(`\n4. LEGAL CONCEPT EXTRACTION:`);
    const legalConcepts = parser.extractLegalConcepts();
    console.log(`   Found ${legalConcepts.length} legal concepts:`);
    legalConcepts.forEach((concept, i) => {
        console.log(`     ${i + 1}. ${concept}`);
    });

    // 5. Entity Extraction
    console.log(`\n5. LEGAL ENTITY EXTRACTION:`);
    const extractor = new LegalEntityExtractor();
    const entities = extractor.extractEntities(parser);

    for (const [entityType, entityList] of Object.entries(entities)) {
        if (entityList.length > 0) {
            console.log(`   ${entityType.replace(/([A-Z])/g, ' $1').trim()}:`);
            entityList.forEach((entity, i) => {
                console.log(`     ${i + 1}. ${entity.text}`);
            });
        }
    }

    // 6. Parse Tree Analysis
    console.log(`\n6. PARSE TREE STRUCTURE ANALYSIS:`);
    const sentences = parseTree.findNodesByType(TreeNodeType.SENTENCE);
    console.log(`   Analyzed ${sentences.length} sentences:`);

    sentences.slice(0, 3).forEach((sentence, i) => { // Show first 3 sentences
        console.log(`\n   Sentence ${i + 1}: ${sentence.content.substring(0, 60)}...`);
        console.log(`     Child nodes: ${sentence.children.length}`);

        // Show citation and legal phrase children
        for (const child of sentence.children) {
            if ([TreeNodeType.CITATION_PHRASE, TreeNodeType.LEGAL_PRINCIPLE].includes(child.nodeType)) {
                console.log(`       ${child.nodeType}: ${child.content}`);
            }
        }
    });

    // 7. Token Analysis by Type
    console.log(`\n7. TOKEN ANALYSIS BY TYPE:`);
    const allSentences = parseTree.findNodesByType(TreeNodeType.SENTENCE);
    const tokenExamples = {};

    for (const sentence of allSentences) {
        for (const token of sentence.tokens) {
            if (!tokenExamples[token.tokenType]) {
                tokenExamples[token.tokenType] = [];
            }
            if (tokenExamples[token.tokenType].length < 3) { // Limit examples
                tokenExamples[token.tokenType].push(token.text);
            }
        }
    }

    for (const [tokenType, examples] of Object.entries(tokenExamples)) {
        if (examples.length > 0) {
            console.log(`   ${tokenType}: ${examples.join(', ')}`);
        }
    }

    // 8. Tree Serialization
    console.log(`\n8. PARSE TREE SERIALIZATION:`);
    const treeDict = parser.toDict();
    console.log(`   Tree structure serialized to dictionary`);
    console.log(`   Root node has ${treeDict.children.length} child sentences`);
    if (treeDict.children.length > 0) {
        console.log(`   First sentence has ${treeDict.children[0].tokens.length} tokens`);
    }

    // 9. Advanced Pattern Analysis
    console.log(`\n9. ADVANCED PATTERN ANALYSIS:`);

    // Find sentences with multiple citations
    const multiCitationSentences = [];
    for (const sentence of sentences) {
        const citationCount = sentence.getCitations().length;
        if (citationCount > 1) {
            multiCitationSentences.push([sentence, citationCount]);
        }
    }

    if (multiCitationSentences.length > 0) {
        console.log(`   Sentences with multiple citations:`);
        for (const [sentence, count] of multiCitationSentences) {
            console.log(`     ${count} citations: ${sentence.content.substring(0, 80)}...`);
        }
    }

    // Find legal principle chains
    const principleSentences = [];
    for (const sentence of sentences) {
        const legalNodes = sentence.findNodesByType(TreeNodeType.LEGAL_PRINCIPLE);
        if (legalNodes.length > 0) {
            principleSentences.push([sentence, legalNodes.length]);
        }
    }

    if (principleSentences.length > 0) {
        console.log(`\n   Sentences with legal principles:`);
        for (const [sentence, count] of principleSentences) {
            console.log(`     ${count} principles: ${sentence.content.substring(0, 80)}...`);
        }
    }

    return {
        parser,
        parseTree,
        entities,
        statistics: stats,
        examples: {
            tokens,
            citations,
            legalConcepts
        }
    };
}

// Export classes and functions for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        TokenType,
        TreeNodeType,
        Token,
        ParseTreeNode,
        LegalTokenizer,
        LegalParseTree,
        LegalEntityExtractor,
        demonstrateNlpTokenTrees
    };
}

// Run demonstration if this file is executed directly
if (typeof require !== 'undefined' && require.main === module) {
    demonstrateNlpTokenTrees();
}