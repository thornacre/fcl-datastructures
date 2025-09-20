/**
 * DOM Trees for LegalDocML/XML Structure Parsing (JavaScript)
 * ==========================================================
 *
 * This module demonstrates DOM tree manipulation for parsing UK legal documents
 * in LegalDocML format used by Find Case Law (FCL). Provides client-side XML
 * parsing capabilities for legal document processing.
 *
 * Key FCL Use Cases:
 * - Parse judgment XML files in web browsers
 * - Extract metadata for search result display
 * - Navigate document structure for reading interface
 * - Transform XML to JSON for frontend frameworks
 * - Client-side document validation and preview
 */

/**
 * Structured metadata extracted from legal documents
 */
class JudgmentMetadata {
    constructor({
        neutralCitation = '',
        court = '',
        date = new Date(),
        judges = [],
        caseName = '',
        uri = '',
        subjectMatter = []
    } = {}) {
        this.neutralCitation = neutralCitation;
        this.court = court;
        this.date = date;
        this.judges = judges;
        this.caseName = caseName;
        this.uri = uri;
        this.subjectMatter = subjectMatter;
    }
}

/**
 * Parser for LegalDocML documents used in FCL.
 *
 * LegalDocML Structure:
 * - akomaNtoso: Root element
 * - judgment: Main document container
 * - meta: Metadata section
 * - preface: Document header
 * - body: Main content
 * - conclusions: Final statements
 */
class LegalDocMLParser {
    constructor() {
        this.namespaces = {
            'akn': 'http://docs.oasis-open.org/legaldocml/ns/akn/3.0',
            'uk': 'https://caselaw.nationalarchives.gov.uk/akn'
        };
    }

    /**
     * Parse a complete judgment XML document.
     *
     * @param {string} xmlContent - The XML content to parse
     * @returns {Object} Parsed judgment data
     */
    parseJudgmentXML(xmlContent) {
        try {
            // Parse XML using DOMParser (browser) or xml2js (Node.js)
            const parser = new DOMParser();
            const xmlDoc = parser.parseFromString(xmlContent, 'text/xml');

            // Check for parsing errors
            const parserError = xmlDoc.querySelector('parsererror');
            if (parserError) {
                throw new Error(`XML parsing error: ${parserError.textContent}`);
            }

            // Extract metadata
            const metadata = this._extractMetadata(xmlDoc);

            // Extract document sections
            const sections = this._extractSections(xmlDoc);

            // Build structured representation
            const judgmentData = {
                metadata: metadata ? this._metadataToObject(metadata) : {},
                sections: sections,
                domStructure: this._buildDOMTree(xmlDoc),
                extractedText: this._extractPlainText(xmlDoc),
                citationsFound: this._extractCitations(xmlDoc),
                paragraphCount: xmlDoc.querySelectorAll('paragraph').length,
                pageCount: xmlDoc.querySelectorAll('neutralCitation').length
            };

            return judgmentData;

        } catch (error) {
            return {
                error: `Failed to parse XML: ${error.message}`,
                metadata: {},
                sections: []
            };
        }
    }

    /**
     * Extract structured metadata from judgment header
     * @param {Document} xmlDoc - Parsed XML document
     * @returns {JudgmentMetadata|null} Extracted metadata
     */
    _extractMetadata(xmlDoc) {
        try {
            // Find neutral citation - e.g., [2023] UKSC 1
            const citationElem = xmlDoc.querySelector('neutralCitation');
            const neutralCitation = citationElem ? citationElem.textContent.trim() : 'Unknown';

            // Extract court information
            const courtElem = xmlDoc.querySelector('court');
            const court = courtElem ? courtElem.textContent.trim() : 'Unknown Court';

            // Find judgment date
            const dateElem = xmlDoc.querySelector('FRBRdate');
            const dateStr = dateElem ? dateElem.getAttribute('date') : '2023-01-01';
            const judgmentDate = new Date(dateStr);

            // Extract judges
            const judges = [];
            const judgeElems = xmlDoc.querySelectorAll('judge');
            judgeElems.forEach(judgeElem => {
                const judgeName = judgeElem.textContent.trim();
                if (judgeName) {
                    judges.push(judgeName);
                }
            });

            // If no judges found in specific tags, look in header
            if (judges.length === 0) {
                const prefaceElem = xmlDoc.querySelector('preface');
                if (prefaceElem) {
                    // Look for common patterns like "Lord Smith", "Lady Jones"
                    const judgePattern = /(?:Lord|Lady|Mr Justice|Mrs Justice|Sir|Dame)\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*/g;
                    const headerText = prefaceElem.textContent;
                    const foundJudges = headerText.match(judgePattern) || [];
                    judges.push(...foundJudges.slice(0, 5)); // Limit to 5 judges
                }
            }

            // Extract case name
            const caseNameElem = xmlDoc.querySelector('docTitle');
            const caseName = caseNameElem ? caseNameElem.textContent.trim() : 'Unknown Case';

            // Extract URI
            const uriElem = xmlDoc.querySelector('FRBRuri');
            const uri = uriElem ? uriElem.getAttribute('value') : `/judgment/${neutralCitation.replace(/\s/g, '_')}`;

            // Extract subject matter/keywords
            const subjectMatter = [];
            const keywordElems = xmlDoc.querySelectorAll('keyword');
            keywordElems.forEach(keywordElem => {
                const keyword = keywordElem.textContent.trim();
                if (keyword) {
                    subjectMatter.push(keyword);
                }
            });

            return new JudgmentMetadata({
                neutralCitation,
                court,
                date: judgmentDate,
                judges,
                caseName,
                uri,
                subjectMatter
            });

        } catch (error) {
            console.error('Error extracting metadata:', error);
            return null;
        }
    }

    /**
     * Extract document sections with hierarchy
     * @param {Document} xmlDoc - Parsed XML document
     * @returns {Array} Array of section objects
     */
    _extractSections(xmlDoc) {
        const sections = [];

        // Extract preface (header information)
        const prefaceElem = xmlDoc.querySelector('preface');
        if (prefaceElem) {
            sections.push({
                type: 'preface',
                title: 'Document Header',
                content: prefaceElem.textContent.trim(),
                subsections: []
            });
        }

        // Extract body paragraphs
        const bodyElem = xmlDoc.querySelector('body');
        if (bodyElem) {
            const paragraphs = bodyElem.querySelectorAll('paragraph');
            const bodyContent = [];

            // Limit to first 20 paragraphs
            Array.from(paragraphs).slice(0, 20).forEach((para, index) => {
                const paraText = para.textContent.trim();
                if (paraText) {
                    bodyContent.push({
                        paragraphId: `para_${index + 1}`,
                        content: paraText.length > 500 ? paraText.substring(0, 500) + '...' : paraText,
                        wordCount: paraText.split(/\s+/).length
                    });
                }
            });

            sections.push({
                type: 'body',
                title: 'Judgment Body',
                content: `${paragraphs.length} paragraphs`,
                subsections: bodyContent
            });
        }

        // Extract conclusions
        const conclusionsElem = xmlDoc.querySelector('conclusions');
        if (conclusionsElem) {
            sections.push({
                type: 'conclusions',
                title: 'Conclusions',
                content: conclusionsElem.textContent.trim(),
                subsections: []
            });
        }

        return sections;
    }

    /**
     * Build a simplified DOM tree representation
     * @param {Document} xmlDoc - Parsed XML document
     * @returns {Object} DOM tree structure
     */
    _buildDOMTree(xmlDoc) {
        const root = xmlDoc.querySelector('akomaNtoso') || xmlDoc.querySelector('judgment');
        if (!root) {
            return {};
        }

        const buildTreeNode = (element) => {
            const node = {
                tag: element.tagName.toLowerCase(),
                attributes: {},
                text: element.textContent.trim().substring(0, 100),
                children: []
            };

            // Extract attributes
            if (element.attributes) {
                for (let i = 0; i < element.attributes.length; i++) {
                    const attr = element.attributes[i];
                    node.attributes[attr.name] = attr.value;
                }
            }

            // Add children (limit depth to avoid too large structures)
            if (element.children.length < 50) {
                Array.from(element.children).forEach(child => {
                    if (child.nodeType === Node.ELEMENT_NODE) {
                        node.children.push(buildTreeNode(child));
                    }
                });
            }

            return node;
        };

        return buildTreeNode(root);
    }

    /**
     * Extract plain text content for full-text search
     * @param {Document} xmlDoc - Parsed XML document
     * @returns {string} Plain text content
     */
    _extractPlainText(xmlDoc) {
        // Get text content and clean it up
        const text = xmlDoc.documentElement.textContent || '';

        // Clean up whitespace
        const cleanText = text
            .replace(/\s+/g, ' ')  // Replace multiple whitespace with single space
            .trim();

        return cleanText.substring(0, 5000); // Limit to first 5000 characters
    }

    /**
     * Extract legal citations found in the document
     * @param {Document} xmlDoc - Parsed XML document
     * @returns {Array} Array of found citations
     */
    _extractCitations(xmlDoc) {
        const text = xmlDoc.documentElement.textContent || '';

        // Common UK citation patterns
        const citationPatterns = [
            /\[(?:19|20)\d{2}\]\s+(?:UKSC|UKHL|EWCA|EWHC|UKUT|UKFTT)\s+\d+/gi,  // Modern neutral citations
            /\[(?:19|20)\d{2}\]\s+\d+\s+(?:WLR|All ER|AC|Ch|QB|Cr App R)/gi,     // Law reports
            /(?:19|20)\d{2}\s+(?:SLT|SC|SCLR)\s+\d+/gi                           // Scottish citations
        ];

        const citations = [];
        citationPatterns.forEach(pattern => {
            const found = text.match(pattern) || [];
            citations.push(...found.slice(0, 10)); // Limit to 10 per pattern
        });

        // Remove duplicates and return
        return [...new Set(citations)];
    }

    /**
     * Convert JudgmentMetadata to plain object
     * @param {JudgmentMetadata} metadata - Metadata instance
     * @returns {Object} Plain object representation
     */
    _metadataToObject(metadata) {
        return {
            neutralCitation: metadata.neutralCitation,
            court: metadata.court,
            date: metadata.date.toISOString(),
            judges: metadata.judges,
            caseName: metadata.caseName,
            uri: metadata.uri,
            subjectMatter: metadata.subjectMatter
        };
    }
}

/**
 * Utility functions for DOM manipulation and analysis
 */
class DOMUtils {
    /**
     * Query XML document with namespace support
     * @param {Document} doc - XML document
     * @param {string} selector - CSS selector or XPath
     * @param {Object} namespaces - Namespace mappings
     * @returns {NodeList} Matching nodes
     */
    static queryWithNamespace(doc, selector, namespaces = {}) {
        // Simple implementation - could be enhanced with full XPath support
        return doc.querySelectorAll(selector);
    }

    /**
     * Get element text with fallback options
     * @param {Element} element - DOM element
     * @param {string} fallback - Fallback text
     * @returns {string} Element text or fallback
     */
    static getTextContent(element, fallback = '') {
        return element ? element.textContent.trim() : fallback;
    }

    /**
     * Extract attributes as object
     * @param {Element} element - DOM element
     * @returns {Object} Attributes object
     */
    static getAttributes(element) {
        const attrs = {};
        if (element && element.attributes) {
            for (let i = 0; i < element.attributes.length; i++) {
                const attr = element.attributes[i];
                attrs[attr.name] = attr.value;
            }
        }
        return attrs;
    }

    /**
     * Find elements by pattern in text content
     * @param {Document} doc - XML document
     * @param {RegExp} pattern - Search pattern
     * @param {string} elementType - Element type to search within
     * @returns {Array} Matching elements with their text
     */
    static findByPattern(doc, pattern, elementType = '*') {
        const elements = doc.querySelectorAll(elementType);
        const matches = [];

        Array.from(elements).forEach(element => {
            const text = element.textContent;
            const found = text.match(pattern);
            if (found) {
                matches.push({
                    element: element,
                    text: text.trim(),
                    matches: found
                });
            }
        });

        return matches;
    }
}

/**
 * Demonstrate DOM tree operations with sample UK judgment data
 */
function demonstrateLegalDocMLParsing() {
    // Sample LegalDocML document (simplified)
    const sampleXML = `<?xml version="1.0" encoding="UTF-8"?>
    <akomaNtoso xmlns="http://docs.oasis-open.org/legaldocml/ns/akn/3.0">
      <judgment name="judgment">
        <meta>
          <identification source="#tna">
            <FRBRWork>
              <FRBRuri value="/akn/uk/uksc/2023/42"/>
              <FRBRdate date="2023-11-15"/>
              <FRBRauthor href="#uksc"/>
              <FRBRcountry value="uk"/>
            </FRBRWork>
          </identification>
        </meta>
        <preface>
          <neutralCitation>[2023] UKSC 42</neutralCitation>
          <court>SUPREME COURT</court>
          <docTitle>R (on the application of Miller) v The Prime Minister</docTitle>
          <judge>Lord Reed (President)</judge>
          <judge>Lady Hale (Deputy President)</judge>
          <judge>Lord Kerr</judge>
        </preface>
        <body>
          <paragraph id="para_1">
            <content>
              <p>This appeal concerns the constitutional principles surrounding the exercise of prerogative powers by the Executive and their relationship with Parliament.</p>
            </content>
          </paragraph>
          <paragraph id="para_2">
            <content>
              <p>The case of <cite>[2019] UKSC 41</cite> established important precedents regarding parliamentary sovereignty. We must also consider <cite>[2017] UKSC 5</cite> and the principles outlined therein.</p>
            </content>
          </paragraph>
          <paragraph id="para_3">
            <content>
              <p>The respondent argues that Article 50 of the Treaty on European Union creates obligations which cannot be fulfilled without parliamentary approval, citing <cite>[2016] EWHC 2768 (Admin)</cite>.</p>
            </content>
          </paragraph>
        </body>
        <conclusions>
          <paragraph>
            <content>
              <p>For the reasons given above, this appeal is allowed. The order of the Divisional Court is restored.</p>
            </content>
          </paragraph>
        </conclusions>
      </judgment>
    </akomaNtoso>`;

    console.log("=== LegalDocML DOM Tree Parsing Demo (JavaScript) ===\n");

    // Initialize parser
    const parser = new LegalDocMLParser();

    // Parse the sample document
    const result = parser.parseJudgmentXML(sampleXML);

    // Display results
    console.log("1. EXTRACTED METADATA:");
    const metadata = result.metadata;
    Object.entries(metadata).forEach(([key, value]) => {
        if (Array.isArray(value)) {
            console.log(`   ${key}: ${value.join(', ')}`);
        } else {
            console.log(`   ${key}: ${value}`);
        }
    });

    console.log(`\n2. DOCUMENT STRUCTURE:`);
    console.log(`   Sections found: ${result.sections.length}`);
    result.sections.forEach(section => {
        console.log(`   - ${section.type}: ${section.title}`);
        if (section.subsections.length > 0) {
            console.log(`     Subsections: ${section.subsections.length}`);
        }
    });

    console.log(`\n3. DOM TREE ANALYSIS:`);
    const domTree = result.domStructure;
    console.log(`   Root element: ${domTree.tag || 'unknown'}`);
    console.log(`   Child elements: ${domTree.children ? domTree.children.length : 0}`);
    console.log(`   Attributes: ${JSON.stringify(domTree.attributes || {})}`);

    console.log(`\n4. CITATIONS EXTRACTED:`);
    const citations = result.citationsFound;
    citations.forEach(citation => {
        console.log(`   - ${citation}`);
    });

    console.log(`\n5. CONTENT STATISTICS:`);
    console.log(`   Paragraph count: ${result.paragraphCount}`);
    console.log(`   Text length: ${result.extractedText.length} characters`);
    console.log(`   Citations found: ${citations.length}`);

    console.log(`\n6. SAMPLE EXTRACTED TEXT:`);
    console.log(`   ${result.extractedText.substring(0, 300)}...`);

    // Demonstrate DOM navigation using DOMParser
    console.log(`\n7. DOM NAVIGATION EXAMPLE:`);
    const parser2 = new DOMParser();
    const xmlDoc = parser2.parseFromString(sampleXML, 'text/xml');

    // Find all paragraphs
    const paragraphs = xmlDoc.querySelectorAll('paragraph');
    console.log(`   Found ${paragraphs.length} paragraphs`);

    // Find citations within paragraphs
    Array.from(paragraphs).forEach((para, index) => {
        const cites = para.querySelectorAll('cite');
        if (cites.length > 0) {
            console.log(`   Paragraph ${index + 1} contains ${cites.length} citations:`);
            Array.from(cites).forEach(cite => {
                console.log(`     - ${cite.textContent}`);
            });
        }
    });

    // Demonstrate specific data extraction
    console.log(`\n8. SPECIFIC DATA EXTRACTION:`);

    // Extract judge names
    const judges = xmlDoc.querySelectorAll('judge');
    console.log(`   Judges presiding:`);
    Array.from(judges).forEach(judge => {
        console.log(`     - ${judge.textContent}`);
    });

    // Extract court and citation
    const court = xmlDoc.querySelector('court');
    const citation = xmlDoc.querySelector('neutralCitation');
    console.log(`   Court: ${court ? court.textContent : 'Not found'}`);
    console.log(`   Citation: ${citation ? citation.textContent : 'Not found'}`);

    // Demonstrate utility functions
    console.log(`\n9. UTILITY FUNCTIONS DEMO:`);

    // Find elements by pattern
    const citationPattern = /\[\d{4}\]\s+\w+\s+\d+/g;
    const foundCitations = DOMUtils.findByPattern(xmlDoc, citationPattern, 'cite');
    console.log(`   Citations found by pattern: ${foundCitations.length}`);
    foundCitations.forEach(match => {
        console.log(`     - "${match.matches[0]}" in element: ${match.element.tagName}`);
    });

    return result;
}

// Export for use in Node.js or browser environments
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        LegalDocMLParser,
        JudgmentMetadata,
        DOMUtils,
        demonstrateLegalDocMLParsing
    };
}

// Auto-run demo if script is executed directly
if (typeof window !== 'undefined') {
    // Browser environment
    document.addEventListener('DOMContentLoaded', () => {
        console.log('LegalDocML DOM Parser loaded in browser');
        // Uncomment to run demo: demonstrateLegalDocMLParsing();
    });
} else if (typeof require !== 'undefined' && require.main === module) {
    // Node.js environment - run demo
    demonstrateLegalDocMLParsing();
}