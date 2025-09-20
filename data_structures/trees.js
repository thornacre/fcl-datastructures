class JudgmentNode {
    /**
     * Represents a node in a legal document hierarchy.
     * Example: UK Supreme Court judgment structure
     */
    constructor(name, type = 'section', content = '') {
        this.name = name;
        this.type = type; // 'judgment', 'section', 'paragraph', 'citation'
        this.content = content;
        this.children = [];
        this.metadata = {};
    }

    addChild(child) {
        this.children.push(child);
        return child;
    }

    *traverseDFS() {
        // Depth-first traversal for document processing
        yield this;
        for (const child of this.children) {
            yield* child.traverseDFS();
        }
    }

    findAllByType(nodeType) {
        // Find all nodes of a specific type (e.g., all citations)
        const results = [];
        for (const node of this.traverseDFS()) {
            if (node.type === nodeType) {
                results.push(node);
            }
        }
        return results;
    }
}

// Example: UK Supreme Court Judgment [2023] UKSC 42
const judgment = new JudgmentNode("R v Smith [2023] UKSC 42", 'judgment');
judgment.metadata = {
    court: 'UK Supreme Court',
    date: '2023-11-15',
    neutral_citation: '[2023] UKSC 42'
};

// Add judgment sections
const intro = judgment.addChild(new JudgmentNode("Introduction", 'section'));
const facts = judgment.addChild(new JudgmentNode("Facts", 'section'));
const law = judgment.addChild(new JudgmentNode("Legal Framework", 'section'));
const analysis = judgment.addChild(new JudgmentNode("Analysis", 'section'));

// Add paragraphs to introduction
intro.addChild(new JudgmentNode(
    "Paragraph 1",
    'paragraph',
    "This appeal concerns the interpretation of Section 1 of the Criminal Justice Act 2003."
));

// Add citations within legal framework
law.addChild(new JudgmentNode(
    "Citation: Donoghue v Stevenson",
    'citation',
    "[1932] AC 562"
));
law.addChild(new JudgmentNode(
    "Citation: R v Jogee",
    'citation',
    "[2016] UKSC 8"
));

// Example usage
console.log(`Judgment: ${judgment.name}`);
console.log(`Total sections: ${judgment.children.length}`);
console.log(`All citations: ${judgment.findAllByType('citation').map(n => n.content)}`);