class JudgmentNode:
    """
    Represents a node in a legal document hierarchy.
    Example: UK Supreme Court judgment structure
    """
    def __init__(self, name, node_type='section', content=''):
        self.name = name
        self.type = node_type  # 'judgment', 'section', 'paragraph', 'citation'
        self.content = content
        self.children = []
        self.metadata = {}

    def add_child(self, child):
        self.children.append(child)
        return child

    def traverse_dfs(self):
        """Depth-first traversal for document processing"""
        yield self
        for child in self.children:
            yield from child.traverse_dfs()

    def find_all_by_type(self, node_type):
        """Find all nodes of a specific type (e.g., all citations)"""
        return [node for node in self.traverse_dfs() if node.type == node_type]


# Example: UK Supreme Court Judgment [2023] UKSC 42
if __name__ == "__main__":
    judgment = JudgmentNode("R v Smith [2023] UKSC 42", node_type='judgment')
    judgment.metadata = {
        'court': 'UK Supreme Court',
        'date': '2023-11-15',
        'neutral_citation': '[2023] UKSC 42'
    }

    # Add judgment sections
    intro = judgment.add_child(JudgmentNode("Introduction", node_type='section'))
    facts = judgment.add_child(JudgmentNode("Facts", node_type='section'))
    law = judgment.add_child(JudgmentNode("Legal Framework", node_type='section'))
    analysis = judgment.add_child(JudgmentNode("Analysis", node_type='section'))

    # Add paragraphs to introduction
    intro.add_child(JudgmentNode(
        "Paragraph 1",
        node_type='paragraph',
        content="This appeal concerns the interpretation of Section 1 of the Criminal Justice Act 2003."
    ))

    # Add citations within legal framework
    law.add_child(JudgmentNode(
        "Citation: Donoghue v Stevenson",
        node_type='citation',
        content="[1932] AC 562"
    ))
    law.add_child(JudgmentNode(
        "Citation: R v Jogee",
        node_type='citation',
        content="[2016] UKSC 8"
    ))

    # Example usage
    print(f"Judgment: {judgment.name}")
    print(f"Total sections: {len(judgment.children)}")
    print(f"All citations: {[node.content for node in judgment.find_all_by_type('citation')]}")