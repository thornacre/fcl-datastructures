class CitationGraph:
    """
    Graph structure for tracking case law citations.
    Each node is a judgment, edges represent citations.
    """
    def __init__(self):
        self.adjacency_list = {}  # {case_id: [cited_case_ids]}
        self.metadata = {}  # {case_id: {court, date, name}}

    def add_judgment(self, case_id, name, court, date):
        if case_id not in self.adjacency_list:
            self.adjacency_list[case_id] = []
        self.metadata[case_id] = {
            'name': name,
            'court': court,
            'date': date
        }

    def add_citation(self, citing_case, cited_case):
        if citing_case in self.adjacency_list:
            self.adjacency_list[citing_case].append(cited_case)

    def find_most_cited(self, top_n=5):
        """Find the most influential cases by citation count"""
        citation_counts = {}
        for citations in self.adjacency_list.values():
            for cited in citations:
                citation_counts[cited] = citation_counts.get(cited, 0) + 1

        return sorted(citation_counts.items(), key=lambda x: x[1], reverse=True)[:top_n]

    def find_citation_chain(self, start_case, target_case, visited=None):
        """Find citation path between two cases"""
        if visited is None:
            visited = set()

        if start_case == target_case:
            return [start_case]

        visited.add(start_case)

        for cited in self.adjacency_list.get(start_case, []):
            if cited not in visited:
                path = self.find_citation_chain(cited, target_case, visited)
                if path:
                    return [start_case] + path

        return None


# Example: UK case law citation network
if __name__ == "__main__":
    graph = CitationGraph()

    # Add landmark UK cases
    graph.add_judgment('[2023] UKSC 42', 'R v Smith', 'UK Supreme Court', '2023-11-15')
    graph.add_judgment('[2016] UKSC 8', 'R v Jogee', 'UK Supreme Court', '2016-02-18')
    graph.add_judgment('[1932] AC 562', 'Donoghue v Stevenson', 'House of Lords', '1932-05-26')
    graph.add_judgment('[2020] EWCA Civ 123', 'Jones v Brown', 'Court of Appeal', '2020-03-10')
    graph.add_judgment('[2019] EWHC 456', 'Re ABC Ltd', 'High Court', '2019-07-22')

    # Add citations (newer cases cite older ones)
    graph.add_citation('[2023] UKSC 42', '[2016] UKSC 8')
    graph.add_citation('[2023] UKSC 42', '[1932] AC 562')
    graph.add_citation('[2020] EWCA Civ 123', '[1932] AC 562')
    graph.add_citation('[2019] EWHC 456', '[1932] AC 562')
    graph.add_citation('[2016] UKSC 8', '[1932] AC 562')

    # Find most influential cases
    print("Most cited cases:")
    for case_id, count in graph.find_most_cited(3):
        metadata = graph.metadata.get(case_id, {})
        print(f"  {case_id}: {metadata.get('name', 'Unknown')} - {count} citations")

    # Find citation path
    path = graph.find_citation_chain('[2023] UKSC 42', '[1932] AC 562')
    if path:
        print(f"\nCitation path from R v Smith to Donoghue v Stevenson:")
        print(" -> ".join(path))