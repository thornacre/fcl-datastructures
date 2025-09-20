class JudgmentMetadataIndex:
    """
    Hash table for O(1) access to judgment metadata.
    Used for quick lookups by neutral citation.
    """
    def __init__(self):
        self.index = {}  # Hash table for metadata
        self.court_index = {}  # Secondary index by court

    def add_judgment(self, neutral_citation, metadata):
        """Add judgment with O(1) insertion"""
        self.index[neutral_citation] = metadata

        # Update court index
        court = metadata.get('court')
        if court:
            if court not in self.court_index:
                self.court_index[court] = []
            self.court_index[court].append(neutral_citation)

    def get_judgment(self, neutral_citation):
        """O(1) lookup by neutral citation"""
        return self.index.get(neutral_citation)

    def get_by_court(self, court):
        """Get all judgments from a specific court"""
        citations = self.court_index.get(court, [])
        return [self.index[citation] for citation in citations]

    def search_by_year(self, year):
        """Linear search - demonstrates when hash tables aren't optimal"""
        results = []
        for citation, metadata in self.index.items():
            if metadata.get('date', '').startswith(str(year)):
                results.append((citation, metadata))
        return results


# Example: FCL judgment metadata index
if __name__ == "__main__":
    index = JudgmentMetadataIndex()

    # Add UK judgments with metadata
    judgments = [
        ('[2023] UKSC 42', {
            'name': 'R v Smith',
            'court': 'UK Supreme Court',
            'date': '2023-11-15',
            'keywords': ['criminal', 'appeal', 'sentencing'],
            'judge': 'Lord Reed',
            'file_url': '/judgments/uksc/2023/42.xml'
        }),
        ('[2023] EWCA Crim 789', {
            'name': 'R v Johnson',
            'court': 'Court of Appeal Criminal Division',
            'date': '2023-06-20',
            'keywords': ['fraud', 'conspiracy'],
            'judge': 'Lord Justice Holroyde',
            'file_url': '/judgments/ewca/crim/2023/789.xml'
        }),
        ('[2023] EWHC 1234 (Ch)', {
            'name': 'Re XYZ Company Ltd',
            'court': 'High Court Chancery Division',
            'date': '2023-04-10',
            'keywords': ['insolvency', 'directors duties'],
            'judge': 'Mr Justice Zacaroli',
            'file_url': '/judgments/ewhc/ch/2023/1234.xml'
        })
    ]

    # Populate the index
    for citation, metadata in judgments:
        index.add_judgment(citation, metadata)

    # O(1) lookup example
    print("Direct lookup [2023] UKSC 42:")
    judgment = index.get_judgment('[2023] UKSC 42')
    if judgment:
        print(f"  Name: {judgment['name']}")
        print(f"  Court: {judgment['court']}")
        print(f"  Keywords: {', '.join(judgment['keywords'])}")

    # Get all Supreme Court cases
    print("\nAll UK Supreme Court cases:")
    for judgment in index.get_by_court('UK Supreme Court'):
        print(f"  - {judgment['name']} ({judgment['date']})")

    # Search by year (demonstrates linear search)
    print("\nAll 2023 judgments:")
    for citation, metadata in index.search_by_year(2023):
        print(f"  {citation}: {metadata['name']}")