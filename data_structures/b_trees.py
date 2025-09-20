"""
B-Tree Data Structure for FCL Legal Judgment Date Range Queries

This implementation provides efficient date-based querying for legal judgments,
optimized for the frequent date range searches in legal databases.

Use cases:
- Find all judgments within a specific date range
- Chronological browsing of legal decisions
- Statistical analysis of case frequency over time
- Efficient temporal filtering for large legal databases
"""

from datetime import datetime, date
from typing import List, Optional, Tuple, Any
import bisect

class LegalJudgment:
    """Represents a legal judgment with essential metadata."""

    def __init__(self, case_name: str, citation: str, judgment_date: date,
                 court: str, neutral_citation: str = "", judges: List[str] = None):
        self.case_name = case_name
        self.citation = citation
        self.judgment_date = judgment_date
        self.court = court
        self.neutral_citation = neutral_citation
        self.judges = judges or []

    def __str__(self):
        return f"{self.case_name} {self.citation} ({self.judgment_date})"

    def __repr__(self):
        return self.__str__()

class BTreeNode:
    """Node in the B-tree structure optimized for date-based legal data."""

    def __init__(self, is_leaf: bool = False):
        self.keys = []          # List of dates (sorted)
        self.values = []        # List of judgment lists for each date
        self.children = []      # List of child nodes
        self.is_leaf = is_leaf

    def is_full(self, max_degree: int) -> bool:
        """Check if node has maximum number of keys."""
        return len(self.keys) >= max_degree - 1

class LegalJudgmentBTree:
    """
    B-tree optimized for legal judgment date range queries.
    Supports efficient insertion, search, and range queries by date.
    """

    def __init__(self, max_degree: int = 5):
        """
        Initialize B-tree for legal judgments.

        Args:
            max_degree (int): Maximum number of children per node (degree)
        """
        self.max_degree = max_degree
        self.min_degree = max_degree // 2
        self.root = BTreeNode(is_leaf=True)
        self.total_judgments = 0

    def insert(self, judgment: LegalJudgment):
        """
        Insert a legal judgment into the B-tree, indexed by date.

        Args:
            judgment (LegalJudgment): The judgment to insert
        """
        if self.root.is_full(self.max_degree):
            # Split root if full
            new_root = BTreeNode(is_leaf=False)
            new_root.children.append(self.root)
            self._split_child(new_root, 0)
            self.root = new_root

        self._insert_non_full(self.root, judgment)
        self.total_judgments += 1

    def _insert_non_full(self, node: BTreeNode, judgment: LegalJudgment):
        """Insert judgment into a node that is not full."""
        i = len(node.keys) - 1
        judgment_date = judgment.judgment_date

        if node.is_leaf:
            # Find insertion position
            pos = bisect.bisect_left([key for key in node.keys], judgment_date)

            if pos < len(node.keys) and node.keys[pos] == judgment_date:
                # Date already exists, add to existing list
                node.values[pos].append(judgment)
            else:
                # Insert new date
                node.keys.insert(pos, judgment_date)
                node.values.insert(pos, [judgment])
        else:
            # Find child to insert into
            while i >= 0 and judgment_date < node.keys[i]:
                i -= 1
            i += 1

            if node.children[i].is_full(self.max_degree):
                self._split_child(node, i)
                if judgment_date > node.keys[i]:
                    i += 1

            self._insert_non_full(node.children[i], judgment)

    def _split_child(self, parent: BTreeNode, index: int):
        """Split a full child node."""
        full_child = parent.children[index]
        new_child = BTreeNode(is_leaf=full_child.is_leaf)

        mid_index = self.min_degree - 1

        # Move half the keys and values to new node
        new_child.keys = full_child.keys[mid_index + 1:]
        new_child.values = full_child.values[mid_index + 1:]
        full_child.keys = full_child.keys[:mid_index]
        full_child.values = full_child.values[:mid_index]

        if not full_child.is_leaf:
            new_child.children = full_child.children[mid_index + 1:]
            full_child.children = full_child.children[:mid_index + 1]

        # Move middle key up to parent
        parent.children.insert(index + 1, new_child)
        parent.keys.insert(index, full_child.keys[mid_index])
        parent.values.insert(index, full_child.values[mid_index])

    def search_by_date(self, target_date: date) -> List[LegalJudgment]:
        """
        Find all judgments on a specific date.

        Args:
            target_date (date): The date to search for

        Returns:
            List[LegalJudgment]: All judgments on that date
        """
        return self._search_node(self.root, target_date)

    def _search_node(self, node: BTreeNode, target_date: date) -> List[LegalJudgment]:
        """Search for date in a specific node."""
        i = 0
        while i < len(node.keys) and target_date > node.keys[i]:
            i += 1

        if i < len(node.keys) and target_date == node.keys[i]:
            return node.values[i][:]  # Return copy of judgment list

        if node.is_leaf:
            return []

        return self._search_node(node.children[i], target_date)

    def range_query(self, start_date: date, end_date: date) -> List[LegalJudgment]:
        """
        Find all judgments within a date range (inclusive).

        Args:
            start_date (date): Start of date range
            end_date (date): End of date range

        Returns:
            List[LegalJudgment]: All judgments in the date range, sorted by date
        """
        if start_date > end_date:
            return []

        results = []
        self._range_query_node(self.root, start_date, end_date, results)

        # Sort results by date, then by case name
        results.sort(key=lambda j: (j.judgment_date, j.case_name))
        return results

    def _range_query_node(self, node: BTreeNode, start_date: date,
                         end_date: date, results: List[LegalJudgment]):
        """Recursively collect judgments in date range."""
        i = 0

        # Process each key in the node
        while i < len(node.keys):
            # If not leaf, check left child first
            if not node.is_leaf:
                self._range_query_node(node.children[i], start_date, end_date, results)

            # Check if current key is in range
            if start_date <= node.keys[i] <= end_date:
                results.extend(node.values[i])
            elif node.keys[i] > end_date:
                # No need to check further keys or right children
                return

            i += 1

        # Check rightmost child if not leaf
        if not node.is_leaf:
            self._range_query_node(node.children[i], start_date, end_date, results)

    def get_chronological_cases(self, limit: int = None) -> List[LegalJudgment]:
        """
        Get all cases in chronological order.

        Args:
            limit (int): Maximum number of cases to return

        Returns:
            List[LegalJudgment]: Cases sorted by date
        """
        results = []
        self._collect_all_cases(self.root, results)

        # Sort by date
        results.sort(key=lambda j: j.judgment_date)

        if limit:
            return results[:limit]
        return results

    def _collect_all_cases(self, node: BTreeNode, results: List[LegalJudgment]):
        """Recursively collect all cases from the tree."""
        if node.is_leaf:
            for value_list in node.values:
                results.extend(value_list)
        else:
            for i in range(len(node.keys)):
                self._collect_all_cases(node.children[i], results)
                results.extend(node.values[i])
            # Don't forget the rightmost child
            self._collect_all_cases(node.children[-1], results)

    def get_statistics(self) -> dict:
        """Get statistics about the legal judgment database."""
        all_cases = self.get_chronological_cases()

        if not all_cases:
            return {"total_cases": 0}

        # Calculate date range
        earliest = min(case.judgment_date for case in all_cases)
        latest = max(case.judgment_date for case in all_cases)

        # Count by court
        court_counts = {}
        for case in all_cases:
            court_counts[case.court] = court_counts.get(case.court, 0) + 1

        # Count by year
        year_counts = {}
        for case in all_cases:
            year = case.judgment_date.year
            year_counts[year] = year_counts.get(year, 0) + 1

        return {
            "total_cases": len(all_cases),
            "date_range": f"{earliest} to {latest}",
            "earliest_case": earliest,
            "latest_case": latest,
            "court_distribution": court_counts,
            "yearly_distribution": year_counts
        }


def demo_fcl_legal_btree():
    """Demonstration of B-tree with realistic FCL legal judgment data."""

    # Initialize B-tree for legal judgments
    legal_btree = LegalJudgmentBTree(max_degree=5)

    # Sample UK legal judgments with realistic data
    sample_judgments = [
        LegalJudgment(
            "Donoghue v Stevenson",
            "[1932] AC 562",
            date(1932, 5, 26),
            "House of Lords",
            "[1932] UKHL 100",
            ["Lord Atkin", "Lord Thankerton", "Lord Macmillan"]
        ),
        LegalJudgment(
            "Carlill v Carbolic Smoke Ball Company",
            "[1893] 1 QB 256",
            date(1892, 12, 7),
            "Court of Appeal",
            "[1892] EWCA Civ 1",
            ["Lindley LJ", "Bowen LJ", "A.L. Smith LJ"]
        ),
        LegalJudgment(
            "Rylands v Fletcher",
            "(1868) LR 3 HL 330",
            date(1868, 7, 17),
            "House of Lords",
            "[1868] UKHL 1",
            ["Lord Cairns", "Lord Cranworth"]
        ),
        LegalJudgment(
            "R v Brown",
            "[1994] 1 AC 212",
            date(1993, 3, 11),
            "House of Lords",
            "[1993] UKHL 19",
            ["Lord Templeman", "Lord Jauncey", "Lord Lowry"]
        ),
        LegalJudgment(
            "Pepper v Hart",
            "[1993] AC 593",
            date(1992, 11, 26),
            "House of Lords",
            "[1992] UKHL 3",
            ["Lord Griffiths", "Lord Ackner", "Lord Oliver"]
        ),
        LegalJudgment(
            "Caparo Industries plc v Dickman",
            "[1990] 2 AC 605",
            date(1990, 2, 8),
            "House of Lords",
            "[1990] UKHL 2",
            ["Lord Bridge", "Lord Roskill", "Lord Ackner"]
        ),
        LegalJudgment(
            "R v R (Rape: Marital Exemption)",
            "[1992] 1 AC 599",
            date(1991, 10, 23),
            "House of Lords",
            "[1991] UKHL 12",
            ["Lord Keith", "Lord Jauncey", "Lord Ackner"]
        ),
        LegalJudgment(
            "Miller v Jackson",
            "[1977] QB 966",
            date(1977, 5, 3),
            "Court of Appeal",
            "[1977] EWCA Civ 6",
            ["Lord Denning MR", "Geoffrey Lane LJ", "Cumming-Bruce LJ"]
        ),
        LegalJudgment(
            "Central London Property Trust Ltd v High Trees House Ltd",
            "[1947] KB 130",
            date(1946, 11, 2),
            "King's Bench Division",
            "[1946] EWHC KB 1",
            ["Denning J"]
        ),
        LegalJudgment(
            "Hedley Byrne & Co Ltd v Heller & Partners Ltd",
            "[1964] AC 465",
            date(1963, 5, 28),
            "House of Lords",
            "[1963] UKHL 4",
            ["Lord Reid", "Lord Morris", "Lord Hodson"]
        ),
        # Add some contemporary cases
        LegalJudgment(
            "R (Miller) v The Prime Minister",
            "[2019] UKSC 41",
            date(2019, 9, 24),
            "Supreme Court",
            "[2019] UKSC 41",
            ["Lady Hale", "Lord Reed", "Lord Kerr"]
        ),
        LegalJudgment(
            "Test Claimants in the FII Group Litigation v Revenue and Customs",
            "[2020] UKSC 47",
            date(2020, 11, 4),
            "Supreme Court",
            "[2020] UKSC 47",
            ["Lord Reed", "Lord Hodge", "Lord Lloyd-Jones"]
        )
    ]

    print("=== FCL Legal Judgment B-Tree Demo ===\n")

    # Insert all judgments
    print("Inserting legal judgments into B-tree...")
    for judgment in sample_judgments:
        legal_btree.insert(judgment)

    print(f"Successfully inserted {legal_btree.total_judgments} judgments\n")

    # Display database statistics
    print("=== Database Statistics ===")
    stats = legal_btree.get_statistics()
    print(f"Total cases: {stats['total_cases']}")
    print(f"Date range: {stats['date_range']}")
    print(f"Earliest case: {stats['earliest_case']}")
    print(f"Latest case: {stats['latest_case']}")

    print("\nCourt distribution:")
    for court, count in sorted(stats['court_distribution'].items()):
        print(f"  {court}: {count} cases")

    print(f"\nYearly distribution (top 5):")
    yearly_sorted = sorted(stats['yearly_distribution'].items(),
                          key=lambda x: x[1], reverse=True)
    for year, count in yearly_sorted[:5]:
        print(f"  {year}: {count} cases")

    # Demonstrate specific date search
    print("\n=== Specific Date Search ===")
    search_date = date(1932, 5, 26)
    results = legal_btree.search_by_date(search_date)
    print(f"Cases on {search_date}:")
    for judgment in results:
        print(f"  - {judgment.case_name} ({judgment.citation})")

    # Demonstrate date range queries
    print("\n=== Date Range Queries ===")

    # 20th century landmark cases
    print(f"\nLandmark cases from 1900-1999:")
    century_cases = legal_btree.range_query(date(1900, 1, 1), date(1999, 12, 31))
    for judgment in century_cases[:5]:  # Show first 5
        print(f"  - {judgment.judgment_date}: {judgment.case_name}")
        print(f"    {judgment.citation} ({judgment.court})")

    # Recent Supreme Court cases
    print(f"\nRecent Supreme Court cases (2010 onwards):")
    recent_cases = legal_btree.range_query(date(2010, 1, 1), date(2025, 12, 31))
    for judgment in recent_cases:
        print(f"  - {judgment.judgment_date}: {judgment.case_name}")
        print(f"    {judgment.citation} ({judgment.court})")

    # House of Lords cases in specific decade
    print(f"\nHouse of Lords cases from 1990-1999:")
    hl_90s = legal_btree.range_query(date(1990, 1, 1), date(1999, 12, 31))
    hl_cases = [j for j in hl_90s if j.court == "House of Lords"]
    for judgment in hl_cases:
        print(f"  - {judgment.judgment_date}: {judgment.case_name}")
        print(f"    Judges: {', '.join(judgment.judges[:2])}{'...' if len(judgment.judges) > 2 else ''}")

    # Chronological listing
    print(f"\n=== Chronological Case Listing (First 8) ===")
    chronological = legal_btree.get_chronological_cases(limit=8)
    for i, judgment in enumerate(chronological, 1):
        print(f"{i:2d}. {judgment.judgment_date}: {judgment.case_name}")
        print(f"     {judgment.citation} - {judgment.court}")
        if judgment.neutral_citation:
            print(f"     Neutral citation: {judgment.neutral_citation}")


if __name__ == "__main__":
    demo_fcl_legal_btree()