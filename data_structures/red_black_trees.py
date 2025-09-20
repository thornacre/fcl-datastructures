"""
Red-Black Trees for Sorted Judgment Collections
===============================================

This module implements self-balancing binary search trees (Red-Black Trees)
for maintaining sorted collections of judgments in Find Case Law (FCL).
Provides efficient insertion, deletion, and range query operations.

Key FCL Use Cases:
- Maintaining chronologically sorted judgment collections
- Efficient range queries by date, citation, or relevance score
- Balanced tree structure for consistent O(log n) performance
- Supporting sorted iteration through large judgment datasets
- Timeline views and date-based filtering operations
"""

from typing import Optional, List, Tuple, Any, Callable, Iterator
from dataclasses import dataclass
from datetime import datetime, date
from enum import Enum
import json


class Color(Enum):
    """Node colors for Red-Black Tree"""
    RED = "RED"
    BLACK = "BLACK"


@dataclass
class Judgment:
    """Represents a UK legal judgment with sorting criteria"""
    neutral_citation: str
    case_name: str
    court: str
    judgment_date: date
    uri: str
    relevance_score: float = 0.0

    def __str__(self) -> str:
        return f"{self.neutral_citation}: {self.case_name}"


class RBNode:
    """Red-Black Tree node containing judgment data"""

    def __init__(self, key: Any, value: Judgment, color: Color = Color.RED):
        self.key = key
        self.value = value
        self.color = color
        self.left: Optional['RBNode'] = None
        self.right: Optional['RBNode'] = None
        self.parent: Optional['RBNode'] = None

    def __str__(self) -> str:
        color_str = "R" if self.color == Color.RED else "B"
        return f"[{color_str}] {self.key}: {self.value.case_name}"


class RedBlackTree:
    """
    Self-balancing binary search tree for sorted judgment collections.
    Maintains Red-Black Tree properties for guaranteed O(log n) operations.
    """

    def __init__(self, key_function: Optional[Callable[[Judgment], Any]] = None):
        """
        Initialize Red-Black Tree with optional custom key function.

        Args:
            key_function: Function to extract sort key from Judgment objects.
                         Defaults to sorting by judgment_date.
        """
        self.nil = RBNode(None, None, Color.BLACK)  # Sentinel node
        self.root = self.nil
        self.size = 0

        # Default to sorting by judgment date
        self.key_function = key_function or (lambda j: j.judgment_date)

    def insert(self, judgment: Judgment) -> None:
        """Insert a judgment into the tree"""
        key = self.key_function(judgment)
        new_node = RBNode(key, judgment, Color.RED)
        new_node.left = self.nil
        new_node.right = self.nil

        # Standard BST insertion
        parent = None
        current = self.root

        while current != self.nil:
            parent = current
            if key < current.key:
                current = current.left
            elif key > current.key:
                current = current.right
            else:
                # Key already exists, update the value
                current.value = judgment
                return

        new_node.parent = parent

        if parent is None:
            self.root = new_node
        elif key < parent.key:
            parent.left = new_node
        else:
            parent.right = new_node

        self.size += 1

        # Fix Red-Black Tree properties
        self._insert_fixup(new_node)

    def delete(self, key: Any) -> bool:
        """Delete a judgment by key"""
        node = self._search_node(key)
        if node == self.nil:
            return False

        self._delete_node(node)
        self.size -= 1
        return True

    def search(self, key: Any) -> Optional[Judgment]:
        """Search for a judgment by key"""
        node = self._search_node(key)
        return node.value if node != self.nil else None

    def range_query(self, start_key: Any, end_key: Any) -> List[Judgment]:
        """
        Find all judgments with keys in the range [start_key, end_key].
        Useful for date range queries or score range filtering.
        """
        result = []
        self._range_query_helper(self.root, start_key, end_key, result)
        return result

    def get_by_date_range(self, start_date: date, end_date: date) -> List[Judgment]:
        """Get judgments within a specific date range"""
        if self.key_function.__name__ == '<lambda>' or 'judgment_date' in str(self.key_function):
            return self.range_query(start_date, end_date)
        else:
            # If not sorting by date, need to scan all nodes
            result = []
            for judgment in self.inorder_traversal():
                if start_date <= judgment.judgment_date <= end_date:
                    result.append(judgment)
            return result

    def get_by_court(self, court: str) -> List[Judgment]:
        """Get all judgments from a specific court"""
        result = []
        for judgment in self.inorder_traversal():
            if judgment.court == court:
                result.append(judgment)
        return result

    def inorder_traversal(self) -> Iterator[Judgment]:
        """Iterate through judgments in sorted order"""
        yield from self._inorder_helper(self.root)

    def get_minimum(self) -> Optional[Judgment]:
        """Get judgment with minimum key"""
        node = self._minimum(self.root)
        return node.value if node != self.nil else None

    def get_maximum(self) -> Optional[Judgment]:
        """Get judgment with maximum key"""
        node = self._maximum(self.root)
        return node.value if node != self.nil else None

    def get_sorted_list(self, reverse: bool = False) -> List[Judgment]:
        """Get all judgments as a sorted list"""
        judgments = list(self.inorder_traversal())
        return judgments[::-1] if reverse else judgments

    def get_statistics(self) -> dict:
        """Get tree statistics"""
        height = self._height(self.root)
        return {
            'size': self.size,
            'height': height,
            'is_balanced': height <= 2 * self._log2(self.size + 1) if self.size > 0 else True,
            'black_height': self._black_height(self.root)
        }

    def _insert_fixup(self, node: RBNode) -> None:
        """Fix Red-Black Tree properties after insertion"""
        while node.parent and node.parent.color == Color.RED:
            if node.parent == node.parent.parent.left:
                uncle = node.parent.parent.right

                if uncle.color == Color.RED:
                    # Case 1: Uncle is red
                    node.parent.color = Color.BLACK
                    uncle.color = Color.BLACK
                    node.parent.parent.color = Color.RED
                    node = node.parent.parent
                else:
                    if node == node.parent.right:
                        # Case 2: Node is right child
                        node = node.parent
                        self._left_rotate(node)

                    # Case 3: Node is left child
                    node.parent.color = Color.BLACK
                    node.parent.parent.color = Color.RED
                    self._right_rotate(node.parent.parent)
            else:
                # Symmetric cases (parent is right child)
                uncle = node.parent.parent.left

                if uncle.color == Color.RED:
                    node.parent.color = Color.BLACK
                    uncle.color = Color.BLACK
                    node.parent.parent.color = Color.RED
                    node = node.parent.parent
                else:
                    if node == node.parent.left:
                        node = node.parent
                        self._right_rotate(node)

                    node.parent.color = Color.BLACK
                    node.parent.parent.color = Color.RED
                    self._left_rotate(node.parent.parent)

        self.root.color = Color.BLACK

    def _delete_node(self, node: RBNode) -> None:
        """Delete a node and maintain Red-Black properties"""
        y = node
        y_original_color = y.color

        if node.left == self.nil:
            x = node.right
            self._transplant(node, node.right)
        elif node.right == self.nil:
            x = node.left
            self._transplant(node, node.left)
        else:
            y = self._minimum(node.right)
            y_original_color = y.color
            x = y.right

            if y.parent == node:
                x.parent = y
            else:
                self._transplant(y, y.right)
                y.right = node.right
                y.right.parent = y

            self._transplant(node, y)
            y.left = node.left
            y.left.parent = y
            y.color = node.color

        if y_original_color == Color.BLACK:
            self._delete_fixup(x)

    def _delete_fixup(self, node: RBNode) -> None:
        """Fix Red-Black Tree properties after deletion"""
        while node != self.root and node.color == Color.BLACK:
            if node == node.parent.left:
                sibling = node.parent.right

                if sibling.color == Color.RED:
                    sibling.color = Color.BLACK
                    node.parent.color = Color.RED
                    self._left_rotate(node.parent)
                    sibling = node.parent.right

                if sibling.left.color == Color.BLACK and sibling.right.color == Color.BLACK:
                    sibling.color = Color.RED
                    node = node.parent
                else:
                    if sibling.right.color == Color.BLACK:
                        sibling.left.color = Color.BLACK
                        sibling.color = Color.RED
                        self._right_rotate(sibling)
                        sibling = node.parent.right

                    sibling.color = node.parent.color
                    node.parent.color = Color.BLACK
                    sibling.right.color = Color.BLACK
                    self._left_rotate(node.parent)
                    node = self.root
            else:
                # Symmetric cases
                sibling = node.parent.left

                if sibling.color == Color.RED:
                    sibling.color = Color.BLACK
                    node.parent.color = Color.RED
                    self._right_rotate(node.parent)
                    sibling = node.parent.left

                if sibling.right.color == Color.BLACK and sibling.left.color == Color.BLACK:
                    sibling.color = Color.RED
                    node = node.parent
                else:
                    if sibling.left.color == Color.BLACK:
                        sibling.right.color = Color.BLACK
                        sibling.color = Color.RED
                        self._left_rotate(sibling)
                        sibling = node.parent.left

                    sibling.color = node.parent.color
                    node.parent.color = Color.BLACK
                    sibling.left.color = Color.BLACK
                    self._right_rotate(node.parent)
                    node = self.root

        node.color = Color.BLACK

    def _left_rotate(self, x: RBNode) -> None:
        """Perform left rotation"""
        y = x.right
        x.right = y.left

        if y.left != self.nil:
            y.left.parent = x

        y.parent = x.parent

        if x.parent is None:
            self.root = y
        elif x == x.parent.left:
            x.parent.left = y
        else:
            x.parent.right = y

        y.left = x
        x.parent = y

    def _right_rotate(self, y: RBNode) -> None:
        """Perform right rotation"""
        x = y.left
        y.left = x.right

        if x.right != self.nil:
            x.right.parent = y

        x.parent = y.parent

        if y.parent is None:
            self.root = x
        elif y == y.parent.right:
            y.parent.right = x
        else:
            y.parent.left = x

        x.right = y
        y.parent = x

    def _transplant(self, u: RBNode, v: RBNode) -> None:
        """Replace subtree rooted at u with subtree rooted at v"""
        if u.parent is None:
            self.root = v
        elif u == u.parent.left:
            u.parent.left = v
        else:
            u.parent.right = v
        v.parent = u.parent

    def _search_node(self, key: Any) -> RBNode:
        """Search for a node by key"""
        current = self.root
        while current != self.nil and key != current.key:
            if key < current.key:
                current = current.left
            else:
                current = current.right
        return current

    def _minimum(self, node: RBNode) -> RBNode:
        """Find minimum node in subtree"""
        while node.left != self.nil:
            node = node.left
        return node

    def _maximum(self, node: RBNode) -> RBNode:
        """Find maximum node in subtree"""
        while node.right != self.nil:
            node = node.right
        return node

    def _range_query_helper(self, node: RBNode, start_key: Any, end_key: Any, result: List[Judgment]) -> None:
        """Helper method for range queries"""
        if node == self.nil:
            return

        # If current key is in range, add to result
        if start_key <= node.key <= end_key:
            result.append(node.value)

        # Recursively search left subtree if needed
        if start_key <= node.key:
            self._range_query_helper(node.left, start_key, end_key, result)

        # Recursively search right subtree if needed
        if node.key <= end_key:
            self._range_query_helper(node.right, start_key, end_key, result)

    def _inorder_helper(self, node: RBNode) -> Iterator[Judgment]:
        """Helper for inorder traversal"""
        if node != self.nil:
            yield from self._inorder_helper(node.left)
            yield node.value
            yield from self._inorder_helper(node.right)

    def _height(self, node: RBNode) -> int:
        """Calculate height of subtree"""
        if node == self.nil:
            return 0
        return 1 + max(self._height(node.left), self._height(node.right))

    def _black_height(self, node: RBNode) -> int:
        """Calculate black height of subtree"""
        if node == self.nil:
            return 0

        left_height = self._black_height(node.left)
        right_height = self._black_height(node.right)

        # Add 1 if current node is black
        height_increment = 1 if node.color == Color.BLACK else 0
        return left_height + height_increment

    @staticmethod
    def _log2(n: int) -> int:
        """Calculate log base 2"""
        if n <= 1:
            return 0
        return 1 + RedBlackTree._log2(n // 2)


class JudgmentCollection:
    """
    High-level collection manager using Red-Black Trees for different sorting criteria.
    Maintains multiple sorted views of the same judgment data.
    """

    def __init__(self):
        # Multiple trees for different sorting criteria
        self.by_date = RedBlackTree(lambda j: j.judgment_date)
        self.by_citation = RedBlackTree(lambda j: j.neutral_citation)
        self.by_relevance = RedBlackTree(lambda j: -j.relevance_score)  # Descending order

        # Keep track of all judgments for cross-referencing
        self.judgments = {}  # citation -> judgment

    def add_judgment(self, judgment: Judgment) -> None:
        """Add a judgment to all sorted collections"""
        # Remove existing judgment if present
        if judgment.neutral_citation in self.judgments:
            self.remove_judgment(judgment.neutral_citation)

        # Add to all trees
        self.by_date.insert(judgment)
        self.by_citation.insert(judgment)
        self.by_relevance.insert(judgment)

        # Store in main collection
        self.judgments[judgment.neutral_citation] = judgment

    def remove_judgment(self, citation: str) -> bool:
        """Remove a judgment from all collections"""
        if citation not in self.judgments:
            return False

        judgment = self.judgments[citation]

        # Remove from all trees
        self.by_date.delete(judgment.judgment_date)
        self.by_citation.delete(judgment.neutral_citation)
        self.by_relevance.delete(-judgment.relevance_score)

        # Remove from main collection
        del self.judgments[citation]
        return True

    def get_by_citation(self, citation: str) -> Optional[Judgment]:
        """Get judgment by neutral citation"""
        return self.judgments.get(citation)

    def get_recent_judgments(self, limit: int = 10) -> List[Judgment]:
        """Get most recent judgments"""
        judgments = self.by_date.get_sorted_list(reverse=True)
        return judgments[:limit]

    def get_by_date_range(self, start_date: date, end_date: date) -> List[Judgment]:
        """Get judgments within date range"""
        return self.by_date.get_by_date_range(start_date, end_date)

    def get_by_relevance(self, limit: int = 10) -> List[Judgment]:
        """Get highest scoring judgments"""
        judgments = self.by_relevance.get_sorted_list()
        return judgments[:limit]

    def get_by_court(self, court: str) -> List[Judgment]:
        """Get all judgments from specific court"""
        return self.by_date.get_by_court(court)

    def search_citations(self, prefix: str) -> List[Judgment]:
        """Search for judgments with citations starting with prefix"""
        result = []
        for judgment in self.by_citation.inorder_traversal():
            if judgment.neutral_citation.startswith(prefix):
                result.append(judgment)
        return result

    def get_collection_stats(self) -> dict:
        """Get comprehensive collection statistics"""
        courts = {}
        years = {}

        for judgment in self.judgments.values():
            # Court statistics
            courts[judgment.court] = courts.get(judgment.court, 0) + 1

            # Year statistics
            year = judgment.judgment_date.year
            years[year] = years.get(year, 0) + 1

        return {
            'total_judgments': len(self.judgments),
            'date_tree_stats': self.by_date.get_statistics(),
            'citation_tree_stats': self.by_citation.get_statistics(),
            'relevance_tree_stats': self.by_relevance.get_statistics(),
            'courts': courts,
            'years': years
        }


def demonstrate_red_black_trees():
    """Demonstrate Red-Black Tree implementation with UK legal data."""

    print("=== Red-Black Trees for Sorted Judgments Demo ===\n")

    # Sample UK judgments
    judgments_data = [
        {
            'citation': '[2023] UKSC 15',
            'case_name': 'R (Miller) v Prime Minister',
            'court': 'UKSC',
            'date': date(2023, 5, 15),
            'uri': 'https://caselaw.nationalarchives.gov.uk/uksc/2023/15',
            'score': 0.95
        },
        {
            'citation': '[2023] EWCA Civ 892',
            'case_name': 'Smith v Secretary of State for Work and Pensions',
            'court': 'EWCA',
            'date': date(2023, 8, 22),
            'uri': 'https://caselaw.nationalarchives.gov.uk/ewca/civ/2023/892',
            'score': 0.87
        },
        {
            'citation': '[2023] EWHC 1456 (Admin)',
            'case_name': 'Jones v Local Authority',
            'court': 'EWHC',
            'date': date(2023, 6, 30),
            'uri': 'https://caselaw.nationalarchives.gov.uk/ewhc/admin/2023/1456',
            'score': 0.73
        },
        {
            'citation': '[2023] UKHL 7',
            'case_name': 'Williams v Crown Prosecution Service',
            'court': 'UKHL',
            'date': date(2023, 3, 10),
            'uri': 'https://caselaw.nationalarchives.gov.uk/ukhl/2023/7',
            'score': 0.91
        },
        {
            'citation': '[2023] UKFTT 234 (TC)',
            'case_name': 'Brown v HMRC',
            'court': 'UKFTT',
            'date': date(2023, 9, 5),
            'uri': 'https://caselaw.nationalarchives.gov.uk/ukftt/tc/2023/234',
            'score': 0.62
        }
    ]

    # Create judgment objects
    judgments = [
        Judgment(
            neutral_citation=data['citation'],
            case_name=data['case_name'],
            court=data['court'],
            judgment_date=data['date'],
            uri=data['uri'],
            relevance_score=data['score']
        )
        for data in judgments_data
    ]

    # 1. Basic Red-Black Tree Operations
    print("1. BASIC RED-BLACK TREE (sorted by date):")
    date_tree = RedBlackTree(lambda j: j.judgment_date)

    print("   Inserting judgments:")
    for judgment in judgments:
        date_tree.insert(judgment)
        print(f"   Added: {judgment.neutral_citation} ({judgment.judgment_date})")

    print(f"\n   Tree statistics: {date_tree.get_statistics()}")

    # 2. Sorted Traversal
    print(f"\n2. CHRONOLOGICAL ORDER (in-order traversal):")
    for i, judgment in enumerate(date_tree.inorder_traversal(), 1):
        print(f"   {i}. {judgment.judgment_date}: {judgment.case_name}")

    # 3. Range Queries
    print(f"\n3. DATE RANGE QUERIES:")
    start_date = date(2023, 5, 1)
    end_date = date(2023, 8, 31)

    range_results = date_tree.range_query(start_date, end_date)
    print(f"   Judgments between {start_date} and {end_date}:")
    for judgment in range_results:
        print(f"     - {judgment.judgment_date}: {judgment.case_name}")

    # 4. Judgment Collection with Multiple Sort Orders
    print(f"\n4. JUDGMENT COLLECTION (multiple sorted views):")
    collection = JudgmentCollection()

    for judgment in judgments:
        collection.add_judgment(judgment)

    print(f"   Collection statistics:")
    stats = collection.get_collection_stats()
    print(f"     Total judgments: {stats['total_judgments']}")
    print(f"     Courts: {stats['courts']}")
    print(f"     Years: {stats['years']}")

    # 5. Different Sorting Criteria
    print(f"\n5. SORTING BY DIFFERENT CRITERIA:")

    # Most recent judgments
    print(f"   Most recent judgments:")
    recent = collection.get_recent_judgments(3)
    for judgment in recent:
        print(f"     - {judgment.judgment_date}: {judgment.case_name}")

    # Highest relevance
    print(f"\n   Highest relevance scores:")
    relevant = collection.get_by_relevance(3)
    for judgment in relevant:
        print(f"     - {judgment.relevance_score:.2f}: {judgment.case_name}")

    # By court
    print(f"\n   Supreme Court judgments:")
    uksc_judgments = collection.get_by_court('UKSC')
    for judgment in uksc_judgments:
        print(f"     - {judgment.neutral_citation}: {judgment.case_name}")

    # 6. Search Operations
    print(f"\n6. SEARCH OPERATIONS:")

    # Citation search
    citation_search = collection.search_citations('[2023] EW')
    print(f"   Citations starting with '[2023] EW':")
    for judgment in citation_search:
        print(f"     - {judgment.neutral_citation}: {judgment.case_name}")

    # Specific judgment lookup
    specific = collection.get_by_citation('[2023] UKSC 15')
    if specific:
        print(f"\n   Found judgment: {specific.case_name}")
        print(f"     Court: {specific.court}")
        print(f"     Date: {specific.judgment_date}")
        print(f"     Relevance: {specific.relevance_score}")

    # 7. Tree Balance Verification
    print(f"\n7. TREE BALANCE VERIFICATION:")
    for tree_name, tree in [('Date', collection.by_date),
                           ('Citation', collection.by_citation),
                           ('Relevance', collection.by_relevance)]:
        stats = tree.get_statistics()
        print(f"   {tree_name} tree:")
        print(f"     Size: {stats['size']}")
        print(f"     Height: {stats['height']}")
        print(f"     Balanced: {stats['is_balanced']}")
        print(f"     Black height: {stats['black_height']}")

    return {
        'date_tree': date_tree,
        'collection': collection,
        'statistics': stats,
        'sample_queries': {
            'range_results': range_results,
            'recent_judgments': recent,
            'relevant_judgments': relevant,
            'citation_search': citation_search
        }
    }


if __name__ == "__main__":
    demonstrate_red_black_trees()