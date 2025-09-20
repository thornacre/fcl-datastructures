"""
Segment Trees for Range Query Optimization
==========================================

This module implements segment trees for efficient range query operations
in Find Case Law (FCL). Provides fast aggregation queries over date ranges,
score ranges, and other numerical attributes of legal documents.

Key FCL Use Cases:
- Fast date range queries for judgment filtering by time periods
- Efficient aggregation of relevance scores across document collections
- Range-based statistics computation (min, max, sum, average)
- Timeline visualization data preparation
- Performance optimization for complex search filters
"""

from typing import List, Optional, Any, Callable, Union, Tuple
from dataclasses import dataclass
from datetime import datetime, date
from abc import ABC, abstractmethod
import math


@dataclass
class JudgmentMetrics:
    """Represents metrics for a legal judgment"""
    judgment_id: str
    date: date
    relevance_score: float
    word_count: int
    citation_count: int
    court_level: int  # Hierarchy level (1=Supreme, 2=Appeal, etc.)

    def __str__(self) -> str:
        return f"{self.judgment_id}: score={self.relevance_score}, date={self.date}"


class SegmentTreeNode:
    """Node in a segment tree"""

    def __init__(self, start: int, end: int, value: Any = None):
        self.start = start
        self.end = end
        self.value = value
        self.left: Optional['SegmentTreeNode'] = None
        self.right: Optional['SegmentTreeNode'] = None
        self.lazy: Any = None  # For lazy propagation


class SegmentTree:
    """
    Generic segment tree implementation for range queries and updates.
    Supports various aggregation functions (sum, min, max, custom).
    """

    def __init__(self, data: List[Union[int, float]],
                 operation: Callable[[Any, Any], Any] = None,
                 identity: Any = None):
        """
        Initialize segment tree with data and aggregation operation.

        Args:
            data: List of numerical values
            operation: Binary function for combining values (default: sum)
            identity: Identity element for the operation (default: 0)
        """
        self.n = len(data)
        self.data = data[:]

        # Default to sum operation
        self.operation = operation or (lambda x, y: x + y)
        self.identity = identity if identity is not None else 0

        # Build the tree
        if self.n > 0:
            self.root = self._build(0, self.n - 1)
        else:
            self.root = None

    def query(self, left: int, right: int) -> Any:
        """
        Query the aggregate value in range [left, right].

        Args:
            left: Left boundary (inclusive)
            right: Right boundary (inclusive)

        Returns:
            Aggregated value over the range
        """
        if self.root is None or left > right:
            return self.identity
        return self._query(self.root, 0, self.n - 1, left, right)

    def update(self, index: int, value: Union[int, float]) -> None:
        """
        Update value at specific index.

        Args:
            index: Index to update
            value: New value
        """
        if 0 <= index < self.n:
            self.data[index] = value
            if self.root:
                self._update(self.root, 0, self.n - 1, index, value)

    def range_update(self, left: int, right: int, delta: Union[int, float]) -> None:
        """
        Update all values in range [left, right] by adding delta.
        Uses lazy propagation for efficiency.

        Args:
            left: Left boundary (inclusive)
            right: Right boundary (inclusive)
            delta: Value to add to all elements in range
        """
        if self.root and left <= right:
            self._range_update(self.root, 0, self.n - 1, left, right, delta)

    def _build(self, start: int, end: int) -> SegmentTreeNode:
        """Build segment tree recursively"""
        if start == end:
            # Leaf node
            return SegmentTreeNode(start, end, self.data[start])

        mid = (start + end) // 2
        node = SegmentTreeNode(start, end)
        node.left = self._build(start, mid)
        node.right = self._build(mid + 1, end)

        # Combine values from children
        node.value = self.operation(node.left.value, node.right.value)
        return node

    def _query(self, node: SegmentTreeNode, start: int, end: int,
               left: int, right: int) -> Any:
        """Query range recursively"""
        # Push down lazy updates
        self._push_lazy(node, start, end)

        # No overlap
        if right < start or left > end:
            return self.identity

        # Complete overlap
        if left <= start and end <= right:
            return node.value

        # Partial overlap
        mid = (start + end) // 2
        left_result = self._query(node.left, start, mid, left, right)
        right_result = self._query(node.right, mid + 1, end, left, right)

        return self.operation(left_result, right_result)

    def _update(self, node: SegmentTreeNode, start: int, end: int,
                index: int, value: Union[int, float]) -> None:
        """Update single element recursively"""
        if start == end:
            # Leaf node
            node.value = value
            return

        mid = (start + end) // 2
        if index <= mid:
            self._update(node.left, start, mid, index, value)
        else:
            self._update(node.right, mid + 1, end, index, value)

        # Update internal node
        node.value = self.operation(node.left.value, node.right.value)

    def _range_update(self, node: SegmentTreeNode, start: int, end: int,
                     left: int, right: int, delta: Union[int, float]) -> None:
        """Range update with lazy propagation"""
        # Push existing lazy updates
        self._push_lazy(node, start, end)

        # No overlap
        if right < start or left > end:
            return

        # Complete overlap
        if left <= start and end <= right:
            node.lazy = (node.lazy or 0) + delta
            self._push_lazy(node, start, end)
            return

        # Partial overlap
        mid = (start + end) // 2
        self._range_update(node.left, start, mid, left, right, delta)
        self._range_update(node.right, mid + 1, end, left, right, delta)

        # Update current node
        self._push_lazy(node.left, start, mid)
        self._push_lazy(node.right, mid + 1, end)
        node.value = self.operation(node.left.value, node.right.value)

    def _push_lazy(self, node: SegmentTreeNode, start: int, end: int) -> None:
        """Apply lazy updates to node"""
        if node.lazy is not None and node.lazy != 0:
            # Apply lazy update to current node
            if self.operation == (lambda x, y: x + y):  # Sum operation
                node.value += node.lazy * (end - start + 1)
            else:
                # For other operations, this needs to be customized
                node.value += node.lazy

            # Propagate to children
            if start != end:  # Not a leaf
                if node.left:
                    node.left.lazy = (node.left.lazy or 0) + node.lazy
                if node.right:
                    node.right.lazy = (node.right.lazy or 0) + node.lazy

            node.lazy = None


class DateRangeSegmentTree:
    """
    Specialized segment tree for date-based range queries on legal judgments.
    Maps dates to indices for efficient range operations.
    """

    def __init__(self, judgments: List[JudgmentMetrics]):
        """
        Initialize with judgment data.

        Args:
            judgments: List of judgment metrics
        """
        self.judgments = sorted(judgments, key=lambda j: j.date)

        # Create date to index mapping
        self.dates = [j.date for j in self.judgments]
        self.date_to_index = {date: i for i, date in enumerate(self.dates)}

        # Build multiple trees for different metrics
        self.relevance_tree = SegmentTree(
            [j.relevance_score for j in self.judgments],
            operation=lambda x, y: x + y  # Sum
        )

        self.count_tree = SegmentTree(
            [1 for _ in self.judgments],  # Count of judgments
            operation=lambda x, y: x + y
        )

        self.max_score_tree = SegmentTree(
            [j.relevance_score for j in self.judgments],
            operation=max,
            identity=float('-inf')
        )

        self.min_score_tree = SegmentTree(
            [j.relevance_score for j in self.judgments],
            operation=min,
            identity=float('inf')
        )

    def query_date_range(self, start_date: date, end_date: date) -> dict:
        """
        Query statistics for judgments in date range.

        Args:
            start_date: Start of date range
            end_date: End of date range

        Returns:
            Dictionary with aggregated statistics
        """
        # Find index range
        start_idx = self._find_date_index(start_date, left_bound=True)
        end_idx = self._find_date_index(end_date, left_bound=False)

        if start_idx > end_idx or start_idx >= len(self.judgments):
            return {
                'count': 0,
                'total_relevance': 0.0,
                'average_relevance': 0.0,
                'max_relevance': None,
                'min_relevance': None,
                'judgments': []
            }

        # Query all trees
        count = self.count_tree.query(start_idx, end_idx)
        total_relevance = self.relevance_tree.query(start_idx, end_idx)
        max_relevance = self.max_score_tree.query(start_idx, end_idx)
        min_relevance = self.min_score_tree.query(start_idx, end_idx)

        # Get actual judgments in range
        judgments_in_range = self.judgments[start_idx:end_idx + 1]

        return {
            'count': count,
            'total_relevance': total_relevance,
            'average_relevance': total_relevance / count if count > 0 else 0.0,
            'max_relevance': max_relevance if max_relevance != float('-inf') else None,
            'min_relevance': min_relevance if min_relevance != float('inf') else None,
            'judgments': judgments_in_range,
            'date_range': (self.dates[start_idx], self.dates[end_idx])
        }

    def update_judgment_score(self, judgment_id: str, new_score: float) -> bool:
        """
        Update relevance score for a specific judgment.

        Args:
            judgment_id: ID of judgment to update
            new_score: New relevance score

        Returns:
            True if judgment was found and updated
        """
        for i, judgment in enumerate(self.judgments):
            if judgment.judgment_id == judgment_id:
                old_score = judgment.relevance_score
                judgment.relevance_score = new_score

                # Update trees
                self.relevance_tree.update(i, new_score)
                self.max_score_tree.update(i, new_score)
                self.min_score_tree.update(i, new_score)

                return True
        return False

    def get_timeline_data(self, num_buckets: int = 12) -> List[dict]:
        """
        Get timeline data for visualization, grouped into buckets.

        Args:
            num_buckets: Number of time buckets to create

        Returns:
            List of dictionaries with timeline statistics
        """
        if not self.judgments:
            return []

        start_date = self.dates[0]
        end_date = self.dates[-1]

        # Calculate bucket size
        total_days = (end_date - start_date).days
        bucket_days = max(1, total_days // num_buckets)

        timeline = []
        current_date = start_date

        for i in range(num_buckets):
            bucket_end = current_date + datetime.timedelta(days=bucket_days)
            if i == num_buckets - 1:  # Last bucket
                bucket_end = end_date

            bucket_stats = self.query_date_range(current_date, bucket_end)

            timeline.append({
                'bucket_index': i,
                'start_date': current_date,
                'end_date': bucket_end,
                'judgment_count': bucket_stats['count'],
                'average_relevance': bucket_stats['average_relevance'],
                'max_relevance': bucket_stats['max_relevance'],
                'total_relevance': bucket_stats['total_relevance']
            })

            current_date = bucket_end + datetime.timedelta(days=1)

        return timeline

    def _find_date_index(self, target_date: date, left_bound: bool = True) -> int:
        """
        Binary search to find date index.

        Args:
            target_date: Date to find
            left_bound: If True, find leftmost position; if False, rightmost

        Returns:
            Index of date position
        """
        left, right = 0, len(self.dates) - 1
        result = len(self.dates)  # Default to beyond range

        while left <= right:
            mid = (left + right) // 2

            if left_bound:
                if self.dates[mid] >= target_date:
                    result = mid
                    right = mid - 1
                else:
                    left = mid + 1
            else:
                if self.dates[mid] <= target_date:
                    result = mid
                    left = mid + 1
                else:
                    right = mid - 1

        return result


class ScoreRangeTree:
    """
    Segment tree specialized for relevance score range queries.
    Enables efficient filtering by score thresholds.
    """

    def __init__(self, judgments: List[JudgmentMetrics], score_precision: int = 100):
        """
        Initialize score range tree.

        Args:
            judgments: List of judgment metrics
            score_precision: Number of score buckets (0.0 to 1.0 mapped to 0 to precision-1)
        """
        self.judgments = judgments
        self.score_precision = score_precision

        # Create score buckets
        self.score_buckets = [[] for _ in range(score_precision)]

        # Distribute judgments into buckets
        for judgment in judgments:
            bucket_idx = min(score_precision - 1,
                           int(judgment.relevance_score * score_precision))
            self.score_buckets[bucket_idx].append(judgment)

        # Build segment tree with counts per bucket
        bucket_counts = [len(bucket) for bucket in self.score_buckets]
        self.count_tree = SegmentTree(bucket_counts, operation=lambda x, y: x + y)

        # Build trees for other aggregations
        bucket_max_scores = []
        bucket_min_scores = []

        for bucket in self.score_buckets:
            if bucket:
                bucket_max_scores.append(max(j.relevance_score for j in bucket))
                bucket_min_scores.append(min(j.relevance_score for j in bucket))
            else:
                bucket_max_scores.append(0.0)
                bucket_min_scores.append(0.0)

        self.max_score_tree = SegmentTree(bucket_max_scores, operation=max, identity=0.0)
        self.min_score_tree = SegmentTree(bucket_min_scores, operation=min, identity=1.0)

    def query_score_range(self, min_score: float, max_score: float) -> dict:
        """
        Query judgments within score range.

        Args:
            min_score: Minimum score (inclusive)
            max_score: Maximum score (inclusive)

        Returns:
            Dictionary with statistics and matching judgments
        """
        # Convert scores to bucket indices
        min_bucket = max(0, int(min_score * self.score_precision))
        max_bucket = min(self.score_precision - 1, int(max_score * self.score_precision))

        if min_bucket > max_bucket:
            return {
                'count': 0,
                'judgments': [],
                'score_range': (min_score, max_score)
            }

        # Query segment tree
        count = self.count_tree.query(min_bucket, max_bucket)

        # Collect matching judgments
        matching_judgments = []
        for bucket_idx in range(min_bucket, max_bucket + 1):
            for judgment in self.score_buckets[bucket_idx]:
                if min_score <= judgment.relevance_score <= max_score:
                    matching_judgments.append(judgment)

        # Sort by score descending
        matching_judgments.sort(key=lambda j: j.relevance_score, reverse=True)

        return {
            'count': len(matching_judgments),
            'judgments': matching_judgments,
            'score_range': (min_score, max_score),
            'bucket_range': (min_bucket, max_bucket)
        }

    def get_score_distribution(self) -> List[dict]:
        """
        Get distribution of judgments across score buckets.

        Returns:
            List of bucket statistics
        """
        distribution = []

        for i, bucket in enumerate(self.score_buckets):
            bucket_min = i / self.score_precision
            bucket_max = (i + 1) / self.score_precision

            if bucket:
                avg_score = sum(j.relevance_score for j in bucket) / len(bucket)
                max_score = max(j.relevance_score for j in bucket)
                min_score = min(j.relevance_score for j in bucket)
            else:
                avg_score = max_score = min_score = 0.0

            distribution.append({
                'bucket_index': i,
                'score_range': (bucket_min, bucket_max),
                'count': len(bucket),
                'average_score': avg_score,
                'max_score': max_score,
                'min_score': min_score
            })

        return distribution


def demonstrate_segment_trees():
    """Demonstrate segment tree implementations with UK legal judgment data."""

    print("=== Segment Trees for Range Query Optimization Demo ===\n")

    # Sample UK legal judgment metrics
    import datetime
    sample_judgments = [
        JudgmentMetrics(
            judgment_id="uksc_2023_15",
            date=date(2023, 5, 15),
            relevance_score=0.95,
            word_count=12500,
            citation_count=45,
            court_level=1
        ),
        JudgmentMetrics(
            judgment_id="ewca_2023_892",
            date=date(2023, 8, 22),
            relevance_score=0.87,
            word_count=8900,
            citation_count=32,
            court_level=2
        ),
        JudgmentMetrics(
            judgment_id="ewhc_2023_1456",
            date=date(2023, 6, 30),
            relevance_score=0.73,
            word_count=6700,
            citation_count=18,
            court_level=3
        ),
        JudgmentMetrics(
            judgment_id="ukhl_2023_7",
            date=date(2023, 3, 10),
            relevance_score=0.91,
            word_count=15200,
            citation_count=67,
            court_level=1
        ),
        JudgmentMetrics(
            judgment_id="ukftt_2023_234",
            date=date(2023, 9, 5),
            relevance_score=0.62,
            word_count=4500,
            citation_count=12,
            court_level=5
        ),
        JudgmentMetrics(
            judgment_id="ewca_2023_445",
            date=date(2023, 7, 18),
            relevance_score=0.84,
            word_count=9800,
            citation_count=28,
            court_level=2
        ),
        JudgmentMetrics(
            judgment_id="ewhc_2023_789",
            date=date(2023, 4, 25),
            relevance_score=0.78,
            word_count=7200,
            citation_count=22,
            court_level=3
        ),
    ]

    # 1. Basic Segment Tree Demo
    print("1. BASIC SEGMENT TREE OPERATIONS:")
    values = [j.relevance_score for j in sample_judgments]
    print(f"   Relevance scores: {[round(v, 2) for v in values]}")

    sum_tree = SegmentTree(values, operation=lambda x, y: x + y)
    max_tree = SegmentTree(values, operation=max, identity=0.0)
    min_tree = SegmentTree(values, operation=min, identity=1.0)

    print(f"   Sum of all scores: {sum_tree.query(0, len(values) - 1):.2f}")
    print(f"   Max score: {max_tree.query(0, len(values) - 1):.2f}")
    print(f"   Min score: {min_tree.query(0, len(values) - 1):.2f}")

    # Range queries
    print(f"\n   Range queries (indices 1-4):")
    print(f"     Sum: {sum_tree.query(1, 4):.2f}")
    print(f"     Max: {max_tree.query(1, 4):.2f}")
    print(f"     Min: {min_tree.query(1, 4):.2f}")

    # 2. Date Range Segment Tree
    print(f"\n2. DATE RANGE QUERIES:")
    date_tree = DateRangeSegmentTree(sample_judgments)

    # Query specific date ranges
    date_ranges = [
        (date(2023, 5, 1), date(2023, 7, 31)),   # Spring/Summer 2023
        (date(2023, 8, 1), date(2023, 9, 30)),   # Late 2023
        (date(2023, 1, 1), date(2023, 12, 31)),  # All 2023
    ]

    for start_date, end_date in date_ranges:
        result = date_tree.query_date_range(start_date, end_date)
        print(f"\n   Date range {start_date} to {end_date}:")
        print(f"     Count: {result['count']}")
        print(f"     Average relevance: {result['average_relevance']:.3f}")
        print(f"     Max relevance: {result['max_relevance']:.3f}")
        print(f"     Min relevance: {result['min_relevance']:.3f}")
        print(f"     Judgments: {[j.judgment_id for j in result['judgments'][:3]]}...")

    # 3. Timeline Analysis
    print(f"\n3. TIMELINE ANALYSIS:")
    timeline = date_tree.get_timeline_data(num_buckets=6)

    print(f"   Timeline data (6 buckets):")
    for bucket in timeline:
        print(f"     Bucket {bucket['bucket_index']}: "
              f"{bucket['start_date']} to {bucket['end_date']}")
        print(f"       Judgments: {bucket['judgment_count']}")
        print(f"       Avg relevance: {bucket['average_relevance']:.3f}")

    # 4. Score Range Queries
    print(f"\n4. SCORE RANGE QUERIES:")
    score_tree = ScoreRangeTree(sample_judgments, score_precision=100)

    score_ranges = [
        (0.9, 1.0),   # High relevance
        (0.7, 0.9),   # Medium relevance
        (0.0, 0.7),   # Lower relevance
    ]

    for min_score, max_score in score_ranges:
        result = score_tree.query_score_range(min_score, max_score)
        print(f"\n   Score range {min_score} to {max_score}:")
        print(f"     Count: {result['count']}")
        print(f"     Judgments:")
        for judgment in result['judgments'][:3]:
            print(f"       {judgment.judgment_id}: {judgment.relevance_score:.3f}")

    # 5. Score Distribution Analysis
    print(f"\n5. SCORE DISTRIBUTION ANALYSIS:")
    distribution = score_tree.get_score_distribution()

    # Show non-empty buckets
    print(f"   Score distribution (non-empty buckets):")
    non_empty_buckets = [b for b in distribution if b['count'] > 0]
    for bucket in non_empty_buckets[:5]:  # Show first 5
        score_range = bucket['score_range']
        print(f"     {score_range[0]:.2f}-{score_range[1]:.2f}: "
              f"{bucket['count']} judgments, "
              f"avg={bucket['average_score']:.3f}")

    # 6. Dynamic Updates
    print(f"\n6. DYNAMIC UPDATES:")
    print(f"   Original uksc_2023_15 score: "
          f"{next(j.relevance_score for j in sample_judgments if j.judgment_id == 'uksc_2023_15')}")

    # Update score
    updated = date_tree.update_judgment_score("uksc_2023_15", 0.99)
    print(f"   Update successful: {updated}")

    # Query again to show update effect
    result = date_tree.query_date_range(date(2023, 5, 1), date(2023, 5, 31))
    print(f"   Updated May 2023 stats:")
    print(f"     Average relevance: {result['average_relevance']:.3f}")
    print(f"     Max relevance: {result['max_relevance']:.3f}")

    # 7. Range Updates Demo
    print(f"\n7. RANGE UPDATES (Lazy Propagation):")
    update_tree = SegmentTree([1, 2, 3, 4, 5], operation=lambda x, y: x + y)

    print(f"   Original array: {update_tree.data}")
    print(f"   Sum of range [1,3]: {update_tree.query(1, 3)}")

    # Range update: add 10 to indices 1-3
    update_tree.range_update(1, 3, 10)
    print(f"   After adding 10 to range [1,3]:")
    print(f"   Sum of range [1,3]: {update_tree.query(1, 3)}")
    print(f"   Sum of entire array: {update_tree.query(0, 4)}")

    # 8. Performance Comparison
    print(f"\n8. PERFORMANCE BENEFITS:")
    print(f"   Segment tree advantages:")
    print(f"     - Range queries: O(log n) vs O(n) linear scan")
    print(f"     - Range updates: O(log n) with lazy propagation")
    print(f"     - Multiple query types on same data structure")
    print(f"     - Memory efficient for large datasets")

    # Example with larger dataset
    large_judgments = sample_judgments * 1000  # Simulate 7000 judgments
    large_date_tree = DateRangeSegmentTree(large_judgments)

    print(f"\n   Large dataset example ({len(large_judgments)} judgments):")
    large_result = large_date_tree.query_date_range(date(2023, 1, 1), date(2023, 12, 31))
    print(f"     Query result: {large_result['count']} judgments")
    print(f"     Average relevance: {large_result['average_relevance']:.3f}")

    return {
        'date_tree': date_tree,
        'score_tree': score_tree,
        'basic_trees': {
            'sum_tree': sum_tree,
            'max_tree': max_tree,
            'min_tree': min_tree
        },
        'timeline': timeline,
        'score_distribution': distribution,
        'sample_judgments': sample_judgments
    }


if __name__ == "__main__":
    demonstrate_segment_trees()