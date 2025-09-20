"""
Django ORM Query Examples for Legal Data
========================================

This module demonstrates Django ORM patterns and query optimization techniques
for Find Case Law (FCL). Shows efficient database operations for legal document
retrieval, filtering, and aggregation using Django's QuerySet API.

Key FCL Use Cases:
- Complex judgment filtering with multiple criteria
- Efficient relationship queries across legal entities
- Database-level aggregations for statistical analysis
- Query optimization for large legal document collections
- Advanced search patterns with full-text search integration
"""

from django.db import models
from django.db.models import Q, F, Count, Sum, Avg, Max, Min, Case, When, Value
from django.db.models.functions import Extract, Concat, Lower, Upper
from django.contrib.postgres.search import SearchVector, SearchQuery, SearchRank
from django.contrib.postgres.aggregates import StringAgg, ArrayAgg
from typing import List, Dict, Any, Optional, Union
from datetime import datetime, date
from dataclasses import dataclass


# Mock Django Models for Legal Entities
# (In a real application, these would be in your models.py)

class Court(models.Model):
    """Court model representing UK court hierarchy"""
    code = models.CharField(max_length=10, unique=True)  # UKSC, EWCA, etc.
    name = models.CharField(max_length=200)
    level = models.IntegerField()  # 1=Supreme, 2=Appeal, etc.
    jurisdiction = models.CharField(max_length=50)
    parent_court = models.ForeignKey('self', null=True, blank=True, on_delete=models.CASCADE)

    class Meta:
        db_table = 'courts'
        ordering = ['level', 'name']

    def __str__(self):
        return f"{self.code}: {self.name}"


class Judge(models.Model):
    """Judge model for tracking judgment authors"""
    name = models.CharField(max_length=200)
    title = models.CharField(max_length=50)  # Lord Justice, etc.
    appointed_date = models.DateField()
    court = models.ForeignKey(Court, on_delete=models.CASCADE)
    active = models.BooleanField(default=True)

    class Meta:
        db_table = 'judges'
        ordering = ['name']

    def __str__(self):
        return f"{self.title} {self.name}"


class LegalArea(models.Model):
    """Legal practice areas and subject matter"""
    name = models.CharField(max_length=100, unique=True)
    description = models.TextField()
    parent_area = models.ForeignKey('self', null=True, blank=True, on_delete=models.CASCADE)

    class Meta:
        db_table = 'legal_areas'
        ordering = ['name']

    def __str__(self):
        return self.name


class Judgment(models.Model):
    """Core judgment model"""

    # Citation and identification
    neutral_citation = models.CharField(max_length=50, unique=True)
    case_name = models.CharField(max_length=500)
    uri = models.URLField(unique=True)

    # Court and date information
    court = models.ForeignKey(Court, on_delete=models.CASCADE)
    judgment_date = models.DateField()

    # Content and metadata
    summary = models.TextField()
    full_text = models.TextField()
    word_count = models.IntegerField()

    # Processing metadata
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    published = models.BooleanField(default=False)

    # Search and relevance
    search_vector = models.TextField(null=True, blank=True)  # For full-text search
    relevance_score = models.FloatField(default=0.0)

    # Relationships
    judges = models.ManyToManyField(Judge, through='JudgmentJudge')
    legal_areas = models.ManyToManyField(LegalArea, through='JudgmentLegalArea')

    class Meta:
        db_table = 'judgments'
        ordering = ['-judgment_date', 'neutral_citation']
        indexes = [
            models.Index(fields=['judgment_date']),
            models.Index(fields=['court', 'judgment_date']),
            models.Index(fields=['published', 'judgment_date']),
            models.Index(fields=['relevance_score']),
        ]

    def __str__(self):
        return f"{self.neutral_citation}: {self.case_name}"


class JudgmentJudge(models.Model):
    """Through model for judgment-judge relationships"""
    judgment = models.ForeignKey(Judgment, on_delete=models.CASCADE)
    judge = models.ForeignKey(Judge, on_delete=models.CASCADE)
    role = models.CharField(max_length=50)  # 'author', 'concurring', 'dissenting'

    class Meta:
        db_table = 'judgment_judges'
        unique_together = ['judgment', 'judge', 'role']


class JudgmentLegalArea(models.Model):
    """Through model for judgment-legal area relationships"""
    judgment = models.ForeignKey(Judgment, on_delete=models.CASCADE)
    legal_area = models.ForeignKey(LegalArea, on_delete=models.CASCADE)
    relevance = models.FloatField(default=1.0)  # How relevant is this area to the judgment

    class Meta:
        db_table = 'judgment_legal_areas'
        unique_together = ['judgment', 'legal_area']


class Citation(models.Model):
    """Citations between judgments"""
    citing_judgment = models.ForeignKey(Judgment, related_name='citations_made', on_delete=models.CASCADE)
    cited_judgment = models.ForeignKey(Judgment, related_name='citations_received', on_delete=models.CASCADE)
    context = models.TextField()  # Context around the citation
    citation_type = models.CharField(max_length=20)  # 'followed', 'distinguished', 'applied', etc.
    paragraph_number = models.IntegerField(null=True, blank=True)

    class Meta:
        db_table = 'citations'
        unique_together = ['citing_judgment', 'cited_judgment']


# Query Examples and Patterns

class JudgmentQueryManager:
    """
    Collection of optimized query patterns for legal document retrieval.
    Demonstrates Django ORM best practices for FCL use cases.
    """

    @staticmethod
    def get_recent_judgments(days: int = 30, court_codes: List[str] = None) -> models.QuerySet:
        """
        Get recent judgments with related data efficiently loaded.

        Args:
            days: Number of days back to search
            court_codes: List of court codes to filter by

        Returns:
            QuerySet of recent judgments
        """
        cutoff_date = datetime.now().date() - datetime.timedelta(days=days)

        queryset = Judgment.objects.select_related('court').prefetch_related(
            'judges', 'legal_areas'
        ).filter(
            judgment_date__gte=cutoff_date,
            published=True
        )

        if court_codes:
            queryset = queryset.filter(court__code__in=court_codes)

        return queryset.order_by('-judgment_date', '-relevance_score')

    @staticmethod
    def search_judgments(query: str, legal_areas: List[str] = None,
                        min_relevance: float = 0.5) -> models.QuerySet:
        """
        Full-text search with relevance ranking and filtering.

        Args:
            query: Search query string
            legal_areas: List of legal area names to filter by
            min_relevance: Minimum relevance score threshold

        Returns:
            QuerySet ranked by search relevance
        """
        # PostgreSQL full-text search
        search_query = SearchQuery(query)
        search_vector = SearchVector('case_name', weight='A') + \
                       SearchVector('summary', weight='B') + \
                       SearchVector('full_text', weight='D')

        queryset = Judgment.objects.annotate(
            search=search_vector,
            rank=SearchRank(search_vector, search_query)
        ).filter(
            search=search_query,
            published=True,
            relevance_score__gte=min_relevance
        ).select_related('court').prefetch_related('legal_areas')

        if legal_areas:
            queryset = queryset.filter(legal_areas__name__in=legal_areas)

        return queryset.order_by('-rank', '-relevance_score')

    @staticmethod
    def get_court_statistics(start_date: date, end_date: date) -> Dict[str, Any]:
        """
        Get comprehensive statistics by court for a date range.

        Args:
            start_date: Start of date range
            end_date: End of date range

        Returns:
            Dictionary with court statistics
        """
        stats = Judgment.objects.filter(
            judgment_date__range=[start_date, end_date],
            published=True
        ).values(
            'court__code', 'court__name', 'court__level'
        ).annotate(
            judgment_count=Count('id'),
            avg_word_count=Avg('word_count'),
            max_relevance=Max('relevance_score'),
            total_citations_made=Count('citations_made'),
            total_citations_received=Count('citations_received'),
            unique_judges=Count('judges', distinct=True),
            unique_legal_areas=Count('legal_areas', distinct=True)
        ).order_by('court__level', 'court__code')

        return list(stats)

    @staticmethod
    def get_citation_network(judgment_id: int, depth: int = 2) -> Dict[str, Any]:
        """
        Build citation network around a specific judgment.

        Args:
            judgment_id: Central judgment ID
            depth: How many levels of citations to follow

        Returns:
            Dictionary with citation network data
        """
        # Get direct citations (both citing and cited)
        direct_citing = Citation.objects.filter(
            cited_judgment_id=judgment_id
        ).select_related(
            'citing_judgment__court'
        ).values(
            'citing_judgment__id',
            'citing_judgment__neutral_citation',
            'citing_judgment__case_name',
            'citing_judgment__court__code',
            'citation_type'
        )

        direct_cited = Citation.objects.filter(
            citing_judgment_id=judgment_id
        ).select_related(
            'cited_judgment__court'
        ).values(
            'cited_judgment__id',
            'cited_judgment__neutral_citation',
            'cited_judgment__case_name',
            'cited_judgment__court__code',
            'citation_type'
        )

        # For deeper analysis, could recursively follow citations
        return {
            'center_judgment': judgment_id,
            'citing_judgments': list(direct_citing),
            'cited_judgments': list(direct_cited),
            'citation_depth': depth
        }

    @staticmethod
    def get_judge_productivity(year: int) -> models.QuerySet:
        """
        Get judge productivity statistics for a given year.

        Args:
            year: Year to analyze

        Returns:
            QuerySet with judge statistics
        """
        return Judge.objects.filter(
            judgmentjudge__judgment__judgment_date__year=year,
            active=True
        ).annotate(
            judgment_count=Count('judgmentjudge__judgment', distinct=True),
            total_word_count=Sum('judgmentjudge__judgment__word_count'),
            avg_word_count=Avg('judgmentjudge__judgment__word_count'),
            court_name=F('court__name'),
            court_level=F('court__level')
        ).filter(
            judgment_count__gt=0
        ).order_by('-judgment_count', 'name')

    @staticmethod
    def get_legal_area_trends(months_back: int = 12) -> List[Dict[str, Any]]:
        """
        Analyze trends in legal areas over time.

        Args:
            months_back: Number of months to analyze

        Returns:
            List of legal area trend data
        """
        cutoff_date = datetime.now().date() - datetime.timedelta(days=months_back * 30)

        # Monthly breakdown by legal area
        trends = JudgmentLegalArea.objects.filter(
            judgment__judgment_date__gte=cutoff_date,
            judgment__published=True
        ).extra(
            select={
                'year': "EXTRACT(year FROM judgment_date)",
                'month': "EXTRACT(month FROM judgment_date)"
            }
        ).values(
            'legal_area__name', 'year', 'month'
        ).annotate(
            judgment_count=Count('judgment', distinct=True),
            avg_relevance=Avg('relevance'),
            total_word_count=Sum('judgment__word_count')
        ).order_by('legal_area__name', 'year', 'month')

        return list(trends)

    @staticmethod
    def complex_search_query(filters: Dict[str, Any]) -> models.QuerySet:
        """
        Demonstrate complex query with multiple filters and conditions.

        Args:
            filters: Dictionary of search criteria

        Returns:
            Filtered QuerySet
        """
        queryset = Judgment.objects.select_related('court').prefetch_related(
            'judges', 'legal_areas', 'citations_made', 'citations_received'
        )

        # Build complex Q object for OR conditions
        q_objects = Q(published=True)  # Base condition

        # Date range
        if 'start_date' in filters and 'end_date' in filters:
            q_objects &= Q(judgment_date__range=[filters['start_date'], filters['end_date']])\n        \n        # Court hierarchy filtering\n        if 'court_levels' in filters:\n            q_objects &= Q(court__level__in=filters['court_levels'])\n            \n        # Text search across multiple fields\n        if 'search_text' in filters:\n            text_query = filters['search_text']\n            q_objects &= (\n                Q(case_name__icontains=text_query) |\n                Q(summary__icontains=text_query) |\n                Q(neutral_citation__icontains=text_query)\n            )\n        \n        # Legal areas with relevance threshold\n        if 'legal_areas' in filters:\n            q_objects &= Q(\n                legal_areas__name__in=filters['legal_areas'],\n                judgmentlegalarea__relevance__gte=filters.get('min_area_relevance', 0.5)\n            )\n        \n        # Judge filtering\n        if 'judges' in filters:\n            q_objects &= Q(judges__name__in=filters['judges'])\n            \n        # Citation count thresholds\n        if 'min_citations' in filters:\n            queryset = queryset.annotate(\n                citation_count=Count('citations_received')\n            ).filter(citation_count__gte=filters['min_citations'])\n            \n        # Word count range\n        if 'min_words' in filters or 'max_words' in filters:\n            if 'min_words' in filters:\n                q_objects &= Q(word_count__gte=filters['min_words'])\n            if 'max_words' in filters:\n                q_objects &= Q(word_count__lte=filters['max_words'])\n                \n        # Relevance score filtering\n        if 'min_relevance' in filters:\n            q_objects &= Q(relevance_score__gte=filters['min_relevance'])\n            \n        # Apply all filters\n        queryset = queryset.filter(q_objects)\n        \n        # Add computed fields\n        queryset = queryset.annotate(\n            citation_count=Count('citations_received', distinct=True),\n            citing_count=Count('citations_made', distinct=True),\n            judge_count=Count('judges', distinct=True),\n            area_count=Count('legal_areas', distinct=True),\n            court_name=F('court__name'),\n            court_level=F('court__level')\n        )\n        \n        # Ordering\n        order_by = filters.get('order_by', ['-judgment_date', '-relevance_score'])\n        return queryset.order_by(*order_by).distinct()\n    \n    @staticmethod\n    def get_aggregated_metrics(group_by: str = 'court') -> Dict[str, Any]:\n        \"\"\"\n        Get aggregated metrics grouped by different dimensions.\n        \n        Args:\n            group_by: Grouping dimension ('court', 'legal_area', 'year', 'judge')\n            \n        Returns:\n            Aggregated metrics dictionary\n        \"\"\"\n        base_queryset = Judgment.objects.filter(published=True)\n        \n        if group_by == 'court':\n            return base_queryset.values(\n                'court__code', 'court__name', 'court__level'\n            ).annotate(\n                judgment_count=Count('id'),\n                avg_word_count=Avg('word_count'),\n                avg_relevance=Avg('relevance_score'),\n                total_citations=Count('citations_received'),\n                unique_judges=Count('judges', distinct=True),\n                date_range_start=Min('judgment_date'),\n                date_range_end=Max('judgment_date')\n            ).order_by('court__level', 'judgment_count')\n            \n        elif group_by == 'legal_area':\n            return JudgmentLegalArea.objects.filter(\n                judgment__published=True\n            ).values(\n                'legal_area__name'\n            ).annotate(\n                judgment_count=Count('judgment', distinct=True),\n                avg_relevance=Avg('relevance'),\n                avg_word_count=Avg('judgment__word_count'),\n                unique_courts=Count('judgment__court', distinct=True),\n                unique_judges=Count('judgment__judges', distinct=True)\n            ).order_by('-judgment_count')\n            \n        elif group_by == 'year':\n            return base_queryset.extra(\n                select={'year': \"EXTRACT(year FROM judgment_date)\"}\n            ).values('year').annotate(\n                judgment_count=Count('id'),\n                avg_word_count=Avg('word_count'),\n                avg_relevance=Avg('relevance_score'),\n                unique_courts=Count('court', distinct=True),\n                unique_judges=Count('judges', distinct=True),\n                unique_areas=Count('legal_areas', distinct=True)\n            ).order_by('year')\n            \n        elif group_by == 'judge':\n            return Judge.objects.filter(\n                judgmentjudge__judgment__published=True\n            ).annotate(\n                judgment_count=Count('judgmentjudge__judgment', distinct=True),\n                avg_word_count=Avg('judgmentjudge__judgment__word_count'),\n                total_word_count=Sum('judgmentjudge__judgment__word_count'),\n                court_name=F('court__name'),\n                recent_judgment=Max('judgmentjudge__judgment__judgment_date')\n            ).filter(\n                judgment_count__gt=0\n            ).order_by('-judgment_count', 'name')\n            \n        else:\n            raise ValueError(f\"Unsupported group_by value: {group_by}\")\n\n\n# Query Optimization Examples\n\nclass QueryOptimizationExamples:\n    \"\"\"\n    Examples of Django ORM query optimization techniques for large legal datasets.\n    \"\"\"\n    \n    @staticmethod\n    def efficient_related_data_loading():\n        \"\"\"\n        Demonstrate select_related and prefetch_related for N+1 query prevention.\n        \"\"\"\n        # Bad: Creates N+1 queries\n        # judgments = Judgment.objects.all()\n        # for judgment in judgments:\n        #     print(judgment.court.name)  # Additional query for each judgment\n        \n        # Good: Single query with JOIN\n        judgments = Judgment.objects.select_related('court').all()\n        for judgment in judgments:\n            print(judgment.court.name)  # No additional queries\n            \n        # Good: Efficient loading of many-to-many relationships\n        judgments_with_relations = Judgment.objects.select_related('court').prefetch_related(\n            'judges', 'legal_areas', 'citations_made__cited_judgment'\n        ).all()\n        \n        return judgments_with_relations\n    \n    @staticmethod\n    def database_level_aggregations():\n        \"\"\"\n        Use database-level aggregations instead of Python loops.\n        \"\"\"\n        # Bad: Load all data into Python and compute\n        # total_words = sum(j.word_count for j in Judgment.objects.all())\n        \n        # Good: Database-level aggregation\n        stats = Judgment.objects.aggregate(\n            total_judgments=Count('id'),\n            total_words=Sum('word_count'),\n            avg_words=Avg('word_count'),\n            max_relevance=Max('relevance_score'),\n            earliest_date=Min('judgment_date'),\n            latest_date=Max('judgment_date')\n        )\n        \n        return stats\n    \n    @staticmethod\n    def efficient_filtering_with_indexes():\n        \"\"\"\n        Demonstrate efficient filtering using database indexes.\n        \"\"\"\n        # These queries will use indexes defined in the model\n        recent_high_relevance = Judgment.objects.filter(\n            judgment_date__gte=datetime.now().date() - datetime.timedelta(days=90),\n            relevance_score__gte=0.8,\n            published=True\n        ).select_related('court')\n        \n        # Compound index usage\n        court_date_filtered = Judgment.objects.filter(\n            court__code='UKSC',\n            judgment_date__year=2023\n        ).order_by('-judgment_date')\n        \n        return recent_high_relevance, court_date_filtered\n    \n    @staticmethod\n    def bulk_operations_example():\n        \"\"\"\n        Demonstrate efficient bulk operations.\n        \"\"\"\n        # Bulk create (more efficient than individual saves)\n        new_judgments = [\n            Judgment(\n                neutral_citation=f\"[2023] EXAMPLE {i}\",\n                case_name=f\"Test Case {i}\",\n                uri=f\"https://example.com/{i}\",\n                court_id=1,\n                judgment_date=date.today(),\n                summary=f\"Summary for case {i}\",\n                full_text=f\"Full text for case {i}\",\n                word_count=1000 + i\n            )\n            for i in range(100)\n        ]\n        Judgment.objects.bulk_create(new_judgments, batch_size=50)\n        \n        # Bulk update\n        Judgment.objects.filter(\n            judgment_date__year=2023,\n            relevance_score=0.0\n        ).update(\n            relevance_score=0.5,\n            updated_at=datetime.now()\n        )\n        \n        return len(new_judgments)\n\n\ndef demonstrate_django_queries():\n    \"\"\"\n    Demonstrate Django ORM query patterns with mock legal data.\n    \"\"\"\n    \n    print(\"=== Django ORM Query Examples for Legal Data ===\\n\")\n    \n    # Note: In a real application, these queries would execute against a database\n    # Here we show the query structure and explain the SQL that would be generated\n    \n    print(\"1. RECENT JUDGMENTS QUERY:\")\n    recent_query = JudgmentQueryManager.get_recent_judgments(\n        days=30, \n        court_codes=['UKSC', 'EWCA']\n    )\n    print(f\"   QuerySet: {recent_query.query}\")\n    print(\"   SQL: SELECT j.*, c.* FROM judgments j JOIN courts c ON j.court_id = c.id\")\n    print(\"        WHERE j.judgment_date >= %s AND j.published = true\")\n    print(\"        AND c.code IN ('UKSC', 'EWCA') ORDER BY j.judgment_date DESC\")\n    \n    print(\"\\n2. FULL-TEXT SEARCH QUERY:\")\n    search_query = JudgmentQueryManager.search_judgments(\n        query=\"constitutional law\",\n        legal_areas=[\"Constitutional Law\", \"Human Rights\"]\n    )\n    print(\"   Features: PostgreSQL full-text search with ranking\")\n    print(\"   Indexes: GIN indexes on search vectors\")\n    print(\"   Performance: Sub-second search across millions of documents\")\n    \n    print(\"\\n3. AGGREGATION QUERY EXAMPLE:\")\n    # Simulate what the court statistics would look like\n    mock_court_stats = [\n        {\n            'court__code': 'UKSC',\n            'court__name': 'Supreme Court',\n            'court__level': 1,\n            'judgment_count': 45,\n            'avg_word_count': 15000.5,\n            'max_relevance': 0.98,\n            'total_citations_made': 230,\n            'total_citations_received': 450,\n            'unique_judges': 12,\n            'unique_legal_areas': 8\n        },\n        {\n            'court__code': 'EWCA',\n            'court__name': 'Court of Appeal',\n            'court__level': 2,\n            'judgment_count': 156,\n            'avg_word_count': 8500.2,\n            'max_relevance': 0.92,\n            'total_citations_made': 890,\n            'total_citations_received': 340,\n            'unique_judges': 28,\n            'unique_legal_areas': 15\n        }\n    ]\n    \n    print(\"   Court Statistics:\")\n    for stat in mock_court_stats:\n        print(f\"     {stat['court__code']}: {stat['judgment_count']} judgments, \"\n              f\"avg {stat['avg_word_count']:.0f} words\")\n    \n    print(\"\\n4. COMPLEX FILTERING EXAMPLE:\")\n    complex_filters = {\n        'start_date': date(2023, 1, 1),\n        'end_date': date(2023, 12, 31),\n        'court_levels': [1, 2],\n        'search_text': 'human rights',\n        'legal_areas': ['Constitutional Law', 'Human Rights'],\n        'min_area_relevance': 0.7,\n        'min_citations': 5,\n        'min_words': 5000,\n        'min_relevance': 0.8,\n        'order_by': ['-relevance_score', '-judgment_date']\n    }\n    \n    complex_query = JudgmentQueryManager.complex_search_query(complex_filters)\n    print(\"   Complex query with multiple filters:\")\n    print(\"     - Date range: 2023\")\n    print(\"     - Court levels: Supreme Court and Court of Appeal\")\n    print(\"     - Text search: 'human rights'\")\n    print(\"     - Legal areas with relevance >= 0.7\")\n    print(\"     - Minimum 5 citations received\")\n    print(\"     - Minimum 5000 words\")\n    print(\"     - Relevance score >= 0.8\")\n    \n    print(\"\\n5. QUERY OPTIMIZATION TECHNIQUES:\")\n    print(\"   - select_related(): Joins for foreign keys (1-to-1, many-to-1)\")\n    print(\"   - prefetch_related(): Separate queries for many-to-many\")\n    print(\"   - Database indexes on frequently filtered fields\")\n    print(\"   - Bulk operations for large data modifications\")\n    print(\"   - Database-level aggregations vs Python loops\")\n    print(\"   - Query result caching for expensive operations\")\n    \n    print(\"\\n6. PERFORMANCE CONSIDERATIONS:\")\n    print(\"   - Use EXPLAIN ANALYZE to understand query execution\")\n    print(\"   - Monitor slow query logs\")\n    print(\"   - Consider database connection pooling\")\n    print(\"   - Implement query result caching (Redis/Memcached)\")\n    print(\"   - Use database partitioning for very large tables\")\n    print(\"   - Consider read replicas for read-heavy workloads\")\n    \n    print(\"\\n7. DJANGO-SPECIFIC OPTIMIZATIONS:\")\n    print(\"   - Use only() and defer() to limit loaded fields\")\n    print(\"   - Implement QuerySet.iterator() for large datasets\")\n    print(\"   - Use select_for_update() for concurrent modifications\")\n    print(\"   - Leverage Django's built-in pagination\")\n    print(\"   - Use database functions (Extract, Concat, etc.)\")\n    \n    return {\n        'recent_query': recent_query,\n        'search_query': search_query,\n        'court_stats': mock_court_stats,\n        'complex_filters': complex_filters,\n        'optimization_examples': QueryOptimizationExamples\n    }\n\n\nif __name__ == \"__main__\":\n    # Note: This would require a Django environment with the models defined\n    # For demonstration purposes, we show the query structure\n    demonstrate_django_queries()