"""
NumPy Arrays for Legal Data Numerical Analysis
==============================================

This module demonstrates numerical analysis using NumPy arrays for legal data
processing in Find Case Law (FCL). NumPy provides efficient operations for
statistical analysis, similarity scoring, and data aggregation.

Key FCL Use Cases:
- Case similarity scoring using vector embeddings
- Statistical analysis of judgment trends
- Court performance metrics calculation
- Citation network analysis
- Search relevance scoring
- Document clustering and classification
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt


@dataclass
class CaseSimilarity:
    """Represents similarity between two legal cases"""
    case1_citation: str
    case2_citation: str
    similarity_score: float
    matching_features: List[str]
    vector_distance: float


@dataclass
class CourtStatistics:
    """Statistical data for a court"""
    court_name: str
    total_cases: int
    avg_case_length: float
    citation_frequency: float
    subject_areas: Dict[str, int]
    temporal_trends: np.ndarray


class LegalDataAnalyzer:
    """
    Numerical analysis tools for legal document processing using NumPy.

    Features:
    - Vector-based similarity calculation
    - Statistical trend analysis
    - Performance metrics computation
    - Citation network analysis
    """

    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        self.case_embeddings = {}
        self.court_stats = {}

    def calculate_case_similarities(self, cases: List[Dict[str, Any]]) -> np.ndarray:
        """
        Calculate pairwise similarities between legal cases using TF-IDF vectors.

        Args:
            cases: List of case dictionaries with 'text' and 'citation' keys

        Returns:
            NumPy array of similarity scores (n_cases x n_cases)

        Example usage for FCL:
        - Find similar judgments for legal research
        - Identify precedent relationships
        - Cluster cases by legal topic
        """
        if len(cases) < 2:
            return np.array([])

        # Extract text content for vectorization
        case_texts = [case.get('text', '') for case in cases]
        citations = [case.get('citation', f'Case_{i}') for i, case in enumerate(cases)]

        # Create TF-IDF vectors
        tfidf_matrix = self.vectorizer.fit_transform(case_texts)

        # Calculate cosine similarity matrix
        similarity_matrix = cosine_similarity(tfidf_matrix)

        # Store embeddings for later use
        for i, citation in enumerate(citations):
            self.case_embeddings[citation] = tfidf_matrix[i].toarray().flatten()

        print(f"Calculated similarities for {len(cases)} cases")
        print(f"Similarity matrix shape: {similarity_matrix.shape}")
        print(f"Average similarity: {np.mean(similarity_matrix[similarity_matrix != 1.0]):.3f}")

        return similarity_matrix

    def find_most_similar_cases(self, cases: List[Dict[str, Any]],
                               target_case_idx: int,
                               n_similar: int = 5) -> List[CaseSimilarity]:
        """
        Find the most similar cases to a target case.

        Args:
            cases: List of case dictionaries
            target_case_idx: Index of target case
            n_similar: Number of similar cases to return

        Returns:
            List of CaseSimilarity objects
        """
        similarity_matrix = self.calculate_case_similarities(cases)

        if similarity_matrix.size == 0 or target_case_idx >= len(cases):
            return []

        target_citation = cases[target_case_idx].get('citation', f'Case_{target_case_idx}')
        similarities = similarity_matrix[target_case_idx]

        # Get indices of most similar cases (excluding self)
        similar_indices = np.argsort(similarities)[::-1][1:n_similar+1]

        similar_cases = []
        for idx in similar_indices:
            if idx < len(cases):
                case_citation = cases[idx].get('citation', f'Case_{idx}')
                similarity_score = similarities[idx]

                # Identify matching features (simplified)
                matching_features = self._identify_matching_features(
                    cases[target_case_idx], cases[idx]
                )

                # Calculate vector distance
                vector_distance = np.linalg.norm(
                    self.case_embeddings.get(target_citation, np.array([])) -
                    self.case_embeddings.get(case_citation, np.array([]))
                )

                similar_cases.append(CaseSimilarity(
                    case1_citation=target_citation,
                    case2_citation=case_citation,
                    similarity_score=similarity_score,
                    matching_features=matching_features,
                    vector_distance=vector_distance
                ))

        return similar_cases

    def analyze_court_performance(self, cases_by_court: Dict[str, List[Dict[str, Any]]]) -> Dict[str, CourtStatistics]:
        """
        Perform statistical analysis of court performance metrics.

        Args:
            cases_by_court: Dictionary mapping court names to lists of cases

        Returns:
            Dictionary of court statistics

        FCL Use Cases:
        - Compare court productivity
        - Analyze case complexity trends
        - Identify specialization patterns
        """
        court_stats = {}

        for court_name, cases in cases_by_court.items():
            if not cases:
                continue

            # Calculate basic statistics
            case_lengths = np.array([len(case.get('text', '')) for case in cases])
            citation_counts = np.array([len(case.get('citations', [])) for case in cases])

            # Temporal analysis
            dates = []
            for case in cases:
                date_str = case.get('date', '2023-01-01')
                try:
                    dates.append(datetime.fromisoformat(date_str.replace('Z', '+00:00')))
                except:
                    dates.append(datetime(2023, 1, 1))

            # Create temporal trend (cases per month)
            if dates:
                date_array = np.array(dates)
                min_date = min(dates)
                max_date = max(dates)
                months = pd.date_range(min_date, max_date, freq='M')
                trend = np.zeros(len(months))

                for i, month in enumerate(months):
                    month_cases = sum(1 for d in dates if d.year == month.year and d.month == month.month)
                    trend[i] = month_cases
            else:
                trend = np.array([0])

            # Subject area analysis
            subject_areas = {}
            for case in cases:
                subjects = case.get('subject_matter', [])
                for subject in subjects:
                    subject_areas[subject] = subject_areas.get(subject, 0) + 1

            court_stats[court_name] = CourtStatistics(
                court_name=court_name,
                total_cases=len(cases),
                avg_case_length=float(np.mean(case_lengths)) if len(case_lengths) > 0 else 0.0,
                citation_frequency=float(np.mean(citation_counts)) if len(citation_counts) > 0 else 0.0,
                subject_areas=subject_areas,
                temporal_trends=trend
            )

        self.court_stats = court_stats
        return court_stats

    def calculate_citation_network_metrics(self, cases: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze citation networks between cases using graph metrics.

        Args:
            cases: List of cases with citation information

        Returns:
            Dictionary of network metrics

        FCL Applications:
        - Identify influential precedents
        - Find citation clusters
        - Measure case authority
        """
        # Build citation adjacency matrix
        n_cases = len(cases)
        citation_matrix = np.zeros((n_cases, n_cases))

        case_citations = {case.get('citation', f'Case_{i}'): i for i, case in enumerate(cases)}

        # Fill adjacency matrix
        for i, case in enumerate(cases):
            cited_cases = case.get('citations', [])
            for cited in cited_cases:
                if cited in case_citations:
                    j = case_citations[cited]
                    citation_matrix[i][j] = 1

        # Calculate network metrics
        in_degree = np.sum(citation_matrix, axis=0)  # How many times each case is cited
        out_degree = np.sum(citation_matrix, axis=1)  # How many cases each case cites

        # Authority score (simplified PageRank)
        authority_scores = self._calculate_authority_scores(citation_matrix)

        # Clustering coefficient
        clustering_coeff = self._calculate_clustering_coefficient(citation_matrix)

        # Network density
        possible_edges = n_cases * (n_cases - 1)
        actual_edges = np.sum(citation_matrix)
        density = actual_edges / possible_edges if possible_edges > 0 else 0

        return {
            'citation_matrix': citation_matrix,
            'in_degree_centrality': in_degree,
            'out_degree_centrality': out_degree,
            'authority_scores': authority_scores,
            'clustering_coefficient': clustering_coeff,
            'network_density': density,
            'most_cited_cases': self._get_most_cited_cases(cases, in_degree),
            'most_citing_cases': self._get_most_citing_cases(cases, out_degree)
        }

    def perform_trend_analysis(self, temporal_data: Dict[str, List[Tuple[datetime, float]]]) -> Dict[str, Any]:
        """
        Perform time series analysis on legal data trends.

        Args:
            temporal_data: Dictionary mapping metric names to (date, value) tuples

        Returns:
            Dictionary of trend analysis results

        FCL Use Cases:
        - Track judgment volume trends
        - Analyze seasonal patterns
        - Forecast case loads
        """
        trend_results = {}

        for metric_name, data_points in temporal_data.items():
            if not data_points:
                continue

            # Sort by date
            sorted_data = sorted(data_points, key=lambda x: x[0])
            dates = np.array([point[0] for point in sorted_data])
            values = np.array([point[1] for point in sorted_data])

            # Convert dates to numeric values for analysis
            date_nums = np.array([(d - dates[0]).days for d in dates])

            # Linear trend analysis
            if len(values) > 1:
                slope, intercept = np.polyfit(date_nums, values, 1)
                trend_line = slope * date_nums + intercept

                # Calculate trend metrics
                trend_strength = np.corrcoef(date_nums, values)[0, 1] if len(values) > 1 else 0
                volatility = np.std(values) if len(values) > 1 else 0
                growth_rate = slope / np.mean(values) if np.mean(values) != 0 else 0

                # Seasonal analysis (if enough data points)
                seasonal_pattern = self._detect_seasonal_pattern(dates, values)

                trend_results[metric_name] = {
                    'slope': slope,
                    'intercept': intercept,
                    'trend_line': trend_line,
                    'trend_strength': trend_strength,
                    'volatility': volatility,
                    'growth_rate': growth_rate,
                    'seasonal_pattern': seasonal_pattern,
                    'data_points': len(values),
                    'date_range': (dates[0], dates[-1])
                }

        return trend_results

    def calculate_search_relevance_scores(self, query_terms: List[str],
                                        documents: List[Dict[str, Any]]) -> np.ndarray:
        """
        Calculate relevance scores for search results using numerical methods.

        Args:
            query_terms: List of search terms
            documents: List of document dictionaries

        Returns:
            NumPy array of relevance scores

        FCL Applications:
        - Rank search results
        - Personalized recommendations
        - Query expansion
        """
        if not documents:
            return np.array([])

        # Create query vector
        query_text = ' '.join(query_terms)
        all_texts = [query_text] + [doc.get('text', '') for doc in documents]

        # Vectorize all texts
        tfidf_matrix = self.vectorizer.fit_transform(all_texts)
        query_vector = tfidf_matrix[0]
        doc_vectors = tfidf_matrix[1:]

        # Calculate cosine similarity with query
        similarities = cosine_similarity(query_vector, doc_vectors).flatten()

        # Boost scores based on additional factors
        boosted_scores = np.copy(similarities)

        for i, doc in enumerate(documents):
            # Court authority boost
            court = doc.get('court', '').lower()
            if 'supreme' in court:
                boosted_scores[i] *= 1.5
            elif 'appeal' in court:
                boosted_scores[i] *= 1.3
            elif 'high' in court:
                boosted_scores[i] *= 1.2

            # Recency boost
            date_str = doc.get('date', '2020-01-01')
            try:
                doc_date = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                days_old = (datetime.now() - doc_date).days
                recency_factor = np.exp(-days_old / 365.0)  # Exponential decay
                boosted_scores[i] *= (1.0 + 0.2 * recency_factor)
            except:
                pass

            # Citation count boost
            citation_count = len(doc.get('citations', []))
            citation_boost = 1.0 + 0.1 * np.log(1 + citation_count)
            boosted_scores[i] *= citation_boost

        return boosted_scores

    def _identify_matching_features(self, case1: Dict[str, Any], case2: Dict[str, Any]) -> List[str]:
        """Identify common features between two cases"""
        features = []

        # Same court
        if case1.get('court') == case2.get('court'):
            features.append('same_court')

        # Similar subject matter
        subjects1 = set(case1.get('subject_matter', []))
        subjects2 = set(case2.get('subject_matter', []))
        if subjects1 & subjects2:
            features.append('shared_subjects')

        # Same year
        try:
            date1 = datetime.fromisoformat(case1.get('date', '2020-01-01'))
            date2 = datetime.fromisoformat(case2.get('date', '2020-01-01'))
            if date1.year == date2.year:
                features.append('same_year')
        except:
            pass

        return features

    def _calculate_authority_scores(self, citation_matrix: np.ndarray,
                                  iterations: int = 50, damping: float = 0.85) -> np.ndarray:
        """Calculate authority scores using PageRank algorithm"""
        n = citation_matrix.shape[0]
        if n == 0:
            return np.array([])

        # Initialize scores
        scores = np.ones(n) / n

        # Normalize citation matrix
        out_degree = np.sum(citation_matrix, axis=1)
        transition_matrix = np.zeros_like(citation_matrix)

        for i in range(n):
            if out_degree[i] > 0:
                transition_matrix[i] = citation_matrix[i] / out_degree[i]
            else:
                transition_matrix[i] = np.ones(n) / n

        # PageRank iterations
        for _ in range(iterations):
            new_scores = (1 - damping) / n + damping * transition_matrix.T @ scores
            scores = new_scores

        return scores

    def _calculate_clustering_coefficient(self, adj_matrix: np.ndarray) -> float:
        """Calculate clustering coefficient for the citation network"""
        n = adj_matrix.shape[0]
        if n < 3:
            return 0.0

        clustering_sum = 0
        valid_nodes = 0

        for i in range(n):
            neighbors = np.where(adj_matrix[i] == 1)[0]
            if len(neighbors) < 2:
                continue

            # Count triangles
            triangles = 0
            possible_triangles = len(neighbors) * (len(neighbors) - 1) // 2

            for j in range(len(neighbors)):
                for k in range(j + 1, len(neighbors)):
                    if adj_matrix[neighbors[j]][neighbors[k]] == 1:
                        triangles += 1

            if possible_triangles > 0:
                clustering_sum += triangles / possible_triangles
                valid_nodes += 1

        return clustering_sum / valid_nodes if valid_nodes > 0 else 0.0

    def _get_most_cited_cases(self, cases: List[Dict[str, Any]],
                            in_degrees: np.ndarray, n: int = 5) -> List[Dict[str, Any]]:
        """Get the most cited cases"""
        sorted_indices = np.argsort(in_degrees)[::-1][:n]
        return [{'case': cases[i], 'citations': int(in_degrees[i])}
                for i in sorted_indices if i < len(cases)]

    def _get_most_citing_cases(self, cases: List[Dict[str, Any]],
                             out_degrees: np.ndarray, n: int = 5) -> List[Dict[str, Any]]:
        """Get cases that cite the most other cases"""
        sorted_indices = np.argsort(out_degrees)[::-1][:n]
        return [{'case': cases[i], 'citing_count': int(out_degrees[i])}
                for i in sorted_indices if i < len(cases)]

    def _detect_seasonal_pattern(self, dates: np.ndarray, values: np.ndarray) -> Dict[str, float]:
        """Detect seasonal patterns in time series data"""
        if len(dates) < 12:
            return {}

        # Group by month
        monthly_values = {}
        for date, value in zip(dates, values):
            month = date.month
            if month not in monthly_values:
                monthly_values[month] = []
            monthly_values[month].append(value)

        # Calculate monthly averages
        monthly_averages = {}
        for month, month_values in monthly_values.items():
            monthly_averages[month] = np.mean(month_values)

        return monthly_averages


def demonstrate_legal_data_analysis():
    """Demonstrate numerical analysis with sample UK legal data"""

    print("=== Legal Data Numerical Analysis Demo ===\n")

    # Sample UK legal cases data
    sample_cases = [
        {
            'citation': '[2023] UKSC 15',
            'court': 'Supreme Court',
            'date': '2023-05-15',
            'text': 'This case concerns constitutional law and the relationship between Parliament and the Executive. The principles of parliamentary sovereignty are fundamental to our constitutional framework. We must consider the precedents established in previous constitutional cases.',
            'subject_matter': ['Constitutional Law', 'Parliamentary Sovereignty'],
            'citations': ['[2019] UKSC 41', '[2017] UKSC 5']
        },
        {
            'citation': '[2023] EWCA Civ 892',
            'court': 'Court of Appeal',
            'date': '2023-08-22',
            'text': 'The appellant challenges the decision on grounds of procedural fairness and natural justice. Administrative law principles require that decisions are made fairly and with proper consideration of all relevant factors. The Wednesbury principles apply to this review.',
            'subject_matter': ['Administrative Law', 'Judicial Review'],
            'citations': ['[2019] UKSC 41', '[2018] EWCA Civ 234']
        },
        {
            'citation': '[2023] EWHC 1456 (Admin)',
            'court': 'High Court',
            'date': '2023-06-30',
            'text': 'This judicial review concerns the lawfulness of the defendant\'s decision-making process. Administrative law requires compliance with statutory procedures and consideration of relevant factors. The principles established in constitutional cases provide guidance.',
            'subject_matter': ['Administrative Law', 'Statutory Interpretation'],
            'citations': ['[2023] UKSC 15', '[2018] EWCA Civ 234']
        },
        {
            'citation': '[2023] UKSC 28',
            'court': 'Supreme Court',
            'date': '2023-09-10',
            'text': 'The constitutional principles governing the exercise of prerogative powers are well-established. Parliamentary sovereignty remains the cornerstone of our constitutional system. This case builds upon previous Supreme Court decisions regarding Executive power.',
            'subject_matter': ['Constitutional Law', 'Prerogative Powers'],
            'citations': ['[2023] UKSC 15', '[2019] UKSC 41']
        },
        {
            'citation': '[2023] EWCA Crim 567',
            'court': 'Court of Appeal Criminal Division',
            'date': '2023-07-18',
            'text': 'The appellant argues that the conviction is unsafe due to procedural irregularities during the trial. Criminal procedure requires strict adherence to statutory provisions and common law principles. The evidence must be evaluated according to established precedents.',
            'subject_matter': ['Criminal Law', 'Criminal Procedure'],
            'citations': ['[2022] UKSC 32', '[2021] EWCA Crim 123']
        }
    ]

    # Initialize analyzer
    analyzer = LegalDataAnalyzer()

    # 1. Case Similarity Analysis
    print("1. CASE SIMILARITY ANALYSIS:")
    similarity_matrix = analyzer.calculate_case_similarities(sample_cases)
    print(f"   Similarity matrix shape: {similarity_matrix.shape}")

    # Find similar cases to the first case
    similar_cases = analyzer.find_most_similar_cases(sample_cases, 0, 3)
    print(f"   Cases similar to {sample_cases[0]['citation']}:")
    for sim_case in similar_cases:
        print(f"     - {sim_case.case2_citation}: {sim_case.similarity_score:.3f} similarity")
        print(f"       Features: {', '.join(sim_case.matching_features)}")

    # 2. Court Performance Analysis
    print(f"\n2. COURT PERFORMANCE ANALYSIS:")
    cases_by_court = {}
    for case in sample_cases:
        court = case['court']
        if court not in cases_by_court:
            cases_by_court[court] = []
        cases_by_court[court].append(case)

    court_stats = analyzer.analyze_court_performance(cases_by_court)
    for court_name, stats in court_stats.items():
        print(f"   {court_name}:")
        print(f"     Total cases: {stats.total_cases}")
        print(f"     Avg case length: {stats.avg_case_length:.0f} characters")
        print(f"     Avg citations per case: {stats.citation_frequency:.1f}")
        print(f"     Subject areas: {', '.join(stats.subject_areas.keys())}")

    # 3. Citation Network Analysis
    print(f"\n3. CITATION NETWORK ANALYSIS:")
    network_metrics = analyzer.calculate_citation_network_metrics(sample_cases)
    print(f"   Network density: {network_metrics['network_density']:.3f}")
    print(f"   Clustering coefficient: {network_metrics['clustering_coefficient']:.3f}")

    print("   Most cited cases:")
    for case_info in network_metrics['most_cited_cases'][:3]:
        citation = case_info['case'].get('citation', 'Unknown')
        citations = case_info['citations']
        print(f"     - {citation}: {citations} citations")

    print("   Authority scores:")
    authority_scores = network_metrics['authority_scores']
    for i, score in enumerate(authority_scores):
        citation = sample_cases[i]['citation']
        print(f"     - {citation}: {score:.3f}")

    # 4. Search Relevance Scoring
    print(f"\n4. SEARCH RELEVANCE SCORING:")
    query_terms = ['constitutional', 'parliament', 'sovereignty']
    relevance_scores = analyzer.calculate_search_relevance_scores(query_terms, sample_cases)

    print(f"   Query: {' '.join(query_terms)}")
    print("   Relevance scores:")
    for i, score in enumerate(relevance_scores):
        citation = sample_cases[i]['citation']
        court = sample_cases[i]['court']
        print(f"     - {citation} ({court}): {score:.3f}")

    # 5. Trend Analysis
    print(f"\n5. TREND ANALYSIS:")

    # Create sample temporal data
    base_date = datetime(2023, 1, 1)
    temporal_data = {
        'case_volume': [
            (base_date + timedelta(days=30*i), 10 + 5*np.sin(i*0.5) + np.random.normal(0, 2))
            for i in range(12)
        ],
        'avg_case_length': [
            (base_date + timedelta(days=30*i), 5000 + 1000*np.sin(i*0.3) + np.random.normal(0, 500))
            for i in range(12)
        ]
    }

    trend_results = analyzer.perform_trend_analysis(temporal_data)
    for metric_name, results in trend_results.items():
        print(f"   {metric_name}:")
        print(f"     Trend strength: {results['trend_strength']:.3f}")
        print(f"     Growth rate: {results['growth_rate']:.3f}")
        print(f"     Volatility: {results['volatility']:.1f}")

    # 6. Statistical Summary
    print(f"\n6. STATISTICAL SUMMARY:")
    all_similarities = similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)]
    print(f"   Average case similarity: {np.mean(all_similarities):.3f}")
    print(f"   Similarity std deviation: {np.std(all_similarities):.3f}")
    print(f"   Max similarity: {np.max(all_similarities):.3f}")
    print(f"   Min similarity: {np.min(all_similarities):.3f}")

    # Case length statistics
    case_lengths = [len(case['text']) for case in sample_cases]
    print(f"   Average case length: {np.mean(case_lengths):.0f} characters")
    print(f"   Case length std dev: {np.std(case_lengths):.0f} characters")

    # Citation statistics
    citation_counts = [len(case['citations']) for case in sample_cases]
    print(f"   Average citations per case: {np.mean(citation_counts):.1f}")
    print(f"   Total unique citations: {len(set().union(*[case['citations'] for case in sample_cases]))}")

    return {
        'similarity_matrix': similarity_matrix,
        'court_stats': court_stats,
        'network_metrics': network_metrics,
        'relevance_scores': relevance_scores,
        'trend_results': trend_results
    }


if __name__ == "__main__":
    demonstrate_legal_data_analysis()