"""
DataFrame Data Structure for FCL Legal Data Analysis

This implementation provides tabular data analysis capabilities for legal case data,
optimized for UK legal database operations and statistical analysis.

Use cases:
- Statistical analysis of legal judgments
- Citation pattern analysis
- Court performance metrics
- Legal trend analysis over time
- Comparative case law studies
"""

import pandas as pd
import numpy as np
from datetime import datetime, date
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any, Optional, Tuple

class LegalDataFrameAnalyzer:
    """
    DataFrame wrapper specialized for legal case data analysis.
    Provides FCL-specific analytical methods for UK legal databases.
    """

    def __init__(self, data: List[Dict[str, Any]] = None):
        """
        Initialize legal data analyzer.

        Args:
            data (List[Dict]): List of legal case dictionaries
        """
        if data:
            self.df = pd.DataFrame(data)
            self._preprocess_legal_data()
        else:
            self.df = pd.DataFrame()

    def _preprocess_legal_data(self):
        """Preprocess legal data for analysis."""
        # Convert date columns to datetime
        if 'judgment_date' in self.df.columns:
            self.df['judgment_date'] = pd.to_datetime(self.df['judgment_date'])

        # Extract year from judgment date
        if 'judgment_date' in self.df.columns:
            self.df['year'] = self.df['judgment_date'].dt.year
            self.df['month'] = self.df['judgment_date'].dt.month
            self.df['decade'] = (self.df['year'] // 10) * 10

        # Create citation type column
        if 'neutral_citation' in self.df.columns:
            self.df['citation_type'] = self.df['neutral_citation'].apply(self._extract_citation_type)

        # Calculate case name length (proxy for case complexity)
        if 'case_name' in self.df.columns:
            self.df['case_name_length'] = self.df['case_name'].str.len()

        # Count number of judges
        if 'judges' in self.df.columns:
            self.df['judge_count'] = self.df['judges'].apply(
                lambda x: len(x) if isinstance(x, list) else 0
            )

    def _extract_citation_type(self, citation: str) -> str:
        """Extract court type from neutral citation."""
        if pd.isna(citation) or not citation:
            return "Unknown"

        citation_upper = citation.upper()
        if "UKSC" in citation_upper:
            return "Supreme Court"
        elif "UKHL" in citation_upper:
            return "House of Lords"
        elif "EWCA" in citation_upper:
            return "Court of Appeal"
        elif "EWHC" in citation_upper:
            return "High Court"
        elif "UKUT" in citation_upper:
            return "Upper Tribunal"
        elif "EWCOP" in citation_upper:
            return "Court of Protection"
        else:
            return "Other"

    def add_cases(self, cases: List[Dict[str, Any]]):
        """Add new cases to the analyzer."""
        new_df = pd.DataFrame(cases)
        if not new_df.empty:
            self.df = pd.concat([self.df, new_df], ignore_index=True)
            self._preprocess_legal_data()

    def get_basic_statistics(self) -> Dict[str, Any]:
        """Get basic statistics about the legal dataset."""
        if self.df.empty:
            return {"error": "No data available"}

        stats = {
            "total_cases": len(self.df),
            "date_range": {
                "earliest": self.df['judgment_date'].min().strftime('%Y-%m-%d') if 'judgment_date' in self.df.columns else None,
                "latest": self.df['judgment_date'].max().strftime('%Y-%m-%d') if 'judgment_date' in self.df.columns else None
            },
            "court_distribution": self.df['court'].value_counts().to_dict() if 'court' in self.df.columns else {},
            "cases_by_decade": self.df['decade'].value_counts().sort_index().to_dict() if 'decade' in self.df.columns else {},
            "average_judges_per_case": self.df['judge_count'].mean() if 'judge_count' in self.df.columns else 0
        }

        return stats

    def analyze_temporal_trends(self) -> Dict[str, Any]:
        """Analyze temporal trends in legal judgments."""
        if 'year' not in self.df.columns:
            return {"error": "No temporal data available"}

        # Cases per year
        yearly_counts = self.df['year'].value_counts().sort_index()

        # Cases per month (seasonal patterns)
        monthly_counts = self.df['month'].value_counts().sort_index()

        # Court activity over time
        court_trends = {}
        if 'court' in self.df.columns:
            for court in self.df['court'].unique():
                court_data = self.df[self.df['court'] == court]
                court_trends[court] = court_data['year'].value_counts().sort_index().to_dict()

        return {
            "yearly_distribution": yearly_counts.to_dict(),
            "monthly_distribution": monthly_counts.to_dict(),
            "court_trends_by_year": court_trends,
            "busiest_year": yearly_counts.idxmax() if not yearly_counts.empty else None,
            "quietest_year": yearly_counts.idxmin() if not yearly_counts.empty else None
        }

    def analyze_citation_patterns(self) -> Dict[str, Any]:
        """Analyze citation patterns and court hierarchies."""
        if 'citation_type' not in self.df.columns:
            return {"error": "No citation data available"}

        citation_analysis = {
            "citation_type_distribution": self.df['citation_type'].value_counts().to_dict(),
            "citation_trends_by_year": {}
        }

        # Citation trends over time
        if 'year' in self.df.columns:
            for citation_type in self.df['citation_type'].unique():
                type_data = self.df[self.df['citation_type'] == citation_type]
                citation_analysis["citation_trends_by_year"][citation_type] = (
                    type_data['year'].value_counts().sort_index().to_dict()
                )

        return citation_analysis

    def find_judicial_patterns(self) -> Dict[str, Any]:
        """Analyze patterns in judicial participation."""
        if 'judges' not in self.df.columns:
            return {"error": "No judicial data available"}

        # Flatten judges list and count appearances
        all_judges = []
        for judges_list in self.df['judges']:
            if isinstance(judges_list, list):
                all_judges.extend(judges_list)

        judge_counter = Counter(all_judges)

        # Panel size analysis
        panel_analysis = {
            "most_active_judges": dict(judge_counter.most_common(10)),
            "average_panel_size": self.df['judge_count'].mean() if 'judge_count' in self.df.columns else 0,
            "panel_size_distribution": self.df['judge_count'].value_counts().to_dict() if 'judge_count' in self.df.columns else {}
        }

        return panel_analysis

    def compare_courts(self) -> Dict[str, Any]:
        """Compare different courts' characteristics."""
        if 'court' not in self.df.columns:
            return {"error": "No court data available"}

        court_comparison = {}

        for court in self.df['court'].unique():
            court_data = self.df[self.df['court'] == court]

            court_stats = {
                "total_cases": len(court_data),
                "date_range": {
                    "earliest": court_data['judgment_date'].min().strftime('%Y-%m-%d') if 'judgment_date' in court_data.columns else None,
                    "latest": court_data['judgment_date'].max().strftime('%Y-%m-%d') if 'judgment_date' in court_data.columns else None
                },
                "average_panel_size": court_data['judge_count'].mean() if 'judge_count' in court_data.columns else 0,
                "case_complexity_proxy": court_data['case_name_length'].mean() if 'case_name_length' in court_data.columns else 0
            }

            court_comparison[court] = court_stats

        return court_comparison

    def search_cases(self, **kwargs) -> pd.DataFrame:
        """
        Search cases based on multiple criteria.

        Args:
            **kwargs: Search criteria (court, year_from, year_to, case_name_contains, etc.)

        Returns:
            pd.DataFrame: Filtered cases
        """
        filtered_df = self.df.copy()

        if 'court' in kwargs:
            filtered_df = filtered_df[filtered_df['court'] == kwargs['court']]

        if 'year_from' in kwargs and 'year' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['year'] >= kwargs['year_from']]

        if 'year_to' in kwargs and 'year' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['year'] <= kwargs['year_to']]

        if 'case_name_contains' in kwargs and 'case_name' in filtered_df.columns:
            filtered_df = filtered_df[
                filtered_df['case_name'].str.contains(kwargs['case_name_contains'], case=False, na=False)
            ]

        if 'judge_name' in kwargs and 'judges' in filtered_df.columns:
            judge_mask = filtered_df['judges'].apply(
                lambda judges: kwargs['judge_name'] in judges if isinstance(judges, list) else False
            )
            filtered_df = filtered_df[judge_mask]

        return filtered_df

    def export_analysis_report(self, filename: str = "fcl_legal_analysis.html"):
        """Export comprehensive analysis report."""
        if self.df.empty:
            print("No data to analyze")
            return

        # Create comprehensive analysis
        basic_stats = self.get_basic_statistics()
        temporal_trends = self.analyze_temporal_trends()
        citation_patterns = self.analyze_citation_patterns()
        judicial_patterns = self.find_judicial_patterns()
        court_comparison = self.compare_courts()

        # Generate HTML report
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>FCL Legal Database Analysis Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                h1, h2 {{ color: #2c3e50; }}
                .stat-section {{ margin: 20px 0; padding: 15px; border-left: 4px solid #3498db; }}
                .court {{ background-color: #f8f9fa; padding: 10px; margin: 10px 0; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <h1>FCL Legal Database Analysis Report</h1>
            <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>

            <div class="stat-section">
                <h2>Basic Statistics</h2>
                <p><strong>Total Cases:</strong> {basic_stats['total_cases']}</p>
                <p><strong>Date Range:</strong> {basic_stats['date_range']['earliest']} to {basic_stats['date_range']['latest']}</p>
                <p><strong>Average Judges per Case:</strong> {basic_stats['average_judges_per_case']:.2f}</p>
            </div>

            <div class="stat-section">
                <h2>Court Distribution</h2>
                <table>
                    <tr><th>Court</th><th>Number of Cases</th></tr>
        """

        for court, count in sorted(basic_stats['court_distribution'].items()):
            html_content += f"<tr><td>{court}</td><td>{count}</td></tr>"

        html_content += """
                </table>
            </div>
        </body>
        </html>
        """

        with open(filename, 'w', encoding='utf-8') as f:
            f.write(html_content)

        print(f"Analysis report exported to {filename}")


def demo_fcl_legal_dataframes():
    """Demonstration of DataFrame analysis with realistic FCL legal data."""

    # Sample UK legal cases data
    sample_legal_data = [
        {
            "case_name": "Donoghue v Stevenson",
            "citation": "[1932] AC 562",
            "neutral_citation": "[1932] UKHL 100",
            "judgment_date": "1932-05-26",
            "court": "House of Lords",
            "judges": ["Lord Atkin", "Lord Thankerton", "Lord Macmillan"],
            "legal_area": "Tort Law",
            "significance": "Established modern negligence law"
        },
        {
            "case_name": "Carlill v Carbolic Smoke Ball Company",
            "citation": "[1893] 1 QB 256",
            "neutral_citation": "[1892] EWCA Civ 1",
            "judgment_date": "1892-12-07",
            "court": "Court of Appeal",
            "judges": ["Lindley LJ", "Bowen LJ", "A.L. Smith LJ"],
            "legal_area": "Contract Law",
            "significance": "Unilateral contract formation"
        },
        {
            "case_name": "Rylands v Fletcher",
            "citation": "(1868) LR 3 HL 330",
            "neutral_citation": "[1868] UKHL 1",
            "judgment_date": "1868-07-17",
            "court": "House of Lords",
            "judges": ["Lord Cairns", "Lord Cranworth"],
            "legal_area": "Tort Law",
            "significance": "Strict liability for dangerous activities"
        },
        {
            "case_name": "R v Brown",
            "citation": "[1994] 1 AC 212",
            "neutral_citation": "[1993] UKHL 19",
            "judgment_date": "1993-03-11",
            "court": "House of Lords",
            "judges": ["Lord Templeman", "Lord Jauncey", "Lord Lowry"],
            "legal_area": "Criminal Law",
            "significance": "Consent in assault cases"
        },
        {
            "case_name": "Pepper v Hart",
            "citation": "[1993] AC 593",
            "neutral_citation": "[1992] UKHL 3",
            "judgment_date": "1992-11-26",
            "court": "House of Lords",
            "judges": ["Lord Griffiths", "Lord Ackner", "Lord Oliver"],
            "legal_area": "Constitutional Law",
            "significance": "Use of Hansard in statutory interpretation"
        },
        {
            "case_name": "Caparo Industries plc v Dickman",
            "citation": "[1990] 2 AC 605",
            "neutral_citation": "[1990] UKHL 2",
            "judgment_date": "1990-02-08",
            "court": "House of Lords",
            "judges": ["Lord Bridge", "Lord Roskill", "Lord Ackner"],
            "legal_area": "Tort Law",
            "significance": "Three-stage test for duty of care"
        },
        {
            "case_name": "Miller v Jackson",
            "citation": "[1977] QB 966",
            "neutral_citation": "[1977] EWCA Civ 6",
            "judgment_date": "1977-05-03",
            "court": "Court of Appeal",
            "judges": ["Lord Denning MR", "Geoffrey Lane LJ", "Cumming-Bruce LJ"],
            "legal_area": "Tort Law",
            "significance": "Nuisance and public benefit"
        },
        {
            "case_name": "R (Miller) v The Prime Minister",
            "citation": "[2019] UKSC 41",
            "neutral_citation": "[2019] UKSC 41",
            "judgment_date": "2019-09-24",
            "court": "Supreme Court",
            "judges": ["Lady Hale", "Lord Reed", "Lord Kerr", "Lord Wilson", "Lord Carnwath"],
            "legal_area": "Constitutional Law",
            "significance": "Limits on executive power"
        },
        {
            "case_name": "Test Claimants in the FII Group Litigation v Revenue and Customs",
            "citation": "[2020] UKSC 47",
            "neutral_citation": "[2020] UKSC 47",
            "judgment_date": "2020-11-04",
            "court": "Supreme Court",
            "judges": ["Lord Reed", "Lord Hodge", "Lord Lloyd-Jones"],
            "legal_area": "Tax Law",
            "significance": "EU law and restitution"
        },
        {
            "case_name": "Cambridge Water Co Ltd v Eastern Counties Leather plc",
            "citation": "[1994] 2 AC 264",
            "neutral_citation": "[1994] UKHL 5",
            "judgment_date": "1994-01-09",
            "court": "House of Lords",
            "judges": ["Lord Goff", "Lord Jauncey", "Lord Lowry"],
            "legal_area": "Tort Law",
            "significance": "Environmental liability and foreseeability"
        }
    ]

    print("=== FCL Legal DataFrame Analysis Demo ===\n")

    # Initialize analyzer with sample data
    analyzer = LegalDataFrameAnalyzer(sample_legal_data)

    print("Loaded legal case database with realistic UK cases\n")

    # Basic statistics
    print("=== Basic Database Statistics ===")
    basic_stats = analyzer.get_basic_statistics()
    print(f"Total cases: {basic_stats['total_cases']}")
    print(f"Date range: {basic_stats['date_range']['earliest']} to {basic_stats['date_range']['latest']}")
    print(f"Average judges per case: {basic_stats['average_judges_per_case']:.2f}")

    print(f"\nCases by decade:")
    for decade, count in sorted(basic_stats['cases_by_decade'].items()):
        print(f"  {decade}s: {count} cases")

    print(f"\nCourt distribution:")
    for court, count in sorted(basic_stats['court_distribution'].items()):
        print(f"  {court}: {count} cases")

    # Temporal analysis
    print(f"\n=== Temporal Trends Analysis ===")
    temporal_analysis = analyzer.analyze_temporal_trends()
    print(f"Busiest year: {temporal_analysis['busiest_year']}")
    print(f"Quietest year: {temporal_analysis['quietest_year']}")

    print(f"\nMonthly distribution (seasonal patterns):")
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    for month, count in sorted(temporal_analysis['monthly_distribution'].items()):
        print(f"  {month_names[month-1]}: {count} cases")

    # Citation pattern analysis
    print(f"\n=== Citation Pattern Analysis ===")
    citation_analysis = analyzer.analyze_citation_patterns()
    print(f"Citation type distribution:")
    for ctype, count in sorted(citation_analysis['citation_type_distribution'].items()):
        print(f"  {ctype}: {count} cases")

    # Judicial patterns
    print(f"\n=== Judicial Participation Analysis ===")
    judicial_analysis = analyzer.find_judicial_patterns()
    print(f"Average panel size: {judicial_analysis['average_panel_size']:.2f} judges")

    print(f"\nMost active judges (top 5):")
    for judge, count in list(judicial_analysis['most_active_judges'].items())[:5]:
        print(f"  {judge}: {count} cases")

    print(f"\nPanel size distribution:")
    for size, count in sorted(judicial_analysis['panel_size_distribution'].items()):
        judge_word = "judge" if size == 1 else "judges"
        print(f"  {size} {judge_word}: {count} cases")

    # Court comparison
    print(f"\n=== Court Comparison Analysis ===")
    court_comparison = analyzer.compare_courts()
    for court, stats in court_comparison.items():
        print(f"\n{court}:")
        print(f"  Total cases: {stats['total_cases']}")
        print(f"  Date range: {stats['date_range']['earliest']} to {stats['date_range']['latest']}")
        print(f"  Average panel size: {stats['average_panel_size']:.2f}")
        print(f"  Case complexity proxy: {stats['case_complexity_proxy']:.1f} chars")

    # Search demonstrations
    print(f"\n=== Search and Filter Examples ===")

    # Search by court
    house_of_lords_cases = analyzer.search_cases(court="House of Lords")
    print(f"\nHouse of Lords cases: {len(house_of_lords_cases)}")
    for _, case in house_of_lords_cases.head(3).iterrows():
        print(f"  - {case['case_name']} ({case['judgment_date'].strftime('%Y')})")

    # Search by date range
    modern_cases = analyzer.search_cases(year_from=1990)
    print(f"\nCases from 1990 onwards: {len(modern_cases)}")
    for _, case in modern_cases.head(3).iterrows():
        print(f"  - {case['case_name']} ({case['judgment_date'].strftime('%Y')})")

    # Search by case name content
    tort_cases = analyzer.search_cases(case_name_contains="v")
    print(f"\nCases with 'v' in name: {len(tort_cases)}")

    # Search by judge
    lord_ackner_cases = analyzer.search_cases(judge_name="Lord Ackner")
    print(f"\nCases involving Lord Ackner: {len(lord_ackner_cases)}")
    for _, case in lord_ackner_cases.iterrows():
        print(f"  - {case['case_name']} ({case['judgment_date'].strftime('%Y')})")

    print(f"\n=== DataFrame Operations Demo ===")
    print(f"Raw DataFrame shape: {analyzer.df.shape}")
    print(f"Columns: {list(analyzer.df.columns)}")
    print(f"\nFirst few rows:")
    print(analyzer.df[['case_name', 'court', 'year', 'judge_count']].head())


if __name__ == "__main__":
    demo_fcl_legal_dataframes()