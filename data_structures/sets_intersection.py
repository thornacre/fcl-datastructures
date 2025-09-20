"""
Set Intersection Data Structure for FCL Document Filtering Operations

This implementation provides efficient set operations for legal document filtering,
optimized for complex queries across UK legal databases.

Use cases:
- Multi-criteria document filtering (by court, legal area, date, etc.)
- Boolean search operations on legal texts
- Tag-based case classification and filtering
- Intersecting multiple search result sets
- Building complex legal database queries
"""

from typing import Set, List, Dict, Any, Optional, Union
from collections import defaultdict
import re
from datetime import datetime, date

class LegalDocument:
    """Represents a legal document with filterable attributes."""

    def __init__(self, doc_id: str, case_name: str, citation: str,
                 court: str, legal_areas: List[str], keywords: List[str],
                 judgment_date: date, neutral_citation: str = "",
                 judges: List[str] = None, content_tags: List[str] = None):
        self.doc_id = doc_id
        self.case_name = case_name
        self.citation = citation
        self.court = court
        self.legal_areas = set(legal_areas) if legal_areas else set()
        self.keywords = set(keyword.lower() for keyword in keywords) if keywords else set()
        self.judgment_date = judgment_date
        self.neutral_citation = neutral_citation
        self.judges = set(judges) if judges else set()
        self.content_tags = set(content_tags) if content_tags else set()

    def __str__(self):
        return f"{self.case_name} ({self.citation})"

    def __repr__(self):
        return self.__str__()

class LegalDocumentFilterEngine:
    """
    Advanced filtering engine using set operations for legal documents.
    Supports complex Boolean queries and multi-attribute filtering.
    """

    def __init__(self):
        self.documents = {}  # doc_id -> LegalDocument
        # Inverted indices for efficient filtering
        self.court_index = defaultdict(set)          # court -> set of doc_ids
        self.legal_area_index = defaultdict(set)     # legal_area -> set of doc_ids
        self.keyword_index = defaultdict(set)        # keyword -> set of doc_ids
        self.judge_index = defaultdict(set)          # judge -> set of doc_ids
        self.tag_index = defaultdict(set)            # tag -> set of doc_ids
        self.year_index = defaultdict(set)           # year -> set of doc_ids

    def add_document(self, document: LegalDocument):
        """Add a legal document to the filtering engine."""
        doc_id = document.doc_id
        self.documents[doc_id] = document

        # Update inverted indices
        self.court_index[document.court].add(doc_id)

        for area in document.legal_areas:
            self.legal_area_index[area].add(doc_id)

        for keyword in document.keywords:
            self.keyword_index[keyword].add(doc_id)

        for judge in document.judges:
            self.judge_index[judge].add(doc_id)

        for tag in document.content_tags:
            self.tag_index[tag].add(doc_id)

        if document.judgment_date:
            year = document.judgment_date.year
            self.year_index[year].add(doc_id)

    def add_documents(self, documents: List[LegalDocument]):
        """Add multiple documents to the engine."""
        for doc in documents:
            self.add_document(doc)

    def filter_by_court(self, courts: Union[str, List[str]]) -> Set[str]:
        """Get document IDs for specific court(s)."""
        if isinstance(courts, str):
            courts = [courts]

        result_set = set()
        for court in courts:
            result_set.update(self.court_index.get(court, set()))

        return result_set

    def filter_by_legal_area(self, areas: Union[str, List[str]]) -> Set[str]:
        """Get document IDs for specific legal area(s)."""
        if isinstance(areas, str):
            areas = [areas]

        result_set = set()
        for area in areas:
            result_set.update(self.legal_area_index.get(area, set()))

        return result_set

    def filter_by_keywords(self, keywords: Union[str, List[str]],
                          match_all: bool = False) -> Set[str]:
        """
        Get document IDs containing specific keyword(s).

        Args:
            keywords: Single keyword or list of keywords
            match_all: If True, require ALL keywords; if False, require ANY keyword
        """
        if isinstance(keywords, str):
            keywords = [keywords]

        # Convert to lowercase for case-insensitive search
        keywords = [kw.lower() for kw in keywords]

        if not keywords:
            return set()

        if match_all:
            # Intersection: documents must contain ALL keywords
            result_set = self.keyword_index.get(keywords[0], set()).copy()
            for keyword in keywords[1:]:
                result_set &= self.keyword_index.get(keyword, set())
        else:
            # Union: documents must contain ANY keyword
            result_set = set()
            for keyword in keywords:
                result_set |= self.keyword_index.get(keyword, set())

        return result_set

    def filter_by_judges(self, judges: Union[str, List[str]],
                        match_all: bool = False) -> Set[str]:
        """Get document IDs for cases involving specific judge(s)."""
        if isinstance(judges, str):
            judges = [judges]

        if not judges:
            return set()

        if match_all:
            # All specified judges must be involved
            result_set = self.judge_index.get(judges[0], set()).copy()
            for judge in judges[1:]:
                result_set &= self.judge_index.get(judge, set())
        else:
            # Any of the specified judges
            result_set = set()
            for judge in judges:
                result_set |= self.judge_index.get(judge, set())

        return result_set

    def filter_by_date_range(self, start_year: int = None,
                           end_year: int = None) -> Set[str]:
        """Get document IDs for cases within a date range."""
        if start_year is None and end_year is None:
            return set(self.documents.keys())

        result_set = set()

        if start_year is None:
            start_year = min(self.year_index.keys()) if self.year_index else 1900
        if end_year is None:
            end_year = max(self.year_index.keys()) if self.year_index else 2100

        for year in range(start_year, end_year + 1):
            result_set |= self.year_index.get(year, set())

        return result_set

    def filter_by_tags(self, tags: Union[str, List[str]],
                      match_all: bool = False) -> Set[str]:
        """Get document IDs containing specific content tag(s)."""
        if isinstance(tags, str):
            tags = [tags]

        if not tags:
            return set()

        if match_all:
            result_set = self.tag_index.get(tags[0], set()).copy()
            for tag in tags[1:]:
                result_set &= self.tag_index.get(tag, set())
        else:
            result_set = set()
            for tag in tags:
                result_set |= self.tag_index.get(tag, set())

        return result_set

    def complex_filter(self, **criteria) -> Set[str]:
        """
        Perform complex filtering using multiple criteria with set intersections.

        Supported criteria:
        - courts: List of courts
        - legal_areas: List of legal areas
        - keywords: List of keywords (with keywords_match_all option)
        - judges: List of judges (with judges_match_all option)
        - start_year, end_year: Date range
        - tags: List of content tags (with tags_match_all option)
        """
        # Start with all documents
        result_set = set(self.documents.keys())

        # Apply court filter
        if 'courts' in criteria:
            court_set = self.filter_by_court(criteria['courts'])
            result_set &= court_set

        # Apply legal area filter
        if 'legal_areas' in criteria:
            area_set = self.filter_by_legal_area(criteria['legal_areas'])
            result_set &= area_set

        # Apply keyword filter
        if 'keywords' in criteria:
            keywords_match_all = criteria.get('keywords_match_all', False)
            keyword_set = self.filter_by_keywords(criteria['keywords'], keywords_match_all)
            result_set &= keyword_set

        # Apply judge filter
        if 'judges' in criteria:
            judges_match_all = criteria.get('judges_match_all', False)
            judge_set = self.filter_by_judges(criteria['judges'], judges_match_all)
            result_set &= judge_set

        # Apply date range filter
        if 'start_year' in criteria or 'end_year' in criteria:
            date_set = self.filter_by_date_range(
                criteria.get('start_year'),
                criteria.get('end_year')
            )
            result_set &= date_set

        # Apply tag filter
        if 'tags' in criteria:
            tags_match_all = criteria.get('tags_match_all', False)
            tag_set = self.filter_by_tags(criteria['tags'], tags_match_all)
            result_set &= tag_set

        return result_set

    def boolean_search(self, and_criteria: List[Dict] = None,
                      or_criteria: List[Dict] = None,
                      not_criteria: List[Dict] = None) -> Set[str]:
        """
        Perform Boolean search operations (AND, OR, NOT).

        Args:
            and_criteria: List of filter criteria that must ALL match
            or_criteria: List of filter criteria where ANY can match
            not_criteria: List of filter criteria to exclude
        """
        result_set = set(self.documents.keys())

        # AND operation: intersection of all criteria
        if and_criteria:
            for criteria in and_criteria:
                criteria_set = self.complex_filter(**criteria)
                result_set &= criteria_set

        # OR operation: union of any criteria
        if or_criteria:
            or_result_set = set()
            for criteria in or_criteria:
                criteria_set = self.complex_filter(**criteria)
                or_result_set |= criteria_set
            result_set &= or_result_set

        # NOT operation: exclude documents matching criteria
        if not_criteria:
            for criteria in not_criteria:
                criteria_set = self.complex_filter(**criteria)
                result_set -= criteria_set

        return result_set

    def get_documents(self, doc_ids: Set[str]) -> List[LegalDocument]:
        """Retrieve document objects for given IDs."""
        return [self.documents[doc_id] for doc_id in doc_ids
                if doc_id in self.documents]

    def search_and_get_documents(self, **criteria) -> List[LegalDocument]:
        """Convenience method to filter and return document objects."""
        doc_ids = self.complex_filter(**criteria)
        return self.get_documents(doc_ids)

    def get_filter_statistics(self) -> Dict[str, Any]:
        """Get statistics about the filtering indices."""
        return {
            "total_documents": len(self.documents),
            "courts": {
                "count": len(self.court_index),
                "distribution": {court: len(docs) for court, docs in self.court_index.items()}
            },
            "legal_areas": {
                "count": len(self.legal_area_index),
                "distribution": dict(sorted(
                    [(area, len(docs)) for area, docs in self.legal_area_index.items()],
                    key=lambda x: x[1], reverse=True
                )[:10])  # Top 10
            },
            "keywords": {
                "count": len(self.keyword_index),
                "most_common": dict(sorted(
                    [(kw, len(docs)) for kw, docs in self.keyword_index.items()],
                    key=lambda x: x[1], reverse=True
                )[:10])  # Top 10
            },
            "judges": {
                "count": len(self.judge_index),
                "most_active": dict(sorted(
                    [(judge, len(docs)) for judge, docs in self.judge_index.items()],
                    key=lambda x: x[1], reverse=True
                )[:10])  # Top 10
            },
            "years": {
                "range": f"{min(self.year_index.keys()) if self.year_index else 'N/A'} - {max(self.year_index.keys()) if self.year_index else 'N/A'}",
                "distribution": dict(sorted(self.year_index.items()))
            }
        }


def demo_fcl_legal_filtering():
    """Demonstration of set-based filtering with realistic FCL legal data."""

    # Initialize the filtering engine
    filter_engine = LegalDocumentFilterEngine()

    # Sample UK legal documents with rich metadata
    sample_documents = [
        LegalDocument(
            "1", "Donoghue v Stevenson", "[1932] AC 562", "House of Lords",
            ["Tort Law", "Negligence"], ["negligence", "duty of care", "neighbour principle"],
            date(1932, 5, 26), "[1932] UKHL 100",
            ["Lord Atkin", "Lord Thankerton", "Lord Macmillan"],
            ["landmark", "precedent", "liability"]
        ),
        LegalDocument(
            "2", "Carlill v Carbolic Smoke Ball Company", "[1893] 1 QB 256", "Court of Appeal",
            ["Contract Law"], ["unilateral contract", "offer", "acceptance", "consideration"],
            date(1892, 12, 7), "[1892] EWCA Civ 1",
            ["Lindley LJ", "Bowen LJ", "A.L. Smith LJ"],
            ["contract formation", "advertisement", "precedent"]
        ),
        LegalDocument(
            "3", "Rylands v Fletcher", "(1868) LR 3 HL 330", "House of Lords",
            ["Tort Law", "Property Law"], ["strict liability", "non-natural use", "escape"],
            date(1868, 7, 17), "[1868] UKHL 1",
            ["Lord Cairns", "Lord Cranworth"],
            ["strict liability", "landmark", "property"]
        ),
        LegalDocument(
            "4", "R v Brown", "[1994] 1 AC 212", "House of Lords",
            ["Criminal Law"], ["consent", "assault", "bodily harm", "public policy"],
            date(1993, 3, 11), "[1993] UKHL 19",
            ["Lord Templeman", "Lord Jauncey", "Lord Lowry"],
            ["consent", "assault", "criminal"]
        ),
        LegalDocument(
            "5", "Pepper v Hart", "[1993] AC 593", "House of Lords",
            ["Constitutional Law", "Statutory Interpretation"], ["hansard", "parliamentary material", "interpretation"],
            date(1992, 11, 26), "[1992] UKHL 3",
            ["Lord Griffiths", "Lord Ackner", "Lord Oliver"],
            ["constitutional", "interpretation", "parliament"]
        ),
        LegalDocument(
            "6", "Caparo Industries plc v Dickman", "[1990] 2 AC 605", "House of Lords",
            ["Tort Law", "Negligence"], ["duty of care", "proximity", "fair just reasonable"],
            date(1990, 2, 8), "[1990] UKHL 2",
            ["Lord Bridge", "Lord Roskill", "Lord Ackner"],
            ["negligence", "duty", "three-stage test"]
        ),
        LegalDocument(
            "7", "Miller v Jackson", "[1977] QB 966", "Court of Appeal",
            ["Tort Law", "Property Law"], ["nuisance", "public benefit", "injunction"],
            date(1977, 5, 3), "[1977] EWCA Civ 6",
            ["Lord Denning MR", "Geoffrey Lane LJ", "Cumming-Bruce LJ"],
            ["nuisance", "public benefit", "cricket"]
        ),
        LegalDocument(
            "8", "R (Miller) v The Prime Minister", "[2019] UKSC 41", "Supreme Court",
            ["Constitutional Law", "Administrative Law"], ["prorogation", "justiciability", "executive power"],
            date(2019, 9, 24), "[2019] UKSC 41",
            ["Lady Hale", "Lord Reed", "Lord Kerr", "Lord Wilson"],
            ["constitutional", "executive", "parliament", "justiciability"]
        ),
        LegalDocument(
            "9", "Cambridge Water Co Ltd v Eastern Counties Leather plc", "[1994] 2 AC 264", "House of Lords",
            ["Tort Law", "Environmental Law"], ["strict liability", "foreseeability", "environmental"],
            date(1994, 1, 9), "[1994] UKHL 5",
            ["Lord Goff", "Lord Jauncey", "Lord Lowry"],
            ["environmental", "liability", "foreseeability"]
        ),
        LegalDocument(
            "10", "White v Chief Constable of South Yorkshire Police", "[1999] 2 AC 455", "House of Lords",
            ["Tort Law", "Negligence"], ["psychiatric injury", "secondary victim", "rescue"],
            date(1998, 12, 3), "[1998] UKHL 45",
            ["Lord Steyn", "Lord Hoffmann", "Lord Griffiths"],
            ["psychiatric", "secondary victim", "rescue"]
        )
    ]

    print("=== FCL Legal Document Filtering Demo ===\n")

    # Add all documents to the engine
    filter_engine.add_documents(sample_documents)
    print(f"Loaded {len(sample_documents)} legal documents into filtering engine\n")

    # Display filtering statistics
    print("=== Filtering Engine Statistics ===")
    stats = filter_engine.get_filter_statistics()
    print(f"Total documents: {stats['total_documents']}")
    print(f"Courts represented: {stats['courts']['count']}")
    print(f"Legal areas: {stats['legal_areas']['count']}")
    print(f"Unique keywords: {stats['keywords']['count']}")
    print(f"Judges: {stats['judges']['count']}")
    print(f"Year range: {stats['years']['range']}")

    print(f"\nCourt distribution:")
    for court, count in stats['courts']['distribution'].items():
        print(f"  {court}: {count} cases")

    print(f"\nTop legal areas:")
    for area, count in list(stats['legal_areas']['distribution'].items())[:5]:
        print(f"  {area}: {count} cases")

    # Demonstrate basic filtering
    print(f"\n=== Basic Filtering Examples ===")

    # Filter by court
    hl_cases = filter_engine.filter_by_court("House of Lords")
    print(f"\nHouse of Lords cases: {len(hl_cases)}")
    for doc in filter_engine.get_documents(hl_cases)[:3]:
        print(f"  - {doc.case_name}")

    # Filter by legal area
    tort_cases = filter_engine.filter_by_legal_area("Tort Law")
    print(f"\nTort Law cases: {len(tort_cases)}")
    for doc in filter_engine.get_documents(tort_cases)[:3]:
        print(f"  - {doc.case_name}")

    # Filter by keywords (ANY)
    negligence_cases = filter_engine.filter_by_keywords(["negligence", "duty of care"])
    print(f"\nCases with 'negligence' OR 'duty of care': {len(negligence_cases)}")
    for doc in filter_engine.get_documents(negligence_cases):
        print(f"  - {doc.case_name}")

    # Filter by keywords (ALL)
    strict_negligence = filter_engine.filter_by_keywords(["negligence", "duty"], match_all=True)
    print(f"\nCases with BOTH 'negligence' AND 'duty': {len(strict_negligence)}")
    for doc in filter_engine.get_documents(strict_negligence):
        print(f"  - {doc.case_name}")

    # Demonstrate complex filtering
    print(f"\n=== Complex Multi-Criteria Filtering ===")

    # Modern tort law cases
    modern_tort = filter_engine.complex_filter(
        legal_areas=["Tort Law"],
        start_year=1990,
        courts=["House of Lords", "Supreme Court"]
    )
    print(f"\nModern tort law cases (1990+, House of Lords/Supreme Court): {len(modern_tort)}")
    for doc in filter_engine.get_documents(modern_tort):
        print(f"  - {doc.case_name} ({doc.judgment_date.year})")

    # Cases involving specific judges
    lord_ackner_cases = filter_engine.complex_filter(
        judges=["Lord Ackner"]
    )
    print(f"\nCases involving Lord Ackner: {len(lord_ackner_cases)}")
    for doc in filter_engine.get_documents(lord_ackner_cases):
        print(f"  - {doc.case_name}")

    # Constitutional cases with parliamentary relevance
    constitutional_parliament = filter_engine.complex_filter(
        legal_areas=["Constitutional Law"],
        keywords=["parliament"],
        tags=["constitutional"]
    )
    print(f"\nConstitutional cases involving parliament: {len(constitutional_parliament)}")
    for doc in filter_engine.get_documents(constitutional_parliament):
        print(f"  - {doc.case_name}")

    # Demonstrate Boolean search operations
    print(f"\n=== Boolean Search Operations ===")

    # AND operation: Tort law cases from the 1990s
    and_search = filter_engine.boolean_search(
        and_criteria=[
            {"legal_areas": ["Tort Law"]},
            {"start_year": 1990, "end_year": 1999}
        ]
    )
    print(f"\nTort law cases from 1990s (AND operation): {len(and_search)}")
    for doc in filter_engine.get_documents(and_search):
        print(f"  - {doc.case_name} ({doc.judgment_date.year})")

    # OR operation: Either House of Lords or landmark cases
    or_search = filter_engine.boolean_search(
        or_criteria=[
            {"courts": ["House of Lords"]},
            {"tags": ["landmark"]}
        ]
    )
    print(f"\nHouse of Lords OR landmark cases (OR operation): {len(or_search)}")
    for doc in filter_engine.get_documents(or_search)[:5]:
        print(f"  - {doc.case_name}")

    # NOT operation: Tort law cases that are NOT negligence cases
    not_search = filter_engine.boolean_search(
        and_criteria=[{"legal_areas": ["Tort Law"]}],
        not_criteria=[{"keywords": ["negligence"]}]
    )
    print(f"\nTort law cases excluding negligence (NOT operation): {len(not_search)}")
    for doc in filter_engine.get_documents(not_search):
        print(f"  - {doc.case_name}")

    # Complex Boolean search
    complex_boolean = filter_engine.boolean_search(
        and_criteria=[
            {"start_year": 1980},  # Modern cases
            {"tags": ["precedent"]}  # Precedent-setting
        ],
        or_criteria=[
            {"courts": ["House of Lords"]},
            {"courts": ["Supreme Court"]}
        ],
        not_criteria=[
            {"legal_areas": ["Criminal Law"]}  # Exclude criminal law
        ]
    )
    print(f"\nComplex search (modern precedent cases, HL/SC, non-criminal): {len(complex_boolean)}")
    for doc in filter_engine.get_documents(complex_boolean):
        print(f"  - {doc.case_name} ({doc.court})")

    # Demonstrate set intersection patterns
    print(f"\n=== Set Intersection Patterns ===")

    # Find overlaps between different criteria
    tort_set = filter_engine.filter_by_legal_area("Tort Law")
    hl_set = filter_engine.filter_by_court("House of Lords")
    modern_set = filter_engine.filter_by_date_range(start_year=1990)

    # Multiple intersections
    tort_hl = tort_set & hl_set
    tort_modern = tort_set & modern_set
    hl_modern = hl_set & modern_set
    all_three = tort_set & hl_set & modern_set

    print(f"Tort Law cases: {len(tort_set)}")
    print(f"House of Lords cases: {len(hl_set)}")
    print(f"Modern cases (1990+): {len(modern_set)}")
    print(f"Tort ) House of Lords: {len(tort_hl)}")
    print(f"Tort ) Modern: {len(tort_modern)}")
    print(f"House of Lords ) Modern: {len(hl_modern)}")
    print(f"All three criteria: {len(all_three)}")

    if all_three:
        print(f"\nCases meeting all three criteria:")
        for doc in filter_engine.get_documents(all_three):
            print(f"  - {doc.case_name}")


if __name__ == "__main__":
    demo_fcl_legal_filtering()