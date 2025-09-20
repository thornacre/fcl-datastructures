"""
Regex Patterns for Legal Reference Extraction
=============================================

This module provides comprehensive regex patterns for extracting legal references
from UK legal documents in Find Case Law (FCL). It handles various citation formats,
case names, and legal document identifiers used in the UK legal system.

Key FCL Use Cases:
- Extract neutral citations from judgment text
- Identify case law references and precedents
- Parse statutory references and legislation
- Extract court names and hearing dates
- Identify judges and legal parties
- Find legal document identifiers (URIs, DOIs)
"""

import re
from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass
from enum import Enum
import json


class CitationType(Enum):
    """Types of legal citations"""
    NEUTRAL = "neutral"           # [2023] UKSC 15
    LAW_REPORT = "law_report"     # [2023] 1 WLR 234
    EUROPEAN = "european"         # C-123/45
    STATUTORY = "statutory"       # Section 1 of the Act
    SECONDARY = "secondary"       # SI 2023/123


@dataclass
class ExtractedCitation:
    """Represents an extracted legal citation"""
    full_text: str
    citation_type: CitationType
    year: Optional[int]
    court: Optional[str]
    number: Optional[str]
    series: Optional[str]
    start_pos: int
    end_pos: int
    confidence: float = 1.0


class UKLegalPatterns:
    """
    Comprehensive regex patterns for UK legal document processing.
    """

    def __init__(self):
        self._compile_patterns()

    def _compile_patterns(self):
        """Compile all regex patterns for efficiency"""

        # Modern UK neutral citations: [YYYY] COURT NUMBER
        self.neutral_citation_pattern = re.compile(
            r'\[(?P<year>(?:19|20)\d{2})\]\s+'
            r'(?P<court>UKSC|UKHL|UKPC|EWCA|EWHC|EWCOP|EWFC|'
            r'UKUT|UKFTT|UKEAT|UKET|SCSC|SSCS|HESC|AACR|'
            r'NICty|NICA|NIQB|CSOH|CSIH|SAC|SLT|SC|SCLR)\s+'
            r'(?P<number>\d+)(?:\s*\((?P<division>[^)]+)\))?',
            re.IGNORECASE
        )

        # Traditional law report citations
        self.law_report_pattern = re.compile(
            r'\[(?P<year>(?:19|20)\d{2})\]\s+'
            r'(?P<volume>\d+)\s+'
            r'(?P<series>WLR|All\s*ER|AC|Ch|QB|Cr\s*App\s*R|'
            r'Lloyd\'s\s*Rep|BCLC|BCC|IRLR|ICR|PIQR|RTR|'
            r'Fam|FLR|FCR|CMLR|ECR|SLT|SC|SCLR)\s+'
            r'(?P<page>\d+)',
            re.IGNORECASE
        )

        # Case names: R v Smith, Smith v Jones
        self.case_name_pattern = re.compile(
            r'(?:(?P<applicant>R\s*(?:\([^)]+\))?|'
            r'[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*(?:\s+(?:Ltd|Plc|Inc|Corp))?)\s+'
            r'(?:v\.?|and)\s+'
            r'(?P<respondent>[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*(?:\s+(?:Ltd|Plc|Inc|Corp))?))',
            re.IGNORECASE
        )

        # Statutory references
        self.statute_pattern = re.compile(
            r'(?:(?:Section|s\.|ss\.|Schedule|Sch\.?|Part|Article|Art\.?|Regulation|reg\.?)\s+)'
            r'(?P<section>\d+(?:[A-Z])?(?:\(\d+\))*(?:-\d+)?)\s+'
            r'(?:of\s+)?(?:the\s+)?'
            r'(?P<act>[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:Act|Regulations?|Rules?|Order)\s+\d{4})',
            re.IGNORECASE
        )

        # European citations
        self.european_citation_pattern = re.compile(
            r'(?:Case\s+)?(?P<type>C|T)-(?P<number>\d+)/(?P<year>\d{2,4})(?:\s+(?P<parties>[^,.\n]+))?',
            re.IGNORECASE
        )

        # Judges
        self.judge_pattern = re.compile(
            r'(?:(?:Lord|Lady|Sir|Dame|Mr|Mrs|Ms)\s+)?'
            r'(?:Justice|J\.?)\s+(?P<surname>[A-Z][a-z]+(?:-[A-Z][a-z]+)?)|'
            r'(?P<title>Lord|Lady)\s+(?P<name>[A-Z][a-z]+(?:\s+of\s+[A-Z][a-z]+)?)',
            re.IGNORECASE
        )


class LegalTextExtractor:
    """
    Legal text extraction using compiled regex patterns.
    """

    def __init__(self):
        self.patterns = UKLegalPatterns()
        self.extraction_stats = {
            'total_extractions': 0,
            'by_type': {},
            'confidence_distribution': []
        }

    def extract_all_citations(self, text: str) -> List[ExtractedCitation]:
        """Extract all legal citations from text with type classification."""
        citations = []

        # Extract neutral citations
        for match in self.patterns.neutral_citation_pattern.finditer(text):
            citation = ExtractedCitation(
                full_text=match.group(0),
                citation_type=CitationType.NEUTRAL,
                year=int(match.group('year')),
                court=match.group('court'),
                number=match.group('number'),
                series=None,
                start_pos=match.start(),
                end_pos=match.end(),
                confidence=0.95
            )
            citations.append(citation)

        # Extract law report citations
        for match in self.patterns.law_report_pattern.finditer(text):
            citation = ExtractedCitation(
                full_text=match.group(0),
                citation_type=CitationType.LAW_REPORT,
                year=int(match.group('year')),
                court=None,
                number=match.group('page'),
                series=match.group('series'),
                start_pos=match.start(),
                end_pos=match.end(),
                confidence=0.90
            )
            citations.append(citation)

        # Extract European citations
        for match in self.patterns.european_citation_pattern.finditer(text):
            year_val = match.group('year')
            full_year = int(year_val) if len(year_val) == 4 else 2000 + int(year_val)

            citation = ExtractedCitation(
                full_text=match.group(0),
                citation_type=CitationType.EUROPEAN,
                year=full_year,
                court='ECJ' if match.group('type') == 'C' else 'CFI',
                number=match.group('number'),
                series=None,
                start_pos=match.start(),
                end_pos=match.end(),
                confidence=0.85
            )
            citations.append(citation)

        # Sort by position and update stats
        citations.sort(key=lambda c: c.start_pos)
        self._update_stats(citations)
        return citations

    def extract_case_names(self, text: str) -> List[Dict[str, any]]:
        """Extract case names from legal text"""
        case_names = []

        for match in self.patterns.case_name_pattern.finditer(text):
            case_names.append({
                'text': match.group(0),
                'applicant': match.group('applicant') or '',
                'respondent': match.group('respondent') or '',
                'start_pos': match.start(),
                'end_pos': match.end()
            })

        return case_names

    def extract_statutes(self, text: str) -> List[Dict[str, any]]:
        """Extract statutory references"""
        statutes = []

        for match in self.patterns.statute_pattern.finditer(text):
            statutes.append({
                'text': match.group(0),
                'section': match.group('section'),
                'act': match.group('act'),
                'start_pos': match.start(),
                'end_pos': match.end()
            })

        return statutes

    def extract_judges(self, text: str) -> List[Dict[str, any]]:
        """Extract judge names from legal text"""
        judges = []

        for match in self.patterns.judge_pattern.finditer(text):
            surname = match.group('surname') if 'surname' in match.groupdict() and match.group('surname') else ''
            title = match.group('title') if 'title' in match.groupdict() and match.group('title') else ''
            name = match.group('name') if 'name' in match.groupdict() and match.group('name') else ''

            judges.append({
                'text': match.group(0),
                'surname': surname,
                'title': title,
                'full_name': name or surname,
                'start_pos': match.start(),
                'end_pos': match.end()
            })

        return judges

    def validate_citation(self, citation: str) -> Tuple[bool, str, List[str]]:
        """Validate and normalize a legal citation."""
        errors = []
        normalized = citation.strip()

        # Check neutral citation format
        neutral_match = self.patterns.neutral_citation_pattern.match(normalized)
        if neutral_match:
            year = int(neutral_match.group('year'))
            court = neutral_match.group('court')
            number = neutral_match.group('number')

            if year < 1990 or year > 2030:
                errors.append(f"Year {year} is outside expected range (1990-2030)")

            valid_courts = {
                'UKSC', 'UKHL', 'UKPC', 'EWCA', 'EWHC', 'EWCOP',
                'EWFC', 'UKUT', 'UKFTT', 'UKEAT', 'UKET'
            }
            if court not in valid_courts:
                errors.append(f"Court code '{court}' not recognized")

            normalized = f"[{year}] {court} {number}"
            if neutral_match.group('division'):
                normalized += f" ({neutral_match.group('division')})"

            return len(errors) == 0, normalized, errors

        # Check law report format
        law_report_match = self.patterns.law_report_pattern.match(normalized)
        if law_report_match:
            year = int(law_report_match.group('year'))
            volume = law_report_match.group('volume')
            series = law_report_match.group('series')
            page = law_report_match.group('page')

            if year < 1900 or year > 2030:
                errors.append(f"Year {year} is outside expected range (1900-2030)")

            normalized = f"[{year}] {volume} {series} {page}"
            return len(errors) == 0, normalized, errors

        errors.append("Citation format not recognized")
        return False, citation, errors

    def get_extraction_statistics(self) -> Dict[str, any]:
        """Get statistics about extractions performed"""
        return {
            'total_extractions': self.extraction_stats['total_extractions'],
            'by_type': dict(self.extraction_stats['by_type']),
            'avg_confidence': sum(self.extraction_stats['confidence_distribution']) /
                            len(self.extraction_stats['confidence_distribution'])
                            if self.extraction_stats['confidence_distribution'] else 0
        }

    def _update_stats(self, citations: List[ExtractedCitation]) -> None:
        """Update extraction statistics"""
        self.extraction_stats['total_extractions'] += len(citations)

        for citation in citations:
            type_name = citation.citation_type.value
            self.extraction_stats['by_type'][type_name] = (
                self.extraction_stats['by_type'].get(type_name, 0) + 1
            )
            self.extraction_stats['confidence_distribution'].append(citation.confidence)


def demonstrate_regex_patterns():
    """Demonstrate regex pattern extraction with UK legal text"""

    print("=== Legal Regex Pattern Extraction Demo ===\n")

    # Sample UK legal text
    sample_text = '''
    In the matter of R (on the application of Miller) v The Prime Minister [2023] UKSC 15,
    the Supreme Court considered the constitutional principles established in the earlier
    decision of [2019] UKSC 41. The judgment was delivered by Lord Reed (President),
    Lady Hale (Deputy President), and Lord Kerr on 15th March 2023.

    The Court referred to Section 4 of the Human Rights Act 1998 and the principles
    established in Ghaidan v Godin-Mendoza [2004] 2 AC 557. The case also considered
    European authorities including Case C-123/45 Commission v Germany.

    Mr Justice Swift, sitting in the High Court in [2023] EWHC 1456 (Admin), had
    previously considered similar issues in Smith v Jones [2022] EWCA Civ 234.
    '''

    # Initialize extractor
    extractor = LegalTextExtractor()

    # 1. Extract all citations
    print("1. EXTRACTED CITATIONS:")
    citations = extractor.extract_all_citations(sample_text)

    for i, citation in enumerate(citations, 1):
        print(f"   {i}. {citation.full_text}")
        print(f"      Type: {citation.citation_type.value}")
        print(f"      Year: {citation.year}")
        print(f"      Court: {citation.court}")
        print(f"      Confidence: {citation.confidence:.2f}")

    # 2. Extract case names
    print(f"\n2. EXTRACTED CASE NAMES:")
    case_names = extractor.extract_case_names(sample_text)

    for i, case in enumerate(case_names, 1):
        print(f"   {i}. {case['text']}")
        print(f"      Applicant: {case['applicant']}")
        print(f"      Respondent: {case['respondent']}")

    # 3. Extract judges
    print(f"\n3. EXTRACTED JUDGES:")
    judges = extractor.extract_judges(sample_text)

    for i, judge in enumerate(judges, 1):
        print(f"   {i}. {judge['text']}")
        print(f"      Title: {judge['title']}")
        print(f"      Name: {judge['full_name']}")

    # 4. Extract statutory references
    print(f"\n4. EXTRACTED STATUTORY REFERENCES:")
    statutes = extractor.extract_statutes(sample_text)

    for i, statute in enumerate(statutes, 1):
        print(f"   {i}. {statute['text']}")
        print(f"      Section: {statute['section']}")
        print(f"      Act: {statute['act']}")

    # 5. Citation validation
    print(f"\n5. CITATION VALIDATION:")
    test_citations = [
        "[2023] UKSC 15",
        "[2019] 2 WLR 456",
        "[2023] BADCOURT 99",  # Invalid court
        "[1899] UKSC 1",       # Invalid year
        "Case C-123/45"
    ]

    for citation in test_citations:
        is_valid, normalized, errors = extractor.validate_citation(citation)
        print(f"   '{citation}':")
        print(f"      Valid: {is_valid}")
        print(f"      Normalized: {normalized}")
        if errors:
            print(f"      Errors: {', '.join(errors)}")

    # 6. Extraction statistics
    print(f"\n6. EXTRACTION STATISTICS:")
    stats = extractor.get_extraction_statistics()
    print(f"   Total extractions: {stats['total_extractions']}")
    print(f"   Average confidence: {stats['avg_confidence']:.2f}")
    print(f"   By type: {stats['by_type']}")

    return {
        'extractor': extractor,
        'citations': citations,
        'case_names': case_names,
        'judges': judges,
        'statutes': statutes,
        'statistics': stats
    }


if __name__ == "__main__":
    demonstrate_regex_patterns()