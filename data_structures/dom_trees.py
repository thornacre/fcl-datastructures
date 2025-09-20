"""
DOM Trees for LegalDocML/XML Structure Parsing
==============================================

This module demonstrates DOM tree manipulation for parsing UK legal documents
in LegalDocML format used by Find Case Law (FCL). LegalDocML is an XML-based
standard for legal document markup.

Key FCL Use Cases:
- Parse judgment XML files from UK courts
- Extract metadata (citation, court, judges, date)
- Navigate document structure (header, body, conclusions)
- Transform XML to JSON for API responses
- Validate document structure against schema
"""

from xml.etree import ElementTree as ET
from bs4 import BeautifulSoup
import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime


@dataclass
class JudgmentMetadata:
    """Structured metadata extracted from legal documents"""
    neutral_citation: str
    court: str
    date: datetime
    judges: List[str]
    case_name: str
    uri: str
    subject_matter: List[str]


class LegalDocMLParser:
    """
    Parser for LegalDocML documents used in FCL.

    LegalDocML Structure:
    - akomaNtoso: Root element
    - judgment: Main document container
    - meta: Metadata section
    - preface: Document header
    - body: Main content
    - conclusions: Final statements
    """

    def __init__(self):
        self.namespaces = {
            'akn': 'http://docs.oasis-open.org/legaldocml/ns/akn/3.0',
            'uk': 'https://caselaw.nationalarchives.gov.uk/akn'
        }

    def parse_judgment_xml(self, xml_content: str) -> Dict[str, Any]:
        """
        Parse a complete judgment XML document.

        Example UK judgment structure:
        ```xml
        <akomaNtoso xmlns="http://docs.oasis-open.org/legaldocml/ns/akn/3.0">
          <judgment name="judgment">
            <meta>
              <identification source="#tna">
                <FRBRWork>
                  <FRBRuri value="/akn/uk/uksc/2023/1"/>
                  <FRBRdate date="2023-01-15"/>
                  <FRBRauthor href="#uksc"/>
                  <FRBRcountry value="uk"/>
                </FRBRWork>
              </identification>
            </meta>
            <preface>
              <p class="cite">[2023] UKSC 1</p>
              <p class="court">SUPREME COURT</p>
            </preface>
            <body>
              <paragraph>
                <content>
                  <p>Judgment content...</p>
                </content>
              </paragraph>
            </body>
          </judgment>
        </akomaNtoso>
        ```
        """
        try:
            # Parse with BeautifulSoup for better namespace handling
            soup = BeautifulSoup(xml_content, 'xml')

            # Extract metadata
            metadata = self._extract_metadata(soup)

            # Extract document sections
            sections = self._extract_sections(soup)

            # Build structured representation
            judgment_data = {
                'metadata': metadata.__dict__ if metadata else {},
                'sections': sections,
                'dom_structure': self._build_dom_tree(soup),
                'extracted_text': self._extract_plain_text(soup),
                'citations_found': self._extract_citations(soup),
                'paragraph_count': len(soup.find_all('paragraph') or []),
                'page_count': len(soup.find_all('neutralCitation') or [])
            }

            return judgment_data

        except Exception as e:
            return {'error': f'Failed to parse XML: {str(e)}', 'metadata': {}, 'sections': []}

    def _extract_metadata(self, soup: BeautifulSoup) -> Optional[JudgmentMetadata]:
        """Extract structured metadata from judgment header"""
        try:
            # Find neutral citation - e.g., [2023] UKSC 1
            citation_elem = soup.find('neutralCitation')
            neutral_citation = citation_elem.get_text().strip() if citation_elem else "Unknown"

            # Extract court information
            court_elem = soup.find('court')
            court = court_elem.get_text().strip() if court_elem else "Unknown Court"

            # Find judgment date
            date_elem = soup.find('FRBRdate')
            date_str = date_elem.get('date') if date_elem else "2023-01-01"
            judgment_date = datetime.fromisoformat(date_str)

            # Extract judges
            judges = []
            for judge_elem in soup.find_all('judge'):
                judge_name = judge_elem.get_text().strip()
                if judge_name:
                    judges.append(judge_name)

            # If no judges found in specific tags, look in header
            if not judges:
                header_text = soup.find('preface')
                if header_text:
                    # Look for common patterns like "Lord Smith", "Lady Jones"
                    import re
                    judge_pattern = r'(?:Lord|Lady|Mr Justice|Mrs Justice|Sir|Dame)\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*'
                    found_judges = re.findall(judge_pattern, header_text.get_text())
                    judges.extend(found_judges[:5])  # Limit to 5 judges

            # Extract case name
            case_name_elem = soup.find('docTitle')
            case_name = case_name_elem.get_text().strip() if case_name_elem else "Unknown Case"

            # Extract URI
            uri_elem = soup.find('FRBRuri')
            uri = uri_elem.get('value') if uri_elem else f"/judgment/{neutral_citation.replace(' ', '_')}"

            # Extract subject matter/keywords
            subject_matter = []
            for keyword_elem in soup.find_all('keyword'):
                keyword = keyword_elem.get_text().strip()
                if keyword:
                    subject_matter.append(keyword)

            return JudgmentMetadata(
                neutral_citation=neutral_citation,
                court=court,
                date=judgment_date,
                judges=judges,
                case_name=case_name,
                uri=uri,
                subject_matter=subject_matter
            )

        except Exception as e:
            print(f"Error extracting metadata: {e}")
            return None

    def _extract_sections(self, soup: BeautifulSoup) -> List[Dict[str, Any]]:
        """Extract document sections with hierarchy"""
        sections = []

        # Extract preface (header information)
        preface = soup.find('preface')
        if preface:
            sections.append({
                'type': 'preface',
                'title': 'Document Header',
                'content': preface.get_text().strip(),
                'subsections': []
            })

        # Extract body paragraphs
        body = soup.find('body')
        if body:
            paragraphs = body.find_all('paragraph')
            body_content = []

            for i, para in enumerate(paragraphs[:20], 1):  # Limit to first 20 paragraphs
                para_text = para.get_text().strip()
                if para_text:
                    body_content.append({
                        'paragraph_id': f'para_{i}',
                        'content': para_text[:500] + ('...' if len(para_text) > 500 else ''),
                        'word_count': len(para_text.split())
                    })

            sections.append({
                'type': 'body',
                'title': 'Judgment Body',
                'content': f'{len(paragraphs)} paragraphs',
                'subsections': body_content
            })

        # Extract conclusions
        conclusions = soup.find('conclusions')
        if conclusions:
            sections.append({
                'type': 'conclusions',
                'title': 'Conclusions',
                'content': conclusions.get_text().strip(),
                'subsections': []
            })

        return sections

    def _build_dom_tree(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """Build a simplified DOM tree representation"""
        root = soup.find('akomaNtoso') or soup.find('judgment')
        if not root:
            return {}

        def build_tree_node(element) -> Dict[str, Any]:
            node = {
                'tag': element.name,
                'attributes': dict(element.attrs) if element.attrs else {},
                'text': element.get_text(strip=True)[:100] if element.get_text(strip=True) else '',
                'children': []
            }

            # Limit depth to avoid too large structures
            if len(element.find_all()) < 50:
                for child in element.find_all(recursive=False):
                    if child.name:  # Skip text nodes
                        node['children'].append(build_tree_node(child))

            return node

        return build_tree_node(root)

    def _extract_plain_text(self, soup: BeautifulSoup) -> str:
        """Extract plain text content for full-text search"""
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()

        # Get text and clean it up
        text = soup.get_text()
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)

        return text[:5000]  # Limit to first 5000 characters

    def _extract_citations(self, soup: BeautifulSoup) -> List[str]:
        """Extract legal citations found in the document"""
        text = soup.get_text()

        # Common UK citation patterns
        import re
        citation_patterns = [
            r'\[(?:19|20)\d{2}\]\s+(?:UKSC|UKHL|EWCA|EWHC|UKUT|UKFTT)\s+\d+',  # Modern neutral citations
            r'\[(?:19|20)\d{2}\]\s+\d+\s+(?:WLR|All ER|AC|Ch|QB|Cr App R)',     # Law reports
            r'(?:19|20)\d{2}\s+(?:SLT|SC|SCLR)\s+\d+',                          # Scottish citations
        ]

        citations = []
        for pattern in citation_patterns:
            found = re.findall(pattern, text, re.IGNORECASE)
            citations.extend(found[:10])  # Limit to 10 per pattern

        return list(set(citations))  # Remove duplicates


def demonstrate_legaldocml_parsing():
    """Demonstrate DOM tree operations with sample UK judgment data"""

    # Sample LegalDocML document (simplified)
    sample_xml = '''<?xml version="1.0" encoding="UTF-8"?>
    <akomaNtoso xmlns="http://docs.oasis-open.org/legaldocml/ns/akn/3.0">
      <judgment name="judgment">
        <meta>
          <identification source="#tna">
            <FRBRWork>
              <FRBRuri value="/akn/uk/uksc/2023/42"/>
              <FRBRdate date="2023-11-15"/>
              <FRBRauthor href="#uksc"/>
              <FRBRcountry value="uk"/>
            </FRBRWork>
          </identification>
        </meta>
        <preface>
          <neutralCitation>[2023] UKSC 42</neutralCitation>
          <court>SUPREME COURT</court>
          <docTitle>R (on the application of Miller) v The Prime Minister</docTitle>
          <judge>Lord Reed (President)</judge>
          <judge>Lady Hale (Deputy President)</judge>
          <judge>Lord Kerr</judge>
        </preface>
        <body>
          <paragraph id="para_1">
            <content>
              <p>This appeal concerns the constitutional principles surrounding the exercise of prerogative powers by the Executive and their relationship with Parliament.</p>
            </content>
          </paragraph>
          <paragraph id="para_2">
            <content>
              <p>The case of <cite>[2019] UKSC 41</cite> established important precedents regarding parliamentary sovereignty. We must also consider <cite>[2017] UKSC 5</cite> and the principles outlined therein.</p>
            </content>
          </paragraph>
          <paragraph id="para_3">
            <content>
              <p>The respondent argues that Article 50 of the Treaty on European Union creates obligations which cannot be fulfilled without parliamentary approval, citing <cite>[2016] EWHC 2768 (Admin)</cite>.</p>
            </content>
          </paragraph>
        </body>
        <conclusions>
          <paragraph>
            <content>
              <p>For the reasons given above, this appeal is allowed. The order of the Divisional Court is restored.</p>
            </content>
          </paragraph>
        </conclusions>
      </judgment>
    </akomaNtoso>'''

    print("=== LegalDocML DOM Tree Parsing Demo ===\n")

    # Initialize parser
    parser = LegalDocMLParser()

    # Parse the sample document
    result = parser.parse_judgment_xml(sample_xml)

    # Display results
    print("1. EXTRACTED METADATA:")
    metadata = result['metadata']
    for key, value in metadata.items():
        if isinstance(value, list):
            print(f"   {key}: {', '.join(map(str, value))}")
        else:
            print(f"   {key}: {value}")

    print(f"\n2. DOCUMENT STRUCTURE:")
    print(f"   Sections found: {len(result['sections'])}")
    for section in result['sections']:
        print(f"   - {section['type']}: {section['title']}")
        if section['subsections']:
            print(f"     Subsections: {len(section['subsections'])}")

    print(f"\n3. DOM TREE ANALYSIS:")
    dom_tree = result['dom_structure']
    print(f"   Root element: {dom_tree.get('tag', 'unknown')}")
    print(f"   Child elements: {len(dom_tree.get('children', []))}")
    print(f"   Attributes: {dom_tree.get('attributes', {})}")

    print(f"\n4. CITATIONS EXTRACTED:")
    citations = result['citations_found']
    for citation in citations:
        print(f"   - {citation}")

    print(f"\n5. CONTENT STATISTICS:")
    print(f"   Paragraph count: {result['paragraph_count']}")
    print(f"   Text length: {len(result['extracted_text'])} characters")
    print(f"   Citations found: {len(citations)}")

    print(f"\n6. SAMPLE EXTRACTED TEXT:")
    print(f"   {result['extracted_text'][:300]}...")

    # Demonstrate DOM navigation
    print(f"\n7. DOM NAVIGATION EXAMPLE:")
    soup = BeautifulSoup(sample_xml, 'xml')

    # Find all paragraphs
    paragraphs = soup.find_all('paragraph')
    print(f"   Found {len(paragraphs)} paragraphs")

    # Find citations within paragraphs
    for i, para in enumerate(paragraphs, 1):
        cites = para.find_all('cite')
        if cites:
            print(f"   Paragraph {i} contains {len(cites)} citations:")
            for cite in cites:
                print(f"     - {cite.get_text()}")

    # Demonstrate XPath-like queries
    print(f"\n8. SPECIFIC DATA EXTRACTION:")

    # Extract judge names
    judges = soup.find_all('judge')
    print(f"   Judges presiding:")
    for judge in judges:
        print(f"     - {judge.get_text()}")

    # Extract court and citation
    court = soup.find('court')
    citation = soup.find('neutralCitation')
    print(f"   Court: {court.get_text() if court else 'Not found'}")
    print(f"   Citation: {citation.get_text() if citation else 'Not found'}")


if __name__ == "__main__":
    demonstrate_legaldocml_parsing()