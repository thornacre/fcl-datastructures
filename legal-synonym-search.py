"""
Legal Synonym Search

Implements synonym matching for legal terminology to help litigants in person
find relevant judgments regardless of their specific phrasing.

Maps common/layperson terms to formal legal terminology and vice versa.

Author: Thornacre
Version: 1.0.0
"""

from typing import List, Dict, Set, Optional
from dataclasses import dataclass, field


@dataclass
class SynonymMatch:
    """Represents a matched synonym with its canonical term."""
    original: str
    canonical: str
    synonyms: Set[str]
    category: str


class LegalSynonymMatcher:
    """
    Handles synonym expansion for legal terminology.

    Enables litigants in person to find relevant judgments using
    everyday language that maps to formal legal terms.

    Example:
        >>> matcher = LegalSynonymMatcher()
        >>> expanded = matcher.expand_query("I need help with child custody")
        >>> print(expanded.all_terms)
        {'custody', 'child arrangements', 'residence', 'contact order', ...}
    """

    # Legal synonym groups: canonical_term -> set of equivalent terms
    # Organized by legal domain
    SYNONYM_GROUPS: Dict[str, Dict[str, Set[str]]] = {
        "family_law": {
            "child arrangements": {
                "custody", "child custody", "access", "visitation",
                "residence order", "contact order", "parental responsibility",
                "living arrangements", "care arrangements", "time with children"
            },
            "injunction": {
                "restraining order", "non-molestation order", "protection order",
                "occupation order", "exclusion order", "harassment order",
                "court order protection", "stay away order", "no contact order"
            },
            "divorce": {
                "dissolution", "marriage dissolution", "decree nisi",
                "decree absolute", "divorce proceedings", "marriage breakdown",
                "separation", "legal separation", "judicial separation"
            },
            "maintenance": {
                "alimony", "spousal support", "spousal maintenance",
                "financial support", "periodical payments", "child support",
                "child maintenance", "cms", "child maintenance service"
            },
            "ancillary relief": {
                "financial remedy", "financial settlement", "divorce settlement",
                "asset division", "property settlement", "financial order",
                "lump sum order", "pension sharing", "clean break"
            },
            "parental alienation": {
                "parent alienation", "alienating behaviour", "hostile parenting",
                "implacable hostility", "parental manipulation",
                "turning child against parent", "poisoning relationship"
            }
        },
        "civil_law": {
            "claimant": {
                "plaintiff", "applicant", "petitioner", "complainant",
                "injured party", "aggrieved party", "pursuer"
            },
            "defendant": {
                "respondent", "accused", "defending party", "alleged wrongdoer"
            },
            "negligence": {
                "breach of duty", "failure of care", "carelessness",
                "duty of care breach", "professional negligence", "clinical negligence",
                "medical negligence"
            },
            "damages": {
                "compensation", "monetary award", "financial compensation",
                "pecuniary loss", "non-pecuniary loss", "general damages",
                "special damages", "exemplary damages", "aggravated damages"
            },
            "limitation period": {
                "time limit", "statute of limitations", "limitation act",
                "time bar", "deadline for claim", "time to sue"
            },
            "small claims": {
                "small claims track", "small claims court", "minor claims",
                "low value claims", "fast track", "multi-track"
            }
        },
        "housing_law": {
            "eviction": {
                "possession", "possession proceedings", "notice to quit",
                "section 21", "section 8", "no fault eviction",
                "possession order", "bailiff eviction", "unlawful eviction"
            },
            "tenant": {
                "renter", "lessee", "occupier", "lodger", "licensee",
                "assured tenant", "secure tenant", "protected tenant"
            },
            "landlord": {
                "lessor", "property owner", "freeholder", "housing association",
                "social landlord", "private landlord", "letting agent"
            },
            "disrepair": {
                "housing disrepair", "property condition", "damp", "mould",
                "structural defects", "unfitness", "hazard", "health and safety"
            },
            "deposit": {
                "tenancy deposit", "security deposit", "deposit protection",
                "deposit scheme", "dps", "tds", "mydeposits"
            }
        },
        "employment_law": {
            "unfair dismissal": {
                "wrongful termination", "wrongful dismissal", "sacked unfairly",
                "fired unfairly", "unjust dismissal", "constructive dismissal"
            },
            "redundancy": {
                "layoff", "laid off", "job loss", "redundant",
                "redundancy payment", "redundancy selection"
            },
            "discrimination": {
                "workplace discrimination", "employment discrimination",
                "protected characteristic", "equal treatment", "victimisation",
                "harassment at work", "bullying"
            },
            "tribunal": {
                "employment tribunal", "et", "industrial tribunal",
                "labour court", "employment appeal tribunal", "eat"
            },
            "grievance": {
                "workplace complaint", "formal complaint", "grievance procedure",
                "disciplinary", "disciplinary procedure", "acas"
            }
        },
        "criminal_law": {
            "bail": {
                "conditional bail", "unconditional bail", "bail conditions",
                "remand", "custody", "release on bail", "bail hearing"
            },
            "sentence": {
                "sentencing", "punishment", "penalty", "custodial sentence",
                "suspended sentence", "community order", "fine"
            },
            "not guilty": {
                "acquittal", "acquitted", "found innocent", "cleared",
                "discharged", "case dismissed"
            },
            "guilty": {
                "conviction", "convicted", "found guilty", "guilty plea",
                "guilty verdict"
            },
            "appeal": {
                "criminal appeal", "appeal against conviction",
                "appeal against sentence", "court of appeal", "grounds of appeal"
            }
        },
        "procedural": {
            "without prejudice": {
                "off the record", "settlement discussions", "confidential negotiations",
                "protected communications"
            },
            "disclosure": {
                "discovery", "document disclosure", "evidence disclosure",
                "witness statements", "bundle", "court bundle"
            },
            "hearing": {
                "court hearing", "trial", "court date", "listed for hearing",
                "directions hearing", "case management hearing"
            },
            "adjournment": {
                "postponement", "delay", "put off", "rescheduled",
                "stood over", "vacated"
            },
            "litigant in person": {
                "lip", "self-represented", "unrepresented", "pro se",
                "acting in person", "no lawyer", "no solicitor"
            },
            "costs": {
                "legal costs", "court costs", "solicitor fees", "barrister fees",
                "disbursements", "costs order", "no order as to costs"
            }
        }
    }

    def __init__(self, additional_synonyms: Optional[Dict[str, Dict[str, Set[str]]]] = None):
        """
        Initialize the legal synonym matcher.

        Args:
            additional_synonyms: Extra synonym groups to add to defaults
        """
        self.synonym_groups = self.SYNONYM_GROUPS.copy()
        if additional_synonyms:
            for category, terms in additional_synonyms.items():
                if category in self.synonym_groups:
                    self.synonym_groups[category].update(terms)
                else:
                    self.synonym_groups[category] = terms

        # Build reverse lookup: any term -> (canonical, category, all_synonyms)
        self._build_reverse_index()

    def _build_reverse_index(self):
        """Build reverse index for fast term lookup."""
        self.term_index: Dict[str, tuple] = {}

        for category, terms in self.synonym_groups.items():
            for canonical, synonyms in terms.items():
                # Index the canonical term
                self.term_index[canonical.lower()] = (canonical, category, synonyms)
                # Index all synonyms
                for syn in synonyms:
                    self.term_index[syn.lower()] = (canonical, category, synonyms)

    def find_matches(self, text: str) -> List[SynonymMatch]:
        """
        Find all legal terms in text and return their synonym groups.

        Args:
            text: Input text to search for legal terms

        Returns:
            List of SynonymMatch objects for found terms
        """
        text_lower = text.lower()
        matches = []
        found_canonicals = set()

        # Sort terms by length (longest first) to match phrases before words
        sorted_terms = sorted(self.term_index.keys(), key=len, reverse=True)

        for term in sorted_terms:
            if term in text_lower:
                canonical, category, synonyms = self.term_index[term]
                # Avoid duplicate matches for same canonical term
                if canonical not in found_canonicals:
                    found_canonicals.add(canonical)
                    matches.append(SynonymMatch(
                        original=term,
                        canonical=canonical,
                        synonyms=synonyms | {canonical},
                        category=category
                    ))

        return matches

    def expand_query(self, query: str) -> 'ExpandedQuery':
        """
        Expand a search query with all relevant synonyms.

        Args:
            query: Original search query

        Returns:
            ExpandedQuery with original terms and all expansions
        """
        matches = self.find_matches(query)

        all_terms = set()
        expansions = {}

        for match in matches:
            all_terms.update(match.synonyms)
            expansions[match.original] = {
                "canonical": match.canonical,
                "alternatives": match.synonyms,
                "category": match.category
            }

        # Add original query words that weren't matched
        original_words = set(query.lower().split())
        matched_terms = set()
        for match in matches:
            matched_terms.add(match.original)
        unmatched = original_words - matched_terms
        all_terms.update(unmatched)

        return ExpandedQuery(
            original=query,
            matches=matches,
            all_terms=all_terms,
            expansions=expansions
        )

    def get_category_terms(self, category: str) -> Dict[str, Set[str]]:
        """Get all terms for a specific legal category."""
        return self.synonym_groups.get(category, {})

    def suggest_alternatives(self, term: str) -> Optional[Set[str]]:
        """
        Suggest alternative terms for a given legal term.

        Args:
            term: Legal term to find alternatives for

        Returns:
            Set of alternative terms or None if not found
        """
        term_lower = term.lower()
        if term_lower in self.term_index:
            canonical, _, synonyms = self.term_index[term_lower]
            return synonyms | {canonical}
        return None


@dataclass
class ExpandedQuery:
    """Represents a query expanded with legal synonyms."""
    original: str
    matches: List[SynonymMatch]
    all_terms: Set[str]
    expansions: Dict[str, Dict]

    @property
    def has_legal_terms(self) -> bool:
        """Check if query contains recognized legal terms."""
        return len(self.matches) > 0

    @property
    def categories_matched(self) -> Set[str]:
        """Get all legal categories matched in the query."""
        return {m.category for m in self.matches}

    def to_search_terms(self, include_original: bool = True) -> List[str]:
        """
        Convert to list of search terms for database query.

        Args:
            include_original: Whether to include original query words

        Returns:
            List of terms to search for
        """
        return list(self.all_terms)

    def to_elasticsearch_should(self, field: str = "content") -> List[Dict]:
        """
        Generate Elasticsearch 'should' clauses for synonym matching.

        Args:
            field: Field to search in

        Returns:
            List of match clauses for bool query
        """
        clauses = []
        for term in self.all_terms:
            clauses.append({
                "match": {
                    field: {
                        "query": term,
                        "boost": 2.0 if term in [m.canonical for m in self.matches] else 1.0
                    }
                }
            })
        return clauses

    def to_mongodb_text_search(self) -> str:
        """
        Generate MongoDB text search string with all synonyms.

        Returns:
            Space-separated search string for $text operator
        """
        # Wrap multi-word terms in quotes
        terms = []
        for term in self.all_terms:
            if ' ' in term:
                terms.append(f'"{term}"')
            else:
                terms.append(term)
        return ' '.join(terms)


# Example usage
if __name__ == "__main__":
    matcher = LegalSynonymMatcher()

    # Example 1: Family law query
    print("=" * 70)
    print("Example 1: Family Law - Child Custody Query")
    print("=" * 70)
    query1 = "I need help with child custody and getting a restraining order"
    expanded1 = matcher.expand_query(query1)

    print(f"Original query: {query1}")
    print(f"\nLegal terms found:")
    for match in expanded1.matches:
        print(f"  • '{match.original}' → canonical: '{match.canonical}'")
        print(f"    Category: {match.category}")
        print(f"    Alternatives: {', '.join(list(match.synonyms)[:5])}...")

    print(f"\nAll search terms ({len(expanded1.all_terms)}):")
    print(f"  {expanded1.all_terms}")

    # Example 2: Housing law query
    print("\n" + "=" * 70)
    print("Example 2: Housing Law - Eviction Query")
    print("=" * 70)
    query2 = "My landlord is trying to evict me and hasn't fixed the damp"
    expanded2 = matcher.expand_query(query2)

    print(f"Original query: {query2}")
    print(f"\nMatched categories: {expanded2.categories_matched}")
    print(f"\nExpanded search terms:")
    for term in sorted(expanded2.all_terms):
        print(f"  • {term}")

    # Example 3: Employment query with suggestions
    print("\n" + "=" * 70)
    print("Example 3: Employment - Suggesting Alternatives")
    print("=" * 70)
    term = "sacked unfairly"
    alternatives = matcher.suggest_alternatives(term)
    print(f"Alternatives for '{term}':")
    for alt in sorted(alternatives or []):
        print(f"  • {alt}")

    # Example 4: Generate database queries
    print("\n" + "=" * 70)
    print("Example 4: Generated Database Queries")
    print("=" * 70)
    query4 = "divorce settlement and maintenance payments"
    expanded4 = matcher.expand_query(query4)

    print(f"Original: {query4}")
    print(f"\nMongoDB text search:")
    print(f"  {expanded4.to_mongodb_text_search()[:100]}...")

    print(f"\nElasticsearch should clauses (first 3):")
    es_clauses = expanded4.to_elasticsearch_should()
    for clause in es_clauses[:3]:
        print(f"  {clause}")
    print(f"  ... and {len(es_clauses) - 3} more clauses")

    # Example 5: Litigant in person query
    print("\n" + "=" * 70)
    print("Example 5: Self-Represented Litigant Query")
    print("=" * 70)
    query5 = "I have no solicitor and need to know about court costs"
    expanded5 = matcher.expand_query(query5)

    print(f"Original: {query5}")
    print(f"Recognized as: {[m.canonical for m in expanded5.matches]}")
    print(f"\nThis query would also find documents about:")
    for match in expanded5.matches:
        alts = list(match.synonyms - {match.original})[:3]
        print(f"  • {match.canonical}: {', '.join(alts)}")
