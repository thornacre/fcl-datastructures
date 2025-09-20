"""
NLP Token Trees for Legal Text Analysis
======================================

This module implements parse trees and token trees for natural language processing
of legal documents in Find Case Law (FCL). Provides structured representation
of legal text for advanced analysis, citation extraction, and semantic understanding.

Key FCL Use Cases:
- Parsing legal document structure (paragraphs, sections, citations)
- Extracting entity relationships and legal concepts
- Building dependency trees for complex legal sentences
- Supporting advanced search with syntactic understanding
- Enabling structured data extraction from judgment text
"""

from typing import List, Dict, Optional, Any, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
import re
from abc import ABC, abstractmethod


class TokenType(Enum):
    """Types of tokens in legal text"""
    WORD = "word"
    PUNCTUATION = "punctuation"
    CITATION = "citation"
    STATUTE = "statute"
    CASE_NAME = "case_name"
    COURT = "court"
    DATE = "date"
    PARAGRAPH_REF = "paragraph_ref"
    SECTION_REF = "section_ref"
    LEGAL_CONCEPT = "legal_concept"
    ENTITY = "entity"


class TreeNodeType(Enum):
    """Types of parse tree nodes"""
    ROOT = "root"
    SENTENCE = "sentence"
    PHRASE = "phrase"
    CLAUSE = "clause"
    NOUN_PHRASE = "noun_phrase"
    VERB_PHRASE = "verb_phrase"
    PREPOSITIONAL_PHRASE = "prepositional_phrase"
    CITATION_PHRASE = "citation_phrase"
    LEGAL_PRINCIPLE = "legal_principle"


@dataclass
class Token:
    """Represents a single token in legal text"""
    text: str
    token_type: TokenType
    position: int
    lemma: Optional[str] = None
    pos_tag: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return f"{self.text} ({self.token_type.value})"


@dataclass
class ParseTreeNode:
    """Node in a parse tree for legal text structure"""
    node_type: TreeNodeType
    content: str
    tokens: List[Token] = field(default_factory=list)
    children: List['ParseTreeNode'] = field(default_factory=list)
    parent: Optional['ParseTreeNode'] = None
    attributes: Dict[str, Any] = field(default_factory=dict)
    span: Tuple[int, int] = (0, 0)  # Start and end positions

    def add_child(self, child: 'ParseTreeNode') -> None:
        """Add a child node"""
        child.parent = self
        self.children.append(child)

    def get_text(self) -> str:
        """Get the text content of this node and its children"""
        if self.tokens:
            return " ".join(token.text for token in self.tokens)
        elif self.children:
            return " ".join(child.get_text() for child in self.children)
        else:
            return self.content

    def find_nodes_by_type(self, node_type: TreeNodeType) -> List['ParseTreeNode']:
        """Find all descendant nodes of a specific type"""
        result = []
        if self.node_type == node_type:
            result.append(self)

        for child in self.children:
            result.extend(child.find_nodes_by_type(node_type))

        return result

    def get_citations(self) -> List[Token]:
        """Extract all citation tokens from this node"""
        citations = []

        # Check direct tokens
        for token in self.tokens:
            if token.token_type == TokenType.CITATION:
                citations.append(token)

        # Check children
        for child in self.children:
            citations.extend(child.get_citations())

        return citations

    def __str__(self) -> str:
        return f"{self.node_type.value}: {self.content[:50]}..."


class LegalTokenizer:
    """
    Specialized tokenizer for legal text that recognizes legal-specific patterns.
    """

    def __init__(self):
        # Legal citation patterns
        self.citation_patterns = [
            r'\[(\d{4})\]\s+([A-Z]{2,})\s+(\d+)',  # [2023] UKSC 15
            r'\[(\d{4})\]\s+([A-Z]{2,})\s+([A-Za-z]+)\s+(\d+)',  # [2023] EWCA Civ 892
            r'\((\d{4})\)\s+([A-Z]{2,})\s+(\d+)',  # (2023) AC 123
            r'(\d+)\s+([A-Z]{2,})\s+(\d+)',  # 2023 AC 123
        ]

        # Court patterns
        self.court_patterns = [
            r'\b(Supreme Court|House of Lords|Court of Appeal|High Court)\b',
            r'\b(UKSC|UKHL|EWCA|EWHC|UKUT|UKFTT)\b',
            r'\b(Divisional Court|Crown Court|Magistrates[\'']?\s*Court)\b',
        ]

        # Legal concept patterns
        self.legal_concepts = {
            'negligence', 'tort', 'contract', 'breach', 'damages', 'liability',
            'jurisdiction', 'statutory', 'common law', 'precedent', 'ratio',
            'obiter', 'judgment', 'appeal', 'cross-appeal', 'permission',
            'judicial review', 'human rights', 'constitutional', 'administrative',
            'procedural', 'substantive', 'reasonable', 'proportionate'
        }

        # Paragraph and section reference patterns
        self.ref_patterns = [
            r'\[(\d+)\]',  # [123]
            r'para(?:graph)?\s*(\d+)',  # paragraph 123
            r'section\s*(\d+)',  # section 123
            r's\.?\s*(\d+)',  # s. 123 or s 123
        ]

    def tokenize(self, text: str) -> List[Token]:
        """Tokenize legal text into structured tokens"""
        tokens = []
        position = 0

        # Simple word-based tokenization with legal pattern recognition
        words = re.findall(r'\S+', text)

        for word in words:
            token_type = self._classify_token(word, text, position)

            # Create base token
            token = Token(
                text=word,
                token_type=token_type,
                position=position,
                lemma=self._get_lemma(word),
                pos_tag=self._get_pos_tag(word)
            )

            # Add metadata for specific token types
            if token_type == TokenType.CITATION:
                token.metadata['citation_type'] = self._get_citation_type(word)
            elif token_type == TokenType.COURT:
                token.metadata['court_level'] = self._get_court_level(word)

            tokens.append(token)
            position += 1

        return tokens

    def _classify_token(self, word: str, full_text: str, position: int) -> TokenType:
        """Classify a token based on legal text patterns"""
        # Remove punctuation for classification
        clean_word = re.sub(r'[^\w\s]', '', word).lower()

        # Check for citations
        for pattern in self.citation_patterns:
            if re.search(pattern, word):
                return TokenType.CITATION

        # Check for courts
        for pattern in self.court_patterns:
            if re.search(pattern, word, re.IGNORECASE):
                return TokenType.COURT

        # Check for legal concepts
        if clean_word in self.legal_concepts:
            return TokenType.LEGAL_CONCEPT

        # Check for references
        for pattern in self.ref_patterns:
            if re.search(pattern, word, re.IGNORECASE):
                return TokenType.PARAGRAPH_REF

        # Check for dates
        if re.match(r'\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}', word):
            return TokenType.DATE

        # Check for punctuation
        if re.match(r'^[^\w\s]+$', word):
            return TokenType.PUNCTUATION

        # Default to word
        return TokenType.WORD

    def _get_lemma(self, word: str) -> str:
        """Get lemmatized form of word (simplified)"""
        # Simple lemmatization for common legal terms
        lemma_map = {
            'courts': 'court',
            'judgments': 'judgment',
            'appeals': 'appeal',
            'applications': 'application',
            'decisions': 'decision',
            'orders': 'order',
            'proceedings': 'proceeding'
        }
        return lemma_map.get(word.lower(), word.lower())

    def _get_pos_tag(self, word: str) -> str:
        """Get part-of-speech tag (simplified)"""
        # Simple POS tagging based on patterns
        if re.match(r'^[A-Z][a-z]+$', word):
            return 'NOUN'
        elif word.lower() in {'the', 'a', 'an'}:
            return 'DET'
        elif word.lower() in {'and', 'or', 'but'}:
            return 'CONJ'
        elif re.match(r'^[^\w\s]+$', word):
            return 'PUNCT'
        else:
            return 'WORD'

    def _get_citation_type(self, word: str) -> str:
        """Determine the type of citation"""
        if 'UKSC' in word or 'UKHL' in word:
            return 'supreme_court'
        elif 'EWCA' in word:
            return 'court_of_appeal'
        elif 'EWHC' in word:
            return 'high_court'
        elif 'UKUT' in word or 'UKFTT' in word:
            return 'tribunal'
        else:
            return 'other'

    def _get_court_level(self, word: str) -> str:
        """Determine court hierarchy level"""
        word_upper = word.upper()
        if any(court in word_upper for court in ['UKSC', 'SUPREME']):
            return 'supreme'
        elif any(court in word_upper for court in ['UKHL', 'LORDS']):
            return 'lords'
        elif any(court in word_upper for court in ['EWCA', 'APPEAL']):
            return 'appeal'
        elif any(court in word_upper for court in ['EWHC', 'HIGH']):
            return 'high'
        else:
            return 'lower'


class LegalParseTree:
    """
    Parse tree for legal document structure analysis.
    Builds hierarchical representation of legal text.
    """

    def __init__(self):
        self.root = ParseTreeNode(TreeNodeType.ROOT, "Legal Document")
        self.tokenizer = LegalTokenizer()

    def parse_document(self, text: str) -> ParseTreeNode:
        """Parse a legal document into a structured tree"""
        # Reset root
        self.root = ParseTreeNode(TreeNodeType.ROOT, "Legal Document")

        # Split into sentences
        sentences = self._split_sentences(text)

        for i, sentence_text in enumerate(sentences):
            sentence_node = self._parse_sentence(sentence_text, i)
            self.root.add_child(sentence_node)

        return self.root

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences, handling legal text patterns"""
        # Enhanced sentence splitting for legal text
        # Handle abbreviations and citations properly
        sentences = []

        # Simple sentence splitting (can be enhanced with more sophisticated rules)
        parts = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)

        for part in parts:
            if part.strip():
                sentences.append(part.strip())

        return sentences

    def _parse_sentence(self, sentence: str, sentence_id: int) -> ParseTreeNode:
        """Parse a single sentence into structured components"""
        sentence_node = ParseTreeNode(
            TreeNodeType.SENTENCE,
            sentence,
            span=(0, len(sentence))
        )

        # Tokenize the sentence
        tokens = self.tokenizer.tokenize(sentence)
        sentence_node.tokens = tokens

        # Extract structured components
        self._extract_phrases(sentence_node, tokens)

        return sentence_node

    def _extract_phrases(self, sentence_node: ParseTreeNode, tokens: List[Token]) -> None:
        """Extract phrases and legal constructs from tokens"""
        i = 0
        while i < len(tokens):
            # Try to identify citation phrases
            citation_phrase = self._extract_citation_phrase(tokens, i)
            if citation_phrase:
                sentence_node.add_child(citation_phrase)
                i += len(citation_phrase.tokens)
                continue

            # Try to identify legal principle phrases
            legal_phrase = self._extract_legal_phrase(tokens, i)
            if legal_phrase:
                sentence_node.add_child(legal_phrase)
                i += len(legal_phrase.tokens)
                continue

            # Default: create word-level nodes
            if tokens[i].token_type != TokenType.PUNCTUATION:
                word_node = ParseTreeNode(
                    TreeNodeType.PHRASE,
                    tokens[i].text,
                    tokens=[tokens[i]]
                )
                sentence_node.add_child(word_node)

            i += 1

    def _extract_citation_phrase(self, tokens: List[Token], start: int) -> Optional[ParseTreeNode]:
        """Extract citation phrases from token sequence"""
        citation_tokens = []
        i = start

        # Look for citation patterns
        while i < len(tokens) and i < start + 10:  # Limit search window
            token = tokens[i]

            if token.token_type == TokenType.CITATION:
                citation_tokens.append(token)
                i += 1

                # Check for additional citation components
                while i < len(tokens) and i < start + 10:
                    next_token = tokens[i]
                    if (next_token.token_type in [TokenType.WORD, TokenType.PUNCTUATION] and
                        any(pattern in next_token.text for pattern in ['v', 'vs', '(', ')'])):
                        citation_tokens.append(next_token)
                        i += 1
                    else:
                        break

                if citation_tokens:
                    content = " ".join(t.text for t in citation_tokens)
                    return ParseTreeNode(
                        TreeNodeType.CITATION_PHRASE,
                        content,
                        tokens=citation_tokens
                    )
            i += 1

        return None

    def _extract_legal_phrase(self, tokens: List[Token], start: int) -> Optional[ParseTreeNode]:
        """Extract legal principle or concept phrases"""
        legal_tokens = []
        i = start

        # Look for sequences of legal concepts
        while i < len(tokens) and i < start + 8:
            token = tokens[i]

            if token.token_type == TokenType.LEGAL_CONCEPT:
                legal_tokens.append(token)

                # Include surrounding context words
                j = i + 1
                while j < len(tokens) and j < start + 8:
                    next_token = tokens[j]
                    if (next_token.token_type in [TokenType.WORD, TokenType.LEGAL_CONCEPT] and
                        next_token.text.lower() not in ['the', 'a', 'an']):
                        legal_tokens.append(next_token)
                        j += 1
                    else:
                        break

                if len(legal_tokens) >= 2:  # Require at least 2 tokens for a phrase
                    content = " ".join(t.text for t in legal_tokens)
                    return ParseTreeNode(
                        TreeNodeType.LEGAL_PRINCIPLE,
                        content,
                        tokens=legal_tokens
                    )

            i += 1

        return None

    def extract_citations(self) -> List[str]:
        """Extract all citations from the parse tree"""
        citation_nodes = self.root.find_nodes_by_type(TreeNodeType.CITATION_PHRASE)
        return [node.get_text() for node in citation_nodes]

    def extract_legal_concepts(self) -> List[str]:
        """Extract all legal concepts and principles"""
        legal_nodes = self.root.find_nodes_by_type(TreeNodeType.LEGAL_PRINCIPLE)
        return [node.get_text() for node in legal_nodes]

    def get_statistics(self) -> Dict[str, Any]:
        """Get parse tree statistics"""
        total_nodes = self._count_nodes(self.root)
        sentences = self.root.find_nodes_by_type(TreeNodeType.SENTENCE)
        citations = self.root.find_nodes_by_type(TreeNodeType.CITATION_PHRASE)
        legal_principles = self.root.find_nodes_by_type(TreeNodeType.LEGAL_PRINCIPLE)

        # Token type distribution
        all_tokens = []
        for sentence in sentences:
            all_tokens.extend(sentence.tokens)

        token_types = {}
        for token in all_tokens:
            token_types[token.token_type.value] = token_types.get(token.token_type.value, 0) + 1

        return {
            'total_nodes': total_nodes,
            'sentences': len(sentences),
            'citations': len(citations),
            'legal_principles': len(legal_principles),
            'total_tokens': len(all_tokens),
            'token_type_distribution': token_types
        }

    def _count_nodes(self, node: ParseTreeNode) -> int:
        """Recursively count all nodes in tree"""
        count = 1
        for child in node.children:
            count += self._count_nodes(child)
        return count

    def to_dict(self) -> Dict[str, Any]:
        """Convert parse tree to dictionary representation"""
        return self._node_to_dict(self.root)

    def _node_to_dict(self, node: ParseTreeNode) -> Dict[str, Any]:
        """Convert a node to dictionary format"""
        return {
            'type': node.node_type.value,
            'content': node.content,
            'tokens': [
                {
                    'text': token.text,
                    'type': token.token_type.value,
                    'position': token.position,
                    'lemma': token.lemma,
                    'pos_tag': token.pos_tag
                }
                for token in node.tokens
            ],
            'children': [self._node_to_dict(child) for child in node.children],
            'attributes': node.attributes,
            'span': node.span
        }


class LegalEntityExtractor:
    """
    Specialized entity extractor for legal documents.
    Identifies and classifies legal entities, relationships, and concepts.
    """

    def __init__(self):
        self.case_name_patterns = [
            r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+v\.?\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
            r'R\s+v\.?\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
            r'Re\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
        ]

        self.statute_patterns = [
            r'([A-Z][a-z\s]+Act)\s+(\d{4})',
            r'(Human Rights Act|Data Protection Act|Companies Act)',
            r'section\s+(\d+)\s+of\s+the\s+([A-Z][a-z\s]+Act\s+\d{4})',
        ]

    def extract_entities(self, parse_tree: LegalParseTree) -> Dict[str, List[Dict[str, Any]]]:
        """Extract legal entities from parsed document"""
        entities = {
            'case_names': [],
            'statutes': [],
            'courts': [],
            'citations': [],
            'parties': [],
            'judges': []
        }

        # Extract from each sentence
        sentences = parse_tree.root.find_nodes_by_type(TreeNodeType.SENTENCE)

        for sentence in sentences:
            sentence_text = sentence.get_text()

            # Extract case names
            for pattern in self.case_name_patterns:
                matches = re.finditer(pattern, sentence_text)
                for match in matches:
                    entities['case_names'].append({
                        'text': match.group(0),
                        'parties': match.groups(),
                        'span': match.span()
                    })

            # Extract statutes
            for pattern in self.statute_patterns:
                matches = re.finditer(pattern, sentence_text)
                for match in matches:
                    entities['statutes'].append({
                        'text': match.group(0),
                        'components': match.groups(),
                        'span': match.span()
                    })

            # Extract court mentions
            court_tokens = [token for token in sentence.tokens if token.token_type == TokenType.COURT]
            for token in court_tokens:
                entities['courts'].append({
                    'text': token.text,
                    'court_level': token.metadata.get('court_level', 'unknown'),
                    'position': token.position
                })

            # Extract citations
            citation_tokens = [token for token in sentence.tokens if token.token_type == TokenType.CITATION]
            for token in citation_tokens:
                entities['citations'].append({
                    'text': token.text,
                    'citation_type': token.metadata.get('citation_type', 'unknown'),
                    'position': token.position
                })

        return entities


def demonstrate_nlp_token_trees():
    """Demonstrate NLP token trees with UK legal text."""

    print("=== NLP Token Trees for Legal Text Analysis Demo ===\n")

    # Sample UK legal judgment text
    legal_text = """
    In the case of Smith v Secretary of State for Work and Pensions [2023] EWCA Civ 892,
    the Court of Appeal considered the application of judicial review principles to
    administrative decisions. The appellant argued that the respondent had failed to
    follow proper procedural requirements under section 12 of the Social Security Act 1998.

    Lord Justice Brown held that the test for negligence established in Donoghue v Stevenson
    [1932] AC 562 remains the cornerstone of tort law. The court found that reasonable
    care had not been exercised in this case.

    The Supreme Court in R (Miller) v Prime Minister [2023] UKSC 15 established important
    constitutional principles regarding parliamentary sovereignty and the separation of powers.
    """

    # 1. Tokenization Demo
    print("1. LEGAL TEXT TOKENIZATION:")
    tokenizer = LegalTokenizer()

    sample_sentence = "Smith v Secretary of State [2023] EWCA Civ 892 applied negligence principles."
    tokens = tokenizer.tokenize(sample_sentence)

    print(f"   Sample sentence: {sample_sentence}")
    print(f"   Tokens:")
    for token in tokens:
        print(f"     {token.text:15} | {token.token_type.value:12} | {token.lemma}")

    # 2. Parse Tree Construction
    print(f"\n2. PARSE TREE CONSTRUCTION:")
    parser = LegalParseTree()
    parse_tree = parser.parse_document(legal_text)

    print(f"   Document parsed successfully")
    stats = parser.get_statistics()
    print(f"   Statistics:")
    for key, value in stats.items():
        if key != 'token_type_distribution':
            print(f"     {key}: {value}")

    print(f"   Token type distribution:")
    for token_type, count in stats['token_type_distribution'].items():
        print(f"     {token_type}: {count}")

    # 3. Citation Extraction
    print(f"\n3. CITATION EXTRACTION:")
    citations = parser.extract_citations()
    print(f"   Found {len(citations)} citations:")
    for i, citation in enumerate(citations, 1):
        print(f"     {i}. {citation}")

    # 4. Legal Concept Extraction
    print(f"\n4. LEGAL CONCEPT EXTRACTION:")
    legal_concepts = parser.extract_legal_concepts()
    print(f"   Found {len(legal_concepts)} legal concepts:")
    for i, concept in enumerate(legal_concepts, 1):
        print(f"     {i}. {concept}")

    # 5. Entity Extraction
    print(f"\n5. LEGAL ENTITY EXTRACTION:")
    extractor = LegalEntityExtractor()
    entities = extractor.extract_entities(parser)

    for entity_type, entity_list in entities.items():
        if entity_list:
            print(f"   {entity_type.replace('_', ' ').title()}:")
            for i, entity in enumerate(entity_list, 1):
                print(f"     {i}. {entity['text']}")

    # 6. Parse Tree Analysis
    print(f"\n6. PARSE TREE STRUCTURE ANALYSIS:")
    sentences = parse_tree.find_nodes_by_type(TreeNodeType.SENTENCE)
    print(f"   Analyzed {len(sentences)} sentences:")

    for i, sentence in enumerate(sentences[:3], 1):  # Show first 3 sentences
        print(f"\n   Sentence {i}: {sentence.content[:60]}...")
        print(f"     Child nodes: {len(sentence.children)}")

        # Show citation and legal phrase children
        for child in sentence.children:
            if child.node_type in [TreeNodeType.CITATION_PHRASE, TreeNodeType.LEGAL_PRINCIPLE]:
                print(f"       {child.node_type.value}: {child.content}")

    # 7. Token Dependency Analysis
    print(f"\n7. TOKEN ANALYSIS BY TYPE:")
    all_sentences = parse_tree.find_nodes_by_type(TreeNodeType.SENTENCE)
    token_examples = {}

    for sentence in all_sentences:
        for token in sentence.tokens:
            if token.token_type not in token_examples:
                token_examples[token.token_type] = []
            if len(token_examples[token.token_type]) < 3:  # Limit examples
                token_examples[token.token_type].append(token.text)

    for token_type, examples in token_examples.items():
        if examples:
            print(f"   {token_type.value}: {', '.join(examples)}")

    # 8. Tree Serialization
    print(f"\n8. PARSE TREE SERIALIZATION:")
    tree_dict = parser.to_dict()
    print(f"   Tree structure serialized to dictionary")
    print(f"   Root node has {len(tree_dict['children'])} child sentences")
    print(f"   First sentence has {len(tree_dict['children'][0]['tokens'])} tokens")

    # 9. Advanced Pattern Matching
    print(f"\n9. ADVANCED PATTERN ANALYSIS:")

    # Find sentences with multiple citations
    multi_citation_sentences = []
    for sentence in sentences:
        citation_count = len(sentence.get_citations())
        if citation_count > 1:
            multi_citation_sentences.append((sentence, citation_count))

    if multi_citation_sentences:
        print(f"   Sentences with multiple citations:")
        for sentence, count in multi_citation_sentences:
            print(f"     {count} citations: {sentence.content[:80]}...")

    # Find legal principle chains
    principle_sentences = []
    for sentence in sentences:
        legal_nodes = sentence.find_nodes_by_type(TreeNodeType.LEGAL_PRINCIPLE)
        if legal_nodes:
            principle_sentences.append((sentence, len(legal_nodes)))

    if principle_sentences:
        print(f"\n   Sentences with legal principles:")
        for sentence, count in principle_sentences:
            print(f"     {count} principles: {sentence.content[:80]}...")

    return {
        'parser': parser,
        'parse_tree': parse_tree,
        'entities': entities,
        'statistics': stats,
        'examples': {
            'tokens': tokens,
            'citations': citations,
            'legal_concepts': legal_concepts
        }
    }


if __name__ == "__main__":
    demonstrate_nlp_token_trees()