# LegalDocML (Akoma Ntoso) Schema Documentation

**Version:** 1.0.0
**Author:** Thornacre
**Purpose:** Provides data re-users clarity on XML output structure for court judgments

## References

- [Akoma Ntoso](http://www.akomantoso.org/)
- [OASIS LegalDocML TC](https://www.oasis-open.org/committees/legaldocml/)
- [UK National Archives Caselaw](https://caselaw.nationalarchives.gov.uk/)

---

## 1. Namespace Declarations

Root element must declare the Akoma Ntoso namespace:

```xml
<akomaNtoso
    xmlns="http://docs.oasis-open.org/legaldocml/ns/akn/3.0"
    xmlns:uk="https://caselaw.nationalarchives.gov.uk/akn">
```

---

## 2. Document Structure Hierarchy

```
<akomaNtoso>
└── <judgment>                          # Root document type
    ├── <meta>                          # Metadata section
    │   ├── <identification>            # Document identifiers (FRBR)
    │   ├── <publication>               # Publication info
    │   ├── <classification>            # Subject classification
    │   ├── <lifecycle>                 # Document events
    │   ├── <references>                # Citations & references
    │   └── <proprietary>               # Custom metadata
    ├── <header>                        # Judgment header
    │   └── <p>                         # Court name, case number
    ├── <judgmentBody>                  # Main content
    │   ├── <introduction>              # Parties, background
    │   ├── <background>                # Case history
    │   ├── <arguments>                 # Legal arguments
    │   ├── <remedies>                  # Orders made
    │   └── <decision>                  # Judgment decision
    ├── <conclusions>                   # Final orders
    └── <attachments>                   # Appendices
```

---

## 3. Key Elements Reference

### `<identification>`

**Purpose:** Uniquely identifies the document using FRBR (Functional Requirements for Bibliographic Records)

**Children:**
| Element | Description |
|---------|-------------|
| `<FRBRWork>` | Abstract work identifier |
| `<FRBRExpression>` | Specific version/language |
| `<FRBRManifestation>` | Physical format |

**Example:**
```xml
<identification source="#tna">
    <FRBRWork>
        <FRBRthis value="/uk/judgment/ewca/civ/2024/123"/>
        <FRBRuri value="/uk/judgment/ewca/civ/2024/123"/>
        <FRBRdate date="2024-03-15" name="judgment"/>
        <FRBRauthor href="#ewca"/>
        <FRBRcountry value="uk"/>
    </FRBRWork>
    <FRBRExpression>
        <FRBRthis value="/uk/judgment/ewca/civ/2024/123/eng"/>
        <FRBRuri value="/uk/judgment/ewca/civ/2024/123/eng"/>
        <FRBRdate date="2024-03-15" name="judgment"/>
        <FRBRauthor href="#ewca"/>
        <FRBRlanguage language="eng"/>
    </FRBRExpression>
</identification>
```

---

### `<classification>`

**Purpose:** Subject matter and legal topic categorization

**Children:**
| Element | Description |
|---------|-------------|
| `<keyword>` | Subject keywords with controlled vocabulary |

**Example:**
```xml
<classification source="#tna">
    <keyword value="family" showAs="Family Law" dictionary="#topics"/>
    <keyword value="children" showAs="Children" dictionary="#topics"/>
    <keyword value="custody" showAs="Child Arrangements" dictionary="#topics"/>
</classification>
```

---

### `<references>`

**Purpose:** Citations to legislation, cases, and authorities

**Children:**
| Element | Description |
|---------|-------------|
| `<TLCOrganization>` | Courts, bodies |
| `<TLCPerson>` | Judges, parties |
| `<TLCReference>` | Case citations |
| `<passiveRef>` | Legislation references |

**Example:**
```xml
<references source="#tna">
    <TLCOrganization eId="ewca" href="/uk/court/ewca" showAs="Court of Appeal"/>
    <TLCPerson eId="judge-smith" href="/uk/judge/smith" showAs="Lord Justice Smith"/>
    <TLCReference eId="ref-1" href="/uk/judgment/uksc/2020/45" showAs="[2020] UKSC 45"/>
    <passiveRef href="/uk/act/2014/6" showAs="Children and Families Act 2014"/>
</references>
```

---

### `<judgmentBody>`

**Purpose:** Contains the substantive content of the judgment

**Attributes:**
| Attribute | Description |
|-----------|-------------|
| `eId` | Element identifier |

**Children:**
| Element | Description |
|---------|-------------|
| `<introduction>` | Opening, parties, representation |
| `<background>` | Factual and procedural background |
| `<arguments>` | Legal submissions |
| `<remedies>` | Orders sought |
| `<decision>` | Court's analysis and decision |

---

### `<party>`

**Purpose:** Identifies parties to the proceedings

**Attributes:**
| Attribute | Description |
|-----------|-------------|
| `refersTo` | Reference to TLCPerson |
| `as` | Role (appellant, respondent, claimant, defendant) |

**Example:**
```xml
<party refersTo="#party-1" as="#appellant">John Smith</party>
<party refersTo="#party-2" as="#respondent">Jane Smith</party>
```

---

### `<paragraph>`

**Purpose:** Numbered paragraph within judgment body

**Attributes:**
| Attribute | Description |
|-----------|-------------|
| `eId` | Unique paragraph identifier (e.g., "para_1") |

**Children:**
| Element | Description |
|---------|-------------|
| `<num>` | Paragraph number |
| `<content>` | Paragraph content container |
| `<p>` | Text content |

**Example:**
```xml
<paragraph eId="para_1">
    <num>1.</num>
    <content>
        <p>This is an appeal against the order of HHJ Smith dated
        15 January 2024 concerning child arrangements.</p>
    </content>
</paragraph>
```

---

### `<ref>`

**Purpose:** Inline citation reference

**Attributes:**
| Attribute | Description |
|-----------|-------------|
| `href` | URI to cited document |

**Example:**
```xml
<ref href="/uk/act/1989/41">Children Act 1989</ref>
<ref href="/uk/judgment/uksc/2020/45">[2020] UKSC 45</ref>
```

---

## 4. Attribute Conventions

| Attribute | Format | Description |
|-----------|--------|-------------|
| `eId` | `elementType_number` | Element identifier, unique within document (e.g., `para_1`, `sec_2`) |
| `href` | `/jurisdiction/type/year/number` | URI reference to external or internal resource |
| `refersTo` | `#elementId` | Reference to element defined in `<references>` |
| `value` | varies | Attribute value for structured data |
| `showAs` | text | Human-readable display text |
| `dictionary` | URI | Controlled vocabulary reference |

---

## 5. Common Patterns

### Neutral Citation
```xml
<neutralCitation>[2024] EWCA Civ 123</neutralCitation>
```

### Court Reference
```xml
<courtType>Court of Appeal (Civil Division)</courtType>
```

### Date Handling
```xml
<docDate date="2024-03-15">15 March 2024</docDate>
```

### Judge Reference
```xml
<judge refersTo="#judge-smith">Lord Justice Smith</judge>
```

---

## 6. Court Identifiers

| Court | Identifier | Full Name |
|-------|------------|-----------|
| UKSC | `uksc` | Supreme Court |
| EWCA Civ | `ewca/civ` | Court of Appeal (Civil Division) |
| EWCA Crim | `ewca/crim` | Court of Appeal (Criminal Division) |
| EWHC QB | `ewhc/qb` | High Court (King's Bench Division) |
| EWHC Ch | `ewhc/ch` | High Court (Chancery Division) |
| EWHC Fam | `ewhc/fam` | High Court (Family Division) |
| EWHC Admin | `ewhc/admin` | High Court (Administrative Court) |
| UKUT | `ukut` | Upper Tribunal |
| UKFTT | `ukftt` | First-tier Tribunal |

---

## 7. Party Roles

| Role | Value | Description |
|------|-------|-------------|
| Appellant | `appellant` | Party appealing a decision |
| Respondent | `respondent` | Party responding to an appeal |
| Claimant | `claimant` | Party bringing a civil claim |
| Defendant | `defendant` | Party defending against a claim |
| Applicant | `applicant` | Party making an application |
| Petitioner | `petitioner` | Party filing a petition |
| Intervener | `intervener` | Third party permitted to participate |

---

## 8. Validation Requirements

### Required Elements

For a valid judgment document:

- [x] `<akomaNtoso>` root with namespace
- [x] `<judgment>` document type
- [x] `<meta>` with `<identification>`
- [x] `<judgmentBody>` with at least one `<paragraph>`

### Recommended Elements

- `<classification>` for searchability
- `<references>` for citation linking
- `<header>` for display purposes

---

## 9. Encoding & Format

| Property | Value |
|----------|-------|
| Encoding | UTF-8 |
| Line endings | LF (Unix-style) |
| Indentation | 2 spaces (recommended) |
| Max line length | 120 characters (recommended) |

---

## 10. Complete Example

```xml
<?xml version="1.0" encoding="UTF-8"?>
<akomaNtoso xmlns="http://docs.oasis-open.org/legaldocml/ns/akn/3.0"
            xmlns:uk="https://caselaw.nationalarchives.gov.uk/akn">
  <judgment name="judgment">
    <meta>
      <identification source="#tna">
        <FRBRWork>
          <FRBRthis value="/uk/judgment/ewca/civ/2024/123"/>
          <FRBRuri value="/uk/judgment/ewca/civ/2024/123"/>
          <FRBRdate date="2024-03-15" name="judgment"/>
          <FRBRauthor href="#ewca-civ"/>
          <FRBRcountry value="uk"/>
        </FRBRWork>
        <FRBRExpression>
          <FRBRthis value="/uk/judgment/ewca/civ/2024/123/eng"/>
          <FRBRuri value="/uk/judgment/ewca/civ/2024/123/eng"/>
          <FRBRdate date="2024-03-15" name="judgment"/>
          <FRBRauthor href="#ewca-civ"/>
          <FRBRlanguage language="eng"/>
        </FRBRExpression>
      </identification>

      <classification source="#tna">
        <keyword value="family-law" showAs="Family Law" dictionary="#topics"/>
        <keyword value="children" showAs="Children" dictionary="#topics"/>
        <keyword value="child-arrangements" showAs="Child Arrangements" dictionary="#topics"/>
      </classification>

      <references source="#tna">
        <TLCOrganization eId="ewca-civ" href="/uk/court/ewca/civ"
                         showAs="Court of Appeal (Civil Division)"/>
        <TLCPerson eId="judge-williams" href="/uk/judge/williams"
                   showAs="Lord Justice Williams"/>
        <TLCPerson eId="party-smith" href="#" showAs="Smith"/>
        <TLCPerson eId="party-jones" href="#" showAs="Jones"/>
        <TLCRole eId="appellant" href="#" showAs="Appellant"/>
        <TLCRole eId="respondent" href="#" showAs="Respondent"/>
      </references>
    </meta>

    <header>
      <courtType>Court of Appeal (Civil Division)</courtType>
      <neutralCitation>[2024] EWCA Civ 123</neutralCitation>
      <docketNumber>A2/2023/1234</docketNumber>
      <docDate date="2024-03-15">15 March 2024</docDate>
      <parties>
        <party refersTo="#party-smith" as="#appellant">Smith</party>
        <party refersTo="#party-jones" as="#respondent">Jones</party>
      </parties>
      <judges>
        <judge refersTo="#judge-williams">Lord Justice Williams</judge>
      </judges>
    </header>

    <judgmentBody>
      <introduction>
        <p>Before Lord Justice Williams</p>
      </introduction>

      <paragraph eId="para_1">
        <num>1.</num>
        <content>
          <p>This is an appeal against the order of HHJ Smith dated
          15 January 2024 concerning child arrangements for the two
          children of the family.</p>
        </content>
      </paragraph>

      <paragraph eId="para_2">
        <num>2.</num>
        <content>
          <p>The appellant father appeals on three grounds. First, that
          the judge failed to give adequate weight to the children's
          expressed wishes pursuant to <ref href="/uk/act/1989/41">
          Children Act 1989</ref> section 1(3)(a).</p>
        </content>
      </paragraph>

      <decision>
        <p>For the reasons given above, the appeal is dismissed.</p>
      </decision>
    </judgmentBody>
  </judgment>
</akomaNtoso>
```

---

## 11. URI Patterns

### Judgments
```
/uk/judgment/{court}/{year}/{number}

Examples:
/uk/judgment/uksc/2024/1
/uk/judgment/ewca/civ/2024/123
/uk/judgment/ewhc/fam/2024/456
```

### Legislation
```
/uk/act/{year}/{number}
/uk/si/{year}/{number}

Examples:
/uk/act/1989/41          # Children Act 1989
/uk/act/2014/6           # Children and Families Act 2014
/uk/si/2010/2955         # Family Procedure Rules 2010
```

### Courts
```
/uk/court/{court-id}

Examples:
/uk/court/uksc
/uk/court/ewca/civ
/uk/court/ewhc/fam
```

---

## 12. Document Lifecycle States

| Status | Description |
|--------|-------------|
| `draft` | Document in preparation |
| `final` | Judgment handed down |
| `published` | Published to public database |
| `corrected` | Post-publication correction applied |
| `superseded` | Replaced by later version |

---

## Questions?

For technical queries about this schema, contact the data team or raise an issue in the repository.
