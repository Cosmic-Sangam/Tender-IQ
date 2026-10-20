# Annotation Guidelines — TenderIQ NER

## Overview

This document defines the entity labels used in TenderIQ's custom
Named Entity Recognition (NER) model. These guidelines ensure
consistent annotation across all tender documents.

## Entity Labels

### 1. CERTIFICATION
**Definition:** Industry certifications, quality standards, and
compliance certifications that a bidder must possess.

**Positive examples:**
- "ISO 9001:2015"
- "ISO 27001:2013"
- "CMMI Level 3"
- "CMMI Level 5"
- "SOC 2 Type II"

**Negative examples (NOT certification):**
- "GST registration" → DOCUMENT
- "Company registration" → ELIGIBILITY

**Boundary rule:** Include the version number if present.
- ✓ "ISO 9001:2015" (include ":2015")
- ✗ "ISO 9001" (if the text says "ISO 9001:2015")

---

### 2. FINANCIAL_REQUIREMENT
**Definition:** Monetary amounts, percentages, and financial
thresholds mentioned as requirements or costs.

**Positive examples:**
- "Rs. 50 Crore"
- "Rs. 30 Lakh"
- "10% of the total contract value"
- "Rs. 15 Crore"

**Negative examples:**
- "Rs. 5,000" (tender fee) → Could be FINANCIAL_REQUIREMENT
  if it's a requirement, but minor fees are borderline.

**Boundary rule:** Include the currency and unit.
- ✓ "Rs. 50 Crore"
- ✗ "50 Crore" (missing currency)

---

### 3. DEADLINE
**Definition:** Specific dates, times, and duration-based deadlines.

**Positive examples:**
- "15th February 2024, 3:00 PM"
- "1st April 2024"
- "12 weeks from date of purchase order"
- "18 months from commencement"

**Boundary rule:** Include the time if present.
- ✓ "15th February 2024, 3:00 PM"
- ✗ "15th February" (incomplete)

---

### 4. TECHNICAL_REQUIREMENT
**Definition:** Technology specifications, frameworks, databases,
hardware, and performance metrics.

**Positive examples:**
- "Java Spring Boot"
- "PostgreSQL version 14"
- "React.js"
- "10,000 concurrent users"
- "99.9% uptime"
- "NVIDIA A100 80GB GPUs"
- "NIC Cloud"

**Boundary rule:** Include version numbers and specifications.
- ✓ "PostgreSQL version 14"
- ✗ "PostgreSQL" (if version is mentioned)

---

### 5. EXPERIENCE
**Definition:** Required years of experience or number of past projects.

**Positive examples:**
- "10 years of experience"
- "minimum of 10 years"
- "3 similar projects"
- "at least 2 AI/ML projects"

---

### 6. PENALTY
**Definition:** Penalty clauses, liquidated damages, and SLA violations.

**Positive examples:**
- "penalty of Rs. 50,000 per hour"
- "1% per week"
- "Liquidated damages of 0.5%"
- "penalty of Rs. 1 Lakh per incident"

---

### 7. DOCUMENT
**Definition:** Documents and certificates that bidders must submit.

**Positive examples:**
- "GST registration certificate"
- "PAN card"
- "Certificate of Incorporation"
- "Power of Attorney"
- "Audited financial statements"
- "OEM authorization letter"

---

### 8. ELIGIBILITY
**Definition:** Organizational eligibility criteria for bidders.

**Positive examples:**
- "registered company in India"
- "company registered under the Companies Act"
- "OEM or authorized dealer"
- "partnership firm registered"

---

## General Rules

1. **Longest match:** Always annotate the longest meaningful span.
2. **No nesting:** Entities should not overlap.
3. **Context matters:** The same phrase may have different labels
   in different contexts.
4. **When in doubt:** Leave it unannotated rather than guessing.