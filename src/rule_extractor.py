"""
TenderIQ — Universal Rule-Based Information Extractor
======================================================
Supports multiple tender formats: Indian govt (GeM, CPPP), 
international (World Bank), construction, IT services.
"""

import re
from src.logger_config import get_logger

logger = get_logger(__name__)


class RuleExtractor:
    """Regex-based extractor for diverse tender document layouts."""

    def __init__(self):
        # Universal currency matcher
        currency = r'(?:Rs\.?|INR|₹|\$|USD|EUR|€|£|GBP)\s*[\d,]+(?:\.\d+)?'
        unit = r'(?:\s*(?:Crore|Cr|Lakh|Lacs?|Million|Mn|M|Thousand|K|Billion|Bn))?'
        money = currency + unit

        # ── TENDER / BID / RFP ID ──
        # Matches: GEM/2026/B/8129260, NIC/IT/2024/0847, 2024/IT/RFP-100
        self.tender_id_pattern = re.compile(
            r'(?:Bid\s*Number|Tender\s*(?:Reference\s*Number|Ref(?:erence)?|No|Num|ID)|'
            r'RFP\s*(?:Reference|No)|NIT\s*No|Solicitation\s*No)'
            r'[.:\s\n\-/]*'
            r'([A-Za-z0-9]{2,15}(?:[/\-][A-Za-z0-9_\.]+){1,10})',
            re.IGNORECASE
        )

        # ── EMD / BID SECURITY ──
        self.emd_pattern = re.compile(
            r'(?:EMD|Earnest\s+Money\s+Deposit|Bid\s+Security|Bid\s+Security\s+Amount)'
            r'[\s\n\-:/]*(?:is|of|payable|amount)?[\s\n\-:/]*(' + money + r')',
            re.IGNORECASE
        )

        # ── TENDER FEE / DOCUMENT COST ──
        self.tender_fee_pattern = re.compile(
            r'(?:Tender\s+(?:fee|cost|price)|Cost\s+of\s+(?:Tender|Document|Bid)|'
            r'Document\s+Fee|Bid\s+Document\s+Fee)'
            r'[\s\n\-:/]*(?:is|of)?[\s\n\-:/]*(' + currency + r')',
            re.IGNORECASE
        )

        # ── ESTIMATED COST / PROJECT VALUE ──
        self.estimated_cost_pattern = re.compile(
            r'(?:Estimated\s+(?:cost|project\s+value|value|budget)|'
            r'Project\s+value|Contract\s+Value|Approximate\s+Cost|Budget)'
            r'[\s\n\-:/]*(?:\w+\s+){0,5}?(' + money + r')',
            re.IGNORECASE
        )

        # ── MINIMUM TURNOVER / REVENUE ──
        self.turnover_pattern = re.compile(
            r'(?:(?:average\s+)?(?:annual\s+)?turnover|revenue|financial\s+turnover)'
            r'[\s\n\-:/]*(?:\w+\s+){0,5}?(?:at\s+least\s+|minimum\s+(?:of\s+)?|of\s+)?'
            r'(' + money + r')',
            re.IGNORECASE
        )

        # ── EXPERIENCE / YEARS ──
        self.experience_pattern = re.compile(
            r'(?:(\d+)\s*(?:\+)?\s*years?(?:\s+of)?\s+(?:relevant\s+)?experience'
            r'|experience\s+(?:of\s+)?(?:at\s+least\s+|minimum\s+(?:of\s+)?)?(\d+)\s*years'
            r'|minimum\s+(?:of\s+)?(\d+)\s*years?)',
            re.IGNORECASE
        )

        # ── BID / SUBMISSION DEADLINE ──
        self.deadline_pattern = re.compile(
            r'(?:Bid\s+End\s+Date(?:/Time)?|bid\s+submission\s+deadline|'
            r'last\s+date\s+(?:for\s+)?submission|submission\s+deadline|'
            r'closing\s+date|bid\s+closing\s+date|due\s+date|final\s+date)'
            r'[\s\n\-:/]*'
            r'('
            r'\d{1,2}[\s\-\/]+(?:January|February|March|April|May|June|July|August|September|October|November|December|\d{1,2})[\s\-\/]+\d{2,4}'
            r'(?:[\s,]+\d{1,2}:\d{2}(?::\d{2})?)?'
            r'|\d{2,4}[\-/]\d{1,2}[\-/]\d{1,2}(?:[\s,T]+\d{1,2}:\d{2}(?::\d{2})?)?'
            r')',
            re.IGNORECASE
        )

        # ── BID OPENING DATE ──
        self.opening_date_pattern = re.compile(
            r'(?:Bid\s+Opening\s+Date(?:/Time)?|opening\s+date|'
            r'technical\s+bid\s+opening)'
            r'[\s\n\-:/]*'
            r'(\d{1,2}[\s\-\/]+(?:January|February|March|April|May|June|July|August|September|October|November|December|\d{1,2})[\s\-\/]+\d{2,4}'
            r'(?:[\s,]+\d{1,2}:\d{2}(?::\d{2})?)?'
            r'|\d{2,4}[\-/]\d{1,2}[\-/]\d{1,2}(?:[\s,T]+\d{1,2}:\d{2}(?::\d{2})?)?)',
            re.IGNORECASE
        )

        # ── MINISTRY / DEPARTMENT / ORGANIZATION ──
        self.organisation_pattern = re.compile(
            r'(?:Organisation\s+Name|Organization\s+Name|Issuing\s+Organization|'
            r'Issued\s+by|Procuring\s+Entity|Name\s+of\s+(?:the\s+)?Department)'
            r'[\s\n\-:/]*'
            r'([A-Z][A-Za-z0-9\s\(\)&,\'\.\-]{5,100}?)(?:\n|$|काया)',
            re.IGNORECASE
        )

        self.ministry_pattern = re.compile(
            r'(?:Ministry[/\s]*State\s+Name|Ministry\s+Name|Ministry/State)'
            r'[\s\n\-:/]*'
            r'(Ministry\s+(?:of\s+)?[A-Z][A-Za-z\s&,\'\.\-]{5,80}?)(?:\n|$|वभाग)',
            re.IGNORECASE
        )

        self.department_pattern = re.compile(
            r'(?:Department\s+Name|Department)'
            r'[\s\n\-:/]*'
            r'(Department\s+(?:of\s+)?[A-Z][A-Za-z\s&,\'\.\-]{5,80}?)(?:\n|$|संगठन)',
            re.IGNORECASE
        )

        # ── BID VALIDITY ──
        self.bid_validity_pattern = re.compile(
            r'(?:Bid\s+(?:Offer\s+)?Validity|Validity\s+Period|Offer\s+Validity)'
            r'[\s\n\-:/]*(?:\(.*?\))?'
            r'[\s\n\-:/]*(\d{1,3})\s*(?:\(Days\)|Days?)',
            re.IGNORECASE
        )

        # ── QUANTITY ──
        self.quantity_pattern = re.compile(
            r'(?:Total\s+Quantity|Quantity|Qty)'
            r'[\s\n\-:/]*(\d+(?:,\d+)*)',
            re.IGNORECASE
        )

        # ── ITEM CATEGORY (GeM specific) ──
        self.item_category_pattern = re.compile(
            r'(?:Item\s+Category|Category|Product\s+Category)'
            r'[\s\n\-:/]*'
            r'([A-Z][A-Za-z0-9\s\(\)&,\-]{3,80}?)(?:\n|वष|$)',
            re.IGNORECASE
        )

        # ── PERFORMANCE GUARANTEE ──
        self.performance_guarantee_pattern = re.compile(
            r'(?:Performance\s+(?:Bank\s+)?(?:Guarantee|Security))'
            r'[\s\n\-:/]*(\d+(?:\.\d+)?\s*%)',
            re.IGNORECASE
        )

        # ── UPTIME / SLA ──
        self.uptime_pattern = re.compile(
            r'(\d{2,3}(?:\.\d+)?\s*%\s*(?:uptime|availability|SLA))',
            re.IGNORECASE
        )

        # ── CONCURRENT USERS ──
        self.concurrent_users_pattern = re.compile(
            r'([\d,]+\s+concurrent\s+(?:users|connections|sessions))',
            re.IGNORECASE
        )

        # ── PENALTY ──
        self.penalty_amount_pattern = re.compile(
            r'penalty\s+(?:of|is|@)\s*(' + currency + r')',
            re.IGNORECASE
        )

        # ── LIQUIDATED DAMAGES ──
        self.liquidated_damages_pattern = re.compile(
            r'(liquidated\s+damages?[\s\w]{0,20}\d+(?:\.\d+)?\s*%)',
            re.IGNORECASE
        )

        logger.debug("RuleExtractor initialized with 17 universal patterns")

    def extract(self, text: str, page_number: int = 0) -> dict:
        if not text or not text.strip():
            return {}

        results = {}

        def first_match(pattern, field_name):
            match = pattern.search(text)
            if match:
                value = None
                for g in match.groups():
                    if g is not None:
                        value = g.strip()
                        break
                if value and len(value) > 1:
                    results[field_name] = {
                        "value": value,
                        "page": page_number
                    }

        first_match(self.tender_id_pattern, "tender_id")
        first_match(self.emd_pattern, "emd")
        first_match(self.tender_fee_pattern, "tender_fee")
        first_match(self.estimated_cost_pattern, "estimated_cost")
        first_match(self.turnover_pattern, "minimum_turnover")
        first_match(self.experience_pattern, "experience_years")
        first_match(self.deadline_pattern, "bid_deadline")
        first_match(self.opening_date_pattern, "bid_opening_date")
        first_match(self.organisation_pattern, "organisation")
        first_match(self.ministry_pattern, "ministry")
        first_match(self.department_pattern, "department")
        first_match(self.bid_validity_pattern, "bid_validity_days")
        first_match(self.quantity_pattern, "total_quantity")
        first_match(self.item_category_pattern, "item_category")
        first_match(self.performance_guarantee_pattern, "performance_guarantee")
        first_match(self.uptime_pattern, "uptime_requirement")
        first_match(self.concurrent_users_pattern, "concurrent_users")
        first_match(self.penalty_amount_pattern, "penalty_amount")
        first_match(self.liquidated_damages_pattern, "liquidated_damages")

        return results

    def extract_all_pages(self, pages: list) -> dict:
        merged = {}
        for page in pages:
            page_results = self.extract(page["text"], page_number=page["page_number"])
            for field, data in page_results.items():
                if field not in merged:
                    merged[field] = data
        return merged