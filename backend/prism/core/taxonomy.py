from __future__ import annotations

from enum import Enum


class EventType(str, Enum):
    GEOPOLITICAL = "Geopolitical"
    MACROECONOMIC = "Macroeconomic"
    CREDIT_EVENT = "Credit Event"
    MERGER_ACQUISITION = "Merger/Acquisition"
    PRODUCT_LAUNCH = "Product Launch"
    REGULATORY = "Regulatory"
    EARNINGS = "Earnings"
    SUPPLY_CHAIN = "Supply Chain"
    LEADERSHIP_CHANGE = "Leadership Change"
    CYBERSECURITY = "Cybersecurity"
    LEGAL = "Legal"
    BANKRUPTCY = "Bankruptcy"
    MARKET_SHOCK = "Market Shock"
    OTHER = "Other"

    @classmethod
    def parse(cls, value: str | EventType) -> EventType:
        if isinstance(value, cls):
            return value
        text = str(value).strip().lower()
        for member in cls:
            if text in (member.value.lower(), member.name.lower()):
                return member
        raise ValueError(f"Unknown event type: {value!r}")
