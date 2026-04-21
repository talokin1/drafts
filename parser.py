import json
import re
from typing import Any, Dict, Iterable, List, Optional

from bs4 import BeautifulSoup


FLAT_COMPANY_FIELDS = [
    "edrpou",
    "company_name",
    "short_name",
    "status",
    "registration_date",
    "legal_form",
    "address",
    "manager",
    "main_activity",
    "phone",
    "email",
    "website",
    "profile_json",
    "beneficiaries_json",
    "authorized_persons_json",
    "contacts_json",
    "json_ld_json",
    "other_sections_json",
]


SECTION_BUCKETS = {
    "catalog-company-file": "profile",
    "catalog-company-beneficiary": "beneficiaries",
    "catalog-company-beneficiaries": "beneficiaries",
    "catalog-company-authorized-persons": "authorized_persons",
    "catalog-company-representatives": "authorized_persons",
    "catalog-company-persons": "authorized_persons",
    "catalog-company-contact": "contacts",
    "catalog-company-contacts": "contacts",
}


def clean_text(value: Optional[str]) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


def _normalized(value: str) -> str:
    return clean_text(value).casefold()


def _to_json(value: Any) -> str:
    if value in ({}, [], None):
        return ""
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _merge_value(target: Dict[str, Any], key: str, value: str) -> None:
    if not key or not value:
        return
    if key not in target:
        target[key] = value
        return
    existing = target[key]
    if isinstance(existing, list):
        if value not in existing:
            existing.append(value)
    elif existing != value:
        target[key] = [existing, value]


def _first_text(element: Any, selectors: Iterable[str]) -> str:
    for selector in selectors:
        found = element.select_one(selector)
        if found:
            text = clean_text(found.get_text(" ", strip=True))
            if text:
                return text
    return ""


def _parse_table_rows(section: Any) -> Dict[str, Any]:
    data: Dict[str, Any] = {}
    for row in section.select("tr"):
        cells = row.find_all(["td", "th"], recursive=False)
        if len(cells) < 2:
            continue
        key = clean_text(cells[0].get_text(" ", strip=True))
        value = clean_text(cells[1].get_text(" ", strip=True))
        _merge_value(data, key, value)
    return data


def _parse_seo_rows(section: Any) -> Dict[str, Any]:
    data: Dict[str, Any] = {}
    for row in section.select(".seo-table-row"):
        key = _first_text(row, [".seo-table-col-1"])
        if not key:
            continue

        value = _first_text(
            row,
            [
                ".copy-file-field",
                ".seo-table-col-2",
                "td:nth-of-type(2)",
            ],
        )
        _merge_value(data, key, value)
    return data


def _parse_definition_rows(section: Any) -> Dict[str, Any]:
    data: Dict[str, Any] = {}
    for term in section.select("dt"):
        value_node = term.find_next_sibling("dd")
        if not value_node:
            continue
        key = clean_text(term.get_text(" ", strip=True))
        value = clean_text(value_node.get_text(" ", strip=True))
        _merge_value(data, key, value)
    return data


def _parse_key_value_section(section: Any) -> Dict[str, Any]:
    data: Dict[str, Any] = {}
    for parser in (_parse_seo_rows, _parse_table_rows, _parse_definition_rows):
        data.update(parser(section))
    return data


def _find_value(data: Dict[str, Any], keywords: Iterable[str]) -> str:
    normalized_keywords = [_normalized(keyword) for keyword in keywords]
    for key, value in data.items():
        normalized_key = _normalized(key)
        if any(keyword in normalized_key for keyword in normalized_keywords):
            if isinstance(value, list):
                return "; ".join(str(item) for item in value if item)
            return str(value or "")
    return ""


def _extract_json_ld(soup: BeautifulSoup) -> List[Any]:
    records: List[Any] = []
    for script in soup.select('script[type="application/ld+json"]'):
        raw = clean_text(script.string or script.get_text(" ", strip=True))
        if not raw:
            continue
        try:
            records.append(json.loads(raw))
        except json.JSONDecodeError:
            records.append(raw)
    return records


def _extract_contacts(soup: BeautifulSoup) -> Dict[str, Any]:
    contacts: Dict[str, Any] = {}

    emails = sorted(
        set(
            re.findall(
                r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}",
                soup.get_text(" ", strip=True),
                flags=re.IGNORECASE,
            )
        )
    )
    phones = sorted(
        set(
            clean_text(match)
            for match in re.findall(
                r"(?:\+?38)?0\d{2}[\s().-]*\d{3}[\s().-]*\d{2}[\s().-]*\d{2}",
                soup.get_text(" ", strip=True),
            )
        )
    )
    websites = sorted(
        set(
            link.get("href", "").strip()
            for link in soup.select('a[href^="http"]')
            if "youcontrol.com.ua" not in link.get("href", "")
        )
    )

    mailto = sorted(
        set(
            link.get("href", "").replace("mailto:", "").strip()
            for link in soup.select('a[href^="mailto:"]')
            if link.get("href")
        )
    )
    tel = sorted(
        set(
            clean_text(link.get("href", "").replace("tel:", ""))
            for link in soup.select('a[href^="tel:"]')
            if link.get("href")
        )
    )

    if mailto:
        emails = sorted(set(emails + mailto))
    if tel:
        phones = sorted(set(phones + tel))
    if emails:
        contacts["emails"] = emails
    if phones:
        contacts["phones"] = phones
    if websites:
        contacts["websites"] = websites

    return contacts


def _extract_title(soup: BeautifulSoup) -> str:
    for selector in ("h1", ".company-title", ".seo-header-title"):
        found = soup.select_one(selector)
        if found:
            text = clean_text(found.get_text(" ", strip=True))
            if text:
                return text
    if soup.title:
        return clean_text(soup.title.get_text(" ", strip=True))
    return ""


def parse_company_html(html: str, edrpou: str = "") -> Dict[str, Any]:
    soup = BeautifulSoup(html or "", "lxml")

    profile: Dict[str, Any] = {}
    beneficiaries: Dict[str, Any] = {}
    authorized_persons: Dict[str, Any] = {}
    contacts = _extract_contacts(soup)
    other_sections: Dict[str, Any] = {}

    for section in soup.select('[id^="catalog-company"]'):
        section_id = section.get("id", "")
        parsed = _parse_key_value_section(section)
        if not parsed:
            continue

        bucket = SECTION_BUCKETS.get(section_id)
        if bucket == "profile":
            profile.update(parsed)
        elif bucket == "beneficiaries":
            beneficiaries.update(parsed)
        elif bucket == "authorized_persons":
            authorized_persons.update(parsed)
        elif bucket == "contacts":
            contacts.update(parsed)
        else:
            other_sections[section_id] = parsed

    if not profile:
        profile = _parse_key_value_section(soup)

    company_name = _find_value(
        profile,
        [
            "повна назва",
            "найменування юридичної особи",
            "назва юридичної особи",
            "company name",
        ],
    ) or _extract_title(soup)

    record = {
        "edrpou": edrpou
        or _find_value(profile, ["єдрпоу", "едрпоу", "edrpou", "код"]),
        "company_name": company_name,
        "short_name": _find_value(profile, ["скорочена назва", "short name"]),
        "status": _find_value(profile, ["стан", "статус", "status"]),
        "registration_date": _find_value(
            profile,
            ["дата реєстрації", "дата державної реєстрації", "registration date"],
        ),
        "legal_form": _find_value(
            profile,
            ["організаційно-правова форма", "правова форма", "legal form"],
        ),
        "address": _find_value(
            profile,
            ["місцезнаходження", "адреса", "address"],
        ),
        "manager": _find_value(profile, ["керівник", "директор", "manager"]),
        "main_activity": _find_value(
            profile,
            ["основний вид діяльності", "основний квед", "квед", "activity"],
        ),
        "phone": "; ".join(contacts.get("phones", []))
        if isinstance(contacts.get("phones"), list)
        else str(contacts.get("phones", "") or ""),
        "email": "; ".join(contacts.get("emails", []))
        if isinstance(contacts.get("emails"), list)
        else str(contacts.get("emails", "") or ""),
        "website": "; ".join(contacts.get("websites", []))
        if isinstance(contacts.get("websites"), list)
        else str(contacts.get("websites", "") or ""),
        "profile_json": _to_json(profile),
        "beneficiaries_json": _to_json(beneficiaries),
        "authorized_persons_json": _to_json(authorized_persons),
        "contacts_json": _to_json(contacts),
        "json_ld_json": _to_json(_extract_json_ld(soup)),
        "other_sections_json": _to_json(other_sections),
    }

    return record
