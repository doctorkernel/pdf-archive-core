from __future__ import annotations

import io
import json
import re
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Callable, Optional

import requests
from pypdf import PdfReader


DEFAULT_CATEGORY = "bill"
CATEGORY_CHOICES = ("invoice", "receipt", "bill", "electricity", "gas")
DATE_PATTERNS = (
    "%m/%d/%Y",
    "%m-%d-%Y",
    "%Y-%m-%d",
    "%B %d, %Y",
    "%b %d, %Y",
)

DebugLogger = Optional[Callable[[str], None]]


@dataclass
class LMStudioConfig:
    base_url: str
    model: str
    batch_size: int = 5
    max_input_tokens: int = 6000
    debug: bool = False
    timeout: int = 120


@dataclass
class DocumentInput:
    source_name: str
    pdf_bytes: bytes
    fallback_date: datetime
    sender: str = ""
    subject: str = ""
    extracted_text: Optional[str] = None
    page_texts: Optional[list[str]] = None


@dataclass
class ParsedMetadata:
    title: str
    category: str
    company_name: Optional[str] = None
    issue_date: Optional[date] = None
    due_date: Optional[date] = None
    needs_review: bool = False


@dataclass
class ArchiveDecision:
    source_name: str
    title: str
    category: str
    company_name: Optional[str]
    document_date: date
    document_date_source: str
    extracted_text: str
    needs_review: bool = False


def sanitize_filename_part(text: str, max_len: int = 80) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r'[\\/:*?"<>|]+', "", text)
    text = text.strip(" .-_")
    if not text:
        text = "Untitled"
    return text[:max_len].rstrip(" .-_")


def extract_pdf_text(pdf_bytes: bytes) -> str:
    pages = extract_pdf_page_texts(pdf_bytes)
    text = "\n".join(pages)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_pdf_page_texts(pdf_bytes: bytes) -> list[str]:
    reader = PdfReader(io.BytesIO(pdf_bytes))
    pages = []
    for page in reader.pages:
        pages.append(page.extract_text() or "")
    return [re.sub(r"\n{3,}", "\n\n", page).strip() for page in pages]


def try_parse_date(value: str) -> Optional[date]:
    normalized = value.strip().replace("  ", " ")
    for fmt in DATE_PATTERNS:
        try:
            return datetime.strptime(normalized, fmt).date()
        except ValueError:
            continue
    return None


def extract_field_date(text: str, labels: list[str]) -> Optional[date]:
    escaped = "|".join(re.escape(label) for label in labels)
    patterns = [
        rf"(?im)\b(?:{escaped})\b\s*[:#-]?\s*([A-Za-z]{{3,9}}\s+\d{{1,2}},\s+\d{{4}})",
        rf"(?im)\b(?:{escaped})\b\s*[:#-]?\s*(\d{{1,2}}[/-]\d{{1,2}}[/-]\d{{4}})",
        rf"(?im)\b(?:{escaped})\b\s*[:#-]?\s*(\d{{4}}-\d{{2}}-\d{{2}})",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            parsed = try_parse_date(match.group(1))
            if parsed:
                return parsed
    return None


def choose_document_date(
    text: str,
    fallback_date: datetime,
    preferred_issue_date: Optional[date] = None,
    preferred_due_date: Optional[date] = None,
    fallback_text: Optional[str] = None,
) -> tuple[date, str]:
    if preferred_issue_date:
        return preferred_issue_date, "document"
    if preferred_due_date:
        return preferred_due_date, "due"

    issued_date = extract_field_date(
        text,
        ["invoice date", "issue date", "issued", "date issued", "bill date", "statement date", "date paid"],
    )
    if issued_date:
        return issued_date, "document"

    due_date = extract_field_date(text, ["due date", "payment due", "date due"])
    if due_date:
        return due_date, "due"

    if fallback_text and fallback_text != text:
        issued_date = extract_field_date(
            fallback_text,
            ["invoice date", "issue date", "issued", "date issued", "bill date", "statement date", "date paid"],
        )
        if issued_date:
            return issued_date, "document-fallback"

        due_date = extract_field_date(fallback_text, ["due date", "payment due", "date due"])
        if due_date:
            return due_date, "due-fallback"

    return fallback_date.astimezone().date(), "fallback"


def looks_like_company_name(line: str) -> bool:
    stripped = line.strip()
    if not (2 <= len(stripped) <= 60):
        return False
    lowered = stripped.lower()
    banned = (
        "invoice",
        "receipt",
        "bill to",
        "payment history",
        "description",
        "amount due",
        "date paid",
        "date due",
        "subtotal",
        "total",
        "page ",
        "united states",
        "california",
        "texas",
        "payment address",
        "pay online",
    )
    if any(token in lowered for token in banned):
        return False
    if "@" in stripped:
        return False
    if re.search(r"\d{3,}", stripped):
        return False
    if re.search(r"\b(street|st|avenue|ave|road|rd|drive|dr|boulevard|blvd|lane|ln|court|ct|suite|ste|floor|fl|box|pmb)\b", lowered):
        return False
    if stripped.upper() == stripped and len(stripped.split()) > 6:
        return False
    words = stripped.split()
    if not words:
        return False
    caps = sum(1 for word in words if word[:1].isupper())
    return caps >= 1


def score_company_name(line: str) -> int:
    score = 0
    if "," in line:
        score += 2
    lowered = line.lower()
    if re.search(r"\b(inc|llc|ltd|corp|corporation|company|co|pbc|plc)\b", lowered):
        score += 3
    if len(line.split()) <= 4:
        score += 1
    return score


def extract_company_name(text: str) -> Optional[str]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    anchor_patterns = [
        r"(?im)^bill to$",
        r"(?im)^invoice number\b",
        r"(?im)^receipt number\b",
        r"(?im)^date paid\b",
        r"(?im)^date due\b",
    ]
    for anchor_pattern in anchor_patterns:
        match = re.search(anchor_pattern, text)
        if match:
            anchor_line = match.group(0).strip().lower()
            for index, line in enumerate(lines):
                if line.strip().lower() == anchor_line:
                    candidates = [
                        candidate
                        for candidate in reversed(lines[max(0, index - 6):index + 1])
                        if looks_like_company_name(candidate)
                    ]
                    if candidates:
                        best = max(candidates, key=score_company_name)
                        return sanitize_filename_part(best, max_len=40)

    candidates = [line for line in lines[:16] if looks_like_company_name(line)]
    if candidates:
        best = max(candidates, key=score_company_name)
        return sanitize_filename_part(best, max_len=40)
    return None


def extract_company_name_from_sender(sender: str) -> Optional[str]:
    match = re.search(r"<([^>]+)>", sender)
    email = match.group(1) if match else sender
    email = email.strip().lower()
    if "@" not in email:
        return None

    domain = email.split("@", 1)[1]
    company_part = domain.split(".", 1)[0]
    company_part = re.sub(r"^(mail|email|info|support|billing|notifications?)\.", "", company_part)
    company_part = re.sub(r"[-_]+", " ", company_part).strip()
    if not company_part:
        return None

    words = [word for word in company_part.split() if word not in {"mail", "email"}]
    if not words:
        return None
    return sanitize_filename_part(" ".join(word.capitalize() for word in words), max_len=40)


def extract_company_name_from_subject(subject: str) -> Optional[str]:
    cleaned = re.sub(r"(?i)\b(invoice|receipt|statement|bill|payment|paid|your|from)\b", " ", subject)
    cleaned = re.sub(r"[^A-Za-z0-9&' -]+", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" -")
    if not cleaned:
        return None

    words = cleaned.split()
    company_words = []
    for word in words[:4]:
        if re.search(r"\d", word):
            break
        company_words.append(word)
    if not company_words:
        return None
    return sanitize_filename_part(" ".join(company_words), max_len=40)


def classify_by_keywords(text: str) -> str:
    lowered = text.lower()
    rules = [
        ("electricity", ["electricity", "electric", "kilowatt", "kwh", "meter read", "utility electric"]),
        ("gas", ["natural gas", "gas service", "therms", "gas usage", "utility gas"]),
        ("receipt", ["receipt", "amount paid", "payment history", "receipt number", "payment method"]),
        ("invoice", ["invoice", "amount due", "remit", "invoice number", "tax invoice"]),
        ("bill", ["bill", "statement", "balance due", "payment due"]),
    ]
    scores = {category: 0 for category in CATEGORY_CHOICES}
    for category, keywords in rules:
        for keyword in keywords:
            scores[category] += lowered.count(keyword)
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else DEFAULT_CATEGORY


def heuristic_title(text: str, original_filename: str) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    for line in lines[:20]:
        if 4 <= len(line) <= 70 and not re.search(r"\b(page|date|account|invoice #|statement period)\b", line.lower()):
            candidate = sanitize_filename_part(line, max_len=50)
            if candidate:
                return candidate

    stem = Path(original_filename).stem
    return sanitize_filename_part(stem.replace("_", " ").replace("-", " "), max_len=50)


def original_filename_title(original_filename: str) -> str:
    stem = Path(original_filename).stem
    return sanitize_filename_part(stem.replace("_", " ").replace("-", " "), max_len=70)


def should_keep_original_filename(text: str, derived_title: str, category: str, company_name: Optional[str]) -> bool:
    normalized_text = re.sub(r"\s+", " ", text).strip()
    if len(normalized_text) < 40:
        return True

    lowered_title = derived_title.strip().lower()
    if not lowered_title:
        return True

    generic_titles = {"invoice", "receipt", "statement", "bill", "untitled"}
    if lowered_title in generic_titles:
        return True

    if category in {"invoice", "receipt", "bill"} and not company_name and lowered_title in generic_titles:
        return True

    return False


def build_archive_title(
    text: str,
    original_filename: str,
    category: str,
    sender: str = "",
    subject: str = "",
    preferred_company_name: Optional[str] = None,
) -> str:
    base_title = heuristic_title(text, original_filename)
    company_name = None
    if category in {"invoice", "receipt", "bill"}:
        company_name = (
            preferred_company_name
            or extract_company_name(text)
            or extract_company_name_from_sender(sender)
            or extract_company_name_from_subject(subject)
        )
        if company_name:
            lowered_base = base_title.lower()
            if company_name.lower() not in lowered_base:
                if any(keyword in lowered_base for keyword in ("invoice", "receipt", "statement", "bill")):
                    titled = sanitize_filename_part(f"{company_name} {base_title}", max_len=70)
                    if not should_keep_original_filename(text, titled, category, company_name):
                        return titled
                elif category == "receipt":
                    titled = sanitize_filename_part(f"{company_name} Receipt", max_len=70)
                    if not should_keep_original_filename(text, titled, category, company_name):
                        return titled
                else:
                    titled = sanitize_filename_part(f"{company_name} Invoice", max_len=70)
                    if not should_keep_original_filename(text, titled, category, company_name):
                        return titled
    if should_keep_original_filename(text, base_title, category, company_name):
        return original_filename_title(original_filename)
    return base_title


def significant_words(text: str) -> set[str]:
    words = set(re.findall(r"[A-Za-z]{4,}", text.lower()))
    stop_words = {
        "page",
        "this",
        "that",
        "with",
        "from",
        "your",
        "have",
        "will",
        "date",
        "service",
        "statement",
        "invoice",
        "receipt",
        "patient",
        "class",
    }
    return {word for word in words if word not in stop_words}


def detect_mixed_document(page_texts: list[str]) -> bool:
    if len(page_texts) < 2:
        return False
    first = page_texts[0].strip()
    second = page_texts[1].strip()
    if len(first) < 80 or len(second) < 80:
        return False

    first_words = significant_words(first)
    second_words = significant_words(second)
    if not first_words or not second_words:
        return False

    overlap = len(first_words & second_words)
    ratio = overlap / max(1, min(len(first_words), len(second_words)))
    return ratio < 0.08


def is_low_signal_text(text: str, min_chars: int = 80) -> bool:
    normalized = re.sub(r"\s+", " ", text).strip()
    return len(normalized) < min_chars


def is_low_signal_document(document: DocumentInput) -> bool:
    full_text = document.extracted_text or ""
    page_texts = document.page_texts or []
    primary_text = page_texts[0] if page_texts else full_text
    return is_low_signal_text(primary_text) and is_low_signal_text(full_text, min_chars=120)


def build_lm_studio_document(doc_id: int, document: DocumentInput) -> dict[str, object]:
    page_texts = document.page_texts or []
    first_page_text = page_texts[0] if page_texts else (document.extracted_text or "")
    second_page_text = page_texts[1] if len(page_texts) > 1 else ""
    return {
        "id": doc_id,
        "original_filename": document.source_name,
        "sender": document.sender,
        "subject": document.subject,
        "page_count": len(page_texts) if page_texts else 1,
        "first_page_text_excerpt": first_page_text[:9000],
        "second_page_text_excerpt": second_page_text[:5000],
        "full_pdf_text_excerpt": (document.extracted_text or "")[:12000],
    }


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, len(text) // 4)


def estimate_payload_document_tokens(document_payload: dict[str, object]) -> int:
    return estimate_tokens(json.dumps(document_payload, ensure_ascii=True))


def batch_documents_for_lm_studio(
    documents: list[DocumentInput],
    lm_config: LMStudioConfig,
    debug_logger: DebugLogger = None,
) -> list[list[DocumentInput]]:
    prompt_overhead = 600
    batches: list[list[DocumentInput]] = []
    current_batch: list[DocumentInput] = []
    current_tokens = prompt_overhead

    for document in documents:
        if is_low_signal_document(document):
            if current_batch:
                batches.append(current_batch)
                current_batch = []
                current_tokens = prompt_overhead
            batches.append([document])
            if debug_logger and lm_config.debug:
                debug_logger(
                    f"LM Studio isolating low-signal document source={document.source_name} into a single-document batch"
                )
            continue

        doc_payload = build_lm_studio_document(0, document)
        estimated_doc_tokens = estimate_payload_document_tokens(doc_payload)

        exceeds_budget = current_batch and (current_tokens + estimated_doc_tokens > lm_config.max_input_tokens)
        exceeds_count = current_batch and (len(current_batch) >= lm_config.batch_size)
        if exceeds_budget or exceeds_count:
            batches.append(current_batch)
            current_batch = []
            current_tokens = prompt_overhead

        current_batch.append(document)
        current_tokens += estimated_doc_tokens

        if debug_logger and lm_config.debug:
            debug_logger(
                f"LM Studio token estimate source={document.source_name} doc_tokens={estimated_doc_tokens} "
                f"batch_tokens={current_tokens} max_tokens={lm_config.max_input_tokens}"
            )

    if current_batch:
        batches.append(current_batch)
    return batches


def parse_lm_studio_metadata_item(item: dict[str, object], document: DocumentInput) -> ParsedMetadata:
    full_text = document.extracted_text or ""
    page_texts = document.page_texts or []
    primary_text = page_texts[0] if page_texts else full_text
    title = sanitize_filename_part(str(item.get("title", "") or ""), max_len=50)
    category = str(item.get("category", "") or "").strip().lower()
    company_name = sanitize_filename_part(str(item.get("company_name", "") or ""), max_len=40)
    issue_date = try_parse_date(str(item.get("issue_date", "") or ""))
    due_date = try_parse_date(str(item.get("due_date", "") or ""))
    needs_review = bool(item.get("needs_review", False))

    if category not in CATEGORY_CHOICES:
        category = classify_by_keywords(primary_text)
    if not title:
        title = build_archive_title(
            primary_text,
            document.source_name,
            category,
            sender=document.sender,
            subject=document.subject,
            preferred_company_name=company_name or None,
        )

    return ParsedMetadata(
        title=title,
        category=category,
        company_name=company_name or None,
        issue_date=issue_date,
        due_date=due_date,
        needs_review=needs_review,
    )


def strip_json_fences(content: str) -> str:
    stripped = content.strip()
    fence_match = re.match(r"^```(?:json)?\s*(.*?)\s*```$", stripped, re.DOTALL | re.IGNORECASE)
    if fence_match:
        return fence_match.group(1).strip()
    return stripped


def extract_metadata_batch_with_lm_studio(
    documents: list[DocumentInput],
    lm_config: LMStudioConfig,
    debug_logger: DebugLogger = None,
) -> list[Optional[ParsedMetadata]]:
    payload_documents = [build_lm_studio_document(index, document) for index, document in enumerate(documents)]
    prompt = """
You are extracting metadata from PDFs for local archival.

Return strict JSON only with this shape:
{"documents":[{"id":0,"title":"","category":"","company_name":"","issue_date":null,"due_date":null,"needs_review":false}]}

Rules:
- Return one object for each input document id.
- category must be exactly one of: invoice, receipt, bill, electricity, gas.
- title must be 2 to 7 words if present.
- Base the title/category/company primarily on the first page.
- Do not use names, titles, or organizations from any other document in the batch.
- If a field is not clearly supported by the current document, leave it blank or null instead of guessing.
- Prefer vendor/company name plus document type if obvious.
- If category is invoice, receipt, or bill, include the vendor/company name in the title.
- Do not include dates in the title.
- Do not include file extensions.
- company_name should be the issuing company, not the billed customer and not an address or country.
- issue_date and due_date must be YYYY-MM-DD when known, otherwise null.
- Set needs_review to true when page 2 or later appears unrelated to page 1 or the PDF appears to contain mixed documents.
""".strip()

    url = lm_config.base_url.rstrip("/") + "/v1/chat/completions"
    payload = {
        "model": lm_config.model,
        "temperature": 0.1,
        "messages": [
            {"role": "system", "content": "You return strict JSON."},
            {"role": "user", "content": prompt},
            {"role": "user", "content": json.dumps({"documents": payload_documents}, ensure_ascii=True)},
        ],
    }
    if lm_config.debug and debug_logger:
        debug_logger(f"LM Studio request model={lm_config.model} base_url={lm_config.base_url} batch_size={len(documents)}")
        debug_logger(f"LM Studio request payload: {json.dumps(payload, ensure_ascii=True)[:12000]}")

    response = requests.post(url, json=payload, timeout=lm_config.timeout)
    if not response.ok:
        if debug_logger:
            debug_logger(f"LM Studio HTTP error status={response.status_code} body={response.text[:4000]}")
        response.raise_for_status()
    content = response.json()["choices"][0]["message"]["content"]
    if lm_config.debug and debug_logger:
        debug_logger(f"LM Studio raw response: {content[:12000]}")
    parsed = json.loads(strip_json_fences(content))
    raw_documents = parsed.get("documents", [])
    by_id = {
        int(item.get("id")): parse_lm_studio_metadata_item(item, documents[int(item.get("id"))])
        for item in raw_documents
        if str(item.get("id", "")).isdigit() and 0 <= int(item["id"]) < len(documents)
    }
    return [by_id.get(index) for index in range(len(documents))]


def analyze_documents_batch(
    documents: list[DocumentInput],
    lm_config: Optional[LMStudioConfig] = None,
    debug_logger: DebugLogger = None,
) -> list[ArchiveDecision]:
    prepared: list[DocumentInput] = []
    for document in documents:
        page_texts = document.page_texts if document.page_texts is not None else extract_pdf_page_texts(document.pdf_bytes)
        extracted_text = document.extracted_text if document.extracted_text is not None else re.sub(r"\n{3,}", "\n\n", "\n".join(page_texts)).strip()
        prepared.append(
            DocumentInput(
                source_name=document.source_name,
                pdf_bytes=document.pdf_bytes,
                fallback_date=document.fallback_date,
                sender=document.sender,
                subject=document.subject,
                extracted_text=extracted_text,
                page_texts=page_texts,
            )
        )

    parsed_results: list[Optional[ParsedMetadata]] = [None] * len(prepared)
    if lm_config and lm_config.model:
        indexed_documents = list(enumerate(prepared))
        batches = batch_documents_for_lm_studio(prepared, lm_config, debug_logger=debug_logger)
        offset = 0
        for batch in batches:
            batch_indices = [index for index, _ in indexed_documents[offset:offset + len(batch)]]
            try:
                batch_results = extract_metadata_batch_with_lm_studio(batch, lm_config, debug_logger=debug_logger)
            except Exception as exc:
                if debug_logger:
                    debug_logger(f"LM Studio batch failed; falling back to heuristics. error={exc}")
                batch_results = [None] * len(batch)
            for global_index, result in zip(batch_indices, batch_results):
                parsed_results[global_index] = result
            offset += len(batch)

    decisions: list[ArchiveDecision] = []
    for document, parsed in zip(prepared, parsed_results):
        full_text = document.extracted_text or ""
        page_texts = document.page_texts or []
        primary_text = page_texts[0] if page_texts else full_text
        needs_review = detect_mixed_document(page_texts)
        low_signal = is_low_signal_text(primary_text) and is_low_signal_text(full_text, min_chars=120)
        if parsed is None:
            category = classify_by_keywords(primary_text)
            title = build_archive_title(primary_text, document.source_name, category, sender=document.sender, subject=document.subject)
            company_name = None
            issue_date = None
            due_date = None
        else:
            category = parsed.category
            title = parsed.title
            company_name = parsed.company_name
            issue_date = parsed.issue_date
            due_date = parsed.due_date
            needs_review = needs_review or parsed.needs_review

        if low_signal:
            title = original_filename_title(document.source_name)
            needs_review = True

        document_date, document_date_source = choose_document_date(
            primary_text,
            document.fallback_date,
            preferred_issue_date=issue_date,
            preferred_due_date=due_date,
            fallback_text=full_text,
        )
        decisions.append(
            ArchiveDecision(
                source_name=document.source_name,
                title=title,
                category=category,
                company_name=company_name,
                document_date=document_date,
                document_date_source=document_date_source,
                extracted_text=full_text,
                needs_review=needs_review,
            )
        )
    return decisions


def analyze_document(
    document: DocumentInput,
    lm_config: Optional[LMStudioConfig] = None,
    debug_logger: DebugLogger = None,
) -> ArchiveDecision:
    return analyze_documents_batch([document], lm_config=lm_config, debug_logger=debug_logger)[0]


def build_relative_output_path(
    decision: ArchiveDecision,
    include_category: bool = True,
    filename_suffix: str = "",
    folder_style: str = "flat_year_month",
) -> Path:
    if folder_style == "flat_year_month":
        parent = Path(decision.document_date.strftime("%Y-%m"))
    elif folder_style == "nested_year_monthword":
        parent = Path(decision.document_date.strftime("%Y")) / decision.document_date.strftime("%m-%B")
    else:
        raise ValueError(f"Unsupported folder_style: {folder_style}")

    filename = f"{decision.document_date.strftime('%Y-%m-%d')} {decision.title}"
    if include_category:
        filename += f" ({decision.category})"
    if filename_suffix:
        filename += filename_suffix
    filename = sanitize_filename_part(filename, max_len=140) + ".pdf"
    return parent / filename


def unique_output_path(base_dir: Path, filename: str) -> Path:
    candidate = base_dir / filename
    if not candidate.exists():
        return candidate

    stem = candidate.stem
    suffix = candidate.suffix
    counter = 2
    while True:
        alt = base_dir / f"{stem} [{counter}]{suffix}"
        if not alt.exists():
            return alt
        counter += 1
