import argparse
import csv
import io
import logging
from typing import Dict, Iterator, List

from parser import FLAT_COMPANY_FIELDS, parse_company_html
from proxy_manager import ProxyManager
from requester import CompanyRequester, RequestConfig


LOGGER = logging.getLogger(__name__)


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )


def iter_edrpou_codes(csv_path: str) -> Iterator[str]:
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as handle:
        sample = handle.read(2048)
        handle.seek(0)
        try:
            has_header = csv.Sniffer().has_header(sample)
        except (csv.Error, ValueError):
            first_line = sample.splitlines()[0] if sample.splitlines() else ""
            has_header = any(ch.isalpha() for ch in first_line)
        if has_header:
            reader = csv.DictReader(handle)
            for row in reader:
                code = (
                    row.get("edrpou")
                    or row.get("EDRPOU")
                    or row.get("EDRPOU_CODE")
                    or row.get("code")
                    or row.get("Code")
                    or next(iter(row.values()), "")
                )
                normalized = "".join(ch for ch in str(code) if ch.isdigit())
                if normalized:
                    yield normalized
        else:
            reader = csv.reader(handle)
            for row in reader:
                if not row:
                    continue
                normalized = "".join(ch for ch in row[0] if ch.isdigit())
                if normalized:
                    yield normalized


def coerce_record(record: Dict[str, object]) -> Dict[str, str]:
    flat_record: Dict[str, str] = {}
    for field in FLAT_COMPANY_FIELDS:
        flat_record[field] = str(record.get(field, "") or "")
    return flat_record


def flush_rows(output_path: str, rows: List[Dict[str, str]], write_header: bool) -> bool:
    if not rows:
        return write_header
    mode = "w" if write_header else "a"
    with open(output_path, mode, encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FLAT_COMPANY_FIELDS)
        if write_header:
            writer.writeheader()
            write_header = False
        writer.writerows(rows)
    return write_header


def format_preview(rows: List[Dict[str, str]], limit: int) -> str:
    preview_rows = rows[:limit]
    if not preview_rows:
        return "No parsed rows to preview."
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=FLAT_COMPANY_FIELDS)
    writer.writeheader()
    writer.writerows(preview_rows)
    return buffer.getvalue().strip()


def build_requester(args: argparse.Namespace) -> CompanyRequester:
    proxy_manager = ProxyManager.from_file(
        args.proxy_file,
        max_failures=args.proxy_max_failures,
    )
    config = RequestConfig(
        timeout_seconds=args.timeout,
        max_attempts=args.max_attempts,
        backoff_base_seconds=args.backoff_base,
        backoff_jitter_seconds=args.backoff_jitter,
        short_delay_range=(args.short_delay_min, args.short_delay_max),
        long_delay_range=(args.long_delay_min, args.long_delay_max),
        long_delay_every=args.long_delay_every,
    )
    return CompanyRequester(config=config, proxy_manager=proxy_manager)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch and parse YouControl company pages.")
    parser.add_argument("--input", required=True, help="Path to input CSV with EDRPOU codes.")
    parser.add_argument("--output", default="company_results.csv", help="Path to output CSV.")
    parser.add_argument("--proxy-file", default=None, help="Optional proxy list file.")
    parser.add_argument("--proxy-max-failures", type=int, default=3, help="Disable proxy after this many failures.")
    parser.add_argument("--batch-size", type=int, default=20, help="Flush rows every N parsed companies.")
    parser.add_argument("--timeout", type=float, default=30.0, help="HTTP timeout in seconds.")
    parser.add_argument("--max-attempts", type=int, default=4, help="Max fetch attempts per company.")
    parser.add_argument("--backoff-base", type=float, default=2.0, help="Backoff base.")
    parser.add_argument("--backoff-jitter", type=float, default=1.0, help="Backoff jitter ceiling.")
    parser.add_argument("--short-delay-min", type=float, default=1.0, help="Minimum short delay.")
    parser.add_argument("--short-delay-max", type=float, default=3.0, help="Maximum short delay.")
    parser.add_argument("--long-delay-min", type=float, default=5.0, help="Minimum occasional long delay.")
    parser.add_argument("--long-delay-max", type=float, default=15.0, help="Maximum occasional long delay.")
    parser.add_argument("--long-delay-every", type=int, default=7, help="Apply long delay every N requests.")
    parser.add_argument("--log-level", default="INFO", help="Logging level.")
    parser.add_argument("--preview-rows", type=int, default=5, help="How many parsed rows to print at the end.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)

    requester = build_requester(args)
    pending_rows: List[Dict[str, str]] = []
    preview_rows: List[Dict[str, str]] = []
    write_header = True

    LOGGER.info(
        "Started: input=%s output=%s proxy_file=%s",
        args.input,
        args.output,
        args.proxy_file or "none",
    )

    for index, edrpou in enumerate(iter_edrpou_codes(args.input), start=1):
        LOGGER.info("Processing #%s: EDRPOU %s", index, edrpou)
        html = requester.fetch_company_page(edrpou)
        if not html:
            LOGGER.warning("Skipped #%s: EDRPOU %s (no HTML returned)", index, edrpou)
            continue

        record = coerce_record(parse_company_html(html, edrpou=edrpou))
        pending_rows.append(record)
        LOGGER.info(
            "Parsed #%s: EDRPOU %s -> %s",
            index,
            edrpou,
            record.get("company_name") or "company name not found",
        )

        if len(preview_rows) < args.preview_rows:
            preview_rows.append(record)

        if len(pending_rows) >= args.batch_size:
            write_header = flush_rows(args.output, pending_rows, write_header)
            LOGGER.info("Saved batch: %s rows -> %s", args.batch_size, args.output)
            pending_rows.clear()

    remaining_rows = len(pending_rows)
    flush_rows(args.output, pending_rows, write_header)
    if remaining_rows:
        LOGGER.info("Saved final batch: %s rows -> %s", remaining_rows, args.output)
    LOGGER.info("Completed: output=%s", args.output)
    print(format_preview(preview_rows, args.preview_rows))


if __name__ == "__main__":
    main()
