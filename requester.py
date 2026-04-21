import logging
import random
from dataclasses import dataclass, field
import time
from typing import Dict, Optional, Tuple
from urllib.parse import urlsplit

import requests

from proxy_manager import ProxyManager, ProxyRecord


LOGGER = logging.getLogger(__name__)

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/134.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_3) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.3 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/133.0.0.0 Safari/537.36",
]
ACCEPT_HEADERS = [
    "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
]
ACCEPT_LANGUAGES = [
    "uk-UA,uk;q=0.9,en-US;q=0.7,en;q=0.6",
    "uk;q=0.9,en-US;q=0.8,en;q=0.7",
    "en-US,en;q=0.8,uk;q=0.6",
]
REFERERS = [
    "https://www.google.com/",
    "https://youcontrol.com.ua/",
    "https://duckduckgo.com/",
]
CAPTCHA_MARKERS = (
    "captcha",
    "cf-challenge",
    "attention required",
    "cloudflare",
    "verify you are human",
)


def proxy_label(proxy: Optional[ProxyRecord]) -> str:
    if not proxy:
        return "direct"
    try:
        parsed = urlsplit(proxy.raw)
    except ValueError:
        return "proxy"

    host = parsed.hostname or "proxy"
    try:
        parsed_port = parsed.port
    except ValueError:
        parsed_port = None
    port = f":{parsed_port}" if parsed_port else ""
    scheme = parsed.scheme or "http"
    auth = "***@" if parsed.username or parsed.password else ""
    return f"{scheme}://{auth}{host}{port}"


@dataclass
class RateLimiter:
    short_delay_range: Tuple[float, float] = (1.0, 3.0)
    long_delay_range: Tuple[float, float] = (5.0, 15.0)
    long_delay_every: int = 7
    request_count: int = 0

    def wait(self) -> None:
        self.request_count += 1
        delay = random.uniform(*self.short_delay_range)
        if self.request_count % self.long_delay_every == 0:
            delay += random.uniform(*self.long_delay_range)
        LOGGER.debug("Waiting %.1fs before next request", delay)
        time.sleep(delay)


@dataclass
class RequestConfig:
    timeout_seconds: float = 30.0
    max_attempts: int = 4
    backoff_base_seconds: float = 2.0
    backoff_jitter_seconds: float = 1.0
    rotate_session_per_request: bool = True
    respect_block_pages: bool = True
    short_delay_range: Tuple[float, float] = (1.0, 3.0)
    long_delay_range: Tuple[float, float] = (5.0, 15.0)
    long_delay_every: int = 7


@dataclass
class CompanyRequester:
    config: RequestConfig = field(default_factory=RequestConfig)
    proxy_manager: ProxyManager = field(default_factory=ProxyManager)
    base_url: str = "https://youcontrol.com.ua/catalog/company_details/{edrpou}/"

    def __post_init__(self) -> None:
        self.rate_limiter = RateLimiter(
            short_delay_range=self.config.short_delay_range,
            long_delay_range=self.config.long_delay_range,
            long_delay_every=self.config.long_delay_every,
        )

    def _build_headers(self) -> Dict[str, str]:
        return {
            "User-Agent": random.choice(USER_AGENTS),
            "Accept": random.choice(ACCEPT_HEADERS),
            "Accept-Language": random.choice(ACCEPT_LANGUAGES),
            "Referer": random.choice(REFERERS),
            "Cache-Control": "no-cache",
            "Pragma": "no-cache",
            "Upgrade-Insecure-Requests": "1",
            "DNT": "1",
        }

    def _new_session(self) -> requests.Session:
        session = requests.Session()
        session.headers.update(self._build_headers())
        return session

    def _sleep_backoff(self, attempt: int) -> None:
        delay = (self.config.backoff_base_seconds ** attempt) + random.uniform(
            0,
            self.config.backoff_jitter_seconds,
        )
        LOGGER.debug("Backoff %.1fs before retry", delay)
        time.sleep(delay)

    def _response_looks_blocked(self, response: requests.Response) -> bool:
        if response.status_code in {403, 429}:
            return True
        text = response.text.lower()
        return any(marker in text for marker in CAPTCHA_MARKERS)

    def fetch_company_page(self, edrpou: str) -> Optional[str]:
        url = self.base_url.format(edrpou=edrpou)

        for attempt in range(1, self.config.max_attempts + 1):
            self.rate_limiter.wait()
            session = self._new_session()
            proxy = self.proxy_manager.get_random_proxy()
            proxy_map = proxy.requests_proxy if proxy else None

            try:
                response = session.get(
                    url,
                    timeout=self.config.timeout_seconds,
                    proxies=proxy_map,
                )
                LOGGER.info(
                    "HTTP %s for EDRPOU %s (attempt %s/%s, proxy %s)",
                    response.status_code,
                    edrpou,
                    attempt,
                    self.config.max_attempts,
                    proxy_label(proxy),
                )

                if self._response_looks_blocked(response):
                    if proxy:
                        self.proxy_manager.mark_failure(proxy)
                    if self.config.respect_block_pages:
                        LOGGER.warning(
                            "Blocked response for EDRPOU %s: HTTP %s (attempt %s/%s, proxy %s)",
                            edrpou,
                            response.status_code,
                            attempt,
                            self.config.max_attempts,
                            proxy_label(proxy),
                        )
                        self._sleep_backoff(attempt)
                        return None
                    self._sleep_backoff(attempt)
                    continue

                if 500 <= response.status_code < 600:
                    if proxy:
                        self.proxy_manager.mark_failure(proxy)
                    self._sleep_backoff(attempt)
                    continue

                response.raise_for_status()
                if proxy:
                    self.proxy_manager.mark_success(proxy)
                return response.text

            except requests.RequestException as exc:
                if proxy:
                    self.proxy_manager.mark_failure(proxy)
                LOGGER.warning(
                    "Request failed for EDRPOU %s (attempt %s/%s, proxy %s): %s",
                    edrpou,
                    attempt,
                    self.config.max_attempts,
                    proxy_label(proxy),
                    exc,
                )
                self._sleep_backoff(attempt)
            finally:
                session.close()

        LOGGER.error(
            "Request exhausted for EDRPOU %s after %s attempts",
            edrpou,
            self.config.max_attempts,
        )
        return None
