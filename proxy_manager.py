import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional


@dataclass
class ProxyRecord:
    raw: str
    failures: int = 0
    successes: int = 0
    disabled: bool = False


@dataclass
class ProxyManager:
    proxies: List[ProxyRecord] = field(default_factory=list)
    max_failures: int = 3

    @classmethod
    def from_file(cls, path: Optional[str], max_failures: int = 3) -> "ProxyManager":
        if not path:
            return cls([], max_failures=max_failures)
        lines = Path(path).read_text(encoding="utf-8").splitlines()
        return cls.from_lines(lines, max_failures=max_failures)

    @classmethod
    def from_lines(cls, lines: Iterable[str], max_failures: int = 3) -> "ProxyManager":
        proxies = []
        for line in lines:
            normalized = cls.normalize_proxy(line)
            if normalized:
                proxies.append(ProxyRecord(raw=normalized))
        return cls(proxies=proxies, max_failures=max_failures)

    @staticmethod
    def normalize_proxy(value: str) -> Optional[str]:
        line = value.strip()
        if not line or line.startswith("#"):
            return None
        if "://" not in line:
            line = f"http://{line}"
        return line

    def has_proxies(self) -> bool:
        return any(not proxy.disabled for proxy in self.proxies)

    def get_random_proxy(self) -> Optional[ProxyRecord]:
        active = [proxy for proxy in self.proxies if not proxy.disabled]
        if not active:
            return None
        return random.choice(active)

    def mark_success(self, proxy: Optional[ProxyRecord]) -> None:
        if not proxy:
            return
        proxy.successes += 1

    def mark_failure(self, proxy: Optional[ProxyRecord]) -> None:
        if not proxy:
            return
        proxy.failures += 1
        if proxy.failures >= self.max_failures:
            proxy.disabled = True
