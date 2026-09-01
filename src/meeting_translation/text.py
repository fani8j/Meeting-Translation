from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

from .domain import Correction


@dataclass
class StabilityState:
    previous: str = ""
    published: str = ""
    revision: int = 0


class StabilityTracker:
    def __init__(self) -> None:
        self._segments: dict[str, StabilityState] = {}

    def update(self, segment_id: str, text: str, final: bool) -> tuple[str, int] | None:
        normalized = _normalize_space(text)
        state = self._segments.setdefault(segment_id, StabilityState())
        if final:
            candidate = normalized
        elif not state.previous:
            state.previous = normalized
            return None
        else:
            candidate = _common_prefix(state.previous, normalized).rstrip()
        state.previous = normalized
        if len(candidate) <= len(state.published) or len(candidate) - len(state.published) < 2:
            if not final or candidate == state.published:
                return None
        state.published = candidate
        state.revision += 1
        if final:
            self._segments.pop(segment_id, None)
        return candidate, state.revision

    def discard(self, segment_id: str) -> None:
        self._segments.pop(segment_id, None)


class Glossary:
    def __init__(self, path: Path) -> None:
        payload = json.loads(path.read_text(encoding="utf-8"))
        self.version = int(payload["version"])
        self.entries = tuple(payload["entries"])
        self.digest = hashlib.sha256(path.read_bytes()).hexdigest()[:12]
        aliases: list[tuple[str, str]] = []
        for entry in self.entries:
            canonical = entry["canonical"]
            aliases.extend((alias, canonical) for alias in entry.get("aliases", []))
        self.aliases = tuple(sorted(aliases, key=lambda item: len(item[0]), reverse=True))

    @property
    def prompt_terms(self) -> str:
        return ", ".join(entry["canonical"] for entry in self.entries)

    def apply(self, text: str) -> tuple[str, tuple[Correction, ...]]:
        corrected = text
        changes: list[Correction] = []
        for alias, canonical in self.aliases:
            if alias.casefold() == canonical.casefold():
                continue
            pattern = re.compile(re.escape(alias), flags=re.IGNORECASE)
            if not pattern.search(corrected):
                continue
            before = pattern.search(corrected)
            assert before is not None
            corrected = pattern.sub(canonical, corrected)
            changes.append(Correction(before=before.group(0), after=canonical, reason="verified_glossary"))
        return corrected, tuple(changes)


def _common_prefix(left: str, right: str) -> str:
    end = min(len(left), len(right))
    index = 0
    while index < end and left[index] == right[index]:
        index += 1
    return left[:index]


def _normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()
