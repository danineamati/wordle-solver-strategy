"""Replace sklearn export_text split lines with short human-readable conditions."""

from __future__ import annotations

import re
from typing import Optional

# Feature comparison only (preserves leading "|---" tree drawing).
_EXPR_TILE = re.compile(r"([a-z])_tile\s*([<>]=?|<|>)\s*([\d.]+)")
_EXPR_COUNT = re.compile(r"\b(n_green|n_yellow|n_absent)\s*([<>]=?|<|>)\s*([\d.]+)")


def _almost_half_integer(th: float) -> bool:
    return abs(th * 2 - round(th * 2)) < 0.02


def _human_tile(letter: str, op: str, th: float) -> Optional[str]:
    L = letter.upper()
    if abs(th - 0.5) < 0.02:
        if op in ("<=", "<"):
            return f"{L} is ⬛ absent (gray)"
        if op in (">", ">="):
            return f"{L} is 🟨 or 🟩 (yellow or green)"
    if abs(th - 1.5) < 0.02:
        if op in ("<=", "<"):
            return f"{L} is not 🟩 (⬛ or 🟨)"
        if op in (">", ">="):
            return f"{L} is 🟩 green"
    return None


def _count_noun(feat: str) -> str:
    if feat == "n_green":
        return "🟩 green"
    if feat == "n_yellow":
        return "🟨 yellow"
    return "⬛ gray (absent)"


def _human_count(feat: str, op: str, th: float) -> Optional[str]:
    if not _almost_half_integer(th):
        return None
    left_max = int(round(th - 0.5))
    noun = _count_noun(feat)

    if op in ("<=", "<"):
        k = left_max
        if k <= 0:
            return f"no {noun} letters"
        return f"at most {k} {noun} letter(s)"

    if op in (">", ">="):
        k = left_max + 1
        if feat == "n_absent" and k >= 5:
            return "all five letters are ⬛ gray (absent)"
        if k == 1:
            return f"at least one {noun} letter"
        return f"at least {k} {noun} letter(s)"

    return None


def humanize_decision_tree_export(rules: str) -> str:
    """
    Rewrite sklearn ``export_text`` split conditions as short English phrases
    (tile codes 0/1/2 and half-integer count thresholds from ``DecisionTreeClassifier``).
    Unrecognized thresholds are left unchanged.
    """

    def repl_tile(m: re.Match) -> str:
        h = _human_tile(m.group(1), m.group(2), float(m.group(3)))
        return h if h is not None else m.group(0)

    def repl_count(m: re.Match) -> str:
        h = _human_count(m.group(1), m.group(2), float(m.group(3)))
        return h if h is not None else m.group(0)

    out = _EXPR_TILE.sub(repl_tile, rules)
    out = _EXPR_COUNT.sub(repl_count, out)
    return out
