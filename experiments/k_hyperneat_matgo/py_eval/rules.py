from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RuleSet:
    name: str
    description: str
    go_min_score: int
    bak_multipliers: dict[str, int]
    use_early_stop: bool


RULE_SETS: dict[str, RuleSet] = {
    "A": RuleSet(
        name="Unified Project Rules",
        description="Project default Matgo rules (Go/Stop starts at 7 points)",
        go_min_score=7,
        bak_multipliers={"gwang": 2, "pi": 2, "mongBak": 2},
        use_early_stop=True,
    ),
}


def normalize_rule_key(rule_key: str | None) -> str:
    key = str(rule_key or "A").strip()
    return key if key in RULE_SETS else "A"
