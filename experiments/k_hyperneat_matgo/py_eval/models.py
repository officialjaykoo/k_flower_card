from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PpukState:
    active: bool = False
    streak: int = 0
    last_turn_no: int = 0
    last_source: str | None = None
    last_month: int | None = None


@dataclass(frozen=True)
class Card:
    id: str
    month: int
    category: str
    name: str
    combo_tags: tuple[str, ...] = ()
    pi_value: int = 0
    bonus: dict[str, int] | None = None
    asset: str = ""
    gukjin_transformed: bool = False
    pass_card: bool = False


@dataclass
class CapturedCards:
    kwang: list[Card] = field(default_factory=list)
    ribbon: list[Card] = field(default_factory=list)
    five: list[Card] = field(default_factory=list)
    junk: list[Card] = field(default_factory=list)


@dataclass
class PlayerEvents:
    pansseul: int = 0
    ppuk: int = 0
    jjob: int = 0
    shaking: int = 0
    bomb: int = 0
    ddadak: int = 0
    jabbeok: int = 0
    yeon_ppuk: int = 0


@dataclass
class PlayerState:
    label: str = ""
    hand: list[Card] = field(default_factory=list)
    captured: CapturedCards = field(default_factory=CapturedCards)
    go_count: int = 0
    events: PlayerEvents = field(default_factory=PlayerEvents)
    gukjin_mode: str = "five"
    gukjin_locked: bool = False
    turn_count: int = 0
    last_go_base: int = 0
    president_hold: bool = False
    president_hold_month: int | None = None
    shaking_declared_months: list[int] = field(default_factory=list)
    held_bonus_cards: list[Card] = field(default_factory=list)
    ppuk_state: PpukState = field(default_factory=PpukState)
    gold: int = 100_000
    declared_stop: bool = False
    score: int = 0


@dataclass
class EngineState:
    phase: str = "playing"
    current_turn: str = "human"
    pending_go_stop: str | None = None
    pending_match: dict[str, object] | None = None
    pending_president: dict[str, object] | None = None
    pending_shaking_confirm: dict[str, object] | None = None
    pending_gukjin_choice: dict[str, object] | None = None
