"""S-dual shot-budget allocation for V10 scaling studies."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Iterable, Tuple

from .contracts import sha256_of

S_TOTAL = "S-total"
S_SETTING = "S-setting"
S_IDEAL = "S-ideal"
SHOT_BUDGET_MODES = (S_TOTAL, S_SETTING, S_IDEAL)
V10_SHOT_MODE_ALIASES = {
    "fixed_total": S_TOTAL,
    "fixed_per_setting": S_SETTING,
    "ideal": S_IDEAL,
}
# Map the V2.4 rung-plan shot-mode vocabulary (``S_total`` / ``S_setting`` /
# ``S_ideal``) to the V10 alias vocabulary accepted by
# ``allocate_measurement_shots``.  Used so the per-cell measurement/shot-budget
# identity is derived from the actual cell contract instead of hard-coded.
RUNG_SHOT_MODE_TO_V10 = {
    "S_total": "fixed_total",
    "S_setting": "fixed_per_setting",
    "S_ideal": "ideal",
}


@dataclass(frozen=True)
class V10ShotBudget:
    mode: str
    setting_ids: Tuple[str, ...]
    shots_by_setting: Dict[str, int]
    total_shots: int
    probability_only: bool

    def validate(self) -> None:
        if self.mode not in SHOT_BUDGET_MODES:
            raise ValueError(f"unknown shot budget mode: {self.mode}")
        if set(self.setting_ids) != set(self.shots_by_setting):
            raise ValueError("every and only registered settings require a shot allocation")
        if self.probability_only != (self.mode == S_IDEAL):
            raise ValueError("only S-ideal is probability-only")
        if self.total_shots != sum(self.shots_by_setting.values()):
            raise ValueError("total_shots does not match per-setting allocation")

    def sha256(self) -> str:
        return sha256_of(asdict(self))


def allocate_shots(setting_ids: Iterable[str], mode: str, budget: int) -> V10ShotBudget:
    ids = tuple(setting_ids)
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("setting_ids must be non-empty and unique")
    if budget < 0:
        raise ValueError("budget must be non-negative")
    if mode == S_TOTAL:
        base, remainder = divmod(int(budget), len(ids))
        # Remainder ownership is a function of setting_id, not input order.
        # This keeps a fixed-total manifest stable when registry enumeration is
        # refactored without changing the physical settings.
        extra_ids = set(sorted(ids)[:remainder])
        allocation = {sid: base + int(sid in extra_ids) for sid in ids}
    elif mode == S_SETTING:
        allocation = {sid: int(budget) for sid in ids}
    elif mode == S_IDEAL:
        allocation = {sid: 0 for sid in ids}
    else:
        raise ValueError(f"mode must be one of {SHOT_BUDGET_MODES}")
    result = V10ShotBudget(
        mode=mode, setting_ids=ids, shots_by_setting=allocation,
        total_shots=sum(allocation.values()), probability_only=mode == S_IDEAL,
    )
    result.validate()
    return result


def allocate_measurement_shots(
    setting_ids: Iterable[str], shot_mode: str, budget: int,
) -> V10ShotBudget:
    """Allocate using the unambiguous V10 public shot-mode vocabulary."""
    if shot_mode not in V10_SHOT_MODE_ALIASES:
        raise ValueError("shot_mode must be ideal, fixed_total, or fixed_per_setting")
    return allocate_shots(setting_ids, V10_SHOT_MODE_ALIASES[shot_mode], budget)
