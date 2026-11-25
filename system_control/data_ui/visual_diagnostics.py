"""Physics-aware diagnostic plots shared by the 3Q and 4Q Data UIs."""

from __future__ import annotations

from math import sqrt
from typing import Any, Mapping

import numpy as np
from matplotlib.patches import Patch

try:
    from .rho_fidelity_summary import normalized_core_density, primary_variant
except Exception:
    from rho_fidelity_summary import normalized_core_density, primary_variant


_EPS = 1e-12
_GROUP_COLORS = (
    "#4c78a8",
    "#f58518",
    "#54a24b",
    "#e45756",
    "#72b7b2",
    "#b279a2",
    "#ff9da6",
    "#9d755d",
    "#bab0ac",
)


def draw_state_family_chart(
    figure,
    probabilities: Mapping[str, Any] | None,
    groups: Mapping[str, Any] | None = None,
    *,
    message: str = "waiting for rho",
    top_n: int = 10,
) -> None:
    """Draw fine state-family probabilities as a grouped horizontal bar chart."""

    figure.clear()
    ax = figure.add_subplot(111)
    probabilities = probabilities or {}
    groups = groups or {}
    rows = sorted(
        ((str(label), _probability(value)) for label, value in probabilities.items()),
        key=lambda item: item[1],
        reverse=True,
    )[:top_n]
    if not rows:
        _draw_empty(ax, message)
        figure.tight_layout(pad=0.5)
        return

    labels = [label for label, _value in rows]
    values = np.asarray([value for _label, value in rows], dtype=float)
    group_names = [str(groups.get(label, "unassigned")) for label in labels]
    color_map = _group_color_map(group_names)
    colors = [color_map[group] for group in group_names]
    y = np.arange(len(rows))

    ax.barh(y, values * 100.0, color=colors, alpha=0.86)
    ax.set_yticks(y)
    ax.set_yticklabels([_family_label(label, group) for label, group in zip(labels, group_names)], fontsize=7)
    ax.invert_yaxis()
    ax.set_xlim(0.0, max(25.0, float(np.max(values) * 112.0)))
    ax.set_xlabel("classifier probability (%)", fontsize=8)
    ax.set_title("Fine state-family probability by physical group", fontsize=9)
    ax.grid(axis="x", alpha=0.24)
    ax.tick_params(axis="x", labelsize=7)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    for yi, value in zip(y, values):
        ax.text(value * 100.0 + 0.6, yi, f"{value * 100.0:.1f}%", va="center", fontsize=7)

    legend_handles = [Patch(facecolor=color, edgecolor="none", alpha=0.86) for color in color_map.values()]
    legend_labels = list(color_map.keys())
    if legend_handles:
        ax.legend(legend_handles, legend_labels, loc="lower right", fontsize=7, frameon=False)
    figure.tight_layout(pad=0.5)


def draw_w_subspace_summary(
    figure,
    rho: Any,
    num_qubits: int,
    summary: Mapping[str, Any] | None = None,
    *,
    message: str = "waiting for rho",
) -> None:
    """Draw W-mode population and pair-coherence phase diagnostics."""

    figure.clear()
    if rho is None:
        ax = figure.add_subplot(111)
        _draw_empty(ax, message)
        figure.tight_layout(pad=0.5)
        return

    try:
        core, core_population = normalized_core_density(np.asarray(rho, dtype=complex), int(num_qubits))
    except Exception as exc:
        ax = figure.add_subplot(111)
        _draw_empty(ax, f"W-subspace unavailable: {exc}")
        figure.tight_layout(pad=0.5)
        return

    ax_pop = figure.add_subplot(1, 2, 1)
    ax_phase = figure.add_subplot(1, 2, 2, projection="polar")
    _draw_w_population(ax_pop, core, core_population, int(num_qubits), summary or {})
    _draw_w_coherence_phase(ax_phase, core, int(num_qubits))
    subtitle = _summary_subtitle(summary or {}, core_population)
    if subtitle:
        figure.suptitle(subtitle, fontsize=8)
    figure.tight_layout(pad=0.55, rect=(0, 0, 1, 0.93 if subtitle else 1))


def _draw_w_population(ax, core: np.ndarray, core_population: float, num_qubits: int, summary: Mapping[str, Any]) -> None:
    diagonal = np.real(np.diag(core))
    populations = np.clip(diagonal, 0.0, 1.0)
    x = np.arange(num_qubits)
    ax.bar(x, populations, color="#4c78a8", alpha=0.86, label="measured")
    target = _target_mode_probabilities(summary, num_qubits)
    if target is not None:
        ax.scatter(x, target, color="#2f2f2f", marker="D", s=18, label="target", zorder=3)
        for xi, value in zip(x, target):
            ax.plot([xi - 0.26, xi + 0.26], [value, value], color="#2f2f2f", linewidth=1.0, alpha=0.65)
    ax.set_ylim(0.0, max(1.0, float(np.max(populations)) * 1.15 if populations.size else 1.0))
    ax.set_xticks(x)
    ax.set_xticklabels([f"W{i + 1}" for i in range(num_qubits)], fontsize=7)
    ax.set_ylabel("P(mode | W)", fontsize=8)
    ax.set_title(f"W-mode population, P_W={core_population:.3f}", fontsize=9)
    ax.grid(axis="y", alpha=0.24)
    ax.tick_params(axis="y", labelsize=7)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    if target is not None:
        ax.legend(loc="upper right", fontsize=7, frameon=False)


def _draw_w_coherence_phase(ax, core: np.ndarray, num_qubits: int) -> None:
    populations = np.clip(np.real(np.diag(core)), 0.0, 1.0)
    pairs = []
    for i in range(num_qubits):
        for j in range(i + 1, num_qubits):
            denom = sqrt(max(float(populations[i] * populations[j]), 0.0))
            radius = 0.0 if denom <= _EPS else min(1.0, abs(core[i, j]) / denom)
            angle = float(np.angle(core[i, j]))
            pairs.append((angle, radius, f"{i + 1}-{j + 1}"))
    if not pairs or max(radius for _angle, radius, _label in pairs) <= _EPS:
        ax.text(0.5, 0.5, "no resolved\npair coherence", ha="center", va="center", transform=ax.transAxes, fontsize=8)
    else:
        for angle, radius, label in pairs:
            color = "#4c78a8" if radius >= 0.66 else "#f58518" if radius >= 0.33 else "#bab0ac"
            ax.plot([angle, angle], [0.0, radius], color=color, linewidth=1.4, alpha=0.72)
            ax.scatter([angle], [radius], color=color, s=24)
            ax.text(angle, min(1.08, radius + 0.08), label, ha="center", va="center", fontsize=7)
    ax.set_ylim(0.0, 1.08)
    ax.set_title("Pair coherence phase", fontsize=9)
    ax.set_yticklabels([])
    ax.tick_params(axis="x", labelsize=7)
    ax.grid(alpha=0.24)


def _summary_subtitle(summary: Mapping[str, Any], core_population: float) -> str:
    _key, metrics = primary_variant(summary)
    metrics = metrics or {}
    pieces = [f"P_W={core_population:.3f}"]
    for label, key, percent in (
        ("F", "fidelity", False),
        ("purity", "purity", False),
        ("OOD", "ood", True),
    ):
        value = metrics.get(key)
        if value is None:
            continue
        try:
            number = float(value)
        except Exception:
            continue
        pieces.append(f"{label}={number * 100.0:.1f}%" if percent else f"{label}={number:.3f}")
    return " | ".join(pieces)


def _target_mode_probabilities(summary: Mapping[str, Any], num_qubits: int) -> np.ndarray | None:
    amplitudes = summary.get("target_amplitudes")
    if amplitudes is None:
        return None
    try:
        amps = np.asarray(amplitudes, dtype=float).reshape(-1)[:num_qubits]
    except Exception:
        return None
    if amps.size != num_qubits:
        return None
    weights = np.square(np.abs(amps))
    total = float(np.sum(weights))
    if total <= _EPS:
        return None
    return weights / total


def _group_color_map(group_names: list[str]) -> dict[str, str]:
    unique = []
    for group in group_names:
        if group not in unique:
            unique.append(group)
    return {group: _GROUP_COLORS[index % len(_GROUP_COLORS)] for index, group in enumerate(unique)}


def _family_label(label: str, group: str) -> str:
    text = str(label).replace("_", " ")
    if len(text) > 28:
        text = text[:25] + "..."
    group_text = str(group).replace("_", " ")
    return f"{text}\n{group_text}"


def _probability(value: Any) -> float:
    try:
        return max(0.0, min(1.0, float(value or 0.0)))
    except Exception:
        return 0.0


def _draw_empty(ax, message: str) -> None:
    ax.clear()
    ax.axis("off")
    ax.text(0.5, 0.5, str(message), ha="center", va="center", fontsize=9, color="#555555")
