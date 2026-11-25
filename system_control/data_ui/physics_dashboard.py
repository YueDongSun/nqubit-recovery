"""High-density physics and model-inference dashboard for the Data UIs."""

from __future__ import annotations

from math import log2, sqrt
from typing import Any, Mapping

import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QFrame,
    QGridLayout,
    QHeaderView,
    QLabel,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

try:
    from .rho_fidelity_summary import normalized_core_density, top_fidelity_fits
except Exception:
    from rho_fidelity_summary import normalized_core_density, top_fidelity_fits


_EPS = 1e-12
_METRIC_SPECS = (
    ("best_fidelity", "Best F", "best computed fidelity", "#047857"),
    ("runner_up_fidelity", "Second F", "next-highest computed fidelity", "#15803d"),
    ("estimator_agreement", "F agreement", "spread across all F definitions", "#0f766e"),
    ("w_population", "P(W)", "single-excitation support", "#2563eb"),
    ("conditional_target_fidelity", "F(target | W)", "target overlap conditioned on W", "#15803d"),
    ("purity", "Purity", "PSD-projected Tr(rho^2)", "#0f766e"),
    ("entropy_bits", "Entropy", "von Neumann entropy", "#0369a1"),
    ("largest_eigenvalue", "Largest eigenvalue", "spectral concentration", "#4338ca"),
    ("pair_coherence", "Pair coherence", "normalized mean pair magnitude", "#7c3aed"),
    ("phase_rms_deg", "Target phase RMS", "lower is better", "#b45309"),
    ("effective_modes", "Effective modes", "participation number in W", "#1d4ed8"),
    ("population_distance", "Population mismatch", "total variation from target", "#c2410c"),
    ("negativity_mean", "Mean negativity", "one-vs-rest entanglement", "#6d28d9"),
    ("negativity_min", "Min negativity", "weakest one-vs-rest cut", "#7e22ce"),
    ("outside_w", "Outside W", "population outside W subspace", "#b91c1c"),
    ("ood", "OOD", "model-domain distance", "#c2410c"),
)
_INFERENCE_ROWS = (
    "Displayed rho",
    "Best F definition",
    "F ranking",
    "Diagnosis model",
    "Family model",
    "Primary mode",
    "Primary family",
    "Dominant error",
    "Estimator agreement",
    "Inference state",
)


class PhysicsMetricTile(QFrame):
    def __init__(self, title: str, subtitle: str, color: str):
        super().__init__()
        self._base_subtitle = subtitle
        self.setObjectName("physicsMetricTile")
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setStyleSheet(
            "QFrame#physicsMetricTile {"
            " background: #f7f8fa; border: 1px solid #c9ced6; border-radius: 4px;"
            "}"
        )
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 3, 6, 3)
        layout.setSpacing(0)

        title_label = QLabel(title)
        title_label.setStyleSheet("font-size: 9px; color: #4b5563; border: none;")
        self.value_label = QLabel("--")
        self.value_label.setStyleSheet(
            f"font-size: 16px; font-weight: 700; color: {color}; border: none;"
        )
        self.subtitle_label = QLabel(subtitle)
        self.subtitle_label.setStyleSheet("font-size: 8px; color: #6b7280; border: none;")
        self.subtitle_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.subtitle_label.setToolTip(subtitle)
        layout.addWidget(title_label)
        layout.addWidget(self.value_label)
        layout.addWidget(self.subtitle_label)

    def set_metric(self, value: str, subtitle: str | None = None) -> None:
        detail = str(subtitle or self._base_subtitle)
        self.value_label.setText(str(value))
        self.subtitle_label.setText(detail)
        self.subtitle_label.setToolTip(detail)


class PhysicsMetricGrid(QWidget):
    def __init__(self):
        super().__init__()
        layout = QGridLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setHorizontalSpacing(4)
        layout.setVerticalSpacing(3)
        self.tiles = {}
        for index, (key, title, subtitle, color) in enumerate(_METRIC_SPECS):
            tile = PhysicsMetricTile(title, subtitle, color)
            self.tiles[key] = tile
            layout.addWidget(tile, index // 8, index % 8)
        for column in range(8):
            layout.setColumnStretch(column, 1)

    def update_metrics(self, metrics: Mapping[str, Any] | None) -> None:
        metrics = metrics or {}
        if not metrics.get("available"):
            for tile in self.tiles.values():
                tile.set_metric("--")
            return

        self.tiles["best_fidelity"].set_metric(
            _decimal(metrics.get("best_fidelity")),
            f"{metrics.get('display_variant_label', '--')} · {metrics.get('display_metric_label', '--')}",
        )
        self.tiles["runner_up_fidelity"].set_metric(
            _decimal(metrics.get("runner_up_fidelity")),
            f"margin {_decimal(metrics.get('fidelity_margin'))}",
        )
        self.tiles["estimator_agreement"].set_metric(
            _percent(metrics.get("estimator_agreement")),
            f"sigma(F)={_decimal(metrics.get('estimator_spread'))}; N={int(metrics.get('estimator_count', 0) or 0)}",
        )
        self.tiles["w_population"].set_metric(_percent(metrics.get("w_population")))
        self.tiles["conditional_target_fidelity"].set_metric(_decimal(metrics.get("conditional_target_fidelity")))
        self.tiles["purity"].set_metric(
            _decimal(metrics.get("purity")),
            f"min eig {_signed(metrics.get('min_eigenvalue'))}; neg mass {_decimal(metrics.get('negative_eigenmass'))}",
        )
        self.tiles["entropy_bits"].set_metric(
            _bits(metrics.get("entropy_bits")),
            f"S/Smax {_percent(metrics.get('entropy_normalized'))}",
        )
        self.tiles["largest_eigenvalue"].set_metric(_decimal(metrics.get("largest_eigenvalue")))
        self.tiles["pair_coherence"].set_metric(_percent(metrics.get("pair_coherence")))
        self.tiles["phase_rms_deg"].set_metric(
            _degrees(metrics.get("phase_rms_deg")),
            f"phase alignment {_percent(metrics.get('phase_alignment'))}",
        )
        self.tiles["effective_modes"].set_metric(
            _decimal(metrics.get("effective_modes")),
            f"amplitude CV {_percent(metrics.get('amplitude_cv'))}",
        )
        self.tiles["population_distance"].set_metric(_percent(metrics.get("population_distance")))
        self.tiles["negativity_mean"].set_metric(
            _decimal(metrics.get("negativity_mean")),
            "mean over single-qubit cuts",
        )
        self.tiles["negativity_min"].set_metric(
            _decimal(metrics.get("negativity_min")),
            "weakest single-qubit cut",
        )
        self.tiles["outside_w"].set_metric(
            _percent(metrics.get("outside_w")),
            f"multi-click {_percent(metrics.get('multi_click_leakage'))}",
        )
        self.tiles["ood"].set_metric(
            _percent(metrics.get("ood")),
            f"model confidence {_percent(metrics.get('model_confidence'))}",
        )


class InferenceSummaryTable(QTableWidget):
    def __init__(self):
        super().__init__(len(_INFERENCE_ROWS), 2)
        self._items = {}
        self.setHorizontalHeaderLabels(["Inference / provenance", "Current result"])
        self.verticalHeader().setVisible(False)
        self.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.setSelectionMode(QAbstractItemView.NoSelection)
        self.setAlternatingRowColors(True)
        self.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.horizontalHeader().setFixedHeight(18)
        self.setFixedHeight(20 + len(_INFERENCE_ROWS) * 16)
        for row, label in enumerate(_INFERENCE_ROWS):
            self.setRowHeight(row, 16)
            name_item = QTableWidgetItem(label)
            value_item = QTableWidgetItem("--")
            name_item.setTextAlignment(Qt.AlignLeft | Qt.AlignVCenter)
            value_item.setTextAlignment(Qt.AlignLeft | Qt.AlignVCenter)
            self.setItem(row, 0, name_item)
            self.setItem(row, 1, value_item)
            self._items[label] = value_item

    def update_metrics(self, metrics: Mapping[str, Any] | None) -> None:
        metrics = metrics or {}
        values = {
            "Displayed rho": metrics.get("display_variant_label") or "--",
            "Best F definition": metrics.get("display_metric_label") or "--",
            "F ranking": _f_ranking_text(metrics),
            "Diagnosis model": metrics.get("diagnosis_backend") or "--",
            "Family model": metrics.get("state_family_backend") or "--",
            "Primary mode": _top_text(metrics.get("top_mode")),
            "Primary family": _top_text(metrics.get("top_family")),
            "Dominant error": _top_text(metrics.get("top_error")),
            "Estimator agreement": _estimator_text(metrics),
            "Inference state": metrics.get("inference_state") or "waiting for rho",
        }
        for label, value in values.items():
            self._items[label].setText(str(value))
            self._items[label].setToolTip(str(value))


def select_best_fidelity_candidate(summary: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """Select the highest finite F value while preserving metric provenance."""

    fits = top_fidelity_fits(summary or {}, limit=10000)
    if not fits:
        return None
    best = dict(fits[0])
    best["rank"] = 1
    if len(fits) > 1:
        best["runner_up"] = dict(fits[1])
    best["candidate_count"] = len(fits)
    return best


def compute_physics_dashboard_metrics(
    rho: Any,
    num_qubits: int,
    summary: Mapping[str, Any] | None = None,
    diagnosis: Mapping[str, Any] | None = None,
    *,
    rho_variants: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Compute display-only physics metrics from the rho that produced best F."""

    summary = summary or {}
    diagnosis = diagnosis or {}
    rho_variants = rho_variants or {}
    audit = summary.get("model_audit", {}) or {}
    best = select_best_fidelity_candidate(summary)
    display_variant = (best or {}).get("variant")
    display_rho = rho_variants.get(display_variant) if display_variant else None
    if display_rho is None:
        display_rho = rho

    selected_metrics = ((summary.get("variants", {}) or {}).get(display_variant, {}) or {})
    variant_diagnosis = selected_metrics.get("diagnosis") or {}
    selected_diagnosis = variant_diagnosis if variant_diagnosis.get("available") else diagnosis
    runner_up = (best or {}).get("runner_up") or {}
    result = {
        "available": False,
        "primary_variant": display_variant,
        "primary_metrics": selected_metrics,
        "display_variant": display_variant,
        "display_variant_label": (best or {}).get("variant_label"),
        "display_metric": (best or {}).get("metric"),
        "display_metric_label": (best or {}).get("metric_label"),
        "selected_metrics": selected_metrics,
        "best_fidelity": _optional_float((best or {}).get("fidelity")),
        "runner_up_fidelity": _optional_float(runner_up.get("fidelity")),
        "diagnosis_backend": selected_diagnosis.get("backend") or audit.get("diagnosis_backend"),
        "state_family_backend": selected_diagnosis.get("state_family_backend") or audit.get("state_family_backend"),
        "top_mode": _top_probability(selected_diagnosis.get("mode_probabilities")),
        "top_family": _top_probability(selected_diagnosis.get("state_family_probabilities")),
        "top_error": _top_probability(selected_diagnosis.get("error_source_probabilities")),
        "inference_state": audit.get("message") or selected_diagnosis.get("message") or "waiting for rho",
    }
    if display_rho is None or best is None:
        return result

    num_qubits = int(num_qubits)
    rho_array = np.asarray(display_rho, dtype=complex)
    expected_dim = 2 ** num_qubits
    if rho_array.shape != (expected_dim, expected_dim):
        result["inference_state"] = f"invalid rho shape {rho_array.shape}"
        return result

    physical_rho, spectrum = _project_density_matrix(rho_array)
    core, w_population = normalized_core_density(physical_rho, num_qubits)
    populations = np.clip(np.real(np.diag(core)), 0.0, 1.0)
    population_total = float(np.sum(populations))
    if population_total > _EPS:
        populations = populations / population_total

    target_amplitudes = _vector(summary.get("target_amplitudes"), num_qubits, 1.0)
    target_phases = _vector(summary.get("target_phases_rad"), num_qubits, 0.0)
    target_state = target_amplitudes * np.exp(1j * target_phases)
    target_norm = float(np.linalg.norm(target_state))
    if target_norm <= _EPS:
        target_state = np.ones(num_qubits, dtype=complex) / sqrt(num_qubits)
    else:
        target_state = target_state / target_norm
    target_populations = np.square(np.abs(target_state))
    conditional_target_fidelity = float(
        np.clip(np.real(np.vdot(target_state, core @ target_state)), 0.0, 1.0)
    )

    strength_matrix = np.zeros((num_qubits, num_qubits), dtype=float)
    phase_error_matrix = np.zeros_like(strength_matrix)
    for index, population in enumerate(populations):
        strength_matrix[index, index] = float(population)

    pair_strengths = []
    pair_phase_errors = []
    pair_weights = []
    for i in range(num_qubits):
        for j in range(i + 1, num_qubits):
            pair_weight = sqrt(max(float(populations[i] * populations[j]), 0.0))
            if pair_weight <= _EPS:
                continue
            strength = float(np.clip(abs(core[i, j]) / pair_weight, 0.0, 1.0))
            target_phase = float(target_phases[i] - target_phases[j])
            phase_error = float(np.angle(np.exp(1j * (np.angle(core[i, j]) - target_phase))))
            strength_matrix[i, j] = strength_matrix[j, i] = strength
            phase_error_matrix[i, j] = np.rad2deg(phase_error)
            phase_error_matrix[j, i] = -np.rad2deg(phase_error)
            pair_strengths.append(strength)
            pair_phase_errors.append(phase_error)
            pair_weights.append(pair_weight)

    if pair_weights:
        weights = np.asarray(pair_weights, dtype=float)
        strengths = np.asarray(pair_strengths, dtype=float)
        errors = np.asarray(pair_phase_errors, dtype=float)
        pair_coherence = float(np.average(strengths, weights=weights))
        phase_weights = weights * np.maximum(strengths, _EPS)
        phase_rms_deg = float(np.rad2deg(sqrt(np.average(np.square(errors), weights=phase_weights))))
        phase_alignment = float(np.clip((1.0 + np.average(np.cos(errors), weights=phase_weights)) / 2.0, 0.0, 1.0))
    else:
        pair_coherence = 0.0
        phase_rms_deg = 180.0
        phase_alignment = 0.0

    all_fits = top_fidelity_fits(summary, limit=10000)
    fidelity_values = np.asarray([float(item["fidelity"]) for item in all_fits], dtype=float)
    estimator_spread = float(np.std(fidelity_values)) if fidelity_values.size > 1 else 0.0
    estimator_agreement = float(np.clip(1.0 - estimator_spread / 0.15, 0.0, 1.0))
    best_fidelity = float(best["fidelity"])
    runner_up_fidelity = _optional_float(runner_up.get("fidelity"))

    ood = _optional_float(selected_metrics.get("ood"))
    if ood is None:
        ood = _optional_float(selected_diagnosis.get("unknown_probability"))
    top_family = result.get("top_family") or {}
    model_confidence = 1.0 - ood if ood is not None else _optional_float(top_family.get("probability"))
    effective_modes = 1.0 / max(float(np.sum(np.square(populations))), _EPS)
    amplitude_vector = np.sqrt(np.clip(populations, 0.0, None))
    amplitude_mean = float(np.mean(amplitude_vector))
    amplitude_cv = float(np.std(amplitude_vector) / amplitude_mean) if amplitude_mean > _EPS else 0.0
    population_distance = float(0.5 * np.sum(np.abs(populations - target_populations)))
    negativities = _single_qubit_negativities(physical_rho, num_qubits)

    result.update(
        {
            "available": True,
            "display_rho": physical_rho,
            "best_fidelity": best_fidelity,
            "target_fidelity": best_fidelity,
            "runner_up_fidelity": runner_up_fidelity,
            "fidelity_margin": None if runner_up_fidelity is None else best_fidelity - runner_up_fidelity,
            "w_population": float(np.clip(w_population, 0.0, 1.0)),
            "outside_w": float(np.clip(1.0 - w_population, 0.0, 1.0)),
            "conditional_target_fidelity": conditional_target_fidelity,
            "purity": spectrum["purity"],
            "entropy_bits": spectrum["entropy_bits"],
            "entropy_normalized": spectrum["entropy_normalized"],
            "largest_eigenvalue": spectrum["largest_eigenvalue"],
            "min_eigenvalue": spectrum["min_eigenvalue"],
            "negative_eigenmass": spectrum["negative_eigenmass"],
            "pair_coherence": pair_coherence,
            "phase_rms_deg": phase_rms_deg,
            "phase_alignment": phase_alignment,
            "effective_modes": effective_modes,
            "amplitude_cv": amplitude_cv,
            "population_distance": population_distance,
            "negativity_mean": float(np.mean(negativities)) if negativities else 0.0,
            "negativity_min": float(np.min(negativities)) if negativities else 0.0,
            "single_qubit_negativities": negativities,
            "multi_click_leakage": float(summary.get("multi_click_leakage", 0.0) or 0.0),
            "ood": ood,
            "model_confidence": model_confidence,
            "estimator_spread": estimator_spread,
            "estimator_count": int(fidelity_values.size),
            "estimator_agreement": estimator_agreement,
            "mode_populations": populations,
            "target_mode_populations": target_populations,
            "coherence_strength_matrix": strength_matrix,
            "phase_error_deg_matrix": phase_error_matrix,
        }
    )
    return result


def draw_physics_dashboard(figure, metrics: Mapping[str, Any] | None, num_qubits: int) -> None:
    figure.clear()
    metrics = metrics or {}
    if not metrics.get("available"):
        ax = figure.add_subplot(111)
        ax.axis("off")
        ax.text(0.5, 0.5, "waiting for rho and inference", ha="center", va="center", fontsize=10, color="#555555")
        figure.tight_layout(pad=0.5)
        return

    ax_quality = figure.add_subplot(1, 3, 1)
    labels = ["Best F", "P(W)", "F(target|W)", "Purity", "Coherence", "Phase align", "Min negativity", "Confidence"]
    values = np.asarray(
        [
            metrics.get("best_fidelity", 0.0),
            metrics.get("w_population", 0.0),
            metrics.get("conditional_target_fidelity", 0.0),
            metrics.get("purity", 0.0),
            metrics.get("pair_coherence", 0.0),
            metrics.get("phase_alignment", 0.0),
            metrics.get("negativity_min", 0.0),
            metrics.get("model_confidence", 0.0),
        ],
        dtype=float,
    )
    values = np.nan_to_num(values, nan=0.0)
    colors = ["#15803d" if value >= 0.8 else "#2563eb" if value >= 0.6 else "#d97706" if value >= 0.4 else "#b91c1c" for value in values]
    y = np.arange(len(labels))
    ax_quality.barh(y, values, color=colors, alpha=0.88)
    ax_quality.set_yticks(y)
    ax_quality.set_yticklabels(labels, fontsize=6)
    ax_quality.invert_yaxis()
    ax_quality.set_xlim(0.0, max(1.0, float(np.max(values)) * 1.08))
    ax_quality.set_xticks([0.0, 0.5, 1.0])
    ax_quality.tick_params(axis="x", labelsize=6)
    ax_quality.set_title("Best-F physical profile", fontsize=8)
    ax_quality.grid(axis="x", alpha=0.22)
    for yi, value in zip(y, values):
        ax_quality.text(min(ax_quality.get_xlim()[1] * 0.97, value + 0.02), yi, f"{value:.3f}", va="center", fontsize=6)
    for spine in ("top", "right", "left"):
        ax_quality.spines[spine].set_visible(False)

    ax_population = figure.add_subplot(1, 3, 2)
    populations = np.asarray(metrics.get("mode_populations"), dtype=float)
    target = np.asarray(metrics.get("target_mode_populations"), dtype=float)
    x = np.arange(int(num_qubits))
    width = 0.36
    ax_population.bar(x - width / 2, populations, width, label="best-F rho", color="#2563eb", alpha=0.86)
    ax_population.bar(x + width / 2, target, width, label="target", color="#374151", alpha=0.68)
    ax_population.set_xticks(x)
    ax_population.set_xticklabels([f"W{i + 1}" for i in range(int(num_qubits))], fontsize=6)
    ax_population.set_ylim(0.0, max(1.0, float(np.max(np.r_[populations, target])) * 1.12))
    ax_population.set_title(
        f"W population: Neff={metrics.get('effective_modes', 0.0):.2f}, TV={metrics.get('population_distance', 0.0):.3f}",
        fontsize=8,
    )
    ax_population.tick_params(axis="y", labelsize=6)
    ax_population.grid(axis="y", alpha=0.22)
    ax_population.legend(loc="upper right", fontsize=6, frameon=False)
    for spine in ("top", "right"):
        ax_population.spines[spine].set_visible(False)

    ax_phase = figure.add_subplot(1, 3, 3)
    strengths = np.asarray(metrics.get("coherence_strength_matrix"), dtype=float)
    phase_errors = np.asarray(metrics.get("phase_error_deg_matrix"), dtype=float)
    norm = Normalize(vmin=-180.0, vmax=180.0)
    from matplotlib import colormaps

    cmap = colormaps["twilight_shifted"]
    rgb = cmap(norm(phase_errors))[..., :3]
    saturation = np.clip(strengths[..., None], 0.08, 1.0)
    rgb = 1.0 - (1.0 - rgb) * saturation
    ax_phase.imshow(rgb, vmin=0.0, vmax=1.0)
    labels_w = [f"W{i + 1}" for i in range(int(num_qubits))]
    ax_phase.set_xticks(np.arange(int(num_qubits)))
    ax_phase.set_yticks(np.arange(int(num_qubits)))
    ax_phase.set_xticklabels(labels_w, fontsize=6)
    ax_phase.set_yticklabels(labels_w, fontsize=6)
    ax_phase.set_title("Target-relative pair coherence", fontsize=8)
    for i in range(int(num_qubits)):
        for j in range(int(num_qubits)):
            text = f"P {strengths[i, j]:.2f}" if i == j else f"C {strengths[i, j]:.2f}\n{phase_errors[i, j]:+.0f}°"
            ax_phase.text(j, i, text, ha="center", va="center", fontsize=5.5, color="#111827")
    scalar = ScalarMappable(norm=norm, cmap=cmap)
    colorbar = figure.colorbar(scalar, ax=ax_phase, fraction=0.045, pad=0.04)
    colorbar.set_label("phase error (deg)", fontsize=6)
    colorbar.ax.tick_params(labelsize=5)
    figure.tight_layout(pad=0.55)


def _project_density_matrix(rho: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    hermitian = 0.5 * (rho + rho.conj().T)
    trace = float(np.real(np.trace(hermitian)))
    if abs(trace) > _EPS:
        hermitian = hermitian / trace
    eigenvalues, eigenvectors = np.linalg.eigh(hermitian)
    min_eigenvalue = float(np.min(eigenvalues))
    negative_eigenmass = float(np.sum(np.abs(eigenvalues[eigenvalues < 0.0])))
    probabilities = np.clip(np.real(eigenvalues), 0.0, None)
    total = float(np.sum(probabilities))
    if total <= _EPS:
        probabilities = np.ones_like(probabilities) / probabilities.size
    else:
        probabilities = probabilities / total
    physical = eigenvectors @ np.diag(probabilities) @ eigenvectors.conj().T
    nonzero = probabilities[probabilities > _EPS]
    entropy_bits = float(-np.sum(nonzero * np.log2(nonzero)))
    max_entropy = log2(probabilities.size) if probabilities.size > 1 else 1.0
    return physical, {
        "purity": float(np.sum(np.square(probabilities))),
        "entropy_bits": entropy_bits,
        "entropy_normalized": float(entropy_bits / max_entropy),
        "largest_eigenvalue": float(np.max(probabilities)),
        "min_eigenvalue": min_eigenvalue,
        "negative_eigenmass": negative_eigenmass,
    }


def _single_qubit_negativities(rho: np.ndarray, num_qubits: int) -> list[float]:
    dims = [2] * int(num_qubits)
    tensor = rho.reshape(dims + dims)
    values = []
    for qubit in range(int(num_qubits)):
        axes = list(range(2 * int(num_qubits)))
        axes[qubit], axes[qubit + int(num_qubits)] = axes[qubit + int(num_qubits)], axes[qubit]
        partial_transpose = np.transpose(tensor, axes).reshape(rho.shape)
        eigenvalues = np.linalg.eigvalsh(0.5 * (partial_transpose + partial_transpose.conj().T))
        values.append(float(np.sum(np.abs(eigenvalues[eigenvalues < 0.0]))))
    return values


def _vector(value: Any, size: int, default: float) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=float).reshape(-1)
    except Exception:
        array = np.asarray([], dtype=float)
    if array.size < size:
        array = np.pad(array, (0, size - array.size), constant_values=default)
    return array[:size]


def _top_probability(values: Any) -> dict[str, Any] | None:
    if not isinstance(values, Mapping) or not values:
        return None
    label, value = max(values.items(), key=lambda item: float(item[1] or 0.0))
    return {"label": str(label), "probability": float(value or 0.0)}


def _top_text(value: Any) -> str:
    if not isinstance(value, Mapping):
        return "--"
    label = str(value.get("label") or "--").replace("_", " ")
    probability = _optional_float(value.get("probability"))
    return label if probability is None else f"{label} ({probability * 100.0:.1f}%)"


def _f_ranking_text(metrics: Mapping[str, Any]) -> str:
    best = _optional_float(metrics.get("best_fidelity"))
    second = _optional_float(metrics.get("runner_up_fidelity"))
    if best is None:
        return "--"
    text = f"best {best:.4f}"
    if second is not None:
        text += f"; second {second:.4f}; delta {best - second:.4f}"
    return text


def _estimator_text(metrics: Mapping[str, Any]) -> str:
    count = int(metrics.get("estimator_count", 0) or 0)
    spread = _optional_float(metrics.get("estimator_spread"))
    if not count or spread is None:
        return "--"
    return f"sigma(F)={spread:.4f} across {count} computed F values"


def _optional_float(value: Any) -> float | None:
    try:
        number = float(value)
    except Exception:
        return None
    return number if np.isfinite(number) else None


def _percent(value: Any) -> str:
    number = _optional_float(value)
    return "--" if number is None else f"{number * 100.0:.1f}%"


def _decimal(value: Any) -> str:
    number = _optional_float(value)
    return "--" if number is None else f"{number:.3f}"


def _signed(value: Any) -> str:
    number = _optional_float(value)
    return "--" if number is None else f"{number:+.3e}"


def _degrees(value: Any) -> str:
    number = _optional_float(value)
    return "--" if number is None else f"{number:.1f} deg"


def _bits(value: Any) -> str:
    number = _optional_float(value)
    return "--" if number is None else f"{number:.3f} bit"
