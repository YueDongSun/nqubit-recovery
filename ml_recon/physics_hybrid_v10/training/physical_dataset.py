"""On-demand AOD-W_N physical stream dataset with no persisted raw rows."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from typing import Iterator, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from ml_recon.physics_hybrid_v10.synthetic_manifest import (
    V10CounterBasedRng,
    V10SyntheticManifest,
)
from ml_recon.data_v2_4.hashing import sha256_json
from ml_recon.physics_hybrid_v10.four_cell_layout import (
    TRACK_A_FOUR_CELL_SEMANTICS,
    four_cell_layout_metadata,
    require_four_cell_layout,
)
from ml_recon.v10_recoverability_foundry.contracts.physical_budget import (
    Budget_provenance,
    Operation_registry_binding,
    Physical_budget,
    allocate_mode_budget,
)
from ml_recon.v10_recoverability_foundry.data.aod_operation_schedule import (
    population_pair_quadrature_candidates,
)
from ml_recon.v10_recoverability_foundry.data.aod_w_renderer import render_aod_w
from ml_recon.v10_recoverability_foundry.physics.aod_detector_forward import (
    Aod_noise_model,
)
from ml_recon.v10_recoverability_foundry.physics.aod_w_state import Aod_w_state

PHYSICAL_DATASET_RECIPE_SCHEMA = "V10PhysicalAodDatasetRecipeV3"
PHYSICAL_OPERATION_POISSON_RECIPE_SCHEMA = "V10PhysicalAodDatasetRecipeV4"
PHYSICAL_UNCERTAINTY_COORDINATE_SCHEMA = (
    "V10PhysicalUncertaintyRelativeLogAmplitudePhaseV1"
)
DEFAULT_N_VALUES = (3, 4, 6, 8, 12, 18, 24, 36)


@dataclass(frozen=True)
class Physical_dataset_recipe:
    n_values: tuple[int, ...] = DEFAULT_N_VALUES
    requested_mode_shots: int = 1024
    samples_per_n: int = 256
    global_seed: int = 20261001
    schedule_family: str = "W_COMPLETE_GENERAL_N"
    noise_per_cell_success_probability: float = 0.005
    split_sizes: tuple[int, int, int] | None = None
    four_cell_layout_schema: str = "V10FourCellLayoutV1"
    four_cell_semantics: str = TRACK_A_FOUR_CELL_SEMANTICS
    four_cell_layout_sha: str = ""
    operation_input_recipe: str = "legacy_v1"
    sample_log_brightness_sigma: float = 0.0
    synthetic_operation_log_rate_sigma: float = 0.0
    registered_synthetic_operation_profile: bool = False
    query_operation_fraction: float = 0.5

    def validate(self) -> None:
        if (
            not isinstance(self.n_values, tuple)
            or not self.n_values
            or any(
                type(value) is not int or value not in DEFAULT_N_VALUES
                for value in self.n_values
            )
            or tuple(sorted(set(self.n_values))) != self.n_values
        ):
            raise ValueError(
                "physical recipe N values must be a canonical primary panel subset"
            )
        if (
            type(self.samples_per_n) is not int
            or self.samples_per_n < 1
            or type(self.requested_mode_shots) is not int
            or self.requested_mode_shots < 1
            or type(self.global_seed) is not int
            or self.global_seed < 0
            or self.schedule_family != "W_COMPLETE_GENERAL_N"
        ):
            raise ValueError(
                "physical recipe has invalid scope, sample pool, budget or seed"
            )
        if (
            isinstance(self.noise_per_cell_success_probability, bool)
            or not np.isfinite(self.noise_per_cell_success_probability)
            or not 0
            <= self.noise_per_cell_success_probability
            <= 1 / max(self.n_values)
        ):
            raise ValueError(
                "per-cell working point must be valid for every possible state in the panel"
            )
        if self.split_sizes is not None and (
            not isinstance(self.split_sizes, tuple)
            or len(self.split_sizes) != 3
            or any(type(value) is not int or value < 1 for value in self.split_sizes)
        ):
            raise ValueError(
                "explicit train/validation/test sizes must be positive integer triples"
            )
        if self.operation_input_recipe not in ("legacy_v1", "operation_poisson_v1"):
            raise ValueError("operation input recipe must be legacy_v1 or operation_poisson_v1")
        for name in ("sample_log_brightness_sigma", "synthetic_operation_log_rate_sigma"):
            value = getattr(self, name)
            if isinstance(value, bool) or not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be a finite nonnegative synthetic hyperparameter")
        if type(self.registered_synthetic_operation_profile) is not bool:
            raise ValueError("registered synthetic operation profile must be strict bool")
        if (
            isinstance(self.query_operation_fraction, bool)
            or not np.isfinite(self.query_operation_fraction)
            or not 0.0 < self.query_operation_fraction < 1.0
        ):
            raise ValueError("query operation fraction must lie strictly in (0, 1)")
        expected_layout = four_cell_layout_metadata(TRACK_A_FOUR_CELL_SEMANTICS)
        if (
            self.four_cell_layout_schema != expected_layout["four_cell_layout_schema"]
            or self.four_cell_semantics != expected_layout["four_cell_semantics"]
            or self.four_cell_layout_sha not in ("", expected_layout["four_cell_layout_sha"])
        ):
            raise ValueError("physical recipe four-cell layout identity must be canonical")

    def to_dict(self) -> dict:
        self.validate()
        if self.operation_input_recipe == "legacy_v1":
            # This exact V3 projection is a replay boundary for frozen E0 data.
            value = {
                name: getattr(self, name)
                for name in (
                    "n_values", "requested_mode_shots", "samples_per_n", "global_seed",
                    "schedule_family", "noise_per_cell_success_probability", "split_sizes",
                    "four_cell_layout_schema", "four_cell_semantics", "four_cell_layout_sha",
                )
            }
        else:
            value = {"schema": PHYSICAL_OPERATION_POISSON_RECIPE_SCHEMA, **asdict(self)}
        if not value["four_cell_layout_sha"]:
            value["four_cell_layout_sha"] = four_cell_layout_metadata(
                TRACK_A_FOUR_CELL_SEMANTICS
            )["four_cell_layout_sha"]
        return value

    def sha256(self) -> str:
        return sha256_json(self.to_dict())

    def stream_recipe(self) -> dict:
        return {**self.to_dict(), "recipe_sha256": self.sha256()}


class Physical_aod_dataset(Dataset[dict[str, object]]):
    """Map-style source that renders each sample at access time only."""

    def __init__(
        self,
        manifest: V10SyntheticManifest,
        *,
        split: str,
        recipe: Physical_dataset_recipe = Physical_dataset_recipe(),
    ) -> None:
        if split not in ("train", "validation", "test"):
            raise ValueError("split must be train, validation, or test")
        recipe.validate()
        manifest.validate()
        if recipe.global_seed != manifest.global_seed or not set(
            recipe.n_values
        ) <= set(manifest.N_values):
            raise ValueError(
                "physical recipe seed/N scope must agree with its source manifest"
            )
        self.manifest, self.split, self.recipe = manifest, split, recipe
        self.recipe_sha = recipe.sha256()
        entries = []
        requested = (
            None
            if recipe.split_sizes is None
            else recipe.split_sizes[("train", "validation", "test").index(split)]
        )
        for n_index, n_mode in enumerate(recipe.n_values):
            candidates = [
                (n_mode, index)
                for index in range(recipe.samples_per_n)
                if manifest.split_for_sample(self._latent_group_key(n_mode, index))
                == split
            ]
            if requested is not None:
                quota = requested // len(recipe.n_values) + int(
                    n_index < requested % len(recipe.n_values)
                )
                if len(candidates) < quota:
                    raise ValueError(
                        "declared pool cannot fill the frozen split quota; enlarge the pool before admission"
                    )
                candidates = candidates[:quota]
            entries.extend(candidates)
        self._entries = tuple(entries)
        if len(self._entries) != len(set(self._entries)):
            raise ValueError("physical split membership contains duplicate rows")
        self.latent_group_keys = tuple(
            self._latent_group_key(n_mode, index)
            for n_mode, index in self._entries
        )
        self.sample_ids = tuple(
            self._sample_id(n_mode, index) for n_mode, index in self._entries
        )
        self.n_mode_counts = {
            n_mode: sum(entry_n_mode == n_mode for entry_n_mode, _ in self._entries)
            for n_mode in recipe.n_values
        }
        self.membership_sha = sha256_json(
            {
                "schema": (PHYSICAL_DATASET_RECIPE_SCHEMA if recipe.operation_input_recipe == "legacy_v1" else PHYSICAL_OPERATION_POISSON_RECIPE_SCHEMA),
                "split": split,
                "manifest_sha": manifest.sha256(),
                "recipe_sha": self.recipe_sha,
                "latent_group_keys": list(self.latent_group_keys),
                "sample_ids": list(self.sample_ids),
                "n_mode_counts": self.n_mode_counts,
            }
        )

    def _sample_id(self, n_mode: int, index: int) -> str:
        return f"physical_aod:{self.recipe_sha}:N{n_mode}:{index:012d}"

    @staticmethod
    def _latent_group_key(n_mode: int, index: int) -> str:
        """CRN identity excluding schedule, budget, and model-arm choices."""
        return f"physical_aod_latent:K1:N{n_mode}:{index:012d}"

    def __len__(self) -> int:
        return len(self._entries)

    def n_mode_at(self, item_index: int) -> int:
        return self._entries[item_index][0]

    def __getitem__(self, item_index: int) -> dict[str, object]:
        n_mode, local_index = self._entries[item_index]
        sample_id = self._sample_id(n_mode, local_index)
        latent_group_key = self._latent_group_key(n_mode, local_index)
        rng = V10CounterBasedRng(
            latent_group_key,
            generator_sha=self.manifest.generator_sha,
            global_seed=self.recipe.global_seed,
            split=self.split,
        )
        schedule = population_pair_quadrature_candidates(n_mode)
        registry = Operation_registry_binding(
            schedule.operation_ids,
            schedule.operation_roles,
            len(schedule.operation_ids),
            schedule.sha256(),
            hashlib.sha256((schedule.sha256() + "registry").encode()).hexdigest(),
            hashlib.sha256((schedule.sha256() + "commands").encode()).hexdigest(),
            "negative_read_phase",
            hashlib.sha256(f"mode_coordinates:{n_mode}".encode()).hexdigest(),
        )
        repeats = allocate_mode_budget(
            n_mode=n_mode,
            operation_count=len(schedule.operation_ids),
            requested_mode_shots=self.recipe.requested_mode_shots,
        )
        budget = Physical_budget(
            n_mode,
            registry,
            repeats,
            Budget_provenance(
                "synthetic_fixture",
                "physical_aod_stream",
                self.recipe_sha,
                "SYNTHETIC_DECLARED",
            ),
            requested_mode_shots=self.recipe.requested_mode_shots,
        )
        state_rng = rng.stream("state")
        coefficients = state_rng.normal(size=n_mode) + 1j * state_rng.normal(
            size=n_mode
        )
        state = Aod_w_state(coefficients, np.ones(n_mode, dtype=bool))
        # The legacy forward fixes a source-rate baseline from its actual herald
        # probability. Registered operation offsets alter lambda_o, while the
        # declared synthetic exposure remains a separate unit opportunity.
        legacy_rendered = render_aod_w(
            state=state, schedule=schedule, budget=budget,
            noise=Aod_noise_model(per_cell_success_probability=self.recipe.noise_per_cell_success_probability),
            keyed_rng=rng, recipe_id=self.recipe_sha, group_id=f"{self.split}:N{n_mode}:{local_index}",
        )
        legacy_herald = np.asarray(legacy_rendered.truth["success_probability_by_operation"], dtype=np.float64)
        mu_base = -np.log1p(-np.clip(legacy_herald, 0.0, 1.0 - 1e-12))
        profile_offsets = self._registered_synthetic_operation_offsets(schedule.operation_ids)
        operation_exposure = np.ones_like(profile_offsets, dtype=np.float32)
        read_throughput = schedule.read_throughput.astype(np.float32)
        brightness_drift = float(rng.stream("operation_global_brightness_drift").normal(scale=self.recipe.sample_log_brightness_sigma))
        operation_log_rate_offset = profile_offsets + brightness_drift
        operation_rate = (mu_base * np.exp(operation_log_rate_offset)).astype(np.float32)
        operation_mean = (operation_exposure * operation_rate).astype(np.float32)
        rendered = legacy_rendered if self.recipe.operation_input_recipe == "legacy_v1" else render_aod_w(
            state=state, schedule=schedule, budget=budget,
            noise=Aod_noise_model(per_cell_success_probability=self.recipe.noise_per_cell_success_probability),
            keyed_rng=rng, recipe_id=self.recipe_sha, group_id=f"{self.split}:N{n_mode}:{local_index}",
            operation_excitation_mean=operation_mean,
            topology_observable_mask=np.zeros(len(schedule.operation_ids), dtype=bool),
        )
        read_vectors = schedule.amplitudes * np.exp(-1j * schedule.phases)
        # This synthetic recipe declares unit exposure; it is never a measured
        # calibration receipt or an apparatus parameter.
        poisson_truth = rendered.truth if self.recipe.operation_input_recipe == "operation_poisson_v1" else {}
        sector_probabilities = (
            np.stack((poisson_truth["poisson_p0"], poisson_truth["poisson_p1"], poisson_truth["poisson_pge2"]), axis=-1).astype(np.float32)
            if poisson_truth else np.stack((np.exp(-operation_mean), operation_mean * np.exp(-operation_mean), 1.0 - np.exp(-operation_mean) * (1.0 + operation_mean)), axis=-1).astype(np.float32)
        )
        observed_population = rendered.observation["four_cell_counts"][:, 3].astype(
            np.float64
        )
        observed_amplitude = np.sqrt(
            (observed_population[:n_mode] + 1e-6)
            / (observed_population[:n_mode].sum() + n_mode * 1e-6)
        )
        active_reference_mode = int(
            np.flatnonzero(np.abs(state.coefficients) > 1e-10)[0]
        )
        reference_phase = np.angle(state.coefficients[active_reference_mode])
        # Log-amplitude ratio and relative phase share the W-head's intrinsic
        # Aitchison/Helmert control units. The truth reference removes only the
        # unobservable gauge and is never an input feature.
        target_amplitude_error = np.log(
            np.abs(state.coefficients).clip(1e-10)
        ) - np.log(observed_amplitude.clip(1e-10))
        target_phase_error = np.angle(
            state.coefficients * np.exp(-1j * reference_phase)
        )
        sector = 1  # primary stream is K=1; higher sectors are unsupported
        coordinates = np.stack(
            (np.linspace(-1.0, 1.0, n_mode), np.zeros(n_mode), np.ones(n_mode)), axis=-1
        ).astype(np.float32)
        query_mask = np.zeros(len(schedule.operation_ids), dtype=bool)
        if self.recipe.operation_input_recipe == "operation_poisson_v1":
            query_mask = rng.stream("operation_query_mask").random(len(query_mask)) < self.recipe.query_operation_fraction
            query_mask[0], query_mask[-1] = False, True
        query_stream_sha = sha256_json({
            "sample_id": rng.sample_id, "generator_sha": rng.generator_sha,
            "global_seed": rng.global_seed, "split": rng.split,
            "stream": "operation_query_mask", "query_mask": query_mask.tolist(),
        })
        return {
            "sample_id": sample_id,
            "latent_group_key": latent_group_key,
            "split": self.split,
            "N": n_mode,
            "recipe_sha": self.recipe_sha,
            "physical_budget_sha": rendered.metadata["physical_budget_sha"],
            **{
                key: rendered.observation[key]
                for key in four_cell_layout_metadata(TRACK_A_FOUR_CELL_SEMANTICS)
            },
            "four_cell_counts": rendered.observation["four_cell_counts"].astype(
                np.float32
            ),
            "read_vectors": read_vectors.astype(np.complex64),
            "operation_ids": tuple(schedule.operation_ids),
            "operation_roles": tuple(schedule.operation_roles),
            "operation_registry_sha": registry.operation_registry_sha,
            "operation_command_sha": registry.command_table_sha,
            "operation_coordinate_sha": registry.mode_coordinate_schema_sha,
            "operation_exposure_per_opportunity": operation_exposure,
            "operation_exposure_mask": np.ones_like(operation_exposure, dtype=bool),
            "read_throughput": read_throughput,
            "read_throughput_mask": np.ones_like(read_throughput, dtype=bool),
            "operation_log_rate_offset": operation_log_rate_offset.astype(np.float32),
            "registered_synthetic_operation_profile_source": "UNIFORM_DEFAULT" if not self.recipe.registered_synthetic_operation_profile else "GLOBAL_SEED_GENERATOR_OPERATION_ID",
            "registered_synthetic_operation_profile_sha": sha256_json({"operation_ids": list(schedule.operation_ids), "offsets": profile_offsets.tolist(), "global_seed": self.recipe.global_seed, "generator_sha": self.manifest.generator_sha}),
            "target_effective_mu_o": np.asarray(poisson_truth.get("operation_excitation_mean", operation_mean), dtype=np.float32),
            "target_effective_mu_o_mask": np.ones_like(operation_exposure, dtype=bool),
            "target_excitation_rate_o": operation_rate,
            "target_excitation_rate_o_mask": np.ones_like(operation_exposure, dtype=bool),
            "target_sector_probabilities": sector_probabilities,
            "target_sector_probability_mask": np.ones_like(operation_exposure, dtype=bool),
            "diagnostic_k2_local_probability": np.asarray(poisson_truth.get("local_k2_mark_probability", np.zeros((len(schedule.operation_ids), n_mode))), dtype=np.float32).sum(-1),
            "diagnostic_k2_cross_probability": np.triu(np.asarray(poisson_truth.get("cross_k2_mark_probability", np.zeros((len(schedule.operation_ids), n_mode, n_mode))), dtype=np.float32), k=1).sum((-1, -2)),
            "target_k2_topology_mask": np.zeros_like(operation_exposure, dtype=bool),
            "mode_features": coordinates,
            "repeats_by_operation": rendered.observation["repeats_by_operation"].astype(
                np.float32
            ),
            "crn_fingerprint": hashlib.sha256(
                (latent_group_key + self.split).encode()
            ).hexdigest(),
            "target_amplitude_error": target_amplitude_error.astype(np.float32),
            "target_phase_error": target_phase_error.astype(np.float32),
            "target_reference_mode": active_reference_mode,
            "target_coordinate_schema": PHYSICAL_UNCERTAINTY_COORDINATE_SCHEMA,
            "target_sector": sector,
            "target_measurement_probability": rendered.truth[
                "four_cell_probabilities"
            ].astype(np.float32),
            "target_R1_re": state.density_matrix().real.astype(np.float32),
            "target_R1_im": state.density_matrix().imag.astype(np.float32),
            "target_conditional_rho1_re": state.density_matrix().real.astype(np.float32),
            "target_conditional_rho1_im": state.density_matrix().imag.astype(np.float32),
            "target_W_populations": np.square(np.abs(state.coefficients)).astype(
                np.float32
            ),
            "target_mechanism_mask": np.zeros(15, dtype=bool),
            "operation_input_recipe": self.recipe.operation_input_recipe,
            "operation_query_mask": query_mask,
            "operation_query_mask_stream_sha": query_stream_sha,
        }

    def _registered_synthetic_operation_offsets(self, operation_ids: Sequence[str]) -> np.ndarray:
        """Stable diagnostic profile: global recipe identity and operation ID only."""
        if not self.recipe.registered_synthetic_operation_profile:
            return np.zeros(len(operation_ids), dtype=np.float64)
        offsets = []
        for operation_id in operation_ids:
            payload = f"{self.recipe.global_seed}|{self.manifest.generator_sha}|{operation_id}".encode()
            seed = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
            offsets.append(np.random.Generator(np.random.Philox(seed)).normal(scale=self.recipe.synthetic_operation_log_rate_sigma))
        return np.asarray(offsets, dtype=np.float64)


class Homogeneous_n_batch_sampler(Sampler[list[int]]):
    """Stable, epoch-seeded batches that never mix mode cardinalities."""

    def __init__(
        self,
        dataset: Physical_aod_dataset,
        batch_size: int,
        *,
        epoch: int = 0,
        shuffle: bool = True,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.dataset, self.batch_size, self.epoch = dataset, int(batch_size), int(epoch)
        if type(shuffle) is not bool:
            raise ValueError("sampler shuffle must be a strict boolean")
        self.shuffle = shuffle

    def __iter__(self) -> Iterator[list[int]]:
        groups: dict[int, list[int]] = {}
        for index in range(len(self.dataset)):
            groups.setdefault(self.dataset.n_mode_at(index), []).append(index)
        random = np.random.default_rng(self.dataset.recipe.global_seed + self.epoch)
        batches = []
        for n_mode in sorted(groups):
            indices = groups[n_mode]
            if self.shuffle:
                random.shuffle(indices)
            batches.append(
                [
                    indices[start : start + self.batch_size]
                    for start in range(0, len(indices), self.batch_size)
                ]
            )
        for round_index in range(max((len(group) for group in batches), default=0)):
            for group in batches:
                if round_index < len(group):
                    yield group[round_index]

    def __len__(self) -> int:
        return sum(
            (
                sum(
                    1
                    for index in range(len(self.dataset))
                    if self.dataset.n_mode_at(index) == n_mode
                )
                + self.batch_size
                - 1
            )
            // self.batch_size
            for n_mode in self.dataset.recipe.n_values
        )


def collate_physical_aod_batch(rows: Sequence[dict[str, object]]) -> dict[str, object]:
    """Pad a homogeneous batch; truth labels never enter the model whitelist."""
    if not rows or len({int(row["N"]) for row in rows}) != 1:
        raise ValueError("physical AOD batches must be nonempty and homogeneous in N")
    n_mode = int(rows[0]["N"])
    for row in rows:
        require_four_cell_layout(row, expected_semantics=TRACK_A_FOUR_CELL_SEMANTICS)
    operation_count = max(np.asarray(row["four_cell_counts"]).shape[0] for row in rows)
    batch_size = len(rows)
    counts = torch.zeros(batch_size, operation_count, 4)
    target_counts = torch.zeros(batch_size, operation_count, 4)
    reads = torch.zeros(batch_size, operation_count, n_mode, dtype=torch.complex64)
    repeats = torch.zeros(batch_size, operation_count)
    operation_mask = torch.zeros(batch_size, operation_count, dtype=torch.bool)
    exposure = torch.zeros(batch_size, operation_count)
    throughput = torch.zeros(batch_size, operation_count)
    exposure_mask = torch.zeros(batch_size, operation_count, dtype=torch.bool)
    throughput_mask = torch.zeros(batch_size, operation_count, dtype=torch.bool)
    effective_mu = torch.zeros(batch_size, operation_count)
    effective_mu_mask = torch.zeros(batch_size, operation_count, dtype=torch.bool)
    excitation_rate = torch.zeros(batch_size, operation_count)
    excitation_rate_mask = torch.zeros(batch_size, operation_count, dtype=torch.bool)
    sector_probabilities = torch.zeros(batch_size, operation_count, 3)
    sector_probability_mask = torch.zeros(batch_size, operation_count, dtype=torch.bool)
    k2_local = torch.zeros(batch_size, operation_count)
    k2_cross = torch.zeros(batch_size, operation_count)
    k2_topology_mask = torch.zeros(batch_size, operation_count, dtype=torch.bool)
    for index, row in enumerate(rows):
        count = torch.from_numpy(np.asarray(row["four_cell_counts"]))
        op_count = count.shape[0]
        counts[index, :op_count], reads[index, :op_count] = (
            count,
            torch.from_numpy(np.asarray(row["read_vectors"])),
        )
        target_counts[index, :op_count] = count
        row_repeats = torch.from_numpy(np.asarray(row["repeats_by_operation"]))
        repeats[index, :op_count] = row_repeats
        operation_mask[index, :op_count] = row_repeats.gt(0)
        for target, name, dtype in (
            (exposure, "operation_exposure_per_opportunity", torch.float32),
            (throughput, "read_throughput", torch.float32),
            (effective_mu, "target_effective_mu_o", torch.float32),
            (excitation_rate, "target_excitation_rate_o", torch.float32),
            (k2_local, "diagnostic_k2_local_probability", torch.float32),
            (k2_cross, "diagnostic_k2_cross_probability", torch.float32),
        ):
            target[index, :op_count] = torch.as_tensor(row[name], dtype=dtype)
        for target, name in ((exposure_mask, "operation_exposure_mask"), (throughput_mask, "read_throughput_mask"), (effective_mu_mask, "target_effective_mu_o_mask"), (excitation_rate_mask, "target_excitation_rate_o_mask"), (sector_probability_mask, "target_sector_probability_mask"), (k2_topology_mask, "target_k2_topology_mask")):
            target[index, :op_count] = torch.as_tensor(row[name], dtype=torch.bool)
        sector_probabilities[index, :op_count] = torch.as_tensor(row["target_sector_probabilities"])
    operation_poisson = rows[0]["operation_input_recipe"] == "operation_poisson_v1"
    query_mask = torch.zeros_like(operation_mask)
    if operation_poisson:
        for index, row in enumerate(rows):
            query = np.asarray(row["operation_query_mask"], dtype=bool)
            if query.size < 2 or not np.any(query) or np.all(query):
                raise ValueError("operation query stream must contain context and query")
            query_mask[index, :query.size] = torch.from_numpy(query)
    observed_count_mask = operation_mask & ~query_mask
    likelihood_target_mask = operation_mask & query_mask
    counts = counts * observed_count_mask.unsqueeze(-1)
    return {
        "physical_classifier_schema": "V10PhysicalClassifierBatchV2",
        **four_cell_layout_metadata(TRACK_A_FOUR_CELL_SEMANTICS),
        "four_cell_counts": counts,
        "target_four_cell_counts": target_counts,
        "target_herald_counts": target_counts[..., 2] + target_counts[..., 3],
        "read_vectors": reads,
        "mode_features": torch.stack(
            [torch.from_numpy(np.asarray(row["mode_features"])) for row in rows]
        ),
        "mode_mask": torch.ones(batch_size, n_mode, dtype=torch.bool),
        "operation_mask": operation_mask,
        "observed_count_mask": observed_count_mask,
        "likelihood_target_mask": likelihood_target_mask,
        "operation_exposure_per_opportunity": exposure,
        "operation_exposure_mask": exposure_mask,
        "read_throughput": throughput,
        "read_throughput_mask": throughput_mask,
        "target_effective_mu_o": effective_mu,
        "target_effective_mu_o_mask": effective_mu_mask,
        "target_excitation_rate_o": excitation_rate,
        "target_excitation_rate_o_mask": excitation_rate_mask,
        "target_sector_probabilities": sector_probabilities,
        "target_sector_probability_mask": sector_probability_mask,
        "diagnostic_k2_local_probability": k2_local,
        "diagnostic_k2_cross_probability": k2_cross,
        "target_k2_topology_mask": k2_topology_mask,
        "repeats_by_operation": repeats,
        "target_amplitude_error": torch.stack(
            [
                torch.from_numpy(np.asarray(row["target_amplitude_error"]))
                for row in rows
            ]
        ),
        "target_phase_error": torch.stack(
            [torch.from_numpy(np.asarray(row["target_phase_error"])) for row in rows]
        ),
        "target_reference_mode": torch.tensor(
            [row["target_reference_mode"] for row in rows]
        ),
        "target_coordinate_schema": PHYSICAL_UNCERTAINTY_COORDINATE_SCHEMA,
        "target_mode_mask": torch.ones(batch_size, n_mode, dtype=torch.bool),
        "target_sector": torch.tensor([row["target_sector"] for row in rows]),
        "target_sector_mask": torch.ones(batch_size, dtype=torch.bool),
        "target_measurement_probability": torch.stack(
            [
                torch.nn.functional.pad(
                    torch.from_numpy(np.asarray(row["target_measurement_probability"])),
                    (
                        0,
                        0,
                        0,
                        operation_count
                        - len(np.asarray(row["target_measurement_probability"])),
                    ),
                )
                for row in rows
            ]
        ),
        "target_R1_re": torch.stack(
            [torch.from_numpy(np.asarray(row["target_R1_re"])) for row in rows]
        ),
        "target_R1_im": torch.stack(
            [torch.from_numpy(np.asarray(row["target_R1_im"])) for row in rows]
        ),
        "target_conditional_rho1_re": torch.stack([torch.from_numpy(np.asarray(row["target_conditional_rho1_re"])) for row in rows]),
        "target_conditional_rho1_im": torch.stack([torch.from_numpy(np.asarray(row["target_conditional_rho1_im"])) for row in rows]),
        "target_W_populations": torch.stack(
            [torch.from_numpy(np.asarray(row["target_W_populations"])) for row in rows]
        ),
        "target_measurement_mask": likelihood_target_mask if operation_poisson else operation_mask,
        "sample_id": [str(row["sample_id"]) for row in rows],
        "acquisition_group_id": [str(row["latent_group_key"]) for row in rows],
        "operation_ids": [tuple(row["operation_ids"]) for row in rows],
        "operation_roles": [tuple(row["operation_roles"]) for row in rows],
        "operation_registry_sha": [str(row["operation_registry_sha"]) for row in rows],
        "operation_command_sha": [str(row["operation_command_sha"]) for row in rows],
        "operation_coordinate_sha": [str(row["operation_coordinate_sha"]) for row in rows],
        "registered_synthetic_operation_profile_source": [str(row["registered_synthetic_operation_profile_source"]) for row in rows],
        "registered_synthetic_operation_profile_sha": [str(row["registered_synthetic_operation_profile_sha"]) for row in rows],
        "operation_query_mask_stream_sha": [str(row["operation_query_mask_stream_sha"]) for row in rows],
        "calibration_receipt_mask": torch.zeros(batch_size, operation_count, dtype=torch.bool),
        "cell_id": [f"n_{n_mode}"] * batch_size,
        "recipe_sha": str(rows[0]["recipe_sha"]),
        "operation_input_recipe": str(rows[0]["operation_input_recipe"]),
    }
