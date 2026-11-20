"""Load a portable shared B鈫扖 controller from the strict backbone plus fitted adapter export."""

from __future__ import annotations

import torch

from .classifier_n_full import V10ClassifierNFull
from .temporal_recovery_policy import (
    FEEDBACK_CONFIG,
    HIERARCHY_CONFIG,
    RecoveryController,
)
from .temporal_recovery_training import save_pt
from .temporal_recovery_long_training import checkpoint_digest


def export_and_load(root, device="cpu"):
    """Return one classifier object across N, a zero-hierarchy core and all four fitted controllers."""
    import json

    reg = json.loads((root / "shared_route.json").read_text())
    if not reg["shared_classifier_object"] or reg["control_evaluated_sizes"] != [3, 4]:
        raise ValueError("source did not use one classifier across N")
    backbone_path = root / reg["shared_core_file"]
    backbone = torch.load(backbone_path, weights_only=True, map_location="cpu")
    if (
        backbone["schema"] != "TrackBSharedBackboneV1"
        or backbone["feedback_config"] != FEEDBACK_CONFIG
    ):
        raise ValueError("unexpected strict classifier-plus-feedback constructor")
    state = torch.load(root / "long_state.pt", weights_only=True, map_location="cpu")
    if (
        state["binding"] != reg["binding"]
        or state["core_blocks"] != 128
        or state["adapter_blocks"] != 256
    ):
        raise ValueError("controller optimizer stage or binding mismatch")
    if set(state["states"]) != {"current", "summary", "ordered", "learner"}:
        raise ValueError("missing shared control adapter")
    ref = backbone["feedback_state_dict"]
    adapters = {}
    for name, incoming in state["states"].items():
        if any(
            not torch.equal(incoming.get("feedback." + k, torch.empty(0)), v)
            for k, v in ref.items()
        ):
            raise ValueError(
                "frozen feedback differs between complete envelope and adapter"
            )
        adapters[name] = {
            k: v for k, v in incoming.items() if not k.startswith("feedback.")
        }
    spec = dict(
        schema="shared_native_control_adapters_v1",
        backbone_file="shared_backbone.pt",
        backbone_sha256=checkpoint_digest(backbone_path),
        binding=reg["binding"],
        class_name="ml_recon.physics_hybrid_v10.temporal_recovery_policy.RecoveryController",
        hierarchy_config=HIERARCHY_CONFIG,
        raw_encoder={
            "in_features": 11,
            "hidden_features": 24,
            "out_features": 6,
            "activation": "GELU",
            "raw_query_in": 6,
            "raw_query_out": 9,
        },
        state_dicts=adapters,
        qualified_structure=reg["partition_control_qualified"],
        classifier_inference_domain="effective_single_event_v1",
        supported_control_sizes=[3, 4],
    )
    save_pt(spec, root / "shared_control_adapters.pt")
    return load_shared_route(root, device)


def load_shared_route(root, device="cpu"):
    """Strictly restore one classifier and each complete trained control path, with no optimizer dependency."""
    spec = torch.load(
        root / "shared_control_adapters.pt", map_location="cpu", weights_only=True
    )
    if (
        spec["schema"] != "shared_native_control_adapters_v1"
        or spec["hierarchy_config"] != HIERARCHY_CONFIG
    ):
        raise ValueError("unknown shared control architecture")
    file = root / spec["backbone_file"]
    if checkpoint_digest(file) != spec["backbone_sha256"]:
        raise ValueError("strict backbone differs from the portable adapter export")
    backbone = torch.load(file, map_location="cpu", weights_only=True)
    fields = {
        "schema",
        "classifier_config",
        "classifier_state_dict",
        "feedback_config",
        "feedback_state_dict",
        "hierarchy_config",
    }
    if (
        set(backbone) != fields
        or backbone["schema"] != "TrackBSharedBackboneV1"
        or backbone["feedback_config"] != FEEDBACK_CONFIG
    ):
        raise ValueError("invalid complete frozen shared backbone")
    if (
        backbone["classifier_config"].get("observation_law")
        != "effective_single_event_v1"
    ):
        raise ValueError("wrong public observation semantics")
    if (
        type(spec["qualified_structure"]) != bool
        or spec["binding"].get("partition_control_qualified")
        != spec["qualified_structure"]
        or (
            spec["qualified_structure"]
            and not backbone["classifier_config"]
            .get("training_ideas", {})
            .get("esm", False)
        )
    ):
        raise ValueError("inconsistent ESM qualification in reconstructed control")
    cls = V10ClassifierNFull(**backbone["classifier_config"])
    cls.load_state_dict(backbone["classifier_state_dict"], strict=True)
    cls.partition_control_qualified = bool(spec["qualified_structure"])
    cls = cls.to(device).requires_grad_(False).eval()
    models = {}
    for arm in ("zero", "current", "summary", "ordered", "learner"):
        ctrl = RecoveryController()
        ctrl.feedback.load_state_dict(backbone["feedback_state_dict"], strict=True)
        if arm != "zero":
            want = {k for k in ctrl.state_dict() if not k.startswith("feedback.")}
            incoming = spec["state_dicts"][arm]
            if set(incoming) != want:
                raise ValueError("incomplete temporal adapter")
            full = ctrl.state_dict()
            for k, v in incoming.items():
                if (
                    v.shape != full[k].shape
                    or v.dtype != full[k].dtype
                    or not torch.isfinite(v).all()
                ):
                    raise ValueError("invalid temporal adapter parameter: " + k)
                full[k] = v
            ctrl.load_state_dict(full, strict=True)
        models[arm] = ctrl.to(device).requires_grad_(False).eval()
    return cls, models, spec
