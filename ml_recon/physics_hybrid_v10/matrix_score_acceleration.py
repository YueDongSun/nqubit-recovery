"""Reuse immutable simulated panels and move only frozen public-input forwards to a device."""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

import torch

from .training_ideas_data import public_projector_inputs


@contextmanager
def _exclusive(path):
    # Shared cloud workers publish per-input cache entries on the same POSIX tmpfs.
    import fcntl

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield


def cached_tensor_tree(folder, name, create):
    """Publish a complete panel or classical fit once. An interrupted writer leaves no final file."""
    if folder is None:
        return create()
    path = Path(folder) / (name + ".pt")
    with _exclusive(path.with_suffix(".lock")):
        if path.exists():
            return torch.load(path, map_location="cpu", weights_only=True)
        value = create()
        with NamedTemporaryFile(dir=path.parent, delete=False) as f:
            temporary = Path(f.name)
        try:
            torch.save(value, temporary)
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
        return value


def prior_digest(prior):
    return hashlib.sha256(
        prior.detach().cpu().contiguous().numpy().tobytes()
    ).hexdigest()


class FrozenPublicForward:
    """A frozen projector evaluator. The exact nominal public input is evaluated once per weight."""

    def __init__(self, model, bank, n, device, batch):
        if model.training or batch < 1:
            raise ValueError(
                "frozen forward requires eval mode and a positive microbatch"
            )
        self.model = model.to(device)
        self.device = device
        self.batch = batch
        self.size = n
        self.training_ideas_config = model.training_ideas_config
        self.training = False
        self.nominal = public_projector_inputs(bank)
        self.output = None

    @torch.inference_mode()
    def __call__(self, observation, *, N, fine_identifiable=False):
        if N != self.size or fine_identifiable:
            raise ValueError(
                "unexpected N or measurement geometry for frozen projector"
            )
        public = public_projector_inputs(observation)
        same = set(public) == set(self.nominal) and all(
            field is self.nominal[key] for key, field in public.items()
        )
        if same and self.output is not None:
            return self.output
        length = len(public["counts"])
        pieces = []
        for start in range(0, length, self.batch):
            segment = {
                key: value[start : start + self.batch].to(self.device)
                for key, value in public.items()
            }
            out = self.model(segment, N=N, fine_identifiable=False)
            pieces.append(
                {
                    key: value.cpu() if torch.is_tensor(value) else value
                    for key, value in out.items()
                }
            )
        combined = {}
        for key, value in pieces[0].items():
            if key == "idea_noise_balance":
                # The original module reduces the gate distribution over the entire panel.
                gates = torch.cat([part["idea_noise_gates"] for part in pieces], dim=0)
                combined[key] = (gates.mean(0) - 1.0 / gates.shape[1]).square().mean()
            elif key == "stretchability_values":
                # This is the shared class axis -(N-1)..(N-1), independent of the state index.
                if not all(torch.equal(part[key], value) for part in pieces):
                    raise ValueError("stretchability class axis changed between chunks")
                combined[key] = value
            elif torch.is_tensor(value) and value.ndim:
                combined[key] = torch.cat([part[key] for part in pieces], dim=0)
                if len(combined[key]) != length:
                    raise ValueError(
                        f"output lacks the registered state dimension: {key}"
                    )
            elif all(part[key] == value for part in pieces):
                combined[key] = value
            else:
                raise ValueError(f"non-state output changed between chunks: {key}")
        if same:
            self.output = combined
        return combined
