"""Offline task-specific scores for a frozen projector model on a labeled synthetic panel."""

from __future__ import annotations

import torch

from .training_ideas import NOISE_CLASS_NAMES


def _noise_report(logits, target, valid, train_counts):
    """Rows of the confusion matrix are true classes. Unsupported classes retain their column."""
    k = len(NOISE_CLASS_NAMES)
    scores = logits[valid].double()
    labels = target[valid].long()
    if scores.shape != (labels.numel(), k) or not torch.isfinite(scores).all():
        raise ValueError("invalid mechanism logits on the labeled panel")
    probability = scores.softmax(dim=1)
    predicted = probability.argmax(dim=1)
    confusion = torch.bincount(k * labels + predicted, minlength=k * k).reshape(k, k)
    support, calls, hits = confusion.sum(1), confusion.sum(0), confusion.diag()
    by_class = []
    for i, name in enumerate(NOISE_CLASS_NAMES):
        total, chosen, correct = int(support[i]), int(calls[i]), int(hits[i])
        by_class.append(dict(
            name=name, support=total, predicted=chosen,
            recall=correct / total if total else None,
            precision=correct / chosen if chosen else None,
            f1=2 * correct / (total + chosen) if total else None,
        ))
    brier = (probability - torch.nn.functional.one_hot(labels, k)).square().sum(1)
    result = dict(
        labeled_states=len(labels), masked_states=int((~valid).sum()),
        true_by_predicted=confusion.tolist(), by_class=by_class,
        macro_f1_supported=float(torch.tensor([c["f1"] for c in by_class if c["support"]]).mean())
        if labels.numel() else None,
        multiclass_brier=float(brier.mean()) if labels.numel() else None,
    )
    if train_counts is not None:
        counts = torch.as_tensor(train_counts, device=logits.device)
        if counts.shape != (k,) or (counts < 0).any() or not counts.sum() > 0:
            raise ValueError("invalid labeled TRAIN class counts")
        freq = counts.double() / counts.sum()
        errors = (freq - torch.nn.functional.one_hot(labels, k)).square().sum(1)
        majority = int(counts.argmax())
        result["train_prior"] = dict(
            class_counts=counts.tolist(), probability=freq.tolist(), modal_class=majority,
            accuracy=float((labels == majority).double().mean()) if labels.numel() else None,
            multiclass_brier=float(errors.mean()) if labels.numel() else None,
            paired_network_minus_prior_brier=float((brier - errors).mean()) if labels.numel() else None,
        )
    return result


def _phase_edges(output, batch):
    """Score phase modulo 2π on the informative visible STAR edges."""
    edges = batch["phase_edges"]
    left, right = edges.unbind(-1)
    occupied = torch.complex(batch["analyzer_re"], batch["analyzer_im"]).abs() > 1e-10
    settings = occupied.shape[1]
    left_present = occupied.gather(2, left[:, None].expand(-1, settings, -1))
    right_present = occupied.gather(2, right[:, None].expand(-1, settings, -1))
    exposed = batch["setting_mask"].bool() & (batch["shots"] > 0)
    visible = ((exposed & (occupied.sum(-1) == 2))[:, :, None] & left_present & right_present).any(1)
    valid = batch["target_phase_edge_mask"].bool()
    if (valid & ~visible).any():
        raise ValueError("informative phase edge has no exposed visible pair setting")
    labels = batch["target_phase_edge_cos_sin"]
    prefix = "conditional_R1" if "conditional_R1_re" in output else "R1"
    row = torch.arange(len(valid), device=left.device)[:, None]
    re, im = (output[prefix + suffix][row, left, right].double() for suffix in ("_re", "_im"))
    defined = valid & ((re != 0) | (im != 0))
    cos, sin = labels.double().unbind(-1)
    error = torch.atan2(im * cos - re * sin, re * cos + im * sin).abs()
    return dict(
        states=len(valid), possible=int(valid.numel()), visible=int(visible.sum()),
        informative=int(valid.sum()), defined_predictions=int(defined.sum()),
        zero_predicted_coherence=int((valid & ~defined).sum()),
        informative_star_connected_fraction=float(valid.all(-1).double().mean()),
        mean_absolute_wrapped_radians=float(error[defined].mean()) if defined.any() else None,
    )


def _reachability(output, batch):
    mask = batch["target_rsm_mask"].bool()
    target = batch["target_rsm_fraction"][mask].double()
    error = output["idea_rsm_fraction"][mask].double() - target
    if not torch.isfinite(target).all() or ((target < 0) | (target > 1)).any():
        raise ValueError("reachable-fraction labels must lie in [0,1]")
    bands = []
    for label, active in (("low", target < 1 / 3), ("mid", (target >= 1 / 3) & (target < 2 / 3)),
                          ("high", target >= 2 / 3)):
        bands.append(dict(band=label, states=int(active.sum()),
                          mae=float(error[active].abs().mean()) if active.any() else None,
                          bias=float(error[active].mean()) if active.any() else None))
    return dict(labeled_states=int(mask.sum()), masked_states=int((~mask).sum()),
                mae=float(error.abs().mean()) if error.numel() else None,
                cutoffs=[1 / 3, 2 / 3], bands=bands)


def _gram_summary(gram, trace):
    """Participation dimension and relative response along common log-amplitude/phase."""
    selected = gram[trace > 0]
    if not len(selected):
        return dict(states_with_positive_trace=0, participation_dimension=None,
                    common_log_amplitude_response=None, common_phase_response=None)
    normed = selected / trace[trace > 0, None, None]
    dim = normed.shape[-1] // 2
    response = [normed[..., offset::2].sum(-1).norm(dim=-1) / dim ** 0.5 for offset in (0, 1)]
    return dict(states_with_positive_trace=len(selected),
                participation_dimension=float((1 / normed.square().sum((-2, -1))).mean()),
                common_log_amplitude_response=float(response[0].mean()),
                common_phase_response=float(response[1].mean()))


@torch.no_grad()
def representation_report(output: dict, batch: dict, *, train_class_counts=None) -> dict:
    """Separate class discrimination, pair structure and continuous control geometry."""
    target = batch["target_operational_noise"]
    valid = batch["target_operational_noise_mask"].bool()
    noise = _noise_report(output["operational_noise_logits"], target, valid, train_class_counts)
    family = torch.where(valid, target, -1)
    pair = output["pair_entanglement_probs"]
    pair = (pair + pair.transpose(-1, -2)) / 2
    pairs = batch["target_pair_mask"].bool().triu(1)
    truth = batch["target_pair"].bool()
    groups = []
    for f in family.unique(sorted=True).tolist():
        selected = family == f
        mask = selected[:, None, None] & pairs
        obs, label = pair[mask], truth[mask]
        choice = obs >= 0.5
        groups.append(dict(
            family="nominal" if f == -1 else NOISE_CLASS_NAMES[f], states=int(selected.sum()),
            unordered_pairs=int(mask.sum()), positive_pairs=int(label.sum()),
            pair_brier=float((obs - label.to(obs.dtype)).square().mean()) if obs.numel() else None,
            true_positive=int((choice & label).sum()), false_positive=int((choice & ~label).sum()),
            false_negative=int((~choice & label).sum()), true_negative=int((~choice & ~label).sum()),
        ))
    report = dict(noise=noise, esm_by_simulated_family=groups, pair_threshold=0.5)
    if "target_hyperedge_mask" in batch:
        mask = batch["target_hyperedge_mask"].bool()
        label = batch["target_hyperedge"][mask]
        prob = output["hyperedge_entanglement_probs"][mask]
        report["esm_hyperedges"] = dict(
            labeled=int(mask.sum()), positives=int(label.sum()),
            brier=float((prob.double() - label.double()).square().mean()) if label.numel() else None,
        )
    if "target_phase_edge_mask" in batch:
        report["phase_edges"] = _phase_edges(output, batch)
    if "idea_rsm_fraction" in output:
        report["rsm"] = _reachability(output, batch)
    if "idea_csm_gram" in output:
        use = batch["target_csm_mask"].bool()
        target_gram = batch["target_csm_gram"][use].double()
        pred_gram = output["idea_csm_gram"][use].double()
        target_trace = target_gram.diagonal(dim1=-2, dim2=-1).sum(-1)
        pred_trace = pred_gram.diagonal(dim1=-2, dim2=-1).sum(-1)
        supported = (target_trace > 0) & (pred_trace > 0)
        target_shape = target_gram[supported] / target_trace[supported, None, None]
        pred_shape = pred_gram[supported] / pred_trace[supported, None, None]
        report["csm_geometry"] = dict(
            labeled_states=int(use.sum()), shape_scored=int(supported.sum()),
            nonpositive_teacher_trace=int((target_trace <= 0).sum()),
            nonpositive_predicted_trace=int((pred_trace <= 0).sum()),
            trace_normalized_frobenius_squared=float(
                (pred_shape - target_shape).square().sum((-2, -1)).mean()
            ) if supported.any() else None,
            mean_abs_log_trace_ratio=float(
                (pred_trace[supported].log() - target_trace[supported].log()).abs().mean()
            ) if supported.any() else None,
            teacher=_gram_summary(target_gram, target_trace),
            predicted=_gram_summary(pred_gram, pred_trace),
        )
        if "target_csm_fixed_trace" in batch:
            fixed = batch["target_csm_fixed_trace"][use].double()
            active = fixed > 0
            report["csm_geometry"]["nuisance_profile"] = dict(
                states=len(fixed), states_with_fixed_information=int(active.sum()),
                mean_fixed_trace=float(fixed.mean()) if len(fixed) else None,
                mean_profiled_trace=float(target_trace.mean()) if len(fixed) else None,
                mean_retained_fraction=float((target_trace[active] / fixed[active]).mean()) if active.any() else None,
            )
    return report
