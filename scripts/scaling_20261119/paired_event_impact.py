"""Paired observation-law sensitivity and zero-update replay of a frozen STAR core."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import torch
from nqrec.physics.single_event_mc import (
    calibrate_source_weights, diagonal_fock2_event_probability, four_cell_probabilities,
    k1_event_probability, sample_counts,
)


ROOT = Path(__file__).resolve().parents[2]
SEED = 2026112017
N_STATES = 96
READS_PER_MODE = 8192
SOURCE_P2 = 0.0004
K2_SIGNAL = 0.375
K2_ACCEPT = 0.75
ETA_SIGNAL = 0.2
ETA_IDLER = 0.03
IDLER_BACKGROUND = 0.001


def load_source(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def tensor(value):
    return torch.as_tensor(value, dtype=torch.float64)


def paired_states(n, *, seed=SEED, rows=N_STATES):
    rng = np.random.default_rng([seed, n])
    amplitudes = rng.uniform(0.2, 1, (rows, n))
    phase = rng.uniform(-np.pi, np.pi, (rows, n))
    phase[:, 0] = 0
    mode = amplitudes * np.exp(1j * phase) / np.linalg.norm(amplitudes, axis=1, keepdims=True)
    visibility = rng.choice([1., .7], rows)
    pure = mode[:, :, None] * mode[:, None, :].conj()
    density = pure * visibility[:, None, None]
    index = np.arange(n)
    density[:, index, index] = abs(mode) ** 2
    analyzers = np.zeros((2 * (n - 1), n), dtype=np.complex128)
    for leaf in range(1, n):
        pair = 2 * (leaf - 1)
        analyzers[pair:pair + 2, 0] = 2 ** -.5
        analyzers[pair:pair + 2, leaf] = np.array([1, -1j]) * 2 ** -.5
    return mode, visibility, torch.from_numpy(density), torch.from_numpy(analyzers)


def new_cells(density, analyzers, rate, background, family):
    n = density.shape[-1]
    p2 = 0 if family == 'k1' else SOURCE_P2
    source = calibrate_source_weights(tensor(rate), tensor(ETA_SIGNAL), tensor(background),
                                      double_probability=tensor(p2), double_signal_event_probability=tensor(K2_SIGNAL))
    transfer = analyzers * np.sqrt(ETA_IDLER)
    r1 = k1_event_probability(density[:, None], transfer, tensor(IDLER_BACKGROUND))
    r0 = torch.full_like(r1, IDLER_BACKGROUND)
    r2 = torch.zeros_like(r1)
    if p2:
        local = torch.full((n,), float(family == 'local') / n, dtype=torch.float64)
        cross = torch.triu(torch.ones((n, n), dtype=torch.float64), diagonal=1)
        cross *= float(family == 'cross') * 2 / (n * (n - 1))
        r2 = diagonal_fock2_event_probability(
            transfer, local, cross, two_photon_single_event_acceptance=tensor(K2_ACCEPT),
            background=tensor(IDLER_BACKGROUND)).expand_as(r1)
    response = torch.stack([r0, r1, r2], dim=-1)
    return four_cell_probabilities(source['joint'], response[..., None].expand(*response.shape, 2))


def estimate(values):
    values = np.asarray(values)
    rng = np.random.default_rng(SEED + 1)
    draws = rng.integers(len(values), size=(4000, len(values)))
    return dict(mean=float(values.mean()), interval95=np.quantile(values[draws].mean(1), [.025, .975]).tolist())


def frozen_risk(model, counts, exposure, truth):
    legacy_order = counts[..., [0, 2, 1, 3]]
    if not (legacy_order.sum(-1) == exposure).all():
        raise ValueError('Invalid STAR per-setting exposure')
    x = np.column_stack((np.log1p(legacy_order.reshape(len(counts), -1)),
                         np.full(len(counts), np.log(exposure))))
    with torch.no_grad():
        rho = model(tensor(x))
    return (rho - truth).abs().square().sum((1, 2)).numpy()


def run(checkpoint, output, bank_dir):
    from ml_recon.physics_hybrid_v10.training_ideas_data import readout_probability_from_rho
    started = time.monotonic()
    archive = load_source('archive_star', ROOT / 'scripts/scaling_20261119/star_data.py')
    model_source = Path(__file__).with_name('unified_model.py')
    net_class = load_source('archived_net', model_source).Unified_star_density_net
    checkpoint = Path(checkpoint)
    with checkpoint.open("rb") as stream:
        checkpoint_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
        stream.seek(0)
        state = torch.load(stream, map_location='cpu', weights_only=True)
    model = net_class(hidden=state['leaf_encoder.0.weight'].shape[0],
                      rank_columns=(state['leaf_decoder.2.bias'].numel()-1)//2)
    model.load_state_dict(state, strict=True)
    model.eval()
    out = []
    for n in (3, 4, 8):
        mode, vis, rho, analyzers = paired_states(n)
        exposure = READS_PER_MODE * n
        stream_seed = [SEED, 3, n]
        archived_risk = None
        if n == 8:
            legacy = archive.observation_probabilities(mode, vis)[..., [0, 2, 1, 3]]
            old_counts = sample_counts(tensor(legacy), exposure, np.random.default_rng(stream_seed))
            archived_risk = frozen_risk(model, old_counts, exposure, rho)
        for rate in (.01, .025, .04):
            old = readout_probability_from_rho(rho, analyzers.expand(N_STATES, -1, -1),
                herald_probability=tensor(rate), read_efficiency=tensor(ETA_IDLER),
                background_probability=tensor(IDLER_BACKGROUND))
            for background in (.0005, .002):
                for family in ('k1', 'local', 'cross'):
                    new = new_cells(rho, analyzers, rate, background, family)
                    raw = sample_counts(new, exposure, np.random.default_rng(stream_seed))
                    if not (raw.sum(-1) == exposure).all() or float((new[..., 1]+new[..., 3]-rate).abs().max()) > 1e-13:
                        raise ArithmeticError('Failed rate or count conservation')
                    log_ratio = (new.log() - old.log()).numpy()
                    row = dict(n=n, total_signal=rate, signal_background=background, family=family,
                        expected_nll_excess_nats_per_read=estimate((new.numpy()*log_ratio).sum(-1).mean(-1)),
                        realized_nll_excess_nats_per_read=estimate((raw*log_ratio/exposure).sum(-1).mean(-1)),
                        missed_signal_idler_probability=estimate(new[..., 2].mean(-1).numpy()),
                        legacy_missed_signal_idler_probability=float(old[..., 2].mean()),
                        simulated_signal_rate=(raw[...,1]+raw[...,3]).mean()/exposure)
                    if n == 8:
                        risk = frozen_risk(model, raw, exposure, rho)
                        row['corrected_frozen_risk'] = estimate(risk)
                        row['archived_star_frozen_risk'] = estimate(archived_risk)
                        row['paired_change_from_archived_star_risk'] = estimate(risk-archived_risk)
                    out.append(row)
                    if rate == .025 and background == .002:
                        dest = bank_dir / f'n{n}_{family}.npz'
                        np.savez_compressed(dest, counts=raw, analyzers=analyzers.numpy(),
                            read_opportunities=np.full(raw.shape[:2], exposure), command=np.full((len(raw),n),n**-.5),
                            private_density=rho.numpy(), public_calibration=[rate,ETA_SIGNAL,background,ETA_IDLER,IDLER_BACKGROUND],
                            metadata_json=json.dumps(dict(law='effective_single_event_v1',schema='paired_star_impact_v1',
                                cell_order=['00','10','01','11'],s_branch_state='shared',k2=family,
                                source_p2=0 if family=='k1' else SOURCE_P2,k2_signal=K2_SIGNAL,k2_accept=K2_ACCEPT,
                                local_fraction={'k1':None,'local':1,'cross':0}[family],private_fields=['private_density'])))
    result=dict(schema='paired_star_impact_v1',seed=SEED,independent_states_per_cell=N_STATES,
        checkpoint=checkpoint.name, checkpoint_sha256=checkpoint_sha256,
        n8_training_updates_performed_here=0,observations='common states, paired RNG seed per N, positive-exposure STAR with read phase sign -1',
        probability_reference='PR44 conditional-projector decoder matched in total signal and idler response',
        frozen_reference='archived STAR law and supplied frozen core; migration changes operating point and law together',
        state_prior='independent uniform positive amplitude [0.2,1], uniform phases with mode 0 reference, V=0.7 or 1',
        public_calibration=dict(eta_signal=ETA_SIGNAL,eta_idler=ETA_IDLER,idler_background=IDLER_BACKGROUND),
        k2_hypothesis=dict(source_probability=SOURCE_P2,signal_acceptance=K2_SIGNAL,two_photon_acceptance=K2_ACCEPT),
        per_operation_reads='8192*N',observed_paid_reads_per_cell={str(n):N_STATES*8192*n*2*(n-1) for n in (3,4,8)},
        bootstrap='4000 resamples of states; pointwise descriptive sensitivity grid; no best-cell selection',
        scientific_limit='Synthetic change of operating law; the archived stationary performance result remains tied to its original conditions.',
        runtime_seconds=time.monotonic()-started,rows=out)
    output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--bank-dir',type=Path,required=True)
    args=parser.parse_args()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.bank_dir.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1)
    run(args.checkpoint,args.output,args.bank_dir)
