"""Build and verify the centrally generated population banks for six long fits."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import shutil
from zipfile import ZIP_STORED, ZipFile

import numpy as np

if __package__:
    from .amp_data import FAMILY_NAMESPACE, generate, load_bank, save_bank
else:
    from amp_data import FAMILY_NAMESPACE, generate, load_bank, save_bank


SIZES = (32, 64, 128, 256)
BUDGETS = ((30, 1), (300, 16))
ROW_START = 1_000_000
ROWS = {"long": {"train": 8192, "validation": 512},
        "qualification": {"train": 64, "validation": 32}}
SEEDS = (61, 62, 63)
ARMS = ("first_order", "second_order")
BANK_SCHEMA = "high_n_long_structured_banks_v1"
RUN_FILES = ("amp_data.py", "amp_fit.py", "amp_control.py", "long_campaign.py", "long_model.py", "long_fit.py")
ROLE_IDS = {"long_train": 21, "long_validation": 22,
            "long_qualification_train": 23, "long_qualification_validation": 24}


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def manifest_identity(record):
    content = json.dumps(record, allow_nan=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def write_json(path, obj):
    with path.open("x", encoding="utf-8") as out:
        json.dump(obj, out, allow_nan=False, indent=2)
        out.write("\n")


def cells(profile):
    for role, rows in ROWS[profile].items():
        stream_role = ("long_" if profile == "long" else "long_qualification_") + role
        for n in SIZES:
            for shots, rounds in BUDGETS:
                name = f"{role}_n{n}_c{shots}_k{rounds}.npz"
                yield role, rows, stream_role, n, shots, rounds, name


def check_metadata(batch, expected):
    role, rows, stream_role, n, shots, rounds, name = expected
    metadata = batch["metadata"]
    fields = dict(n_modes=n, shots=shots, rounds=rounds, rows=rows, start=ROW_START,
                  role=stream_role, role_id=ROLE_IDS[stream_role], family="structured",
                  seed_namespace=FAMILY_NAMESPACE["structured"], n_settings=4*n-3)
    if any(metadata.get(key) != value for key, value in fields.items()):
        raise ValueError(f"Unexpected source or role at {name}")
    if (metadata["trials"] != metadata["trials_per_setting_per_round"] * rounds or
            metadata["population_read_opportunities_per_row"] != n * metadata["trials"] or
            metadata["full_chain_read_opportunities_per_row"] != (4*n-3)*metadata["trials"]):
        raise ValueError(f"Inconsistent physical exposure at {name}")


def source_record(source):
    config = source / "configs/round6_device.json"
    params = json.loads(config.read_text(encoding="utf-8"))
    return {"config_sha256": file_digest(config),
            "calibration_csv_sha256": file_digest(source / params["calibration_csv"]),
            "generator_sha256": file_digest(Path(__file__).with_name("amp_data.py")),
            "nqrec_source": {p.relative_to(source).as_posix(): file_digest(p)
                             for p in sorted((source/"src/nqrec").rglob("*.py"))},
            "python": platform.python_version(), "numpy": np.__version__}


def build_banks(source, bank_dir, qualification=False):
    profile = "qualification" if qualification else "long"
    source, bank_dir = Path(source).resolve(strict=True), Path(bank_dir)
    provenance = source_record(source)
    if shutil.disk_usage(bank_dir.parent).free < (1 << 30):
        raise OSError("Bank build requires 1 GiB of free staging space")
    bank_dir.mkdir(exist_ok=False)
    records = {}
    for item in cells(profile):
        role, rows, stream_role, n, shots, rounds, name = item
        batch = generate(n, rows, stream_role, nqrec_root=source, start=ROW_START,
                         shots=shots, rounds=rounds, family="structured")
        check_metadata(batch, item)
        path = bank_dir / name
        save_bank(path, batch)
        records[name] = {"bytes": path.stat().st_size, "sha256": file_digest(path)}
        print(f"{profile} {name} {records[name]['bytes']} bytes", flush=True)
        del batch
    content = {"schema": BANK_SCHEMA, "profile": profile, "source": provenance, "files": records}
    bank_id = manifest_identity(content)
    write_json(bank_dir/"bank_manifest.json", {**content, "bank_id": bank_id})
    return bank_id


def load_banks(bank_dir, qualification=False):
    """Verify transferred bytes, roles, generator exposure and paired latent identities."""
    profile = "qualification" if qualification else "long"
    bank_dir = Path(bank_dir)
    manifest = json.loads((bank_dir/"bank_manifest.json").read_text(encoding="utf-8"))
    bank_id = manifest.pop("bank_id")
    if (manifest_identity(manifest) != bank_id or manifest["schema"] != BANK_SCHEMA
            or manifest["profile"] != profile):
        raise ValueError("Incompatible bank manifest, profile or identity")
    expected_names = {item[-1] for item in cells(profile)}
    if set(manifest["files"]) != expected_names:
        raise ValueError("Incomplete or unexpected bank cell set")
    banks = {}
    for item in cells(profile):
        role, _, _, n, shots, rounds, name = item
        path = bank_dir/name
        entry = manifest["files"][name]
        if path.stat().st_size != entry["bytes"] or file_digest(path) != entry["sha256"]:
            raise ValueError(f"Input bank changed: {name}")
        batch = load_bank(path)
        check_metadata(batch, item)
        if (shots, rounds) != BUDGETS[0]:
            other = banks[(role, n, *BUDGETS[0])]
            if any(not np.array_equal(batch[key], other[key]) for key in
                   ("cell_index", "population", "log_weight", "efficiency")):
                raise ValueError(f"Latent/calibration mismatch across budgets: {name}")
        banks[(role, n, shots, rounds)] = batch
    return bank_id, banks


def jobs_manifest(bank_id, qualification_id):
    """Emit portable commands for each arm, optimizer seed and observation cell."""
    jobs = []
    for seed in SEEDS:
        for arm in ARMS:
            job_id = f"{arm}_s{seed}"
            common = ["--arm", arm, "--seed", str(seed), "--device", "cuda", "--microbatch", "16"]
            commands = {}
            for mode, directory in (("qualify", "qualification_banks"), ("train", "banks")):
                commands[mode] = ["python", "-m", "scripts.high_n_amplitude.long_fit", "--mode", mode, *common, "--bank-dir",
                                  directory, "--output", f"runs/{mode}/{job_id}"]
            jobs.append({"job_id": job_id, "seed": seed, "arm": arm,
                         "commands": commands})
    return {"schema": "high_n_amplitude_long_queue_v1", "bank_id": bank_id,
            "qualification_bank_id": qualification_id, "family": "structured",
            "updates_per_fit": 180000, "total_updates": 1080000, "validation_interval": 3000,
            "wall_seconds_per_fit": 86400, "seeds": list(SEEDS),
            "jobs": jobs}


def package(root):
    """Use the retained bank files directly, leaving one transport archive and its receipt."""
    root = Path(root)
    ids = {}
    for directory, qualification in (("banks", False), ("qualification_banks", True)):
        bank_id, banks = load_banks(root/directory, qualification)
        ids[directory] = bank_id
        del banks
    jobs = jobs_manifest(ids["banks"], ids["qualification_banks"])
    code_dir = Path(__file__).resolve().parent
    source = {name: code_dir/name for name in RUN_FILES}
    for path in source.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    archive = root/"amplitude_long.zip"
    if archive.exists() or (root/"transfer.json").exists():
        raise FileExistsError("A finalized package already exists; preserve its receipt")
    staged = archive.with_suffix(".building")
    with ZipFile(staged, "x", compression=ZIP_STORED) as out:
        for name, path in source.items():
            out.write(path, name)
        out.writestr("jobs.json", json.dumps(jobs, indent=2, allow_nan=False)+"\n")
        for directory, profile in (("banks","long"), ("qualification_banks","qualification")):
            for name in ["bank_manifest.json", *(item[-1] for item in cells(profile))]:
                out.write(root/directory/name, f"{directory}/{name}")
    with ZipFile(staged) as zipped:
        if zipped.testzip() or any(zipped.read(n) != p.read_bytes() for n,p in source.items()):
            raise OSError("Archive verification failed")
    staged.rename(archive)
    receipt = {"archive": archive.name, "bytes": archive.stat().st_size,
               "sha256": file_digest(archive), "source_sha256": {n:file_digest(p) for n,p in source.items()},
               "bank_id": ids["banks"], "qualification_bank_id": ids["qualification_banks"]}
    write_json(root/"jobs.json", jobs)
    write_json(root/"transfer.json", receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="operation", required=True)
    build = sub.add_parser("build")
    build.add_argument("--nqrec-root", type=Path, required=True)
    build.add_argument("--root", type=Path, required=True)
    build.add_argument("--profile", choices=tuple(ROWS), required=True)
    pack = sub.add_parser("package")
    pack.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    if args.operation == "build":
        args.root.mkdir(parents=True, exist_ok=True)
        qual = args.profile == "qualification"
        directory = args.root/("qualification_banks" if qual else "banks")
        print(build_banks(args.nqrec_root, directory, qual), flush=True)
    else:
        print(json.dumps(package(args.root), indent=2))


if __name__ == "__main__":
    main()
