"""Atomic writers used by model fitting and evaluation."""
import os
import json
import torch


def atomic_save(data, path):
    scratch = path.with_suffix(path.suffix + ".tmp")
    torch.save(data, scratch)
    os.replace(scratch, path)


def write_json(path, value):
    scratch = path.with_suffix(path.suffix + ".tmp")
    scratch.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(scratch, path)
