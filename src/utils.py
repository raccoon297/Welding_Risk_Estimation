import hashlib
import json
from pathlib import Path

import numpy as np


def categorical_labels(values,allowed=(0,1)):
    """Validate before converting: fractional labels must never be truncated."""
    labels = np.asarray(values,dtype=float)
    if labels.ndim!=1 or not len(labels) or not np.isfinite(labels).all() or not np.isin(labels,allowed).all():
        raise ValueError(f"Nonempty one-dimensional labels in {allowed} required")
    return labels.astype(int)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path,value):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False,default=str)+"\n",encoding="utf-8")


def write_csv(frame,path):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    frame.to_csv(path,index=False,encoding="utf-8-sig",float_format="%.17g")
