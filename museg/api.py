from __future__ import annotations

import pathlib
import tempfile
import uuid

import numpy as np

import docker

from . import itklabels, metaimage


def list_models():
    """List available models."""
    return ["thigh-model3"]


def segment_volumes(volumes, model, *, side="left,right", tempdir=None):
    """segment volumes with specified model"""

    input_type, volumes = _setup_volumes(volumes)

    # check model
    models = list_models()
    if model not in models:
        raise ValueError(f"Unknown model: {model}")

    _pull_if_not_exists(model)

    # prepare volumes
    to_segment = {}
    volume_parts = {}
    for name, vols in volumes.items():
        # checks
        _check_volumes(vols)

        for i, vol in enumerate(vols):
            # split into left and right
            left, right = _split_volume(vol, side)
            volume_parts[name] = []
            if left is not None:
                volume_parts[name].append("left")
                to_segment[f"{name}_left_{i:04d}"] = left
            if right is not None:
                volume_parts[name].append("right")
                to_segment[f"{name}_right_{i:04d}"] = right

    # save into temporary directory, and segment
    segmented = {}
    with tempfile.TemporaryDirectory(dir=tempdir) as tmp:
        tmp = pathlib.Path(tmp)

        # make input and output directories
        indir = tmp / "in"
        indir.mkdir()
        outdir = tmp / "out"
        outdir.mkdir()

        # save input data
        for name, vol in to_segment.items():
            _save_volume(indir, name, vol)

        # run model
        _run_model(model, indir, outdir)

        # recover outputs
        labels = _load_labels(outdir)
        for name, parts in volume_parts.items():
            if "left" in parts:
                left = _load_volume(outdir, f"{name}_left")
            if "right" in parts:
                right = _load_volume(outdir, f"{name}_right")
            vol = _heal_volume(left, right)
            segmented[name] = vol

    # return
    if input_type == "dict":
        return segmented, labels
    elif input_type == "single":
        return next(segmented.values()), labels
    elif input_type == "list":
        return [segmented[name] for name in volumes], labels


#
# private functions


def _load_labels(tmp):
    filename = pathlib.Path(tmp) / "labels.txt"
    return itklabels.read_labels(filename)


def _save_volume(tmp, name, vol):
    filename = (pathlib.Path(tmp) / name).with_suffix(".mha")
    metaimage.write(filename, vol)


def _load_volume(tmp, name):
    filename = (pathlib.Path(tmp) / name).with_suffix(".nii.gz")
    vol, _ = metaimage.read(filename)
    return vol


def _split_volume(volume, side, axis=0):
    """slit volumes according to side"""
    side = side.lower()
    size = volume.shape[axis]
    if "left" in side and "right" in side:
        # split and flip left
        indices = np.arange(size // 2)
        left = np.take(volume, axis=axis, indices=indices[::-1])
        right = np.take(volume, axis=axis, indices=indices + size // 2)
    elif "left" in side:
        # flip left
        left = np.flip(volume, axis=axis)
        right = None
    elif "right" in side:
        # do nothing
        left = None
        right = volume
    else:
        raise ValueError(f"Invalid side: {side}")
    return left, right


def _heal_volume(left, right, *, axis=0):
    if left is not None and right is not None:
        return np.concatenate([np.flip(left, axis=axis), right], axis=axis)
    elif left is not None:
        return np.flip(left, axis=axis)
    elif right is not None:
        return right
    else:
        raise ValueError("Something went wrong")


def _check_volumes(volumes, *, nvolumes=2):
    """safety checks"""
    if not len(volumes) == nvolumes:
        raise ValueError("There should be 2 volumes")
    elif len({vol.shape for vol in volumes}) != 1:
        raise ValueError("All volumes must have the same shape")


def _setup_volumes(volumes):
    if isinstance(volumes, dict):
        input_type = "dict"
        volumes = volumes
    elif isinstance(volumes, list) and all(isinstance(item, np.ndarray) for item in volumes):
        # a single case
        input_type = "single"
        volumes = {str(uuid.uuid1()): volumes}
    elif isinstance(volumes, list) and all(isinstance(item, list) for item in volumes):
        # multiple cases
        input_type = "list"
        volumes = {str(uuid.uuid1()): vol for vol in volumes}
    else:
        raise ValueError("`volumes` should be a (dict of) list of ndarrays")
    return input_type, volumes


def _run_model(model, indir, outdir):
    """Run inference."""
    client = docker.from_env()
    client.containers.run(
        f"fabianbalsiger/museg:{model}",
        remove=True,
        device_requests=[docker.types.DeviceRequest(device_ids=["all"], capabilities=[["gpu"]])],
        volumes={indir.parent: {"bind": "/data", "mode": "rw"}},
    )


def _pull_if_not_exists(model: str):
    """Pull a Docker image if not exists."""
    client = docker.from_env()
    if not len(client.images.list(name=f"fabianbalsiger/museg:{model}")):
        client.images.pull(f"fabianbalsiger/museg:{model}")
