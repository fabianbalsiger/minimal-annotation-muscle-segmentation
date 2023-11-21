from __future__ import annotations

import pathlib
import tempfile
import uuid

import numpy as np

import docker
import SimpleITK as sitk




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
            vol.save(indir / name, '.nii.gz')

        # run model
        _run_model(model, indir, outdir)

        # recover outputs
        
        labels = Labels.load(outdir / 'labels.txt')
        for name, parts in volume_parts.items():
            if "left" in parts:
                left = Volume.load(outdir / f"{name}_left", 'nii.gz')
            if "right" in parts:
                right =  Volume.load(outdir / f"{name}_right", 'nii.gz')
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



class Labels:
    """ label container """
    def __init__(self, data):
        self.data = data

    def load(cls, file):
        with open(file) as fp:
            data = fp.read()
        return cls(data)

    def save(self, file):
        with open(file, 'w') as fp:
            fp.write(self.data)


class Volume:
    """ Image container """
    def __init__(self, obj, **meta):
        try:
            self.array = obj if isinstance(obj, np.ndarray) else np.asarray(getattr(obj, 'array'))
            self.spacing = meta.get('spacing', getattr(obj, 'spacing'))
            self.origin = meta.get('origin', getattr(obj, 'origin'))
            self.transform = meta.get('transform', getattr(obj, 'transform'))
        except AttributeError as exc:
            raise TypeError(f'Missing argument or attribute: {exc.name}')
        
    def __array__(self):
        return self.array
    
    @property
    def shape(self):
        return self.array.shape
    
    @property
    def ndim(self):
        return self.array.ndim
    
    @property
    def metadata(self):
        return {'origin': self.origin, 'spacing': self.spacing, 'transform': self.transform}

    def save(self, file, ext=None):
        im = sitk.GetImageFromArray(self.arr)
        im.SetSpacing(self.spacing)
        im.SetOrigin(self.origin)
        im.SetDirection(self.transform)
        if ext:
            file = pathlib.Path(file).with_suffix(ext)
        sitk.WriteImage(im, file)

    def load(cls, file):
        im = sitk.ReadImage(file)
        array = sitk.GetArrayFromImage(im)
        spacing = im.GetSpacing()
        origin = im.GetOrigin()
        transform = im.GetDirection()
        return cls(array, origin=origin, spacing=spacing, transform=transform)



def _split_volume(volume, side, axis=0):
    """slit volumes according to side"""
    side = side.lower()
    size = volume.shape[axis]
    if "left" in side and "right" in side:
        # split into left and right parts
        indices = np.arange(size // 2)
        left = Volume(np.take(volume, axis=axis, indices=indices), **volume.metadata)
        right = Volume(np.take(volume, axis=axis, indices=indices + size // 2), **volume.metadata)
    elif "left" in side:
        # do nothing
        left = volume
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
        return Volume(np.concatenate([left, right], axis=axis), **left.metadata)
    elif left is not None:
        return left
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
    volumes = {name: Volume(vol) for name, vol in volumes.items()}
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
