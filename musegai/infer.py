import pathlib
import shutil
import logging
import tempfile
import numpy as np
import docker

from . import dockerutils, io

LOGGER = logging.getLogger(__name__)

def infer(model, images, outputs=None, *, side=None, tempdir=None, copy_inputs=False):
    """run inference on images

    Args
        model (str): model name
    """
    side = tame_side(side)

    # names
    indir = "in"
    outdir = "out"
    imagename = "im_{index:03d}_{side}_{channel:04d}.nii.gz"
    roiname = "im_{index:03d}_{side}.nii.gz"

    nimage = len(images)
    if not isinstance(images[0], (tuple, list)):
        images = [(im,) for im in images]

    nchannel = len(images[0])
    if {len(im) for im in images} != {nchannel}:
        raise ValueError(f"Number of channels is not constant")

    if outputs and len(set(outputs)) != nimage:
        raise ValueError("Mismatch in the number of inputs and outputs")

    labels = None
    with tempfile.TemporaryDirectory(dir=tempdir) as tmp:
        LOGGER.info("Setup temporary directory")
        # create folder structure
        tmp = pathlib.Path(tmp)
        (tmp / indir).mkdir()
        (tmp / outdir).mkdir()

        # check and copy each volume
        num = 0
        for index in range(nimage):
            LOGGER.info(f"Loading dataset {index + 1}/{nimage}")
            if side == "LR":
                # split into halves (eg. left and right sides)
                LOGGER.info(f"Split images into halves")
                for channel in range(nchannel):
                    image = io.load(images[index][channel])
                    imageA, imageB = io.split(image, axis=0)
                    io.save(tmp / indir / imagename.format(index=index, side="A", channel=channel), imageA)
                    io.save(tmp / indir / imagename.format(index=index, side="B", channel=channel), imageB)
                num += 2
            else:
                # do not split
                for channel in range(nchannel):
                    image = io.load(images[index][channel])
                    io.save(tmp / indir / imagename.format(index=index, side="X", channel=channel), image)
                num += 1

        LOGGER.info(f"Done copying data (num. datasets: {num})")

        # run model
        dockerutils.run_inference(model, tmp)

        # recover outputs
        rois = []
        for index in range(nimage):
            if side == "LR":
                labelmapA = io.load(tmp / outdir / roiname.format(index=index, side="A"))
                labelmapB = io.load(tmp / outdir / roiname.format(index=index, side="B"))
                # increment left side
                max_label = np.max(labelmapA)
                labelmapB.array[labelmapB.array > 0] += max_label
                labelmap = io.heal(labelmapA, labelmapB, axis=0)
            else:
                labelmap = io.load(tmp / outdir / roiname.format(index=index, side="X"))

            rois.append(labelmap)

        if labels is None:
            # get label names
            labels = io.load_labels(tmp / "labels.txt")

    # fix labels sides
    if side == "LR":
        descr = [d + "_R" if l > 0 else d for l, d in zip(labels.indices, labels.descriptions)]
        descr += [d + "_L" for l, d in zip(labels.indices, labels.descriptions) if l > 0]
        indices = labels.indices + [l + max_label for l in labels.indices if l > 0]
        colors = labels.colors + [c for l, c in zip(labels.indices, labels.colors) if l > 0]
        transparency = labels.transparency + [t for l, t in zip(labels.indices, labels.transparency) if l > 0]
        visibility = labels.visibility + [v for l, v in zip(labels.indices, labels.visibility) if l > 0]
        labels = io.Labels(indices, descr, colors, transparency, visibility)
    elif side == "L":
        labels.descriptions = [d + "_L" if l > 0 else d for l, d in zip(labels.indices, labels.descriptions)]
    elif side == "R":
        labels.descriptions = [d + "_R" if l > 0 else d for l, d in zip(labels.indices, labels.descriptions)]

    # return volumes or files?
    if not outputs:
        return rois, labels
    # for filename, labelmap, image in zip(outputs, rois, images, strict=True):
    for index in range(nimage):
        filename = outputs[index]
        filename.parent.mkdir(exist_ok=True, parents=True)
        io.save(filename, rois[index])
        io.save_labels(filename.parent / "labels.txt", labels)
        if copy_inputs:
            for i, chan in enumerate(images[index]):
                image = io.load(chan)
                io.save(filename.parent / chan.name, image)
                if i == 0:
                    suffixes = ''.join(filename.suffixes)
                    io.save(filename.parent / f'vol{suffixes}', image)


def tame_side(side):
    if side is None:
        return "NA"
    elif not isinstance(side, str):
        raise TypeError(f"Invalid side value: {side}")
    side = side.upper().strip().replace(",", "").replace("+", "")
    if side in ["LEFT", "L"]:
        return "L"
    elif side in ["RIGHT", "R"]:
        return "R"
    elif side in ["LR", "RL", "LEFTRIGHT", "RIGHTLEFT"]:
        return "LR"
    elif side.upper() in ["NONE", "NA"]:
        return "NA"
    raise ValueError(f"Unknown side value: {side}")
