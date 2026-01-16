import pathlib
import shutil
import logging
import tempfile
import numpy as np
import docker

from . import dockerutils, io, metrics

LOGGER = logging.getLogger(__name__)

def test(model, images, refs, preds=None, *, inference=True, side=None, tempdir=None, copy_inputs=False, statsfile=None, figfile=None, figtitle=None):

    side = tame_side(side)

    nimage = len(images)
    if not isinstance(images[0], (tuple, list)):
        images = [(im,) for im in images]

    nchannel = len(images[0])
    if {len(im) for im in images} != {nchannel}:
        raise ValueError("Number of channels is not constant")
    
    if len(refs) != nimage:
        raise ValueError("Mismatch in the number of images and references")

    if preds and len(set(preds)) != nimage:
        raise ValueError("Mismatch in the number of images and predictions")

    if inference:
        # make inference

        # names
        indir = "in"
        outdir = "out"
        imagename = "im_{index:03d}_{side}_{channel:04d}.nii.gz"
        roiname = "im_{index:03d}_{side}.nii.gz"

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
                    idx = None
                    for channel in range(nchannel):
                        image = io.load(images[index][channel])
                        imageA, imageB = io.split(image, axis=0, idx=idx)
                        io.save(tmp / indir / imagename.format(index=index, side="A", channel=channel), imageA)
                        io.save(tmp / indir / imagename.format(index=index, side="B", channel=channel), imageB)
                        # keep same split index for all channels
                        idx = imageA.shape[0]
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

            # recover predictions
            predictions = []
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

                predictions.append(labelmap)

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

    else: 
        # check predictions
        for index in range(nimage):
            filename = preds[index]
            if not io.is_image(filename):
                raise ValueError(f'Missing prediction: #{index}')
            
        # load labels
        path = pathlib.Path(preds[0]).parent
        labels = io.load_labels(path / 'labels.txt')
        
    # get labels for each image
    labels_preds = [labels.to_dict() for _ in range(len(images))]
            
    # check references
    labels_refs = []
    for index in range(nimage):
        filename = refs[index]
        if not io.is_image(filename):
            raise ValueError(f'Missing reference: #{index}')
        labels_ref = io.load_labels(pathlib.Path(filename).parent / 'labels.txt')
        labels_refs.append(labels_ref.to_dict())
        
    # load images
    references = [io.load(file) for file in refs]
    predictions = [io.load(file) for file in preds]
        
    # process segmentations
    kw = {
        'labels': labels_refs,
        'labels_preds': labels_preds,
        'options': {'b_iou.d': 5, 'nsd.tau': 3},
    }
    process = metrics.Process(references, predictions, **kw)
    stats = ['dsc', 'b_iou', 'nsd', 'hd95']
    data = process(stats, subset=None)
    
    # store stats
    if statsfile:
        io.save_csv(statsfile, data)

    # plot
    if figfile:
        detailed = ['dsc', 'hd95', 'nsd']
        title = figtitle or model
        fig = metrics.plot_metrics(data, detailed=detailed, title=title)
        fig.savefig(figfile, dpi=300)

    if inference:
        # store predictions
        for index in range(nimage):
            filename = preds[index]
            filename.parent.mkdir(exist_ok=True, parents=True)
            io.save(filename, predictions[index])
            io.save_labels(filename.parent / "labels.txt", labels)
            if copy_inputs:
                for chan in images[index]:
                    image = io.load(chan)
                    io.save(filename.parent / chan.name, image)


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
