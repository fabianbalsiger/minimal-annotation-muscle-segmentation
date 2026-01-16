"""Command line interface for muscle segmentation."""

from __future__ import annotations
import logging
import pathlib
import re
import sys

import click

from musegai import api


@click.group()
def cli(): ...

@cli.command()
def list():
    """ List available models """
    models = api.list_models()
    # no argument: list available models
    click.echo("Available segmentation models:")
    for available_model in models:
        click.echo(f"\t{available_model}")
    sys.exit(0)


@cli.command(context_settings={"show_default": True})
@click.argument("model")
@click.argument("images", nargs=-1)
@click.option('-o', "--overwrite", is_flag=True, help="Overwrite already existing files.")
@click.option('-y', "--yes", is_flag=True, help="Bypass confirmation.")
@click.option("-r", "--root", type=click.Path(exists=True), help="Input root directory.")
@click.option("-d", "--dest", type=click.Path(), help="Output root directory.")
@click.option("--filename", default="roi", help="Segmentation filename.")
@click.option("--dirname", help="Segmentation parent folder name.")
@click.option("-f", "--format", default=".nii.gz", type=click.Choice([".nii.gz", ".mha", ".mhd", ".hdr"]))
@click.option("--side", default="LR", type=click.Choice(["L", "R", "LR", "NA"]), help="Limb's side(s) in image")
@click.option("--tempdir", type=click.Path(exists=True), help="Location for nnUNet temporary files.")
@click.option("-v", "--verbose", is_flag=True, help="Show more information")
def segment(images, dest, dirname, filename, format, model, side, tempdir, verbose, overwrite, yes, root):
    """Apply segmentation model on dataset

    \b
    MODEL: segmentation model to use (cf. `museg-ai list`)
    IMAGES: path expression to specify the images to segment
        Multiple channels are assumed if:
        - image files are numbered
        - multiple path expressions are passed
    
    """
    if verbose:
        logging.basicConfig(level=logging.INFO)

    # copy inputs to outputs
    copy_inputs = (dirname is not None) or (dest is not None)

    root = pathlib.Path(root) if root else pathlib.Path(".")

    if not images:
        # no argument: list available models
        models = api.list_models()
        click.echo("Available segmentation models:")
        for available_model in models:
            click.echo(f"\t{available_model}")
        sys.exit(0)

    # get nchannel from model metadata
    try:
        model_info = api.get_model_info(model)
    except ValueError as exc:
        click.echo(f'Error: {exc}')
        sys.exit(1)
    nchannel = int(model_info["nchannel"])

    if 1 < len(images) != nchannel:
        click.echo(f"Expecting {nchannel} channels, got {len(images)} expression(s).")
        sys.exit(1)

    # find images
    image_files = []
    for expr in images:
        if pathlib.Path(expr).is_absolute():
            # assume a directory
            click.echo(f"\tAbsolute paths are not accepted: {expr}")
        files = sorted(root.rglob(expr))
        image_files.append(files)

    num_files = {expr: len(files) for expr, files in zip(images, image_files)}
    if any(num == 0 for num in num_files):
        click.echo(f"No image files found, check expression(s): {[ex for ex in num_files if num_files[ex] == 0]}")
        sys.exit(1)

    images = {}
    if len(image_files) == 1:
        # numbered channels
        regex = re.compile(r"(.+?)(\d*)\.([\.\w]+)$")
        for file in image_files[0]:
            match = regex.match(str(file))
            if not match or not api.is_image(file):
                continue
            common, index, ext = match.groups()
            images.setdefault(common, []).append(file)
    else:
        # one expression per channel
        regex = re.compile(r"(.+?)\.([\.\w]+)$")
        for files in image_files:
            for file in files:
                match = regex.match(str(file))
                if not match or not api.is_image(file):
                    continue
                common = file.parent
                images.setdefault(common, []).append(file)
    images = {name: sorted(ims)[:nchannel] for name, ims in images.items()}

    # check number of channels
    names = sorted(images)
    bad_numchan = {name: len(images[name]) for name in names if len(images[name]) < nchannel}
    if bad_numchan:
        click.echo(f"Did not find {nchannel} channel(s) in:")
        for name in bad_numchan:
            click.echo(f'\t{name}: {", ".join(f"[{i + 1}] {file}" for i, file in enumerate(images[name]))}')
        sys.exit(1)

    # destination
    dest = pathlib.Path(root if dest is None else dest)
    dest.mkdir(exist_ok=True, parents=True)
    destloc = {name: images[name][0].relative_to(root).parent for name in names}
    if dirname:
        destloc = {name: loc.parent / dirname for name, loc in destloc.items()}
    labels = {name: (dest / destloc[name] / filename).with_suffix(format) for name in names}

    click.echo(f"Found {len(images)} image to segment (num. channels: {nchannel}):")
    for i, name in enumerate(names):
        click.echo(f"({i+1})")
        for j in range(nchannel):
            click.echo(f"\tchan. {j + 1:02d}: {images[name][j]}")
        click.echo(f"\tlabels:   {labels[name]}")
    if not yes:
        click.confirm("Are all images correctly selected?", abort=True)

    # dealing whith already existent files in output directory
    if not overwrite:
        for name in names:
            if labels[name].is_file():
                click.echo(f"Output file already exists: {labels[name]}, skipping")
                images.pop(name)
                labels.pop(name)

    if not images:
        click.echo("Nothing to do.")
        sys.exit(0)

    # side
    if side == "NA":
        side = None

    inputs = [images[name] for name in images]
    outputs = [labels[name] for name in images]

    # segment images
    click.echo(f"Segmenting {len(images)} volume(s)...")
    api.run_model(model, inputs, outputs, side=side, tempdir=tempdir, copy_inputs=copy_inputs)

    click.echo("Done.")


@cli.command(context_settings={"show_default": True})
@click.argument("model")
@click.argument("images")
@click.argument("refs")
@click.option("--labelfile", type=click.Path(exists=True), required=True, help="ITK-Snap label file")
@click.option("--train/--no-train", default=True, help="Train model.")
@click.option("--dockerfile/--no-dockerfile", default=True, help="Make dockerfile.")
@click.option("--preprocess/--no-preprocess", default=True, help="enable or not the preprocessing part")
@click.option("-r", "--root", type=click.Path(exists=True), help="Root directory for training data.")
@click.option("-d", "--dest", type=click.Path(), help="Output directory for model files.")
@click.option("--nchannel", type=int, default=1, help="Expected number of channels")
@click.option("--folds", help="specify fold numbers to train as tuple")
@click.option("--nepoch", type=click.Choice(["1", "10", "20", "50", "100", "250", "500", "750", "1000"]), default="250", help="Number of epochs")
# @click.option("--split", is_flag=True, help="Split datasets into left and right parts")
@click.option("--continue", "continue_training", is_flag=True, help="Continue training.")
@click.option("-v", "--verbose", is_flag=True)
def train(model, images, refs, train, dockerfile, nchannel, labelfile, root, dest, verbose, folds, preprocess, nepoch, continue_training):
    """Train segmentation model on dataset

    \b
    MODEL: segmentation model to create (eg.: `museg-legs:model1`)
    IMAGES: path expression to specify training images
    REFS: path expression to specify reference segmentation
    
    """

    if verbose:
        logging.basicConfig(level=logging.INFO)
    nepoch = int(nepoch)

    # output dir
    if not dest:
        dest = pathlib.Path(".") / model.replace(":", "_")
    else:
        dest = pathlib.Path(dest.strip())

    if set(pathlib.Path(dest).glob("*")) and train:
        click.confirm(f"Output folder `{dest}` is not empty, do you want to continue?", abort=True)

    if folds is None:
        folds = (0, 1, 2, 3, 4)
    else:
        folds = tuple(map(int, folds.split(",")))

    # find images
    if pathlib.Path(images).is_absolute():
        # assume a directory
        image_files = sorted(pathlib.Path(images).rglob("*"))
        roi_files = sorted(pathlib.Path(refs).rglob("*"))
    else:
        root = pathlib.Path(root) if root else pathlib.Path(".")
        image_files = sorted(root.rglob(images))
        roi_files = sorted(root.rglob(refs))

    if not image_files:
        click.echo(f"No image file found, check expression: {images}")
    if not roi_files:
        click.echo(f"No label file found, check expression: {refs}")

    regex = re.compile(r"(.+?)(\d*)\.([\.\w]+)$")
    images = {}
    for file in image_files:
        match = regex.match(str(file))
        if not match or not api.is_image(file):
            continue
        prefix, index, ext = match.groups()
        images.setdefault(prefix, []).append(file)

    regex = re.compile(r"(.+?)(20\d\d\d\d\d\d)(.+)")
    rois, roi_dates = {}, {}
    for file in roi_files:
        match = regex.match(str(file))
        if not api.is_image(file):
            continue
        elif not match:  # no date
            prefix = str(file)
            date = 0
        else:
            prefix, date, _ = match.groups()
        rois.setdefault(prefix, []).append(file)
        roi_dates.setdefault(prefix, []).append(date)

    images = [tuple(sorted(images[prefix]))[:nchannel] for prefix in sorted(images)]
    rois = [tuple(sorted(rois[prefix]))[-1] for prefix in sorted(rois)]
    roi_dates = tuple(roi_dates.values())

    nimage = len(images)
    if len(rois) != nimage:
        click.echo(f"Error: found {nimage} images and {len(rois)} rois.")
        for ims in images:
            click.echo(f'\t{", ".join(map(str, ims))}')
        for im in rois:
            click.echo(f"\t{im}")
        sys.exit(0)

    if not images:
        click.echo(f"No image files were found")
        sys.exit(0)

    # check num channels
    channels = {len(ims) for ims in images}
    if not channels == {nchannel}:
        click.echo(f"Error: invalid number of channels (must be {nchannel})")
        for ims in images:
            click.echo(f'\t{", ".join(map(str, ims))}')
        sys.exit(0)

    # check training data
    click.echo(f"Found {len(images)} images and matching rois:")
    for i in range(nimage):
        click.echo(f"({i+1})")
        for j in range(nchannel):
            click.echo(f"\tchan. {j + 1:02d}: {images[i][j]}")
        click.echo(f"\tlabels:   {rois[i]}")
        if len(roi_dates[i]) > 1:
            click.echo(f'\t(latest among: {", ".join(roi_dates[i])})')
    ans = click.confirm("Are all images/rois correctly matched?", abort=True)

    # train model
    split_axis = 0 #None if not split else 0
    api.train_model(
        model,
        images,
        rois,
        labelfile,
        dest,
        split_axis=split_axis,
        train_model=train,
        make_dockerfile=dockerfile,
        folds=folds,
        preprocess=preprocess,
        nepoch=nepoch,
        continue_training=continue_training,
    )


@cli.command(context_settings={"show_default": True})
@click.argument("model")
@click.argument("data", nargs=-1)
# @click.option('-o', "--overwrite", is_flag=True, help="Overwrite already existing files.")
@click.option("-r", "--root", type=click.Path(exists=True), help="Input root directory.")
@click.option("-d", "--dest", type=click.Path(), help="Output root directory.")
@click.option("--filename", default="roi", help="Segmentation filename.")
@click.option("--dirname", help="Segmentation parent folder name.")
@click.option("-f", "--format", default=".nii.gz", type=click.Choice([".nii.gz", ".mha", ".mhd", ".hdr"]))
@click.option("--side", default="LR", type=click.Choice(["L", "R", "LR", "NA"]), help="Limb's side(s) in image")
@click.option("--tempdir", type=click.Path(exists=True), help="Location for nnUNet temporary files.")
@click.option("-v", "--verbose", is_flag=True, help="Show more information")
def test(model, data, root, dest, filename, dirname, format, side, tempdir, verbose):
    """Test segmentation model on dataset

    \b  
    MODEL: segmentation model to use (cf. `museg-ai list`)
    DATA: path expressions for: image dataset, [prediction segmentations], reference segmentations
    \b
    case 1: DATA = <images> <references>
        Run inference on images before comparing to references.
    case 2: DATA = <images> <predictions> <references>
        Predictions are provided, inference is not run.

    """
    if verbose:
        logging.basicConfig(level=logging.INFO)

    # copy inputs to outputs
    copy_inputs = (dirname is not None) or (dest is not None)

    # root directory
    root = pathlib.Path(root) if root else pathlib.Path(".")

    if not data:
        # no argument: list available models
        models = api.list_models()
        click.echo("Available segmentation models:")
        for available_model in models:
            click.echo(f"\t{available_model}")
        sys.exit(0)

    elif len(data) == 2:
        images, refs = data
        preds = None

    elif len(data) == 3:
        images, preds, refs = data
    
    else:
        click.echo(f"Invalid number of data arguments: {len(data)}")
        sys.exit(0)

    # make inference?
    make_inference = preds is None

    # side
    if side == "NA":
        side = None

    # get nchannel from model metadata
    try:
        model_info = api.get_model_info(model)
    except ValueError as exc:
        click.echo(f'Error: {exc}')
        sys.exit(1)
    nchannel = int(model_info["nchannel"])

    # find images
    if pathlib.Path(images).is_absolute():
        # assume a directory
        image_files = sorted(pathlib.Path(images).rglob("*"))
        refs_files = sorted(pathlib.Path(refs).rglob("*"))
        preds_files = None if preds is None else sorted(pathlib.Path(preds).rglob("*"))
    else:
        root = pathlib.Path(root) if root else pathlib.Path(".")
        image_files = sorted(root.rglob(images))
        refs_files = sorted(root.rglob(refs))
        preds_files = None if preds is None else sorted(root.rglob(preds))

    if not image_files:
        click.echo(f"No image file found, check expression: {images}")
    if not refs_files:
        click.echo(f"No reference label file found, check expression: {refs}")
    if preds and not refs_files:
        click.echo(f"No prediction label file found, check expression: {preds}")

    # sort images
    regex = re.compile(r"(.+?)(\d*)\.([\.\w]+)$")
    images = {}
    for file in image_files:
        match = regex.match(str(file))
        if not match or not api.is_image(file):
            continue
        prefix, index, ext = match.groups()
        images.setdefault(prefix, []).append(file)

    # sort predictions
    if preds:
        regex = re.compile(r"(.+?)\.([\.\w]+)$")
        preds = {}
        for file in preds_files:
            match = regex.match(str(file))
            if not match or not api.is_image(file):
                continue
            prefix, ext = match.groups()
            preds[prefix] = file

    # sort references
    regex = re.compile(r"(.+?)(20\d\d\d\d\d\d)(.+)")
    refs, refs_dates = {}, {}
    for file in refs_files:
        match = regex.match(str(file))
        if not api.is_image(file):
            continue
        elif not match:  # no date
            prefix = str(file)
            date = 0
        else:
            prefix, date, _ = match.groups()
        refs.setdefault(prefix, []).append(file)
        refs_dates.setdefault(prefix, []).append(date)

    # destination
    names = sorted(images)
    dest = pathlib.Path(root if dest is None else dest)
    dest.mkdir(exist_ok=True, parents=True)
    destloc = {name: images[name][0].relative_to(root).parent for name in names}
    if dirname:
        destloc = {name: loc.parent / dirname for name, loc in destloc.items()}
    

    # list images and label files
    images = [tuple(sorted(images[prefix]))[:nchannel] for prefix in sorted(images)]
    refs = [tuple(sorted(refs[prefix]))[-1] for prefix in sorted(refs)]
    refs_dates = tuple(refs_dates.values())
    if preds:
        preds = [preds[prefix] for prefix in sorted(preds)]
    nimage = len(images)

    # check images
    if not images:
        click.echo(f"No image files were found")
        sys.exit(0)

    # check number of references
    if len(refs) != nimage:
        click.echo(f"Error: found {nimage} images and {len(refs)} reference labels.")
        for ims in images:
            click.echo(f'\t{", ".join(map(str, ims))}')
        for im in refs:
            click.echo(f"\t{im}")
        sys.exit(0)

    # check predictions
    if preds and len(preds) != nimage:
        click.echo(f"Error: found {nimage} images and {len(refs)} reference labels.")
        for ims in images:
            click.echo(f'\t{", ".join(map(str, ims))}')
        for im in preds:
            click.echo(f"\t{im}")
        sys.exit(0)

    # check num channels
    channels = {len(ims) for ims in images}
    if not channels == {nchannel}:
        click.echo(f"Error: invalid number of channels (must be {nchannel})")
        for ims in images:
            click.echo(f'\t{", ".join(map(str, ims))}')
        sys.exit(0)

    # check test data
    click.echo(f"Found {len(images)} images and matching rois:")
    for i in range(nimage):
        click.echo(f"({i+1})")
        for j in range(nchannel):
            click.echo(f"\tchan. {j + 1:02d}: {images[i][j]}")
        click.echo(f"\tref.:   {refs[i]}")
        if len(refs_dates[i]) > 1:
            click.echo(f'\t(latest among: {", ".join(refs_dates[i])})')
        if preds:
            click.echo(f"\tpred.:  {preds[i]}")
    click.confirm("Are all images/rois correctly matched?", abort=True)

    # prediction
    if preds is None:
        preds = {name: (dest / destloc[name] / filename).with_suffix(format) for name in names}

    # test model
    click.echo(f"Testing {model} on {len(images)} volume(s)...")
    opts = {
        "inference": make_inference,
        'tempdir': tempdir,
        'copy_inputs': copy_inputs,
        'side': side,
        'statsfile': dest / 'stats.csv',
        'figfile': dest / 'stats.png',
        'figtitle': f'{model} - {root.relative_to('.')}/',
    }
    api.test_model(model, images, refs, preds, **opts)
    
    click.echo("Done.")



@cli.command(context_settings={"show_default": True})
@click.argument("model")
@click.argument("folder", type=click.Path())
@click.option("-v", "--verbose", is_flag=True)
def dockerize(model, folder, verbose):
    """Put trained model into docker """

    if verbose:
        logging.basicConfig(level=logging.INFO)

    api.dockerize_model(model, folder)
    click.echo("Done.")


if __name__ == "__main__":
    cli()  # pylint: disable=no-value-for-parameter
