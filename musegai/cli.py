"""Command line interface for muscle segmentation."""
from __future__ import annotations

import pathlib
import re
import sys

import click

from musegai import api


@click.command(context_settings={"show_default": True})
@click.argument("volumes", type=click.Path(exists=True), nargs=-1)
@click.option("-d", "--dest", type=click.Path(), help="Output directory.")
@click.option("--model", default="thigh-model3", help="Specify the segmentation model.")
@click.option("--side", default="left+right", type=click.Choice(api.SIDES), help="Specify the limb's side(s).")
@click.option("--tempdir", type=click.Path(exists=True), help="Location for temporary files.")
def cli(volumes, dest, model, side, tempdir):
    """Automatic muscle segmentation command line tool.

    \b
    VOLUMES can be:
        - (nothing): show available segmentation models
        - two matching Dixon volumes to segment
        - a single directory with numbered pairs of matching Dixon volumes
    """
    if not volumes:
        # no argument: list available models
        click.echo("Available segmentation models:")
        for available_model in api.list_models():
            click.echo(f"\t{available_model}")
        sys.exit(0)

    if (len(volumes) == 1) and pathlib.Path(volumes[0]).is_dir():
        # a folder with volume pairs
        root = pathlib.Path(volumes[0])
        regex = re.compile(r"(.+?)(\d+).[\w.]+$")
        volumes = {}
        for file in sorted(root.glob("*")):
            match = regex.match(file.name)
            if not match:
                continue
            name, _ = match.groups()
            vol = api.Volume.load(file)
            volumes.setdefault(name, []).append(vol)
            if len(volumes[name]) > 2:
                click.echo(f"Expecting two volume files with prefix: {name}")
        click.echo(f"Found {len(volumes)} volume pair(s) to segment:")
        for name in volumes:
            click.echo(f"\t{name}")

    elif len(volumes) == 2 and all(pathlib.Path(file).is_file() for file in volumes):
        # individual files
        root = "."
        volumes = [api.Volume.load(file) for file in volumes]
        name = volumes[0].info["name"]
        volumes = {name: volumes}
        click.echo(f"Found one volume pair to segment: {name}")

    else:
        # invalid number of arguments
        click.echo("Expecting two volume files or a single directory")
        sys.exit(0)

    # destination
    dest = pathlib.Path(root if dest is None else dest)
    dest.mkdir(exist_ok=True, parents=True)
    destfiles = {name: (dest / name).with_suffix(volumes[name][0].info["extension"]) for name in volumes}
    for name in destfiles:
        if destfiles[name].is_file():
            click.echo(f"Output file already exists: {destfiles[name]}, skipping")
            volumes.pop(name)

    if not volumes:
        click.echo("Nothing to do.")
        sys.exit(0)

    # segment volumes
    click.echo(f"Segmenting {len(volumes)} volume(s)...")
    segmented, labels = api.segment_volumes(volumes, model, side=side, tempdir=tempdir)

    # save
    click.echo(f"Saving results to `{dest}`")
    labels.save(dest / "labels.txt")
    for name, vol in segmented.items():
        vol.save(destfiles[name])

    click.echo("Done.")


if __name__ == "__main__":
    cli()  # pylint: disable=no-value-for-parameter
