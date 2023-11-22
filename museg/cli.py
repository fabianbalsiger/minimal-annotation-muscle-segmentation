import pathlib
import sys

import click
import api


@click.command()
@click.argument("volumes", type=click.Path(exists=True), nargs=-1)
@click.option("-d", "--dest", type=click.Path(), help="Output filename.")
@click.option("--model", default="thigh-model3", help="Specify the segmentation model.")
@click.option("--side", default="left+right", help="Specify the limb's side(s).")
@click.option("--fmt", default=".mha", help="Output format.")
@click.option("--tempdir", type=click.Path(exists=True), help="Location for temporary files.")
def cli(volumes, dest, model, side, fmt, tempdir):
    """Automatic muscle segmentation."""

    if not volumes:
        click.echo("Available segmentation models:")
        for model in api.list_models():
            click.echo(f"\t{model}")
        sys.exit(0)

    elif len(volumes) == 2:
        volumes = [api.Volume.load(file) for file in volumes]
        name = volumes[0].info["name"]
        volumes = {name: volumes}

    else:
        click.echo("You must provide two volume files")
        sys.exit(0)

    # destination
    if dest is None:
        dest = (pathlib.Path(".") / name).with_suffix(fmt)
    else:
        dest = pathlib.Path(dest)
        dest.parent.mkdir(exist_ok=True, parents=True)
    if dest.is_file():
        click.echo(f"Output file already exists: {dest}")
        sys.exit(0)

    # segment volumes
    segmented, labels = api.segment_volumes(volumes, model, side=side, tempdir=tempdir)

    # save
    labels.save(dest.parent / "labels.txt")
    for name, vol in segmented.items():
        vol.save(dest, volumes[name][0].info["extension"])


if __name__ == "__main__":
    cli()
