import pathlib
import sys
import click
import SimpleITK as sitk

from musegai import api, itklabels

@click.command()
@click.argument('volumes', type=click.Path(exists=True), nargs=-1)
@click.option('-d', '--dest', type=click.Path(), help='Output filename.')
@click.option('--model', default='TODO', help='Specify the segmentation model.')
@click.option('--side', default='left+right', help='Specify the limb\'s side(s).')
@click.option('--fmt', default='.mha', help='Output format.')
@click.option('--tempdir', type=click.Path(exists=True), help='Location for temporary files.')
def cli(volumes, dest, model, side, fmt, tempdir):
    """ Automatic muscle segmentation """

    if not volumes:
        click.echo('Available segmentation models:')
        for model in api.list_models():
            click.echo(f'\t{model}')
        sys.exit(0)

    elif len(volumes) == 2:
        name = pathlib.Path(volumes[0]).stem
        inputs = {name: [load_volume(vol) for vol in volumes]}
        volumes = {name: [data['array'] for data in inputs[name]] for vol in inputs}
        info = {name: inputs[name][0]['info'] for name in inputs}
        
    else:
        click.echo('You must provide two volume files')
        sys.exit(0)


    # destination
    if dest is None:
        dest = (pathlib.Path('.') / name).with_suffix(fmt)
    else:
        dest = pathlib.Path(dest)
        dest.parent.mkdir(exist_ok=True, parents=True)
    if dest.is_file():
        click.echo(f'Output file already exists: {dest}')
        sys.exit(0)


    # segment volumes
    segmented, labels = api.segment_volumes(volumes, model, side=side, tempdir=tempdir)


    # save
    itklabels.write_labels(dest.parent / 'labels.txt', labels)
    for name, vol in segmented.items():
        save_volume(dest, vol, info[name])


#
# functions

def load_volume(file):
    im = sitk.ReadImage(file)
    array = sitk.GetArrayFromImage(im).T
    info = {'spacing': im.GetSpacing(), 'origin': im.GetOrigin(), 'direction': im.GetDirection() }
    return {'array': array, 'info': info}

def save_volume(file, vol, info):
    im = sitk.GetImageFromArray(vol.T)
    im.SetSpacing(info['spacing'])
    im.SetOrigin(info['origin'])
    im.SetDirection(info['direction'])
    sitk.WriteImage(im, file)


if __name__ == '__main__':
    cli()