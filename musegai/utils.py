import pathlib
import re
from . import api

def find_files(root, pattern, nchannel=1):
    """ find files in root directory based on pattern """
    root = pathlib.Path(root)
    files = sorted(root.rglob(pattern))

    # handle numbered files
    regex = re.compile(r"(.+?)(\d*)\.([\.\w]+)$")
    prefixes = {}
    for file in files:
        match = regex.match(str(file))
        if not match or not api.is_image(file):
            continue
        prefix, index, ext = match.groups()
        prefixes.setdefault(prefix, []).append(file)
    
    # check number of channels
    files = [tuple(sorted(prefixes[prefix]))[:nchannel] for prefix in sorted(prefixes)]

    # check number of channels
    nchan = {len(channels) for channels in files}
    if not nchan == {nchannel}:
        raise ValueError(f"Error: invalid number of channels (must be {nchannel})")

    return files


def print_files(files, msg=None):
    if msg:
        print(msg)
    for i in range(len(files)):
        print(f"({i+1})")
        for j, file in enumerate(files[i]):
            print(f"\t{j + 1:02d}: {file}")


def print_files_compare(files1, files2, msg=None):
    if len(files1) != len(files2):
        raise ValueError()
    if msg:
        print(msg)
    for i in range(len(files1)):
        print(f"({i+1:02d})")
        for j, file in enumerate(files1[i]):
            print(f"\t[{j + 1}] {file}")
        for j, file in enumerate(files2[i]):
            print(f"\t[{j + 1}] {file}")