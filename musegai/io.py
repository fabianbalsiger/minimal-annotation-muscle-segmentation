import pathlib
import re
import numpy as np
import SimpleITK as sitk
import csv


def is_image(obj):
    if isinstance(obj, Image):
        return True
    elif isinstance(obj, (str, pathlib.Path)):
        return Image.is_imagefile(obj)
    raise TypeError(f"Invalid image object or file: {obj}")


def load(obj, **kwargs):
    """load image file"""
    if isinstance(obj, (np.ndarray, Image)):
        # copy image/array obj
        return Image(obj, **kwargs)
    # else assume a file file
    return Image.load(obj, **kwargs)


def save(file, image, ext=None):
    image.save(file, ext=ext)


def load_labels(obj):
    if isinstance(obj, Labels):
        return obj
    elif isinstance(obj, (str, pathlib.Path)):
        return Labels.load(obj)
    raise TypeError(f"Invalid labels object or file: {obj}")


def save_labels(filename, labels):
    labels.save(filename)


def init_labels(num):
    indices = list(range(num))
    descr = [f"Label {i + 1}" for i in range(num)]
    return Labels(indices, descr)


def split(image, axis, auto=True):
    """split images along axis"""
    if not axis in tuple(range(image.ndim)):
        raise ValueError(f"Invalid axis: {axis}")
    
    if auto:
        mask = image.array > np.percentile(np.unique(image.array), 5)
        center = np.mean(np.stack(np.nonzero(mask)), axis=1)
        nx = int(center[axis] + 0.5)
    else:
        nx = image.shape[axis] // 2

    # first half
    slices = [slice(nx) if i == axis else slice(None) for i, n in enumerate(image.shape)]
    first = Image(image.array[tuple(slices)], **image.metadata)
    # second half
    slices = [slice(nx, n) if i == axis else slice(None) for i, n in enumerate(image.shape)]
    origin = list(image.origin)
    origin[axis] = image.origin[axis] + image.spacing[axis] * nx
    second = Image(image.array[tuple(slices)], **{**image.metadata, "origin": origin})
    return first, second


def heal(imageA, imageB, axis):
    """heal splitted images"""
    arr = np.concatenate([imageA, imageB], axis=axis)
    return Image(arr, **imageA.metadata)


class Image:
    """Image container."""

    EXTENSIONS = [".mha", ".mhd", ".hdr", ".nii", ".nii.gz"]

    def __init__(self, obj, **meta):
        try:
            self.array = obj if isinstance(obj, np.ndarray) else np.asarray(obj)
            self.origin = meta.pop("origin", None) or getattr(obj, "origin")
            self.spacing = meta.pop("spacing", None) or getattr(obj, "spacing")
            self.transform = meta.pop("transform", None) or getattr(obj, "transform")
        except AttributeError as exc:
            raise TypeError(f"Missing argument or attribute: {exc.name}") from exc
        self.info = {**meta.pop("info", {}), **meta}

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
        return {
            "origin": self.origin,
            "spacing": self.spacing,
            "transform": self.transform,
            "info": self.info,
        }

    def save(self, file, ext=None):
        file = pathlib.Path(file)
        array = self.array.T
        if np.issubdtype(self.array.dtype, np.integer):
            array = array.astype(np.uint8)
        im = sitk.GetImageFromArray(array)
        im.SetSpacing(self.spacing)
        im.SetOrigin(self.origin)
        im.SetDirection(self.transform)
        if not file.suffix and ext:
            file = pathlib.Path(file).with_suffix(ext)
        sitk.WriteImage(im, file)

    @classmethod
    def load(cls, file, **kwargs):
        file = pathlib.Path(file)
        name, ext = cls._get_file_ext(file)
        im = sitk.ReadImage(file)
        array = sitk.GetArrayFromImage(im).T
        spacing = im.GetSpacing()
        origin = im.GetOrigin()
        transform = im.GetDirection()
        info = {"extension": ext, "name": name}

        return cls(array, origin=origin, spacing=spacing, transform=transform, info=info, **kwargs)

    @classmethod
    def is_imagefile(cls, file):
        return any(str(file).endswith(suffix) for suffix in cls.EXTENSIONS)

    @classmethod
    def _get_file_ext(cls, filename):
        filename = pathlib.Path(filename)
        for ext in cls.EXTENSIONS:
            if str(filename).endswith(ext):
                name = filename.name.split(ext)[0]
                return name, ext
        else:
            raise TypeError(f"Unknown file type: {filename}")


class Labels:
    """dict-like Label container"""

    RE_LABEL = re.compile(r'(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+([\.\d]+)\s+(\d)\s+(\d)\s+"([\w\s\-]+)"$')

    def __init__(self, indices, descriptions, colors=None, transparency=None, visibility=None):
        self.indices = list(map(int, indices))
        nlabels = len(self.indices)
        assert len(descriptions) == nlabels
        self.descriptions = list(map(str, descriptions))
        if colors is not None:
            assert len(colors) == nlabels
            self.colors = list(tuple(map(int, color)) for color in colors)
        else:
            self.colors = [tuple(np.random.randint(0, 255, 3)) for _ in range(nlabels)]
        if transparency is not None:
            assert len(transparency) == nlabels
            self.transparency = list(map(float, transparency))
        else:
            self.transparency = [1] * nlabels
        if visibility is not None:
            assert len(visibility) == nlabels
            self.visibility = list(map(int, visibility))
        else:
            self.visibility = [1] * nlabels

    def __repr__(self):
        return f"Labels({len(self)})"

    def __len__(self):
        return len(self.indices)

    def __iter__(self):
        return iter(self.indices)

    def __getitem__(self, item):
        if isinstance(item, int):
            dct = dict(zip(self.indices, self.descriptions))
        elif isinstance(item, str):
            unique = set()
            dct = {d: i for d, i in zip(self.descriptions,self.indices) if not (d in unique or unique.add(d))}
        else:
            raise ValueError(f"Invalid item type: {item}")
        return dct[item]

    def to_dict(self):
        return dict(zip(self.indices, self.descriptions))

    def append(self, description, *, color=None, transparency=1, visibility=1):
        if color is None:
            color = np.random.randint(0, 255, 3)
        color = tuple(color)
        self.indices.append(max(self.indices) + 1)
        self.descriptions.append(str(description))
        self.colors.append(color)
        self.transparency.append(transparency)
        self.visibility.append(visibility)

    def remove(self, item, reindex=True):
        if isinstance(item, int):
            index = self.indices.index(item)
        if isinstance(item, str):
            index = self.descriptions.index(item)
        else:
            raise ValueError(f"Invalid item type: {item}")
        indices = [i for i in self.indices if i != index]
        return self.subset(indices, reindex=reindex)

    def subset(self, indices, reindex=True):
        num = len(indices)
        true_indices = {self.indices.index(i) for i in indices}
        return Labels(
            list(range(num)) if reindex else indices,
            [self.descriptions[i] for i in true_indices],
            [self.colors[i] for i in true_indices],
            [self.transparency[i] for i in true_indices],
            [self.visibility[i] for i in true_indices],
        )

    @classmethod
    def load(cls, file):
        indices, descr, colors, transp, visib = [], [], [], [], []
        with open(file, "r") as fp:
            for line in fp:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                # parse line
                match = cls.RE_LABEL.match(line)
                if not match:
                    raise ValueError(f"Invalid syntax in file {file}: {line}")
                idx, r, g, b, a, v, m, d = match.groups()
                indices.append(int(idx))
                colors.append((int(r), int(g), int(b)))
                transp.append(float(a))
                visib.append(int(v))
                descr.append(d)
        return Labels(indices, descr, colors, transp, visib)

    def save(self, file):
        with open(file, "w") as fp:
            fp.write(self.HEADER)
            for i in range(len(self)):
                idx = self.indices[i]
                r, g, b = self.colors[i]
                a = self.transparency[i]
                v = self.visibility[i]
                d = self.descriptions[i]
                line = f'{idx:5d} {r:5d} {g:5d} {b:5d} {a:9.2f} {v:2d} {1:2d}    "{d}"\n'
                fp.write(line)

    HEADER = """################################################
# Label Description File
# File format: 
# IDX   -R-  -G-  -B-  -A--  VIS MSH  LABEL
# Fields:
#    IDX:   Zero-based index 
#    -R-:   Red color component (0..255)
#    -G-:   Green color component (0..255)
#    -B-:   Blue color component (0..255)
#    -A-:   Label transparency (0.00 .. 1.00)
#    VIS:   Label visibility (0 or 1)
#    MSH:   Label mesh visibility (0 or 1)
#  LABEL:   Label description 
################################################
"""


# write excel
def save_csv(filename, data):
    fieldnames = []
    [key for row in data for key in row if not (key in fieldnames or fieldnames.append(key))]
    with open(filename, 'w', newline='', encoding='utf-8') as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames, dialect='excel')
        writer.writeheader()
        for row in data:
            writer.writerow(row)