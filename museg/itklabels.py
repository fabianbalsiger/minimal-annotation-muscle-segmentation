# -*- coding: utf-8 -*-
""" i/o for ITKSnap-like labels """

import re
import itertools
from collections import abc

COLOR_MAP = [
    [228, 26, 28],
    [55, 126, 184],
    [77, 175, 74],
    [152, 78, 163],
    [255, 127, 0],
    [255, 255, 51],
    [166, 86, 40],
    [247, 129, 191],
    [153, 153, 153],
]

PREAMBLE = (
    "################################################\n"
    "# Label Description File\n"
    "# File format: \n"
    "# IDX   -R-  -G-  -B-  -A--  VIS MSH  LABEL\n"
    "# Fields:\n"
    "#    IDX:   Zero-based index \n"
    "#    -R-:   Red color component (0..255)\n"
    "#    -G-:   Green color component (0..255)\n"
    "#    -B-:   Blue color component (0..255)\n"
    "#    -A-:   Label transparency (0.00 .. 1.00)\n"
    "#    VIS:   Label visibility (0 or 1)\n"
    "#    MSH:   Label mesh visibility (0 or 1)\n"
    "#  LABEL:   Label description \n"
    "################################################\n"
)


def read_labels(filename, **kwargs):
    return Labels.load(filename, **kwargs)


def write_labels(filename, labels, **kwargs):
    return Labels.save(filename, labels, **kwargs)


class Labels(abc.Mapping):
    """dict-like class containing the ROI labels

    # create
    # using an (index, decsription) dict
    labels = Labels({1: "label_1", 2: "label_2"})
    labels = Labels.load(filename) # load a label file

    # save
    labels.save(filename) # save labels to file
    Labels.save(filename, labels) # save labels to file

    # access
    name = labels[1] # get description for index 1

    name = labels.descriptions[0] # get first label description
    index = labels.indices[0] # get first index value
    rgba = labels.rgba[0] # get first label rgba tuple

    # set
    labels.name[1] = "another name"
    etc.

    """

    # label_re_syntax = r"(?P<name>[\w\. -\"\+]+)"
    label_re_syntax = r"(?P<name>[^\t]+)"
    re_parse_label = re.compile(label_re_syntax)
    re_parse_line = re.compile(
        (
            r"^\s*(?P<index>\d+)"
            r"\s+(?P<R>\d+)"
            r"\s+(?P<G>\d+)"
            r"\s+(?P<B>\d+)"
            r"\s+(?P<A>\d|(\d\.\d+))"
            r"\s+(?P<vis>[0|1])"
            r"\s+(?P<meshvis>[0|1])"
            r"\s+\"" + label_re_syntax + r"\"\s*$"
        )
    )

    @classmethod
    def load(cls, filename, **kwargs):
        """load label file"""
        labels = {}
        with open(filename, "r") as fp:
            lines = fp.read().splitlines()

            for line in lines:
                if "#" in line:
                    line = line[: line.find("#")].strip()
                if not line:
                    # empty line
                    continue

                # parse line
                match = cls.re_parse_line.match(line)
                if not match:
                    raise RuntimeError(f"Could not parse line: {line}")

                # index is the first number
                index = int(match.group("index"))

                # color, including alpha
                rgba = (
                    int(match.group("R")),
                    int(match.group("G")),
                    int(match.group("B")),
                    float(match.group("A")),
                )
                desc = match.group("name")
                labels[index] = {"LABEL": desc, "RGBA": rgba}

        return cls(labels, **kwargs)

    @classmethod
    def save(cls, filename, labels, **kwargs):
        """save labels to file"""
        if isinstance(labels, dict):
            labels = cls(labels, **kwargs)
        elif not isinstance(labels, cls):
            raise ValueError("Invalid type for labels: %s" % labels)

        with open(filename, "w") as f:
            # write preamble
            f.write(PREAMBLE)
            for index in labels:
                info = labels.data[index]
                rgba = info["RGBA"]
                descr = info["LABEL"]
                vis = info["VIS"]
                msh = info["MSH"]
                alpha = float(rgba[3])
                line = ('{:>5} {:>5}{:>5}{:>5} {:>9}{:>3}{:>3}    "{:}" \n').format(
                    index, rgba[0], rgba[1], rgba[2], alpha, vis, msh, descr
                )
                f.write(line)

    def __init__(self, labels, allow_duplicates=False):
        """init label object"""
        if isinstance(labels, Labels):
            self.data = labels.data.copy()
            return
        elif not isinstance(labels, dict):
            raise ValueError("A dict is required")

        self.color_gen = itertools.cycle(COLOR_MAP)
        data = {}
        descriptions = set()
        for index in sorted(labels):
            info = labels[index]
            if isinstance(info, str):
                # label description only
                info = {"LABEL": info}
            elif not isinstance(info, dict):
                raise ValueError("Invalid label info: %s" % info)

            # full description
            descr = info["LABEL"]
            rgba = info.get("RGBA", (0, 0, 0, 0.0) if index == 0 else self.newcolor())
            vis = info.get("VIS", 1)
            meshvis = info.get("MSH", 1)

            # if not isinstance(descr, str):
            if not self.re_parse_label.match(descr):
                raise ValueError("Invalid label description: %s" % descr)
            elif not allow_duplicates and descr in descriptions:
                raise ValueError("Duplicate description: %s" % descr)

            if (
                len(rgba) != 4
                or not all(isinstance(v, int) for v in rgba[:3])
                or not float(rgba[-1]) <= 1
            ):
                raise ValueError("Invalid RGBA format: %s" % str(rgba))

            if not (vis in [0, 1] and meshvis in [0, 1]):
                raise ValueError("Invalid visibility values: %s" % str(info))
            data[index] = {
                "LABEL": descr,
                "RGBA": tuple(rgba),
                "VIS": vis,
                "MSH": meshvis,
            }
            descriptions.add(descr)

        self.data = data

    @property
    def indices(self):
        return list(self.data)

    def __repr__(self):
        return str(self.data)

    def __len__(self):
        return len(self.indices)

    def __iter__(self):
        return iter(sorted(self.indices))

    def __eq__(self, other):
        return self.data == other.data

    def __contains__(self, index):
        return index in self.indices

    def __getitem__(self, index):
        """return label description for given index"""
        return self.data[index]["LABEL"]

    def __setitem__(self, index, descr):
        """set label description"""
        self.data[index]["LABEL"] = descr

    def __delitem__(self, index):
        del self.data[index]

    def keys(self):
        return self.data.keys()

    def values(self):
        return (item["LABEL"] for item in self.data.values())

    def get(self, index, default=None):
        """get label description"""
        try:
            return self[index]
        except KeyError:
            return default

    def pop(self, index, **kwargs):
        return self.data.pop(index, **kwargs)

    def getindex(self, name):
        """return index of given name (reverse of .__getitem__)"""
        return {self.data[index]["LABEL"]: index for index in self.indices}[name]

    @property
    def colors(self):
        """return dict of colors"""
        return dict((idx, self.data[idx]["RGBA"]) for idx in self)

    @property
    def palette(self):
        """return RBG palette (eg. for using with PIL)"""
        palette = []
        for i in range(256):
            if i in self:
                palette.extend(self.data[i]["RGBA"][:-1])
            else:
                palette.extend((0, 0, 0))
        return palette

    def newcolor(self, alpha=1.0):
        """return a different color every time"""
        return next(self.color_gen) + [alpha]
