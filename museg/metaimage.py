# coding=utf-8

""" I/O for metaimage format """

import os
import sys
import array
import io

import gzip, zlib
import logging
from collections import OrderedDict

import numpy as np

LOGGER = logging.getLogger(__name__)

FILE_EXTENSIONS = [".mhd", ".mha"]

IS_BIG_ENDIAN = sys.byteorder != "little"

NUMPY_TO_METAIO_TYPES = {
    "float32": "MET_FLOAT",
    "float64": "MET_DOUBLE",
    "uint8": "MET_UCHAR",
    "int8": "MET_CHAR",
    "uint16": "MET_USHORT",
    "int16": "MET_SHORT",
    "uint32": "MET_UINT",
    "int32": "MET_INT",
    "uint64": "MET_ULONG",
    "int64": "MET_LONG",
    "bool": "MET_UCHAR",
}

METAIO_TO_NUMPY_TYPES = {
    "MET_FLOAT": np.float32,
    "MET_DOUBLE": np.float64,
    "MET_UCHAR": np.uint8,
    "MET_CHAR": np.int8,
    "MET_USHORT": np.uint16,
    "MET_SHORT": np.int16,
    "MET_UINT": np.uint32,
    "MET_INT": np.int32,
    "MET_ULONG": np.uint64,
    "MET_LONG": np.int64,
    "MET_LONG_LONG": np.int64,
}


def read(filename, **kwargs):
    """load metaimage file and return numpy array and tags"""
    metaimage = MetaImage.load(filename, **kwargs)
    return metaimage.data, metaimage.tags


def write(filename, data, **kwargs):
    """write array and tags to metaimage file"""
    metaimage = MetaImage(data, **kwargs)
    MetaImage.save(filename, metaimage)


class MetaImage:
    """Simple metaimage i/o class
    Usage:
        data, tags = MetaImage.read(filename)
        MetaImage.write(filename, data, tags)

    """

    @property
    def data(self):
        return self._data

    @property
    def tags(self):
        return self._tags

    def __init__(self, data, isvector=None, ignore_errors=False, tags={}, **kwargs):
        """init metaimage from data array and tag values"""
        LOGGER.debug("Init metaimage")

        tags = {**tags, **kwargs}
        if isinstance(data, MetaImage):
            # if data is a MetaImage
            data = np.array(data.data)  # copy array
            _tags = data.tags.copy()

        else:
            # some other data type is passed
            data = np.asarray(data)
            _tags = MetaImageTags.from_array(data, isvector=isvector)

        self._data = data
        self._tags = _tags
        self.update(tags, ignore_errors=ignore_errors)

    def update(self, tags={}, ignore_errors=False, **kwargs):
        """update tag values"""
        tags = {**tags, **kwargs}
        if not tags:
            return

        LOGGER.debug("update metaimage")

        # reshape shape
        ndims = tags.get("NDims", self.tags["NDims"])
        dimsize = tags.get("DimSize", self.tags["DimSize"][:ndims])
        nchan = tags.get(
            "ElementNumberOfChannels", self.tags.get("ElementNumberOfChannels", 1)
        )
        chansize = [nchan] if nchan > 1 else []
        if {"DimSize", "ElementNumberOfChannels"} & set(tags):
            newshape = list(dimsize) + chansize
            self._data = self._data.reshape(newshape)

        # update data type
        if "ElementType" in tags:
            # change data type
            metaiotype = tags["ElementType"]
            dtype = METAIO_TO_NUMPY_TYPES[metaiotype]
            self._data = self._data.astype(dtype)
        else:
            # fix datatype
            dtype = self.data.dtype
            tags["ElementType"] = NUMPY_TO_METAIO_TYPES[str(dtype)]

        # update byte order
        if "BinaryDataByteOrderMSB" in tags:
            byte_order = tags["BinaryDataByteOrderMSB"] and ">" or "<"
            dtype = np.dtype(byte_order + self.data.dtype.char)
            self._data = self._data.astype(dtype)
        else:
            # fix byte order tag
            byteorder = self.data.dtype.byteorder
            byte_order_msb = {"=": IS_BIG_ENDIAN, ">": True, "<": False, "|": False}[
                byteorder
            ]
            tags["BinaryDataByteOrderMSB"] = byte_order_msb

        # fix anatomical orientation
        anat = tags.get(
            "AnatomicalOrientation", self.tags.get("AnatomicalOrientation", "RAI")
        )
        if ndims > 3:
            tags["AnatomicalOrientation"] = "???"
        else:
            tags["AnatomicalOrientation"] = anat

        # update MetaImageTags
        tags = MetaImageTags(self.tags, **tags)
        self.check(self.data, tags, ignore_errors=ignore_errors)
        self._tags = tags

    @classmethod
    def load(cls, filename, ignore_errors=False):
        """load metaimage from file"""
        LOGGER.debug("load metaimage")
        tags = MetaImageTags.load(filename, ignore_errors=ignore_errors)

        if tags["ElementDataFile"] == "LOCAL":
            with open(filename, "rb") as f:
                line = []
                while not b"ElementDataFile" in line:
                    line = f.readline()
                # data: remaining of file
                raw_data = f.read()
        else:
            try:
                dirname = os.path.dirname(filename)
                datafile = os.path.join(dirname, tags["ElementDataFile"])
                with open(datafile, "rb") as f:
                    raw_data = f.read()
            except IOError:
                raise IOError("Could not find data from file: %s" % filename)

        # if compressed
        if tags.get("CompressedData", False):
            try:
                raw_data = zlib.decompress(raw_data)
            except zlib.error:
                try:
                    raw_data = gzip.decompress(raw_data)
                except gzip.BadGzipFile:
                    raise ValueError("Unknown compression type")

        # data type
        element_type = np.dtype(METAIO_TO_NUMPY_TYPES[tags["ElementType"]]).char
        byte_order = tags["BinaryDataByteOrderMSB"] and ">" or "<"
        dtype = np.dtype(byte_order + element_type)

        # make data array
        dimsize = tags["DimSize"]
        nchan = tags.get("ElementNumberOfChannels", 1)
        shape = dimsize[::-1] + [nchan] if nchan > 1 else dimsize[::-1]
        data = np.frombuffer(raw_data, dtype=dtype).reshape(shape).T
        if nchan > 1:
            data = np.moveaxis(data, 0, -1)

        # create new image
        image = cls(data, ignore_errors=ignore_errors, tags=tags)
        return image

    @classmethod
    def check(cls, data, tags, ignore_errors=False):
        """check and fix tags errors"""
        LOGGER.debug("check metaimage data")

        # check data
        ndim = data.ndim
        shape = list(data.shape)
        data_type = NUMPY_TO_METAIO_TYPES[str(data.dtype)]

        nchan = tags.get("ElementNumberOfChannels", 1)
        chansize = [nchan] if nchan > 1 else []

        if ndim != tags["NDims"] + 1 * (nchan > 1):
            raise ValueError(f"Invalid NDims: {tags['NDims']}")

        if shape != tags["DimSize"] + chansize:
            #   if shape != chansize + tags["DimSize"]:
            raise ValueError(f"Invalid DimSize: {tags['DimSize']}")

        if data_type != tags["ElementType"]:
            raise ValueError(f"Invalid data type: {tags['ElementType']}")

        if ignore_errors:
            return

        # check geometrical tags
        ndims = tags["NDims"]

        origin = tags.get("Origin", tags.get("Position", tags.get("Offset")))
        if origin and len(origin) != ndims:
            raise ValueError(f"Invalid Origin/Position/Offset tag: {origin}")

        spacing = tags.get("ElementSpacing")
        if spacing and len(spacing) != ndims:
            raise ValueError(f"Invalid ElementSpacing tag: {spacing}")

        rotation = tags.get(
            "Rotation", tags.get("Orientation", tags.get("TransformMatrix"))
        )
        if rotation and len(rotation) != ndims ** 2:
            raise ValueError(
                f"Invalid Rotation/Orientation/TransformMatrix tag: {rotation}"
            )

        center = tags.get("CenterOfRotation")
        if center and len(center) != ndims:
            raise ValueError(f"Invalid CenterOfRotation tag: {center}")

    @classmethod
    def save(cls, filename, metaimage, **tags):
        """save metaimage to file"""
        LOGGER.debug("save metaimage")

        # new set of tags
        tags = MetaImageTags(**dict(metaimage.tags, **tags))

        # raw data
        nchan = tags.get("ElementNumberOfChannels", 1)
        if nchan > 1:
            raw_data = bytes(np.moveaxis(metaimage.data, -1, 0).T.data)
        else:
            raw_data = bytes(metaimage.data.T.data)

        # compress
        if tags.get("CompressedData", False):
            # is compressed
            raw_data = zlib.compress(raw_data)
            # raw_data = gzip.compress(raw_data)
            # set compressed data size tag
            tags["CompressedDataSize"] = len(raw_data)

        # set destination from tags or filename (not metaimage!)
        # if not "ElementDataFile" in tags:
        basename, ext = os.path.splitext(filename)
        if not ext:
            filename += ".mha"
            dest = "LOCAL"
        elif ext == ".mha":
            dest = "LOCAL"
        elif ext == ".mhd":
            raw = ".zraw" if tags.get("CompressedData", False) else ".raw"
            dest = os.path.basename(basename) + raw
        tags["ElementDataFile"] = dest

        # write header
        cls.check(metaimage.data, tags)
        MetaImageTags.save(filename, tags)

        # write data
        if dest == "LOCAL":
            with open(filename, "ab") as f:
                # append to header
                f.write(raw_data)
        else:
            dirname = os.path.dirname(filename)
            datafile = os.path.join(dirname, dest)
            with open(datafile, "wb") as f:
                f.write(raw_data)


#
# MetaImage tags


class MetaImageTags(OrderedDict):
    """metaimage tag collection"""

    @classmethod
    def load(cls, filename, ignore_errors=False):
        """load tags from filename"""
        LOGGER.debug("load metaimage tags")
        tags = {}
        with open(filename, "rb") as f:
            while True:
                line = f.readline()

                if not line:
                    break

                line = line.strip()
                if not line:
                    continue

                split = line.split(b"=")
                tag = split[0].strip().decode("utf8")
                value = split[1].strip().decode("utf8")

                # store value
                tags[tag] = value

                if tag == "ElementDataFile":
                    break

        return cls(ignore_errors=ignore_errors, **tags)

    @classmethod
    def save(cls, filename, tags):
        """save tags to file"""
        LOGGER.debug("save metaimage tags")
        assert isinstance(tags, cls)

        with open(filename, "w") as f:
            f.write(tags.to_string())

    @classmethod
    def from_array(cls, array, isvector=None):
        """init Tags from numpy array"""
        tags = {}
        if isvector:
            tags["NDims"] = array.ndim - 1
            tags["DimSize"] = array.shape[:-1]
            # tags["DimSize"] = array.shape[1:]
            tags["ElementNumberOfChannels"] = array.shape[-1]
            # tags["ElementNumberOfChannels"] = array.shape[0]
        else:
            tags["NDims"] = array.ndim
            tags["DimSize"] = array.shape

        try:
            tags["ElementType"] = NUMPY_TO_METAIO_TYPES[str(array.dtype)]
        except KeyError:
            raise TypeError(f"Invalid dtype: {array.dtype}")

        # required geometrical tags
        ndims = tags["NDims"]
        tags["Offset"] = [0] * ndims
        tags["ElementSpacing"] = [1] * ndims
        tags["TransformMatrix"] = [
            1 * (i % (ndims + 1) == 0) for i in range(ndims ** 2)
        ]

        return MetaImageTags(**tags)

    def __init__(self, tags={}, ignore_errors=False, **kwargs):
        """init tag collection"""
        LOGGER.debug("Init metaimage tags")
        if isinstance(tags, MetaImageTags):
            super().update(tags)
            tags = {}
        tags = {**tags, **kwargs}
        self.update(ignore_errors=ignore_errors, **tags)

    def update(self, tags={}, ignore_errors=False, **kwargs):
        """update tag values"""
        LOGGER.debug("update metaimage tags")
        tags = {**tags, **kwargs}

        traits = METAIMAGE_TAG_TRAITS
        for trait in traits:
            # iter on all tags
            for name in trait.names:
                if name in tags:
                    # tag is provided
                    value = tags.pop(name)
                    break
            else:
                # tag not provided
                if trait.name in self:
                    # using pre-existing value
                    value = self[trait.name]
                elif trait.default is not None:
                    # use default value
                    value = trait.default
                elif trait.required and not ignore_errors:
                    # raise
                    raise ValueError("Missing required tag: %s" % trait.name)
                else:
                    # pass
                    continue

            # check and cast value
            value = trait(value)

            # update values
            self[trait.name] = value

        if tags:
            for tag in tags:
                LOGGER.info("Unknown tag: {}".format(tag))

    def to_string(self):
        """return string representation"""
        string = ""
        alltags = [trait.name for trait in METAIMAGE_TAG_TRAITS]
        for tag in alltags:
            value = self.get(tag)
            if value is None:
                continue
            elif not isinstance(value, list):
                value = [value]
            string += "{} = {}\n".format(tag, " ".join([str(v) for v in value]))
        return string


#
# Tag traits


class TagTrait:
    """metaimage Tag trait"""

    def __repr__(self):
        return f"{self.name} (default={self.default})"

    def __init__(
        self,
        name,
        size=None,
        type=None,
        choices=None,
        default=None,
        required=True,
    ):
        names = name if isinstance(name, list) else [name]
        self.name = names[0]
        self.names = names
        self.size = size
        self.type = type
        self.choices = choices
        self.required = required
        if size is not None:
            if size > 0 and default and len(default) != size:
                raise ValueError(f"Invalid default: {default}")
            elif size < 0 and default:
                raise ValueError("Cannot set default with unknown size")
        self.default = default

    def __call__(self, value):
        """check and cast tag's value"""
        if isinstance(value, str):
            while "  " in value:
                # remove multiple white-space
                value = value.replace("  ", " ")
            if " " in value:
                # split into list of values
                value = value.split(" ")

        elif hasattr(value, "__len__"):
            value = list(value)

        if self.size is None:
            if self.size is None and isinstance(value, list):
                raise ValueError(f"Tag {self.name} must be a singleton")

        elif self.size is not None:
            if self.size > 0 and len(value) != self.size:
                raise ValueError(
                    f"Tag {self.name} must be a sequence of length {self.size}"
                )

        values = value
        if not isinstance(values, list):
            values = [value]

        # check type and cast
        if self.type is bool:
            try:
                values = [
                    {True: True, False: False, "True": True, "False": False}[value]
                    for value in values
                ]
            except KeyError:
                raise TypeError(
                    "Invalid boolean value for tag '%s': %s" % (self.name, values)
                )
        elif self.type:
            # check value type
            try:
                values = [self.type(value) for value in values]
            except (ValueError, TypeError):
                raise TypeError(
                    "Invalid value type for tag '%s': %s" % (self.name, values)
                )
        elif self.choices:
            # check values
            if not all(value in self.choices for value in values):
                raise TypeError("Invalid value for tag '%s': %s" % (self.name, values))

        # return
        if self.size is None:
            return values[0]  # singleton
        return values


METAIMAGE_TAG_TRAITS = [
    # base tags
    TagTrait("ObjectType", choices=["Image"], default="Image"),
    TagTrait("NDims", type=int, required=False),
    TagTrait("BinaryData", type=bool, default="True"),
    TagTrait("BinaryDataByteOrderMSB", type=bool, default="False"),
    TagTrait("CompressedData", type=bool, default="False"),
    TagTrait("AnatomicalOrientation", type=str, default="RAI"),
    TagTrait(["Offset", "Position"], type=float, size=-1),
    TagTrait(["TransformMatrix", "Rotation"], type=float, size=-1),
    TagTrait("ElementSpacing", type=float, size=-1),
    # not required
    TagTrait("CenterOfRotation", type=float, size=-1, required=False),
    TagTrait("HeaderSize", type=int, required=False),
    TagTrait("CompressedDataSize", type=int, required=False),
    TagTrait("Comment", type=str, required=False),
    TagTrait("ObjectSubType", type=str, required=False),
    TagTrait("TransformType", type=str, required=False),
    TagTrait("Name", type=str, required=False),
    TagTrait("ID", type=int, required=False),
    TagTrait("ParentID", type=int, required=False),
    TagTrait("Color", type=float, size=4, required=False),
    TagTrait("Modality", type=str, required=False),
    TagTrait("SequenceID", type=int, size=4, required=False),
    TagTrait("ElementByteOrderMSB", type=str, required=False),
    TagTrait("ElementMin", type=float, required=False),
    TagTrait("ElementMax", type=float, required=False),
    TagTrait("ElementNumberOfChannels", type=int, required=False),
    TagTrait("DimSize", type=int, size=-1),
    TagTrait("ElementSize", type=float, required=False),
    # last
    TagTrait("ElementType", type=str, default="MET_USHORT", required=False),
    TagTrait("ElementDataFile", type=str, default="LOCAL", required=False),
]

METAIMAGE_TAG_DICT = {
    name: trait for trait in METAIMAGE_TAG_TRAITS for name in trait.names
}

"""
MetaObject Tags

The tags of MetaObject are:

    Comment
        MET_STRING
        User defined - arbitrary
    ObjectType
        MET_STRING
        Defined by derived objects – e.g., Tube, Image
    ObjectSubType
        MET_STRING
        Defined by derived objects – currently not used
    TransformType
        MET_STRING
        Defined by derived objects – e.g., Rigid
    NDims
        MET_INT
        Defined at object instantiation
    Name
        MET_STRING
        User defined
    ID
        MET_INT
        User defined else -1
    ParentID
        MET_INT
        User defined else -1
    BinaryData
        MET_STRING
        Are the data associated with this object stored at Binary or ASCII
        Defined by derived objects
    ElementByteOrderMSB
        MET_STRING
    BinaryDataByteOrderMSB
        MET_STRING
    Color
        MET_FLOAT_ARRAY[4]
        R, G, B, alpha (opacity)
    Position
        MET_FLOAT_ARRAY[NDims]
        X, Y, Z,… of real-world coordinate of 0,0,0 index of image)
    Orientation/TransformMatrix
        MET_FLOAT_MATRIX[NDims][NDims]
        [0][0],[0][1],[0][2] specify X, Y, Z… direction in real-world of X-axis of image
        [1][0],[1][1],[1][2] specify X, Y, Z… direction in real-world of Y-axis of image, etc.
    AnatomicalOrientation
        MET_STRING
        Specify anatomic ordering of the axis. Use only [R|L] | [A|P] | [S|I] per axis. For example, if the three letter code for (column index, row index, slice index is) ILP, then the origin is at the superior, right, anterior corner of the volume, and therefore the axes run from superior to inferior, from right to left, from anterior to posterior.
    ElementSpacing
        MET_FLOAT_ARRAY[NDims]
        The distance between voxel centers
    DimSize
        MET_INT_ARRAY[NDims]
        Number of elements per axis in data
    HeaderSize
        MET_INT
        Number of Bytes to skip at the head of each data file.
        Specify –1 to have MetaImage calculate the header size based on the assumption that the data occurs at the end of the file.
        Specify 0 if the data occurs at the begining of the file.
    Modality
        MET_STRING
        One of enum type: MET_MOD_CT, MET_MOD_MR, MET_MOD_US… See metaImageTypes.h
    SequenceID
        MET_INT_ARRAY[4]
        Four values comprising a DICOM sequence: Study, Series, Image numbers
    ElementMin
        MET_FLOAT
        Minimum value in the data
    ElementMax
        MET_FLOAT
        Maximum value in the data
    ElementNumberOfChannels
        MET_INT
        Number of values (of type ElementType) per voxel
    ElementSize
        MET_FLOAT_ARRAY[NDims]
        Physical size of each voxel
    ElementType
        MET_STRING
        One of enum type: MET_UCHAR, MET_CHAR… See metaTypes.h
    ElementDataFile
        MET_STRING
        One of the following:
            Name of the file to be loaded
            A printf-style string followed by the min, max, and step values to be used to pass an argument to the string to create list of file names to be loaded (must be (N-1)D blocks of data per file).
            LIST [X] – This specifies that starting on the next line is a list of files (one filename per line) in which the data is stored. Each file (by default) contains an (N-1)D block of data. If a second argument is given, its first character must be a number that specifies the dimension of the data in each file. For example ElementDataFile = LIST 2D means that there will be a 2D block of data per file.
            LOCAL – Indicates that the data begins at the beginning of the next line.

"""
