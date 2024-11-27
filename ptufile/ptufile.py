# ptufile.py

# Copyright (c) 2023-2024, Christoph Gohlke
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

"""Read PicoQuant PTU and related files.

Ptufile is a Python library to read image and metadata from PicoQuant PTU
and related files: PHU, PCK, PCO, PFS, PUS, and PQRES.
PTU files contain time correlated single photon counting (TCSPC)
measurement data and instrumentation parameters.

:Author: `Christoph Gohlke <https://www.cgohlke.com>`_
:License: BSD 3-Clause
:Version: 2024.11.26
:DOI: `10.5281/zenodo.10120021 <https://doi.org/10.5281/zenodo.10120021>`_

Quickstart
----------

Install the ptufile package and all dependencies from the
`Python Package Index <https://pypi.org/project/ptufile/>`_::

    python -m pip install -U "ptufile[all]"

See `Examples`_ for using the programming interface.

Source code and support are available on
`GitHub <https://github.com/cgohlke/ptufile>`_.

Requirements
------------

This revision was tested with the following requirements and dependencies
(other versions may work):

- `CPython <https://www.python.org>`_ 3.10.11, 3.11.9, 3.12.7, 3.13.0 64-bit
- `NumPy <https://pypi.org/project/numpy>`_ 2.1.3
- `Xarray <https://pypi.org/project/xarray>`_ 2024.11.0 (recommended)
- `Matplotlib <https://pypi.org/project/matplotlib/>`_ 3.9.2 (optional)
- `Tifffile <https://pypi.org/project/tifffile/>`_ 2024.9.20 (optional)
- `Numcodecs <https://pypi.org/project/numcodecs/>`_ 0.14.1 (optional)
- `Cython <https://pypi.org/project/cython/>`_ 3.0.11 (build)

Revisions
---------

2024.11.26

- Support bi-directional scanning (FLIMbee scanner).
- Drop support for Python 3.9.

2024.10.10

- Also trim leading channels without photons (breaking).
- Add property to identify channels with photons.

2024.9.14

- Improve typing.

2024.7.13

- Detect point scans in image mode.
- Deprecate Python 3.9, support Python 3.13.

2024.5.24

- Fix docstring examples not correctly rendered on GitHub.

2024.4.24

- Build wheels with NumPy 2.

2024.2.20

- Change definition of PtuFile.frequency (breaking).
- Add option to specify number of bins returned by decode_histogram.
- Add option to return histograms of one period.

2024.2.15

- Add PtuFile.scanner property.
- Add numcodecs compatible PTU codec.

2024.2.8

- Support sinusoidal scanning correction.

2024.2.2

- Change positive dtime parameter from index to size (breaking).
- Fix segfault with ImgHdr_TimePerPixel = 0.
- Rename MultiHarp to Generic conforming with changes in PicoQuant reference.

2023.11.16

- Fix empty line when first record is start marker.

2023.11.13

- Change image histogram dimension order to TYXCH (breaking).
- Change frame start to start of first line in frame (breaking).
- Improve trimming of incomplete frames (breaking).
- Remove trim_dtime option (breaking).
- Fix selection handling in PtuFile.decode_image.
- Add option to trim T, C, and H axes of image histograms.
- Add option to decode histograms to memory-mapped or user-provided arrays.
- Add ``__getitem__`` interface to image histogram.

2023.11.1

- Initial alpha release.

Notes
-----

`PicoQuant GmbH <https://www.picoquant.com/>`_ is a manufacturer of photonic
components and instruments.

The PicoQuant unified file formats are documented at the
`PicoQuant-Time-Tagged-File-Format-Demos
<https://github.com/PicoQuant/PicoQuant-Time-Tagged-File-Format-Demos/tree/master/doc>`_.

The following features are currently not implemented: PT2 and PT3 files,
decoding images from T2 formats, bidirectional sinusoidal scanning, and
deprecated image reconstruction. Line-scanning is not tested.

Other modules for reading or writing PicoQuant files are:

- `Read_PTU.py
  <https://github.com/PicoQuant/PicoQuant-Time-Tagged-File-Format-Demos/blob/master/PTU/Python/Read_PTU.py>`_
- `readPTU_FLIM <https://github.com/SumeetRohilla/readPTU_FLIM>`_
- `fastFLIM <https://github.com/RobertMolenaar-UT/fastFLIM>`_
- `PyPTU <https://gitlab.inria.fr/jrye/pyptu>`_
- `PTU_Reader <https://github.com/UU-cellbiology/PTU_Reader>`_
- `PTU_Writer <https://github.com/ekatrukha/PTU_Writer>`_
- `FlimReader <https://github.com/flimfit/FlimReader>`_
- `tttrlib <https://github.com/Fluorescence-Tools/tttrlib>`_
- `picoquantio <https://github.com/tsbischof/picoquantio>`_
- `ptuparser <https://pypi.org/project/ptuparser/>`_
- `phconvert <https://github.com/Photon-HDF5/phconvert/>`_
- `trattoria <https://pypi.org/project/trattoria/>`_
  (wrapper of `trattoria-core <https://pypi.org/project/trattoria-core/>`_ and
  `tttr-toolbox <https://github.com/GCBallesteros/tttr-toolbox/>`_)
- `napari-flim-phasor-plotter
  <https://github.com/zoccoler/napari-flim-phasor-plotter/blob/0.0.6/src/napari_flim_phasor_plotter/_io/readPTU_FLIM.py>`_

Examples
--------

Read properties and tags from any type of PicoQuant unified tagged file:

>>> pq = PqFile('tests/Settings.pfs')
>>> pq.magic
<PqFileMagic.PFS: ...>
>>> pq.guid
UUID('86d428e2-cb0b-4964-996c-04456ba6be7b')
>>> pq.tags
{...'CreatorSW_Name': 'SymPhoTime 64', 'CreatorSW_Version': '2.1'...}
>>> pq.close()

Read metadata from a PicoQuant PTU FLIM file:

>>> ptu = PtuFile('tests/FLIM.ptu')
>>> ptu.magic
<PqFileMagic.PTU: ...>
>>> ptu.type
<PtuRecordType.PicoHarpT3: 66307>
>>> ptu.measurement_mode
<PtuMeasurementMode.T3: 3>
>>> ptu.measurement_submode
<PtuMeasurementSubMode.IMAGE: 3>

Decode TTTR records from the PTU file to ``numpy.recarray``:

>>> decoded = ptu.decode_records()

Get global times of frame changes from markers:

>>> decoded['time'][(decoded['marker'] & ptu.frame_change_mask) > 0]
array([1571185680], dtype=uint64)

Decode TTTR records to overall delay-time histograms per channel:

>>> ptu.decode_histogram(dtype='uint8')
array([[ 5,  7,  7, ..., 10,  9,  2]], dtype=uint8)

Get information about the FLIM image histogram in the PTU file:

>>> ptu.shape
(1, 256, 256, 2, 3126)
>>> ptu.dims
('T', 'Y', 'X', 'C', 'H')
>>> ptu.coords
{'T': ..., 'Y': ..., 'X': ..., 'H': ...}
>>> ptu.dtype
dtype('uint16')
>>> ptu.active_channels
(0, 1)

Decode parts of the image histogram to ``numpy.ndarray`` using slice notation.
Slice step sizes define binning, -1 being used to integrate along axis:

>>> ptu[:, ..., 0, ::-1]
array([[[103, ..., 38],
              ...
        [ 47, ..., 30]]], dtype=uint16)

Alternatively, decode the first channel and integrate all histogram bins
to a ``xarray.DataArray``, keeping reduced axes:

>>> ptu.decode_image(channel=0, dtime=-1, asxarray=True)
<xarray.DataArray (T: 1, Y: 256, X: 256, C: 1, H: 1)> ...
array([[[[[103]],
           ...
         [[ 30]]]]], dtype=uint16)
Coordinates:
  * T        (T) float64... 0.05625
  * Y        (Y) float64... -0.0001304 ... 0.0001294
  * X        (X) float64... -0.0001304 ... 0.0001294
  * C        (C) uint8... 0
  * H        (H) float64... 0.0
Attributes...
    frequency:      19999200.0
...
>>> ptu.close()

Preview the image and metadata in a PTU file from the console::

    python -m ptufile tests/FLIM.ptu

"""

from __future__ import annotations

__version__ = '2024.11.26'

__all__ = [
    '__version__',
    'imread',
    'logger',
    'PqFile',
    'PqFileError',
    'PqFileMagic',
    'PhuFile',
    'PtuFile',
    'PtuRecordType',
    'PtuScannerType',
    'PtuScanDirection',
    'PtuMeasurementMode',
    'PtuMeasurementSubMode',
    'PhuMeasurementMode',
    'PhuMeasurementSubMode',
    'T2_RECORD_DTYPE',
    'T3_RECORD_DTYPE',
]

import dataclasses
import enum
import logging
import math
import os
import struct
import sys
import time
import uuid
from functools import cached_property
from typing import TYPE_CHECKING, final, overload

if TYPE_CHECKING:
    from collections.abc import Sequence
    from types import EllipsisType
    from typing import IO, Any, Literal

    from numpy.typing import DTypeLike, NDArray
    from xarray import DataArray

    Dimension = Literal['T', 'C', 'H']
    OutputType = str | IO[bytes] | NDArray[Any] | None


import numpy


@overload
def imread(
    file: str | os.PathLike[str] | IO[bytes],
    /,
    selection: Sequence[int | slice | EllipsisType | None] | None = None,
    *,
    dtype: DTypeLike | None = None,
    channel: int | None = None,
    frame: int | None = None,
    dtime: int | None = None,
    bishift: int | None = None,
    trimdims: Sequence[Dimension] | str | None = None,
    keepdims: bool = True,
    asxarray: Literal[False] = ...,
    out: OutputType = None,
) -> NDArray[Any]: ...


@overload
def imread(
    file: str | os.PathLike[str] | IO[bytes],
    /,
    selection: Sequence[int | slice | EllipsisType | None] | None = None,
    *,
    dtype: DTypeLike | None = None,
    channel: int | None = None,
    frame: int | None = None,
    dtime: int | None = None,
    bishift: int | None = None,
    trimdims: Sequence[Dimension] | str | None = None,
    keepdims: bool = True,
    asxarray: Literal[True] = ...,
    out: OutputType = None,
) -> DataArray: ...


@overload
def imread(
    file: str | os.PathLike[str] | IO[bytes],
    /,
    selection: Sequence[int | slice | EllipsisType | None] | None = None,
    *,
    dtype: DTypeLike | None = None,
    channel: int | None = None,
    frame: int | None = None,
    dtime: int | None = None,
    bishift: int | None = None,
    trimdims: Sequence[Dimension] | str | None = None,
    keepdims: bool = True,
    asxarray: bool = False,
    out: OutputType = None,
) -> NDArray[Any] | DataArray: ...


def imread(
    file: str | os.PathLike[str] | IO[bytes],
    /,
    selection: Sequence[int | slice | EllipsisType | None] | None = None,
    *,
    dtype: DTypeLike | None = None,
    channel: int | None = None,
    frame: int | None = None,
    dtime: int | None = None,
    bishift: int | None = None,
    trimdims: Sequence[Dimension] | str | None = None,
    keepdims: bool = True,
    asxarray: bool = False,
    out: OutputType = None,
) -> NDArray[Any] | DataArray:
    """Return decoded image histogram from T3 mode PTU file.

    Parameters:
        file:
            File name or seekable binary stream.
        selection, dtype, channel, frame, dtime, bishift, keepdims, asxarray,\
        out:
            Passed to :py:meth:`PtuFile.decode_image`.
        trimdims:
            Passed to :py:class:`PtuFile`.

    """
    with PtuFile(file, trimdims=trimdims) as ptu:
        data = ptu.decode_image(
            selection,
            dtype=dtype,
            channel=channel,
            frame=frame,
            dtime=dtime,
            bishift=bishift,
            keepdims=keepdims,
            asxarray=asxarray,
            out=out,
        )
    return data


class PqFileError(Exception):
    """Exception to indicate invalid PicoQuant tagged file structure."""


class PqFileMagic(enum.Enum):
    """PicoQuant file type identifiers."""

    PTU = b'PQTTTR\0\0'
    """TTTR file, PTU, contains raw data in unified TTTR-format."""

    PHU = b'PQHISTO\0'
    """Histogram file, PHU, contains TCSPC histograms."""

    PCK = b'PQCHECK\0'
    """Internal file, PCK, contains post-acquisition analysis results."""

    PCO = b'PQCOMNT\0'
    """Comment file, PCO, contains manually entered text."""

    PFS = b'PQDEFLT\0'
    """Settings file, PFS or PUS, contains factory or user setting defaults."""

    PQRES = b'PQRESLT\0'
    """Result file, PQRES, contains analysis generated during measurement."""


class PhuMeasurementMode(enum.IntEnum):
    """Kind of TCSPC measurement."""

    UNKNOWN = -1
    """Unknown mode."""

    HISTOGRAM = 0
    """Histogram mode."""

    CONTI = 8
    """Conti mode."""

    @classmethod
    def _missing_(cls, value: object) -> object:
        if not isinstance(value, int):
            return None
        obj = cls(-1)  # Unknown
        obj._value_ = value
        return obj


class PhuMeasurementSubMode(enum.IntEnum):
    """Kind of measurement."""

    UNKNOWN = -1
    """Unknown mode."""

    OSCILLOSCOPE = 0
    """Oscilloscope mode."""

    INTEGRATING = 1
    """Integrating mode."""

    TRES = 2
    """Time-Resolved Emission Spectra mode."""

    SEQ = 3
    """Sequence mode."""

    @classmethod
    def _missing_(cls, value: object) -> object:
        if not isinstance(value, int):
            return None
        obj = cls(-1)
        obj._value_ = value
        return obj


class PtuMeasurementMode(enum.IntEnum):
    """Kind of TCSPC Measurement."""

    UNKNOWN = -1
    """Unknown mode."""

    T2 = 2
    """T2 mode."""

    T3 = 3
    """T3 mode."""

    @classmethod
    def _missing_(cls, value: object) -> object:
        if not isinstance(value, int):
            return None
        obj = cls(-1)  # Unknown
        obj._value_ = value
        return obj


class PtuMeasurementSubMode(enum.IntEnum):
    """Kind of measurement."""

    UNKNOWN = -1
    """Unknown mode."""

    POINT = 1
    """Point scan mode."""

    LINE = 2
    """Line scan mode."""

    IMAGE = 3
    """Image scan mode."""

    @classmethod
    def _missing_(cls, value: object) -> object:
        if not isinstance(value, int):
            return None
        if value == 0:
            obj = cls(1)  # Point
        else:
            obj = cls(-1)
        obj._value_ = value
        return obj


class PtuScannerType(enum.IntEnum):
    """Scanner hardware."""

    UNKNOWN = -1
    """Unknown scanner."""

    PI_E710 = 1
    """PI E-710 scanner."""

    LSM = 3
    """PicoQuant LSM scanner."""

    PI_LINEWBS = 5
    """PI Line WB scanner."""

    PI_E725 = 6
    """PI E-725 scanner."""

    PI_E727 = 7
    """PI E-727 scanner."""

    MCL = 8
    """MCL scanner."""

    FLIMBEE = 9
    """PicoQuant FLIMBee scanner."""

    SCANBOX = 10
    """Zeiss ScanBox scanner."""

    @classmethod
    def _missing_(cls, value: object) -> object:
        if not isinstance(value, int):
            return None
        obj = cls(-1)  # Unknown
        obj._value_ = value
        return obj


class PtuScanDirection(enum.IntEnum):
    """Scan direction, defining configuration of fast and slowscan axes."""

    XY = 0
    XZ = 1
    YZ = 2

    @classmethod
    def _missing_(cls, value: object) -> object:
        if not isinstance(value, int) or value != 0:
            return None
        obj = cls(0)  # XY
        obj._value_ = value
        return obj


class PqTagType(enum.IntEnum):
    """Tag type definition."""

    Empty8 = 0xFFFF0008
    Bool8 = 0x00000008
    Int8 = 0x10000008
    BitSet64 = 0x11000008
    Color8 = 0x12000008
    Float8 = 0x20000008
    TDateTime = 0x21000008
    Float8Array = 0x2001FFFF
    AnsiString = 0x4001FFFF
    WideString = 0x4002FFFF
    BinaryBlob = 0xFFFFFFFF


class PtuRecordType(enum.IntEnum):
    """TTTR record format."""

    PicoHarpT3 = 0x00010303  # Picoharp300T3
    PicoHarpT2 = 0x00010203  # Picoharp300T2
    HydraHarpT3 = 0x00010304
    HydraHarpT2 = 0x00010204
    HydraHarp2T3 = 0x01010304
    HydraHarp2T2 = 0x01010204
    TimeHarp260NT3 = 0x00010305
    TimeHarp260NT2 = 0x00010205
    TimeHarp260PT3 = 0x00010306
    TimeHarp260PT2 = 0x00010206
    GenericT2 = 0x00010207  # MultiHarpT2 and Picoharp330T2
    GenericT3 = 0x00010307  # MultiHarpT3 and Picoharp330T3


T2_RECORD_DTYPE = numpy.dtype(
    [
        ('time', numpy.uint64),
        ('channel', numpy.int8),
        ('marker', numpy.uint8),
    ]
)
"""Numpy dtype of decoded T2 records."""

T3_RECORD_DTYPE = numpy.dtype(
    [
        ('time', numpy.uint64),
        ('dtime', numpy.int16),
        ('channel', numpy.int8),
        ('marker', numpy.uint8),
    ]
)
"""Numpy dtype of decoded T3 records."""

FILE_EXTENSIONS = {
    '.ptu': PqFileMagic.PTU,
    '.phu': PqFileMagic.PHU,
    '.pck': PqFileMagic.PCK,
    '.pco': PqFileMagic.PCO,
    '.pfs': PqFileMagic.PFS,
    '.pus': PqFileMagic.PFS,
    '.pqres': PqFileMagic.PQRES,
}
"""File extensions of PicoQuant tagged files."""


class PqFile:
    """PicoQuant unified tagged file.

    PTU, PHU, PCK, PCO, PFS, PUS, and PQRES files contain measurement
    metadata and settings encoded as unified tags.

    ``PqFile`` and subclass instances are not thread safe.
    All attributes are read-only.

    ``PqFile`` and subclass instances must be closed with
    :py:meth:`PqFile.close`, which is automatically called when using the
    'with' context manager.

    Parameters:
        file:
            File name or seekable binary stream.
        fastload:
            If true, only read tags marked for fast loading,
            else read all tags.

    Raises:
        PqFileError: File is not a PicoQuant tagged file or is corrupted.

    """

    filename: str
    """Name of file or empty if binary stream."""

    magic: PqFileMagic
    """PicoQuant file type identifier."""

    version: str
    """File version."""

    tags: dict[str, Any]
    """PicoQuant unified tags."""

    _fh: IO[bytes]
    _close: bool  # file needs to be closed
    _data_offset: int  # position of raw data in file

    _MAGIC: set[PqFileMagic] = set(PqFileMagic)
    _STR_: tuple[str, ...] = ('magic', 'version')  # attributes listed first

    def __init__(
        self,
        file: str | os.PathLike[str] | IO[bytes],
        /,
        *,
        fastload: bool = False,
    ) -> None:
        self.version = ''
        self.tags = {}

        if isinstance(file, (str, os.PathLike)):
            self.filename = os.fspath(file)
            self._close = True
            self._fh = open(file, 'rb')
        elif hasattr(file, 'seek'):
            self.filename = ''
            self._close = False
            self._fh = file
        else:
            raise ValueError(f'cannot open file of type {type(file)}')

        fh = self._fh
        magic = fh.read(8)
        try:
            self.magic = PqFileMagic(magic)
            if self.magic not in self._MAGIC:
                raise ValueError(f'{self.magic} not in {self._MAGIC!r}')
        except Exception as exc:
            self.close()
            raise PqFileError(
                f'{self.filename!r} not a {self.__class__.__name__} {magic=!r}'
            ) from exc

        self.version = fh.read(8).strip(b'\0').decode()
        tags = self.tags

        def errmsg(
            msg: str, tagid: str, index: int, typecode: int, value: Any
        ) -> str:
            return (
                f'{msg} @ {self.name!r} '
                f'{tagid=}, {index=}, {typecode=}, {value=!r}'
            )[:80]

        tagid: str
        index: int
        typecode: int
        value: Any
        ty = PqTagType
        unpack = struct.unpack
        try:
            while True:
                # offset = fh.tell()
                tagid_, index, typecode, value = unpack(
                    '<32siI8s', fh.read(48)
                )
                # print(tagid.strip(b'\0'), index, typecode, value)
                tagid = tagid_.rstrip(b'\0').decode('ascii', errors='ignore')
                # disabled: too many errors in PQRES
                # if offset % 8:
                #     logger().error(
                #         errmsg(
                #             f'tag {offset=} not divisible by 8',
                #             tagid,
                #             index,
                #             typecode,
                #             value,
                #         )
                #     )
                if tagid == 'Header_End':
                    break
                if tagid == 'Fast_Load_End':
                    if fastload:
                        break
                    continue
                if typecode == ty.Empty8:
                    value = None
                elif typecode == ty.Bool8:
                    value = bool(unpack('<q', value)[0])
                elif typecode == ty.Int8:
                    value = unpack('<q', value)[0]
                elif typecode == ty.BitSet64:
                    value = unpack('<q', value)[0]
                elif typecode == ty.Color8:
                    # TODO: unpack to RGB triple
                    value = unpack('<q', value)[0]
                elif typecode == ty.Float8:
                    value = unpack('<d', value)[0]
                elif typecode == ty.TDateTime:
                    value = unpack('<d', value)[0]
                    value = time.gmtime(int((float(value) - 25569) * 86400))
                elif typecode == ty.Float8Array:
                    size = unpack('<q', value)[0]
                    value = unpack(f'<{size // 8}d', fh.read(size))
                elif typecode == ty.AnsiString:
                    size = unpack('<q', value)[0]
                    value = (
                        fh.read(size)
                        .rstrip(b'\0')
                        .decode('windows-1252', errors='ignore')
                    )
                elif typecode == ty.WideString:
                    size = unpack('<q', value)[0]
                    value = fh.read(size).decode('utf-16-le').rstrip('\0')
                elif typecode == ty.BinaryBlob:
                    size = unpack('<q', value)[0]
                    value = fh.read(size)
                    if tagid == 'ChkHistogram':
                        value = numpy.frombuffer(value, dtype=numpy.int64)
                else:
                    logger().error(
                        errmsg(
                            'invalid tag type', tagid, index, typecode, value
                        )
                    )
                    break

                # Although tagids can appear multiple times, later tags have
                # always found to either contain the same or more detailed
                # values. Hence a dict interface seems sufficient.
                if index < 0:
                    if tagid in tags and tags[tagid] != value:
                        logger().warning(
                            errmsg(
                                'duplicate tag', tagid, index, typecode, value
                            )
                            + f' != {tags[tagid]!r}'[:16]
                        )
                    tags[tagid] = value
                elif index == 0:
                    if tagid in tags and tags[tagid][0] != value:
                        logger().warning(
                            errmsg(
                                'duplicate tag', tagid, index, typecode, value
                            )
                            + f' != {tags[tagid][0]!r}'[:16]
                        )
                    tags[tagid] = [value]
                elif tagid not in tags:
                    logger().error(
                        errmsg(
                            'tag index out of order',
                            tagid,
                            index,
                            typecode,
                            value,
                        )
                    )
                    tags[tagid] = [value]
                elif index != len(tags[tagid]):
                    logger().error(
                        errmsg(
                            'tag index out of order',
                            tagid,
                            index,
                            typecode,
                            value,
                        )
                    )
                    tags[tagid].append(value)
                else:
                    tags[tagid].append(value)
        except Exception as exc:
            self.close()
            raise PqFileError(
                errmsg('tag corrupted', tagid, index, typecode, value)
            ) from exc
        self._data_offset = self._fh.tell()

    @property
    def name(self) -> str:
        """Name of file."""
        if self.filename:
            return os.path.basename(self.filename)
        if self._fh.name:
            return self._fh.name
        return repr(self._fh)

    @property
    def guid(self) -> uuid.UUID:
        """Global identifier of file."""
        return uuid.UUID(self.tags['File_GUID'])

    def close(self) -> None:
        """Close file handle and free resources."""
        if self._close:
            try:
                self._fh.close()
            except Exception:
                pass

    def __enter__(self) -> PqFile:
        return self

    def __exit__(  # type: ignore[no-untyped-def]
        self, exc_type, exc_value, traceback
    ) -> None:
        self.close()

    def __repr__(self) -> str:
        return f'<{self.__class__.__name__} {self.name!r}>'

    def __str__(self) -> str:
        return indent(
            repr(self),
            *(
                f'{name}: {getattr(self, name)!r}'[:160]
                for name in self._STR_
                if getattr(self, name) is not None
            ),
            *(
                f'{name}: {getattr(self, name)!r}'[:160]
                for name in dir(self)
                if not (
                    name in self._STR_
                    or name in {'tags', 'name'}
                    or name.startswith('_')
                    or callable(getattr(self, name))
                )
            ),
            indent(
                'tags:',
                *(
                    f'{key}: {value!r}'[:160]
                    for key, value in self.tags.items()
                ),
            ),
        )


@final
class PhuFile(PqFile):
    """PicoQuant histogram file.

    PHU files contain a series of TCSPC histograms in addition to unified tags.

    ``PhuFile`` instances are derived from :py:class:`PqFile`.

    Parameters:
        file:
            File name or seekable binary stream.

    Raises:
        PqFileError: File is not a PicoQuant PHU file or is corrupted.

    """

    _MAGIC: set[PqFileMagic] = {PqFileMagic.PHU}
    _STR_ = ('magic', 'version', 'measurement_mode', 'measurement_submode')

    def __init__(self, file: str | os.PathLike[str] | IO[bytes], /) -> None:
        super().__init__(file)

    def __enter__(self) -> PhuFile:
        return self

    @property
    def measurement_mode(self) -> PhuMeasurementMode:
        """Kind of measurement: HISTOGRAM or CONTI."""
        return PhuMeasurementMode(self.tags['Measurement_Mode'])

    @property
    def measurement_submode(self) -> PhuMeasurementSubMode:
        """Sub-kind of measurement: OSCILLOSCOPE, INTEGRATING, or TRES."""
        return PhuMeasurementSubMode(self.tags['Measurement_SubMode'])

    @property
    def tcspc_resolution(self) -> float:
        """Resolution of TCSPC in s (BaseResolution * iBinningFactor)."""
        return float(self.tags.get('MeasDesc_Resolution', 0.0))

    @property
    def histogram_resolutions(self) -> tuple[float, ...] | None:
        """Base resolution for each histogram."""
        res = self.tags.get('HistResDscr_HWBaseResolution')
        return tuple(res) if res is not None else None

    @property
    def number_histograms(self) -> int:
        """Number of histograms stored in file."""
        return int(self.tags['HistoResult_NumberOfCurves'])

    @overload
    def histograms(
        self,
        index: int | slice | None = None,
        /,
        asxarray: Literal[False] = ...,
    ) -> tuple[NDArray[numpy.uint32], ...]: ...

    @overload
    def histograms(
        self,
        index: int | slice | None = None,
        /,
        asxarray: Literal[True] = ...,
    ) -> tuple[DataArray, ...]: ...

    @overload
    def histograms(
        self,
        index: int | slice | None = None,
        /,
        asxarray: bool = ...,
    ) -> tuple[NDArray[numpy.uint32] | DataArray, ...]: ...

    def histograms(
        self, index: int | slice | None = None, /, asxarray: bool = False
    ) -> tuple[NDArray[numpy.uint32] | DataArray, ...]:
        """Return sequences of histograms from file.

        Parameters:
            index:
                Index of histogram(s) to return.
                By default, all histograms are returned.
            asxarray:
                If true, return histograms as ``xarray.DataArray``,
                else ``numpy.ndarray`` (default).

        """
        if index is None:
            index = slice(None)
        elif isinstance(index, int):
            index = slice(index, index + 1)
        ncurves = self.number_histograms
        if len(self.tags['HistResDscr_DataOffset']) != ncurves:
            raise ValueError('invalid HistResDscr_DataOffset tag')
        if len(self.tags['HistResDscr_HistogramBins']) != ncurves:
            raise ValueError('invalid HistResDscr_HistogramBins tag')

        histograms: list[NDArray[numpy.uint32] | DataArray] = []
        resolution = []
        for offset, nbins, res in zip(
            self.tags['HistResDscr_DataOffset'][index],
            self.tags['HistResDscr_HistogramBins'][index],
            self.tags['HistResDscr_HWBaseResolution'][index],
        ):
            self._fh.seek(offset)
            histograms.append(
                numpy.fromfile(self._fh, dtype='<u4', count=nbins)
            )
            resolution.append(res)
        if asxarray:
            from xarray import DataArray

            histograms = [
                DataArray(
                    h,
                    dims=('H',),
                    coords={
                        # TODO: is this correct?
                        'H': numpy.linspace(
                            0, h.size * r, h.size, endpoint=False
                        )
                    },
                    # name=self.name,
                )
                for h, r in zip(histograms, resolution)
            ]
        # TODO: do not return tuple if index is integer?
        return tuple(histograms)

    def plot(self, *, verbose: bool = False, show: bool = True) -> None:
        """Plot histograms using matplotlib.

        Parameters:
            verbose:
                Print information about histogram arrays.
            show:
                If true (default), display all figures.
                Else, defer to user or environment to display figures.

        """
        from matplotlib import pyplot
        from tifffile import Timer

        t = Timer()
        histograms = self.histograms(asxarray=True)
        if verbose:
            t.print('decode histograms')
            print()
            for hist in histograms:
                print(hist)

        for i, hist in enumerate(histograms):
            y = numpy.trim_zeros(hist.values, trim='b')
            x = hist.coords['H'].values[: y.size]
            pyplot.plot(x, y, label=f'ch {i}')
        pyplot.title(repr(self))
        pyplot.xlabel('delay time [s]')
        pyplot.ylabel('photon count')
        pyplot.legend()
        if show:
            pyplot.show()


@final
class PtuFile(PqFile):
    """PicoQuant time-tagged time-resolved (TTTR) file.

    PTU files contain TTTR records in addition to unified tags.

    ``PtuFile`` is derived from :py:class:`PqFile`.

    Parameters:
        file:
            File name or seekable binary stream.
        trimdims:
            Axes to trim. The default is ``'TCH'``:

            - ``'T'``: remove incomplete first or last frame.
            - ``'C'``: remove leading and trailing channels without photons.
              Else use record type's default :py:attr:`number_channels_max`.
            - ``'H'``: remove trailing delay-time bins without photons.
              Else use record type's default :py:attr:`number_bins_max`.

    Raises:
        PqFileError: File is not a PicoQuant PTU file or is corrupted.

    """

    _trimdims: set[str]
    _asxarray: bool
    _dtype: numpy.dtype[Any]

    _MAGIC: set[PqFileMagic] = {PqFileMagic.PTU}
    _STR_ = (
        'magic',
        'version',
        'type',
        'measurement_mode',
        'measurement_submode',
        'scanner',
        'shape',
        'dims',
        'coords',
    )

    def __init__(
        self,
        file: str | os.PathLike[str] | IO[bytes],
        /,
        *,
        trimdims: Sequence[Dimension] | str | None = None,
    ) -> None:
        super().__init__(file)
        if trimdims is None:
            self._trimdims = {'T', 'C', 'H'}
        else:
            self._trimdims = {ax.upper() for ax in trimdims}
        self._dtype = numpy.dtype(numpy.uint16)
        self._asxarray = False

    def __enter__(self) -> PtuFile:
        return self

    def __getitem__(self, key: Any, /) -> NDArray[Any] | DataArray:
        return self.decode_image(key, keepdims=False)

    @property
    def type(self) -> PtuRecordType:
        """Type of TTTR records.

        Defines the TCSPC device and type of measurement that produced the
        records.

        """
        return PtuRecordType(self.tags['TTResultFormat_TTTRRecType'])

    @property
    def measurement_mode(self) -> PtuMeasurementMode:
        """Kind of TCSPC measurement: T2 or T3."""
        return PtuMeasurementMode(self.tags['Measurement_Mode'])

    @property
    def measurement_submode(self) -> PtuMeasurementSubMode:
        """Sub-kind of measurement: Point, line, or image scan."""
        return PtuMeasurementSubMode(self.tags['Measurement_SubMode'])

    @property
    def measurement_ndim(self) -> int:
        """Dimensionality of measurement."""
        # Measurement_SubMode is not always correct
        submode = self.tags['Measurement_SubMode']
        if (
            submode == 3
            and self.tags.get('ImgHdr_Dimensions', 3) == 3  # optional
            and self.tags.get('ImgHdr_PixY', 1) > 1  # may be missing
        ):
            return 3
        if (
            submode == 2
            and self.tags.get('ImgHdr_Dimensions', 2) == 2  # optional
            and self.tags.get('ImgHdr_PixX', 1) > 1  # may be missing
        ):
            # TODO: need linescan test file
            return 2
        return 1

    @property
    def scanner(self) -> PtuScannerType | None:
        """Scanner hardware, or None if not specified."""
        if 'ImgHdr_Ident' in self.tags:
            return PtuScannerType(self.tags['ImgHdr_Ident'])
        return None

    @property
    def global_resolution(self) -> float:
        """Resolution of time tags in s."""
        return float(self.tags['MeasDesc_GlobalResolution'])

    @property
    def tcspc_resolution(self) -> float:
        """Resolution of TCSPC in s (BaseResolution * iBinningFactor)."""
        return float(self.tags.get('MeasDesc_Resolution', 0.0))

    @property
    def number_records(self) -> int:
        """Number of TTTR records."""
        return int(self.tags['TTResult_NumberOfRecords'])

    @property
    def number_photons(self) -> int:
        """Number of photons counted."""
        return self._info.photons

    @property
    def number_markers(self) -> int:
        """Number of marker events."""
        return self._info.markers

    @property
    def number_images(self) -> int:
        """Number of images separated by frame change markers."""
        return self._info.frames

    @property
    def number_lines(self) -> int:
        """Number of lines marker pairs."""
        return self._info.lines

    @property
    def number_channels_max(self) -> int:
        """Maximum number of channels for record type."""
        return self._info.channels

    @property
    def number_channels(self) -> int:
        """Number of channels, without leading and trailing empty channels."""
        return (
            1
            + self._info.channels_active_last
            - self._info.channels_active_first
        )

    @property
    def active_channels(self) -> tuple[int, ...]:
        """Indices of un-trimmed channels containing photons."""
        channels_active = self._info.channels_active
        return tuple(
            ch
            for ch in range(self._info.channels)
            if channels_active & (1 << ch)
        )

    @property
    def number_bins_max(self) -> int:
        """Maximum delay time for record type."""
        return self._info.bins

    @property
    def number_bins(self) -> int:
        """Highest delay time with photons. Not available for T2 records."""
        return self._info.bins_used

    @property
    def number_bins_in_period(self) -> int:
        """Delay time in one period. Not available for T2 records.

        Same as ``global_resolution / tcspc_resolution``

        """
        nbins = int(math.floor(self.global_resolution / self.tcspc_resolution))
        return max(nbins, 1)

    @property
    def line_start_mask(self) -> int:
        """Marker mask defining line start, or 0 if not defined."""
        value = self.tags.get('ImgHdr_LineStart', None)
        return int(2 ** (value - 1) if value is not None else 0)

    @property
    def line_stop_mask(self) -> int:
        """Marker mask defining line end, or 0 if not defined."""
        value = self.tags.get('ImgHdr_LineStop', None)
        return int(2 ** (value - 1) if value is not None else 0)

    @property
    def frame_change_mask(self) -> int:
        """Marker mask defining image frame change, or 0 if not defined."""
        value = self.tags.get('ImgHdr_Frame', None)
        return int(2 ** (value - 1) if value is not None else 0)

    @property
    def global_pixel_time(self) -> int:
        """Global time per pixel.

        Multiply with global resolution to get time in s.

        """
        if self.tags.get('ImgHdr_TimePerPixel', 0.0) > 0.0:
            pixeltime = (
                float(self.tags['ImgHdr_TimePerPixel'])
                / float(self.tags['MeasDesc_GlobalResolution'])
                / 1e3
            )
        elif self._info.lines > 0:
            pixeltime = self._info.line_time / self.pixels_in_line
        else:
            pixeltime = 1e-3 / float(self.tags['MeasDesc_GlobalResolution'])
        return int(round(pixeltime))

    @property
    def global_line_time(self) -> int:
        """Global time per line, excluding retrace.

        Might be approximate. Multiply with global resolution to get time in s.

        """
        if 'ImgHdr_TimePerPixel' in self.tags:
            linetime = self.pixels_in_line * self.global_pixel_time
        elif self._info.lines > 0:
            linetime = self._info.line_time
        else:
            # point scan: line of one pixel
            linetime = 1e-3 / self.tags['MeasDesc_GlobalResolution']
        return int(round(linetime))

    @property
    def global_frame_time(self) -> int:
        """Global time per image, line, or point scan cycle, excluding retrace.

        Multiply with global resolution to get time in s.

        """
        if self.tags['Measurement_SubMode'] == 3:
            # image, including retrace
            if self._info.frames == 0:
                return self._info.acquisition_time
            else:
                return self._info.acquisition_time // self._info.frames
        if self._info.lines > 0:
            # line scan
            return self._info.line_time
        # point scan
        return self.pixels_in_frame * self.global_pixel_time

    @property
    def global_acquisition_time(self) -> int:
        """Global time of acquisition."""
        # MeasDesc_AcquisitionTime not reliable
        # aqt = self.tags.get('MeasDesc_AcquisitionTime', 0.0)
        # if aqt > 0:
        #     return int(round(1e-3 * aqt / self.global_resolution))
        return self._info.acquisition_time

    @property
    def pixels_in_frame(self) -> int:
        """Number of pixels in one scan cycle."""
        return self.lines_in_frame * self.pixels_in_line

    @property
    def pixels_in_line(self) -> int:
        """Number of pixels in line."""
        ndim = self.measurement_ndim
        if ndim == 3:
            # image
            pixels = self.tags['ImgHdr_PixX']
        elif ndim == 2:
            # line scan
            pixels = round(
                1e-3
                / (
                    self.tags['ImgHdr_TimePerPixel']
                    * self.tags['ImgHdr_LineFrequency']
                )
            )
        else:
            pixels = 1
        return max(1, int(pixels))

    @property
    def lines_in_frame(self) -> int:
        """Number of lines in frame."""
        if self.measurement_ndim == 3:
            return max(1, int(self.tags['ImgHdr_PixY']))
        return 1

    @property
    def pixel_time(self) -> float:
        """Time per pixel in s."""
        return self.global_pixel_time * self.global_resolution

    @property
    def line_time(self) -> float:
        """Average time between line markers or ``pixel_time`` in s."""
        return self.global_line_time * self.global_resolution

    @property
    def frame_time(self) -> float:
        """Time per image, line, or point scan cycle in s.

        Image scan times include retrace.

        """
        return self.global_frame_time * self.global_resolution

    @property
    def acquisition_time(self) -> float:
        """Duration of acquisition in s."""
        # MeasDesc_AcquisitionTime not reliable
        # if 'MeasDesc_AcquisitionTime' in self.tags:
        #     return self.tags['MeasDesc_AcquisitionTime'] * 1e-3
        return self._info.acquisition_time * self.global_resolution

    @property
    def frequency(self) -> float:
        """Repetition frequency in Hz.

        The inverse of :py:attr:`PtuFile.global_resolution`.

        """
        period = float(self.tags.get('MeasDesc_GlobalResolution', 0.0))
        return 1.0 / period if period > 1e-14 else 0.0

    @property
    def syncrate(self) -> int:
        """Sync events per s as recorded at beginning of measurement."""
        return int(self.tags['TTResult_SyncRate'])

    @property
    def is_image(self) -> bool:
        """File contains image data."""
        return (
            self.tags['Measurement_SubMode'] == 3
            and self.tags.get('ImgHdr_Dimensions', 1) == 3
            and 'ImgHdr_PixX' in self.tags  # some Leica PTU are missing this
        )

    @property
    def is_t3(self) -> bool:
        """File contains T3 records."""
        return bool(self.tags['Measurement_Mode'] == 3)
        # return self.tags['TTResultFormat_TTTRRecType'] in {
        #     0x00010303, 0x00010304, 0x01010304,
        #     0x00010305, 0x00010306, 0x00010307,
        # }

    @property
    def is_bidirectional(self) -> bool:
        """Bidirectional scan mode."""
        return bool(self.tags.get('ImgHdr_BiDirect', 0) > 0)

    @property
    def is_sinusoidal(self) -> bool:
        """Sinusoidal scan mode."""
        return bool(self.tags.get('ImgHdr_SinCorrection', 0) != 0)

    @property
    def use_xarray(self) -> bool:
        """Return histograms as ``xarray.DataArray``."""
        return self._asxarray

    @use_xarray.setter
    def use_xarray(self, value: bool | None, /) -> None:
        self._asxarray = bool(value)

    @property
    def dtype(self) -> numpy.dtype[Any]:
        """Data type of image histogram array."""
        return self._dtype

    @dtype.setter
    def dtype(self, dtype: DTypeLike | None, /) -> None:
        dtype = numpy.dtype('uint16' if dtype is None else dtype)
        if dtype.kind != 'u':
            raise ValueError(f'{dtype=!r} not an unsigned integer')
        self._dtype = dtype

    @cached_property
    def shape(self) -> tuple[int, ...]:
        """Shape of image histogram array."""
        if not self.is_t3:
            return ()

        if 'C' in self._trimdims:
            nchannels = max(self.number_channels, 1)
        else:
            nchannels = self.number_channels_max
        if 'H' in self._trimdims:
            nbins = self.number_bins
        else:
            nbins = self.number_bins_max

        ndim = self.measurement_ndim
        if ndim == 3:
            return (
                max(self._info.frames, 1),
                self.lines_in_frame,
                self.pixels_in_line,
                nchannels,
                nbins,
            )
        if ndim == 2:
            return (
                max(self._info.lines, 1),
                self.pixels_in_line,
                nchannels,
                nbins,
            )
        if ndim in {0, 1}:
            return (
                max(1, self._info.photons // self.global_pixel_time),
                nchannels,
                nbins,
            )
        return ()

    @cached_property
    def dims(self) -> tuple[str, ...]:
        """Axes labels for each dimension in image histogram array."""
        if not self.shape:
            return ()
        ndim = self.measurement_ndim
        if ndim == 3:
            return ('T', 'Y', 'X', 'C', 'H')
        if ndim == 2:
            return ('T', 'X', 'C', 'H')
        return ('T', 'C', 'H')

    @property
    def ndims(self) -> int:
        """Number of dimensions in image histogram array."""
        return len(self.dims)

    @property
    def _coords_c(self) -> NDArray[Any]:
        """Coordinate array labelling all channels."""
        if 'C' in self._trimdims:
            return numpy.arange(
                self._info.channels_active_first,
                self._info.channels_active_last + 1,
                dtype=numpy.uint8,
            )
        return numpy.arange(self._info.channels, dtype=numpy.uint8)

    @property
    def _coords_h(self) -> NDArray[Any]:
        """Coordinate array labelling all delay-time bins."""
        if 'H' in self._trimdims:
            nbins = self.number_bins
        else:
            nbins = self.number_bins_max
        return numpy.linspace(  # type: ignore[no-any-return]
            0, nbins * self.tags['MeasDesc_Resolution'], nbins, endpoint=False
        )

    @cached_property
    def coords(self) -> dict[str, NDArray[Any]]:
        """Coordinate arrays labelling each point in image histogram array.

        Coordinates for the time axis are approximate. Exact coordinates are
        returned with :py:meth:`PtuFile.decode_image` as ``xarray.DataArray``.

        """
        if not self.shape:
            return {}
        ndim = self.measurement_ndim
        coords = {}
        shape = self.shape
        # exact time coordinates must be decoded from records
        coords['T'] = numpy.linspace(
            0, shape[0] * self.frame_time, shape[0], endpoint=False
        )
        res = self.tags.get('ImgHdr_PixResol', None)
        if res is not None:
            res *= 1e-6  # um
            if ndim > 2:
                offset = self.tags.get('ImgHdr_Y0', 0.0) * 1e-6  # um
                coords['Y'] = numpy.linspace(
                    offset, offset + shape[-4] * res, shape[-4], endpoint=False
                )
            if ndim > 1:
                offset = self.tags.get('ImgHdr_X0', 0.0) * 1e-6
                coords['X'] = numpy.linspace(
                    offset, offset + shape[-3] * res, shape[-3], endpoint=False
                )
        coords['C'] = self._coords_c
        coords['H'] = self._coords_h
        return coords

    @cached_property
    def _info(self) -> PtuInfo:
        """Information about decoded records."""
        from ._ptufile import decode_info

        lines_in_frame = 0
        if (
            'T' in self._trimdims
            and 'ImgHdr_PixY' in self.tags
            and self.tags['Measurement_SubMode'] == 3
        ):
            lines_in_frame = max(1, self.tags['ImgHdr_PixY'])

        return PtuInfo(
            *decode_info(
                self.read_records(),
                self.tags['TTResultFormat_TTTRRecType'],
                self.line_start_mask,
                self.line_stop_mask,
                self.frame_change_mask,
                lines_in_frame,
            )
        )

    def read_records(self, *, memmap: bool = False) -> NDArray[numpy.uint32]:
        """Return encoded TTTR records from file.

        Parameters:
            memmap:
                If true, memory-map the records in the file.
                By default, read the records from file into main memory.

        """
        if self.tags['TTResultFormat_BitsPerRecord'] not in {0, 32}:
            raise ValueError(
                'invalid bits per record '
                f"{self.tags['TTResultFormat_BitsPerRecord']}"
            )
        count = self.tags['TTResult_NumberOfRecords']
        result: NDArray[numpy.uint32]
        if memmap:
            return numpy.memmap(
                self._fh,
                dtype=numpy.uint32,
                mode='r',
                offset=self._data_offset,
                shape=(count,),
            )
        result = numpy.empty(count, numpy.uint32)
        self._fh.seek(self._data_offset)
        n = self._fh.readinto(result)  # type: ignore[attr-defined]
        if n != count * 4:
            logger().error(f'{self!r} expected {count} records, got {n // 4}')
            result = result[: n // 4]
        return result

    def decode_records(
        self,
        records: NDArray[numpy.uint32] | None = None,
        /,
        *,
        out: OutputType = None,
    ) -> NDArray[Any]:
        """Return decoded TTTR records.

        Parameters:
            records:
                Encoded TTTR records. By default, read records from file.
            out:
                Specifies where to decode records.
                If ``None``, create a new NumPy recarray in main memory.
                If ``'memmap'``, create a memory-mapped recarray in a
                temporary file.
                If a ``numpy.ndarray``, a writable recarray of compatible
                shape and dtype.
                If a ``file name`` or ``open file``, create a
                memory-mapped array in specified file.

        Returns:
            :
                ``numpy.recarray`` of size :py:attr:`number_records` and dtype
                :py:attr:`T3_RECORD_DTYPE` or :py:attr:`T2_RECORD_DTYPE`.

                A channel >= 0 indicates a record contains a photon.
                Else, the record contains an overflow event or marker > 0.

        """
        from ._ptufile import decode_t2_records, decode_t3_records

        if records is None:
            records = self.read_records()
        rectype = self.tags['TTResultFormat_TTTRRecType']
        if self.is_t3:
            result = create_output(out, (records.size,), T3_RECORD_DTYPE)
            decode_t3_records(result, records, rectype)
        else:
            result = create_output(out, (records.size,), T2_RECORD_DTYPE)
            decode_t2_records(result, records, rectype)
        return result

    @overload
    def decode_histogram(
        self,
        /,
        *,
        records: NDArray[numpy.uint32] | None = None,
        dtype: DTypeLike | None = None,
        sampling_time: int | None = None,
        dtime: int | None = None,
        asxarray: Literal[False] = ...,
        out: OutputType = None,
    ) -> NDArray[Any]: ...

    @overload
    def decode_histogram(
        self,
        /,
        *,
        records: NDArray[numpy.uint32] | None = None,
        dtype: DTypeLike | None = None,
        sampling_time: int | None = None,
        dtime: int | None = None,
        asxarray: Literal[True] = ...,
        out: OutputType = None,
    ) -> DataArray: ...

    @overload
    def decode_histogram(
        self,
        /,
        *,
        records: NDArray[numpy.uint32] | None = None,
        dtype: DTypeLike | None = None,
        sampling_time: int | None = None,
        dtime: int | None = None,
        asxarray: bool = ...,
        out: OutputType = None,
    ) -> NDArray[Any] | DataArray: ...

    def decode_histogram(
        self,
        /,
        *,
        records: NDArray[numpy.uint32] | None = None,
        dtype: DTypeLike | None = None,
        sampling_time: int | None = None,
        dtime: int | None = None,
        asxarray: bool = False,
        out: OutputType = None,
    ) -> NDArray[Any] | DataArray:
        """Return histogram of all photons by channel.

        Parameters:
            records:
                Encoded TTTR records. By default, read records from file.
            dtype:
                Unsigned integer type of histogram array.
                The default is ``uint32`` for T3, else ``uint16``.
                Increase the bit depth to avoid overflows.
            sampling_time:
                Global time per sample for T2 mode.
                The default is :py:meth:`PtuFile.global_pixel_time`.
            dtime:
                Specifies number of bins in histogram.
                If 0, return :py:attr:`number_bins_in_period` bins.
                If > 0, return up to specified bin.
            asxarray:
                If true, return ``xarray.DataArray``, else ``numpy.ndarray``
                (default).
            out:
                Specifies where to decode histogram.
                If ``None``, create a new NumPy array in main memory.
                If ``'memmap'``, create a memory-mapped array in a
                temporary file.
                If a ``numpy.ndarray``, a writable, initialized array
                of :py:attr:`shape` and unsigned integer dtype.
                If a ``file name`` or ``open file``, create a
                memory-mapped array in the specified file.

        Returns:
            :
                Decoded TTTR T3 records as 2-dimensional histogram array:

                - ``'C'`` channel
                - ``'H'`` histogram bins

        """
        from ._ptufile import decode_t2_histogram, decode_t3_histogram

        if dtype is None:
            dtype = numpy.uint32 if self.is_t3 else numpy.uint16
        dtype = numpy.dtype(dtype)
        if dtype.kind != 'u':
            raise ValueError(f'not an unsigned integer {dtype=!r}')

        if records is None:
            records = self.read_records()
        rectype = self.tags['TTResultFormat_TTTRRecType']

        if 'C' in self._trimdims:
            first_channel = self._info.channels_active_first
        else:
            first_channel = 0

        if self.is_t3:
            if dtime is None:
                nbins = self.shape[-1]
            elif dtime == 0:
                nbins = self.number_bins_in_period
            elif dtime > 0:
                nbins = dtime
            else:
                raise ValueError(f'{dtime=} < 0')
            histogram = create_output(out, (self.shape[-2], nbins), dtype)
            decode_t3_histogram(histogram, records, rectype, first_channel)
            coords = numpy.linspace(
                0,
                histogram.shape[-1] * self.tags['MeasDesc_Resolution'],
                histogram.shape[-1],
                endpoint=False,
            )
        else:
            if sampling_time is None or sampling_time <= 0:
                sampling_time = self.global_pixel_time
            histogram = create_output(
                out,
                (
                    self.number_channels,
                    max(1, self.global_acquisition_time // sampling_time),
                ),
                dtype,
            )
            decode_t2_histogram(
                histogram, records, rectype, sampling_time, first_channel
            )
            coords = numpy.linspace(
                0, self.acquisition_time, histogram.shape[1], endpoint=False
            )
        if not self._asxarray and not asxarray:
            return histogram

        from xarray import DataArray

        return DataArray(
            histogram,
            dims=('C', 'H'),
            coords={'C': self._coords_c, 'H': coords},  # name=self.name
        )

    @overload
    def decode_image(
        self,
        selection: Sequence[int | slice | EllipsisType | None] | None = None,
        /,
        *,
        records: NDArray[numpy.uint32] | None = None,
        dtype: DTypeLike | None = None,
        frame: int | None = None,
        channel: int | None = None,
        dtime: int | None = None,
        bishift: int | None = None,
        keepdims: bool = True,
        asxarray: Literal[False] = ...,
        out: OutputType = None,
    ) -> NDArray[Any]: ...

    @overload
    def decode_image(
        self,
        selection: Sequence[int | slice | EllipsisType | None] | None = None,
        /,
        *,
        records: NDArray[numpy.uint32] | None = None,
        dtype: DTypeLike | None = None,
        frame: int | None = None,
        channel: int | None = None,
        dtime: int | None = None,
        bishift: int | None = None,
        keepdims: bool = True,
        asxarray: Literal[True] = ...,
        out: OutputType = None,
    ) -> DataArray: ...

    @overload
    def decode_image(
        self,
        selection: Sequence[int | slice | EllipsisType | None] | None = None,
        /,
        *,
        records: NDArray[numpy.uint32] | None = None,
        dtype: DTypeLike | None = None,
        frame: int | None = None,
        channel: int | None = None,
        dtime: int | None = None,
        bishift: int | None = None,
        keepdims: bool = True,
        asxarray: bool = ...,
        out: OutputType = None,
    ) -> NDArray[Any] | DataArray: ...

    def decode_image(
        self,
        selection: Sequence[int | slice | EllipsisType | None] | None = None,
        /,
        *,
        records: NDArray[numpy.uint32] | None = None,
        dtype: DTypeLike | None = None,
        frame: int | None = None,
        channel: int | None = None,
        dtime: int | None = None,
        bishift: int | None = None,
        keepdims: bool = True,
        asxarray: bool = False,
        out: OutputType = None,
    ) -> NDArray[Any] | DataArray:
        """Return T3 mode point, line, or image histogram.

        The histogram may not include photons counted during incomplete frame
        scans or during line retraces.

        Parameters:
            selection:
                Indices for all dimensions:

                    - ``None``: return all items along axis (default).
                    - ``Ellipsis``: return all items along multiple axes.
                    - ``int``: return single item along axis.
                    - ``slice``: return chunk of axis.
                      ``slice.step`` is binning factor.
                      If ``slice.step=-1``, integrate all items along axis.

            records:
                Encoded TTTR records. By default, read records from file.
            dtype:
                Unsigned integer type of image histogram array.
                The default is ``uint16``. Increase the bit depth to avoid
                overflows when integrating.
            frame:
                If < 0, integrate time axis, else return specified frame.
                Overrides ``selection`` for axis ``T``.
            channel:
                If < 0, integrate channel axis, else return specified channel.
                Overrides ``selection`` for axis ``C``.
            dtime:
                Specifies number of bins in image histogram.
                If 0, return :py:attr:`number_bins_in_period` bins.
                If < 0, integrate delay time axis.
                If > 0, return up to specified bin.
                Overrides ``selection`` for axis ``H``.
            bishift:
                Global time shift of odd vs. even lines in bidirectional mode.
                The default is zero.
                Positive shifts invalidate left odd columns, while
                negative shifts invalidate right odd columns.
            keepdims:
                If true (default), reduced axes are left as size-one dimension.
            asxarray:
                If true, return ``xarray.DataArray``, else ``numpy.ndarray``
                (default).
            out:
                Specifies where to decode image histogram.
                If ``None``, create a new NumPy array in main memory.
                If ``'memmap'``, create a memory-mapped array in a
                temporary file.
                If a ``numpy.ndarray``, a writable, initialized array
                of :py:attr:`shape` and unsigned integer dtype.
                If a ``file name`` or ``open file``, create a
                memory-mapped array in the specified file.

        Returns:
            :
                Decoded TTTR T3 records as up to 5-dimensional image array:

                - ``'T'`` time/frame
                - ``'Y'`` slow scan axis for image scans
                - ``'X'`` fast scan axis for line and image scans
                - ``'C'`` channel
                - ``'H'`` histogram bins

        Raises:
            NotImplementedError:
                T2 images, bidirectional sinusoidal scanning, and deprecated
                image reconstruction are not supported.
            IndexError:
                Selection is out of bounds.

        """
        if not self.is_t3:
            # TODO: T2 images
            raise NotImplementedError('not a T3 image')
        if self.is_bidirectional:
            if not self.is_image:
                raise NotImplementedError(
                    'bidirectional scanning only supported for images'
                )
            if self.is_sinusoidal:
                raise NotImplementedError(
                    'bidirectional sinusoidal scanning not supported'
                )
        if self.is_image and 'ImgHdr_LineStart' not in self.tags:
            # TODO: deprecated image reconstruction using
            #   ImgHdr_PixResol, ImgHdr_TStartTo, ImgHdr_TStopTo,
            #   ImgHdr_TStartFro, ImgHdr_TStopFro
            raise NotImplementedError('old-style image reconstruction')

        shape = list(self.shape)
        ndim = len(shape)
        keepaxes: list[slice | int] = [slice(None)] * ndim

        if selection is None:
            selection = [None] * ndim
        else:
            try:
                len(selection)
            except TypeError:
                selection = [selection]  # type: ignore[list-item]

            if len(selection) > ndim:
                raise IndexError(f'too many indices in {selection=}')
            elif len(selection) == ndim:
                selection = list(selection).copy()
                if Ellipsis in selection:
                    selection[selection.index(Ellipsis)] = None
            # elif len(selection) < ndim:
            elif Ellipsis in selection:
                selection = list(selection).copy()
                i = selection.index(Ellipsis)
                selection = (
                    selection[:i]
                    + ([None] * (1 + ndim - len(selection)))
                    + selection[i + 1 :]
                )
            else:
                selection = list(selection) + [None] * (ndim - len(selection))
            if Ellipsis in selection:
                raise IndexError(f'more than one Ellipsis in {selection=}')

        if frame is not None:
            if frame >= shape[0]:
                raise IndexError(f'{frame=} out of range')
            selection[0] = frame if frame >= 0 else slice(None, None, -1)
        if channel is not None:
            if channel >= shape[-2]:
                raise IndexError(f'{channel=} out of range')
            selection[-2] = channel if channel >= 0 else slice(None, None, -1)
        if dtime is not None:
            if dtime == 0:
                dtime = self.number_bins_in_period
            if dtime > 0:
                if dtime > self.number_bins_max:
                    raise IndexError(
                        f'{dtime=} out of range {self.number_bins_max}'
                    )
                selection[-1] = slice(0, dtime, 1)
                shape[-1] = dtime
            else:
                selection[-1] = slice(None, None, -1)

        start = [0] * ndim
        step = [1] * ndim
        for i, (index, size) in enumerate(zip(selection, self.shape)):
            if index is None:
                pass
            elif isinstance(index, int):
                if index < 0:
                    index %= shape[i]
                if not 0 <= index < size:
                    raise IndexError(f'axis {i} {index=} out of range')
                start[i] = index
                shape[i] = 1
                keepaxes[i] = 0
            elif isinstance(index, slice):
                istart = index.start
                if istart is not None:
                    if istart < 0:
                        istart %= shape[i]
                    if not 0 <= istart < shape[i]:
                        raise IndexError(
                            f'axis {i} {index=} start out of range'
                        )
                    start[i] = istart
                if index.stop is not None:
                    istop = index.stop
                    if istop < 0:
                        istop %= shape[i]
                    if not start[i] < istop <= shape[i]:
                        raise IndexError(
                            f'axis {i} {index=} stop out of range'
                        )
                    shape[i] = istop
                shape[i] -= start[i]
                if index.step is not None:
                    if index.step < 0:
                        # negative step size -> integrate all
                        step[i] = shape[i]
                        keepaxes[i] = 0
                    else:
                        step[i] = min(index.step, shape[i])
                    shape[i] = shape[i] // step[i] + min(1, shape[i] % step[i])
            else:
                raise IndexError(
                    f'axis {i} index type {type(index)!r} invalid'
                )

        if self._info.channels_active_first > 0 and 'C' in self._trimdims:
            # set channel offset
            start[-2] += self._info.channels_active_first

        if self.is_sinusoidal:
            pixel_time = 0
            pixel_at_time = sinusoidal_correction(
                self.tags['ImgHdr_SinCorrection'],
                self.global_line_time,
                self.pixels_in_line,
                dtype=numpy.uint16,  # should be enough for pixels_in_line
            )
        else:
            pixel_time = self.global_pixel_time
            pixel_at_time = numpy.empty(0, dtype=numpy.uint16)

        if dtype is None:
            dtype = self._dtype
        else:
            dtype = numpy.dtype(dtype)
            if dtype.kind != 'u':
                raise ValueError(f'not an unsigned integer {dtype=!r}')

        histogram = create_output(out, tuple(shape), dtype)
        times = numpy.zeros(shape[0], numpy.uint64)

        from ._ptufile import decode_t3_image, decode_t3_line, decode_t3_point

        if records is None:
            records = self.read_records()

        if ndim == 5:
            decode_t3_image(
                histogram,
                times,
                records,
                self.tags['TTResultFormat_TTTRRecType'],
                pixel_time,
                pixel_at_time,
                self.line_start_mask,
                self.line_stop_mask,
                self.frame_change_mask,
                *start,
                *step,
                self._info.skip_first_frame,
                self.pixels_in_line if self.is_bidirectional else 0,
                0 if bishift is None else bishift,
            )
        elif ndim == 4:
            # not tested
            decode_t3_line(
                histogram,
                times,
                records,
                self.tags['TTResultFormat_TTTRRecType'],
                self.global_pixel_time,
                self.line_start_mask,
                self.line_stop_mask,
                *start,
                *step,
            )
        elif ndim == 3:
            decode_t3_point(
                histogram,
                times,
                records,
                self.tags['TTResultFormat_TTTRRecType'],
                self.global_pixel_time,
                *start,
                *step,
            )

        if not keepdims:
            histogram = histogram[tuple(keepaxes)]

        if not self._asxarray and not asxarray:
            return histogram

        from xarray import DataArray

        if self._info.channels_active_first > 0 and 'C' in self._trimdims:
            # unset channel offset
            start[-2] -= self._info.channels_active_first

        dims = []
        coords = self.coords.copy()
        for i, ax in enumerate(self.dims):
            if keepdims or keepaxes[i] != 0:
                dims.append(ax)
                if ax in coords:
                    index = slice(
                        start[i], start[i] + shape[i] * step[i], step[i]
                    )
                    coords[ax] = coords[ax][index]
            elif ax in coords:
                del coords[ax]
        if 'H' in dims and len(coords['H']) < shape[-1]:
            coords['H'] = numpy.linspace(
                0,
                shape[-1] * coords['H'][1],
                shape[-1],
                endpoint=False,
            )
        if 'T' in dims:
            coords['T'] = times * self.global_resolution
        attrs = {
            'frequency': self.frequency,
            'max_delaytime': self.number_bins_max,
        }

        return DataArray(
            histogram,
            dims=dims,
            coords=coords,
            attrs=attrs,
            # name=self.name,
        )

    def plot(
        self,
        *,
        samples: int | None = None,
        frame: int | None = None,
        channel: int | None = None,
        dtime: int | None = -1,
        verbose: bool = False,
        show: bool = True,
        **kwargs: Any,
    ) -> None:
        """Plot histograms using matplotlib.

        Parameters:
            samples:
                Number of bins along measurement for T2 mode.
                The default is 1000.
            frame:
                If < 0, integrate time axis, else show specified frame.
                By default all frames are shown. Applies to T3 images.
            channel:
                If < 0, integrate channel axis, else show specified channel.
                By default all channels are shown. Applies to T3 images.
            dtime:
                Specifies number of bins in T3 histograms.
                If < 0 (default), integrate delay time axis of images.
                If 0, show :py:attr:`number_bins_in_period` bins.
                If > 0, show histograms up to specified bin.
                If None, show all bins.
            verbose:
                Print information about histogram arrays.
            show:
                If true (default), display all figures.
                Else, defer to user or environment to display figures.
            **kwargs:
                Additional arguments passed to ``tifffile.imshow``.

        """
        from matplotlib import pyplot
        from tifffile import Timer, imshow

        t = Timer()
        if self.is_t3:
            if self.measurement_ndim > 1:
                t.start()
                histogram: Any = self.decode_image(
                    frame=frame, channel=channel, dtime=dtime, asxarray=True
                )
                if verbose:
                    print()
                    t.print('decode_image')
                    print()
                    print(histogram.squeeze())
                imshow(
                    numpy.transpose(
                        histogram.values, (0, 3, 4, 1, 2)
                    ).squeeze(),
                    title=repr(self),
                    photometric='minisblack',
                    **kwargs,
                )
                pyplot.figure()
            t.start()
            # histogram = histogram.sum(axis=(0, 1, 2), dtype=numpy.uint32)
            dtime = None if dtime is None or dtime < 0 else dtime
            histogram = self.decode_histogram(asxarray=True, dtime=dtime)
        else:
            if samples is None or samples < 1:
                samples = 1000
            histogram = self.decode_histogram(
                sampling_time=self.global_acquisition_time // samples,
                asxarray=True,
            )
        if verbose:
            print()
            t.print('decode_histogram')
            print()
            print(histogram)
        channels = histogram.coords['C'].values
        for i, hist in enumerate(histogram):
            pyplot.plot(
                hist.coords['H'], hist.values, label=f'ch {channels[i]}'
            )
        if 0.0 < self.frequency:
            pyplot.axvline(x=1 / self.frequency, color='0.5', ls=':', lw=0.75)
        pyplot.title(repr(self))
        pyplot.xlabel('delay time [s]' if self.is_t3 else 'time [s]')
        pyplot.ylabel('photon count')
        pyplot.legend()
        if show:
            pyplot.show()


@dataclasses.dataclass
class PtuInfo:
    """Information about decoded TTTR records.

    Returned by ``_ptufile.decode_info``.

    """

    format: int
    """Type of records."""

    records: int
    """Number of records."""

    photons: int
    """Number of photons counted."""

    markers: int
    """Number of marker events."""

    frames: int
    """Number of frames detected. May exclude incomplete frames."""

    lines: int
    """Number of lines between line markers."""

    channels: int
    """Maximum number of channels for record type."""

    channels_active: int
    """Bitfield identifying channels with photons."""

    channels_active_first: int
    """First channel with photons."""

    channels_active_last: int
    """Last channel with photons."""

    bins: int
    """Maximum delay time for record type."""

    bins_used: int
    """Highest delay time observed. Not available for T2 records."""

    skip_first_frame: bool
    """First frame of multi-frame image is incomplete."""

    skip_last_frame: bool
    """Last frame of multi-frame image is incomplete."""

    line_time: int
    """Average global time between line markers."""

    acquisition_time: int
    """Global time of last sync event."""

    def __str__(self) -> str:
        return indent(
            f'{self.__class__.__name__}(',
            *(f'{key}={value},' for key, value in self.__dict__.items()),
            end='\n)',
        )


def sinusoidal_correction(
    sincorrect: float,
    global_line_time: int,
    pixels_in_line: int,
    dtype: DTypeLike = None,
) -> NDArray[Any]:
    """Return pixel indices of global times in line for sinusoidal scanning.

    Parameters:
        sincorrect:
            Percentage of amplitude of sine wave used for measurement.
            The value of the `ImgHdr_SinCorrection` tag.
        global_line_time:
            Global time per line.
        pixels_in_line:
            Number of pixels in line.

    Returns:
        Array of size `global_line_time`, mapping global time in line to
        pixel index in line.

    """
    # TODO: Leica uses fraction of overall period of sinus wave?
    dtype = numpy.dtype(numpy.uint16 if dtype is None else dtype)
    if sincorrect <= 0.0 or sincorrect > 100.0:
        raise ValueError(f'{sincorrect=} out of range')
    if global_line_time < 2:
        raise ValueError(f'{global_line_time=} out of range')
    if pixels_in_line < 2 or pixels_in_line >= numpy.iinfo(dtype).max:
        raise ValueError(f'{pixels_in_line=} out of range')
    limit = math.asin(-sincorrect / 100.0)
    a = numpy.linspace(limit, -limit, global_line_time, endpoint=False)
    a = numpy.sin(a)
    a *= -0.5 * pixels_in_line / a[0]
    a -= a[0]
    return a.astype(dtype)


def create_output(
    out: str | IO[bytes] | NDArray[Any] | None,
    /,
    shape: tuple[int, ...],
    dtype: DTypeLike,
) -> NDArray[Any] | numpy.memmap[Any, Any]:
    """Return NumPy array where images of shape and dtype can be copied."""
    if out is None:
        return numpy.zeros(shape, dtype)
    if isinstance(out, numpy.ndarray):
        out.shape = shape
        return out
    if isinstance(out, str) and out[:6] == 'memmap':
        import tempfile

        tempdir = out[7:] if len(out) > 7 else None
        with tempfile.NamedTemporaryFile(dir=tempdir, suffix='.memmap') as fh:
            return numpy.memmap(fh, shape=shape, dtype=dtype, mode='w+')
    return numpy.memmap(out, shape=shape, dtype=dtype, mode='w+')


def indent(*args: Any, sep: str = '', end: str = '') -> str:
    """Return joined string representations of objects with indented lines."""
    text = (sep + '\n').join(
        arg if isinstance(arg, str) else repr(arg) for arg in args
    )
    return (
        '\n'.join(
            ('    ' + line if line else line)
            for line in text.splitlines()
            if line
        )[4:]
        + end
    )


def logger() -> logging.Logger:
    """Return logging.getLogger('ptufile')."""
    return logging.getLogger(__name__.replace('ptufile.ptufile', 'ptufile'))


def askopenfilename(**kwargs: Any) -> str:
    """Return file name(s) from Tkinter's file open dialog."""
    from tkinter import Tk, filedialog

    root = Tk()
    root.withdraw()
    root.update()
    filenames = filedialog.askopenfilename(**kwargs)
    root.destroy()
    return filenames


def main(argv: list[str] | None = None) -> int:
    """Command line usage main function.

    Preview image and metadata in specified files or all files in directory.

    ``python -m ptufile file_or_directory``

    """
    from glob import glob

    if argv is None:
        argv = sys.argv

    extensions: set[str] | None = set(FILE_EXTENSIONS.keys())
    if len(argv) == 1:
        path = askopenfilename(
            title='Select a TIFF file',
            filetypes=[
                (f'{ext.upper()} files', f'*{ext}') for ext in FILE_EXTENSIONS
            ]
            + [('allfiles', '*')],
        )
        files = [path] if path else []
    elif '*' in argv[1]:
        files = glob(argv[1])
    elif os.path.isdir(argv[1]):
        files = glob(f'{argv[1]}/*.p*')
    else:
        files = argv[1:]
        extensions = None

    for fname in files:
        if (
            extensions
            and os.path.splitext(fname)[-1].lower() not in extensions
        ):
            continue
        try:
            with PqFile(fname) as pq:
                if pq.magic == PqFileMagic.PTU:
                    with PtuFile(fname) as ptu:
                        # TODO: print decoding time
                        print(ptu)
                        print(ptu._info)
                        try:
                            ptu.plot(verbose=True)
                        except NotImplementedError as exc:
                            print('NotImplementedError:', exc)
                elif pq.magic == PqFileMagic.PHU:
                    with PhuFile(fname) as phu:
                        print(phu)
                        phu.plot(verbose=True)
                else:
                    print(pq)
                print()
        except ValueError as exc:
            # raise  # enable for debugging
            print(fname, exc)
            continue

    return 0


if __name__ == '__main__':
    sys.exit(main())
