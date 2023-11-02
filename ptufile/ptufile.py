# ptufile.py

# Copyright (c) 2023, Christoph Gohlke
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

`PicoQuant GmbH <https://www.picoquant.com/>`_ is a manufacturer of
photonic components and instruments.

:Author: `Christoph Gohlke <https://www.cgohlke.com>`_
:License: BSD 3-Clause
:Version: 2023.11.1

Quickstart
----------

Install the ptufile package and all dependencies from the
`Python Package Index <https://pypi.org/project/ptufile/>`_::

    python -m pip install -U ptufile[all]

See `Examples`_ for using the programming interface.

Source code and support are available on
`GitHub <https://github.com/cgohlke/ptufile>`_.

Requirements
------------

This revision was tested with the following requirements and dependencies
(other versions may work):

- `CPython <https://www.python.org>`_ 3.9.13, 3.10.11, 3.11.6, 3.12.0 (64-bit)
- `Numpy <https://pypi.org/project/numpy>`_ 1.25.2
- `Xarray <https://pypi.org/project/xarray>`_ 2023.10.1 (recommended)
- `Matplotlib <https://pypi.org/project/matplotlib/>`_ 3.7.3 (optional)
- `Tifffile <https://pypi.org/project/tifffile/>`_ 2023.9.26 (optional)

Revisions
---------

2023.11.1

- Initial alpha release.

Notes
-----

The API is not stable yet and might change between revisions.

This library has been tested with a limited number of files only.

The following features are currently not implemented: PT2 and PT3 files,
decoding images from T2 formats, bidirectional scanning, sinusoidal correction,
and deprecated image reconstruction.

The PicoQuant unified file formats are documented at the
`PicoQuant-Time-Tagged-File-Format-Demos
<https://github.com/PicoQuant/PicoQuant-Time-Tagged-File-Format-Demos/tree/master/doc>`_

Other Python implementations for reading PicoQuant files are
`Read_PTU.py
<https://github.com/PicoQuant/PicoQuant-Time-Tagged-File-Format-Demos/blob/master/PTU/Python/Read_PTU.py>`_,
`readPTU_FLIM <https://github.com/SumeetRohilla/readPTU_FLIM>`_,
`picoquantio <https://github.com/tsbischof/picoquantio>`_, and
`napari-flim-phasor-plotter
<https://github.com/zoccoler/napari-flim-phasor-plotter/blob/main/src/napari_flim_phasor_plotter/_io/readPTU_FLIM.py>`_.

The development of this library was supported by the
`Chan Zuckerberg Initiative
<https://chanzuckerberg.com/eoss/proposals/phasorpy-a-python-library-for-phasor-analysis-of-flim-and-spectral-imaging>`_.

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

Read metadata from PicoQuant PTU FLIM file:

>>> ptu = PtuFile('tests/FLIM.ptu')
>>> ptu.magic
<PqFileMagic.PTU: ...>
>>> ptu.type
<PtuRecordType.PicoHarpT3: 66307>
>>> ptu.measurement_mode
<PtuMeasurementMode.T3: 3>
>>> ptu.measurement_submode
<PtuMeasurementSubMode.IMAGE: 3>

Decode TTTR records from file to numpy.recarray and get global times of frame
changes from the masks:

>>> decoded = ptu.decode_records()
>>> decoded['time'][(decoded['marker'] & ptu.frame_change_mask) > 0]
array([1571185680], dtype=uint64)

Decode TTTR records to delay-time histogram per channel:

>>> ptu.decode_histogram(dtype='uint8', asxarray=False)
array([[ 5,  7,  7, ..., 10,  9,  2]], dtype=uint8)

Read FLIM histogram as xarray, decoding only the first channel and
integrating all histogram bins:

>>> ptu.shape
(2, 1, 256, 256, 3126)
>>> ptu.dims
('C', 'T', 'Y', 'X', 'H')
>>> ptu.coords
{'T': ..., 'Y': ..., 'X': ..., 'H': ...}
>>> ptu.decode_image(channel=0, dtime=-1, asxarray=True)  # doctest: +SKIP
<xarray.DataArray (C: 1, T: 1, Y: 256, X: 256, H: 1)>
array([[[[[...]]]]], dtype=uint16)
Coordinates:
    * T        (T) float64 0.0
    * Y        (Y) float64 -0.0001304 -0.0001294 ... 0.0001284 0.0001294
    * X        (X) float64 -0.0001304 -0.0001294 ... 0.0001284 0.0001294
    * H        (H) float64 0.0
Dimensions without coordinates: C
...
>>> ptu.close()

View the image and metadata in a PTU file from the console::

    $ python -m ptufile tests/FLIM.ptu

"""

from __future__ import annotations

__version__ = '2023.11.1'

__all__ = [
    'imread',
    'PqFile',
    'PqFileError',
    'PqFileMagic',
    'PhuFile',
    'PtuFile',
    'PtuRecordType',
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
import os
import struct
import sys
import time
import uuid
from functools import cached_property
from typing import TYPE_CHECKING, final

if TYPE_CHECKING:
    from typing import Any, BinaryIO

    from collections.abc import Sequence

    from numpy.typing import NDArray, DTypeLike
    from xarray import DataArray

import numpy


def imread(
    file: str | os.PathLike | BinaryIO,
    selection: Sequence[int | slice | None] | None = None,
    /,
    *,
    trim_dtime: bool = True,
    dtype: DTypeLike | None = None,
    channel: int | None = None,
    frame: int | None = None,
    dtime: int | None = None,
    asxarray: bool = False,
) -> NDArray[Any] | DataArray:
    """Return decoded image histogram from T3 mode PTU file.

    Parameters:
        file:
            File name or seekable binary stream.
        selection, dtype, channel, frame, asxarray:
            Passed to :py:meth:`PtuFile.decode_image`.
        trim_dtime:
            Passed to :py:class:`PtuFile`.

    """
    with PtuFile(file, trim_dtime=trim_dtime) as ptu:
        data = ptu.decode_image(
            selection,
            dtype=dtype,
            channel=channel,
            frame=frame,
            dtime=dtime,
            asxarray=asxarray,
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

    PicoHarpT3 = 0x00010303
    PicoHarpT2 = 0x00010203
    HydraHarpT3 = 0x00010304
    HydraHarpT2 = 0x00010204
    HydraHarp2T3 = 0x01010304
    HydraHarp2T2 = 0x01010204
    TimeHarp260NT3 = 0x00010305
    TimeHarp260NT2 = 0x00010205
    TimeHarp260PT3 = 0x00010306
    TimeHarp260PT2 = 0x00010206
    MultiHarpT2 = 0x00010207
    MultiHarpT3 = 0x00010307


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

FILE_EXTENSIONS = {'.ptu', '.phu', '.pck', '.pco', '.pfs', '.pus', '.pqres'}
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

    _fh: BinaryIO
    _close: bool
    _data_offset: int  # position of raw data in file

    _MAGIC: set[PqFileMagic] = set(PqFileMagic)

    def __init__(
        self,
        file: str | os.PathLike | BinaryIO,
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

        def errmsg(msg, tagid, index, typecode, value):
            return (
                f'{msg} @ {self.name!r} '
                f'{tagid=}, {index=}, {typecode=}, {value=!r}'
            )[:80]

        ty = PqTagType
        unpack = struct.unpack
        try:
            while True:
                # offset = fh.tell()
                tagid, index, typecode, value = unpack('<32siI8s', fh.read(48))
                # print(tagid.strip(b'\0'), index, typecode, value)
                tagid = tagid.rstrip(b'\0').decode('ascii', errors='ignore')
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

    def __exit__(self, exc_type, exc_value, traceback) -> None:  # type: ignore
        self.close()

    def __repr__(self) -> str:
        return f'<{self.__class__.__name__} {self.name!r}>'

    def __str__(self) -> str:
        return indent(
            repr(self),
            f'magic: {self.magic}',
            f'version: {self.version}',
            *(
                f'{name}: {getattr(self, name)!r}'[:160]
                for name in dir(self)
                if not (
                    name in {'tags', 'magic', 'version'}
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
        file: File name or seekable binary stream.

    Raises:
        PqFileError: File is not a PicoQuant PHU file or is corrupted.

    """

    _MAGIC: set[PqFileMagic] = {PqFileMagic.PHU}

    def __init__(self, file: str | os.PathLike | BinaryIO, /) -> None:
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

    def histograms(
        self,
        index: int | slice | None = None,
        /,
        *,
        asxarray: bool = False,
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
            y = numpy.trim_zeros(hist.values, trim='b')  # type: ignore
            x = hist.coords['H'].values[: y.size]  # type: ignore
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
        trim_dtime:
            If true (default), limit the number of T3 bins returned to the
            largest non-zero delay time bin observed, else use record type's
            default.

    Raises:
        PqFileError: File is not a PicoQuant PTU file or is corrupted.

    """

    _trim_dtime: bool

    _MAGIC: set[PqFileMagic] = {PqFileMagic.PTU}

    def __init__(
        self,
        file: str | os.PathLike | BinaryIO,
        /,
        *,
        trim_dtime: bool = True,
    ) -> None:
        super().__init__(file)
        self._trim_dtime = bool(trim_dtime)

    def __enter__(self) -> PtuFile:
        return self

    @property
    def type(self) -> PtuRecordType:
        """Type of TTTR records.

        Defines the TCSPC device and type of measurement that produced the
        records.

        """
        return PtuRecordType(self.tags['TTResultFormat_TTTRRecType'])

    @property
    def number_records(self) -> int:
        """Number of TTTR records."""
        return int(self.tags['TTResult_NumberOfRecords'])

    @property
    def syncrate(self) -> int:
        """Counts per s as recorded at beginning of measurement."""
        return int(self.tags['TTResult_SyncRate'])

    @property
    def measurement_mode(self) -> PtuMeasurementMode:
        """Kind of TCSPC measurement: T2 or T3."""
        return PtuMeasurementMode(self.tags['Measurement_Mode'])

    @property
    def measurement_submode(self) -> PtuMeasurementSubMode:
        """Sub-kind of measurement: Point, line, or image. scan"""
        return PtuMeasurementSubMode(self.tags['Measurement_SubMode'])

    @property
    def global_resolution(self) -> float:
        """Resolution of time tags in s."""
        return float(self.tags['MeasDesc_GlobalResolution'])

    @property
    def tcspc_resolution(self) -> float:
        """Resolution of TCSPC in s (BaseResolution * iBinningFactor)."""
        return float(self.tags.get('MeasDesc_Resolution', 0.0))

    @property
    def number_photons(self) -> int:
        """Number of photons counted."""
        return self._info.photons

    @property
    def number_markers(self) -> int:
        """Number of marker events."""
        return self._info.markers

    @property
    def number_frames(self) -> int:
        """Number of frame markers."""
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
        """Highest channel number with photons."""
        # TODO: only use channels with photons
        # len(ch for ch in self.tags['HWInpChan_Enabled'] if ch) ?
        return self._info.channels_used

    @property
    def number_bins_max(self) -> int:
        """Maximum delay time for record type."""
        return self._info.bins

    @property
    def number_bins(self) -> int:
        """Highest delay time measured. Not available for T2 records."""
        return self._info.bins_used

    @property
    def line_start_mask(self) -> int:
        """Marker mask defining line start, or 0 if not defined."""
        value = self.tags.get('ImgHdr_LineStart', None)
        return 2 ** (value - 1) if value is not None else 0

    @property
    def line_stop_mask(self) -> int:
        """Marker mask defining line end, or 0 if not defined."""
        value = self.tags.get('ImgHdr_LineStop', None)
        return 2 ** (value - 1) if value is not None else 0

    @property
    def frame_change_mask(self) -> int:
        """Marker mask defining frame change, or 0 if not defined."""
        value = self.tags.get('ImgHdr_Frame', None)
        return 2 ** (value - 1) if value is not None else 0

    @property
    def global_pixel_time(self) -> int:
        """Global time per pixel.

        Multiply with global resolution to get time in s.
        """
        if 'ImgHdr_TimePerPixel' in self.tags:
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
        """Approximate global time per line.

        Multiply with global resolution to get time in s.
        """
        if 'ImgHdr_TimePerPixel' in self.tags:
            linetime = self.pixels_in_line * self.global_pixel_time
        elif self._info.lines > 0:
            linetime = self._info.line_time
        else:
            linetime = 1e-3 / self.tags['MeasDesc_GlobalResolution']
        return int(round(linetime))

    @property
    def global_frame_time(self) -> int:
        """Approximate global time per frame.

        Multiply with global resolution to get time in s.
        """
        if self._info.frames > 0:
            # image average includes retrace, etc
            return self._info.frame_time
        if self._info.lines > 0:
            # line scan average includes retrace, etc
            return self._info.line_time
        # does not include retrace, etc
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
        """Number of pixels in frame."""
        return self.lines_in_frame * self.pixels_in_line

    @property
    def pixels_in_line(self) -> int:
        """Number of pixels in line."""
        ndim = self.tags['Measurement_SubMode']
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
        if self.tags['Measurement_SubMode'] == 3:
            # image
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
        """Average time between frame markers or ``line_time`` in s."""
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
        """Repetition frequency in s, or 0 if not applicable."""
        period = self.number_bins_max * float(
            self.tags.get('MeasDesc_Resolution', 0.0)
        )
        return 1 / period if period > 1e-14 else 0

    @cached_property
    def shape(self) -> tuple[int, ...]:
        """Shape of image histogram array."""
        if not self.is_t3:
            return ()
        nbins = self.number_bins if self._trim_dtime else self.number_bins_max
        nchannels = max(self.number_channels, 1)
        ndim = self.tags['Measurement_SubMode']
        if ndim == 3:
            return (
                nchannels,
                max(self.number_frames, 1),
                self.lines_in_frame,
                self.pixels_in_line,
                nbins,
            )
        if ndim == 2:
            return (
                nchannels,
                max(self.number_lines, 1),
                self.pixels_in_line,
                nbins,
            )
        if ndim in {0, 1}:
            return (
                nchannels,
                max(1, self._info.photons // self.global_pixel_time),
                nbins,
            )
        return ()

    @cached_property
    def dims(self) -> tuple[str, ...]:
        """Axes labels for each dimension in image histogram array

        - ``'C'`` channel
        - ``'T'`` time frame
        - ``'Y'`` slow scan axis
        - ``'X'`` fast scan axis
        - ``'H'`` histogram bins

        """
        if not self.shape:
            return ()
        ndim = self.tags['Measurement_SubMode']
        if ndim == 3:
            return ('C', 'T', 'Y', 'X', 'H')
        if ndim == 2:
            return ('C', 'T', 'X', 'H')
        return ('C', 'T', 'H')

    @cached_property
    def coords(self) -> dict[str, NDArray]:
        """Coordinate arrays labelling each point in image histogram array.

        Coordinates for the time axis are approximate. Exact coordinates are
        returned with :py:meth:`PtuFile.decode_image` xarray.Dataset arrays.

        """
        if not self.shape:
            return {}
        ndim = self.tags['Measurement_SubMode']
        coords = {}
        shape = self.shape
        # exact time coordinates must be decoded from records
        coords['T'] = numpy.linspace(
            0, shape[1] * self.frame_time, shape[1], endpoint=False
        )
        res = self.tags.get('ImgHdr_PixResol', None)
        if res is not None:
            res *= 1e-6  # um
            if ndim > 2:
                offset = self.tags.get('ImgHdr_Y0', 0.0) * 1e-6  # um
                coords['Y'] = numpy.linspace(
                    offset, offset + shape[-3] * res, shape[-3], endpoint=False
                )
            if ndim > 1:
                offset = self.tags.get('ImgHdr_X0', 0.0) * 1e-6
                coords['X'] = numpy.linspace(
                    offset, offset + shape[-2] * res, shape[-2], endpoint=False
                )
        coords['H'] = numpy.linspace(
            0,
            shape[-1] * self.tags['MeasDesc_Resolution'],
            shape[-1],
            endpoint=False,
        )
        return coords

    @property
    def is_image(self) -> bool:
        """File contains image data."""
        return (
            self.tags['Measurement_SubMode'] == 3
            and self.tags.get('ImgHdr_Dimensions', 1) == 3
            and 'ImgHdr_PixX' in self.tags
        )

    @property
    def is_t3(self) -> bool:
        """File contains T3 records."""
        return self.tags['Measurement_Mode'] == 3
        # return self.tags['TTResultFormat_TTTRRecType'] in {
        #     0x00010303, 0x00010304, 0x01010304,
        #     0x00010305, 0x00010306, 0x00010307,
        # }

    @property
    def is_bidirectional(self) -> bool:
        """Bidirectional scan mode."""
        return self.tags.get('ImgHdr_BiDirect', False)

    @cached_property
    def _info(self) -> PtuInfo:
        """Information about decoded records."""
        from ._ptufile import _decode_info

        return PtuInfo(
            *_decode_info(
                self.read_records(),
                self.tags['TTResultFormat_TTTRRecType'],
                self.line_start_mask,
                self.line_stop_mask,
                self.frame_change_mask,
            )
        )

    def read_records(self) -> NDArray[numpy.uint32]:
        """Return encoded TTTR records from file."""
        if self.tags['TTResultFormat_BitsPerRecord'] not in {0, 32}:
            raise ValueError(
                'invalid bits per record '
                f"{self.tags['TTResultFormat_BitsPerRecord']}"
            )
        count = self.tags['TTResult_NumberOfRecords']
        result = numpy.empty(count, numpy.uint32)
        self._fh.seek(self._data_offset)
        n = self._fh.readinto(result)  # type: ignore
        if n != count * 4:
            logger().error(f'{self!r} expected {count} records, got {n // 4}')
            result = result[: n // 4]
        return result

    def decode_records(self) -> NDArray[Any]:
        """Return decoded TTTR records from file as.

        The returned ``numpy.recarray`` has :py:attr:`T3_RECORD_DTYPE`
        or :py:attr:`T2_RECORD_DTYPE` dtype.

        A channel > 0 indicates a record contained a photon.
        Else, the record contained an overflow event or marker mask > 0.

        """
        from ._ptufile import _decode_t2_records, _decode_t3_records

        records = self.read_records()
        rectype = self.tags['TTResultFormat_TTTRRecType']
        if self.is_t3:
            result = numpy.zeros(records.size, T3_RECORD_DTYPE)
            _decode_t3_records(result, records, rectype)
        else:
            result = numpy.zeros(records.size, T2_RECORD_DTYPE)
            _decode_t2_records(result, records, rectype)
        return result

    def decode_histogram(
        self,
        /,
        *,
        dtype: DTypeLike | None = None,
        asxarray: bool | None = None,
        sampling_time: int | None = None,
    ) -> NDArray[Any] | DataArray:
        """Return histogram.

        Parameters:
            dtype:
                Unsigned integer type of histogram array.
                The default is ``uint32`` for T3, else ``uint16``.
            asxarray:
                If true, return histogram as ``xarray.DataArray``,
                else ``numpy.ndarray`` (default).
            sampling_time:
                Global time per sample for T2 mode.
                The default is :py:meth:`PtuFile.global_pixel_time`.

        Returns:
            :
                Decoded TTTR T3 records as 2-dimensional histogram:

                - ``'C'`` channel
                - ``'H'`` histogram bins

        """
        from ._ptufile import _decode_t2_histogram, _decode_t3_histogram

        if dtype is None:
            dtype = numpy.uint32 if self.is_t3 else numpy.uint16
        dtype = numpy.dtype(dtype)
        if dtype.kind != 'u':
            raise ValueError(f'not an unsigned integer {dtype=!r}')

        records = self.read_records()
        rectype = self.tags['TTResultFormat_TTTRRecType']

        if self.is_t3:
            histogram = numpy.zeros((self.shape[0], self.shape[-1]), dtype)
            _decode_t3_histogram(histogram, records, rectype)
            coords = self.coords['H']
        else:
            if sampling_time is None or sampling_time <= 0:
                nbins = self.global_acquisition_time // self.global_pixel_time
            else:
                nbins = self.global_acquisition_time // sampling_time
            histogram = numpy.zeros(
                (self.number_channels, max(1, nbins)), dtype
            )
            _decode_t2_histogram(
                histogram,
                records,
                rectype,
                self.global_pixel_time,
            )
            coords = numpy.linspace(
                0, self.acquisition_time, histogram.shape[1], endpoint=False
            )
        if not asxarray:
            return histogram

        from xarray import DataArray

        return DataArray(
            histogram,
            dims=('C', 'H'),
            coords={'H': coords},  # name=self.name
        )

    def decode_image(
        self,
        selection: Sequence[int | slice | None] | None = None,
        /,
        *,
        dtype: DTypeLike | None = None,
        channel: int | None = None,  # -1 integrate, >=0 index
        frame: int | None = None,  # -1 integrate, >=0 index
        dtime: int | None = None,  # -1 integrate, >=0 index
        asxarray: bool | None = None,
    ) -> NDArray[Any] | DataArray:
        """Return T3 mode point, line, or image histogram.

        Parameters:
            selection:
                Sequence of indices for all dimensions:

                    - ``None`` (default) selects all items along axis.
                    - ``int`` selects single item along axis.
                    - ``slice`` with ``step`` setting binning factor
                      and ``step=-1`` integrating all items along axis.

            dtype:
                Unsigned integer type of histogram array.
                The default is ``uint16``.
                Increase to avoid overflows, especially when integrating.
            channel:
                If < 0, integrate channel axis, else select specific channel.
                Overrides ``selection`` for ``C`` axis.
            frame:
                If < 0, integrate time axis, else select specific frame.
                Overrides ``selection`` for ``T`` axis.
            dtime:
                If < 0, integrate time axis, else select specific frame.
                Overrides ``selection`` for ``H`` axis.
            asxarray:
                If true, return histograms as ``xarray.DataArray``,
                else ``numpy.ndarray`` (default).

        Returns:
            :
                Decoded TTTR T3 records as 3-5-dimensional image array:

                - ``'C'`` channel
                - ``'T'`` time/frame
                - ``'Y'`` slow scan axis for image scans
                - ``'X'`` fast scan axis for line and image scans
                - ``'H'`` histogram bins

                Singular ``C``, ``T``, and ``H`` dimensions are not removed.

        Raises:
            NotImplementedError:
                T2 images, bidirectional scanning, sinusoidal correction, and
                deprecated image reconstruction are not supported.
            IndexError:
                Selection is out of bounds.

        """
        # TODO: T2 images
        # TODO: sinusoidal correction
        # TODO: deprecated image reconstruction using
        #   ImgHdr_PixResol, ImgHdr_TStartTo, ImgHdr_TStopTo,
        #   ImgHdr_TStartFro, ImgHdr_TStopFro

        if not self.is_t3:
            raise NotImplementedError('not a T3 image')

        if self.tags.get('ImgHdr_SinCorrection', 0) > 0:
            raise NotImplementedError('sinusoidal correction')
        if self.tags.get('ImgHdr_BiDirect', 0) > 0:
            raise NotImplementedError('bidirectional scanning')
        if self.is_image and 'ImgHdr_LineStart' not in self.tags:
            raise NotImplementedError('old-style image reconstruction')

        shape = list(self.shape)
        ndim = len(shape)

        if selection is None:
            selection = [None] * ndim
        elif len(selection) < ndim:
            selection = list(selection) + [None] * (ndim - len(selection))
        elif len(selection) > ndim:
            raise IndexError('too many indices in selection')
        else:
            selection = list(selection).copy()

        if channel is not None:
            if channel >= shape[0]:
                raise IndexError(f'{channel=} out of range')
            selection[0] = channel if channel >= 0 else slice(None, None, -1)
        if frame is not None:
            if frame >= shape[1]:
                raise IndexError(f'{frame=} out of range')
            selection[1] = frame if frame >= 0 else slice(None, None, -1)
        if dtime is not None:
            if dtime >= shape[-1]:
                raise IndexError(f'{dtime=} out of range')
            selection[-1] = dtime if dtime >= 0 else slice(None, None, -1)

        start = [0] * ndim
        step = [1] * ndim
        for i, (index, size) in enumerate(zip(selection, self.shape)):
            if index is None:
                pass
            elif isinstance(index, int):
                if not 0 <= index < size:
                    raise IndexError(f'axis {i} index out of range')
                start[i] = index
                shape[i] = 1
            elif isinstance(index, slice):
                if index.start is not None:
                    if index.start >= shape[i]:
                        raise IndexError(f'axis {i} slice.start out of range')
                    start[i] = index.start % shape[i]
                if index.stop is not None:
                    if index.stop % shape[i] <= start[i]:
                        raise IndexError(f'axis {i} slice.stop < start')
                    shape[i] = index.stop % shape[i]
                shape[i] -= start[i]
                if index.step is not None:
                    if index.step < 0:
                        # negative step size -> integrate all
                        step[i] = shape[i]
                    else:
                        step[i] = min(index.step, shape[i])
                    shape[i] = shape[i] // step[i] + min(1, shape[i] % step[i])
            else:
                raise IndexError(
                    f'axis {i} index type {type(index)!r} invalid'
                )

        if dtype is None:
            dtype = numpy.uint16
        dtype = numpy.dtype(dtype)
        if dtype.kind != 'u':
            raise ValueError(f'not an unsigned integer {dtype=!r}')

        histogram = numpy.zeros(shape, dtype)
        times = numpy.zeros(shape[1], numpy.uint64)

        from ._ptufile import (
            _decode_t3_image,
            _decode_t3_line,
            _decode_t3_point,
        )

        if ndim == 5:
            _decode_t3_image(
                histogram,
                times,
                self.read_records(),
                self.tags['TTResultFormat_TTTRRecType'],
                self.global_pixel_time,
                self.line_start_mask,
                self.line_stop_mask,
                self.frame_change_mask,
                *start,
                *step,
                self._info.skip_first_frame,
            )
        elif ndim == 4:
            # not tested
            _decode_t3_line(
                histogram,
                times,
                self.read_records(),
                self.tags['TTResultFormat_TTTRRecType'],
                self.global_pixel_time,
                self.line_start_mask,
                self.line_stop_mask,
                *start,
                *step,
            )
        elif ndim == 3:
            _decode_t3_point(
                histogram,
                times,
                self.read_records(),
                self.tags['TTResultFormat_TTTRRecType'],
                self.global_pixel_time,
                *start,
                *step,
            )

        if not asxarray:
            return histogram

        from xarray import DataArray

        coords = self.coords.copy()
        for i, ax in enumerate(self.dims):
            if ax in coords:
                index = slice(start[i], start[i] + shape[i] * step[i], step[i])
                coords[ax] = coords[ax][index]
        coords['T'] = times * self.global_resolution
        attrs = {
            'frequency': self.frequency,
            'max_delaytime': self.number_bins_max,
        }
        return DataArray(
            histogram,
            dims=self.dims,
            coords=coords,
            attrs=attrs,
            # name=self.name,
        )

    def plot(
        self,
        *,
        samples: int | None = None,
        verbose: bool = False,
        show: bool = True,
    ) -> None:
        """Plot histograms using matplotlib.

        Parameters:
            samples:
                Number of bins along measurement for T2 mode.
                The default is 1000.
            verbose:
                Print information about histogram arrays.
            show:
                If true (default), display all figures.
                Else, defer to user or environment to display figures.

        """
        from matplotlib import pyplot
        from tifffile import Timer, imshow

        t = Timer()
        if self.is_t3:
            if (
                self.measurement_submode != PtuMeasurementSubMode.POINT
                and not self.is_bidirectional
            ):
                t.start()
                histogram: Any = self.decode_image(dtime=-1, asxarray=verbose)
                if verbose:
                    print()
                    t.print('decode_image')
                    print()
                    print(histogram)
                imshow(
                    (histogram.values if verbose else histogram).squeeze(),
                    title=repr(self),
                    photometric='minisblack',
                )
                pyplot.figure()
            t.start()
            histogram = self.decode_histogram(asxarray=True)
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
        for i, hist in enumerate(histogram):
            pyplot.plot(hist.coords['H'], hist.values, label=f'ch {i}')
        pyplot.title(repr(self))
        pyplot.xlabel('delay time [s]' if self.is_t3 else 'time [s]')
        pyplot.ylabel('photon count')
        pyplot.ylim(0, None)
        pyplot.legend()
        if show:
            pyplot.show()


@dataclasses.dataclass
class PtuInfo:
    """Information about decoded TTTR records.

    Returned by ``_ptufile._decode_info``.
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
    """Number of frame markers."""

    lines: int
    """Number of lines between line markers."""

    channels: int
    """Maximum number of channels for record type."""

    channels_used: int
    """Highest channel number with photons."""

    bins: int
    """Maximum delay time for record type."""

    bins_used: int
    """Highest delay time observed. Not available for T2 records."""

    skip_first_frame: bool
    """More than two frame markers counted."""

    line_time: int
    """Average global time between line markers."""

    frame_time: int
    """Average global time between frame markers."""

    acquisition_time: int
    """Global time of last sync event."""

    def __str__(self) -> str:
        return '\n'.join(
            (
                f'{self.__class__.__name__}(',
                *(
                    f'    {key}={value},'
                    for key, value in self.__dict__.items()
                ),
                ')',
            )
        )


def indent(*args) -> str:
    """Return joined string representations of objects with indented lines."""
    text = "\n".join(str(arg) for arg in args)
    return "\n".join(
        ("  " + line if line else line) for line in text.splitlines() if line
    )[2:]


def logger() -> logging.Logger:
    """Return logging.getLogger('ptufile')."""
    return logging.getLogger(__name__.replace('ptufile.ptufile', 'ptufile'))


def main(argv: list[str] | None = None) -> int:
    """Command line usage main function.

    Preview image and metadata in specified files or all files in directory.

    ``python -m ptufile file_or_directory``

    """
    from glob import glob

    if argv is None:
        argv = sys.argv

    if len(argv) > 1 and '--doctest' in argv:
        import doctest

        try:
            import ptufile.ptufile as m
        except ImportError:
            m = None  # type: ignore
        doctest.testmod(m, optionflags=doctest.ELLIPSIS)
        return 0

    extensions: set[str] | None = FILE_EXTENSIONS
    if len(argv) == 1:
        files = glob('*.p*')
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
            # enable for debugging
            print(fname, exc)
            continue

    return 0


if __name__ == '__main__':
    sys.exit(main())
