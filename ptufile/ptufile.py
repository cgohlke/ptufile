# ptufile.py

# Copyright (c) 2023-2025, Christoph Gohlke
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

"""Read and write PicoQuant PTU and related files.

Ptufile is a Python library to

1. read data and metadata from PicoQuant PTU and related files
   (PHU, PCK, PCO, PFS, PUS, and PQRES), and
2. write TCSPC histograms to T3 image mode PTU files.

PTU files contain time correlated single photon counting (TCSPC)
measurement data and instrumentation parameters.

:Author: `Christoph Gohlke <https://www.cgohlke.com>`_
:License: BSD 3-Clause
:Version: 2025.1.13
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

- `CPython <https://www.python.org>`_ 3.10.11, 3.11.9, 3.12.8, 3.13.1 64-bit
- `NumPy <https://pypi.org/project/numpy>`_ 2.2.1
- `Xarray <https://pypi.org/project/xarray>`_ 2025.1.1 (recommended)
- `Matplotlib <https://pypi.org/project/matplotlib/>`_ 3.10.0 (optional)
- `Tifffile <https://pypi.org/project/tifffile/>`_ 2025.1.10 (optional)
- `Numcodecs <https://pypi.org/project/numcodecs/>`_ 0.14.1 (optional)
- `Python-dateutil <https://pypi.org/project/python-dateutil/>`_ 2.9.0
  (optional)
- `Cython <https://pypi.org/project/cython/>`_ 3.0.11 (build)

Revisions
---------

2025.1.13

- Fall back to file size if TTResult_NumberOfRecords is zero (#2).

2024.12.28

- Add imwrite function to encode TCSPC image histogram in T3 PTU format.
- Add enums for more PTU tag values.
- Add PqFile.datetime property.
- Read TDateTime tag as datetime instead of struct_time (breaking).
- Rename PtuFile.type property to record_type (breaking).
- Fix reading PHU missing HistResDscr_HWBaseResolution tag.
- Warn if tags are not 8-byte aligned in file.

2024.12.20

- Support bi-directional sinusoidal scanning (WIP).

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

- …

Refer to the CHANGES file for older revisions.

Notes
-----

`PicoQuant GmbH <https://www.picoquant.com/>`_ is a manufacturer of photonic
components and instruments.

The PicoQuant unified file formats are documented at the
`PicoQuant-Time-Tagged-File-Format-Demos
<https://github.com/PicoQuant/PicoQuant-Time-Tagged-File-Format-Demos/tree/master/doc>`_.

The following features are currently not implemented due to the lack of
test files or documentation: PT2 and PT3 files, decoding images from
T2 formats, bidirectional per frame, and deprecated image reconstruction.

Compatibility of written PTU files with other software is limitedly tested,
as are decoding line, bidirectional, and sinusoidal scanning.

Other modules for reading or writing PicoQuant files are
`Read_PTU.py
<https://github.com/PicoQuant/PicoQuant-Time-Tagged-File-Format-Demos/blob/master/PTU/Python/Read_PTU.py>`_,
`readPTU_FLIM <https://github.com/SumeetRohilla/readPTU_FLIM>`_,
`fastFLIM <https://github.com/RobertMolenaar-UT/fastFLIM>`_,
`PyPTU <https://gitlab.inria.fr/jrye/pyptu>`_,
`PTU_Reader <https://github.com/UU-cellbiology/PTU_Reader>`_,
`PTU_Writer <https://github.com/ekatrukha/PTU_Writer>`_,
`FlimReader <https://github.com/flimfit/FlimReader>`_,
`tangy <https://github.com/Peter-Barrow/tangy>`_,
`tttrlib <https://github.com/Fluorescence-Tools/tttrlib>`_,
`picoquantio <https://github.com/tsbischof/picoquantio>`_,
`ptuparser <https://pypi.org/project/ptuparser/>`_,
`phconvert <https://github.com/Photon-HDF5/phconvert/>`_,
`trattoria <https://pypi.org/project/trattoria/>`_ (wrapper of
`trattoria-core <https://pypi.org/project/trattoria-core/>`_,
`tttr-toolbox <https://github.com/GCBallesteros/tttr-toolbox/>`_), and
`napari-flim-phasor-plotter
<https://github.com/zoccoler/napari-flim-phasor-plotter/blob/0.0.6/src/napari_flim_phasor_plotter/_io/readPTU_FLIM.py>`_.

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
>>> ptu.record_type
<PtuRecordType.PicoHarpT3: 66307>
>>> ptu.measurement_mode
<PtuMeasurementMode.T3: 3>
>>> ptu.measurement_submode
<PtuMeasurementSubMode.IMAGE: 3>

Decode TTTR records from the PTU file to ``numpy.recarray``:

>>> decoded = ptu.decode_records()
>>> decoded.dtype
dtype([('time', '<u8'), ('dtime', '<i2'), ('channel', 'i1'), ('marker', 'u1')])

Get global times of frame changes from markers:

>>> decoded['time'][(decoded['marker'] & ptu.frame_change_mask) > 0]
array([1571185680], dtype=uint64)

Decode TTTR records to overall delay-time histograms per channel:

>>> ptu.decode_histogram(dtype='uint8')
array([[ 5,  7,  7, ..., 10,  9,  2]], shape=(2, 3126), dtype=uint8)

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
        [ 47, ..., 30]]],
      shape=(1, 256, 256), dtype=uint16)

Alternatively, decode the first channel and integrate all histogram bins
into a ``xarray.DataArray``, keeping reduced axes:

>>> ptu.decode_image(channel=0, dtime=-1, asxarray=True)
<xarray.DataArray (T: 1, Y: 256, X: 256, C: 1, H: 1)> ...
array([[[[[103]],
           ...
         [[ 30]]]]], shape=(1, 256, 256, 1, 1), dtype=uint16)
Coordinates:
  * T        (T) float64... 0.05625
  * Y        (Y) float64... -0.0001304 ... 0.0001294
  * X        (X) float64... -0.0001304 ... 0.0001294
  * C        (C) uint8... 0
  * H        (H) float64... 0.0
Attributes...
    frequency:      19999200.0
...

Write the TCSPC histogram and metadata to a PicoHarpT3 image mode PTU file:

>>> imwrite(
...     '_test.ptu',
...     ptu[:],
...     ptu.global_resolution,
...     ptu.tcspc_resolution,
...     # optional metadata
...     pixel_time=ptu.pixel_time,
...     record_type=PtuRecordType.PicoHarpT3,
...     comment='Written by ptufile.py',
...     tags={'File_RawData_GUID': [ptu.guid]},
... )

Read back the TCSPC histogram from the file:

>>> tcspc_histogram = imread('_test.ptu')
>>> import numpy
>>> numpy.array_equal(tcspc_histogram, ptu[:])
True

Close the file handle:

>>> ptu.close()

Preview the image and metadata in a PTU file from the console::

    python -m ptufile tests/FLIM.ptu

"""

from __future__ import annotations

__version__ = '2025.1.13'

__all__ = [
    '__version__',
    'imread',
    'imwrite',
    'logger',
    'PqFile',
    'PqFileError',
    'PqFileMagic',
    'PhuFile',
    'PtuFile',
    'PtuWriter',
    'PtuHwFeatures',
    'PtuMeasurementMode',
    'PtuMeasurementSubMode',
    'PtuMeasurementWarnings',
    'PtuRecordType',
    'PtuScanDirection',
    'PtuScannerType',
    'PtuStopReason',
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
import uuid
from datetime import datetime, timedelta
from functools import cached_property
from typing import TYPE_CHECKING, final, overload

if TYPE_CHECKING:
    from collections.abc import Sequence
    from types import EllipsisType
    from typing import IO, Any, Literal

    from numpy.typing import ArrayLike, DTypeLike, NDArray
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


def imwrite(
    file: str | os.PathLike[str] | IO[bytes],
    data: ArrayLike,
    /,
    global_resolution: float,
    tcspc_resolution: float,
    pixel_time: float | None = None,
    *,
    has_frames: bool | None = None,
    record_type: PtuRecordType | None = None,
    pixel_resolution: float | None = None,
    guid: str | uuid.UUID | None = None,
    comment: str | None = None,
    datetime: datetime | None = None,
    tags: dict[str, Any] | None = None,
    mode: Literal['w', 'wb', 'x', 'xb'] | None = None,
) -> None:
    """Write TCSPC histogram to T3 image mode PTU file.

    Parameters:
        file:
            File name or writable binary stream.
        data:
            TCSPC histogram image stack.
            The order of dimensions must be 'TYXCH', 'YXH', 'YXCH',
            or 'TYXH' (with `has_frames=True`).
            The dtype must be unsigned integer.
        global_resolution:
            Resolution of time tags in s, typically in ns range.
            The inverse of the synctime or laser frequency.
            One photon is encoded per time tag.
        tcspc_resolution:
            Resolution of TCSPC in s, typically in ps range.
            The width of a histogram bin.
        pixel_time:
            Time per pixel in s, typically in μs range.
            Photons that cannot be encoded within pixel_time are omitted.
            By default, pixel_time is set just large enough to encode all
            photons.
        has_frames:
            4-dimensional data have frames in first axis ('TYXH'), no channels.
            By default, true if data contains metadata specifying the first
            dimension is 'T', else false.
        record_type, pixel_resolution, guid, comment, datetime, tags, mode:
            Optional parameters passed to :py:class:`PtuWriter`.

    """
    if hasattr(data, 'dims'):
        has_frames = 'T' in data.dims and data.dims[0] == 'T'

    data = numpy.asarray(data)

    if pixel_time is None:
        data = data.reshape(PtuWriter.normalize_shape(data.shape, has_frames))
        pixel_time = global_resolution * max(
            1, float(numpy.max(data.sum(axis=(3, 4), dtype=numpy.uint64)))
        )

    with PtuWriter(
        file,
        data.shape,
        global_resolution,
        tcspc_resolution,
        pixel_time,
        record_type=record_type,
        pixel_resolution=pixel_resolution,
        has_frames=has_frames,
        guid=guid,
        comment=comment,
        datetime=datetime,
        tags=tags,
        mode=mode,
    ) as ptu:
        ptu.write(data)


class PtuWriter:
    """Write TCSPC histogram to T3 image mode PTU file.

    T3 TTTR records allow for a maximum of 63 channels and 32768 bins.
    The TTTR records written can only be used to reconstruct the encoded
    TCSPC histogram image stack, not for higher-than-pixel-time-resolution
    intensity time-trace or correlation analysis.

    Parameters:
        file:
            File name or writable binary stream.
            File names typically end in '.PTU'.
        shape:
            Shape of TCSPC histogram image stack two write.
            The order of dimensions must be 'TYXCH', 'YXH', 'YXCH',
            or 'TYXH' (with `has_frames=True`).
        global_resolution:
            Resolution of time tags in s, typically in ns range.
            The inverse of the synctime or laser frequency.
            One photon is encoded per time tag.
        tcspc_resolution:
            Resolution of TCSPC in s, typically in ps range.
            The width of a histogram bin.
        pixel_time:
            Time per pixel in s, typically in μs range.
            Photons that cannot be encoded within pixel_time are omitted.
        record_type:
            Type of TTTR T3 records to write.
            By default, write ``PicoHarpT3`` records for up to two channels
            and 4096 bins, else ``GenericT3``.
        pixel_resolution:
            Resolution of single pixel in μm. The default is 1 μm.
        has_frames:
            4-dimensional shape has frames in first axis ('TYXH'), no channels.
        guid:
            Windows formatted GUID used as global file identifier.
            By default, a random GUID. Write to File_GUID tag.
        comment:
            File comment. Write to File_Comment tag.
        datetime:
            File creation date and time.
            The default is time at function call.
            Write to File_CreatingTime tag.
        tags:
            Additional tag Id and values to write.
            Critical tags are automatically set and cannot be modified.
            No validation is performed.
            Refer to the "PicoQuant Unified Tag Dictionary" for valid Id and
            values.
        mode:
            Binary file open mode if `file` is file name.
            The default is 'w', which opens files for writing, truncating
            existing files.
            'x' opens files for exclusive creation, failing on existing files.

    Raises:
        ValueError
            Not ``0 < tcspc_resolution <= global_resolution <= pixel_time``.

    """

    _fh: IO[bytes] | None
    _shape: tuple[int, int, int, int, int]
    _record_type: PtuRecordType
    _number_records: int
    _number_records_offset: int
    _number_frames: int
    _number_frames_offset: int
    _global_resolution: float
    _tcspc_resolution: float
    _pixel_time: int

    _line_start = 1
    _line_stop = 2
    _frame_change = 3

    def __init__(
        self,
        file: str | os.PathLike[str] | IO[bytes],
        /,
        shape: tuple[int, ...],
        global_resolution: float,
        tcspc_resolution: float,
        pixel_time: float,
        *,
        record_type: PtuRecordType | None = None,
        pixel_resolution: float | None = None,
        has_frames: bool | None = None,
        guid: str | uuid.UUID | None = None,
        comment: str | None = None,
        datetime: datetime | None = None,
        tags: dict[str, Any] | None = None,
        mode: Literal['w', 'wb', 'x', 'xb'] | None = None,
    ) -> None:
        """Write PTU header to file."""
        # 0 < tcspc_resolution <= global_resolution <= pixel_time
        if tcspc_resolution <= 0.0:
            raise ValueError(f'{tcspc_resolution=} <= 0.0')
        if tcspc_resolution > global_resolution:
            raise ValueError(f'{tcspc_resolution=} > {global_resolution=}')
        if pixel_time < global_resolution:
            raise ValueError(f'{pixel_time=} < {global_resolution=}')

        self._fh = None
        self._number_records = 0
        self._number_records_offset = 0
        self._number_frames = 0
        self._number_frames_offset = 0
        self._global_resolution = global_resolution
        self._tcspc_resolution = tcspc_resolution
        self._pixel_time = int(round(pixel_time / global_resolution))
        self._shape = shape = PtuWriter.normalize_shape(shape, has_frames)

        if record_type is None:
            if shape[3] <= 2 and shape[4] <= 4096:
                record_type = PtuRecordType.PicoHarpT3
            else:
                record_type = PtuRecordType.GenericT3

        if record_type == PtuRecordType.PicoHarpT3:
            if shape[3] > 4 or shape[4] > 4096:
                raise ValueError(
                    f'{record_type=} does not support '
                    f'{shape[3]} channels and {shape[4]} bins'
                )
            self._record_type = PtuRecordType.PicoHarpT3
        elif record_type in {
            PtuRecordType.GenericT3,
            PtuRecordType.HydraHarp2T3,
            PtuRecordType.TimeHarp260NT3,
            PtuRecordType.TimeHarp260PT3,
        }:
            if shape[3] > 63 or shape[4] > 32768:
                raise ValueError(
                    f'{record_type=} does not support '
                    f'{shape[3]} channels and {shape[4]} bins'
                )
            self._record_type = PtuRecordType.GenericT3
        else:
            raise ValueError(f'{record_type=} not supported')

        if comment is None:
            comment = ''

        if guid is None:
            guid = f'{{{uuid.uuid4()}}}'
        elif isinstance(guid, uuid.UUID):
            guid = f'{{{guid}}}'
        elif len(guid) != 38 or guid[9] != '-':
            raise ValueError('invalid GUID')

        if pixel_resolution is None:
            pixel_resolution = 1.0
        elif pixel_resolution <= 0.0:
            raise ValueError(f'{pixel_resolution=} <= 0.0')

        if datetime is None:
            datetime = now()

        critical_tags = {
            # tags not to be overwritten by user
            'Measurement_Mode': PtuMeasurementMode.T3,
            'Measurement_SubMode': PtuMeasurementSubMode.IMAGE,
            'MeasDesc_GlobalResolution': float(self._global_resolution),
            'MeasDesc_Resolution': float(self._tcspc_resolution),
            'MeasDesc_BinningFactor': 1,
            'TTResult_NumberOfRecords': self._number_records,
            'TTResult_SyncRate': int(round(1.0 / self._global_resolution)),
            'TTResultFormat_TTTRRecType': self._record_type,
            'TTResultFormat_BitsPerRecord': 32,
            'ImgHdr_Dimensions': 3,
            'ImgHdr_Ident': PtuScannerType.LSM,
            'ImgHdr_LineStart': self._line_start,
            'ImgHdr_LineStop': self._line_stop,
            'ImgHdr_Frame': self._frame_change,
            'ImgHdr_MaxFrames': 0,
            'ImgHdr_TimePerPixel': pixel_time * 1e3,  # ms
            'ImgHdr_PixX': int(self._shape[2]),
            'ImgHdr_PixY': int(self._shape[1]),
            'ImgHdr_BiDirect': False,
            'ImgHdr_SinCorrection': 0,
        }
        pqtags = {
            # required tags written first, allowed to be overwritten by user
            'File_GUID': guid,
            'File_Comment': comment,
            'File_CreatingTime': datetime,
            'CreatorSW_Name': 'ptufile.py',
            'CreatorSW_Version': __version__,
        }
        pqtags.update(critical_tags)
        pqtags.update(
            # other tags allowed to be overwritten by user
            {
                'ImgHdr_PixResol': float(pixel_resolution),
                'HW_InpChannels': self._shape[3] + 1,  # used by FlimReader
                # 'TTResult_StopReason': PtuStopReason(0)
            }
        )
        if tags is not None:
            # add user tags but do not overwrite critical tags
            pqtags.update(
                {k: v for k, v in tags.items() if k not in critical_tags}
            )

        header_list = [b'PQTTTR\x00\x001.0.00\x00\x00']  # magic and version
        for tagid, value in pqtags.items():
            if isinstance(value, (list, tuple)):
                for index, item in enumerate(value):
                    header_list.append(encode_tag(tagid, item, index))
            else:
                header_list.append(encode_tag(tagid, value))
        header_list.append(encode_tag('Header_End', None))

        header = b''.join(header_list)
        offset = header.find(b'TTResult_NumberOfRecords')
        assert offset > 0
        self._number_records_offset = offset + 40

        offset = header.find(b'ImgHdr_MaxFrames')
        assert offset > 0
        self._number_frames_offset = offset + 40

        if isinstance(file, (str, os.PathLike)):
            if mode is None:
                mode = 'wb'
            elif mode[-1] != 'b':
                mode += 'b'  # type: ignore[assignment]
            self._fh = open(file, mode)
            self._close = True
        elif hasattr(file, 'write') and hasattr(file, 'seek'):
            self._fh = file
            self._close = False
        else:
            raise ValueError(f'cannot write to {type(file)=}')

        self._fh.write(header)

    def write(self, data: ArrayLike, /) -> None:
        """Append T3 encoded TCSPC histogram to file.

        Parameters:
            data:
                TCSPC histogram image stack.
                The shape must be compatible with the shape passed to
                PtuWriter(). The dtype must be unsigned integer.

        """
        from ._ptufile import encode_t3_image

        data = numpy.asarray(data)
        if data.dtype.kind != 'u':
            raise ValueError(f'{data.dtype=} is not an unsigned integer')
        data = data.reshape(self._shape)

        number_photons = int(data.sum(dtype=numpy.uint64))

        if self._record_type == PtuRecordType.PicoHarpT3:
            maxtime = 65536
        else:
            maxtime = 1024

        shape = data.shape
        number_records = (
            number_photons
            # overflows assuming all empty pixels
            + (shape[0] * shape[1] * shape[2] * self._pixel_time) // maxtime
            # line markers
            + shape[0] * shape[1] * 2
            # frame markers
            + shape[0]
        )
        records = numpy.zeros(number_records, dtype=numpy.uint32)

        number_records = encode_t3_image(
            records,
            data,
            self._record_type,
            self._pixel_time,
            int(2 ** (self._line_start - 1)),
            int(2 ** (self._line_stop - 1)),
            int(2 ** (self._frame_change - 1)),
        )
        if number_records < 0:
            raise ValueError(f'{records.size=} too small')

        assert self._fh is not None
        self._fh.write(records[:number_records].tobytes())

        self._number_records += number_records
        self._number_frames += shape[0]

    def close(self) -> None:
        """Close file handle after writing final tag values."""
        if self._fh is None:
            return

        if self._number_records_offset > 0:
            self._fh.seek(self._number_records_offset)
            self._fh.write(struct.pack('<q', self._number_records))
            self._fh.seek(self._number_frames_offset)
            self._fh.write(struct.pack('<q', self._number_frames))

        if self._close:
            try:
                self._fh.close()
            except Exception:
                pass
        self._fh = None

    def __enter__(self) -> PtuWriter:
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.close()

    @staticmethod
    def normalize_shape(
        shape: tuple[int, ...], has_frames: bool | None = None, /
    ) -> tuple[int, int, int, int, int]:
        """Return TCSPC histogram shape normalized to 5D 'TYXCH'."""
        ndim = len(shape)
        if ndim == 5:
            'TYXCH'
            return shape  # type: ignore[return-value]
        if ndim == 3:
            # 'YXH'
            return (1, shape[0], shape[1], 1, shape[2])
        if ndim == 4:
            if has_frames:
                # 'TYXH'
                return (shape[0], shape[1], shape[2], 1, shape[3])
            # 'YXCH'
            return (1, shape[0], shape[1], shape[2], shape[3])

        raise ValueError(f'invalid number of dimensions {len(shape)=}')


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


class PqFileError(Exception):
    """Exception to indicate invalid PicoQuant tagged file structure."""


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
    _number_records_offset: int  # position of TTResult_NumberOfRecords value

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
            raise ValueError(f'cannot open {type(file)=}')

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
            )[:160]

        tagid: str
        index: int
        typecode: int
        value: Any
        unpack = struct.unpack
        try:
            while True:
                offset = fh.tell()
                tagid_, index, typecode, value = unpack(
                    '<32siI8s', fh.read(48)
                )
                # print(tagid.strip(b'\0'), index, typecode, value)
                tagid = tagid_.rstrip(b'\0').decode('ascii', errors='ignore')

                # tags must start on positions divisible by 8
                # disabled for PQRES
                if offset % 8 and self.magic != PqFileMagic.PQRES:
                    logger().error(
                        errmsg(
                            f'tag {offset=} not divisible by 8',
                            tagid,
                            index,
                            typecode,
                            value,
                        )
                    )
                if tagid == 'Header_End':
                    break
                if tagid == 'Fast_Load_End':
                    if fastload:
                        break
                    continue

                # TODO: use dict to dispatch?
                # frequent typecodes
                if typecode == PqTagType.Int8:
                    value = unpack('<q', value)[0]
                elif typecode == PqTagType.Bool8:
                    value = bool(unpack('<q', value)[0])
                elif typecode == PqTagType.Float8:
                    value = unpack('<d', value)[0]
                elif typecode == PqTagType.AnsiString:
                    size = unpack('<q', value)[0]
                    value = (
                        fh.read(size)
                        .rstrip(b'\0')
                        .decode('windows-1252', errors='ignore')
                    )
                elif typecode == PqTagType.Empty8:
                    value = None
                elif typecode == PqTagType.TDateTime:
                    value = unpack('<d', value)[0]
                    value = datetime(1899, 12, 30) + timedelta(days=value)

                # rarer typecodes
                elif typecode == PqTagType.WideString:
                    size = unpack('<q', value)[0]
                    value = fh.read(size).decode('utf-16-le').rstrip('\0')
                elif typecode == PqTagType.BinaryBlob:
                    size = unpack('<q', value)[0]
                    value = fh.read(size)
                    if tagid == 'ChkHistogram':
                        value = numpy.frombuffer(value, dtype=numpy.int64)
                elif typecode == PqTagType.BitSet64:
                    value = unpack('<q', value)[0]
                elif typecode == PqTagType.Color8:
                    # TODO: unpack to RGB triple
                    value = unpack('<q', value)[0]
                elif typecode == PqTagType.Float8Array:
                    size = unpack('<q', value)[0]
                    value = unpack(f'<{size // 8}d', fh.read(size))
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
        if hasattr(self._fh, 'name') and self._fh.name:
            return self._fh.name
        return repr(self._fh)

    @property
    def guid(self) -> uuid.UUID:
        """Global identifier of file."""
        return uuid.UUID(self.tags['File_GUID'])

    @property
    def comment(self) -> str | None:
        """File comment, if any."""
        return self.tags.get('File_Comment')

    @property
    def datetime(self) -> datetime | None:
        """File creation date, if any."""
        if 'File_CreatingTime' not in self.tags:
            return None
        value = self.tags['File_CreatingTime']
        if isinstance(value, datetime):
            return value
        if isinstance(value, str):
            try:
                return datetime.fromisoformat(value)
            except ValueError:
                try:
                    from dateutil import parser

                    return parser.parse(value)
                except Exception:
                    return None
        if not isinstance(value, float):
            return None
        try:
            return datetime(1899, 12, 30) + timedelta(days=value)
        except Exception:
            return None

    def close(self) -> None:
        """Close file handle and free resources."""
        if self._close:
            try:
                self._fh.close()
            except Exception:
                pass

    def __enter__(self) -> PqFile:
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
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
        ncurves = self.number_histograms
        if 'HistResDscr_HWBaseResolution' in self.tags:
            return tuple(self.tags['HistResDscr_HWBaseResolution'])
        if 'HW_BaseResolution' in self.tags:
            return tuple([float(self.tags['HW_BaseResolution'])] * ncurves)
        return None

    @property
    def number_histograms(self) -> int:
        """Number of histograms stored in file."""
        return int(self.tags['HistoResult_NumberOfCurves'])

    # @property
    # def bits_per_bin(self) -> int:
    #     """Number of bits per histogram bin."""
    #     return int(self.tags.get('HistoResult_BitsPerBin', 32))

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
        for offset, nbins in zip(
            self.tags['HistResDscr_DataOffset'][index],
            self.tags['HistResDscr_HistogramBins'][index],
        ):
            self._fh.seek(offset)
            histograms.append(
                numpy.fromfile(self._fh, dtype='<u4', count=nbins)
            )
        if asxarray:
            from xarray import DataArray

            assert self.histogram_resolutions is not None

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
                for h, r in zip(histograms, self.histogram_resolutions)
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
        'record_type',
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
    def record_offset(self) -> int:
        """Position of records in file."""
        return self._data_offset

    @property
    def record_type(self) -> PtuRecordType:
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
            # and self.tags.get('ImgHdr_PixY', 1) > 1  # may be missing
        ):
            return 3
        if (
            submode == 2
            and self.tags.get('ImgHdr_Dimensions', 2) == 2  # optional
            # and self.tags.get('ImgHdr_PixX', 1) > 1  # may be missing
        ):
            # TODO: need linescan test file
            return 2
        return 1

    @property
    def measurement_warnings(self) -> PtuMeasurementWarnings | None:
        """Warnings during measurement, or None if not specified."""
        if 'TTResult_MDescWarningFlags' in self.tags:
            return PtuMeasurementWarnings(
                self.tags['TTResult_MDescWarningFlags']
            )
        return None

    @property
    def hardware_features(self) -> PtuHwFeatures | None:
        """Hardware features, or None if not specified."""
        if 'HW_Features' in self.tags:
            return PtuHwFeatures(self.tags['HW_Features'])
        return None

    @property
    def stop_reason(self) -> PtuStopReason | None:
        """Reason for measurement end, or None if not specified."""
        if 'TTResult_StopReason' in self.tags:
            return PtuStopReason(self.tags['TTResult_StopReason'])
        return None

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

    @cached_property
    def number_records(self) -> int:
        """Number of TTTR records."""
        count = int(self.tags.get('TTResult_NumberOfRecords', 0))
        if count == 0:
            count = (self._fh.seek(0, os.SEEK_END) - self._data_offset) // 4
            if count != 0:
                logger().warning(
                    f'{self!r} TTResult_NumberOfRecords is zero. '
                    'Using remaining file content as records'
                )
        return count

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
        return max(1, int(round(pixeltime)))

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
            # TODO: Warning from the PicoQuant documentation:
            # Attention, in some images one will find a different number
            # of lines than defined by PixY (less or more, even different
            # in every frame), so do not trust this value.
            return max(1, int(self.tags['ImgHdr_PixY']))
        return 1

    @property
    def pixel_time(self) -> float:
        """Time per pixel in s."""
        pixel_time = float(self.tags.get('ImgHdr_TimePerPixel', 0.0))
        if pixel_time > 0.0:
            return pixel_time * 1e-3  # ms to s
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
        count = self.number_records
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
            image:
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
        # TODO: support ReqHdr_ScanningPattern = 1, bidirectional per frame
        if not self.is_t3:
            # TODO: T2 images
            raise NotImplementedError('not a T3 image')
        if self.is_bidirectional and not self.is_image:
            raise NotImplementedError(
                'bidirectional scanning only supported for images'
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
            pixel_at_time = sinusoidal_correction(
                self.tags['ImgHdr_SinCorrection'],
                self.global_line_time,
                self.pixels_in_line,
                dtype=numpy.uint16,  # should be enough for pixels_in_line
            )
        else:
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
                self.global_pixel_time,
                self.global_line_time,
                pixel_at_time,
                self.line_start_mask,
                self.line_stop_mask,
                self.frame_change_mask,
                *start,
                *step,
                0 if bishift is None else bishift,
                self.is_bidirectional,
                self.is_sinusoidal,
                self._info.skip_first_frame,
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
                By default, all frames are shown. Applies to T3 images.
            channel:
                If < 0, integrate channel axis, else show specified channel.
                By default, all channels are shown. Applies to T3 images.
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


class PhuMeasurementMode(enum.IntEnum):
    """Kind of TCSPC measurement (Measurement_Mode tag)."""

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
    """Kind of measurement (Measurement_SubMode tag)."""

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
    """Kind of TCSPC Measurement (Measurement_Mode tag)."""

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
    """Kind of measurement (Measurement_SubMode tag)."""

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
    """Scanner hardware (ImgHdr_Ident tag)."""

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
    """Scan direction (ImgHdr_ScanDirection tag)."""

    XY = 0
    """X-Y scan."""

    XZ = 1
    """X-Z scan."""

    YZ = 2
    """Y-Z scan."""

    @classmethod
    def _missing_(cls, value: object) -> object:
        if not isinstance(value, int) or value != 0:
            return None
        obj = cls(0)  # XY
        obj._value_ = value
        return obj


class PtuStopReason(enum.IntEnum):
    """Reason for measurement end (TTResult_StopReason tag)."""

    TIME_OVER = 0
    MANUAL = 1
    OVERFLOW = 2
    ERROR = 3
    UNKNOWN = -1
    FIFO_OVERRUN = -2
    LEGACY_ERROR = -3
    TCSPC_ERROR = -4
    FILE_ERROR = -5
    OUT_OF_MEMORY = -6
    SUSPENDED = -7
    SYS_ERROR = -8
    QUEUE_OVERRUN = -9
    DATA_XFER_FAIL = -10
    DATA_CHECK_FAIL = -11
    REF_CLK_LOST = -12
    SYNC_LOST = -13

    @classmethod
    def _missing_(cls, value: object) -> object:
        if not isinstance(value, int):
            return None
        obj = cls(-1)  # Unknown
        obj._value_ = value
        return obj


class PtuMeasurementWarnings(enum.IntFlag):
    """Warnings during measurement (TTResult_MDescWarningFlags tag)."""

    SYNC_RATE_ZERO = 0x1
    SYNC_RATE_TOO_LOW = 0x2
    SYNC_RATE_TOO_HIGH = 0x4
    INPT_RATE_ZERO = 0x10
    INPT_RATE_TOO_HIGH = 0x40
    EVENTS_DROPPED = 0x80
    INPT_RATE_RATIO = 0x100
    DIVIDER_GT_ONE = 0x200
    TIME_SPAN_TOO_SMALL = 0x400
    OFFSET_UNNECESSARY = 0x800


class PtuHwFeatures(enum.IntFlag):
    """Hardware features (HW_Features tag)."""

    DLL = 0x1
    TTTR = 0x2
    MARKERS = 0x4
    LOW_RES = 0x8
    TRIG_OUT = 0x10
    PROG_DEADTIME = 0x20
    EXT_FPGA = 0x40
    PROG_HYSTERESES = 0x80
    COINCIDENCE_FILTERING = 0x100
    INPUT_MODES = 0x200


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
    """PicoHarp 300 T3."""

    PicoHarpT2 = 0x00010203
    """PicoHarp 300 T2."""

    HydraHarpT3 = 0x00010304
    """HydraHarp V1.x T3."""

    HydraHarpT2 = 0x00010204
    """HydraHarp V1.x T2."""

    HydraHarp2T3 = 0x01010304
    """HydraHarp V2.x T3."""

    HydraHarp2T2 = 0x01010204
    """HydraHarp V2.x T2."""

    TimeHarp260NT3 = 0x00010305
    """TimeHarp 260N T3."""

    TimeHarp260NT2 = 0x00010205
    """TimeHarp 260N T2."""

    TimeHarp260PT3 = 0x00010306
    """TimeHarp 260P T3."""

    TimeHarp260PT2 = 0x00010206
    """TimeHarp 260P T2."""

    GenericT2 = 0x00010207
    """MultiHarp and Picoharp 330 T2."""

    GenericT3 = 0x00010307
    """MultiHarp and Picoharp 330 T3."""


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


def encode_tag(tagid: str, value: Any, index: int = -1, /) -> bytes:
    """Return encoded PqTag."""
    if value is None:
        typecode = PqTagType.Empty8
        buffer = b'\x00\x00\x00\x00\x00\x00\x00\x00'
    elif isinstance(value, bool):
        # must check bool before int
        typecode = PqTagType.Bool8
        buffer = struct.pack('<q', value)
    elif isinstance(value, enum.IntFlag):
        # must check IntFlag before int
        typecode = PqTagType.BitSet64
        buffer = struct.pack('<q', value)
    elif isinstance(value, int):
        typecode = PqTagType.Int8
        buffer = struct.pack('<q', value)
    elif isinstance(value, float):
        typecode = PqTagType.Float8
        buffer = struct.pack('<d', value)
    elif isinstance(value, str):
        if not value.endswith('\0'):
            value += '\0'
        try:
            typecode = PqTagType.AnsiString
            value_ = value.encode('windows-1252')
        except UnicodeEncodeError:
            typecode = PqTagType.WideString
            value_ = value.encode('utf-16-le')
        value_ += align_bytes(len(value_), 8)
        buffer = struct.pack('<q', len(value_)) + value_
    elif isinstance(value, datetime):
        typecode = PqTagType.TDateTime
        value -= datetime(1899, 12, 30)
        value /= timedelta(days=1)
        buffer = struct.pack('<d', value)
    elif isinstance(value, uuid.UUID):
        typecode = PqTagType.AnsiString
        value = f'{{{value}}}'.encode('windows-1252')
        value += align_bytes(len(value), 8)
        buffer = struct.pack('<q', len(value)) + value
    elif isinstance(value, bytes):
        typecode = PqTagType.BinaryBlob
        value += align_bytes(len(value), 8)
        buffer = struct.pack('<q', len(value)) + value
    else:
        # TODO: support Color8 and Float8Array
        raise ValueError(f'{type(value)=} not supported')

    # tags always have lengths divisible by 8
    assert len(buffer) % 8 == 0
    return struct.pack('<32siI', tagid.encode(), index, typecode) + buffer


def sinusoidal_correction(
    sincorrect: float,
    /,
    global_line_time: int,
    pixels_in_line: int,
    *,
    dtype: DTypeLike = None,
    is_amplitude: bool = True,
) -> NDArray[Any]:
    """Return pixel indices of global times in line for sinusoidal scanning.

    Parameters:
        sincorrect:
            Amount of sine wave used for measurement.
            Either percentage of amplitude (PicoQuant) or period (Leica)
            depending on `is_amplitude`.
            The value of the `ImgHdr_SinCorrection` tag.
        global_line_time:
            Global time per line.
        pixels_in_line:
            Number of pixels in line.
        is_amplitude:
            Correction value is percentage of amplitude of sine wave.
            If false, correction value is percentage of period of sine wave.

    Returns:
        Array of size `global_line_time`, mapping global time in line to
        pixel index in line.

    """
    dtype = numpy.dtype(numpy.uint16 if dtype is None else dtype)
    if sincorrect <= 0.0 or sincorrect > 100.0:
        raise ValueError(f'{sincorrect=} out of range')
    if global_line_time < 2:
        raise ValueError(f'{global_line_time=} out of range')
    if pixels_in_line < 2 or pixels_in_line >= numpy.iinfo(dtype).max:
        raise ValueError(f'{pixels_in_line=} out of range')
    if not is_amplitude:
        sincorrect = math.sin(sincorrect * math.pi / 200.0) * 100.0
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


def now() -> datetime:
    """Return current date and time."""
    return datetime.now()


def align_bytes(size: int, align: int, /) -> bytes:
    """Return trailing bytes to align bytes of size."""
    size %= align
    return b'' if size == 0 else b'\0' * (align - size)


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
