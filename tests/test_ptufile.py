# test_ptufile.py

# Copyright (c) 2023-2026, Christoph Gohlke
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

"""Unittests for the ptufile package.

:Version: 2026.11.5

"""

import contextlib
import datetime
import glob
import io
import itertools
import logging
import mmap
import os
import pathlib
import sys
import sysconfig
import tempfile
import types
import uuid
from typing import Any

import numpy
import pytest
from numpy import array_equal  # faster than numpy.testing.assert_array_equal
from numpy.testing import assert_allclose, assert_almost_equal

import ptufile
import ptufile.numcodecs
from ptufile import (
    FILE_EXTENSIONS,
    T2_RECORD_DTYPE,
    T3_RECORD_DTYPE,
    PhuFile,
    PhuMeasurementMode,
    PhuMeasurementSubMode,
    PqFile,
    PqFileError,
    PqFileType,
    PtuFile,
    PtuMeasurementMode,
    PtuMeasurementSubMode,
    PtuMeasurementWarnings,
    PtuRecordType,
    PtuScannerType,
    PtuStopReason,
    PtuWriter,
    __version__,
    binread,
    binwrite,
    imread,
    imwrite,
)

try:
    import fsspec
except ImportError:
    fsspec = None  # type: ignore[assignment]

try:
    import xarray
except ImportError:
    xarray = None  # type: ignore[assignment]

pyplot: types.ModuleType | None
if os.environ.get('PTUFILE_DISABLE_PYPLOT'):
    pyplot = None
else:
    try:
        from matplotlib import pyplot
    except ImportError:
        pyplot = None


RNG = numpy.random.default_rng(42)

DATA = pathlib.Path(os.path.dirname(__file__)) / 'data'

FILES = [
    # PicoHarpT3
    'napari_flim_phasor_plotter/hazelnut_FLIM_single_image.ptu',
    # PicoHarpT2
    'Samples.sptw/Atto488_diff_cw_total_correlation.ptu',
    # HydraHarpT3
    'picoquant-sample-data/hydraharp/v10_t3.ptu',
    # HydraHarpT2
    'Samples.sptw/NV-Center_for_Antibunching_several.ptu',
    # HydraHarp2T3
    'tttr-data/imaging/pq/Microtime200_HH400/beads.ptu',
    # HydraHarp2T2
    'tttr-data/pq/ptu/pq_ptu_hh_t2_test2.ptu',
    # TODO: TimeHarp260NT3
    # TimeHarp260NT2
    'ptuparser/default_007.ptu',
    # TimeHarp260PT3
    'tttr-data/imaging/pq/Microtime200_TH260/beads.ptu',
    # TODO: TimeHarp260PT2
    # TODO: GenericT2/MultiHarpT2 or Picoharp330T2
    # GenericT3/MultiHarpT3
    'Tutorials.sptw/Hyperosmotic_Shock_MDCK_Cells.ptu',
]


@pytest.mark.skipif(__doc__ is None, reason='__doc__ is None')
def test_version() -> None:
    """Assert ptufile versions match docstrings."""
    ver = ':Version: ' + __version__
    assert ver in __doc__
    assert ver in ptufile.__doc__


def test_import_xarray() -> None:
    """Assert xarray is installed."""
    assert xarray is not None


def test_import_matplotlib() -> None:
    """Assert matplotlib is installed."""
    if os.environ.get('PTUFILE_DISABLE_PYPLOT') == '1':
        assert pyplot is None
    else:
        assert pyplot is not None


@pytest.mark.parametrize('memmap', [False, True])
def test_binread(*, memmap: bool) -> None:
    """Test read and write PicoQuant BIN file."""
    filename = DATA / 'UNC/805.bin'
    data, attrs = binread(filename, memmap=memmap)
    assert attrs['shape'] == (256, 256, 2000)
    assert attrs['pixel_resolution'] == 0.078125
    assert attrs['tcspc_resolution'] == 2.5000000372529032e-11
    assert data.shape == (256, 256, 2000)
    assert data.dtype == numpy.uint32
    assert numpy.sum(data) == 43071870

    with tempfile.NamedTemporaryFile(suffix='.bin', delete=False) as tmp:
        fnout = tmp.name

    try:
        binwrite(
            fnout,
            data,
            attrs['tcspc_resolution'],
            pixel_resolution=attrs['pixel_resolution'],
        )
        del data

        with open(filename, 'rb') as fh:
            data1 = fh.read()
        with open(fnout, 'rb') as fh:
            data2 = fh.read()
        assert data1 == data2
    finally:
        if os.path.exists(fnout):
            os.remove(fnout)


def test_non_pqfile() -> None:
    """Test read non-PicoQuant file fails."""
    filename = DATA / 'FRET_GFP and mRFP.pt3'
    with pytest.raises(PqFileError):  # noqa: SIM117
        with PqFile(filename):
            pass


def test_non_ptu() -> None:
    """Test read non-PTU file fails."""
    filename = DATA / 'Settings.pfs'
    with pytest.raises(PqFileError):  # noqa: SIM117
        with PtuFile(filename):
            pass


def test_pq_fastload() -> None:
    """Test read tags using fastload."""
    filename = (
        DATA / 'napari_flim_phasor_plotter/hazelnut_FLIM_single_image.ptu'
    )
    with PqFile(filename, fastload=True) as pq:
        str(pq)
        assert pq.tags['File_GUID'] == '{4f6e5f68-8289-483d-9d9a-7974b77ef8b8}'
        assert 'HW_ExternalRefClock' not in pq.tags


def test_pck() -> None:
    """Test read PCK file."""
    filename = DATA / 'Tutorials.sptw/IRF_Fluorescein.pck'
    with PqFile(filename) as pq:
        str(pq)
        assert pq.type == PqFileType.PCK
        assert pq.version == '1.0.00'
        assert pq.tags['File_Comment'].startswith('Check point file of ')
        assert array_equal(
            pq.tags['ChkHistogram'][:6], [96, 150, 151, 163, 153, 145]
        )
        attrs = pq.attrs
        assert attrs['type'] == pq.type.name
        assert attrs['name'] == pq.name
        assert attrs['tags'] == pq.tags


def test_pco() -> None:
    """Test read PCO file."""
    filename = DATA / 'Tutorials.sptw/Hyperosmotic_Shock_MDCK_Cell.pco'
    with PqFile(filename) as pq:
        str(pq)
        assert pq.type == PqFileType.PCO
        assert pq.version == '1.0.00'
        assert pq.tags['CreatorSW_Modules'] == 0
        attrs = pq.attrs
        assert attrs['type'] == pq.type.name
        assert attrs['name'] == pq.name
        assert attrs['tags'] == pq.tags


def test_pfs() -> None:
    """Test read PFS file."""
    filename = DATA / 'Settings.pfs'
    with PqFile(filename) as pq:
        str(pq)
        assert pq.type == PqFileType.PFS
        assert pq.version == '1.0.00'
        assert pq.tags['HW_SerialNo'] == '<SerNo. empty>'
        assert pq.tags['Defaults_Begin'] is None
        attrs = pq.attrs
        assert attrs['type'] == pq.type.name
        assert attrs['name'] == pq.name
        assert attrs['tags'] == pq.tags


def test_pqres(caplog: pytest.LogCaptureFixture) -> None:
    """Test read PQRES file."""
    filename = DATA / 'Samples.sptw/AnisotropyImage.pqres'
    with caplog.at_level(logging.ERROR):
        with PqFile(filename) as pq:
            str(pq)
            # assert 'not divisible by 8' in caplog.text
            assert pq.type == PqFileType.PQRES
            assert pq.version == '00.0.1'
            assert pq.tags['VarStatFilterGrpIdx'].startswith(b'\xe7/\x00\x00')
        attrs = pq.attrs
        assert attrs['type'] == pq.type.name
        assert attrs['name'] == pq.name
        assert attrs['tags'] == pq.tags


def test_pqdat(caplog: pytest.LogCaptureFixture) -> None:
    """Test read PQDAT file."""
    filename = DATA / 'Luminosa/FRET_20230606-185222/FittedCurveIRF.pqdat'
    with caplog.at_level(logging.ERROR), PqFile(filename) as pq:
        str(pq)
        # assert 'not divisible by 8' in caplog.text
        assert pq.type == PqFileType.PQDAT
        assert pq.version == '1.0.00'
        assert len(pq.tags['LSDCurveX']) == 1254
        assert len(pq.tags['LSDCurveY']) == 1254
        preview = numpy.frombuffer(
            pq.tags['PreviewImage'][4:], dtype=numpy.uint8
        ).reshape((128, 128, 4))
        assert preview.shape == (128, 128, 4)
        attrs = pq.attrs
        assert attrs['type'] == pq.type.name
        assert attrs['name'] == pq.name
        assert attrs['tags'] == pq.tags


def test_pquni() -> None:
    """Test read PQUNI file."""
    # TODO need better PqUni test data
    filename = DATA / 'UniHarp/MicroBeads.PqUni'
    with PqFile(filename) as pq:
        str(pq)
        # assert 'not divisible by 8' in caplog.text
        assert pq.type == PqFileType.PQUNI
        assert pq.version == '1.0.0.0'
        assert pq.comment is None
        assert pq.datetime == datetime.datetime(
            2025, 11, 14, 12, 15, 46, 665000
        )
        assert pq.guid == uuid.UUID('a8210025-5de6-418a-841c-186da82e169e')

        tags = pq.tags
        assert tags['File_GUID'] == '{A8210025-5DE6-418A-841C-186DA82E169E}'
        assert tags['CreatorSW_Name'] == 'UniHarp'
        assert tags['CreatorSW_Version'] == '1.1.1.130'
        assert tags['File_CreatingTime'] == datetime.datetime(
            2025, 11, 14, 12, 15, 46, 665000
        )
        assert tags['CreatorSW_Modules'] == 0
        assert tags['HistoResult_NumberOfCurves'] == 0

        attrs = pq.attrs
        assert attrs['type'] == pq.type.name
        assert attrs['name'] == pq.name
        assert attrs['tags'] == pq.tags


def test_spqr() -> None:
    """Test read SPQR file."""
    filename = (
        DATA / 'Luminosa/GattaQUant_Cells_FLIM/GattaQUant_Cells_FLIM.spqr'
    )
    with PqFile(filename) as pq:
        str(pq)
        assert pq.type == PqFileType.SPQR
        assert pq.version == '1.0.00'
        assert pq.tags['CreatorSW_Name'] == 'NovaConvert'
        preview = numpy.frombuffer(
            pq.tags['SPQRPrevImage'], dtype=numpy.uint8
        ).reshape((128, 128, 4))
        assert preview.shape == (128, 128, 4)
        assert len(pq.tags['SPQRBinWidths']) == 128
        attrs = pq.attrs
        assert attrs['type'] == pq.type.name
        assert attrs['name'] == pq.name
        assert attrs['tags'] == pq.tags


@pytest.mark.parametrize('filetype', [str, io.BytesIO, mmap])
def test_ptu(filetype: Any) -> None:
    """Test read PTU file."""
    filename: Any = (
        DATA / 'napari_flim_phasor_plotter/hazelnut_FLIM_single_image.ptu'
    )
    if filetype is io.BytesIO:
        filename = open(filename, 'rb')
    try:
        with PtuFile(filename, memmap=filetype is mmap) as ptu:
            str(ptu)
            assert ptu.memmapped == (filetype is mmap)
            assert ptu.type == PqFileType.PTU
            assert ptu.record_type == PtuRecordType.PicoHarpT3
            assert ptu.measurement_mode == PtuMeasurementMode.T3
            assert ptu.measurement_submode == PtuMeasurementSubMode.IMAGE
            assert ptu.scanner == PtuScannerType.LSM
            assert ptu.measurement_ndim == 3
            assert ptu.filehandle is not None
            if filetype is str:
                assert ptu.filename == str(filename.name)
                assert ptu.dirname == str(filename.parent)
            assert ptu.version == '00.0.1'
            assert ptu.comment == ''
            assert ptu.datetime is None
            assert str(ptu.guid) == '4f6e5f68-8289-483d-9d9a-7974b77ef8b8'
            assert ptu.tags['TTResultFormat_BitsPerRecord'] == 32
            assert ptu.tags['\x02HWInpChan_CFDLeveld'] == [100]  # corrupted?
            assert not ptu.tags['HW_ExternalRefClock']

            # BinaryFile properties
            assert ptu.name == 'hazelnut_FLIM_single_image.ptu'
            assert ptu.filesize == 24285248
            assert not ptu.writable
            assert not ptu.closed
            if filetype is str:
                assert ptu.filepath == str(filename)
                assert isinstance(ptu.lock, contextlib.nullcontext)

            # PtuFile properties
            assert ptu.record_offset == 4616
            assert ptu.measurement_warnings is None
            assert ptu.hardware_features is None
            assert ptu.stop_reason == PtuStopReason.MANUAL
            assert ptu.global_resolution == 1.281722635221738e-08
            assert ptu.tcspc_resolution == 9.696969697e-11
            assert ptu.number_records == 6070158
            assert ptu.number_photons == 6065123
            assert ptu.number_markers == 2565
            assert ptu.number_images == 5
            assert ptu.number_planes == 1
            assert ptu.number_lines == 1280
            assert ptu.number_channels_max == 4
            assert ptu.number_channels == 1
            assert ptu.active_channels == (0,)
            assert ptu.number_bins_max == 4096
            assert ptu.number_bins == 139
            assert ptu.number_bins_in_period == 132
            assert ptu.line_start_mask == 1
            assert ptu.line_stop_mask == 2
            assert ptu.frame_change_mask == 4
            assert ptu.global_pixel_time == 494
            assert ptu.global_line_time == 126464
            assert ptu.global_frame_time == 32374784
            assert ptu.global_volume_time == 32374784
            assert ptu.global_acquisition_time == 161873920
            assert ptu.pixels_in_frame == 65536
            assert ptu.pixels_in_line == 256
            assert ptu.lines_in_frame == 256
            assert ptu.pixel_time == 6.331709817995386e-06
            assert ptu.line_time == 0.0016209177134068188
            assert ptu.frame_time == 0.4149549346321456
            assert ptu.volume_time == 0.4149549346321456
            assert ptu.acquisition_time == 2.074774673160728
            assert ptu.frequency == 78020000.0
            assert ptu.syncrate == 78020000
            assert ptu.is_image
            assert ptu.is_t3
            assert not ptu.is_bidirectional
            assert not ptu.is_sinusoidal
            assert not ptu.use_xarray
            assert ptu.cache_records == (filetype is not mmap)
            assert ptu.dtype == numpy.uint16
            assert ptu.shape == (5, 256, 256, 1, 139)
            assert ptu.dims == ('T', 'Y', 'X', 'C', 'H')
            assert ptu.sizes == {'T': 5, 'Y': 256, 'X': 256, 'C': 1, 'H': 139}
            assert ptu.ndims == 5
            assert ptu.nbytes == 91095040
            assert ptu.size == 45547520
            assert ptu.itemsize == 2

            coords = ptu.coords
            assert tuple(coords.keys()) == ('T', 'Y', 'X', 'C', 'H')
            for dim, size in ptu.sizes.items():
                assert coords[dim].size == size
            assert_allclose(coords['T'][[1, -1]], [0.41495493, 1.65981974])
            assert_allclose(coords['Y'][[1, -1]], [1.46326157e-7, 3.731317e-5])
            assert_allclose(coords['X'][[1, -1]], [1.46326157e-7, 3.731317e-5])
            assert array_equal(coords['C'], [0])
            assert_allclose(
                coords['H'][[1, -1]], [9.6969697e-11, 1.3381818e-8]
            )

            attrs = ptu.attrs
            assert attrs['type'] == ptu.type.name
            assert attrs['name'] == ptu.name
            assert attrs['guid'] == str(ptu.guid)
            assert attrs['datetime'] is None
            assert attrs['tags'] == ptu.tags
            assert attrs['filepath'] == ptu.filepath
            assert attrs['version'] == ptu.version
            assert attrs['comment'] == ptu.comment

            assert attrs['acquisition_time'] == ptu.acquisition_time
            assert attrs['active_channels'] == ptu.active_channels
            assert attrs['frame_time'] == ptu.frame_time
            assert attrs['frequency'] == ptu.frequency
            assert (
                attrs['global_acquisition_time'] == ptu.global_acquisition_time
            )
            assert attrs['global_volume_time'] == ptu.global_volume_time
            assert attrs['global_frame_time'] == ptu.global_frame_time
            assert attrs['global_line_time'] == ptu.global_line_time
            assert attrs['global_pixel_time'] == ptu.global_pixel_time
            assert attrs['global_resolution'] == ptu.global_resolution
            assert attrs['line_time'] == ptu.line_time
            assert (
                attrs['max_delaytime'] == ptu.number_bins_max
            )  # for PhasorPy
            assert attrs['measurement_mode'] == ptu.measurement_mode.name
            assert attrs['measurement_submode'] == ptu.measurement_submode.name
            assert attrs['number_bins'] == ptu.number_bins
            assert attrs['number_bins_in_period'] == ptu.number_bins_in_period
            assert attrs['number_bins_max'] == ptu.number_bins_max
            assert attrs['pixel_time'] == ptu.pixel_time
            assert attrs['record_type'] == ptu.record_type.name
            assert attrs['scanner'] == ptu.scanner.name
            assert attrs['syncrate'] == ptu.syncrate
            assert attrs['tcspc_resolution'] == ptu.tcspc_resolution

            # decoding of records is tested separately
    finally:
        if filetype is io.BytesIO:
            filename.close()


@pytest.mark.parametrize('filetype', [str, io.BytesIO, mmap])
def test_phu(filetype: Any) -> None:
    """Test read PHU file."""
    filename: Any = DATA / 'TimeHarp/Decay_Coumarin_6.phu'
    if filetype is io.BytesIO:
        filename = open(filename, 'rb')
    try:
        with PhuFile(filename, memmap=filetype is mmap) as phu:
            str(phu)
            assert phu.type == PqFileType.PHU
            assert phu.measurement_mode == PhuMeasurementMode.HISTOGRAM
            assert phu.measurement_submode == PhuMeasurementSubMode.INTEGRATING
            assert phu.version == '1.1.00'
            assert not phu.tags['HWTriggerOut_On']
            assert phu.tcspc_resolution == 2.5e-11
            assert phu.number_histograms == 4
            # assert phu.histogram_resolutions == (3e-11, 3e-11, 3e-11, 3e-11)

            attrs = phu.attrs
            assert attrs['type'] == phu.type.name
            assert attrs['name'] == phu.name
            assert attrs['tags'] == phu.tags
            assert attrs['measurement_mode'] == phu.measurement_mode.name
            assert attrs['measurement_submode'] == phu.measurement_submode.name
            assert attrs['tcspc_resolution'] == phu.tcspc_resolution
            # assert attrs['histogram_resolutions']==phu.histogram_resolutions

            assert array_equal(
                phu.tags['HistResDscr_DataOffset'],
                [11224, 142296, 273368, 404440],
            )
            if xarray is not None:
                histograms = phu.histograms(asxarray=True)
                assert len(histograms) == 4
                for h in histograms:
                    assert h.shape == (32768,)
                assert histograms[2][1] == 3
                assert array_equal(phu.histograms(2)[0], histograms[2])
                assert array_equal(phu.histograms(-1)[0], histograms[3])
                assert array_equal(
                    phu.histograms(1, asxarray=True)[0], histograms[1]
                )
                with pytest.raises(IndexError):
                    phu.histograms(4)
            if pyplot is not None:
                phu.plot(show=False, verbose=False)
                phu.plot(show=False, verbose=True)
    finally:
        if filetype is io.BytesIO:
            filename.close()


def test_phu_baseres() -> None:
    """Test read PHU file without HistResDscr_HWBaseResolution."""
    # also has a non-ISO datetime string format
    filename = DATA / 'FluoPlot/Naphtal_BuOH_TRES.phu'

    with PhuFile(filename) as phu:
        str(phu)
        assert phu.type == PqFileType.PHU
        assert phu.measurement_mode == PhuMeasurementMode.HISTOGRAM
        assert phu.measurement_submode == PhuMeasurementSubMode.INTEGRATING
        assert phu.version == '1.0.00'
        assert phu.tags['File_CreatingTime'] == '29/11/06 18:51:08'
        assert phu.datetime == datetime.datetime(2006, 11, 29, 18, 51, 8)
        assert 'HistResDscr_HWBaseResolution' not in phu.tags
        assert phu.tcspc_resolution == 1.600000075995922e-11
        assert phu.number_histograms == 42
        if xarray is not None:
            histograms = phu.histograms(asxarray=True)
            assert len(histograms) == 42
            for h in histograms:
                assert h.shape == (65536,)
        if pyplot is not None:
            phu.plot(show=False, verbose=False)
            phu.plot(show=False, verbose=True)


def test_ptu_t3_image() -> None:
    """Test decode T3 image."""
    filename = (
        DATA / 'napari_flim_phasor_plotter/hazelnut_FLIM_single_image.ptu'
    )
    with PtuFile(filename) as ptu:
        str(ptu)
        assert str(ptu.guid) == '4f6e5f68-8289-483d-9d9a-7974b77ef8b8'
        assert ptu.version == '00.0.1'
        # assert ptu._data_offset == 4616
        assert ptu.type == PqFileType.PTU
        assert ptu.record_type == PtuRecordType.PicoHarpT3
        assert ptu.measurement_mode == PtuMeasurementMode.T3
        assert ptu.measurement_submode == PtuMeasurementSubMode.IMAGE
        assert ptu.scanner == PtuScannerType.LSM
        assert ptu.measurement_ndim == 3

        assert ptu.is_image
        assert ptu.is_t3

        assert ptu.shape == (5, 256, 256, 1, 139)
        assert ptu.dims == ('T', 'Y', 'X', 'C', 'H')
        assert tuple(ptu.coords.keys()) == ('T', 'Y', 'X', 'C', 'H')
        assert ptu.active_channels == (0,)
        # TODO: verify coords

        assert ptu.frame_change_mask == 4
        assert ptu.line_start_mask == 1
        assert ptu.line_stop_mask == 2

        assert ptu.acquisition_time == 2.074774673160728
        assert ptu.frame_time == 0.4149549346321456
        assert ptu.frequency == 78020000.0
        assert ptu.global_frame_time == 32374784
        assert ptu.global_line_time == 126464
        assert ptu.global_pixel_time == 494
        assert ptu.global_resolution == 1.281722635221738e-8
        assert ptu.pixel_time == pytest.approx(
            ptu.global_pixel_time * ptu.global_resolution, rel=1e-3
        )
        assert ptu.line_time == pytest.approx(
            ptu.global_line_time * ptu.global_resolution, rel=1e-3
        )
        assert ptu.lines_in_frame == 256
        assert ptu.number_bins == 139
        assert ptu.number_bins_in_period == 132
        assert ptu.number_bins_max == 4096
        assert ptu.number_channels == 1
        assert ptu.number_channels_max == 4
        assert ptu.number_lines == 1280
        assert ptu.number_markers == 2565
        assert ptu.number_photons == 6065123
        assert ptu.number_records == 6070158
        assert ptu.pixels_in_frame == 65536
        assert ptu.pixels_in_line == 256
        assert ptu.syncrate == 78020000
        assert ptu.tcspc_resolution == 9.696969697e-11

        assert len(ptu.read_records()) == ptu.number_records
        # assert ptu.decode_records
        im0 = ptu.decode_image()
        assert im0.shape == (5, 256, 256, 1, 139)
        assert im0.dtype == numpy.uint16

        im = ptu.decode_image(channel=0, frame=2, dtime=-1, dtype='uint32')
        assert im.shape == (1, 256, 256, 1, 1)
        assert im.dtype == numpy.uint32
        assert array_equal(im[0, ..., 0, 0], im0[2, :, :, 0].sum(axis=-1))

        im = ptu.decode_image(channel=0, frame=2, dtime=99, dtype='uint32')
        assert im.shape == (1, 256, 256, 1, 99)
        assert array_equal(
            im[0, :, :, 0].sum(axis=-1), im0[2, :, :, 0, :99].sum(axis=-1)
        )

        im = ptu.decode_image(channel=0, frame=2, dtime=199, dtype='uint32')
        assert im.shape == (1, 256, 256, 1, 199)
        assert array_equal(
            im[0, :, :, 0].sum(axis=-1), im0[2, :, :, 0].sum(axis=-1)
        )

        im = ptu.decode_image(
            [2, slice(0, 32), slice(100, 132), None, slice(None, None, -1)]
        )
        assert im.shape == (1, 32, 32, 1, 1)
        assert array_equal(im[..., 0], im0[2:3, :32, 100:132].sum(axis=-1))

        im = ptu.decode_image(
            [slice(1, None, 2)],  # bin by 2 frames starting from second
            dtime=-1,
        )
        assert im.shape == (2, 256, 256, 1, 1)
        assert array_equal(
            im[1, :, :, :, 0], im0[3:5].sum(axis=0).sum(axis=-1)
        )
        # TODO: verify values

        with pytest.raises(ValueError):
            ptu.decode_image(dtype='int16')


def test_ptu_t3_zstack() -> None:
    """Test decode T3 z-stack image."""
    # https://github.com/cgohlke/ptufile/issues/5
    filename = DATA / 'issue5/RawImage.ptu'
    with PtuFile(filename) as ptu:
        str(ptu)
        assert ptu.record_type == PtuRecordType.GenericT3
        assert ptu.measurement_submode == PtuMeasurementSubMode.IMAGE
        assert ptu.scanner == PtuScannerType.FLIMBEE
        assert ptu.measurement_ndim == 4
        assert ptu.is_image
        assert ptu.number_planes == 11
        assert ptu.number_images == 11
        assert ptu.number_lines == 1408
        assert ptu.number_photons == 1134
        assert ptu.shape == (1, 11, 128, 128, 1, 1248)
        assert ptu.dims == ('T', 'Z', 'Y', 'X', 'C', 'H')
        assert ptu.active_channels == (1,)
        assert ptu.global_pixel_time == 76
        assert ptu.global_volume_time == ptu.global_frame_time * 11
        assert_allclose(ptu.coords['Z'], numpy.arange(11) * 1e-6, atol=1e-9)

        # sparse tag indices
        tags = ptu.tags
        assert tags['DetectorCount'] == 5
        assert tags['Detector'] == [
            'Detector 1 (HPD)',
            'Detector 2 (SPAD)',
            'Detector 3 (SPAD)',
            'Detector 4 (SPAD)',
            'Detector 5 (PDA-23)',
        ]
        assert tags['DetectorInUse'] == [False, True, False, False, False]
        assert tags['DetectorTCSPCChan'] == [None, None, None, None, 10]
        assert tags['DetectorNumTCSPCChan'] == [None, None, None, None, 23]

        im0 = ptu.decode_image()
        assert im0.shape == (1, 11, 128, 128, 1, 1248)
        assert im0.sum() == 338  # photons within line scans
        assert array_equal(
            im0.sum(axis=(0, 2, 3, 4, 5)),
            [84, 113, 30, 23, 19, 11, 19, 15, 13, 5, 6],
        )

        im = ptu.decode_image(plane=3)
        assert im.shape == (1, 1, 128, 128, 1, 1248)
        assert array_equal(im, im0[:, 3:4])

        im = ptu.decode_image(plane=-1, dtime=-1, dtype='uint32')
        assert im.shape == (1, 1, 128, 128, 1, 1)
        assert array_equal(
            im[:, 0, ..., 0], im0.sum(axis=(1, 5), dtype=numpy.uint32)
        )

        im = ptu.decode_image(
            [0, slice(2, 10, 4), slice(0, 64), ..., slice(None, None, -1)],
            dtype='uint32',
        )
        assert im.shape == (1, 2, 64, 128, 1, 1)
        expected = im0[:, 2:10, :64].reshape(1, 2, 4, 64, 128, 1, 1248)
        assert array_equal(
            im[..., 0], expected.sum(axis=(2, 6), dtype=numpy.uint32)
        )

        item = ptu[0, 1, ..., 0, ::-1]
        assert isinstance(item, numpy.ndarray)
        assert item.shape == (128, 128)
        assert array_equal(item, im0[0, 1, :, :, 0].sum(axis=-1))

        if xarray is not None:
            xim = ptu.decode_image(plane=5, dtime=-1, asxarray=True)
            assert xim.dims == ('T', 'Z', 'Y', 'X', 'C', 'H')
            assert_allclose(xim.coords['Z'], [5e-6], atol=1e-9)
            xim = ptu.decode_image(plane=5, keepdims=False, asxarray=True)
            assert xim.dims == ('T', 'Y', 'X', 'C', 'H')
            assert 'Z' not in xim.coords

        with pytest.raises(IndexError):
            ptu.decode_image(plane=11)

    filename = (
        DATA / 'napari_flim_phasor_plotter/hazelnut_FLIM_single_image.ptu'
    )
    with PtuFile(filename) as ptu, pytest.raises(IndexError):
        ptu.decode_image(plane=0)


def test_ptu_channels() -> None:
    """Test decode T3 image with empty leading channels."""
    filename = DATA / 'Tutorials.sptw/Kidney _Cell_FLIM.ptu'
    with PtuFile(filename) as ptu:
        str(ptu)
        assert str(ptu.guid) == 'b767c46e-9693-4ad9-9fcf-7fab5e4377fc'
        assert ptu.version == '1.0.00'
        assert ptu.comment is not None
        assert ptu.comment.startswith('SPAD-CH1523')
        assert ptu.datetime == datetime.datetime(2020, 4, 7, 18, 8, 44, 860000)
        assert ptu.type == PqFileType.PTU
        assert ptu.record_type == PtuRecordType.GenericT3
        assert ptu.measurement_mode == PtuMeasurementMode.T3
        assert ptu.measurement_submode == PtuMeasurementSubMode.IMAGE
        assert ptu.scanner == PtuScannerType.FLIMBEE
        assert ptu.measurement_ndim == 3

        assert ptu.is_image
        assert ptu.is_t3

        assert ptu.shape == (3, 512, 512, 2, 501)
        assert ptu.dims == ('T', 'Y', 'X', 'C', 'H')
        assert tuple(ptu.coords.keys()) == ('T', 'Y', 'X', 'C', 'H')
        assert ptu.active_channels == (2, 3)
        assert array_equal(ptu.coords['C'], (2, 3))
        # TODO: verify coords

        assert ptu.frame_change_mask == 4
        assert ptu.line_start_mask == 1
        assert ptu.line_stop_mask == 2

        assert ptu.acquisition_time == 47.26384024428878
        assert ptu.frame_time == 15.754613414762927
        assert ptu.frequency == 24999920.0
        assert ptu.global_frame_time == 393864075
        assert ptu.global_line_time == 384000
        assert ptu.global_pixel_time == 750
        assert ptu.global_resolution == 4.00001280004096e-8
        assert ptu.pixel_time == pytest.approx(
            ptu.global_pixel_time * ptu.global_resolution, rel=1e-3
        )
        assert ptu.line_time == pytest.approx(
            ptu.global_line_time * ptu.global_resolution, rel=1e-3
        )
        assert ptu.lines_in_frame == 512
        assert ptu.number_bins == 501
        assert ptu.number_bins_in_period == 500
        assert ptu.number_bins_max == 32768
        assert ptu.number_channels == 2
        assert ptu.number_channels_max == 64
        assert ptu.number_lines == 1536
        assert ptu.number_markers == 3074
        assert ptu.number_photons == 41039565
        assert ptu.number_records == 42196537
        assert ptu.pixels_in_frame == 262144
        assert ptu.pixels_in_line == 512
        assert ptu.syncrate == 24999920
        assert ptu.tcspc_resolution == 7.999999968033578e-11

        if xarray is None:
            return  # type: ignore[unreachable]

        histogram = ptu.decode_histogram(asxarray=True)
        assert histogram.shape == (2, 501)
        assert array_equal(histogram.coords['C'], (2, 3))
        assert histogram[1].coords['C'] == 3

        histogram = ptu.decode_histogram(asxarray=True, dtime=0)
        assert histogram.shape == (2, 500)

        histogram = ptu.decode_histogram(asxarray=True, dtime=100)
        assert histogram.shape == (2, 100)

        image = ptu.decode_image(asxarray=True, dtime=-1)
        assert array_equal(image.coords['C'], (2, 3))
        assert image[..., 1, :].coords['C'] == 3
        assert array_equal(ptu._coords_c, (2, 3))

        image = ptu.decode_image(asxarray=True, channel=1, dtime=-1)
        assert array_equal(image.coords['C'], (3,))

        image = ptu.decode_image(
            asxarray=True, channel=1, dtime=-1, keepdims=False
        )
        assert 'C' not in image.coords

    with PtuFile(filename, trimdims='') as ptu:
        str(ptu)
        assert ptu.shape == (3, 512, 512, 64, 32768)
        assert ptu.dims == ('T', 'Y', 'X', 'C', 'H')
        assert tuple(ptu.coords.keys()) == ('T', 'Y', 'X', 'C', 'H')
        assert array_equal(ptu.coords['C'][[0, -1]], (0, 63))
        assert array_equal(ptu._coords_c[[0, -1]], (0, 63))

        if xarray is not None:
            image = ptu.decode_image(asxarray=True, dtime=-1, channel=2)
            assert array_equal(image.coords['C'], (2,))


def test_ptu_t3_sinusoidal() -> None:
    """Test decode T3 image with sinusoidal correction."""
    filename = DATA / 'tttrlib/5kDa_1st_1_1_1.ptu'
    with PtuFile(filename) as ptu:
        str(ptu)
        assert ptu.version == '1.0.00'
        # assert ptu._data_offset == 4616
        assert ptu.type == PqFileType.PTU
        assert ptu.record_type == PtuRecordType.PicoHarpT3
        assert ptu.measurement_mode == PtuMeasurementMode.T3
        assert ptu.measurement_submode == PtuMeasurementSubMode.IMAGE
        assert ptu.scanner == PtuScannerType.LSM
        assert ptu.measurement_ndim == 3

        assert ptu.is_t3
        assert ptu.is_image
        assert ptu.is_sinusoidal
        assert not ptu.is_bidirectional

        assert ptu.tags['ImgHdr_SinCorrection'] == 80

        assert ptu.shape == (122, 512, 512, 1, 3216)
        assert ptu.dims == ('T', 'Y', 'X', 'C', 'H')
        assert tuple(ptu.coords.keys()) == ('T', 'Y', 'X', 'C', 'H')
        assert ptu.active_channels == (0,)
        # TODO: verify coords

        assert ptu.frame_change_mask == 4
        assert ptu.line_start_mask == 1
        assert ptu.line_stop_mask == 2

        assert ptu.acquisition_time == 118.88305798296687
        assert ptu.frame_time == 0.9744512873563691
        assert ptu.frequency == 38898320.0
        assert ptu.global_acquisition_time == 4624351232
        assert ptu.global_frame_time == 37904518
        assert ptu.global_line_time == 18995
        assert ptu.global_pixel_time == 37
        assert ptu.global_resolution == 2.5708051144625268e-8
        assert ptu.pixel_time == pytest.approx(
            ptu.global_pixel_time * ptu.global_resolution, rel=1e-3
        )
        assert ptu.line_time == pytest.approx(
            ptu.global_line_time * ptu.global_resolution, rel=1e-3
        )
        assert ptu.lines_in_frame == 512
        assert ptu.number_bins == 3216
        assert ptu.number_bins_in_period == 3213
        assert ptu.number_bins_max == 4096
        assert ptu.number_channels == 1
        assert ptu.number_channels_max == 4
        assert ptu.number_images == 122
        assert ptu.number_lines == 62464
        assert ptu.number_markers == 125050
        assert ptu.number_photons == 8664782
        assert ptu.number_records == 8860394
        assert ptu.pixels_in_frame == 262144
        assert ptu.pixels_in_line == 512
        assert ptu.syncrate == 38898320
        assert ptu.tcspc_resolution == 7.999999968033578e-12

        records = ptu.read_records()
        assert len(records) == ptu.number_records
        im = ptu.decode_image(frame=-1, dtime=-1, channel=0, keepdims=False)
        assert im.shape == (512, 512)
        assert im[399, 18] == 37

        with pytest.raises(ValueError):
            ptu.decode_image(pixel_time=0)


def test_ptu_t3_bidirectional() -> None:
    """Test decode T3 image acquired with bidirectional scanning."""
    filename = DATA / 'fastFLIM/A2_Shep2_26.ptu'
    with PtuFile(filename) as ptu:
        str(ptu)
        assert ptu.version == '1.0.00'
        # assert ptu._data_offset == 4616
        assert ptu.type == PqFileType.PTU
        assert ptu.record_type == PtuRecordType.GenericT3
        assert ptu.measurement_mode == PtuMeasurementMode.T3
        assert ptu.measurement_submode == PtuMeasurementSubMode.IMAGE
        assert ptu.scanner == PtuScannerType.FLIMBEE
        assert ptu.measurement_ndim == 3

        assert ptu.is_t3
        assert ptu.is_image
        assert ptu.is_bidirectional
        assert not ptu.is_sinusoidal

        assert ptu.tags['ImgHdr_BiDirect']

        assert ptu.shape == (1, 1024, 1024, 2, 502)
        assert ptu.dims == ('T', 'Y', 'X', 'C', 'H')
        assert tuple(ptu.coords.keys()) == ('T', 'Y', 'X', 'C', 'H')
        assert ptu.active_channels == (0, 1)
        # TODO: verify coords

        assert ptu.frame_change_mask == 4
        assert ptu.line_start_mask == 1
        assert ptu.line_stop_mask == 2

        assert ptu.acquisition_time == 192.27363031561703
        assert ptu.frame_time == 192.27363031561703
        assert ptu.frequency == 24999920.0
        assert ptu.global_acquisition_time == 4806825376
        assert ptu.global_frame_time == 4806825376
        assert ptu.global_line_time == 3200000
        assert ptu.global_pixel_time == 3125
        assert ptu.global_resolution == 4.00001280004096e-8
        assert ptu.pixel_time == pytest.approx(
            ptu.global_pixel_time * ptu.global_resolution, rel=1e-3
        )
        assert ptu.line_time == pytest.approx(
            ptu.global_line_time * ptu.global_resolution, rel=1e-3
        )
        assert ptu.lines_in_frame == 1024
        assert ptu.number_bins == 502
        assert ptu.number_bins_in_period == 500
        assert ptu.number_bins_max == 32768
        assert ptu.number_channels == 2
        assert ptu.number_channels_max == 64
        assert ptu.number_images == 1
        assert ptu.number_lines == 1024
        assert ptu.number_markers == 2049
        assert ptu.number_photons == 239232451
        assert ptu.number_records == 242488255
        assert ptu.pixels_in_frame == 1048576
        assert ptu.pixels_in_line == 1024
        assert ptu.syncrate == 24999920
        assert ptu.tcspc_resolution == 7.999999968033578e-11

        records = ptu.read_records()
        assert len(records) == ptu.number_records
        im = ptu.decode_image(frame=-1, dtime=-1, channel=0, keepdims=False)
        assert im.shape == (1024, 1024)
        # TODO: compare to ground truth image from SymPhoTime
        assert im[430, 430] == 1057  # even line
        assert im[431, 431] == 1050  # odd line

        # selection
        m, n = 421, 440
        selection = (0, slice(m, n), slice(m, n), 0, slice(None, None, -1))
        im1 = ptu.decode_image(selection, records=records, keepdims=False)
        assert array_equal(im[m:n, m:n], im1)

        # bin Y starting at odd line
        selection = (
            0,
            slice(m, m + 20, 2),
            slice(m, n),
            0,
            slice(None, None, -1),
        )
        im2 = ptu.decode_image(selection, records=records, keepdims=False)
        expected = im[m : m + 20, m:n].reshape(10, 2, n - m)
        assert array_equal(im2, expected.sum(axis=1, dtype=im.dtype))

        # x-shift by one pixel
        im = ptu.decode_image(
            records=records,
            frame=-1,
            dtime=-1,
            channel=0,
            bishift=-ptu.global_pixel_time,
            keepdims=False,
        )
        assert im[430, 430] == 1057  # even line is same
        assert im[431, 430] == 1050  # odd line shifted


def test_ptu_t3_bidirectional_sinusoidal() -> None:
    """Test decode T3 image acquired with bidirectional sinusoidal scanning."""
    filename = DATA / 'tttr-data/imaging/leica/sp8/d0/G-28_S1_1_1.ptu'
    with PtuFile(filename) as ptu:
        str(ptu)
        assert ptu.version == '1.0.00'
        # assert ptu._data_offset == 5816
        assert ptu.type == PqFileType.PTU
        assert ptu.record_type == PtuRecordType.PicoHarpT3
        assert ptu.measurement_mode == PtuMeasurementMode.T3
        assert ptu.measurement_submode == PtuMeasurementSubMode.IMAGE
        assert ptu.scanner == PtuScannerType.LSM
        assert ptu.measurement_ndim == 3

        assert ptu.is_t3
        assert ptu.is_image
        assert ptu.is_bidirectional
        assert ptu.is_sinusoidal

        assert ptu.tags['ImgHdr_BiDirect']

        assert ptu.shape == (26, 512, 512, 1, 3212)
        assert ptu.dims == ('T', 'Y', 'X', 'C', 'H')
        assert tuple(ptu.coords.keys()) == ('T', 'Y', 'X', 'C', 'H')
        assert ptu.active_channels == (0,)
        # TODO: verify coords

        assert ptu.frame_change_mask == 4
        assert ptu.line_start_mask == 1
        assert ptu.line_stop_mask == 2

        assert ptu.acquisition_time == 14.697189081520644
        assert ptu.frame_time == 0.5652764873231677
        assert ptu.frequency == 19459120.0
        assert ptu.global_acquisition_time == 285994366
        assert ptu.global_frame_time == 10999783
        assert ptu.global_line_time == 9534
        assert ptu.global_pixel_time == 19
        assert ptu.global_resolution == 5.138978535514453e-8
        assert ptu.pixel_time == pytest.approx(
            ptu.global_pixel_time * ptu.global_resolution, rel=1e-3
        )
        assert ptu.line_time == pytest.approx(
            ptu.global_line_time * ptu.global_resolution, rel=1e-3
        )
        assert ptu.lines_in_frame == 512
        assert ptu.number_bins == 3212
        assert ptu.number_bins_in_period == 3211
        assert ptu.number_bins_max == 4096
        assert ptu.number_channels == 1
        assert ptu.number_channels_max == 4
        assert ptu.number_images == 26
        assert ptu.number_lines == 13658
        assert ptu.number_markers == 27341
        assert ptu.number_photons == 5585391
        assert ptu.number_records == 5617095
        assert ptu.pixels_in_frame == 262144
        assert ptu.pixels_in_line == 512
        assert ptu.syncrate == 19459120
        assert ptu.tcspc_resolution == 1.5999999936067155e-11

        records = ptu.read_records()
        assert len(records) == ptu.number_records
        im = ptu.decode_image(frame=-1, dtime=-1, channel=0, keepdims=False)
        assert im.shape == (512, 512)
        # TODO: compare to ground truth image from SymPhoTime
        assert im[430, 430] == 38  # even line
        assert im[431, 431] == 32  # odd line

        # selection
        m, n = 421, 440
        selection = (
            slice(None, None, -1),
            slice(m, n),
            slice(m, n),
            0,
            slice(None, None, -1),
        )
        im1 = ptu.decode_image(selection, records=records, keepdims=False)
        assert array_equal(im[m:n, m:n], im1)

        # x-shift by one pixel
        im = ptu.decode_image(
            records=records,
            frame=-1,
            dtime=-1,
            channel=0,
            bishift=-ptu.global_pixel_time,
            keepdims=False,
        )
        assert im[430, 430] == 38  # even line is same
        assert im[431, 430] == 32  # odd line shifted


@pytest.mark.skip('no test file available')
def test_ptu_t3_line() -> None:
    """Test decode T3 line scan."""


def test_ptu_t3_point() -> None:
    """Test decode T3 point scan."""
    filename = DATA / '1XEGFP_1.ptu'
    with PtuFile(filename) as ptu:
        str(ptu)
        assert str(ptu.guid) == 'dec6a033-99a9-482d-afbd-5b5743a25133'
        assert ptu.version == '1.0.00'
        assert ptu.type == PqFileType.PTU
        assert ptu.record_type == PtuRecordType.PicoHarpT3
        assert ptu.measurement_mode == PtuMeasurementMode.T3
        assert ptu.measurement_submode == PtuMeasurementSubMode.POINT
        assert ptu.scanner == PtuScannerType.PI_E710
        assert ptu.measurement_ndim == 1

        assert not ptu.is_image
        assert not ptu.is_bidirectional
        assert not ptu.is_sinusoidal
        assert ptu.is_t3

        assert ptu.shape == (59998, 2, 1564)
        assert ptu.dims == ('T', 'C', 'H')
        assert tuple(ptu.coords.keys()) == ('T', 'C', 'H')
        assert ptu.active_channels == (0, 1)
        # TODO: verify coords

        assert ptu.frame_change_mask == 0
        assert ptu.line_start_mask == 0
        assert ptu.line_stop_mask == 0

        assert ptu.acquisition_time == 59.998948937161735
        assert ptu.frame_time == 0.0010000110003960143
        assert ptu.frequency == 39998560.0
        assert ptu.global_frame_time == 39999
        assert ptu.global_line_time == 39999
        assert ptu.global_pixel_time == 39999
        assert ptu.global_resolution == 2.5000900032401165e-8
        assert ptu.pixel_time == pytest.approx(
            ptu.global_pixel_time * ptu.global_resolution, rel=1e-3
        )
        assert ptu.line_time == pytest.approx(
            ptu.global_line_time * ptu.global_resolution, rel=1e-3
        )
        assert ptu.lines_in_frame == 1
        assert ptu.number_bins == 1564
        assert ptu.number_bins_in_period == 1562
        assert ptu.number_bins_max == 4096
        assert ptu.number_channels == 2
        assert ptu.number_channels_max == 4
        assert ptu.number_lines == 0
        assert ptu.number_markers == 0
        assert ptu.number_photons == 11516799
        assert ptu.number_records == 11553418
        assert ptu.pixels_in_frame == 1
        assert ptu.pixels_in_line == 1
        assert ptu.syncrate == 39998560
        assert ptu.tcspc_resolution == 1.5999999936067155e-11

        assert len(ptu.read_records()) == ptu.number_records
        # assert ptu.decode_records
        im0 = ptu.decode_image(dtime=-1, dtype='uint32')
        assert im0.shape == (59998, 2, 1)
        assert im0.dtype == numpy.uint32
        # photons in the last incomplete sample are dropped
        assert 0.999 * ptu.number_photons < im0.sum() <= ptu.number_photons
        im = ptu.decode_image(channel=0, frame=2, dtime=-1, dtype='uint32')
        assert im.shape == (1, 1, 1)
        assert im.dtype == numpy.uint32
        # TODO: verify values


@pytest.mark.parametrize('filename', FILES)
def test_ptu_decode_records(filename: str) -> None:
    """Test decode records."""
    with PtuFile(DATA / filename) as ptu:
        decoded = ptu.decode_records()
        assert decoded.dtype == (
            T3_RECORD_DTYPE if ptu.is_t3 else T2_RECORD_DTYPE
        )
        assert decoded.size == ptu.number_records
        assert decoded['time'][-1] == ptu.global_acquisition_time
        assert decoded['channel'].max() + 1 == ptu.number_channels
        assert decoded[decoded['channel'] >= 0].size == ptu.number_photons
        assert decoded[decoded['marker'] > 0].size == ptu.number_markers
        nframes = decoded[decoded['marker'] & ptu.frame_change_mask > 0].size
        if ptu.shape and ptu.tags['Measurement_SubMode'] > 0:
            assert abs(nframes - ptu.shape[0]) < 2
        if ptu.is_t3:
            assert decoded['dtime'].max() + 1 == ptu.number_bins
            assert decoded[decoded['dtime'] >= 0].size == ptu.number_photons
        # TODO: verify values


@pytest.mark.parametrize('asxarray', [False, True])
@pytest.mark.parametrize('filename', FILES)
def test_ptu_decode_histogram(*, filename: str, asxarray: bool) -> None:
    """Test decode histograms."""
    if asxarray and xarray is None:
        pytest.skip('xarray not installed')
    with PtuFile(DATA / filename) as ptu:
        ptu.decode_histogram(asxarray=asxarray)
        # TODO: verify values
        with pytest.raises(ValueError):
            ptu.decode_histogram(dtype='int32')


@pytest.mark.skipif(pyplot is None, reason='matplotlib not installed')
@pytest.mark.parametrize('verbose', [False, True])
@pytest.mark.parametrize('filename', FILES)
def test_ptu_plot(*, filename: str, verbose: bool) -> None:
    """Test plot methods."""
    with PtuFile(DATA / filename) as ptu:
        ptu.plot(show=False, verbose=verbose)


@pytest.mark.parametrize('copy', [False, True])
@pytest.mark.parametrize('memmap', [False, True])
def test_ptu_read_records(*, copy: bool, memmap: bool) -> None:
    """Test PTU read_records method."""
    # the file is tested in test_issue_skip_frame
    filename = DATA / 'Samples.sptw/GUVs.ptu'
    with PtuFile(filename, memmap=memmap) as ptu:
        # explicit copy mode
        records = ptu.read_records(copy=copy)
        assert ptu.cache_records == (not memmap)
        assert isinstance(records, numpy.ndarray), type(records)
        assert records.size == ptu.number_records

        if ptu.cache_records:
            assert ptu._records is not None
            if copy:
                assert records is not ptu._records
                assert not numpy.shares_memory(records, ptu._records)
                assert ptu.read_records(copy=copy) is not records
            else:
                assert records is ptu._records
                assert records is ptu.read_records(copy=copy)

            # default copy=True always returns detached arrays
            records_default = ptu.read_records()
            assert records_default is not ptu._records
            assert not numpy.shares_memory(records_default, ptu._records)

        im0 = ptu.decode_image(records=records, frame=1, channel=1, dtime=-1)
        del records

        # disable caching
        ptu.cache_records = False
        assert not ptu.cache_records
        assert ptu._records is None
        records = ptu.read_records(copy=copy)
        assert ptu._records is None
        assert isinstance(records, numpy.ndarray), type(records)
        assert ptu.read_records(copy=copy) is not records  # not from cache

        # default copy=True with no caching
        assert ptu.read_records() is not records

        im1 = ptu.decode_image(records=records, frame=1, channel=1, dtime=-1)
        assert array_equal(im0, im1)


@pytest.mark.parametrize('output', ['ndarray', 'memmap', 'filename'])
def test_ptu_output(output: str) -> None:
    """Test PTU decoding to different output."""
    # the file is tested in test_issue_skip_frame
    filename = DATA / 'Samples.sptw/GUVs.ptu'
    with PtuFile(filename) as ptu:
        assert ptu.shape == (100, 512, 512, 2, 4096)
        selection = (
            slice(11, 66, 3),
            Ellipsis,
            slice(1, 2),
            slice(None, None, -1),
        )
        shape = (19, 512, 512, 1, 1)
        out: Any
        if output == 'ndarray':
            out = numpy.zeros(shape, 'uint32')
        elif output == 'filename':
            out = tempfile.TemporaryFile()
        else:
            out = 'memmap'
        im = ptu.decode_image(selection, out=out)
        if output == 'ndarray':
            im = out
        else:
            assert isinstance(im, numpy.memmap)
        assert im[:, 281, 373, 0, 0].sum(axis=0) == 157
        if output == 'filename':
            out.close()


def test_ptu_getitem() -> None:
    """Test slice PTU."""
    # the file is tested in test_issue_skip_frame
    filename = DATA / 'Samples.sptw/GUVs.ptu'
    with PtuFile(filename) as ptu:
        assert ptu.shape == (100, 512, 512, 2, 4096)
        assert ptu.dims == ('T', 'Y', 'X', 'C', 'H')
        assert not ptu.use_xarray
        with pytest.raises(ValueError):
            ptu.dtype = 'int32'  # not an unsigned integer
        assert ptu.dtype == 'uint16'
        ptu.dtype = 'uint32'
        assert ptu.dtype == 'uint32'
        # decode
        im0 = ptu.decode_image(
            (slice(11, 66, 3), Ellipsis, slice(1, 2), slice(None, None, -1)),
            keepdims=False,
        )
        assert im0.shape == (19, 512, 512, 1)
        assert im0.dtype == 'uint32'
        assert im0[:, 281, 373, 0].sum(axis=0) == 157
        # slice
        im = ptu[-89:-34:3, ..., :, 1:2, ::-1]
        assert im0.dtype == 'uint32'
        assert array_equal(im, im0)
        ptu.dtype = 'uint16'
        # slice uint16
        im = ptu[11:66:3, ..., -1, ::-1]
        assert im.shape == (19, 512, 512)
        assert im.dtype == 'uint16'
        assert array_equal(im, im0.squeeze())

        if xarray is None:
            return  # type: ignore[unreachable]

        # slice with xarray
        ptu.use_xarray = True
        assert ptu.use_xarray
        im = ptu[11:66:3, ..., 1:2, ::-1]
        assert isinstance(im, xarray.DataArray)
        assert tuple(im.coords.keys()) == ('T', 'Y', 'X', 'C')
        assert im.coords['T'].values[0] == 20.62217778751141
        # sum all
        ptu.dtype = 'uint64'
        im = ptu[::-1, ::-1, ::-1, ::-1, ::-1]
        assert isinstance(im, xarray.DataArray)
        assert im.shape == ()
        assert tuple(im.coords.keys()) == ()

        photons = ptu.decode_image([slice(None, None, -1)] * 5).sum()
        assert im.values == photons  # 21243427
        with pytest.raises(IndexError):
            im = ptu[0, 0, 0, 0, 0, 0]  # too many indices
        with pytest.raises(IndexError):
            im = ptu[0, ..., 0, ..., 0]  # more than one Ellipsis
        with pytest.raises(IndexError):
            im = ptu[101]  # index out of range
        with pytest.raises(IndexError):
            im = ptu[50:49]  # stop < start
        with pytest.raises(IndexError):
            im = ptu[101:102]  # start out of range
        with pytest.raises(IndexError):
            im = ptu[1.0]  # invalid type


def test_issue_datetime() -> None:
    """Test file with datetime stored as float, not str or TDateTime."""
    filename = DATA / 'Zenodo_7656540/2a_FLIM_single_image.ptu'

    with PtuFile(filename) as ptu:
        assert ptu.version == '00.0.1'
        assert ptu.tags['File_CreatingTime'] == 13301655831.929
        # raises OverflowError
        assert ptu.datetime is None


def test_issue_falcon_point() -> None:
    """Test PTU from FALCON with no ImgHdr_PixY."""
    # file produced by FALCON in image mode but no ImgHdr_PixX/Y
    filename = DATA / 'FALCON_ptu_examples/40MHz_example.ptu'

    with PtuFile(filename) as ptu:
        assert ptu.version == '00.0.1'
        assert ptu.type == PqFileType.PTU
        assert ptu.record_type == PtuRecordType.PicoHarpT3
        assert ptu.measurement_mode == PtuMeasurementMode.T3
        assert ptu.measurement_submode == PtuMeasurementSubMode.IMAGE
        assert ptu.scanner == PtuScannerType.LSM
        assert ptu.measurement_ndim == 1

        assert ptu.shape == (15479, 1, 269)
        assert ptu.dims == ('T', 'C', 'H')
        assert tuple(ptu.coords.keys()) == ('T', 'C', 'H')
        assert ptu.active_channels == (0,)
        # TODO: verify coords

        assert ptu._info.skip_first_frame == 0
        assert ptu._info.skip_last_frame == 0

        assert ptu.frame_change_mask == 0
        assert ptu.line_start_mask == 0
        assert ptu.line_stop_mask == 0

        assert ptu.acquisition_time == 15.4790509824
        assert ptu.frame_time == 15.4790509824
        assert ptu.frequency == 312500000.0
        assert ptu.global_frame_time == 4837203432
        assert ptu.global_line_time == 312500
        assert ptu.global_pixel_time == 312500
        assert ptu.global_resolution == 3.2e-9
        assert ptu.pixel_time == pytest.approx(
            ptu.global_pixel_time * ptu.global_resolution, rel=1e-3
        )
        assert ptu.line_time == pytest.approx(
            ptu.global_line_time * ptu.global_resolution, rel=1e-3
        )
        assert ptu.lines_in_frame == 1
        assert ptu.number_bins == 269
        assert ptu.number_bins_in_period == 32
        assert ptu.number_bins_max == 4096
        assert ptu.number_channels == 1
        assert ptu.number_channels_max == 4
        assert ptu.number_lines == 0
        assert ptu.number_markers == 0
        assert ptu.number_photons == 1016546
        assert ptu.number_records == 1090355
        assert ptu.pixels_in_frame == 1
        assert ptu.pixels_in_line == 1
        assert ptu.syncrate == 0
        assert ptu.tcspc_resolution == 9.696969697e-11

        im = ptu.decode_image(channel=0, keepdims=False, dtype='uint32')
        assert im.shape == (15479, 269)
        assert im.dtype == numpy.uint32
        # photons in the last incomplete sample are dropped
        assert 0.999 * ptu.number_photons < im.sum() <= ptu.number_photons


def test_issue_skip_frame() -> None:
    """Test PTU with incomplete last frame."""
    filename = DATA / 'Samples.sptw/GUVs.ptu'
    with PtuFile(filename) as ptu:
        assert ptu.version == '00.0.0'
        assert ptu.type == PqFileType.PTU
        assert ptu.record_type == PtuRecordType.PicoHarpT3
        assert ptu.measurement_mode == PtuMeasurementMode.T3
        assert ptu.measurement_submode == PtuMeasurementSubMode.IMAGE
        assert ptu.scanner == PtuScannerType.LSM
        assert ptu.measurement_ndim == 3

        assert ptu.shape == (100, 512, 512, 2, 4096)
        assert ptu.dims == ('T', 'Y', 'X', 'C', 'H')
        assert tuple(ptu.coords.keys()) == ('T', 'Y', 'X', 'C', 'H')
        assert ptu.active_channels == (0, 1)
        # TODO: verify coords

        assert ptu._info.skip_first_frame == 0
        assert ptu._info.skip_last_frame == 1

        assert ptu.frame_change_mask == 4
        assert ptu.line_start_mask == 1
        assert ptu.line_stop_mask == 2

        assert ptu.acquisition_time == 174.96876283164778
        assert ptu.frame_time == 1.7496875403137495
        assert ptu.frequency == 9999690.0
        assert ptu.global_frame_time == 17496333
        assert ptu.global_line_time == 20480
        assert ptu.global_pixel_time == 40
        assert ptu.global_resolution == 1.0000310009610297e-7
        assert ptu.pixel_time == pytest.approx(
            ptu.global_pixel_time * ptu.global_resolution, rel=1e-3
        )
        assert ptu.line_time == pytest.approx(
            ptu.global_line_time * ptu.global_resolution, rel=1e-3
        )
        assert ptu.lines_in_frame == 512
        assert ptu.number_bins == 4096
        assert ptu.number_bins_in_period == 6250  # > number_bins_max !
        assert ptu.number_bins_max == 4096
        assert ptu.number_channels == 2
        assert ptu.number_channels_max == 4
        assert ptu.number_lines == 51204
        assert ptu.number_markers == 102509
        assert ptu.number_photons == 32976068
        assert ptu.number_records == 33105274
        assert ptu.pixels_in_frame == 262144
        assert ptu.pixels_in_line == 512
        assert ptu.syncrate == 9999690
        assert ptu.tcspc_resolution == 1.5999999936067155e-11

        im = ptu.decode_image(
            [92], channel=1, dtime=-1, keepdims=False, dtype='uint32'
        )
        assert im.shape == (512, 512)
        assert im.dtype == numpy.uint32
        assert im[281, 373] == 3


def test_issue_marker_order() -> None:
    """Test PTU with strange marker order."""
    # the file has `[ | ][][][ | ]`` instead  of `[] | [][] | []`` markers
    # first and last frame are incomplete
    filename = DATA / 'Samples.sptw/CENP-labelled_cells_for_FRET.ptu'
    with PtuFile(filename, trimdims='CH') as ptu:
        assert ptu.version == '00.0.0'
        # assert ptu._data_offset == 4616
        assert ptu.type == PqFileType.PTU
        assert ptu.record_type == PtuRecordType.PicoHarpT3
        assert ptu.measurement_mode == PtuMeasurementMode.T3
        assert ptu.measurement_submode == PtuMeasurementSubMode.IMAGE
        assert ptu.scanner == PtuScannerType.LSM
        assert ptu.measurement_ndim == 3

        assert ptu.shape == (191, 512, 512, 3, 3126)
        assert ptu.dims == ('T', 'Y', 'X', 'C', 'H')
        assert tuple(ptu.coords.keys()) == ('T', 'C', 'H')
        assert ptu.active_channels == (0, 1, 2)
        # TODO: verify coords

        assert ptu._info.skip_first_frame == 0
        assert ptu._info.skip_last_frame == 0
        assert ptu._info.lines == 97596

        assert ptu.frame_change_mask == 4
        assert ptu.line_start_mask == 1
        assert ptu.line_stop_mask == 2

        assert ptu.acquisition_time == 213.87296482079424
        assert ptu.frame_time == 1.1197537179119068
        assert ptu.frequency == 20001617.0
        assert ptu.global_frame_time == 22396885
        assert ptu.global_line_time == 20446
        assert ptu.global_pixel_time == 40
        assert ptu.global_resolution == 4.99959578268097e-8
        assert ptu.pixel_time == pytest.approx(
            ptu.global_pixel_time * ptu.global_resolution, rel=1e-3
        )
        assert ptu.line_time == pytest.approx(
            ptu.global_line_time * ptu.global_resolution, rel=1e-3
        )
        assert ptu.lines_in_frame == 512
        assert ptu.number_bins == 3126
        assert ptu.number_bins_in_period == 3124
        assert ptu.number_bins_max == 4096
        assert ptu.number_channels == 3
        assert ptu.number_channels_max == 4
        assert ptu.number_lines == 97596
        assert ptu.number_markers == 195384
        assert ptu.number_photons == 17601306
        assert ptu.number_records == 17861964
        assert ptu.pixels_in_frame == 262144
        assert ptu.pixels_in_line == 512
        assert ptu.syncrate == 20001617
        assert ptu.tcspc_resolution == 1.5999999936067155e-11

        # image of shape (3, 191, 512, 512, 3126) too large 875 GiB
        im = ptu.decode_image(channel=0, frame=92, dtime=-1, dtype='uint32')
        assert im.shape == (1, 512, 512, 1, 1)
        assert im.dtype == numpy.uint32
        assert im[0, 390, 277] == 6
        del im

    with PtuFile(filename, trimdims='T') as ptu:
        # trim only time dimension
        assert ptu.shape == (189, 512, 512, 4, 4096)
        assert ptu._info.skip_first_frame == 1
        assert ptu._info.skip_last_frame == 1
        im = ptu.decode_image(
            channel=0, frame=91, dtime=-1, dtype='uint8', keepdims=False
        )
        assert im.shape == (512, 512)
        assert im[390, 277] == 6


def test_issue_empty_line() -> None:
    """Test line not empty when first record is start marker."""
    filename = DATA / 'ExampleFLIM/Example_image.sc.ptu'
    with PtuFile(filename) as ptu:
        str(ptu)
        assert ptu.version == '00.0.1'
        assert ptu.type == PqFileType.PTU
        assert ptu.record_type == PtuRecordType.PicoHarpT3
        assert ptu.measurement_mode == PtuMeasurementMode.T3
        assert ptu.measurement_submode == PtuMeasurementSubMode.IMAGE
        assert ptu.scanner == PtuScannerType.LSM
        assert ptu.measurement_ndim == 3

        assert ptu.shape == (1, 256, 256, 1, 133)
        assert ptu.dims == ('T', 'Y', 'X', 'C', 'H')
        assert tuple(ptu.coords.keys()) == ('T', 'Y', 'X', 'C', 'H')
        assert ptu.coords['H'][1] == 9.696969697e-11  # 97 ps
        assert ptu.active_channels == (0,)

        assert ptu._info.skip_first_frame == 0
        assert ptu._info.skip_last_frame == 0

        assert ptu.frame_change_mask == 4
        assert ptu.line_start_mask == 1
        assert ptu.line_stop_mask == 2

        assert ptu.acquisition_time == 0.27299666752114843
        assert ptu.frame_time == 0.27299666752114843
        assert ptu.frequency == 78020000.0
        assert ptu.syncrate == 78020000
        assert ptu.number_markers == 513
        assert ptu.number_photons == 722402
        assert ptu.number_records == 723240
        assert ptu.global_pixel_time == 325  # 324
        assert ptu.pixel_time == pytest.approx(
            ptu.global_pixel_time * ptu.global_resolution, rel=1e-3
        )

        assert ptu.decode_records()['marker'][0] == 1  # start marker
        assert ptu[0, 0, 100, 0, ::-1] == 40  # first line not empty


def test_issue_pixeltime_zero() -> None:
    """Test PTU with zero ImgHdr_TimePerPixel."""
    filename = DATA / 'nc.picoquant.com/DaisyPollen1.ptu'
    with PtuFile(filename) as ptu:
        assert ptu.version == '1.0.00'
        assert ptu.type == PqFileType.PTU
        assert ptu.record_type == PtuRecordType.GenericT3
        assert ptu.measurement_mode == PtuMeasurementMode.T3
        assert ptu.measurement_submode == PtuMeasurementSubMode.IMAGE
        assert ptu.scanner is None
        assert ptu.measurement_ndim == 3

        assert ptu.tags['ImgHdr_TimePerPixel'] == 0  # nasty
        assert ptu.global_pixel_time == 160

        assert ptu.shape == (10, 512, 512, 2, 2510)
        assert ptu.dims == ('T', 'Y', 'X', 'C', 'H')
        assert tuple(ptu.coords.keys()) == ('T', 'Y', 'X', 'C', 'H')
        assert ptu.active_channels == (0, 1)
        # TODO: verify coords

        assert ptu._info.skip_first_frame == 1
        assert ptu._info.skip_last_frame == 1

        assert ptu.frame_change_mask == 4
        assert ptu.line_start_mask == 1
        assert ptu.line_stop_mask == 2

        assert ptu.acquisition_time == 17.6701460317873
        assert ptu.frame_time == 1.767014600678785
        assert ptu.frequency == 40000880.0
        assert ptu.global_frame_time == 70682139
        assert ptu.global_line_time == 81920
        assert ptu.global_pixel_time == 160
        assert ptu.global_resolution == 2.4999450012099732e-8
        assert ptu.pixel_time == pytest.approx(
            ptu.global_pixel_time * ptu.global_resolution, rel=1e-3
        )
        assert ptu.line_time == pytest.approx(
            ptu.global_line_time * ptu.global_resolution, rel=1e-3
        )
        assert ptu.lines_in_frame == 512
        assert ptu.number_bins == 2510
        assert ptu.number_bins_in_period == 2499
        assert ptu.number_bins_max == 32768
        assert ptu.number_channels == 2
        assert ptu.number_channels_max == 64
        assert ptu.number_lines == 5499
        assert ptu.number_markers == 11009
        assert ptu.number_photons == 37047472
        assert ptu.number_records == 37748736
        assert ptu.pixels_in_frame == 262144
        assert ptu.pixels_in_line == 512
        assert ptu.syncrate == 40000880
        assert ptu.tcspc_resolution == 9.999999960041972e-12

        im = ptu.decode_image(
            [9], channel=0, dtime=-1, keepdims=False, dtype='uint32'
        )
        assert im.shape == (512, 512)
        assert im.dtype == numpy.uint32
        assert im[281, 373] == 25

        im = ptu.decode_image(
            [9], channel=0, dtime=-1, pixel_time=0, keepdims=False, dtype='u4'
        )
        assert im.shape == (512, 512)
        assert im.dtype == numpy.uint32
        assert im[281, 373] == 25


def test_issue_pixeltime_off() -> None:
    """Test PTU with ImgHdr_TimePerPixel metadata slightly off."""
    filename = DATA / 'UNC/805_1.ptu'
    with PtuFile(filename) as ptu:
        assert ptu.version == '1.0.00'
        assert ptu.type == PqFileType.PTU
        assert ptu.record_type == PtuRecordType.TimeHarp260PT3
        assert ptu.measurement_mode == PtuMeasurementMode.T3
        assert ptu.measurement_submode == PtuMeasurementSubMode.IMAGE
        assert ptu.scanner == PtuScannerType.PI_E727
        assert ptu.measurement_ndim == 3

        assert ptu.tags['ImgHdr_TimePerPixel'] == 2.0  # slightly off
        assert ptu.global_pixel_time == 40000

        assert ptu.shape == (1, 256, 256, 2, 2002)
        assert ptu.dims == ('T', 'Y', 'X', 'C', 'H')
        assert tuple(ptu.coords.keys()) == ('T', 'Y', 'X', 'C', 'H')
        assert ptu.active_channels == (0, 1)

        assert ptu._info.line_time == 10232923
        assert ptu._info.skip_first_frame == 0
        assert ptu._info.skip_last_frame == 0

        assert ptu.frame_change_mask == 0  # !
        assert ptu.line_start_mask == 4
        assert ptu.line_stop_mask == 8

        assert ptu.acquisition_time == 172.07485649999998
        assert ptu.frame_time == 172.07485649999998
        assert ptu.frequency == 20000000.0
        assert ptu.global_frame_time == 3441497130
        assert ptu.global_line_time == 10240000
        assert ptu.global_pixel_time == 40000
        assert ptu.global_resolution == 5e-08
        assert ptu.pixel_time == pytest.approx(
            ptu.global_pixel_time * ptu.global_resolution, rel=1e-3
        )
        assert ptu.line_time == pytest.approx(
            ptu.global_line_time * ptu.global_resolution, rel=1e-3
        )
        assert ptu.lines_in_frame == 256
        assert ptu.number_bins == 2002
        assert ptu.number_bins_in_period == 1999
        assert ptu.number_bins_max == 32768
        assert ptu.number_channels == 2
        assert ptu.number_channels_max == 64
        assert ptu.number_lines == 256
        assert ptu.number_markers == 512
        assert ptu.number_photons == 56142925
        assert ptu.number_records == 59504234
        assert ptu.pixels_in_frame == 65536
        assert ptu.pixels_in_line == 256
        assert ptu.syncrate == 20000000
        assert ptu.tcspc_resolution == 2.50000003337858e-11

        data = binread(DATA / 'UNC/805.bin')[0].sum(axis=-1, dtype=numpy.int32)

        # use pixel time from metadata
        im = ptu.decode_image(channel=1, frame=0, keepdims=False)
        im = im[..., :2000].sum(axis=-1, dtype=numpy.int32)
        assert numpy.abs(im - data).max() == 298

        # use pixel time from average line time
        im = ptu.decode_image(
            channel=1,
            frame=0,
            pixel_time=ptu._info.line_time / 256 * ptu.global_resolution,
            keepdims=False,
        )
        im = im[..., :2000].sum(axis=-1, dtype=numpy.int32)
        assert numpy.abs(im - data).max() == 8

        # use pixel time from line markers
        im = ptu.decode_image(channel=1, frame=0, pixel_time=0, keepdims=False)
        im = im[..., :2000].sum(axis=-1, dtype=numpy.int32)
        assert numpy.abs(im - data).max() == 8


def test_issue_number_records_zero(caplog: pytest.LogCaptureFixture) -> None:
    """Test PTU with zero TTResult_NumberOfRecords."""
    # https://github.com/cgohlke/ptufile/issues/2
    filename = DATA / 'FLIM_number_records_zero.ptu'
    with PtuFile(filename) as ptu:
        ptu.cache_records = False
        with caplog.at_level(logging.WARNING):
            assert ptu.number_records == 12769472
        assert ptu.tags['TTResult_NumberOfRecords'] == 0
        assert 'invalid TTResult_NumberOfRecords' in caplog.text
        assert len(ptu.read_records()) == 12769472


def test_issue_number_records_negative(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test PTU with negative TTResult_NumberOfRecords."""
    # file >4 GB produced by LAS X software. Received by email on Oct 27, 2025.
    filename = DATA / 'i3S/AlessandroSlide_10x_488nm.ptu'
    with PtuFile(filename) as ptu:
        ptu.cache_records = False
        with caplog.at_level(logging.WARNING):
            assert ptu.number_records == 3167584182
        assert ptu.tags['TTResult_NumberOfRecords'] == -1127383114
        assert 'invalid TTResult_NumberOfRecords' in caplog.text
        assert len(ptu.read_records()) == 3167584182


@pytest.mark.parametrize('memmap', [False, True])
def test_issue_record_number(
    *, memmap: bool, caplog: pytest.LogCaptureFixture
) -> None:
    """Test PTU with too few records."""
    filename = DATA / 'Samples.sptw/Cy5_immo_FLIM+Pol-Imaging.ptu'
    with PtuFile(filename, memmap=memmap) as ptu:
        assert ptu.version == '00.0.0'
        with caplog.at_level(logging.ERROR):
            records = ptu.read_records()
            assert 'expected 3409856 items, got 3364091' in caplog.text
            assert len(records) == 3364091
        decoded = ptu.decode_records()
        assert decoded.size == 3364091
        assert decoded['time'][-1] == ptu.global_acquisition_time
        assert decoded['channel'].max() + 1 == ptu.number_channels
        assert decoded[decoded['channel'] >= 0].size == ptu.number_photons
        assert decoded[decoded['marker'] > 0].size == ptu.number_markers
        str(ptu)


def test_issue_tag_index_order(caplog: pytest.LogCaptureFixture) -> None:
    """Test sparse tag indices."""
    filename = DATA / 'picoquant-sample-data/hydraharp/v10_t2.ptu'
    with caplog.at_level(logging.WARNING):  # noqa: SIM117
        with PqFile(filename) as pq:
            str(pq)
            assert 'UsrHeadName' not in caplog.text
            assert pq.type == PqFileType.PTU
            assert pq.version == '1.0.00'
            assert pq.tags['UsrHeadName'] == [
                None,
                '405.0nm (DC405)',
                None,
                '485.0nm (DC485)',
            ]


def test_tag_parsing() -> None:
    """Test parsing mixed, out-of-order, and truncated tags."""
    from ptufile.ptufile import encode_tag

    magic = b'PQTTTR\x00\x001.0.00\x00\x00'
    header = b''.join(
        [
            magic,
            encode_tag('Scalar', 1),
            encode_tag('Scalar', 'x', 1),  # scalar followed by indexed
            encode_tag('Sparse', 'b1', 1),
            encode_tag('Sparse', 'b0', 0),  # out of order
            encode_tag('Header_End', None),
        ]
    )
    with PqFile(io.BytesIO(header)) as pq:
        assert pq.tags['Scalar'] == [None, 'x']
        assert pq.tags['Sparse'] == ['b0', 'b1']
        assert pq.attrs['guid'] is None  # no File_GUID tag

    truncated = magic + encode_tag('Scalar', 1)[:20]
    with pytest.raises(PqFileError):
        PqFile(io.BytesIO(truncated))


def test_ptu_t2_histogram_untrimmed() -> None:
    """Test decode T2 histogram without trimming channels."""
    filename = DATA / FILES[1]
    with PtuFile(filename) as ptu:
        trimmed = ptu.decode_histogram()
        first = ptu._info.channels_active_first
    with PtuFile(filename, trimdims='TH') as ptu:
        hist = ptu.decode_histogram()
        assert hist.shape[0] == ptu.number_channels_max
        assert array_equal(hist[first : first + trimmed.shape[0]], trimmed)
        if xarray is not None:
            xhist = ptu.decode_histogram(asxarray=True)
            assert xhist.shape[0] == ptu.number_channels_max


def test_ptuwriter_numpy_shape() -> None:
    """Test PtuWriter with numpy integer shape and z-stack."""
    data = RNG.integers(0, 4, (1, 2, 4, 4, 1, 8), dtype=numpy.uint8)
    buf = io.BytesIO()
    shape = tuple(numpy.int64(i) for i in data.shape)
    with PtuWriter(buf, shape, 1e-8, 1e-10, 1e-6) as ptu:  # type: ignore[arg-type]
        ptu.write(data)
    buf.seek(0)
    with PtuFile(buf, trimdims='') as ptu:
        assert ptu.number_planes == 2
        assert array_equal(ptu.decode_image()[..., :1, :8], data)


def test_decode_t3_line_unpaired_stop() -> None:
    """Test decode_t3_line ignores line stop without line start."""
    from ptufile._ptufile import decode_t3_line

    def marker(time: int, mask: int) -> int:
        return time | (mask << 16) | 0xF0000000

    def photon(time: int, dtime: int, channel: int = 0) -> int:
        return time | (dtime << 16) | ((channel + 1) << 28)

    records = numpy.array(
        [
            marker(0, 2),  # unpaired line stop
            marker(10, 1),
            photon(12, 3),
            marker(20, 2),
            marker(30, 1),
            photon(35, 1),
            marker(40, 2),
        ],
        dtype=numpy.uint32,
    )
    histogram = numpy.zeros((2, 2, 1, 4), numpy.uint16)
    times = numpy.zeros(2, numpy.uint64)
    decode_t3_line(
        histogram, times, records, PtuRecordType.PicoHarpT3, 5, 1, 2
    )
    assert histogram[0, 0, 0, 3] == 1
    assert histogram[1, 1, 0, 1] == 1
    assert histogram.sum() == 2
    assert list(times) == [10, 30]


@pytest.mark.skipif(xarray is None, reason='xarray not installed')
@pytest.mark.parametrize(
    ('dtime', 'size'),
    [
        (None, 139),  # last bin with non-zero photons
        (0, 132),  # last bin matching frequency
        (-1, 1),  # integrate bins
        (32, 32),  # specified number of bins
        (145, 145),
    ],
)
def test_issue_dtime(dtime: int | None, size: int) -> None:
    """Test dtime parameter."""
    filename = (
        DATA / 'napari_flim_phasor_plotter/hazelnut_FLIM_single_image.ptu'
    )
    im = imread(
        filename,
        frame=0,
        channel=0,
        dtime=dtime,
        dtype=numpy.uint8,
        asxarray=True,
    )
    assert im.dtype == numpy.uint8
    assert im.shape == (1, 256, 256, 1, size)
    assert im.dims == ('T', 'Y', 'X', 'C', 'H')
    assert tuple(im.coords.keys()) == ('T', 'Y', 'X', 'C', 'H')


def test_issue_line_markers() -> None:
    """Test nonsense line markers."""
    # This file apparently omits line stop markers if no photons were
    # recorded in the line.
    # The line stop markers don't match the timing expected from pixel time
    # and number of pixels in lines.
    # There are many more lines than expected from the frame height.
    # In conclusion, the line time from the `info` object is not reliable.

    filename = DATA / 'Tutorials.sptw/MicroBeads.ptu'
    with PtuFile(filename) as ptu:
        str(ptu)
        assert ptu._info.lines == 300
        assert ptu._info.line_time == 9000  # should be 18000
        assert ptu.global_line_time == 18000

        data = ptu.decode_image(pixel_time=0)
        assert data.shape == (2, 150, 150, 2, 626)
        assert data.sum(dtype=numpy.uint32) == 778696


def test_issue_imspector(caplog: pytest.LogCaptureFixture) -> None:
    """Test file written by Seidel group using Abberior Imspector software."""
    # This file stores wrong marker masks and twice as many lines per frame.
    # Frames and channels need to be "deinterlaced".

    filename = DATA / 'HHU/PQSpcm_2021-12-13_17-53-45.ptu'
    with PtuFile(filename) as ptu:
        assert ptu.tags['CreatorSW_Name'] == 'Imspector'
        assert ptu.tags['ImgHdr_LineStart'] == 0  # invalid
        assert ptu.tags['ImgHdr_LineStop'] == 1  # wrong
        assert ptu.tags['ImgHdr_Frame'] == 2  # wrong
        assert ptu.tags['ImgHdr_PixY'] == 100  # wrong
        assert ptu._info.frames == 0
        assert ptu._info.lines == 0
        str(ptu)

        image = ptu.decode_image(dtime=-1, frame=-1, keepdims=False)
        assert 'invalid line_start' in caplog.text
        assert image.sum() == 0  # empty

    with PtuFile(filename) as ptu:
        # overwrite invalid header values before inspecting or decoding
        ptu.tags['ImgHdr_LineStart'] = 1
        ptu.tags['ImgHdr_LineStop'] = 2
        ptu.tags['ImgHdr_Frame'] = 3
        ptu.tags['ImgHdr_PixY'] *= 2

        assert ptu._info.frames == 61
        assert ptu._info.lines == 12200

        image = ptu.decode_image(dtime=-1, frame=-1, keepdims=False)
        assert image.sum() == 24953

        # deinterlace lines and channels
        channel = [
            image[::2, :, 0] + image[::2, :, 2],
            image[::2, :, 1] + image[::2, :, 3],
            image[1::2, :, 0] + image[1::2, :, 2],
            image[1::2, :, 1] + image[1::2, :, 3],
        ]
        assert channel[0].shape == (100, 100)
        assert channel[0].sum() == 7272
        assert channel[1].sum() == 6658
        assert channel[2].sum() == 3789
        assert channel[3].sum() == 7234


@pytest.mark.skipif(xarray is None, reason='xarray not installed')
def test_imread() -> None:
    """Test imread function."""
    filename = (
        DATA / 'napari_flim_phasor_plotter/hazelnut_FLIM_single_image.ptu'
    )
    im = imread(
        filename,
        [slice(1), None],  # first frame
        channel=0,
        frame=None,
        dtime=0,
        pixel_time=0,
        dtype=numpy.uint8,
        asxarray=True,
    )
    assert im.dtype == numpy.uint8
    assert im.shape == (1, 256, 256, 1, 132)
    assert im.dims == ('T', 'Y', 'X', 'C', 'H')
    assert tuple(im.coords.keys()) == ('T', 'Y', 'X', 'C', 'H')


@pytest.mark.parametrize('pixel_time', [None, 0.1])
@pytest.mark.parametrize('counts', [0, 1, 87])
@pytest.mark.parametrize(
    'shape',
    [
        (31, 33, 63),
        (5, 31, 33, 63),
        (31, 33, 2, 63),
        (5, 31, 33, 2, 63),
        (2, 3, 31, 33, 2, 63),
    ],
)
@pytest.mark.parametrize(
    'record_type', [PtuRecordType.PicoHarpT3, PtuRecordType.GenericT3]
)
def test_imwrite(
    record_type: PtuRecordType,
    pixel_time: float | None,
    shape: tuple[int, ...],
    counts: int,
) -> None:
    """Test imwrite function."""
    if counts:
        data = RNG.integers(0, counts, shape, numpy.uint8)
    else:
        data = numpy.zeros(shape, numpy.uint8)

    dims = 'TYXH' if data.ndim == 4 and data.shape[0] == 5 else None
    if data.ndim == 3:
        shape = (1, 31, 33, 1, 63)
        if counts:
            data[..., -1] = 1
    elif data.ndim == 4:
        if dims:
            shape = (5, 31, 33, 1, 63)
            if counts:
                data[..., -1] = 1
        else:
            shape = (1, 31, 33, 2, 63)
            if counts:
                data[..., :, -1] = 1
    else:
        shape = data.shape
        if counts:
            data[..., :, -1] = 1

    guid = '{b767c46e-9693-4ad9-9fcf-7fab5e4377fc}'
    comment = f'{shape=} \u2764\ufe0f'

    buf = io.BytesIO()
    imwrite(
        buf,
        data,
        global_resolution=4e-8,
        tcspc_resolution=8e-11,
        record_type=record_type,
        pixel_time=pixel_time,
        pixel_resolution=0.5,
        dims=dims,
        comment=comment,
        guid=guid,
        tags={
            'HW_Markers': 4,
            'HWMarkers_Enabled': [True, True, True, True],
            'TTResult_MDescWarningFlags': PtuMeasurementWarnings(2),
        },
    )
    buf.seek(0)

    data = data.reshape(shape)
    if counts == 0:
        data = data[..., :1, :1]
    zstack = data.ndim == 6

    with PtuFile(buf) as ptu:
        str(ptu)
        assert ptu.version == '1.0.00'
        assert ptu.type == PqFileType.PTU
        assert ptu.record_type == record_type
        assert ptu.measurement_mode == PtuMeasurementMode.T3
        assert ptu.measurement_submode == PtuMeasurementSubMode.IMAGE
        assert ptu.scanner == PtuScannerType.LSM
        assert ptu.measurement_ndim == (4 if zstack else 3)
        assert ptu.is_image
        assert ptu.is_t3
        assert ptu.comment == comment
        assert ptu.shape == data.shape
        assert str(ptu.guid) == guid[1:-1]

        assert ptu.global_resolution == 4e-8
        if pixel_time is not None:
            assert ptu.pixel_time == pixel_time
            assert ptu.global_line_time == pixel_time / 4e-8 * shape[-3]
            assert ptu._info.line_time == ptu.global_line_time

        assert ptu.tags['ImgHdr_MaxFrames'] == (
            shape[0] * shape[1] if zstack else shape[0]
        )
        assert ptu.tags['HW_InpChannels'] == shape[-2] + 1
        assert ptu.tags['ImgHdr_PixResol'] == 0.5
        assert ptu.tags['HW_Markers'] == 4
        assert ptu.tags['HWMarkers_Enabled'] == [True, True, True, True]
        assert ptu.tags['TTResult_MDescWarningFlags'] == 2

        data2 = ptu.decode_image()
        assert array_equal(data2, data)
        data2 = ptu.decode_image(pixel_time=0.0)
        assert array_equal(data2, data)


@pytest.mark.parametrize(
    'record_type', [PtuRecordType.PicoHarpT3, PtuRecordType.GenericT3]
)
def test_imwrite_rewrite(record_type: PtuRecordType) -> None:
    """Test imwrite function with real data."""
    filename = DATA / 'Tutorials.sptw/Kidney _Cell_FLIM.ptu'

    buf = io.BytesIO()
    with PtuFile(filename) as ptu0:
        data = ptu0.decode_image()
        assert data.shape == (3, 512, 512, 2, 501)

        imwrite(
            buf,
            data,
            global_resolution=ptu0.global_resolution,  # 4.00001280004096e-8
            tcspc_resolution=ptu0.tcspc_resolution,  # 7.999999968033578e-11
            record_type=record_type,
            pixel_time=ptu0.pixel_time,  # 3e-5
            pixel_resolution=ptu0.tags['ImgHdr_PixResol'],  # 0.504453125
            guid=ptu0.guid,
            datetime=ptu0.datetime,
            comment=ptu0.comment,
            tags={'File_RawData_GUID': [ptu0.guid]},
        )
        buf.seek(0)

        with PtuFile(buf) as ptu:
            str(ptu)
            assert ptu.version == '1.0.00'
            assert ptu.type == PqFileType.PTU
            assert ptu.record_type == record_type
            assert ptu.measurement_mode == PtuMeasurementMode.T3
            assert ptu.measurement_submode == PtuMeasurementSubMode.IMAGE
            assert ptu.scanner == PtuScannerType.LSM
            assert ptu.measurement_ndim == 3
            assert ptu.is_image
            assert ptu.is_t3
            assert ptu.shape == (3, 512, 512, 2, 501)
            assert ptu.dims == ('T', 'Y', 'X', 'C', 'H')
            assert tuple(ptu.coords.keys()) == ('T', 'Y', 'X', 'C', 'H')
            assert ptu.active_channels == (0, 1)
            assert array_equal(ptu.coords['C'], (0, 1))
            # TODO: verify coords

            assert ptu.guid == ptu0.guid
            assert ptu.comment == ptu0.comment
            assert ptu.datetime == ptu0.datetime

            assert ptu.line_start_mask == 1
            assert ptu.line_stop_mask == 2
            assert ptu.frame_change_mask == 4

            assert ptu._info.line_time == 384000

            assert ptu.acquisition_time == 23.593035497713593
            assert ptu.frame_time == 7.864345165904531
            assert ptu.frequency == 24999920.0
            assert ptu.global_frame_time == 196608000
            assert ptu.global_line_time == 384000
            assert ptu.global_pixel_time == 750
            assert ptu.global_resolution == 4.00001280004096e-8
            assert ptu.pixel_time == pytest.approx(
                ptu.global_pixel_time * ptu.global_resolution, rel=1e-3
            )
            assert ptu.line_time == pytest.approx(
                ptu.global_line_time * ptu.global_resolution, rel=1e-3
            )
            assert ptu.lines_in_frame == 512
            assert ptu.number_bins == 501
            assert ptu.number_bins_in_period == 500
            assert ptu.number_channels == 2
            assert ptu.number_lines == 1536
            assert ptu.number_markers == 3075
            assert ptu.number_photons == data.sum(dtype=numpy.uint32)
            assert ptu.number_records in {21116856, 21683856}
            assert ptu.pixels_in_frame == 262144
            assert ptu.pixels_in_line == 512
            assert ptu.syncrate == 24999920
            assert ptu.tcspc_resolution == 7.999999968033578e-11

            assert ptu.tags['File_RawData_GUID'][0] == (
                '{b767c46e-9693-4ad9-9fcf-7fab5e4377fc}'
            )
            assert ptu.tags['ImgHdr_PixResol'] == 0.504453125

            data2 = ptu.decode_image()

    assert array_equal(data2, data)


@pytest.mark.parametrize('count', [0, 1, 87])
@pytest.mark.parametrize(
    'record_type', [PtuRecordType.PicoHarpT3, PtuRecordType.GenericT3]
)
def test_imwrite_pixel(record_type: PtuRecordType, count: int) -> None:
    """Test imwrite function with one pixel."""
    data = numpy.full((1, 1, 1, 1, 1), count, numpy.uint8)

    buf = io.BytesIO()
    imwrite(buf, data, 4e-8, 8e-11, record_type=record_type)
    buf.seek(0)

    with PtuFile(buf) as ptu:
        str(ptu)
        assert ptu.version == '1.0.00'
        assert ptu.type == PqFileType.PTU
        assert ptu.record_type == record_type
        assert ptu.measurement_mode == PtuMeasurementMode.T3
        assert ptu.measurement_submode == PtuMeasurementSubMode.IMAGE
        assert ptu.scanner == PtuScannerType.LSM
        assert ptu.measurement_ndim == 3
        assert ptu.is_image
        assert ptu.is_t3
        assert ptu.shape == (1, 1, 1, 1, 1)
        assert ptu.number_photons == count
        assert ptu.number_records == count + 3
        records = ptu.decode_records()
        histogram = ptu.decode_histogram()
        image = ptu.decode_image()

    assert records.size == count + 3
    assert histogram.shape == (1, 1)
    assert array_equal(image, data)


def test_imwrite_zstack() -> None:
    """Test imwrite function with z-stack data."""
    t, z, y, x, c, h = 2, 3, 16, 16, 2, 32
    data = RNG.integers(0, 3, (t, z, y, x, c, h), numpy.uint16)
    data[..., -1] = 1

    buf = io.BytesIO()
    imwrite(
        buf,
        data,
        global_resolution=4e-8,
        tcspc_resolution=8e-11,
        pixel_time=0.1,
    )
    buf.seek(0)

    with PtuFile(buf) as ptu:
        assert ptu.measurement_ndim == 4
        assert ptu.tags['ImgHdr_Dimensions'] == 4
        assert ptu.tags['ImgHdr_PixZ'] == z
        assert ptu.tags['ImgHdr_MaxFrames'] == t * z
        assert ptu.tags['ImgHdr_PixX'] == x
        assert ptu.tags['ImgHdr_PixY'] == y
        assert ptu.shape == (t, z, y, x, c, h)
        assert ptu.dims == ('T', 'Z', 'Y', 'X', 'C', 'H')
        assert ptu.number_photons == int(data.sum())
        assert array_equal(ptu.decode_image(), data)
        assert array_equal(ptu.decode_image(pixel_time=0.0), data)
        assert array_equal(ptu.decode_image(frame=1, plane=2), data[1:2, 2:3])
        assert array_equal(
            ptu.decode_image(frame=-1, plane=-1, dtype='uint32')[0, 0],
            data.sum(axis=(0, 1), dtype=numpy.uint32),
        )


def test_write_zstack_iterate() -> None:
    """Test PtuWriter writing z-stack one volume at a time."""
    t, z, y, x, c, h = 2, 3, 16, 16, 2, 32
    data = numpy.ones((t, z, y, x, c, h), numpy.uint16)

    buf = io.BytesIO()
    with PtuWriter(buf, (1, z, y, x, c, h), 4e-8, 8e-11, 0.1) as ptu_out:
        for t_idx in range(t):
            ptu_out.write(data[t_idx])
    buf.seek(0)

    with PtuFile(buf) as ptu:
        assert ptu.measurement_ndim == 4
        assert ptu.tags['ImgHdr_Dimensions'] == 4
        assert ptu.tags['ImgHdr_PixZ'] == z
        assert ptu.tags['ImgHdr_MaxFrames'] == t * z
        assert ptu.shape == (t, z, y, x, c, h)
        assert ptu.dims == ('T', 'Z', 'Y', 'X', 'C', 'H')
        assert ptu.number_photons == int(data.sum())
        assert array_equal(ptu.decode_image(), data)


@pytest.mark.parametrize(
    'record_type', [PtuRecordType.PicoHarpT3, PtuRecordType.GenericT3]
)
@pytest.mark.parametrize('chunks', [(1, 1, 1, 1, 1, 1), (2, 1, 3)])
def test_write_zstack_planes(
    record_type: PtuRecordType, chunks: tuple[int, ...]
) -> None:
    """Test PtuWriter writing z-stack in chunks of planes."""
    t, z, y, x, c, h = 2, 3, 8, 7, 2, 32
    data = RNG.integers(0, 3, (t, z, y, x, c, h), numpy.uint16)
    data[..., -1] = 1
    planes = data.reshape(t * z, y, x, c, h)

    buf = io.BytesIO()
    with PtuWriter(
        buf, (1, z, y, x, c, h), 4e-8, 8e-11, 1e-5, record_type=record_type
    ) as ptu_out:
        index = 0
        for n in chunks:
            chunk = planes[index : index + n]
            ptu_out.write(chunk[0] if n == 1 else chunk)
            index += n
    buf.seek(0)

    with PtuFile(buf) as ptu:
        assert ptu.tags['ImgHdr_PixZ'] == z
        assert ptu.tags['ImgHdr_MaxFrames'] == t * z
        assert ptu.shape == (t, z, y, x, c, h)
        assert ptu.number_photons == int(data.sum())
        assert array_equal(ptu.decode_image(), data)
        assert array_equal(ptu.decode_image(pixel_time=0.0), data)
        records = ptu.decode_records()
        # overflow records report time + count for GenericT3
        events = (records['channel'] >= 0) | (records['marker'] > 0)
        times = records['time'][events].astype(numpy.int64)
        assert numpy.all(numpy.diff(times) >= 0)

    buf2 = io.BytesIO()
    with PtuWriter(
        buf2, (1, z, y, x, c, h), 4e-8, 8e-11, 1e-5, record_type=record_type
    ) as ptu_out:
        ptu_out.write(data)
    with PtuFile(buf2) as ptu2:
        records2 = ptu2.decode_records()
    # chunking must not change encoded records
    assert array_equal(records2, records)


def test_write_frame_size_mismatch() -> None:
    """Test PtuWriter.write with data not a multiple of frame size."""
    buf = io.BytesIO()
    with PtuWriter(buf, (1, 3, 4, 5, 1, 6), 4e-8, 8e-11, 1e-6) as ptu:
        with pytest.raises(ValueError):
            ptu.write(numpy.ones((4, 5, 1, 7), numpy.uint8))
        ptu.write(numpy.ones((3, 4, 5, 1, 6), numpy.uint8))


def test_write_zstack_incomplete(caplog: pytest.LogCaptureFixture) -> None:
    """Test PtuWriter warns about incomplete z-stack."""
    t, z, y, x, c, h = 1, 3, 4, 5, 1, 6
    data = numpy.ones((t, z, y, x, c, h), numpy.uint8)

    buf = io.BytesIO()
    with caplog.at_level(logging.WARNING):
        with PtuWriter(buf, data.shape, 4e-8, 8e-11, 1e-6) as ptu_out:
            ptu_out.write(data[0, :2])
        assert 'not a multiple of nplanes=3' in caplog.text
    buf.seek(0)

    with PtuFile(buf) as ptu:
        assert ptu.tags['ImgHdr_MaxFrames'] == 2
        assert ptu.number_photons == int(data[0, :2].sum())


def test_write_none() -> None:
    """Test PtuWriter with no data."""
    buf = io.BytesIO()
    with PtuWriter(buf, (2, 3, 4, 5, 6), 4e-8, 8e-11, 1e-6) as ptu:
        pass
    buf.seek(0)

    with PtuFile(buf) as ptu:
        str(ptu)
        assert ptu.version == '1.0.00'
        assert ptu.type == PqFileType.PTU
        assert ptu.record_type == PtuRecordType.GenericT3
        assert ptu.measurement_mode == PtuMeasurementMode.T3
        assert ptu.measurement_submode == PtuMeasurementSubMode.IMAGE
        assert ptu.scanner == PtuScannerType.LSM
        assert ptu.measurement_ndim == 3
        assert ptu.is_image
        assert ptu.is_t3
        assert ptu.shape == (1, 3, 4, 1, 1)
        assert ptu.number_photons == 0
        assert ptu.number_records == 0

        records = ptu.decode_records()
        assert len(records) == 0

        histogram = ptu.decode_histogram()
        assert array_equal(histogram, numpy.zeros((1, 1), numpy.uint32))

        image = ptu.decode_image()
        assert array_equal(image, numpy.zeros((1, 3, 4, 1, 1), numpy.uint16))


def test_write_iterate() -> None:
    """Test PtuWriter.write iteratively."""
    filename = DATA / 'Samples.sptw/GUVs.ptu'

    with tempfile.NamedTemporaryFile(suffix='.ptu', delete=False) as tmp:
        fnout = tmp.name

    try:
        with PtuFile(filename) as ptu_in:
            assert ptu_in.shape == (100, 512, 512, 2, 4096)

            with PtuWriter(
                fnout,
                ptu_in.shape[1:],
                global_resolution=ptu_in.global_resolution,
                tcspc_resolution=ptu_in.tcspc_resolution,
                pixel_time=ptu_in.pixel_time,
            ) as ptu_out:
                for i in range(2):
                    frame = ptu_in.decode_image(frame=i)
                    assert frame.shape == (1, 512, 512, 2, 4096)  # 4 GB
                    ptu_out.write(frame)

        with PtuFile(fnout) as ptu:
            assert ptu.version == '1.0.00'
            assert ptu.type == PqFileType.PTU
            assert ptu.record_type == PtuRecordType.PicoHarpT3
            assert ptu.measurement_mode == PtuMeasurementMode.T3
            assert ptu.measurement_submode == PtuMeasurementSubMode.IMAGE
            assert ptu.scanner == PtuScannerType.LSM
            assert ptu.measurement_ndim == 3
            assert ptu.is_image
            assert ptu.is_t3
            assert ptu.tags['ImgHdr_MaxFrames'] == 2
            assert ptu.number_records == 685208
            assert ptu.shape == (2, 512, 512, 2, 4096)
            assert array_equal(
                ptu.decode_image(frame=1, dtime=-1, dtype=numpy.uint16),
                frame.sum(axis=-1, dtype=numpy.uint16, keepdims=True),
            )
    finally:
        if os.path.exists(fnout):
            os.remove(fnout)


def test_imwrite_exceptions() -> None:
    """Test imwrite function exceptions."""
    buf = io.BytesIO()
    kwargs: dict[str, Any] = {
        'global_resolution': 4e-8,
        'tcspc_resolution': 8e-11,
    }

    with pytest.raises(ValueError):
        # cannot write to
        imwrite([], numpy.empty((31, 33, 1), numpy.uint8), **kwargs)  # type: ignore[arg-type]

    with pytest.raises(ValueError):
        # not unsigned int
        imwrite(buf, numpy.empty((31, 33, 1), numpy.int8), **kwargs)

    with pytest.raises(ValueError):
        # not enough dimensions
        imwrite(buf, numpy.empty((31, 33), numpy.uint8), **kwargs)

    with pytest.raises(ValueError):
        # too many dimensions
        imwrite(
            buf, numpy.empty((1, 1, 31, 33, 1, 1, 1), numpy.uint8), **kwargs
        )

    with pytest.raises(ValueError):
        # too many channels
        imwrite(buf, numpy.empty((1, 31, 33, 64, 1), numpy.uint8), **kwargs)

    with pytest.raises(ValueError):
        # too many bins
        imwrite(buf, numpy.empty((1, 1, 1, 1, 32769), numpy.uint8), **kwargs)

    with pytest.raises(ValueError):
        # invalid record_type
        imwrite(
            buf,
            numpy.empty((31, 33, 1), numpy.uint8),
            record_type=PtuRecordType.GenericT2,
            **kwargs,
        )

    with pytest.raises(ValueError):
        # invalid global_resolution
        imwrite(
            buf,
            numpy.empty((31, 33, 1), numpy.uint8),
            global_resolution=0.0,
            tcspc_resolution=8e-11,
        )

    with pytest.raises(ValueError):
        # tcspc_resolution > global_resolution
        imwrite(
            buf,
            numpy.empty((31, 33, 1), numpy.uint8),
            global_resolution=4e-8,
            tcspc_resolution=8e-6,
        )

    with pytest.raises(ValueError):
        # pixel_time < global_resolution
        imwrite(
            buf,
            numpy.empty((31, 33, 1), numpy.uint8),
            global_resolution=4e-8,
            tcspc_resolution=8e-11,
            pixel_time=1e-9,
        )

    # with pytest.raises(ValueError):
    #     # global_pixel_time=250 < photons_in_pixel=255
    #     imwrite(
    #         buf,
    #         numpy.full((31, 33, 1), 255, numpy.uint8),
    #         pixel_time=1e-5,
    #         **kwargs,
    #     )

    with pytest.raises(ValueError):
        # invalid guid
        imwrite(buf, numpy.empty((31, 33, 1), numpy.uint8), guid='-', **kwargs)

    with pytest.raises(ValueError):
        # dims length does not match shape ndim
        imwrite(
            buf,
            numpy.empty((31, 33, 63), numpy.uint8),
            dims='TYXH',
            **kwargs,
        )

    with pytest.raises(ValueError):
        # dims contains invalid axis labels
        imwrite(
            buf,
            numpy.empty((31, 33, 1, 63), numpy.uint8),
            dims='ABCD',
            **kwargs,
        )

    with pytest.raises(ValueError):
        # dims not in canonical TZYXCH order
        imwrite(
            buf,
            numpy.empty((31, 33, 63), numpy.uint8),
            dims='HYX',
            **kwargs,
        )


def test_signal_from_ptu() -> None:
    """Test PhasorPy signal_from_ptu function."""
    try:
        from phasorpy.io import signal_from_ptu
    except ImportError:
        pytest.skip('PhasorPy not installed')

    filename = (
        DATA / 'napari_flim_phasor_plotter/hazelnut_FLIM_single_image.ptu'
    )
    signal = signal_from_ptu(
        filename, frame=-1, channel=0, dtime=0, keepdims=False
    )
    assert signal.values.sum(dtype=numpy.uint64) == 6064854
    assert signal.dtype == numpy.uint16
    assert signal.shape == (256, 256, 132)
    assert signal.dims == ('Y', 'X', 'H')
    assert_almost_equal(
        signal.coords['H'].data[[1, -1]], [0.0969697, 12.7030303], decimal=4
    )
    assert signal.attrs['frequency'] == 78.02
    assert signal.attrs['ptu_tags']['HW_Type'] == 'PicoHarp'

    signal = signal_from_ptu(
        filename,
        frame=-1,
        channel=0,
        dtime=None,
        keepdims=True,
        trimdims='TC',
    )
    assert signal.values.sum(dtype=numpy.uint64) == 6065123
    assert signal.dtype == numpy.uint16
    assert signal.shape == (1, 256, 256, 1, 4096)
    assert signal.dims == ('T', 'Y', 'X', 'C', 'H')
    assert_almost_equal(
        signal.coords['H'].data[[1, -1]], [0.0969697, 397.09091], decimal=4
    )
    assert signal.attrs['frequency'] == 78.02


def test_signal_from_ptu_irf() -> None:
    """Test read PhasorPy signal_from_ptu function with IRF."""
    try:
        from phasorpy.io import signal_from_ptu
    except ImportError:
        pytest.skip('PhasorPy not installed')

    filename = DATA / 'Samples.sptw/Cy5_diff_IRF+FLCS-pattern.ptu'
    signal = signal_from_ptu(filename, channel=None, keepdims=True)
    assert signal.values.sum(dtype=numpy.uint64) == 13268548
    assert signal.dtype == numpy.uint32
    assert signal.shape == (1, 1, 1, 2, 6250)
    assert signal.dims == ('T', 'Y', 'X', 'C', 'H')
    assert_almost_equal(
        signal.coords['H'].data[[1, -1]], [0.007999, 49.991999], decimal=4
    )
    assert pytest.approx(signal.attrs['frequency'], abs=1e-4) == 19.999732
    assert signal.attrs['ptu_tags']['HW_Type'] == 'PicoHarp 300'

    signal = signal_from_ptu(filename, channel=0, keepdims=True)
    assert signal.values.sum(dtype=numpy.uint64) == 6984849
    assert signal.shape == (1, 1, 1, 1, 6250)
    assert signal.dims == ('T', 'Y', 'X', 'C', 'H')

    with pytest.raises(ValueError):
        signal_from_ptu(filename, dtime=-1)

    signal = signal_from_ptu(filename, channel=0, dtime=None, keepdims=False)
    assert signal.values.sum(dtype=numpy.uint64) == 6984849
    assert signal.shape == (1, 1, 4096)
    assert signal.dims == ('Y', 'X', 'H')
    assert_almost_equal(
        signal.coords['H'].data[[1, -1]], [0.007999, 32.759999], decimal=4
    )


def test_ptu_measurement_submode() -> None:
    """Test PtuMeasurementSubMode enum, including undefined values."""
    assert PtuMeasurementSubMode(1) is PtuMeasurementSubMode.POINT
    assert PtuMeasurementSubMode(3) is PtuMeasurementSubMode.IMAGE
    assert PtuMeasurementSubMode(-1) is PtuMeasurementSubMode.UNKNOWN
    # 0 is an alias for POINT
    assert PtuMeasurementSubMode(0) is PtuMeasurementSubMode.POINT
    assert PtuMeasurementSubMode(numpy.int64(0)) is PtuMeasurementSubMode.POINT  # type: ignore[arg-type]
    # undefined values map to UNKNOWN pseudo-members preserving the value
    mode = PtuMeasurementSubMode(42)
    assert mode.name == 'UNKNOWN'
    assert mode.value == 42
    assert mode == 42
    assert isinstance(mode, PtuMeasurementSubMode)
    # undefined values must not overwrite the UNKNOWN member value
    assert PtuMeasurementSubMode.UNKNOWN.value == -1
    assert PtuMeasurementSubMode(numpy.int32(7)).value == 7  # type: ignore[arg-type]
    assert PtuMeasurementSubMode.UNKNOWN.value == -1
    with pytest.raises(ValueError):
        PtuMeasurementSubMode('3')  # type: ignore[arg-type]


@pytest.mark.parametrize(
    'filename',
    tuple(
        itertools.chain.from_iterable(
            glob.glob(f'**/*{ext}', root_dir=DATA, recursive=True)
            for ext in FILE_EXTENSIONS
        )
    ),
)
def test_glob(filename: str) -> None:
    """Test read all PicoQuant files."""
    filename = str(DATA / filename)
    if 'htmlcov' in filename or 'url' in filename or 'defective' in filename:
        pytest.skip()
    with PqFile(filename, memmap=True) as pq:
        str(pq)
        is_ptu = pq.type == PqFileType.PTU
    if is_ptu:
        with PtuFile(filename, memmap=True) as ptu:
            str(ptu)


@pytest.mark.parametrize(
    ('trimdims', 'dtime', 'size'), [('TC', None, 4096), ('TCH', 0, 132)]
)
def test_ptu_zip_sequence(trimdims: str, dtime: int | None, size: int) -> None:
    """Test read Z-stack with imread and tifffile.FileSequence."""
    # requires ~28GB. Do not trim H dimensions such that files match
    from tifffile import FileSequence

    filename = DATA / 'napari_flim_phasor_plotter/hazelnut_FLIM_z_stack.zip'
    with FileSequence(imread, '*.ptu', container=filename) as ptus:
        assert ptus.shape == (11,)
        stack = ptus.asarray(
            channel=0, trimdims=trimdims, dtime=dtime, ioworkers=1
        )
    assert stack.shape == (11, 5, 256, 256, 1, size)  # 11 files, 5 frames each
    assert stack.dtype == 'uint16'


def test_ptu_leica_sequence() -> None:
    """Test read Leica TZ-stack with imread and tifffile.imread."""
    # 410 files. Requires ~16GB.
    import tifffile  # >= 2024.2.12

    filename = DATA / 'Flipper TR time series.sptw/*.ptu'
    stack = tifffile.imread(
        str(filename),  # glob pattern needs to be str
        pattern=r'_(t)(\d+)_(z)(\d+)',
        imread=imread,
        chunkshape=(512, 512, 132),  # shape returned by imread
        chunkdtype='uint8',  # dtype returned by imread
        ioworkers=None,  # use multi-threading
        imreadargs={
            'frame': 0,
            'channel': 0,
            'dtime': 132,  # fix number of bins to 132
            'dtype': 'uint8',  # request uint8 output
            'keepdims': False,
        },
    )
    assert stack.shape == (41, 10, 512, 512, 132)
    assert stack.dtype == 'uint8'
    assert stack[24, 4, 228, 279, 16] == 3


def test_ptu_numcodecs() -> None:
    """Test Leica TZ-stack with tifffile.ZarrFileSequenceStore and fsspec."""
    # 410 files. Requires ~16GB.
    try:
        import tifffile
        import tifffile.zarr
        import zarr
        from kerchunk.utils import refs_as_store
    except ImportError:
        pytest.skip('fsspec, tifffile, or zarr not installed')

    pathname = DATA / 'Flipper TR time series.sptw'
    url = str(pathname).replace('\\', '/')
    jsonfile = str(pathname / 'FLIPPER.json')
    filename = str(pathname / '*.ptu')  # glob pattern needs to be str

    store: Any = tifffile.imread(
        filename,
        pattern=r'_(t)(\d+)_(z)(\d+)',
        imread=imread,
        chunkshape=(512, 512),  # shape returned by imread
        chunkdtype='uint8',  # dtype returned by imread
        imreadargs={
            'frame': 0,
            'channel': 0,
            'dtime': -1,
            'pixel_time': None,
            'dtype': 'uint8',  # request uint8 output
            'trimdims': None,
            'keepdims': False,
        },
        aszarr=True,
    )
    assert isinstance(store, tifffile.zarr.ZarrFileSequenceStore)
    store.write_fsspec(
        jsonfile,
        url=url,
        version=1,
        codec_id='ptufile',
        quote=False,
    )
    store.close()

    ptufile.numcodecs.register_codec()
    stack: Any = zarr.open(refs_as_store(jsonfile), mode='r')
    assert stack.shape == (41, 10, 512, 512)
    assert stack.dtype == 'uint8'
    assert stack[24, 4, 228, 279] == 18


@pytest.mark.skipif(
    not hasattr(sys, '_is_gil_enabled'), reason='Python < 3.13'
)
def test_gil_enabled() -> None:
    """Test that GIL state is consistent with build configuration."""
    assert sys._is_gil_enabled() != sysconfig.get_config_var('Py_GIL_DISABLED')


if __name__ == '__main__':
    import warnings

    # warnings.simplefilter('always')
    warnings.filterwarnings('ignore', category=ImportWarning)
    argv = sys.argv
    argv.append('--cov-report=html')
    argv.append('--cov=ptufile')
    argv.append('--verbose')
    sys.exit(pytest.main(argv))
