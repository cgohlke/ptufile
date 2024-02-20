# test_ptufile.py

# Copyright (c) 2023-2024, Christoph Gohlke
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright
#   notice, this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright
#   notice, this list of conditions and the following disclaimer in the
#   documentation and/or other materials provided with the distribution.
# * Neither the name of the copyright holders nor the names of any
#   contributors may be used to endorse or promote products derived
#   from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

"""Unittests for the ptufile package.

:Version: 2024.2.20

"""

import glob
import io
import logging
import os
import pathlib
import sys

import numpy
import ptufile
import ptufile.numcodecs
import pytest
import xarray
from numpy.testing import assert_array_equal
from ptufile import (
    PhuFile,
    PhuMeasurementMode,
    PhuMeasurementSubMode,
    PqFile,
    PqFileError,
    PqFileMagic,
    PtuFile,
    PtuMeasurementMode,
    PtuMeasurementSubMode,
    PtuRecordType,
    PtuScannerType,
    imread,
)

HERE = pathlib.Path(os.path.dirname(__file__))

FILES = [
    # PicoHarpT3
    'napari_flim_phasor_plotter/hazelnut_FLIM_single_image.ptu',
    # MultiHarpT3
    'Tutorials.sptw/Hyperosmotic_Shock_MDCK_Cells.ptu',
    # PicoHarpT2
    'Samples.sptw/Atto488_diff_cw_total_correlation.ptu',
    # HydraHarpT2
    'Samples.sptw/NV-Center_for_Antibunching_several.ptu',
]


@pytest.mark.skipif(__doc__ is None, reason='__doc__ is None')
def test_version():
    """Assert ptufile versions match docstrings."""
    ver = ':Version: ' + ptufile.__version__
    assert ver in __doc__
    assert ver in ptufile.__doc__


def test_non_pqfile():
    """Test read non-PicoQuant file fails."""
    fname = HERE / 'FRET_GFP and mRFP.pt3'
    with pytest.raises(PqFileError):
        with PqFile(fname):
            pass


def test_non_ptu():
    """Test read non-PTU file fails."""
    fname = HERE / 'Settings.pfs'
    with pytest.raises(PqFileError):
        with PtuFile(fname):
            pass


def test_pq_fastload():
    """Test read tags using fastload."""
    fname = HERE / 'napari_flim_phasor_plotter/hazelnut_FLIM_single_image.ptu'
    with PqFile(fname, fastload=True) as pq:
        str(pq)
        assert pq.tags['File_GUID'] == '{4f6e5f68-8289-483d-9d9a-7974b77ef8b8}'
        assert 'HW_ExternalRefClock' not in pq.tags


def test_pck():
    """Test read PCK file."""
    fname = HERE / 'Tutorials.sptw/IRF_Fluorescein.pck'
    with PqFile(fname) as pq:
        str(pq)
        assert pq.magic == PqFileMagic.PCK
        assert pq.version == '1.0.00'
        assert pq.tags['File_Comment'].startswith('Check point file of ')
        assert_array_equal(
            pq.tags['ChkHistogram'][:6], [96, 150, 151, 163, 153, 145]
        )


def test_pco():
    """Test read PCO file."""
    fname = HERE / 'Tutorials.sptw/Hyperosmotic_Shock_MDCK_Cell.pco'
    with PqFile(fname) as pq:
        str(pq)
        assert pq.magic == PqFileMagic.PCO
        assert pq.version == '1.0.00'
        assert pq.tags['CreatorSW_Modules'] == 0


def test_pfs():
    """Test read PFS file."""
    fname = HERE / 'Settings.pfs'
    with PqFile(fname) as pq:
        str(pq)
        assert pq.magic == PqFileMagic.PFS
        assert pq.version == '1.0.00'
        assert pq.tags['HW_SerialNo'] == '<SerNo. empty>'
        assert pq.tags['Defaults_Begin'] is None


def test_pqres(caplog):
    """Test read PQRES file."""
    fname = HERE / 'Samples.sptw/AnisotropyImage.pqres'
    with caplog.at_level(logging.ERROR):
        with PqFile(fname) as pq:
            str(pq)
            # assert 'not divisible by 8' in caplog.text
            assert pq.magic == PqFileMagic.PQRES
            assert pq.version == '00.0.1'
            assert pq.tags['VarStatFilterGrpIdx'].startswith(b'\xe7/\x00\x00')


@pytest.mark.parametrize('filetype', [str, io.BytesIO])
def test_ptu(filetype):
    """Test read PTU file."""
    fname = HERE / 'napari_flim_phasor_plotter/hazelnut_FLIM_single_image.ptu'
    if filetype != str:
        fname = open(fname, 'rb')
    try:
        with PtuFile(fname) as ptu:
            str(ptu)
            assert ptu.magic == PqFileMagic.PTU
            assert ptu.type == PtuRecordType.PicoHarpT3
            assert ptu.measurement_mode == PtuMeasurementMode.T3
            assert ptu.measurement_submode == PtuMeasurementSubMode.IMAGE
            assert ptu.scanner == PtuScannerType.LSM
            assert ptu.filename == (
                os.fspath(fname) if filetype == str else ''
            )
            assert ptu.version == '00.0.1'
            assert str(ptu.guid) == '4f6e5f68-8289-483d-9d9a-7974b77ef8b8'
            assert ptu.tags['TTResultFormat_BitsPerRecord'] == 32
            assert ptu.tags['\x02HWInpChan_CFDLeveld'] == [100]  # corrupted?
            assert not ptu.tags['HW_ExternalRefClock']
            # decoding of records is tested separately
    finally:
        if filetype != str:
            fname.close()


@pytest.mark.parametrize('filetype', [str, io.BytesIO])
def test_phu(filetype):
    """Test read PHU file."""
    fname = HERE / 'TimeHarp/Decay_Coumarin_6.phu'
    if filetype != str:
        fname = open(fname, 'rb')
    try:
        with PhuFile(fname) as phu:
            str(phu)
            assert phu.magic == PqFileMagic.PHU
            assert phu.measurement_mode == PhuMeasurementMode.HISTOGRAM
            assert phu.measurement_submode == PhuMeasurementSubMode.INTEGRATING
            assert phu.version == '1.1.00'
            assert not phu.tags['HWTriggerOut_On']
            assert phu.tcspc_resolution == 2.5e-11
            assert phu.number_histograms == 4
            assert phu.histogram_resolutions == (3e-11, 3e-11, 3e-11, 3e-11)
            assert_array_equal(
                phu.tags['HistResDscr_DataOffset'],
                [11224, 142296, 273368, 404440],
            )
            histograms = phu.histograms(asxarray=True)
            assert len(histograms) == 4
            for h in histograms:
                assert h.shape == (32768,)
            assert histograms[2][1] == 3
            assert_array_equal(phu.histograms(2)[0], histograms[2])
            phu.plot(show=False, verbose=False)
            phu.plot(show=False, verbose=True)
    finally:
        if filetype != str:
            fname.close()


def test_ptu_t3_image():
    """Test decode T3 image."""
    fname = HERE / 'napari_flim_phasor_plotter/hazelnut_FLIM_single_image.ptu'
    with PtuFile(fname) as ptu:
        str(ptu)
        assert str(ptu.guid) == '4f6e5f68-8289-483d-9d9a-7974b77ef8b8'
        assert ptu.version == '00.0.1'
        # assert ptu._data_offset == 4616
        assert ptu.magic == PqFileMagic.PTU
        assert ptu.type == PtuRecordType.PicoHarpT3
        assert ptu.measurement_mode == PtuMeasurementMode.T3
        assert ptu.measurement_submode == PtuMeasurementSubMode.IMAGE
        assert ptu.scanner == PtuScannerType.LSM

        assert ptu.is_image
        assert ptu.is_t3

        assert ptu.shape == (5, 256, 256, 1, 139)
        assert ptu.dims == ('T', 'Y', 'X', 'C', 'H')
        assert tuple(ptu.coords.keys()) == ('T', 'Y', 'X', 'H')
        # TODO: verify coords

        assert ptu.frame_change_mask == 4
        assert ptu.line_start_mask == 1
        assert ptu.line_stop_mask == 2

        assert ptu.acquisition_time == 2.074774673160728
        assert ptu.frame_time == 0.4149549346321456
        assert ptu.frequency == 78020000.0
        assert ptu.global_frame_time == 32374784
        assert ptu.global_line_time == 126365  # 126464
        assert ptu.global_pixel_time == 494
        assert ptu.global_resolution == 1.281722635221738e-08
        assert ptu.line_time == 0.0016196488079979492
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
        assert ptu.pixel_time == 6.331709817995386e-06
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
        assert_array_equal(im[0, ..., 0, 0], im0[2, :, :, 0].sum(axis=-1))

        im = ptu.decode_image(channel=0, frame=2, dtime=99, dtype='uint32')
        assert im.shape == (1, 256, 256, 1, 99)
        assert_array_equal(
            im[0, :, :, 0].sum(axis=-1), im0[2, :, :, 0, :99].sum(axis=-1)
        )

        im = ptu.decode_image(channel=0, frame=2, dtime=199, dtype='uint32')
        assert im.shape == (1, 256, 256, 1, 199)
        assert_array_equal(
            im[0, :, :, 0].sum(axis=-1), im0[2, :, :, 0].sum(axis=-1)
        )

        im = ptu.decode_image(
            [2, slice(0, 32), slice(100, 132), None, slice(None, None, -1)]
        )
        assert im.shape == (1, 32, 32, 1, 1)
        assert_array_equal(im[..., 0], im0[2:3, :32, 100:132].sum(axis=-1))

        im = ptu.decode_image(
            [slice(1, None, 2)],  # bin by 2 frames starting from second
            dtime=-1,
        )
        assert im.shape == (2, 256, 256, 1, 1)
        assert_array_equal(
            im[1, :, :, :, 0], im0[3:5].sum(axis=0).sum(axis=-1)
        )
        # TODO: verify values

        with pytest.raises(ValueError):
            ptu.decode_image(dtype='int16')


def test_ptu_t3_sinusoidal():
    """Test decode T3 image with sinusoidal correction."""
    fname = HERE / 'tttrlib/5kDa_1st_1_1_1.ptu'
    with PtuFile(fname) as ptu:
        str(ptu)
        assert ptu.version == '1.0.00'
        # assert ptu._data_offset == 4616
        assert ptu.magic == PqFileMagic.PTU
        assert ptu.type == PtuRecordType.PicoHarpT3
        assert ptu.measurement_mode == PtuMeasurementMode.T3
        assert ptu.measurement_submode == PtuMeasurementSubMode.IMAGE
        assert ptu.scanner == PtuScannerType.LSM

        assert ptu.is_image
        assert ptu.is_t3

        assert ptu.tags['ImgHdr_SinCorrection'] == 80

        assert ptu.shape == (122, 512, 512, 1, 3216)
        assert ptu.dims == ('T', 'Y', 'X', 'C', 'H')
        assert tuple(ptu.coords.keys()) == ('T', 'Y', 'X', 'H')
        # TODO: verify coords

        assert ptu.frame_change_mask == 4
        assert ptu.line_start_mask == 1
        assert ptu.line_stop_mask == 2

        assert ptu.acquisition_time == 118.88305798296687
        assert ptu.frame_time == 0.9744512873563691
        assert ptu.frequency == 38898320.0
        assert ptu.global_acquisition_time == 4624351232
        assert ptu.global_frame_time == 37904518
        assert ptu.global_line_time == 18994
        assert ptu.global_pixel_time == 37
        assert ptu.global_resolution == 2.5708051144625268e-08
        assert not ptu.is_bidirectional
        assert ptu.is_image
        assert ptu.is_t3
        assert ptu.line_time == 0.0004882987234410124
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
        assert ptu.pixel_time == 9.511978923511349e-07
        assert ptu.pixels_in_frame == 262144
        assert ptu.pixels_in_line == 512
        assert ptu.syncrate == 38898320
        assert ptu.tcspc_resolution == 7.999999968033578e-12

        records = ptu.read_records()
        assert len(records) == ptu.number_records
        im = ptu.decode_image(
            records=records, frame=-1, dtime=-1, channel=0, keepdims=False
        )
        assert im.shape == (512, 512)
        assert im[399, 18] == 37


@pytest.mark.skip('no test file available')
def test_ptu_t3_line():
    """Test decode T3 line scan."""


def test_ptu_t3_point():
    """Test decode T3 point scan."""
    fname = HERE / '1XEGFP_1.ptu'
    with PtuFile(fname) as ptu:
        str(ptu)
        assert str(ptu.guid) == 'dec6a033-99a9-482d-afbd-5b5743a25133'
        assert ptu.version == '1.0.00'
        assert ptu.magic == PqFileMagic.PTU
        assert ptu.type == PtuRecordType.PicoHarpT3
        assert ptu.measurement_mode == PtuMeasurementMode.T3
        assert ptu.measurement_submode == PtuMeasurementSubMode.POINT
        assert ptu.scanner == PtuScannerType.PI_E710

        assert not ptu.is_image
        assert ptu.is_t3

        assert ptu.shape == (287, 2, 1564)
        assert ptu.dims == ('T', 'C', 'H')
        assert tuple(ptu.coords.keys()) == ('T', 'H')
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
        assert ptu.global_resolution == 2.5000900032401165e-08
        assert ptu.line_time == 0.0010000110003960143
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
        assert ptu.pixel_time == 0.0010000110003960143
        assert ptu.pixels_in_frame == 1
        assert ptu.pixels_in_line == 1
        assert ptu.syncrate == 39998560
        assert ptu.tcspc_resolution == 1.5999999936067155e-11

        assert len(ptu.read_records()) == ptu.number_records
        # assert ptu.decode_records
        im0 = ptu.decode_image()
        assert im0.shape == (287, 2, 1564)
        assert im0.dtype == numpy.uint16
        im = ptu.decode_image(channel=0, frame=2, dtime=-1, dtype='uint32')
        assert im.shape == (1, 1, 1)
        assert im.dtype == numpy.uint32
        # TODO: verify values


@pytest.mark.parametrize('fname', FILES)
def test_ptu_decode_records(fname):
    """Test decode records."""
    with PtuFile(HERE / fname) as ptu:
        decoded = ptu.decode_records()
        assert decoded.size == ptu.number_records
        assert decoded['time'][-1] == ptu.global_acquisition_time
        assert decoded['channel'].max() + 1 == ptu.number_channels
        assert decoded[decoded['channel'] >= 0].size == ptu.number_photons
        assert decoded[decoded['marker'] > 0].size == ptu.number_markers
        nframes = decoded[decoded['marker'] & ptu.frame_change_mask > 0].size
        if ptu.shape:
            assert abs(nframes - ptu.shape[0]) < 2
        if ptu.is_t3:
            assert decoded['dtime'].max() + 1 == ptu.number_bins
            assert decoded[decoded['dtime'] >= 0].size == ptu.number_photons
        # TODO: verify values


@pytest.mark.parametrize('asxarray', [False, True])
@pytest.mark.parametrize('fname', FILES)
def test_ptu_decode_histogram(fname, asxarray):
    """Test decode histograms."""
    with PtuFile(HERE / fname) as ptu:
        ptu.decode_histogram(asxarray=asxarray)
        # TODO: verify values
        with pytest.raises(ValueError):
            ptu.decode_histogram(dtype='int32')


@pytest.mark.parametrize('verbose', [False, True])
@pytest.mark.parametrize('fname', FILES)
def test_ptu_plot(fname, verbose):
    """Test plot methods."""
    with PtuFile(HERE / fname) as ptu:
        ptu.plot(show=False, verbose=verbose)


def test_ptu_read_records():
    """Test PTU read_records method."""
    # the file is tested in test_issue_skip_frame
    fname = HERE / 'Samples.sptw/GUVs.ptu'
    with PtuFile(fname) as ptu:
        records = ptu.read_records(memmap=True)
        assert records.size == ptu.number_records
        im0 = ptu.decode_image(records=records, frame=1, channel=1, dtime=-1)
        im1 = ptu.decode_image(frame=1, channel=1, dtime=-1)
        assert_array_equal(im0, im1)


@pytest.mark.parametrize('output', ['ndarray', 'memmap', 'fname'])
def test_ptu_output(output):
    """Test PTU decoding to different output."""
    # the file is tested in test_issue_skip_frame
    fname = HERE / 'Samples.sptw/GUVs.ptu'
    with PtuFile(fname) as ptu:
        assert ptu.shape == (100, 512, 512, 2, 4096)
        selection = (
            slice(11, 66, 3),
            Ellipsis,
            slice(1, 2),
            slice(None, None, -1),
        )
        shape = (19, 512, 512, 1, 1)
        if output == 'ndarray':
            out = numpy.zeros(shape, 'uint32')
        elif output == 'fname':
            import tempfile

            out = tempfile.TemporaryFile()
        else:
            out = 'memmap'
        im = ptu.decode_image(selection, out=out)
        if output == 'ndarray':
            im = out
        else:
            assert isinstance(im, numpy.memmap)
        assert im[:, 281, 373, 0, 0].sum(axis=0) == 157
        if output == 'fname':
            out.close()


def test_ptu_getitem():
    """Test slice PTU."""
    # the file is tested in test_issue_skip_frame
    fname = HERE / 'Samples.sptw/GUVs.ptu'
    with PtuFile(fname) as ptu:
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
        assert_array_equal(im, im0)
        ptu.dtype = 'uint16'
        # slice uint16
        im = ptu[11:66:3, ..., -1, ::-1]
        assert im.shape == (19, 512, 512)
        assert im.dtype == 'uint16'
        assert_array_equal(im, im0.squeeze())
        # slice with xarray
        ptu.use_xarray = True
        assert ptu.use_xarray
        im = ptu[11:66:3, ..., 1:2, ::-1]
        assert isinstance(im, xarray.DataArray)
        assert tuple(im.coords.keys()) == ('T', 'Y', 'X')
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


def test_issue_skip_frame():
    """Test PTU with incomplete last frame."""
    fname = HERE / 'Samples.sptw/GUVs.ptu'
    with PtuFile(fname) as ptu:
        assert ptu.version == '00.0.0'
        assert ptu.magic == PqFileMagic.PTU
        assert ptu.type == PtuRecordType.PicoHarpT3
        assert ptu.measurement_mode == PtuMeasurementMode.T3
        assert ptu.measurement_submode == PtuMeasurementSubMode.IMAGE
        assert ptu.scanner == PtuScannerType.LSM

        assert ptu.shape == (100, 512, 512, 2, 4096)
        assert ptu.dims == ('T', 'Y', 'X', 'C', 'H')
        assert tuple(ptu.coords.keys()) == ('T', 'Y', 'X', 'H')
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
        assert ptu.global_resolution == 1.0000310009610297e-07
        assert ptu.line_time == 0.002048063489968189
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
        assert ptu.pixel_time == 4.000124003844119e-06
        assert ptu.pixels_in_frame == 262144
        assert ptu.pixels_in_line == 512
        assert ptu.syncrate == 9999690
        assert ptu.tcspc_resolution == 1.5999999936067155e-11

        im = ptu.decode_image(
            92, channel=1, dtime=-1, keepdims=False, dtype='uint32'
        )
        assert im.shape == (512, 512)
        assert im.dtype == numpy.uint32
        assert im[281, 373] == 3


def test_issue_marker_order():
    """Test PTU with strange marker order."""
    # the file has `[ | ][][][ | ]`` instead  of `[] | [][] | []`` markers
    # first and last frame are incomplete
    fname = HERE / 'Samples.sptw/CENP-labelled_cells_for_FRET.ptu'
    with PtuFile(fname, trimdims='CH') as ptu:
        assert ptu.version == '00.0.0'
        # assert ptu._data_offset == 4616
        assert ptu.magic == PqFileMagic.PTU
        assert ptu.type == PtuRecordType.PicoHarpT3
        assert ptu.measurement_mode == PtuMeasurementMode.T3
        assert ptu.measurement_submode == PtuMeasurementSubMode.IMAGE
        assert ptu.scanner == PtuScannerType.LSM

        assert ptu.shape == (191, 512, 512, 3, 3126)
        assert ptu.dims == ('T', 'Y', 'X', 'C', 'H')
        assert tuple(ptu.coords.keys()) == ('T', 'H')
        # TODO: verify coords

        assert ptu._info.skip_first_frame == 0
        assert ptu._info.skip_last_frame == 0

        assert ptu.frame_change_mask == 4
        assert ptu.line_start_mask == 1
        assert ptu.line_stop_mask == 2

        assert ptu.acquisition_time == 213.87296482079424
        assert ptu.frame_time == 1.1197537179119068
        assert ptu.frequency == 20001617.0
        assert ptu.global_frame_time == 22396885
        assert ptu.global_line_time == 20446
        assert ptu.global_pixel_time == 40
        assert ptu.global_resolution == 4.99959578268097e-08
        assert ptu.line_time == 0.0010222173537269511
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
        assert ptu.pixel_time == 1.999838313072388e-06
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

    with PtuFile(fname, trimdims='T') as ptu:
        # trim only time dimension
        assert ptu.shape == (189, 512, 512, 4, 4096)
        assert ptu._info.skip_first_frame == 1
        assert ptu._info.skip_last_frame == 1
        im = ptu.decode_image(
            channel=0, frame=91, dtime=-1, dtype='uint8', keepdims=False
        )
        assert im.shape == (512, 512)
        assert im[390, 277] == 6


def test_issue_empty_line():
    """Test line not empty when first record is start marker."""
    fname = HERE / 'ExampleFLIM/Example_image.sc.ptu'
    with PtuFile(fname) as ptu:
        str(ptu)
        assert ptu.version == '00.0.1'
        assert ptu.magic == PqFileMagic.PTU
        assert ptu.type == PtuRecordType.PicoHarpT3
        assert ptu.measurement_mode == PtuMeasurementMode.T3
        assert ptu.measurement_submode == PtuMeasurementSubMode.IMAGE
        assert ptu.scanner == PtuScannerType.LSM

        assert ptu.shape == (1, 256, 256, 1, 133)
        assert ptu.dims == ('T', 'Y', 'X', 'C', 'H')
        assert tuple(ptu.coords.keys()) == ('T', 'Y', 'X', 'H')
        assert ptu.coords['H'][1] == 9.696969697e-11  # 97 ps

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
        assert ptu.global_pixel_time == 324
        assert ptu.pixel_time == 4.1527813381184314e-06

        assert ptu.decode_records()['marker'][0] == 1  # start marker
        assert ptu[0, 0, 100, 0, ::-1] == 40  # first line not empty


def test_issue_pixeltime_zero():
    """Test PTU with zero ImgHdr_TimePerPixel."""
    fname = HERE / 'nc.picoquant.com/DaisyPollen1.ptu'
    with PtuFile(fname) as ptu:
        assert ptu.version == '1.0.00'
        assert ptu.magic == PqFileMagic.PTU
        assert ptu.type == PtuRecordType.GenericT3
        assert ptu.measurement_mode == PtuMeasurementMode.T3
        assert ptu.measurement_submode == PtuMeasurementSubMode.IMAGE
        assert ptu.scanner is None

        assert ptu.tags['ImgHdr_TimePerPixel'] == 0  # nasty
        assert ptu.global_pixel_time == 160

        assert ptu.shape == (10, 512, 512, 2, 2510)
        assert ptu.dims == ('T', 'Y', 'X', 'C', 'H')
        assert tuple(ptu.coords.keys()) == ('T', 'Y', 'X', 'H')
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
        assert ptu.global_resolution == 2.4999450012099732e-08
        assert ptu.line_time == 0.00204795494499121
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
        assert ptu.pixel_time == 3.999912001935957e-06
        assert ptu.pixels_in_frame == 262144
        assert ptu.pixels_in_line == 512
        assert ptu.syncrate == 40000880
        assert ptu.tcspc_resolution == 9.999999960041972e-12

        im = ptu.decode_image(
            9, channel=0, dtime=-1, keepdims=False, dtype='uint32'
        )
        assert im.shape == (512, 512)
        assert im.dtype == numpy.uint32
        assert im[281, 373] == 25


def test_issue_record_number(caplog):
    """Test PTU with too few records."""
    fname = HERE / 'Samples.sptw/Cy5_immo_FLIM+Pol-Imaging.ptu'
    with PtuFile(fname) as ptu:
        assert ptu.version == '00.0.0'
        with caplog.at_level(logging.ERROR):
            records = ptu.read_records()
            assert 'expected 3409856 records, got 3364091' in caplog.text
            assert len(records) == 3364091
        decoded = ptu.decode_records()
        assert decoded.size == 3364091
        assert decoded['time'][-1] == ptu.global_acquisition_time
        assert decoded['channel'].max() + 1 == ptu.number_channels
        assert decoded[decoded['channel'] >= 0].size == ptu.number_photons
        assert decoded[decoded['marker'] > 0].size == ptu.number_markers
        str(ptu)


def test_issue_tag_index_order(caplog):
    """Test tag index out of order."""
    fname = HERE / 'picoquant-sample-data/hydraharp/v10_t2.ptu'
    with caplog.at_level(logging.ERROR):
        with PqFile(fname) as pq:
            str(pq)
            assert 'tag index out of order' in caplog.text
            assert 'UsrHeadName' in caplog.text
            assert pq.magic == PqFileMagic.PTU
            assert pq.version == '1.0.00'
            assert pq.tags['UsrHeadName'] == [
                '405.0nm (DC405)',
                '485.0nm (DC485)',
            ]


@pytest.mark.parametrize(
    'dtime, size',
    [
        (None, 139),  # last bin with non-zero photons
        (0, 132),  # last bin matching frequency
        (-1, 1),  # integrate bins
        (32, 32),  # specified number of bins
        (145, 145),
    ]
)
def test_issue_dtime(dtime, size):
    """Test dtime parameter."""
    fname = HERE / 'napari_flim_phasor_plotter/hazelnut_FLIM_single_image.ptu'
    im = imread(
        fname,
        frame=0,
        channel=0,
        dtime=dtime,
        dtype=numpy.uint8,
        asxarray=True,
    )
    assert im.dtype == numpy.uint8
    assert im.shape == (1, 256, 256, 1, size)
    assert im.dims == ('T', 'Y', 'X', 'C', 'H')
    assert tuple(im.coords.keys()) == ('T', 'Y', 'X', 'H')


def test_imread():
    """Test imread function."""
    fname = HERE / 'napari_flim_phasor_plotter/hazelnut_FLIM_single_image.ptu'
    im = imread(
        fname,
        [slice(1), None],  # first frame
        channel=0,
        frame=None,
        dtime=0,
        dtype=numpy.uint8,
        asxarray=True,
    )
    assert im.dtype == numpy.uint8
    assert im.shape == (1, 256, 256, 1, 132)
    assert im.dims == ('T', 'Y', 'X', 'C', 'H')
    assert tuple(im.coords.keys()) == ('T', 'Y', 'X', 'H')


@pytest.mark.parametrize('path', [f for f in HERE.iterdir() if f.is_dir()])
def test_glob(path):
    """Test read all PicoQuant files."""
    for fname in glob.glob(str(path) + '/*.p*'):
        if 'htmlcov' in fname:
            continue
        if fname[-3:] in {'.py', 'pyc', 'pt2', 'pt3', 'pdf', 'phd'}:
            continue
        with PqFile(fname) as pq:
            str(pq)
            is_ptu = pq.magic == PqFileMagic.PTU
        if not is_ptu:
            continue
        with PtuFile(fname) as ptu:
            str(ptu)


@pytest.mark.parametrize(
    'trimdims, dtime, size', [('TC', None, 4096), ('TCH', 0, 132)]
)
def test_ptu_zip_sequence(trimdims, dtime, size):
    """Test read Z-stack with imread and tifffile.FileSequence."""
    # requires ~28GB. Do not trim H dimensions such that files match
    from tifffile import FileSequence

    fname = HERE / 'napari_flim_phasor_plotter/hazelnut_FLIM_z_stack.zip'
    with FileSequence(imread, '*.ptu', container=fname) as ptus:
        assert ptus.shape == (11,)
        stack = ptus.asarray(
            channel=0, trimdims=trimdims, dtime=dtime, ioworkers=1
        )
    assert stack.shape == (11, 5, 256, 256, 1, size)  # 11 files, 5 frames each
    assert stack.dtype == 'uint16'


def test_ptu_leica_sequence():
    """Test read Leica TZ-stack with imread and tifffile.imread."""
    # 410 files. Requires ~16GB.
    import tifffile  # >= 2024.2.12

    fname = HERE / 'Flipper TR time series.sptw/*.ptu'
    stack = tifffile.imread(
        str(fname),  # glob pattern needs to be str
        pattern=r'_(t)(\d+)_(z)(\d+)',
        imread=imread,
        chunkshape=(512, 512, 132),  # shape returned by imread
        chunkdtype='uint8',  # dtype returned by imread
        ioworkers=None,  # use multi-threading
        imreadargs=dict(
            frame=0,
            channel=0,
            dtime=132,  # fix number of bins to 132
            dtype='uint8',  # request uint8 output
            keepdims=False,
        ),
    )
    assert stack.shape == (41, 10, 512, 512, 132)
    assert stack.dtype == 'uint8'
    assert stack[24, 4, 228, 279, 16] == 3


def test_ptu_numcodecs():
    """Test Leica TZ-stack with tifffile.ZarrFileSequenceStore and fsspec."""
    # 410 files. Requires ~16GB.
    import fsspec
    import tifffile  # > 2024.2.12
    import zarr

    pathname = HERE / 'Flipper TR time series.sptw'
    url = str(pathname).replace('\\', '/')
    jsonfile = str(pathname / 'FLIPPER.json')
    fname = str(pathname / '*.ptu')  # glob pattern needs to be str

    store = tifffile.imread(
        fname,
        pattern=r'_(t)(\d+)_(z)(\d+)',
        imread=imread,
        chunkshape=(512, 512),  # shape returned by imread
        chunkdtype='uint8',  # dtype returned by imread
        imreadargs=dict(
            frame=0,
            channel=0,
            dtime=-1,
            dtype='uint8',  # request uint8 output
            keepdims=False,
        ),
        aszarr=True,
    )
    assert isinstance(store, tifffile.ZarrFileSequenceStore)
    store.write_fsspec(
        jsonfile,
        url=url,
        version=1,
        codec_id='ptufile',
        quote=False,
    )
    store.close()
    mapper = fsspec.get_mapper(
        'reference://',
        fo=jsonfile,
        target_protocol='file',
        remote_protocol='file',
    )

    ptufile.numcodecs.register_codec()
    stack = zarr.open(mapper, mode='r')
    assert stack.shape == (41, 10, 512, 512)
    assert stack.dtype == 'uint8'
    assert stack[24, 4, 228, 279] == 18


if __name__ == '__main__':
    import warnings

    # warnings.simplefilter('always')
    warnings.filterwarnings('ignore', category=ImportWarning)
    argv = sys.argv
    argv.append('--cov-report=html')
    argv.append('--cov=ptufile')
    argv.append('--verbose')
    sys.exit(pytest.main(argv))
