# test_ptufile.py

# Copyright (c) 2023, Christoph Gohlke
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

:Version: 2023.11.1

"""

import glob
import io
import logging
import os
import pathlib
import sys

import numpy
import ptufile
import pytest
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


def test_tag_index_invalid(caplog):
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

        assert ptu.is_image
        assert ptu.is_t3

        assert ptu.shape == (1, 4, 256, 256, 139)
        assert ptu.dims == ('C', 'T', 'Y', 'X', 'H')
        assert tuple(ptu.coords.keys()) == ('T', 'Y', 'X', 'H')
        # TODO: verify coords

        assert ptu.frame_change_mask == 4
        assert ptu.line_start_mask == 1
        assert ptu.line_stop_mask == 2

        assert ptu.acquisition_time == 2.074774673160728
        assert ptu.frame_time == 0.4149549346321456
        assert ptu.frequency == 2518315.018307149
        assert ptu.global_frame_time == 32374784
        assert ptu.global_line_time == 126464
        assert ptu.global_pixel_time == 494
        assert ptu.global_resolution == 1.281722635221738e-08
        assert ptu.line_time == 0.0016209177134068188
        assert ptu.lines_in_frame == 256
        assert ptu.number_bins == 139
        assert ptu.number_bins_max == 4095
        assert ptu.number_channels == 1
        assert ptu.number_channels_max == 4
        assert ptu.number_frames == 4
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
        assert im0.shape == (1, 4, 256, 256, 139)
        assert im0.dtype == numpy.uint16
        im = ptu.decode_image(channel=0, frame=2, dtime=-1, dtype='uint32')
        assert im.shape == (1, 1, 256, 256, 1)
        assert im.dtype == numpy.uint32
        assert_array_equal(im[0, 0, ..., 0], im0[0, 2].sum(axis=-1))
        im = ptu.decode_image(
            [None, 2, slice(0, 32), slice(100, 132), slice(None, None, -1)]
        )
        assert im.shape == (1, 1, 32, 32, 1)
        assert_array_equal(im[..., 0], im0[:, 2:3, :32, 100:132].sum(axis=-1))
        im = ptu.decode_image(
            [None, slice(1, None, 2)],  # bin 2 frames starting from second
            dtime=-1,
        )
        assert im.shape == (1, 2, 256, 256, 1)
        assert_array_equal(
            im[:, 1, :, :, 0], im0[:, 3:5].sum(axis=1).sum(axis=-1)
        )
        # TODO: verify values


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

        assert not ptu.is_image
        assert ptu.is_t3

        assert ptu.shape == (2, 287, 1564)
        assert ptu.dims == ('C', 'T', 'H')
        assert tuple(ptu.coords.keys()) == ('T', 'H')
        # TODO: verify coords

        assert ptu.frame_change_mask == 0
        assert ptu.line_start_mask == 0
        assert ptu.line_stop_mask == 0

        assert ptu.acquisition_time == 59.998948937161735
        assert ptu.frame_time == 0.0010000110003960143
        assert ptu.frequency == 15262515.323501265
        assert ptu.global_frame_time == 39999
        assert ptu.global_line_time == 39999
        assert ptu.global_pixel_time == 39999
        assert ptu.global_resolution == 2.5000900032401165e-08
        assert ptu.line_time == 0.0010000110003960143
        assert ptu.lines_in_frame == 1
        assert ptu.number_bins == 1564
        assert ptu.number_bins_max == 4095
        assert ptu.number_channels == 2
        assert ptu.number_channels_max == 4
        assert ptu.number_frames == 0
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
        assert im0.shape == (2, 287, 1564)
        assert im0.dtype == numpy.uint16
        im = ptu.decode_image(channel=0, frame=2, dtime=-1, dtype='uint32')
        assert im.shape == (1, 1, 1)
        assert im.dtype == numpy.uint32
        # TODO: verify values


@pytest.mark.parametrize('fname', FILES)
def test_decode_records(fname):
    """Test decode records."""
    with PtuFile(HERE / fname) as ptu:
        decoded = ptu.decode_records()
        assert decoded.size == ptu.number_records
        assert decoded['time'][-1] == ptu.global_acquisition_time
        assert decoded['channel'].max() + 1 == ptu.number_channels
        assert decoded[decoded['channel'] >= 0].size == ptu.number_photons
        assert decoded[decoded['marker'] > 0].size == ptu.number_markers
        nframes = decoded[decoded['marker'] & ptu.frame_change_mask > 0].size
        assert max(0, nframes - 1) == ptu.number_frames
        if ptu.is_t3:
            assert decoded['dtime'].max() + 1 == ptu.number_bins
            assert decoded[decoded['dtime'] >= 0].size == ptu.number_photons
        # TODO: verify values


@pytest.mark.parametrize('asxarray', [False, True])
@pytest.mark.parametrize('fname', FILES)
def test_decode_histogram(fname, asxarray):
    """Test decode histograms."""
    with PtuFile(HERE / fname) as ptu:
        ptu.decode_histogram(asxarray=asxarray)
        # TODO: verify values


@pytest.mark.parametrize('verbose', [False, True])
@pytest.mark.parametrize('fname', FILES)
def test_plot(fname, verbose):
    """Test plot methods."""
    with PtuFile(HERE / fname) as ptu:
        ptu.plot(show=False, verbose=verbose)


def test_wrong_record_number(caplog):
    """Test PTU with too few records."""
    fname = HERE / 'Samples.sptw/Cy5_immo_FLIM+Pol-Imaging.ptu'
    with PtuFile(HERE / fname) as ptu:
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


def test_imread():
    """Test imread function."""
    fname = HERE / 'napari_flim_phasor_plotter/hazelnut_FLIM_single_image.ptu'
    im = imread(
        fname,
        [None, slice(1)],  # first frame
        channel=0,
        frame=None,
        dtime=-1,
        dtype=numpy.uint16,
        asxarray=True,
    )
    assert im.dtype == numpy.uint16
    assert im.shape == (1, 1, 256, 256, 1)
    assert im.dims == ('C', 'T', 'Y', 'X', 'H')
    assert tuple(im.coords.keys()) == ('T', 'Y', 'X', 'H')


@pytest.mark.parametrize('path', [f for f in HERE.iterdir() if f.is_dir()])
def test_glob(path):
    """Test read all PicoQuant files."""
    for fname in glob.glob(str(path) + '/*.p*'):
        if fname[-3:] in {'.py', 'pyc', 'pt2', 'pt3', 'pdf', 'phd'}:
            continue
        with PqFile(fname) as pq:
            str(pq)
            is_ptu = pq.magic == PqFileMagic.PTU
        if not is_ptu:
            continue
        with PtuFile(fname) as ptu:
            str(ptu)


def test_filesequence():
    """Test read Z-stack with imread and tifffile.FileSequence."""
    # requires ~28GB
    from tifffile import FileSequence

    fname = HERE / 'napari_flim_phasor_plotter/hazelnut_FLIM_z_stack.zip'
    with FileSequence(imread, '*.ptu', container=fname) as ptus:
        assert ptus.shape == (11,)
        stack = ptus.asarray(channel=0, trim_dtime=False, ioworkers=1)
    assert stack.shape == (11, 1, 4, 256, 256, 4095)
    assert stack.dtype == 'uint16'


if __name__ == '__main__':
    import warnings

    # warnings.simplefilter('always')
    warnings.filterwarnings('ignore', category=ImportWarning)
    argv = sys.argv
    argv.append('--cov-report=html')
    argv.append('--cov=ptufile')
    argv.append('--verbose')
    sys.exit(pytest.main(argv))
