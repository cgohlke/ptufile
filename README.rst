Read PicoQuant PTU and related files
====================================

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
