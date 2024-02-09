Read PicoQuant(r) PTU and related files
=======================================

Ptufile is a Python library to read image and metadata from PicoQuant PTU
and related files: PHU, PCK, PCO, PFS, PUS, and PQRES.
PTU files contain time correlated single photon counting (TCSPC)
measurement data and instrumentation parameters.

`PicoQuant GmbH <https://www.picoquant.com/>`_ is a manufacturer of
photonic components and instruments.

:Author: `Christoph Gohlke <https://www.cgohlke.com>`_
:License: BSD 3-Clause
:Version: 2024.2.8
:DOI: `10.5281/zenodo.10120021 <https://doi.org/10.5281/zenodo.10120021>`_

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

- `CPython <https://www.python.org>`_ 3.9.13, 3.10.11, 3.11.8, 3.12.2 (64-bit)
- `Numpy <https://pypi.org/project/numpy>`_ 1.26.3
- `Xarray <https://pypi.org/project/xarray>`_ 2024.1.1 (recommended)
- `Matplotlib <https://pypi.org/project/matplotlib/>`_ 3.8.2 (optional)
- `Tifffile <https://pypi.org/project/tifffile/>`_ 2024.1.30 (optional)

Revisions
---------

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

The API is not stable yet and might change between revisions.

This library has been tested with a limited number of files only.

The following features are currently not implemented: PT2 and PT3 files,
decoding images from T2 formats, bidirectional scanning, and deprecated
image reconstruction.

The PicoQuant unified file formats are documented at the
`PicoQuant-Time-Tagged-File-Format-Demos
<https://github.com/PicoQuant/PicoQuant-Time-Tagged-File-Format-Demos/tree/master/doc>`_.

Other Python modules for reading PicoQuant files are
`Read_PTU.py
<https://github.com/PicoQuant/PicoQuant-Time-Tagged-File-Format-Demos/blob/master/PTU/Python/Read_PTU.py>`_,
`readPTU_FLIM <https://github.com/SumeetRohilla/readPTU_FLIM>`_,
`PyPTU <https://gitlab.inria.fr/jrye/pyptu>`_,
`tttrlib <https://github.com/Fluorescence-Tools/tttrlib>`_,
`picoquantio <https://github.com/tsbischof/picoquantio>`_,
`ptuparser <https://pypi.org/project/trattoria/>`_,
`trattoria <https://pypi.org/project/ptuparser/>`_
(wrapper of `trattoria-core <https://pypi.org/project/trattoria-core/>`_
and `tttr-toolbox
<https://github.com/GCBallesteros/tttr-toolbox/tree/master/tttr-toolbox>`_),
and `napari-flim-phasor-plotter
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

Decode TTTR records from the PTU file to numpy.recarray. Get global times of
frame changes from markers:

>>> decoded = ptu.decode_records()
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

Decode parts of the image histogram to ``numpy.ndarray`` using slice notation.
Slice step sizes define binning, -1 being used to integrate along axis:

>>> ptu[:, ..., 0, ::-1]
array([[[103, ..., 38],
              ...
        [ 47, ..., 30]]], dtype=uint16)

Alternatively, decode the first channel and integrate all histogram bins
to a ``xarray.DataArray``, keeping reduced axes:

>>> ptu.decode_image(channel=0, dtime=-1, asxarray=True)
<xarray.DataArray (T: 1, Y: 256, X: 256, C: 1, H: 1)>
array([[[[[103]],
           ...
         [[ 30]]]]], dtype=uint16)
Coordinates:
  * T        (T) float64 0.05625
  * Y        (Y) float64 -0.0001304 ... 0.0001294
  * X        (X) float64 -0.0001304 ... 0.0001294
  * H        (H) float64 0.0
Dimensions without coordinates: C
Attributes...
    frequency:      15258789.123471113
...
>>> ptu.close()

Preview the image and metadata in a PTU file from the console::

    python -m ptufile tests/FLIM.ptu
