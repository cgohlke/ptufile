Revisions
---------

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
