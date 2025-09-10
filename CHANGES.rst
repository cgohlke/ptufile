Revisions
---------

2025.9.9

- Log error when decoding image with invalid line or frame masks.

2025.7.30

- Add option to specify pixel time for decoding images.
- Add functions to read and write PicoQuant BIN files.
- Drop support for Python 3.10.

2025.5.10

- Mark Cython extension free-threading compatible.
- Support Python 3.14.

2025.2.20

- Rename PqFileMagic to PqFileType (breaking).
- Rename PqFile.magic to PqFile.type (breaking).
- Add PQDAT and SPQR file types.

2025.2.12

- Add options to specify file open modes to PqFile and PtuFile.read_records.
- Add convenience properties to PqFile and PtuFile.
- Cache records read from file.

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
