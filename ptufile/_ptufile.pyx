# _ptufile.pyx
# distutils: language = c
# cython: language_level = 3
# cython: boundscheck = False
# cython: wraparound = False
# cython: cdivision = True
# cython: nonecheck = False
# cython: freethreading_compatible = True

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

"""Decode PicoQuant Time-Tagged Time-Resolved (TTTR) records."""

from libc.math cimport round
from libc.stdint cimport (
    UINT64_MAX,
    int8_t,
    int16_t,
    uint8_t,
    uint16_t,
    uint32_t,
    uint64_t,
)


cdef packed struct t2_t:
    uint64_t time
    int8_t channel
    uint8_t marker
    # uint8_t[6] _align

cdef packed struct t3_t:
    uint64_t time
    int16_t dtime
    int8_t channel
    uint8_t marker
    # uint8_t[4] _align

ctypedef uint32_t (*encode_func_t)(
    const uint32_t time,
    const uint32_t dtime,
    const uint32_t channel,
    const uint32_t overflow,
    const uint32_t marker
) noexcept nogil

ctypedef void (*decode_func_t)(
    const uint32_t record,
    uint32_t* time,
    uint32_t* dtime,
    uint32_t* channel,
    uint64_t* overflow,
    uint8_t* marker,
    uint8_t* special
) noexcept nogil

ctypedef fused uint_t:
    uint8_t
    uint16_t
    uint32_t
    uint64_t


cdef int init_format(
    const uint32_t format,
    decode_func_t* decode,
    ssize_t* bins,  # number_bins_max
    ssize_t* channels,  # number_channels_max
):
    if format == 0x00010303:
        # PicoHarpT3/PicoHarp300
        decode[0] = decode_pt3
        bins[0] = 4096
        channels[0] = 4
    elif format == 0x00010203:
        # PicoHarpT2/PicoHarp300
        decode[0] = decode_pt2
        bins[0] = 0
        channels[0] = 5  # ?
    elif format in {
        0x01010204,  # HydraHarp2T2
        0x00010205,  # TimeHarp260NT2
        0x00010206,  # TimeHarp260PT2
        0x00010207,  # GenericT2 (MultiHarpT2 and Picoharp330T2)
    }:
        decode[0] = decode_ht2
        bins[0] = 0
        channels[0] = 64
    elif format == 0x00010204:
        # HydraHarpT2
        decode[0] = decode_ht2v1
        bins[0] = 0
        channels[0] = 64
    elif format in {
        0x01010304,  # HydraHarp2T3
        0x00010305,  # TimeHarp260NT3
        0x00010306,  # TimeHarp260PT3
        0x00010307,  # GenericT3 (MultiHarpT3 and Picoharp330T3)
    }:
        decode[0] = decode_ht3
        bins[0] = 32768
        channels[0] = 64
    elif format == 0x00010304:  # HydraHarpT3
        decode[0] = decode_ht3v1
        bins[0] = 32768
        channels[0] = 64
    else:
        return 1
    return 0


def decode_info(
    const uint32_t[::1] records,
    const uint32_t format,
    const uint32_t line_start,
    const uint32_t line_stop,
    const uint32_t frame_change,
    const ssize_t lines_in_frame,  # may be zero to disable frame trimming
):
    """Return information about PicoQuant TTTR records."""
    cdef:
        ssize_t nrecords = records.size
        ssize_t i, y, maxchannels, maxbins, skip_first_frame, skip_last_frame
        ssize_t channels_active_first, channels_active_last
        uint64_t nbins, nframes, nphotons, nmarkers, nlines
        uint64_t overflow, time_line_start, time_in_lines, channels_active
        uint32_t itime, idtime, ichannel
        uint8_t ispecial, imarker
        decode_func_t decode_func

    if init_format(format, &decode_func, &maxbins, &maxchannels) != 0:
        raise ValueError(f'no decoder available for {format=:02x}')

    # channels_active is 64-bit
    maxchannels = min(maxchannels, 64)

    # Unfortunately Cython's OpenMP does not support min/max reduction
    # https://github.com/cython/cython/issues/3585#issuecomment-625961911

    with nogil:
        skip_first_frame = 0
        skip_last_frame = 0
        time_line_start = UINT64_MAX
        time_in_lines = 0
        channels_active = 0
        overflow = 0
        itime = 0
        nphotons = 0
        nbins = 0
        nmarkers = 0
        nlines = 0
        nframes = 0
        y = 0
        for i in range(nrecords):
            decode_func(
                records[i],
                &itime,
                &idtime,
                &ichannel,
                &overflow,
                &imarker,
                &ispecial
            )
            if ispecial == 0:
                # photon record
                nphotons += 1
                if ichannel < maxchannels:
                    channels_active |= (<uint64_t> 1) << ichannel
                if idtime > nbins and idtime < maxbins:
                    nbins = idtime
            elif ispecial == 2:
                # marker record
                nmarkers += 1
                if imarker & frame_change:
                    # frame marker
                    if lines_in_frame > y + 1:
                        # not enough lines in frame
                        if nframes == 1:
                            skip_first_frame = 1
                        else:
                            skip_last_frame = 1
                    else:
                        skip_last_frame = 0
                    if not imarker & line_stop:
                        time_line_start = UINT64_MAX
                    y = 0
                if imarker & line_stop:
                    # line stop marker
                    if (
                        time_line_start != UINT64_MAX
                        and (lines_in_frame == 0 or lines_in_frame >= y)
                    ):
                        time_in_lines += (overflow + itime) - time_line_start
                    time_line_start = UINT64_MAX
                if imarker & line_start:
                    # line start marker
                    # TODO: add to time_in_lines if previous line did not stop?
                    time_line_start = overflow + itime
                    if y == 0:
                        # new frame starts at first line
                        # after start or frame change marker
                        nframes += 1
                    y += 1
                    if lines_in_frame == 0 or lines_in_frame >= y:
                        nlines += 1

        channels_active_first = 64
        channels_active_last = 0
        for i in range(64):
            if channels_active & ((<uint64_t> 1) << i):
                if i < channels_active_first:
                    channels_active_first = i
                channels_active_last = i
        if channels_active_first == 64:
            channels_active_first = 0
            channels_active_last = 0

        nbins = 0 if maxbins == 0 else nbins + 1

        if nframes > 1 and y > 0 and lines_in_frame > y + 1:
            skip_last_frame = 1

        if nframes == 0:
            skip_first_frame = 0
            skip_last_frame = 0
        # elif nframes == 1:
        #     # leave incomplete single frame
        #     skip_first_frame = 0
        #     skip_last_frame = 0
        # elif nframes == 2:
        #     if skip_first_frame and skip_last_frame:
        #         # leave both incomplete frames
        #         skip_first_frame = 0
        #         skip_last_frame = 0
        #     elif skip_first_frame or skip_last_frame:
        #         # remove incomplete first xor last frame
        #         nframes -= 1
        else:
            # remove incomplete first and/or last frames
            if skip_first_frame:
                nframes -= 1
            if skip_last_frame and nframes > 0:
                nframes -= 1

    if nlines > 0:
        time_in_lines = <ssize_t> (
            round(<double> time_in_lines / <double> nlines)
        )
    else:
        time_in_lines = 0

    return (
        format,
        nrecords,
        nphotons,
        nmarkers,
        nframes,
        nlines,
        maxchannels,
        channels_active,
        channels_active_first,
        channels_active_last,
        maxbins,
        nbins,
        skip_first_frame,
        skip_last_frame,
        time_in_lines,
        overflow + itime
    )


def decode_t3_point(
    uint_t[:, :, ::1] histogram,
    uint64_t[::1] times,
    const uint32_t[::1] records,
    const uint32_t format,
    const uint64_t pixel_time,
    const ssize_t startt = 0,
    const ssize_t startc = 0,
    const ssize_t starth = 0,
    const ssize_t bint = 1,
    const ssize_t binc = 1,
    const ssize_t binh = 1,
):
    """Return TCSPC histogram from TTTR T3 records of point measurement."""
    cdef:
        ssize_t sizec, sizet, sizeh, stopc, stopt, stoph
        ssize_t nrecords = records.size
        uint64_t overflow, time_global
        uint32_t itime, idtime, ichannel
        uint8_t ispecial, imarker
        ssize_t i, iframe, iframe_binned, maxbins_
        decode_func_t decode_func

    if pixel_time == 0:
        raise ValueError(f'invalid {pixel_time=}')

    if init_format(format, &decode_func, &maxbins_, &i) != 0:
        raise ValueError(f'no decoder available for {format=:02x}')
    if maxbins_ == 0:
        raise NotImplementedError(f'not a T3 {format=:02x}')

    if startc < 0 or startt < 0 or starth < 0:
        raise ValueError(f'invalid {startt=}, {startc=}, or {starth=}')
    if binc < 1 or bint < 1 or binh < 1:
        raise ValueError(f'invalid {bint=}, {binc=}, or {binh=}')
    if times.size != histogram.shape[0]:
        raise ValueError(f'{times.size=} does not match {histogram.shape=}')

    sizet, sizec, sizeh = histogram.shape[:3]

    stopt = startt + sizet * bint
    stopc = startc + sizec * binc
    stoph = starth + sizeh * binh

    with nogil:
        overflow = 0
        iframe = 0
        iframe_binned = -1

        for i in range(nrecords):
            decode_func(
                records[i],
                &itime,
                &idtime,
                &ichannel,
                &overflow,
                &imarker,
                &ispecial
            )
            if ispecial == 0:
                # regular record
                if (
                    ichannel < startc
                    or ichannel >= stopc
                    or idtime < starth
                    or idtime >= stoph
                ):
                    continue
                time_global = overflow + itime
                iframe = time_global // pixel_time
                if iframe < startt:
                    continue
                if iframe >= stopt:
                    break
                if iframe_binned != (iframe - startt) // bint:
                    iframe_binned = (iframe - startt) // bint
                    times[iframe_binned] = time_global

                histogram[
                    iframe_binned,
                    (ichannel - startc) // binc,
                    (idtime - starth) // binh,
                ] += 1
            # elif ispecial == 1:
            #     # overflow
            # elif ispecial == 2:
            #     # no markers


def decode_t3_line(
    uint_t[:, :, :, ::1] histogram,
    uint64_t[::1] times,
    const uint32_t[::1] records,
    const uint32_t format,
    const uint64_t pixel_time,
    const uint32_t line_start,
    const uint32_t line_stop,
    const ssize_t startt = 0,
    const ssize_t startx = 0,
    const ssize_t startc = 0,
    const ssize_t starth = 0,
    const ssize_t bint = 1,
    const ssize_t binx = 1,
    const ssize_t binc = 1,
    const ssize_t binh = 1,
):
    """Return TCSPC histogram from TTTR T3 records of line scan measurement."""
    cdef:
        ssize_t sizec, sizet, sizex, sizeh
        ssize_t stopc, stopt, stopx, stoph
        ssize_t nrecords = records.size
        uint64_t overflow, time_global, time_line_start
        uint32_t itime, idtime, ichannel
        uint8_t ispecial, imarker
        ssize_t i, ix, iframe, iframe_binned, maxbins_
        decode_func_t decode_func

    if pixel_time == 0:
        raise ValueError(f'invalid {pixel_time=}')

    if init_format(format, &decode_func, &maxbins_, &i) != 0:
        raise ValueError(f'no decoder available for {format=:02x}')
    if maxbins_ == 0:
        raise NotImplementedError(f'not a T3 {format=:02x}')

    if startc < 0 or startt < 0 or startx < 0 or starth < 0:
        raise ValueError(
            f'invalid {startt=}, {startx=}, {startc=}, or {starth=}'
        )
    if binc < 1 or bint < 1 or binx < 1 or binh < 1:
        raise ValueError(f'invalid {bint=}, {binx=}, {binc=}, or {binh=}')
    if times.size != histogram.shape[0]:
        raise ValueError(f'{times.size=} does not match {histogram.shape=}')

    sizet, sizex, sizec, sizeh = histogram.shape[:4]

    stopt = startt + sizet * bint
    stopx = startx + sizex * binx
    stopc = startc + sizec * binc
    stoph = starth + sizeh * binh

    with nogil:
        time_line_start = 0
        overflow = 0
        iframe = 0
        iframe_binned = -1
        ix = 0

        for i in range(nrecords):
            decode_func(
                records[i],
                &itime,
                &idtime,
                &ichannel,
                &overflow,
                &imarker,
                &ispecial
            )
            time_global = overflow + itime
            if ispecial == 0:
                # regular record
                if (
                    time_line_start == 0  # no line start marker yet
                    or ichannel < startc
                    or ichannel >= stopc
                    or idtime < starth
                    or idtime >= stoph
                ):
                    continue

                ix = (time_global - time_line_start) // pixel_time
                if ix < startx or ix >= stopx:
                    continue
                histogram[
                    iframe_binned,
                    (ix - startx) // binx,
                    (ichannel - startc) // binc,
                    (idtime - starth) // binh,
                ] += 1

            # elif ispecial == 1:
            #     # overflow
            elif ispecial == 2:
                # marker
                if imarker & line_stop:
                    time_line_start = 0
                    iframe += 1
                    if iframe == stopt:
                        break
                if imarker & line_start and iframe >= startt:
                    time_line_start = time_global
                    if iframe_binned != (iframe - startt) // bint:
                        iframe_binned = (iframe - startt) // bint
                        times[iframe_binned] = time_line_start


def decode_t3_image(
    uint_t[:, :, :, :, ::1] histogram,
    uint64_t[::1] times,
    const uint32_t[::1] records,
    const uint32_t format,
    const ssize_t pixels_in_line,
    ssize_t pixel_time,  # average global time spent in pixel
    ssize_t line_time,  # global time spent in one line
    const uint16_t[::1] pixel_at_time,  # global time in line to pixel index
    const uint32_t line_start,  # mask
    const uint32_t line_stop,  # mask
    const uint32_t frame_change,  # mask
    const ssize_t startt = 0,
    const ssize_t starty = 0,
    const ssize_t startx = 0,
    const ssize_t startc = 0,
    const ssize_t starth = 0,
    const ssize_t bint = 1,
    const ssize_t biny = 1,
    const ssize_t binx = 1,
    const ssize_t binc = 1,
    const ssize_t binh = 1,
    const ssize_t bishift = 0,
    const bint bidirectional = False,
    const bint sinusoidal = False,
    const bint skip_first_frame = 0,
):
    """Return TCSPC histogram from TTTR T3 records of image measurement."""
    cdef:
        ssize_t sizec, sizet, sizey, sizex, sizeh
        ssize_t stopc, stopt, stopy, stopx, stoph
        ssize_t nrecords = records.size
        ssize_t i, j, ix, iy, iy_binned, iframe, iframe_binned, maxbins_, bidiv
        uint64_t overflow, overflowj, time_global, time_line_start
        uint32_t itime, idtime, ichannel
        uint8_t ispecial, imarker
        decode_func_t decode_func
        bint scanline = pixel_time <= 0 or line_time <= 0

    if scanline and pixels_in_line <= 0:
        raise ValueError(f'invalid {pixels_in_line=}')

    if sinusoidal and pixel_at_time.size != line_time:
        raise ValueError(f'invalid {pixel_at_time.size=} != {line_time=}')

    if init_format(format, &decode_func, &maxbins_, &i) != 0:
        raise ValueError(f'no decoder available for {format=:02x}')
    if maxbins_ == 0:
        raise NotImplementedError(f'not a T3 {format=:02x}')

    if startc < 0 or startt < 0 or starty < 0 or startx < 0 or starth < 0:
        raise ValueError(
            f'invalid {startt=}, {starty=}, {startx=}, {startc=}, or {starth=}'
        )
    if binc < 1 or bint < 1 or biny < 1 or binx < 1 or binh < 1:
        raise ValueError(
            f'invalid {bint=}, {biny=}, {binx=}, {binc=}, or {binh=})'
        )
    if times.size != histogram.shape[0]:
        raise ValueError(f'{times.size=} does not match {histogram.shape=}')
    # if wraparound and starth > 0:
    #     raise ValueError(f'can not wrap dtime with {starth=}')

    sizet, sizey, sizex, sizec, sizeh = histogram.shape[:5]

    stopt = startt + sizet * bint
    stopy = starty + sizey * biny
    stopx = startx + sizex * binx
    stopc = startc + sizec * binc
    stoph = starth + sizeh * binh
    bidiv = starty % 2

    with nogil:
        time_line_start = UINT64_MAX
        overflow = 0
        iframe = -2 if skip_first_frame else -1
        iframe_binned = -1
        iy_binned = -1
        iy = -1
        ix = 0

        # TODO: process channels/frames in parallel?
        for i in range(nrecords):
            ispecial = 3

            decode_func(
                records[i],
                &itime,
                &idtime,
                &ichannel,
                &overflow,
                &imarker,
                &ispecial
            )
            time_global = overflow + itime

            if ispecial == 0:
                # regular record
                if (
                    time_line_start == UINT64_MAX  # no line start marker yet
                    or ichannel < startc
                    or ichannel >= stopc
                    or idtime < starth
                    or idtime >= stoph
                    # or (not wraparound and idtime >= stoph)
                ):
                    continue

                ix = <ssize_t> (time_global - time_line_start)

                if bidirectional and iy_binned % 2 != bidiv:
                    # line backward scan
                    # TODO: support bidirectional per frame
                    # TODO: is bishift correct for sinusoidal scanning?
                    ix = line_time - 1 - ix + bishift

                if ix < 0 or ix >= line_time:
                    ix = stopx
                elif sinusoidal:
                    ix = <ssize_t> pixel_at_time[ix]
                else:
                    ix = ix // pixel_time

                if ix >= startx and ix < stopx:
                    # idtime_binned = (idtime - starth) // binh
                    # if wraparound and idtime_binned >= sizeh:
                    #     idtime_binned %= sizeh
                    histogram[
                        iframe_binned,
                        iy_binned,
                        (ix - startx) // binx,
                        (ichannel - startc) // binc,
                        (idtime - starth) // binh,
                    ] += 1

            # elif ispecial == 1:
            #     # overflow
            elif ispecial == 2:
                # marker
                if imarker & frame_change:
                    time_line_start = UINT64_MAX
                    iy = -1
                if imarker & line_stop:
                    time_line_start = UINT64_MAX
                if imarker & line_start:
                    iy += 1
                    if iy == 0:
                        # new frame starts at first line
                        # after start or frame change marker
                        iframe += 1
                        if iframe == stopt:
                            break
                    if iframe >= startt and iy >= starty and iy < stopy:
                        iy_binned = (iy - starty) // biny
                        time_line_start = time_global
                        if iframe_binned != (iframe - startt) // bint:
                            iframe_binned = (iframe - startt) // bint
                            times[iframe_binned] = time_line_start
                        if scanline:
                            # scan line to next marker to determine
                            # line_time and pixel_time
                            overflowj = overflow
                            j = i + 1
                            while True:
                                if j >= nrecords:
                                    break
                                decode_func(
                                    records[j],
                                    &itime,
                                    &idtime,
                                    &ichannel,
                                    &overflowj,
                                    &imarker,
                                    &ispecial
                                )
                                time_global = overflowj + itime
                                if ispecial == 2:
                                    # any marker
                                    break
                                j += 1
                            line_time = <ssize_t> (
                                time_global - time_line_start
                            )
                            # pixel_time = <ssize_t> round(
                            #     <double> line_time / <double> pixels_in_line
                            # )
                            pixel_time = line_time // pixels_in_line
                            pixel_time = pixel_time if pixel_time > 0 else 1
                    else:
                        # line is not part of selection
                        time_line_start = UINT64_MAX

            elif ispecial != 1:
                with gil:
                    raise ValueError(f'invalid records[{i}]={records[i]}')


def decode_t3_histogram(
    uint_t[:, ::1] histogram,
    const uint32_t[::1] records,
    const uint32_t format,
    const ssize_t startc,
):
    """Decode PicoQuant T3 TTTR records to histogram per channel."""
    cdef:
        ssize_t nrecords = records.size
        ssize_t i, nbins, nchannels, maxchannels
        uint64_t overflow
        uint32_t itime, idtime, ichannel
        uint8_t ispecial, imarker
        decode_func_t decode_func

    if init_format(format, &decode_func, &nbins, &maxchannels) != 0:
        raise ValueError(f'no decoder available for {format=:02x}')
    if nbins == 0:
        raise ValueError(f'not a T3 {format=:02x}')
    if startc < 0:
        raise ValueError(f'{startc=} < 0')

    nchannels, nbins = histogram.shape[:2]

    with nogil:
        overflow = 0
        for i in range(nrecords):
            decode_func(
                records[i],
                &itime,
                &idtime,
                &ichannel,
                &overflow,
                &imarker,
                &ispecial
            )
            ichannel -= <uint32_t> startc  # may underflow
            if ispecial == 0 and ichannel < nchannels and idtime < nbins:
                histogram[ichannel, idtime] += 1
                # if wraparound:
                #     histogram[ichannel, idtime % nbins] += 1


def decode_t2_histogram(
    uint_t[:, ::1] histogram,
    const uint32_t[::1] records,
    const uint32_t format,
    const uint64_t bin_time,
    const ssize_t startc,
):
    """Decode PicoQuant T2 TTTR records to histogram per channel."""
    cdef:
        ssize_t nrecords = records.size
        ssize_t i, ibin, nbins, nchannels, maxchannels
        uint64_t overflow
        uint32_t itime, idtime, ichannel
        uint8_t ispecial, imarker
        decode_func_t decode_func

    if bin_time == 0:
        raise ValueError(f'invalid {bin_time=}')

    if init_format(format, &decode_func, &nbins, &maxchannels) != 0:
        raise ValueError(f'no decoder available for {format=:02x}')
    if nbins != 0:
        raise ValueError(f'not a T2 {format=:02x}')
    if bin_time == 0:
        raise ValueError(f'invalid {bin_time=}')
    if startc < 0:
        raise ValueError(f'{startc=} < 0')

    nchannels, nbins = histogram.shape[:2]

    with nogil:
        overflow = 0
        for i in range(nrecords):
            decode_func(
                records[i],
                &itime,
                &idtime,
                &ichannel,
                &overflow,
                &imarker,
                &ispecial
            )
            ibin = <ssize_t> ((overflow + itime) // bin_time)
            if ibin >= nbins:
                break
            ichannel -= <uint32_t> startc  # may underflow
            if ispecial == 0 and ichannel < nchannels:
                histogram[ichannel, ibin] += 1


def decode_t3_records(
    t3_t[::1] decoded,
    const uint32_t[::1] records,
    const uint32_t format
):
    """Decode PicoQuant T3 TTTR records."""
    cdef:
        ssize_t nrecords = min(records.size, decoded.size)
        ssize_t i, nbins, maxchannels
        uint64_t overflow
        uint32_t itime, idtime, ichannel
        uint8_t ispecial, imarker
        decode_func_t decode_func

    if init_format(format, &decode_func, &nbins, &maxchannels) != 0:
        raise ValueError(f'no decoder available for {format=:02x}')
    if nbins == 0:
        raise ValueError(f'not a T3 {format=:02x}')

    with nogil:
        overflow = 0
        for i in range(nrecords):
            decode_func(
                records[i],
                &itime,
                &idtime,
                &ichannel,
                &overflow,
                &imarker,
                &ispecial
            )
            if ispecial == 0:
                # regular record
                decoded[i].time = overflow + itime
                decoded[i].dtime = idtime
                decoded[i].channel = ichannel
                decoded[i].marker = 0
            elif ispecial == 1:
                # overflow
                decoded[i].time = overflow + itime
                decoded[i].dtime = -1
                decoded[i].channel = -1
                decoded[i].marker = 0
            elif ispecial == 2:
                # external marker
                decoded[i].time = overflow + itime
                decoded[i].dtime = -1
                decoded[i].channel = -1
                decoded[i].marker = imarker


def decode_t2_records(
    t2_t[::1] decoded,
    const uint32_t[::1] records,
    const uint32_t format
):
    """Decode PicoQuant T2 TTTR records."""
    cdef:
        ssize_t nrecords = min(records.size, decoded.size)
        ssize_t i, nbins, maxchannels
        uint64_t overflow
        uint32_t itime, idtime, ichannel
        uint8_t ispecial, imarker
        decode_func_t decode_func

    if init_format(format, &decode_func, &nbins, &maxchannels) != 0:
        raise ValueError(f'no decoder available for {format=:02x}')
    if nbins != 0:
        raise ValueError(f'not a T2 {format=:02x}')

    with nogil:
        overflow = 0
        for i in range(nrecords):
            decode_func(
                records[i],
                &itime,
                &idtime,
                &ichannel,
                &overflow,
                &imarker,
                &ispecial
            )
            if ispecial == 0:
                # regular record
                decoded[i].time = overflow + itime
                decoded[i].channel = ichannel
                decoded[i].marker = 0
            elif ispecial == 1:
                # overflow
                decoded[i].time = overflow + itime
                decoded[i].channel = -1
                decoded[i].marker = 0
            elif ispecial == 2:
                # external marker
                decoded[i].time = overflow + itime
                decoded[i].channel = -1
                decoded[i].marker = imarker


cdef void decode_pt3(
    const uint32_t record,
    uint32_t* time,  # nsync
    uint32_t* dtime,
    uint32_t* channel,
    uint64_t* overflow,
    uint8_t* marker,
    uint8_t* special
) noexcept nogil:
    cdef uint32_t tmp

    time[0] = record & 0xffff  # 16 bit nsync
    dtime[0] = (record >> 16) & 0xfff  # 12 bit dtime
    tmp = (record >> 28) & 0xf  # 4 bit channel
    if tmp != 0xf:
        # regular record
        # TODO: understand this undocumented channel decoding
        if tmp > 0 and tmp < 5:
            channel[0] = tmp - 1  # one to zero-based
        else:
            # should not happen
            channel[0] = 4
        special[0] = 0
    elif dtime[0] == 0:
        # overflow
        special[0] = 1
        overflow[0] += 65536
    else:
        # marker
        special[0] = 2
        marker[0] = dtime[0]
        dtime[0] = 0


cdef void decode_pt2(
    const uint32_t record,
    uint32_t* time,  # timetag
    uint32_t* dtime,  # not used
    uint32_t* channel,
    uint64_t* overflow,
    uint8_t* marker,
    uint8_t* special
) noexcept nogil:
    cdef uint32_t tmp

    dtime[0] = 0  # not present in T2 record
    time[0] = record & 0xfffffff  # 28 bit timetag
    tmp = ((record >> 28) & 0xf)  # 4 bit channel
    if tmp != 0xf:
        # regular record
        if tmp < 5:
            channel[0] = tmp
        else:
            # should not happen
            channel[0] = 5
        special[0] = 0
    else:
        tmp = record & 0xf  # lower 4 bits
        if tmp == 0:
            # overflow
            special[0] = 1
            overflow[0] += 210698240
        else:
            # marker
            special[0] = 2
            marker[0] = tmp


cdef void decode_ht3(
    const uint32_t record,
    uint32_t* time,  # nsync
    uint32_t* dtime,
    uint32_t* channel,
    uint64_t* overflow,
    uint8_t* marker,
    uint8_t* special
) noexcept nogil:
    cdef uint32_t tmp

    time[0] = record & 0x3ff  # 10 bit nsync
    dtime[0] = (record >> 10) & 0x7fff  # 15 bit dtime
    tmp = (record >> 31) & 0x1  # 1 bit special
    if tmp == 0:
        # regular record
        special[0] = 0
        channel[0] = (record >> 25) & 0x3f  # 6 bit channel
    else:
        tmp = (record >> 25) & 0x3f  # 6 bit
        if tmp == 0x3f:
            # overflow
            special[0] = 1
            if time[0] <= 1:
                overflow[0] += 1024
            else:
                overflow[0] += time[0] * 1024
        elif 0 < tmp < 16:
            # marker
            special[0] = 2
            marker[0] = tmp


cdef void decode_ht3v1(
    const uint32_t record,
    uint32_t* time,  # nsync
    uint32_t* dtime,
    uint32_t* channel,
    uint64_t* overflow,
    uint8_t* marker,
    uint8_t* special
) noexcept nogil:
    cdef uint32_t tmp

    time[0] = record & 0x3ff  # 10 bit nsync
    dtime[0] = (record >> 10) & 0x7fff  # 15 bit dtime
    tmp = (record >> 31) & 0x1  # 1 bit special
    if tmp == 0:
        # regular record
        special[0] = 0
        channel[0] = (record >> 25) & 0x3f  # 6 bit channel
    else:
        tmp = (record >> 25) & 0x3f  # 6 bit
        if tmp == 0x3f:
            # overflow
            special[0] = 1
            overflow[0] += 1024
        elif 0 < tmp < 16:
            # marker
            special[0] = 2
            marker[0] = tmp


cdef void decode_ht2(
    const uint32_t record,
    uint32_t* time,  # timetag
    uint32_t* dtime,  # not used
    uint32_t* channel,
    uint64_t* overflow,
    uint8_t* marker,
    uint8_t* special
) noexcept nogil:
    cdef uint32_t tmp

    dtime[0] = 0  # not present in T2 record
    time[0] = record & 0x1ffffff  # 25 bit timetag
    tmp = (record >> 31) & 0x1  # 1 bit special
    if tmp == 0:
        # regular record
        special[0] = 0
        channel[0] = (record >> 25) & 0x3f  # 6 bit channel
    else:
        tmp = (record >> 25) & 0x3f  # 6 bit
        if tmp == 0x3f:
            # overflow
            special[0] = 1
            if time[0] <= 1:
                overflow[0] += 33554432
            else:
                overflow[0] += time[0] * 33554432
        elif tmp == 0:
            # regular record
            special[0] = 0
            channel[0] = 0
        elif tmp < 16:
            # marker
            special[0] = 2
            marker[0] = tmp


cdef void decode_ht2v1(
    const uint32_t record,
    uint32_t* time,  # timetag
    uint32_t* dtime,  # not used
    uint32_t* channel,
    uint64_t* overflow,
    uint8_t* marker,
    uint8_t* special
) noexcept nogil:
    cdef uint32_t tmp

    dtime[0] = 0  # not present in T2 record
    time[0] = record & 0x1ffffff  # 25 bit timetag
    tmp = (record >> 31) & 0x1  # 1 bit special
    if tmp == 0:
        # regular record
        special[0] = 0
        channel[0] = (record >> 25) & 0x3f  # 6 bit channel
    else:
        tmp = (record >> 25) & 0x3f  # 6 bit
        if tmp == 0x3f:
            # overflow
            special[0] = 1
            overflow[0] += 33552000
        elif tmp == 0:
            # regular record
            special[0] = 0
            channel[0] = 0
        elif tmp < 16:
            # marker
            special[0] = 2
            marker[0] = tmp


def encode_t3_image(
    uint32_t[::1] records,
    const uint_t[:, :, :, :, ::1] histogram,
    const uint32_t format,
    const uint32_t pixel_time,  # average global time spent in pixel
    const uint32_t line_start,  # mask
    const uint32_t line_stop,  # mask
    const uint32_t frame_change,  # mask
):
    """Return GenericT3 records from TCSPC image histogram.

    No bounds checking is performed writing to records array.

    """
    # TODO: frame and line markers may be combined in one record
    # TODO: record photons across channels at same time (slower)
    # TODO: randomize photon arrival times at fixed count rate

    cdef:
        ssize_t sizet, sizey, sizex, sizec, sizeh
        ssize_t t, y, x, c, h, nrecords
        uint_t count
        uint32_t time, time_in_pixel, overflow, maxtime, maxoverflow, i
        encode_func_t encode

    sizet, sizey, sizex, sizec, sizeh = histogram.shape[:5]

    if format == 0x00010303:
        # PicoHarpT3/PicoHarp300
        encode = encode_pt3
        maxtime = 65536
        maxoverflow = 1
    elif format == 0x00010307:
        # GenericT3 (MultiHarpT3 and Picoharp330T3)
        encode = encode_ht3
        maxtime = 1024
        maxoverflow = 1023
    else:
        raise ValueError(f'{format=} not supported')

    with nogil:
        time = 0
        nrecords = 0

        for t in range(sizet):
            # frame
            for y in range(sizey):
                # line

                # line start marker
                records[nrecords] = encode(time, 0, 0, 0, line_start)
                nrecords += 1

                for x in range(sizex):
                    # pixel
                    time_in_pixel = 0

                    # record all photons at different time
                    for c in range(sizec):
                        # channel
                        for h in range(sizeh):
                            # bin
                            for count in range(histogram[t, y, x, c, h]):
                                # photon
                                records[nrecords] = encode(
                                    time, <uint32_t> h, <uint32_t> c, 0, 0
                                )
                                nrecords += 1

                                # increment time
                                time += 1
                                if time == maxtime:
                                    # overflow
                                    time = 0
                                    records[nrecords] = encode(0, 0, 0, 1, 0)
                                    nrecords += 1

                                time_in_pixel += 1
                                if time_in_pixel == pixel_time:
                                    break
                            if time_in_pixel == pixel_time:
                                break
                        if time_in_pixel == pixel_time:
                            break

                    # move to next pixel
                    # TODO: calculate overflows
                    overflow = 0
                    for i in range(pixel_time - time_in_pixel):
                        time += 1
                        if time == maxtime:
                            # overflow
                            time = 0
                            overflow += 1
                            if overflow == maxoverflow:
                                records[nrecords] = encode(
                                    0, 0, 0, overflow, 0
                                )
                                nrecords += 1
                                overflow = 0
                    if overflow > 0:
                        records[nrecords] = encode(0, 0, 0, overflow, 0)
                        nrecords += 1

                # line end marker
                records[nrecords] = encode(time, 0, 0, 0, line_stop)
                nrecords += 1

            # frame change marker
            records[nrecords] = encode(time, 0, 0, 0, frame_change)
            nrecords += 1

    return nrecords


cdef uint32_t encode_pt3(
    const uint32_t time,
    const uint32_t dtime,
    const uint32_t channel,
    const uint32_t overflow,
    const uint32_t marker,
) noexcept nogil:
    cdef:
        uint32_t record = time  # & <uint32_t> 0xffff  # 16 bit nsync

    if marker != 0:
        # dtime is marker
        record |= marker << <uint32_t> 16
        # record |= (marker & 0xfff) << <uint32_t> 16
        # all channel bits set
        record |= <uint32_t> 0xf0000000
    elif overflow != 0:
        # dtime is 0, all channel bits set
        record |= <uint32_t> 0xf0000000
    else:
        # 12 bit dtime
        record |= dtime << <uint32_t> 16
        # record |= (dtime & 0xfff) << <uint32_t> 16
        # 4 bit channel, one-based
        record |= (channel + 1) << <uint32_t> 28
        # record |= ((channel + 1) & 0xf) << <uint32_t> 28
    return record


cdef uint32_t encode_ht3(
    const uint32_t time,
    const uint32_t dtime,
    const uint32_t channel,
    const uint32_t overflow,
    const uint32_t marker,
) noexcept nogil:
    cdef:
        uint32_t record = time  # & <uint32_t> 0x3ff  # 10 bit nsync

    record |= dtime << <uint32_t> 10  # 15 bit dtime
    # record |= (dtime & <uint32_t> 0x7fff) << <uint32_t> 10  # 15 bit dtime

    if marker != 0:
        # 1 bit special
        record |= <uint32_t> 0x80000000
        # 3 bit marker instead of channel
        record |= marker << <uint32_t> 25
        # record |= (marker & 0xf) << <uint32_t> 25
    elif overflow != 0:
        # 1 bit special and all bits set instead of channel
        record |= <uint32_t> 0xfe000000
        # 10 bit multiple of 1024 increment of nsync instead of nsync
        record |= overflow
        # record |= overflow & <uint32_t> 0x3ff
    else:
        # 6 bit channel
        record |= channel << <uint32_t> 25
        # record |= (channel & <uint32_t> 0x3f) << <uint32_t> 25
    return record
