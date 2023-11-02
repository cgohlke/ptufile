# _ptufile.pyx
# distutils: language = c
# cython: language_level = 3
# cython: boundscheck = False
# cython: wraparound = True
# cython: cdivision = True
# cython: nonecheck = False

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

"""Decode PicoQuant Time-Tagged Time-Resolved (TTTR) records.

:Author: Christoph Gohlke
:License: BSD 3-Clause

"""

# TODO: only decode channels with photons

from libc.stdint cimport int8_t, uint8_t, uint16_t, int16_t, uint32_t, uint64_t

import numpy

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


cdef int _format(
    const uint32_t format,
    decode_func_t* decode,
    ssize_t* bins,
    ssize_t* channels,
):
    if format == 0x00010303:
        # PicoHarpT3
        decode[0] = decode_pt3
        bins[0] = 0xfff
        channels[0] = 4
    elif format == 0x00010203:
        # PicoHarpT2
        decode[0] = decode_pt2
        bins[0] = 0
        channels[0] = 5  # ?
    elif format in {
        0x01010204,  # HydraHarp2T2
        0x00010205,  # TimeHarp260NT2
        0x00010206,  # TimeHarp260PT2
        0x00010207,  # MultiHarpT2
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
        0x00010307,  # MultiHarpT3
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


def _decode_info(
    const uint32_t[::1] records,
    const uint32_t format,
    const uint32_t line_start,
    const uint32_t line_stop,
    const uint32_t frame_change,
):
    """Return information about PicoQuant TTTR records."""
    cdef:
        ssize_t nrecords = records.size
        ssize_t i, maxchannels, maxbins
        uint64_t nchannels, nbins, nframes, nphotons, nmarkers, nlines
        uint64_t overflow, time_line_start, time_in_lines
        uint64_t time_frame_start, time_in_frames
        uint32_t itime, idtime, ichannel
        uint8_t ispecial, imarker
        bint skip_first_frame = 1
        decode_func_t decode_func

    if _format(format, &decode_func, &maxbins, &maxchannels) != 0:
        raise ValueError(f'no decoder available for {format=:02x}')

    # Unfortunately Cython's OpenMP does not support min/max reduction
    # https://github.com/cython/cython/issues/3585#issuecomment-625961911

    with nogil:
        time_in_frames = 0
        time_frame_start = 0
        time_in_lines = 0
        overflow = 0
        itime = 0
        nphotons = 0
        nchannels = 0
        nbins = 0
        nmarkers = 0
        nlines = 0
        nframes = 0
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
                nphotons += 1
                if ichannel > nchannels and ichannel < maxchannels:
                    nchannels = ichannel
                if idtime > nbins and idtime < maxbins:
                    nbins = idtime
            elif ispecial == 2:
                nmarkers += 1
                if imarker & frame_change:
                    if time_frame_start > 0:
                        nframes += 1
                        time_in_frames += (overflow + itime) - time_frame_start
                    time_frame_start = overflow + itime
                if imarker & line_stop:
                    nlines += 1
                    time_in_lines += (overflow + itime) - time_line_start
                    time_line_start = 0
                if imarker & line_start:
                    time_line_start = overflow + itime

    if nframes == 0 and time_frame_start > 0:
        # one frame marker
        skip_first_frame = 0
        nframes = 1
        time_in_frames += (overflow + itime) - time_frame_start

    nchannels += 1
    nbins = 0 if maxbins == 0 else nbins + 1

    return (
        format,
        nrecords,
        nphotons,
        nmarkers,
        nframes,
        nlines,
        maxchannels,
        nchannels,
        maxbins,
        nbins,
        skip_first_frame,
        time_in_lines // nlines if nlines > 0 else 0,
        time_in_frames // nframes if nframes > 0 else 0,
        overflow + itime
    )


def _decode_t3_point(
    uint_t[:, :, ::1] histogram,
    uint64_t[::1] times,
    const uint32_t[::1] records,
    const uint32_t format,
    const uint64_t pixel_time,
    const ssize_t startc = 0,
    const ssize_t startt = 0,
    const ssize_t starth = 0,
    const ssize_t binc = 1,
    const ssize_t bint = 1,
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

    if _format(format, &decode_func, &maxbins_, &i) != 0:
        raise ValueError(f'no decoder available for {format=:02x}')
    if maxbins_ == 0:
        raise NotImplementedError(f'not a T3 {format=:02x}')

    if startc < 0 or startt < 0 or starth < 0:
        raise ValueError(f'invalid {startc=}, {startt=}, or {starth=}')
    if binc < 1 or bint < 1 or binh < 1:
        raise ValueError(f'invalid {binc=}, {bint=}, or {binh=}')
    if times.size != histogram.shape[1]:
        raise ValueError(f'{times.size=} does not match {histogram.shape=}')

    sizec, sizet, sizeh = histogram.shape[:3]

    stopc = startc + sizec * binc
    stopt = startt + sizet * bint
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
                    (ichannel - startc) // binc,
                    iframe_binned,
                    (idtime - starth) // binh,
                ] += 1
            # elif ispecial == 1:
            #     # overflow
            # elif ispecial == 2:
            #     # no markers


def _decode_t3_line(
    uint_t[:, :, :, ::1] histogram,
    uint64_t[::1] times,
    const uint32_t[::1] records,
    const uint32_t format,
    const uint64_t pixel_time,
    const uint32_t line_start,
    const uint32_t line_stop,
    const ssize_t startc = 0,
    const ssize_t startt = 0,
    const ssize_t startx = 0,
    const ssize_t starth = 0,
    const ssize_t binc = 1,
    const ssize_t bint = 1,
    const ssize_t binx = 1,
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

    if _format(format, &decode_func, &maxbins_, &i) != 0:
        raise ValueError(f'no decoder available for {format=:02x}')
    if maxbins_ == 0:
        raise NotImplementedError(f'not a T3 {format=:02x}')

    if startc < 0 or startt < 0 or startx < 0 or starth < 0:
        raise ValueError(
            f'invalid {startc=}, {startt=}, {startx=}, or {starth=}'
        )
    if binc < 1 or bint < 1 or binx < 1 or binh < 1:
        raise ValueError(f'invalid {binc=}, {bint=}, {binx=}, or {binh=}')
    if times.size != histogram.shape[1]:
        raise ValueError(f'{times.size=} does not match {histogram.shape=}')

    sizec, sizet, sizex, sizeh = histogram.shape[:4]

    stopc = startc + sizec * binc
    stopt = startt + sizet * bint
    stopx = startx + sizex * binx
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
                    or iframe < startt
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
                    (ichannel - startc) // binc,
                    iframe_binned,
                    (ix - startx) // binx,
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
                    if (
                        iframe >= startt
                        and iframe_binned != (iframe - startt) // bint
                    ):
                        iframe_binned = (iframe - startt) // bint
                        times[iframe_binned] = time_global
                if imarker & line_start:
                    time_line_start = time_global


def _decode_t3_image(
    uint_t[:, :, :, :, ::1] histogram,
    uint64_t[::1] times,
    const uint32_t[::1] records,
    const uint32_t format,
    const uint64_t pixel_time,
    const uint32_t line_start,
    const uint32_t line_stop,
    const uint32_t frame_change,
    const ssize_t startc = 0,
    const ssize_t startt = 0,
    const ssize_t starty = 0,
    const ssize_t startx = 0,
    const ssize_t starth = 0,
    const ssize_t binc = 1,
    const ssize_t bint = 1,
    const ssize_t biny = 1,
    const ssize_t binx = 1,
    const ssize_t binh = 1,
    const bint skip_first_frame = 1
):
    """Return TCSPC histogram from TTTR T3 records of image measurement."""
    cdef:
        ssize_t sizec, sizet, sizey, sizex, sizeh
        ssize_t stopc, stopt, stopy, stopx, stoph
        ssize_t nrecords = records.size
        uint64_t overflow, time_global, time_line_start
        uint32_t itime, idtime, ichannel
        uint8_t ispecial, imarker
        ssize_t i, ix, iy, iframe, iframe_binned, maxbins_
        decode_func_t decode_func

    if _format(format, &decode_func, &maxbins_, &i) != 0:
        raise ValueError(f'no decoder available for {format=:02x}')
    if maxbins_ == 0:
        raise NotImplementedError(f'not a T3 {format=:02x}')

    if startc < 0 or startt < 0 or starty < 0 or startx < 0 or starth < 0:
        raise ValueError(
            f'invalid {startc=}, {startt=}, {starty=}, {startx=}, or {starth=}'
        )
    if binc < 1 or bint < 1 or biny < 1 or binx < 1 or binh < 1:
        raise ValueError(
            f'invalid {binc=}, {bint=}, {biny=}, {binx=}, or {binh=})'
        )
    if times.size != histogram.shape[1]:
        raise ValueError(f'{times.size=} does not match {histogram.shape=}')

    sizec, sizet, sizey, sizex, sizeh = histogram.shape[:5]

    stopc = startc + sizec * binc
    stopt = startt + sizet * bint
    stopy = starty + sizey * biny
    stopx = startx + sizex * binx
    stoph = starth + sizeh * binh

    with nogil:
        time_line_start = 0
        overflow = 0
        # skip until first frame marker if more than one marker
        iframe = -1 if skip_first_frame else 0
        iframe_binned = -1
        iy = 0
        ix = 0

        # TODO: process channels/frames in parallel?
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
                    or iframe < startt
                    or ichannel < startc
                    or ichannel >= stopc
                    or idtime < starth
                    or idtime >= stoph
                    or iy < starty
                    or iy >= stopy
                ):
                    continue

                ix = (time_global - time_line_start) // pixel_time
                if ix < startx or ix >= stopx:
                    continue
                histogram[
                    (ichannel - startc) // binc,
                    iframe_binned,
                    (iy - starty) // biny,
                    (ix - startx) // binx,
                    (idtime - starth) // binh,
                ] += 1

            # elif ispecial == 1:
            #     # overflow
            elif ispecial == 2:
                # marker
                if imarker & frame_change:
                    iframe += 1
                    if iframe == stopt:
                        break
                    if (
                        iframe >= startt
                        and iframe_binned != (iframe - startt) // bint
                    ):
                        iframe_binned = (iframe - startt) // bint
                        times[iframe_binned] = time_global
                    iy = 0
                if imarker & line_stop:
                    time_line_start = 0
                    iy += 1
                if imarker & line_start:
                    time_line_start = time_global


def _decode_t3_histogram(
    uint_t[:, ::1] histogram,
    const uint32_t[::1] records,
    const uint32_t format
):
    """Decode PicoQuant T3 TTTR records to histogram per channel."""
    cdef:
        ssize_t nrecords = records.size
        ssize_t i, nbins, nchannels, maxchannels
        uint64_t overflow
        uint32_t itime, idtime, ichannel
        uint8_t ispecial, imarker
        decode_func_t decode_func

    if _format(format, &decode_func, &nbins, &maxchannels) != 0:
        raise ValueError(f'no decoder available for {format=:02x}')
    if nbins == 0:
        raise ValueError(f'not a T3 {format=:02x}')

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
            if ispecial == 0 and ichannel < nchannels and idtime < nbins:
                histogram[ichannel, idtime] += 1


def _decode_t2_histogram(
    uint_t[:, ::1] histogram,
    const uint32_t[::1] records,
    const uint32_t format,
    const uint64_t bin_time,
):
    """Decode PicoQuant T2 TTTR records to histogram per channel."""
    cdef:
        ssize_t nrecords = records.size
        ssize_t i, ibin, nbins, nchannels, maxchannels
        uint64_t overflow
        uint32_t itime, idtime, ichannel
        uint8_t ispecial, imarker
        decode_func_t decode_func

    if _format(format, &decode_func, &nbins, &maxchannels) != 0:
        raise ValueError(f'no decoder available for {format=:02x}')
    if nbins != 0:
        raise ValueError(f'not a T2 {format=:02x}')
    if bin_time == 0:
        raise ValueError(f'invalid {bin_time=}')

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
            if ispecial == 0 and ichannel < nchannels:
                histogram[ichannel, ibin] += 1


def _decode_t3_records(
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

    if _format(format, &decode_func, &nbins, &maxchannels) != 0:
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


def _decode_t2_records(
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

    if _format(format, &decode_func, &nbins, &maxchannels) != 0:
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
        if tmp > 0 and tmp < 5:
            channel[0] = tmp - 1
        else:
            # should not happen
            channel[0] = 4
        special[0] = 0
    else:
        if dtime[0] == 0:
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
        if tmp > 0 and tmp < 16:
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
        if tmp > 0 and tmp < 16:
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
        if tmp == 0:
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
        if tmp == 0:
            # regular record
            special[0] = 0
            channel[0] = 0
        elif tmp < 16:
            # marker
            special[0] = 2
            marker[0] = tmp