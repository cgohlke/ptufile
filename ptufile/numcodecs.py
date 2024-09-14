# ptufile/numcodecs.py

# Copyright (c) 2024, Christoph Gohlke
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

"""PTU codec for the Numcodecs package."""

from __future__ import annotations

__all__ = ['register_codec', 'Ptu']

from io import BytesIO
from typing import TYPE_CHECKING

from numcodecs import registry
from numcodecs.abc import Codec

from .ptufile import PtuFile

if TYPE_CHECKING:
    from collections.abc import Sequence
    from types import EllipsisType
    from typing import Any

    from numpy.typing import ArrayLike, DTypeLike, NDArray


class Ptu(Codec):  # type: ignore[misc]
    """Ptu codec for Numcodecs."""

    codec_id = 'ptufile'

    def __init__(
        self,
        *,
        selection: Sequence[int | slice | EllipsisType | None] | None = None,
        dtype: DTypeLike | None = None,
        channel: int | None = None,
        frame: int | None = None,
        dtime: int | None = None,
        trimdims: str | None = None,
        keepdims: bool = True,
    ):
        if selection is not None:
            # TODO: serialize slices, EllipsisType
            raise NotImplementedError(f'{selection=}')
        self.selection = selection
        self.dtype = dtype
        self.channel = channel
        self.frame = frame
        self.dtime = dtime
        self.trimdims = trimdims
        self.keepdims = bool(keepdims)

    def encode(self, buf: ArrayLike) -> None:
        """Return Ptu file as bytes."""
        raise NotImplementedError

    def decode(self, buf: bytes, out: Any | None = None) -> NDArray[Any]:
        """Return decoded image as NumPy array."""
        with BytesIO(buf) as fh:
            with PtuFile(fh) as ptu:
                result = ptu.decode_image(
                    self.selection,
                    dtype=self.dtype,
                    channel=self.channel,
                    frame=self.frame,
                    dtime=self.dtime,
                    keepdims=self.keepdims,
                )
        return result


def register_codec(cls: Codec = Ptu, codec_id: str | None = None) -> None:
    """Register :py:class:`Ptu` codec with Numcodecs."""
    registry.register_codec(cls, codec_id=codec_id)
