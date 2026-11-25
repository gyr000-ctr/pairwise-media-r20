"""imgtile: image tiling with border padding.

The public surface is deliberately small:

``ImgTile().split(image, size, pad=0, mode='reflect')``
    Cut a 2-D rectangular ``image`` (a list of rows) into row-major tiles of
    at most ``size`` by ``size`` pixels.  When ``pad`` is non-zero every tile
    is grown by ``pad`` pixels on all four sides using either ``reflect`` or
    ``replicate`` border handling.

``ImgTile().merge(tiles, shape)``
    Inverse of :meth:`split`.  Tiles produced by :meth:`split` carry a small
    amount of private layout metadata, so padded tiles are cropped back
    exactly and the original pixels are restored (no interpolation is ever
    performed, hence the round-trip error is exactly zero).

All methods are pure: they never mutate their arguments and keep no state on
the instance, which makes a single ``ImgTile`` instance safe to share
between threads.
"""

import math
import numbers

__all__ = ["ImgTile"]

_MODES = ("reflect", "replicate")


# ---------------------------------------------------------------------------
# validation helpers
# ---------------------------------------------------------------------------


def _check_size(value, name):
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise ValueError("%s must be a positive integer" % name)
    value = int(value)
    if value <= 0:
        raise ValueError("%s must be a positive integer" % name)
    return value


def _check_pad(value):
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise ValueError("pad must be a non-negative integer")
    value = int(value)
    if value < 0:
        raise ValueError("pad must be a non-negative integer")
    return value


def _check_mode(mode):
    if not isinstance(mode, str) or mode not in _MODES:
        raise ValueError("mode must be 'reflect' or 'replicate'")
    return mode


def _check_shape(shape):
    if isinstance(shape, (str, bytes)) or not hasattr(shape, "__len__"):
        raise ValueError("shape must be a (height, width) pair")
    try:
        if len(shape) != 2:
            raise ValueError("shape must be a (height, width) pair")
        height = _check_size(shape[0], "shape[0]")
        width = _check_size(shape[1], "shape[1]")
    except (TypeError, IndexError) as exc:  # pragma: no cover - defensive
        raise ValueError("shape must be a (height, width) pair") from exc
    return height, width


def _as_row(row):
    """Return ``row`` as a list, accepting lists/tuples/array-likes."""
    if isinstance(row, list):
        return row
    if isinstance(row, tuple):
        return list(row)
    tolist = getattr(row, "tolist", None)
    if callable(tolist):
        converted = tolist()
        if isinstance(converted, list):
            return converted
    raise ValueError("each image row must be a sequence")


def _check_finite(row):
    isfinite = math.isfinite
    for value in row:
        try:
            finite = isfinite(value)
        except TypeError:
            raise ValueError("image values must be real finite numbers") from None
        if not finite:
            raise ValueError("image values must be finite")


def _as_image(image):
    """Validate a 2-D rectangular finite image and return it as list of rows.

    Input rows are not copied when they are already lists; callers must not
    mutate them (``split`` only slices, so this is safe).
    """
    if not isinstance(image, (list, tuple)):
        tolist = getattr(image, "tolist", None)
        if callable(tolist):
            image = tolist()
        else:
            raise ValueError("image must be a non-empty 2-D sequence")

    if len(image) == 0:
        raise ValueError("image must be a non-empty 2-D sequence")

    rows = []
    width = None
    for raw in image:
        row = _as_row(raw)
        if len(row) == 0:
            raise ValueError("image rows must not be empty")
        if width is None:
            width = len(row)
        elif len(row) != width:
            raise ValueError("image must be rectangular")
        _check_finite(row)
        rows.append(row)
    return rows


def _check_tiles(tiles):
    """Validate tiles and return them as a list of list-of-row lists."""
    if not isinstance(tiles, (list, tuple)):
        tolist = getattr(tiles, "tolist", None)
        if callable(tolist):
            tiles = tolist()
        else:
            raise ValueError("tiles must be a non-empty sequence")
    if len(tiles) == 0:
        raise ValueError("tiles must be a non-empty sequence")

    checked = []
    for tile in tiles:
        if not isinstance(tile, (list, tuple)):
            tolist = getattr(tile, "tolist", None)
            if callable(tolist):
                tile = tolist()
            else:
                raise ValueError("each tile must be a non-empty 2-D sequence")
        checked.append(_as_image(tile))
    return checked


# ---------------------------------------------------------------------------
# border extension
# ---------------------------------------------------------------------------


def _reflect_index(index, length):
    """Map an index through numpy-style ``reflect`` padding.

    The edge sample is not repeated: ``[a, b]`` padded by one becomes
    ``[b, a, b, a]``.  Works for arbitrarily large offsets.
    """
    if length == 1:
        return 0
    period = 2 * (length - 1)
    index %= period
    if index >= length:
        index = period - index
    return index


def _pad_block(block, pad, mode):
    height = len(block)
    width = len(block[0])
    if mode == "replicate":
        row_index = [min(max(i, 0), height - 1) for i in range(-pad, height + pad)]
        col_index = [min(max(j, 0), width - 1) for j in range(-pad, width + pad)]
    else:
        row_index = [_reflect_index(i, height) for i in range(-pad, height + pad)]
        col_index = [_reflect_index(j, width) for j in range(-pad, width + pad)]

    return [[block[i][j] for j in col_index] for i in row_index]


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------


class _Tiles(list):
    """A plain list of tiles plus private layout metadata."""

    _meta = None


class ImgTile:
    """Stateless image tiler.  See the module docstring."""

    def split(self, image, size, pad=0, mode="reflect"):
        """Split ``image`` into row-major tiles and optionally pad borders.

        ``image`` is a non-empty rectangular list of rows; values must be
        finite real numbers.  ``size`` is a positive integer.  ``pad`` adds
        that many pixels on every side of each tile using ``mode``.
        """
        mode = _check_mode(mode)
        size = _check_size(size, "size")
        pad = _check_pad(pad)
        rows = _as_image(image)

        height = len(rows)
        width = len(rows[0])
        cols = (width + size - 1) // size
        line_count = (height + size - 1) // size

        tiles = []
        for y0 in range(0, height, size):
            block_height = min(size, height - y0)
            band = rows[y0:y0 + block_height]
            for x0 in range(0, width, size):
                block_width = min(size, width - x0)
                block = [row[x0:x0 + block_width] for row in band]
                if pad:
                    block = _pad_block(block, pad, mode)
                tiles.append(block)

        result = _Tiles(tiles)
        result._meta = {
            "shape": (height, width),
            "size": size,
            "pad": pad,
            "mode": mode,
            "rows": line_count,
            "cols": cols,
        }
        return result

    def merge(self, tiles, shape):
        """Reassemble ``tiles`` into an image of ``shape`` (height, width).

        Padding added by :meth:`split` is removed.  Tiles without layout
        metadata are accepted as well; their block size and padding are
        inferred from ``shape`` and the tile dimensions.
        """
        height, width = _check_shape(shape)
        checked = _check_tiles(tiles)

        meta = getattr(tiles, "_meta", None)
        if isinstance(meta, dict) and tuple(meta.get("shape", ())) == (height, width):
            return self._merge_known(checked, meta, height, width)
        return self._merge_inferred(checked, height, width)

    # -- merge implementations -------------------------------------------------

    @staticmethod
    def _merge_known(tiles, meta, height, width):
        size = meta["size"]
        pad = meta["pad"]
        rows_count = meta["rows"]
        cols_count = meta["cols"]
        if len(tiles) != rows_count * cols_count:
            raise ValueError("tile count does not match the image shape")

        out = [[None] * width for _ in range(height)]
        index = 0
        for tile_row in range(rows_count):
            y0 = tile_row * size
            block_height = min(size, height - y0)
            for tile_col in range(cols_count):
                x0 = tile_col * size
                block_width = min(size, width - x0)
                block = tiles[index]
                index += 1

                expected_height = block_height + 2 * pad
                expected_width = block_width + 2 * pad
                if len(block) != expected_height or len(block[0]) != expected_width:
                    raise ValueError("tile dimensions do not match the split layout")

                for i in range(block_height):
                    out[y0 + i][x0:x0 + block_width] = block[pad + i][pad:pad + block_width]
        return out

    @staticmethod
    def _merge_inferred(tiles, height, width):
        dims = [(len(tile), len(tile[0])) for tile in tiles]
        count = len(tiles)

        # A single tile is either the image itself or a symmetrically padded
        # version of it.
        if count == 1:
            tile_height, tile_width = dims[0]
            if (tile_height, tile_width) == (height, width):
                return [row[:] for row in tiles[0]]
            dh = tile_height - height
            dw = tile_width - width
            if dh == dw and dh >= 0 and dh % 2 == 0:
                pad = dh // 2
                return [row[pad:pad + width] for row in tiles[0][pad:pad + height]]
            raise ValueError("tile dimensions do not match shape")

        first_height, first_width = dims[0]
        max_pad = min(first_height, first_width) // 2
        layout = None
        for pad in range(max_pad + 1):
            base_height = first_height - 2 * pad
            base_width = first_width - 2 * pad
            if base_height <= 0 or base_width <= 0:
                break

            # The first tile covers the top-left block, so its unpadded size
            # is (min(size, height), min(size, width)).  Recover ``size``
            # from whichever dimension is clipped.
            short_height = base_height < height
            short_width = base_width < width
            if short_height and short_width:
                if base_height != base_width:
                    continue
                size = base_height
            elif short_height:
                if base_height < width:
                    continue
                size = base_height
            elif short_width:
                if base_width < height:
                    continue
                size = base_width
            else:
                # A single unpadded tile covering the whole image; any size
                # at least max(height, width) produces this layout.
                size = max(height, width)

            rows_count = (height + size - 1) // size
            cols_count = (width + size - 1) // size
            if rows_count * cols_count != count:
                continue

            valid = True
            for index, (tile_height, tile_width) in enumerate(dims):
                tile_row, tile_col = divmod(index, cols_count)
                expected_height = min(size, height - tile_row * size) + 2 * pad
                expected_width = min(size, width - tile_col * size) + 2 * pad
                if (tile_height, tile_width) != (expected_height, expected_width):
                    valid = False
                    break
            if valid:
                layout = (size, pad, rows_count, cols_count)
                break

        if layout is None:
            raise ValueError("unable to infer tile layout from shape")

        size, pad, rows_count, cols_count = layout
        out = [[None] * width for _ in range(height)]
        index = 0
        for tile_row in range(rows_count):
            y0 = tile_row * size
            block_height = min(size, height - y0)
            for tile_col in range(cols_count):
                x0 = tile_col * size
                block_width = min(size, width - x0)
                block = tiles[index]
                index += 1
                for i in range(block_height):
                    out[y0 + i][x0:x0 + block_width] = block[pad + i][pad:pad + block_width]
        return out
