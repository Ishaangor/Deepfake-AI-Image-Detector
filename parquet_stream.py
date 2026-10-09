"""Memory-light access to the big binary column of Community Forensics parquet shards.

The shards store `image_data` UNCOMPRESSED / PLAIN with dictionary + fallback pages, each column chunk being one
~1-4 GB blob. pyarrow, DuckDB and pandas all materialise that blob in RAM (the machine has ~2 GB free), so this module
walks the pages itself and returns the *file byte range of every row*, after which single images are read on demand.

    offs, lens = image_ranges(path)                 # int64 arrays, one entry per row, in row order
    for pos, data in iter_rows(path, offs, lens, positions): ...

Only the Parquet page headers (Thrift compact protocol) and RLE/bit-packed dictionary indices are parsed. Validated
against pyarrow in tests/test_parquet_stream.py.
"""
import struct

import numpy as np
import pyarrow.parquet as pq

DICT_PAGE, DATA_PAGE, DATA_PAGE_V2 = 2, 0, 3


def _varint(b, i):
    r = s = 0
    while True:
        x = b[i]
        i += 1
        r |= (x & 0x7F) << s
        if not x & 0x80:
            return r, i
        s += 7


def _zigzag(n):
    return (n >> 1) ^ -(n & 1)


def _thrift_value(b, i, t):
    if t in (1, 2):                                   # bool encoded in the field type
        return t == 1, i
    if t == 3:
        return b[i], i + 1
    if t in (4, 5, 6):
        v, i = _varint(b, i)
        return _zigzag(v), i
    if t == 7:
        return struct.unpack_from("<d", b, i)[0], i + 8
    if t == 8:
        n, i = _varint(b, i)
        return bytes(b[i:i + n]), i + n
    if t in (9, 10):
        h = b[i]
        i += 1
        n, et = h >> 4, h & 0x0F
        if n == 15:
            n, i = _varint(b, i)
        out = []
        for _ in range(n):
            v, i = _thrift_value(b, i, et)
            out.append(v)
        return out, i
    if t == 12:
        return _thrift_struct(b, i)
    raise ValueError(f"unsupported thrift type {t}")


def _thrift_struct(b, i):
    d, last = {}, 0
    while True:
        h = b[i]
        i += 1
        if h == 0:
            return d, i
        delta, t = h >> 4, h & 0x0F
        if delta:
            fid = last + delta
        else:
            v, i = _varint(b, i)
            fid = _zigzag(v)
        d[fid], i = _thrift_value(b, i, t)
        last = fid


def _rle_hybrid(b, i, bit_width, count):
    out = []
    while len(out) < count:
        h, i = _varint(b, i)
        if h & 1:                                      # bit-packed groups of 8
            n = (h >> 1) * 8
            nbytes = (h >> 1) * bit_width
            bits = int.from_bytes(b[i:i + nbytes], "little")
            mask = (1 << bit_width) - 1
            out.extend((bits >> (k * bit_width)) & mask for k in range(n))
            i += nbytes
        else:                                          # run
            n = h >> 1
            w = (bit_width + 7) // 8
            out.extend([int.from_bytes(b[i:i + w], "little")] * n)
            i += w
    return out[:count]


def image_ranges(path, column="image_data", row_group=0):
    pf = pq.ParquetFile(path)
    rg = pf.metadata.row_group(row_group)
    names = [rg.column(i).path_in_schema for i in range(rg.num_columns)]
    ci = names.index(column)
    col = rg.column(ci)
    if str(col.compression) != "UNCOMPRESSED":
        raise ValueError(f"{column} is {col.compression}, only UNCOMPRESSED is supported")
    max_def = pf.metadata.schema.column(ci).max_definition_level
    start = col.dictionary_page_offset if col.has_dictionary_page else col.data_page_offset
    end, total_values = start + col.total_compressed_size, col.num_values
    dict_offs = dict_lens = None
    offs, lens = [], []
    with open(path, "rb") as f:
        pos = start
        while pos < end and len(offs) < total_values:
            f.seek(pos)
            buf = f.read(min(1 << 16, end - pos))
            hdr, hlen = _thrift_struct(buf, 0)
            ptype, size = hdr[1], hdr[3]
            body = pos + hlen
            if ptype == DICT_PAGE:
                n = hdr[7][1]
                dict_offs, dict_lens, p = np.empty(n, np.int64), np.empty(n, np.int64), body
                for k in range(n):
                    f.seek(p)
                    ln = struct.unpack("<I", f.read(4))[0]
                    dict_offs[k], dict_lens[k] = p + 4, ln
                    p += 4 + ln
            elif ptype == DATA_PAGE:
                dh = hdr[5]
                n, enc = dh[1], dh[2]
                f.seek(body)
                small = f.read(min(size, 1 << 20)) if enc in (2, 8) else None
                q = 0
                if max_def > 0:                        # definition levels: 4-byte length + RLE data (no nulls here)
                    q = 4 + struct.unpack_from("<I", small if small is not None else f.read(4), 0)[0]
                if enc in (2, 8):                      # PLAIN_DICTIONARY / RLE_DICTIONARY
                    bw = small[q]
                    idx = _rle_hybrid(small, q + 1, bw, n)
                    offs.extend(dict_offs[idx].tolist())
                    lens.extend(dict_lens[idx].tolist())
                else:                                  # PLAIN: [len][bytes] * n
                    p = body + q
                    for _ in range(n):
                        f.seek(p)
                        ln = struct.unpack("<I", f.read(4))[0]
                        offs.append(p + 4)
                        lens.append(ln)
                        p += 4 + ln
            else:
                raise ValueError(f"unsupported page type {ptype}")
            pos = body + size
    return np.asarray(offs, np.int64), np.asarray(lens, np.int64)


def iter_rows(path, offs, lens, positions):
    """Yield (position, bytes) for the requested row positions, in file order."""
    with open(path, "rb") as f:
        for p in sorted(positions):
            f.seek(int(offs[p]))
            yield p, f.read(int(lens[p]))
