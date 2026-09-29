"""
Minimal OriginLab .opju worksheet-data extractor (no Origin needed).

What it does
------------
.opju (Origin 2018+) stores every worksheet column as a record:
    <len><Workbook_Col[@sheet]> ... <tag> 0x18 <u64 0x18><u64 blob_size><u64 hash>
    0a 05 <varints: alloc, ff ff, n, flag, m=2n-1> 0c <FPC-compressed doubles>
The doubles are compressed with the FPC algorithm (Burtscher & Ratanaworabhan,
FCM + DFCM predictors, 2^12-entry tables, one code nibble per value, low nibble
first). This script decodes those columns and writes one CSV per worksheet.

Only numeric worksheet data is recovered (that is all we need). Graph/text
objects are ignored. Origin's missing-value marker (-1.23456789e-300) -> NaN.
"""
import re
import struct
import sys
import json
from collections import defaultdict, OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd

M64 = (1 << 64) - 1
ORIGIN_MISSING = -1.23456789e-300


def fpc_decode(data, n, logsize, start):
    """Decode n doubles (as uint64) from an FPC stream starting at `start`."""
    mask = (1 << logsize) - 1
    fcm = [0] * (mask + 1)
    dfcm = [0] * (mask + 1)
    h = dh = 0
    last = 0
    p1 = p2 = 0
    out = []
    pos = start
    L = len(data)
    while len(out) < n:
        if pos >= L:
            raise ValueError("ran past end of file")
        code = data[pos]
        pos += 1
        for half in (code & 0xF, code >> 4):
            if len(out) >= n:
                break
            bc = half & 7
            nb = bc + (bc >> 2)
            x = int.from_bytes(data[pos:pos + nb], "little")
            pos += nb
            if half & 8:
                v = x ^ ((last + p2) & M64)
            else:
                v = x ^ p1
            fcm[h] = v
            h = ((h << 6) ^ (v >> 48)) & mask
            p1 = fcm[h]
            st = (v - last) & M64
            dfcm[dh] = st
            dh = ((dh << 2) ^ (st >> 40)) & mask
            p2 = dfcm[dh]
            last = v
            out.append(v)
    return out, pos


def u64_to_f64(vals):
    arr = np.array(vals, dtype=np.uint64).view(np.float64)
    arr = arr.copy()
    arr[np.isclose(arr, ORIGIN_MISSING, rtol=0, atol=1e-310)] = np.nan
    return arr


def read_varint(b, i):
    val = 0
    shift = 0
    while True:
        c = b[i]
        i += 1
        val |= (c & 0x7F) << shift
        shift += 7
        if not c & 0x80:
            return val, i


NAME_RE = re.compile(rb"([A-Za-z][A-Za-z0-9]*)_([A-Z]{1,3})(?:@(\d+))?")
HDR_RE = re.compile(rb"[\x80-\x9f]\x18\x18\x00\x00\x00\x00\x00\x00\x00")


def find_datasets(b):
    """Yield (offset, book, col, sheet_idx, values) for every numeric column."""
    results = []
    for m in HDR_RE.finditer(b):
        q = m.start()
        size, _hash = struct.unpack("<QQ", b[q + 10:q + 26])
        base = q + 26
        if b[base:base + 2] != b"\x0a\x05":
            continue
        # dataset name: nearest length-prefixed Workbook_Col[@n] before header
        back = b[max(0, q - 64):q]
        name = None
        for nm in NAME_RE.finditer(back):
            s = nm.start()
            if s > 0 and back[s - 1] == len(nm.group(0)):
                name = nm
        if name is None:
            continue
        book = name.group(1).decode()
        col = name.group(2).decode()
        sheet = int(name.group(3)) if name.group(3) else 1
        decoded = parse_segments(b, base, size)
        if decoded is None:
            results.append((q, book, col, sheet, None, size))
            continue
        results.append((q, book, col, sheet, decoded, size))
    return results


def parse_segments(b, base, size):
    """Parse the value block that follows a column header.

    Layout (after the 0a 05 marker):
      general : <alloc> ff ff  <count> <type> <m> [payload]  (ff ff <segment>)*
      compact : <count> <m> 0c <FPC payload>
    type 00 + 0x0c -> FPC stream of `count` doubles
    type 00 + 0x50 -> one 8-byte double repeated `count` times
    type 01        -> `count` missing values
    Returns a float64 array (NaN for missing) or None if nothing numeric.
    """
    i = base + 2
    end = base + size + 2

    def read_runs(i, count):
        """Runs inside a type-00 segment: <v><enc>[payload] until `count` rows.
        enc 0x0c: FPC, n=(v+1)/2 | enc 0x50: constant, n=v/2 | enc 0x65: missing, n=v/2"""
        parts, got, numeric = [], 0, False
        while got < count:
            v, i = read_varint(b, i)
            enc = b[i]
            i += 1
            if enc == 0x0C:
                n = (v + 1) // 2
                vals, i = fpc_decode(b, n, 12, i)
                parts.append(u64_to_f64(vals))
                numeric = True
            elif enc == 0x50:
                n = v // 2
                raw = struct.unpack("<Q", b[i:i + 8])[0]
                i += 8
                parts.append(u64_to_f64([raw] * n))
                numeric = True
            elif enc == 0x65:
                n = v // 2
                parts.append(np.full(n, np.nan))
            else:
                raise ValueError("unknown run encoding %#x" % enc)
            got += n
        return parts, numeric, i

    try:
        first, j = read_varint(b, i)
        if b[j:j + 2] != b"\xff\xff":
            # compact form: <count> <runs...>
            parts, numeric, _ = read_runs(j, first)
            return np.concatenate(parts) if numeric else None
        i = j + 2
        parts = []
        any_numeric = False
        while True:
            count, i = read_varint(b, i)
            typ = b[i]
            i += 1
            if typ == 0x01:                      # whole segment missing
                _m, i = read_varint(b, i)
                i += 1                           # trailing 00
                parts.append(np.full(count, np.nan))
            elif typ == 0x00:
                p, numeric, i = read_runs(i, count)
                parts.extend(p)
                any_numeric |= numeric
            else:
                return None
            if b[i:i + 2] == b"\xff\xff" and i < end:
                i += 2
                continue
            break
        if not any_numeric:
            return None
        return np.concatenate(parts)
    except (IndexError, ValueError, struct.error):
        return None


def col_index(c):
    v = 0
    for ch in c:
        v = v * 26 + (ord(ch) - 64)
    return v


def extract(path, sheet_names=None):
    """Return {(book, sheet_idx): DataFrame} with columns in A,B,C.. order."""
    b = Path(path).read_bytes()
    ds = find_datasets(b)
    books = defaultdict(dict)
    skipped = []
    for q, book, col, sheet, vals, size in ds:
        if vals is None:
            skipped.append(f"{book}_{col}@{sheet}")
            continue
        books[(book, sheet)][col] = vals
    out = OrderedDict()
    for key in sorted(books, key=lambda k: (k[0], k[1])):
        cols = books[key]
        order = sorted(cols, key=col_index)
        nmax = max(len(cols[c]) for c in order)
        data = {}
        for c in order:
            v = np.full(nmax, np.nan)
            v[: len(cols[c])] = cols[c]
            data[c] = v
        out[key] = pd.DataFrame(data)
    return out, skipped


if __name__ == "__main__":
    for p in sys.argv[1:]:
        sheets, skipped = extract(p)
        print("==", p)
        for (book, idx), df in sheets.items():
            print(f"  {book} sheet{idx}: shape={df.shape} cols={list(df.columns)}")
        print("  skipped (empty/non-numeric):", skipped)
