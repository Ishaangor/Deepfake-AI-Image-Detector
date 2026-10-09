"""parquet_stream must return byte ranges identical to pyarrow's decoded values (synthetic file, no dataset needed)."""
import os

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from training import parquet_stream as ps


@pytest.mark.parametrize("dictionary", [True, False])
def test_ranges_match_pyarrow(tmp_path, dictionary):
    rng = np.random.RandomState(0)
    uniq = [rng.bytes(rng.randint(50, 3000)) for _ in range(40)]
    vals = [uniq[i] for i in rng.randint(0, 40, 500)] if dictionary else [rng.bytes(rng.randint(1, 900)) for _ in range(500)]
    p = str(tmp_path / "t.parquet")
    pq.write_table(pa.table({"image_name": [str(i) for i in range(500)], "image_data": pa.array(vals, pa.binary())}), p,
                   compression="NONE", use_dictionary=dictionary, data_page_size=4096, row_group_size=300)
    base = 0
    for rg in range(pq.ParquetFile(p).metadata.num_row_groups):
        offs, lens = ps.image_ranges(p, row_group=rg)
        n = len(offs)
        got = dict(ps.iter_rows(p, offs, lens, range(n)))
        assert [got[i] for i in range(n)] == vals[base:base + n]
        base += n
    assert base == 500


def test_real_shard_if_present():
    p = os.path.join("data", "commfor", "raw", "data", "HFCF_small_116.parquet")
    if not os.path.exists(p):
        pytest.skip("shard not downloaded")
    offs, lens = ps.image_ranges(p)
    assert len(offs) == pq.ParquetFile(p).metadata.num_rows
    first = next(ps.iter_rows(p, offs, lens, [0]))[1]
    assert first[:2] in (b"\xff\xd8", b"\x89P")
