"""证实「可见 test.parquet 取自 train.parquet」。

train 里没有 molecule_id / spectrum_id，无法按 ID 比对，只能按**谱内容**比对。
做法：对每张谱算一个内容指纹（m/z 四舍五入 + 强度粗量化），再求交集。

train 2.54M 行、21 个 row group，按 row group 分块读，不整表进内存。

用法:
    python code/_leak.py                      # 全量比对
    python code/_leak.py --sample 200000      # 抽样加速
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C


def fp_of(mz, it, nd=3) -> int:
    """谱内容指纹 -> 64 位整数。m/z 保留 nd 位小数，强度量化到 1%。"""
    m = np.round(np.asarray(mz, dtype=np.float64), nd)
    i = np.round(np.asarray(it, dtype=np.float64) * 100).astype(np.int32)
    h = hashlib.blake2b(digest_size=8)
    h.update(np.ascontiguousarray(m).tobytes())
    h.update(np.ascontiguousarray(i).tobytes())
    return int.from_bytes(h.digest(), "little")


def frame_fps(mz_col, it_col) -> np.ndarray:
    return np.fromiter(
        (fp_of(m, i) for m, i in zip(mz_col, it_col)), dtype=np.uint64, count=len(mz_col)
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=0, help="只读 train 前 N 行")
    a = ap.parse_args()

    te_path = C.find_data_file("test.parquet")
    tr_path = C.find_data_file("train.parquet")

    print(f"[test ] {te_path}")
    te = pd.read_parquet(te_path, columns=["molecule_id", "ms2_mzs",
                                           "ms2_normalized_intensities"])
    te_fp = frame_fps(te["ms2_mzs"], te["ms2_normalized_intensities"])
    te_set = set(int(x) for x in te_fp)
    print(f"        {len(te):,} spectra, {len(te_set):,} unique fingerprints")
    print(f"        {te.molecule_id.nunique():,} molecules")

    print(f"[train] {tr_path}  (分块读取)")
    pf = pq.ParquetFile(tr_path)
    n_rg = pf.metadata.num_row_groups
    hit = 0
    seen = 0
    matched_mols = set()
    for rgi in range(n_rg):
        t = pf.read_row_group(rgi, columns=["ms2_mzs", "ms2_normalized_intensities"])
        mz = t.column("ms2_mzs").to_pylist()
        it = t.column("ms2_normalized_intensities").to_pylist()
        fps = frame_fps(mz, it)
        seen += len(fps)
        hit += sum(1 for x in fps if int(x) in te_set)
        print(f"  rg {rgi+1:2d}/{n_rg}: rows={len(fps):>8,}  cumulative_hits={hit:,}")
        if a.sample and seen >= a.sample:
            break

    print("\n" + "=" * 62)
    print(f"test 唯一谱指纹        : {len(te_set):,}")
    print(f"train 已扫描行数        : {seen:,}")
    print(f"在 train 中命中的指纹   : {hit:,}")
    print(f"*** 命中率 = {hit / max(len(te_set),1) * 100:.2f}% ***")
    print("=" * 62)
    if hit / max(len(te_set), 1) > 0.9:
        print("\n结论：可见 test 几乎全部来自 train。")
        print("      => 在它上面算分是虚高的；本地验证必须从 train 切分。")
    elif hit:
        print("\n结论：部分重叠，需进一步核查（可能是同一化合物重测，而非同一张谱）。")
    else:
        print("\n结论：没有内容重叠。可见 test 是独立的谱（仍可能是 train 的结构）。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
