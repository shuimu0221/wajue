"""B 负责：跑全量检索通道，产出 feat_search.parquet。

用法:
    python code/b_run_search.py                 # 全量
    python code/b_run_search.py --head 100      # 只跑前 N 张谱（调试）
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import b_search as B
import common as C


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--head", type=int, default=0)
    ap.add_argument("--ppm", type=float, default=B.DEFAULT_CFG["ppm"])
    ap.add_argument("--cand", default=os.path.join(C.CAND_DIR, "candidates.parquet"))
    ap.add_argument("--queries", default=None,
                    help="查询谱 parquet（默认读官方 test.parquet）。"
                         "自建验证集时指向 dev_np_spectra_clean.parquet")
    ap.add_argument("--out", default=os.path.join(C.FEAT_DIR, "feat_search.parquet"))
    a = ap.parse_args()

    if a.queries:
        q = pd.read_parquet(a.queries)
        print(f"[queries] 自建验证集: {a.queries}")
    else:
        q = pd.read_parquet(C.find_data_file("test.parquet"))
        q = q.rename(columns={"ms2_mzs": "peaks_mz",
                              "ms2_normalized_intensities": "peaks_int"})
    need = {"molecule_id", "precursor_mz", "adduct", "peaks_mz", "peaks_int"}
    missing = need - set(q.columns)
    if missing:
        raise SystemExit(f"查询谱缺少列: {missing}（实际有 {list(q.columns)}）")
    q = q[["molecule_id", "precursor_mz", "adduct", "peaks_mz", "peaks_int"]]
    if a.head:
        q = q.head(a.head)
    print(f"[queries] {len(q):,} spectra, {q.molecule_id.nunique()} molecules")

    cand = pd.read_parquet(a.cand)
    print(f"[candidates] {len(cand):,} rows, {cand.molecule_id.nunique()} molecules "
          f"({os.path.basename(a.cand)})")

    sc = B.LibraryScanner()
    cfg = {**B.DEFAULT_CFG, "ppm": a.ppm}

    t0 = time.time()
    hits = sc.run(q, cfg=cfg, verbose=True)
    print(f"[search] {time.time()-t0:.1f}s, {len(hits):,} (mol,cand) pairs")

    hits = B.add_lib_hit(hits, cfg)
    cols = ["f_cos", "f_nl", "f_lib_hit", "f_cos_self"]
    out = cand[["molecule_id", "cand_key"]].merge(
        hits, on=["molecule_id", "cand_key"], how="left")
    for c in cols:
        if c not in out:
            out[c] = np.nan

    n_mol = cand.molecule_id.nunique()
    covered = out.loc[out.f_cos.notna(), "molecule_id"].nunique()
    print(f"[coverage] {covered}/{n_mol} 个分子至少有一个候选拿到检索分")
    print(f"[score] f_cos: {out.f_cos.notna().sum():,} 非空, "
          f"max={out.f_cos.max():.3f}")
    print(f"[score] f_cos_self: {out.f_cos_self.notna().sum():,} 非空 "
          f"(=1.0 表示该候选的谱就是查询谱本身)")

    out = out[["molecule_id", "cand_key"] + cols]
    out.to_parquet(a.out, index=False)
    print(f"written {a.out}  ({len(out):,} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
