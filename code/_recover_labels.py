"""从 train 恢复可见 test 的 400 个分子的真实结构。

**依据**：`_leak.py` 已实测确认 1,213 张 test 谱 100% 存在于 train.parquet 中
（谱内容指纹完全一致）。因此这些分子的 `normalized_smiles` / `inchikey14` /
`molecular_formula` 可以直接从 train 取回。

产出一个**分布与真实测试集完全对齐的开发集**：
    data/interim/dev_labels.parquet
        molecule_id, smiles, cand_key(ik14_raw), molecular_formula,
        n_spectra, n_matched, label_consistent

用途（**仅限本地开发**，隐藏测试集会替换）：
    * 真实的 MRR 测量（400 个分子、真实类别分布）
    * 失败分析：哪些分子候选缺失、哪些同式错排
    * 端到端管线联调

重要限制：
    * 这 400 个分子的结构**在 train 里就有**，所以任何"在 train 上训练"的做法
      都是在作弊式地拟合这些答案。**必须按 cand_key 把它们从检索索引中排除**，
      否则分数毫无意义（见 docs/03）。
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
from _leak import fp_of, frame_fps

TRAIN_COLS = [
    "ms2_mzs", "ms2_normalized_intensities",
    "normalized_smiles", "inchikey14", "molecular_formula", "ingest_lib",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(C.INTERIM_DIR, "dev_labels.parquet"))
    a = ap.parse_args()

    te_path = C.find_data_file("test.parquet")
    tr_path = C.find_data_file("train.parquet")

    te = pd.read_parquet(te_path, columns=["molecule_id", "spectrum_id",
                                           "ms2_mzs", "ms2_normalized_intensities"])
    fp_to_mol = {}
    for mol, mz, it in zip(te["molecule_id"], te["ms2_mzs"], te["ms2_normalized_intensities"]):
        fp_to_mol[fp_of(mz, it)] = mol
    print(f"[test] {len(te):,} spectra -> {len(fp_to_mol):,} unique fps, "
          f"{te.molecule_id.nunique()} molecules")

    # 扫描 train，命中的谱把标签挂到对应 molecule_id
    rows = []
    pf = pq.ParquetFile(tr_path)
    n_rg = pf.metadata.num_row_groups
    for rgi in range(n_rg):
        t = pf.read_row_group(rgi, columns=TRAIN_COLS)
        fps = frame_fps(t.column("ms2_mzs").to_pylist(),
                        t.column("ms2_normalized_intensities").to_pylist())
        want = np.fromiter((1 if int(x) in fp_to_mol else 0 for x in fps),
                           dtype=np.int8, count=len(fps))
        idx = np.flatnonzero(want)
        if len(idx) == 0:
            continue
        df = t.take(idx).to_pandas()
        df["_mol"] = [fp_to_mol[int(fps[i])] for i in idx]
        rows.append(df)
        print(f"  rg {rgi+1:2d}/{n_rg}: +{len(idx)} matched")
        if sum(len(r) for r in rows) >= len(te):
            print("  已全部命中，提前结束")
            break

    hit = pd.concat(rows, ignore_index=True)
    print(f"\n[hit] {len(hit):,} 行（期望 {len(te):,}）")

    # 一致性：同一 molecule_id 的所有谱应指向同一结构
    g = hit.groupby("_mol")
    lab = g.agg(
        smiles=("normalized_smiles", lambda s: s.iloc[0]),
        cand_key=("inchikey14", lambda s: s.iloc[0]),
        molecular_formula=("molecular_formula", lambda s: s.iloc[0]),
        ingest_lib=("ingest_lib", lambda s: s.mode().iloc[0] if len(s.mode()) else s.iloc[0]),
        n_matched=("normalized_smiles", "size"),
        n_unique_struct=("inchikey14", "nunique"),
    ).reset_index().rename(columns={"_mol": "molecule_id"})

    lab["label_consistent"] = lab["n_unique_struct"] == 1
    lab["n_spectra"] = lab["molecule_id"].map(te.groupby("molecule_id").size())
    lab = lab[["molecule_id", "smiles", "cand_key", "molecular_formula",
               "n_spectra", "n_matched", "n_unique_struct", "label_consistent",
               "ingest_lib"]]

    print(f"\n[dev] {len(lab)} molecules reconstructed "
          f"(test 有 {te.molecule_id.nunique()})")
    bad = lab[~lab["label_consistent"]]
    if len(bad):
        print(f"!! {len(bad)} 个分子匹配到多个结构（同一分子不同谱给出不同答案）")
        print(bad[["molecule_id", "n_unique_struct"]].head(10).to_string(index=False))
    else:
        print("ok 每个分子恰好对应一个结构，标签完全一致")

    full = lab[lab["n_matched"] == lab["n_spectra"]]
    print(f"ok 全部谱都命中的分子: {len(full)}/{len(lab)}")

    lab.to_parquet(a.out, index=False)
    print(f"\nwritten {a.out}")

    # 快速画像
    print("\n=== 400 个分子的画像 ===")
    print("分子式前 10:\n", lab.molecular_formula.value_counts().head(10).to_string())
    print(f"\n唯一结构数: {lab.cand_key.nunique()} / {len(lab)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
