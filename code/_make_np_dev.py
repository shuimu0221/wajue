"""构建「天然产物」验证集 —— 比 enveda-180 更接近真实测试集。

为什么必须换
------------
W1 的开发集 400 个分子**全部来自 `enveda-180`**（合成类药，含 F 23%）。
官方明确说它与测试集是 "a different region of chemical space"。
真实测试集是**天然产物**。

实测差异（见 experiments.md）：

| 空间 | 查询 | 质量窗口内候选中位 |
|---|---|---|
| enveda-180（合成） | 500 | **86** |
| 天然产物库 | 492 | **9** |

差 10 倍。用合成空间调出来的权重，搬到天然产物上很可能是错的。

做法
----
从天然产物库（riken / gnps / enveda-np-examples / massbank / mona / spectraverse）
抽 N 个有谱的分子当「伪测试集」：
    * 每个分子保留 1–16 张谱（模拟真实每分子谱数）
    * **结构从候选池中排除**（模拟"答案不在库里"，即 class 2/3）
    * 另存一份「含真值」的池子用于测排序上限

输出：
    data/interim/dev_np_labels.parquet   molecule_id, cand_key, smiles, formula
    data/interim/dev_np_spectra.parquet  molecule_id, precursor_mz, adduct, peaks
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import chem
import common as C

NP_LIBS = ["riken", "gnps", "enveda-np-examples", "massbank", "mona", "spectraverse"]
# 只保留测试集可能出现的加合物（官方 10 种）
TEST_ADDUCTS = C.TEST_ADDUCTS


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-mol", type=int, default=400)
    ap.add_argument("--seed", type=int, default=C.SEED)
    ap.add_argument("--max-per-mol", type=int, default=16)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)

    t = pd.read_parquet(C.find_data_file("train.parquet"),
                        columns=["ingest_lib", "inchikey14", "molecular_formula",
                                 "normalized_smiles", "adduct", "precursor_mz",
                                 "ms2_mzs", "ms2_normalized_intensities",
                                 "ionization_mode"])
    np_lib = t[t.ingest_lib.isin(NP_LIBS)].copy()
    print(f"[NP 库] {len(np_lib):,} 谱 / {np_lib.inchikey14.nunique():,} 结构")

    # 只保留测试集 10 种加合物的谱（否则中性质量推不出来）
    np_lib["adduct_n"] = np_lib.adduct.map(C.normalize_adduct)
    ok = np_lib.adduct_n.isin(TEST_ADDUCTS)
    print(f"[加合物] 10 种之一: {ok.sum():,}/{len(np_lib):,} ({ok.mean()*100:.1f}%)")
    np_lib = np_lib[ok]

    # 质量范围对齐真实测试集：157–1159 Da，中位 348（官方原文）
    np_lib["mz_neutral"] = chem.mass_vector(np_lib.molecular_formula)
    np_lib = np_lib.dropna(subset=["mz_neutral"])
    in_range = np_lib.mz_neutral.between(157, 1159)
    print(f"[质量范围] 157–1159 Da: {in_range.sum():,}/{len(np_lib):,}")
    np_lib = np_lib[in_range]

    structs = np_lib.groupby("inchikey14").size()
    pick = rng.choice(structs.index.to_numpy(),
                      size=min(a.n_mol, len(structs)), replace=False)
    sel = np_lib[np_lib.inchikey14.isin(pick)].copy()

    # 每分子最多 max_per_mol 张谱
    sel = (sel.sample(frac=1.0, random_state=a.seed)
              .groupby("inchikey14", group_keys=False)
              .head(a.max_per_mol))

    mol_ids = {k: f"np_{i:04x}" for i, k in enumerate(sorted(sel.inchikey14.unique()))}
    sel["molecule_id"] = sel.inchikey14.map(mol_ids)

    labels = (sel.drop_duplicates("molecule_id")
                 [["molecule_id", "inchikey14", "normalized_smiles", "molecular_formula",
                   "mz_neutral", "ingest_lib"]]
                 .rename(columns={"inchikey14": "cand_key", "normalized_smiles": "smiles"})
                 .reset_index(drop=True))

    spec = sel[["molecule_id", "adduct", "precursor_mz", "ms2_mzs",
                "ms2_normalized_intensities", "ionization_mode"]].rename(columns={
                    "ms2_mzs": "peaks_mz",
                    "ms2_normalized_intensities": "peaks_int"}).reset_index(drop=True)

    print(f"\n[dev_np] {labels.molecule_id.nunique()} 分子 / {len(spec)} 谱")
    print(f"  每分子谱数: 中位 {spec.groupby('molecule_id').size().median():.0f}")
    print(f"  质量: 中位 {labels.mz_neutral.median():.0f} Da "
          f"(真实测试集 348)")
    print(f"  加合物: {spec.adduct.nunique()} 种")
    print(f"  来源: {labels.ingest_lib.value_counts().to_dict()}")
    h = labels.molecular_formula.map(chem.n_heavy)
    print(f"  重原子中位 {h.median():.0f}，含F {labels.molecular_formula.str.contains('F').mean()*100:.1f}% "
          f"(合成集是 23%)")

    labels.to_parquet(os.path.join(C.INTERIM_DIR, "dev_np_labels.parquet"), index=False)
    spec.to_parquet(os.path.join(C.INTERIM_DIR, "dev_np_spectra.parquet"), index=False)
    print(f"\nwritten dev_np_labels.parquet / dev_np_spectra.parquet")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
