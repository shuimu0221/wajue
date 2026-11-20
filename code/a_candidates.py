"""A 负责：候选池构建。

设计要点
--------
1. **绝不整池进内存**：275k 结构按 m/z 排序后存 parquet，检索用 searchsorted。
   若将来接入 PubChem 全量（上亿级），改用按 m/z 分桶的 parquet + row-group 统计。
2. **排除必须按 cand_key**：验证时把 holdout 结构从池中剔除。
   只按 `ingest_lib` 过滤是无效的——同一化合物跨库重复（官方原文确认）。
3. **分子式预筛**：`molecular_formula` 能从精确质量直接反查，
   同分子式的候选是排序阶段的主要竞争者，必须作为特征保留（`f_formula_match`）。

实测数据（见 experiments.md）：
    * 275,301 个 train 唯一结构，质量 31–3,400 Da
    * 10 ppm 窗口下，候选数中位 78、p90 257、max 481
    * 5 ppm 窗口下，候选数中位 55、p90 198、max 479
"""

from __future__ import annotations

import logging
import os
from typing import Iterable, Optional

import numpy as np
import pandas as pd

import chem
import common as C

log = logging.getLogger("casmi.cand")

CAND_SCHEMA = [
    "molecule_id", "cand_key", "smiles", "molecular_formula",
    "mz_neutral", "ppm", "source", "rank_mass",
]


# ---------------------------------------------------------------- 结构池

def build_pool_from_train(train_path: Optional[str] = None,
                          out_path: Optional[str] = None) -> pd.DataFrame:
    """从 train.parquet 抽唯一结构，建可检索的结构池。

    输出列：cand_key(inchikey14), smiles, molecular_formula, mz_neutral
    按 mz_neutral 升序排列（检索依赖有序性）。
    """
    train_path = train_path or C.find_data_file("train.parquet")
    out_path = out_path or os.path.join(C.CAND_DIR, "pool_train.parquet")

    cols = ["inchikey14", "normalized_smiles", "molecular_formula"]
    t = pd.read_parquet(train_path, columns=cols)

    pool = (t.drop_duplicates("inchikey14")
             .rename(columns={"inchikey14": "cand_key",
                              "normalized_smiles": "smiles"})
             .reset_index(drop=True))
    pool["mz_neutral"] = chem.mass_vector(pool["molecular_formula"])
    n_bad = int(pool["mz_neutral"].isna().sum())
    if n_bad:
        log.warning("丢弃 %d 个无法解析分子式的结构", n_bad)
        pool = pool.dropna(subset=["mz_neutral"])

    pool["source"] = "train"
    pool = pool.sort_values("mz_neutral").reset_index(drop=True)
    pool.to_parquet(out_path, index=False)
    log.info("结构池 %d 个 -> %s（%.1f–%.1f Da）",
             len(pool), out_path, pool.mz_neutral.min(), pool.mz_neutral.max())
    return pool


def load_pool(path: Optional[str] = None) -> pd.DataFrame:
    path = path or os.path.join(C.CAND_DIR, "pool_train.parquet")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} 不存在，先跑 build_pool_from_train")
    return pd.read_parquet(path)


# ---------------------------------------------------------------- 检索

def derive_neutral_masses(test: pd.DataFrame, tol_ppm: float = 12.0,
                          tol_da: float = 0.004) -> pd.DataFrame:
    """每个 molecule_id 的中性质量。

    **不能直接取中位数。** 实测（dev_np 的 28 个 miss）：
    某个分子的 7 张谱给出两个质量簇（330.18 和 370.21，差 40 Da，是加合物
    标注错误），中位数落在错误簇上，导致真值永远进不了质量窗口。

    正确做法：**先聚类，取支持数最多的簇，再取该簇中位数**。
    容差用 max(ppm 相对容差, 绝对下限)——实测存在固定 +0.0007 Da 偏移，
    纯 ppm 容差在低质量端过窄。
    """
    rows = []
    mz_col = test["precursor_mz"].to_numpy(dtype=float)
    ad_col = test["adduct"].to_numpy()
    mol_col = test["molecule_id"].to_numpy()
    masses = np.array([np.nan if m is None else m for m in
                       (C.neutral_mass_from_precursor(mz, ad)
                        for mz, ad in zip(mz_col, ad_col))], dtype=float)
    order = np.argsort(mol_col, kind="mergesort")
    mol_sorted = mol_col[order]
    mass_sorted = masses[order]
    bounds = np.flatnonzero(np.r_[True, mol_sorted[1:] != mol_sorted[:-1], True])

    for i in range(len(bounds) - 1):
        mol = mol_sorted[bounds[i]]
        v = mass_sorted[bounds[i]:bounds[i + 1]]
        v = np.sort(v[np.isfinite(v)])
        if len(v) == 0:
            rows.append({"molecule_id": mol, "mz_neutral": np.nan,
                         "n_mass": 0, "ppm_spread": np.nan, "n_clusters": 0})
            continue

        # 单链聚类：相邻质量差超过容差就断开
        tol = np.maximum(np.abs(v) * tol_ppm * 1e-6, tol_da)
        brk = np.r_[True, np.diff(v) > tol[1:]]
        seg_id = np.cumsum(brk) - 1
        sizes = np.bincount(seg_id)
        best = int(np.argmax(sizes))
        cluster = v[seg_id == best]
        spread = ((cluster.max() - cluster.min()) / np.median(cluster) * 1e6
                  if len(cluster) > 1 else 0.0)
        rows.append({"molecule_id": mol,
                     "mz_neutral": float(np.median(cluster)),
                     "n_mass": len(cluster),
                     "n_mass_all": len(v),
                     "ppm_spread": float(spread),
                     "n_clusters": int(len(sizes))})
    return pd.DataFrame(rows)


def candidates_for(
    neutrals: pd.DataFrame,
    pool: pd.DataFrame,
    ppm: float = C.PPM_DEFAULT,
    min_da: float = 0.003,
    max_per_mol: int = C.MAX_CAND_PER_MOL,
    exclude_keys: Optional[Iterable[str]] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """按质量窗口取候选。

    返回 (candidates, stats)：
      candidates : CAND_SCHEMA 的长表
      stats      : 每分子的候选数、是否命中真值（若 pool 含真值）

    `ppm` 是相对容差，`min_da` 是绝对下限——实测存在固定 +0.0007 Da 偏移，
    纯 ppm 容差在低质量端会过窄。

    **exclude_keys 在窗口内逐个剔除**（不是整池过滤，避免重建池）。
    """
    exclude = set(exclude_keys or ())
    mz = pool["mz_neutral"].to_numpy()
    outs = []
    stats = []

    for mol, mzn in zip(neutrals["molecule_id"], neutrals["mz_neutral"]):
        if not np.isfinite(mzn):
            stats.append({"molecule_id": mol, "n_cand": 0, "truncated": 0,
                          "mz_neutral": np.nan})
            continue
        tol = max(abs(mzn) * ppm * 1e-6, min_da)
        lo, hi = np.searchsorted(mz, [mzn - tol, mzn + tol])
        sub = pool.iloc[lo:hi]
        if exclude:
            sub = sub[~sub["cand_key"].isin(exclude)]

        n_all = len(sub)
        if n_all == 0:
            stats.append({"molecule_id": mol, "n_cand": 0, "truncated": 0,
                          "mz_neutral": float(mzn)})
            continue

        sub = sub.assign(ppm=(sub["mz_neutral"] - mzn) / mzn * 1e6)
        sub = sub.reindex(sub["ppm"].abs().sort_values().index)
        trunc = max(0, n_all - max_per_mol)
        sub = sub.head(max_per_mol).reset_index(drop=True)
        sub["molecule_id"] = mol
        sub["rank_mass"] = np.arange(1, len(sub) + 1, dtype=np.int16)
        outs.append(sub[CAND_SCHEMA])
        stats.append({"molecule_id": mol, "n_cand": len(sub), "truncated": trunc,
                      "mz_neutral": float(mzn)})

    cand = pd.concat(outs, ignore_index=True) if outs else pd.DataFrame(columns=CAND_SCHEMA)
    return cand, pd.DataFrame(stats)


# ---------------------------------------------------------------- 主入口

def build_candidates(
    test: pd.DataFrame,
    pool: pd.DataFrame,
    truth: Optional[pd.DataFrame] = None,
    ppm: float = C.PPM_DEFAULT,
    exclude_truth: bool = True,
    out_path: Optional[str] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """端到端建候选池。

    truth: 若给出（dev 集），且 exclude_truth=True，则**把真值结构从池中剔除**，
           用于模拟"答案不在库里"的真实情形（class 2/3）。
           **这是本地评估唯一合法的做法**——否则 400/400 命中是假象。
    """
    neutrals = derive_neutral_masses(test)
    exclude = set(truth["cand_key"]) if (truth is not None and exclude_truth) else None

    cand, stats = candidates_for(neutrals, pool, ppm=ppm, exclude_keys=exclude)

    if truth is not None:
        tmap = dict(zip(truth["molecule_id"], truth["cand_key"]))
        hit = cand.groupby("molecule_id")["cand_key"].apply(set)
        stats["truth_in_pool"] = stats["molecule_id"].map(
            lambda m: tmap.get(m) in hit.get(m, set()) if m in tmap else None)

    if out_path:
        cand.to_parquet(out_path, index=False)
        log.info("候选 %d 行 -> %s", len(cand), out_path)
    return cand, stats


def route_by_spectral_evidence(max_cos: pd.Series, threshold: float = 0.5) -> set[str]:
    """按谱证据决定每个分子用哪个池。

    逻辑（对应官方的 class 1 / class 2 划分）：
        * 谱库里能找到高相似参考谱  -> 答案大概就在库里 -> 用**小池**（精度优先）
        * 找不到                     -> 答案只可能在结构库 -> 用**大池**（覆盖优先）

    实测依据（`experiments.md` 2026-11-20）：
        * 小池（train 275k）  class1 MRR 0.7581，候选/分子 33
        * 大池（merged 720k） class1 MRR 0.7111，候选/分子 69  ← 无差别合并代价 −0.047
        * 大池对 class2 的价值：0 → 0.2193

    ⚠️ **阈值未在真实数据上校准**。dev_np 里所有答案都在 train 池，
    无法验证分流的收益（见 experiments.md 的说明）。上线前必须在
    能代表 class 2 的验证集上重测。

    返回：应该走**大池**的 molecule_id 集合。
    """
    return set(max_cos.index[max_cos < threshold])


def build_candidates_routed(
    test: pd.DataFrame,
    pool_small: pd.DataFrame,
    pool_big: pd.DataFrame,
    big_molecules: set[str],
    truth: Optional[pd.DataFrame] = None,
    ppm: float = C.PPM_DEFAULT,
    exclude_truth: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """分流版建候选：一部分分子走小池，一部分走大池。

    pool_small : 通常是 train 结构池（275k）
    pool_big   : 通常是 train + COCONUT + NPAtlas（720k）
    big_molecules : 来自 `route_by_spectral_evidence()`
    """
    neutrals = derive_neutral_masses(test)
    tmap = dict(zip(truth["molecule_id"], truth["cand_key"])) if truth is not None else {}

    parts, stats = [], []
    for use_big, label in ((False, "small"), (True, "big")):
        sel = neutrals[neutrals["molecule_id"].isin(big_molecules) == use_big]
        if sel.empty:
            continue
        pool = pool_big if use_big else pool_small
        exclude = None
        if truth is not None and exclude_truth:
            exclude = [tmap[m] for m in sel["molecule_id"] if m in tmap]
        cand, st = candidates_for(sel, pool, ppm=ppm, exclude_keys=exclude)
        if len(cand):
            cand["pool"] = label
            parts.append(cand)
        st["pool"] = label
        stats.append(st)

    if not parts:
        return pd.DataFrame(columns=CAND_SCHEMA), pd.DataFrame()
    out = pd.concat(parts, ignore_index=True)
    st = pd.concat(stats, ignore_index=True)

    if truth is not None:
        hit = out.groupby("molecule_id")["cand_key"].apply(set)
        st["truth_in_pool"] = st["molecule_id"].map(
            lambda m: tmap.get(m) in hit.get(m, set()) if m in tmap else None)
    return out, st


def candidates_hash(path: Optional[str] = None) -> str:
    """候选池指纹，用于 experiments.md 的可复现性记录。"""
    import hashlib

    path = path or os.path.join(C.CAND_DIR, "candidates.parquet")
    if not os.path.exists(path):
        return "missing"
    df = pd.read_parquet(path, columns=["molecule_id", "cand_key", "ppm"])
    h = hashlib.blake2b(digest_size=8)
    h.update(f"{len(df)}".encode())
    h.update(pd.util.hash_pandas_object(df, index=False).values.tobytes())
    return h.hexdigest()


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--build-pool", action="store_true")
    ap.add_argument("--build-cands", action="store_true")
    ap.add_argument("--ppm", type=float, default=C.PPM_DEFAULT)
    ap.add_argument("--no-exclude", action="store_true",
                    help="不排除真值（用于观察作弊式上限，不要用于评估）")
    a = ap.parse_args()

    pool = build_pool_from_train() if a.build_pool else load_pool()

    if a.build_cands:
        test = pd.read_parquet(C.find_data_file("test.parquet"),
                               columns=["molecule_id", "adduct", "precursor_mz"])
        truth = pd.read_parquet(os.path.join(C.INTERIM_DIR, "dev_labels.parquet"))
        cand, stats = build_candidates(
            test, pool, truth=truth, ppm=a.ppm,
            exclude_truth=not a.no_exclude,
            out_path=os.path.join(C.CAND_DIR, "candidates.parquet"))
        print(stats.describe().round(2).to_string())
        print(f"\ntruth_in_pool: {stats['truth_in_pool'].sum()}/{len(stats)}")
        print("candidates hash:", candidates_hash())
