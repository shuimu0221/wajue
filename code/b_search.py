"""B 负责：谱库检索通道。

实现策略（内存安全）
------------------
train 有 253 万张谱、约 4 亿个峰，**不能整池进内存**。
做法是**按 row group 流式扫描**：每个行组只读一次，
在行组内按中性质量排序 + searchsorted 定位窗口，只对窗口内少数谱算相似度。

三条通道
--------
* `f_cos`      经典余弦，强度 sqrt 变换（官方推荐的前处理）
* `f_nl`       中性丢失：`precursor_mz - fragment_mz` 后匹配（同系物主力）
* `f_lib_hit`  是否存在高相似参考谱（0/1），供 ranker 判断"有谱证据"

关键：**跨加合物统一到中性质量**再比对，否则 [M+H]+ 的谱和 [M+Na]+ 的谱匹配不上。
测试集只有 10 种加合物（common.TEST_ADDUCTS），无二聚体。
"""

from __future__ import annotations

import logging
import math
import os
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import common as C

log = logging.getLogger("casmi.search")

DEFAULT_CFG = {
    "ppm": 20.0,          # 母离子质量窗口（放宽，因为检索不怕多一点候选）
    "min_da": 0.01,
    "mz_tol": 0.01,       # 碎片峰匹配容差 Da
    "n_peaks": 128,       # 每谱最多保留的峰数（按强度）
    "sqrt_intensity": True,
    "min_peaks": 3,
    "hit_threshold": 0.7,  # f_lib_hit 的判定阈值
}


# ---------------------------------------------------------------- 前处理

def prep_peaks(mz, it, cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    """官方推荐的规范化：取强度 top-N，sqrt 变换，按 m/z 升序。"""
    mz = np.asarray(mz, dtype=np.float32)
    it = np.asarray(it, dtype=np.float32)
    if len(mz) == 0:
        return mz, it
    n = cfg.get("n_peaks", 128)
    if len(mz) > n:
        idx = np.argpartition(it, -n)[-n:]
        mz, it = mz[idx], it[idx]
    order = np.argsort(mz)
    mz, it = mz[order], it[order]
    if cfg.get("sqrt_intensity", True):
        it = np.sqrt(np.clip(it, 0, None))
    norm = float(np.linalg.norm(it))
    if norm > 0:
        it = it / norm
    return mz, it


def cosine_sorted(mz1, i1, mz2, i2, tol: float) -> float:
    """两个**已按 m/z 升序且已归一化**的峰表的余弦。双指针 O(n)。"""
    i = j = 0
    dot = 0.0
    n1, n2 = len(mz1), len(mz2)
    while i < n1 and j < n2:
        d = float(mz1[i]) - float(mz2[j])
        if d < -tol:
            i += 1
        elif d > tol:
            j += 1
        else:
            dot += float(i1[i]) * float(i2[j])
            i += 1
            j += 1
    return dot


def neutral_loss_peaks(precursor_mz: float, mz: np.ndarray, it: np.ndarray):
    """转成中性丢失谱：loss = precursor - fragment，按 loss 升序。"""
    loss = precursor_mz - mz
    keep = loss > 0.5          # 丢掉母离子自身和负值
    loss, inten = loss[keep], it[keep]
    order = np.argsort(loss)
    return loss[order], inten[order]


# ---------------------------------------------------------------- 质量向量化

def spectrum_fp(mz, it, nd: int = 3) -> int:
    """谱内容指纹 -> 64 位整数（与 _leak.py 同口径）。"""
    import hashlib
    m = np.round(np.asarray(mz, dtype=np.float64), nd)
    i = np.round(np.asarray(it, dtype=np.float64) * 100).astype(np.int32)
    h = hashlib.blake2b(digest_size=8)
    h.update(np.ascontiguousarray(m).tobytes())
    h.update(np.ascontiguousarray(i).tobytes())
    return int.from_bytes(h.digest(), "little")


def _neutral_masses(precursor_mz: np.ndarray, adducts: np.ndarray) -> np.ndarray:
    """向量化：加合物 m/z -> 中性质量。未知加合物给 NaN。

    用 dict 缓存查表（加合物只有 ~121 种），避免 np.unique 对 13 万条字符串排序。
    """
    cache: dict[str, tuple[float, float, float]] = {}

    def lookup(a: str) -> tuple[float, float, float]:
        if a not in cache:
            na = C.normalize_adduct(a)
            d = C.ADDUCT_DELTAS.get(na, np.nan)
            n = 2.0 if C.is_dimer(na) else 1.0
            z = abs(C.charge_of(na)) or 1
            cache[a] = (d, n, float(z))
        return cache[a]

    out = np.empty(len(precursor_mz), dtype=np.float64)
    for i, (mz, a) in enumerate(zip(precursor_mz, adducts)):
        d, nd, z = lookup(a)
        out[i] = (mz * z - d) / nd if d == d else np.nan
    return out


# ---------------------------------------------------------------- 扫描

class LibraryScanner:
    """流式扫描 train，按中性质量窗口找相似谱。

    用法：
        sc = LibraryScanner(train_path)
        hits = sc.run(queries, cfg)      # 每行组扫一遍
    """

    def __init__(self, train_path: Optional[str] = None):
        self.path = train_path or C.find_data_file("train.parquet")
        self.pf = pq.ParquetFile(self.path)

    def run(
        self,
        queries: pd.DataFrame,
        cfg: Optional[dict] = None,
        exclude_keys: Optional[Iterable[str]] = None,
        verbose: bool = True,
    ) -> pd.DataFrame:
        """queries: molecule_id, precursor_mz, adduct, peaks_mz, peaks_int

        返回长表：molecule_id, cand_key, f_cos, f_nl（同结构取 max）
        """
        cfg = {**DEFAULT_CFG, **(cfg or {})}
        exclude = set(exclude_keys or ())
        tol_mz = cfg["mz_tol"]

        # 预处理查询谱
        q_mz, q_it, q_nl, q_mass, q_fp = [], [], [], [], []
        for _, r in queries.iterrows():
            mz, it = prep_peaks(r["peaks_mz"], r["peaks_int"], cfg)
            q_mz.append(mz)
            q_it.append(it)
            q_nl.append(neutral_loss_peaks(float(r["precursor_mz"]), mz, it))
            q_mass.append(C.neutral_mass_from_precursor(float(r["precursor_mz"]), r["adduct"]))
            q_fp.append(spectrum_fp(r["peaks_mz"], r["peaks_int"]))
        q_mass = np.array([m if m is not None else np.nan for m in q_mass])
        q_mol = queries["molecule_id"].to_numpy()
        q_fp = np.array(q_fp, dtype=np.uint64)
        n_q = len(queries)

        # 每 (query, cand_key) 只保留最大分
        best: dict[tuple[str, str], tuple[float, float]] = {}

        n_rg = self.pf.metadata.num_row_groups
        for rgi in range(n_rg):
            t = self.pf.read_row_group(rgi, columns=["precursor_mz", "adduct", "inchikey14"])
            pmz_all = t.column("precursor_mz").to_numpy(zero_copy_only=False).astype(np.float64)
            ad_all = t.column("adduct").to_numpy(zero_copy_only=False).astype(str)
            keys_all = t.column("inchikey14").to_numpy(zero_copy_only=False).astype(str)
            lib_mass = _neutral_masses(pmz_all, ad_all)

            ok = np.isfinite(lib_mass)
            if exclude:
                ok &= ~np.isin(keys_all, list(exclude))
            if not ok.any():
                continue
            idx_ok = np.flatnonzero(ok)
            lm = lib_mass[idx_ok]
            order = np.argsort(lm)
            lm_sorted = lm[order]
            idx_sorted = idx_ok[order]

            # 先用质量窗口求出「本行组里哪些行真的需要」，
            # 再只把这些行读出来转 python list。
            # 实测：13 万行里通常只有几百行进窗口，直接 to_pylist 全量会慢 10 倍以上。
            need = np.zeros(len(pmz_all), dtype=bool)
            windows: dict[int, tuple[int, int]] = {}
            for qi in range(n_q):
                if not np.isfinite(q_mass[qi]):
                    continue
                tol = max(abs(q_mass[qi]) * cfg["ppm"] * 1e-6, cfg["min_da"])
                lo, hi = np.searchsorted(lm_sorted, [q_mass[qi] - tol, q_mass[qi] + tol])
                windows[qi] = (lo, hi)
                if hi > lo:
                    need[idx_sorted[lo:hi]] = True
            if not need.any():
                continue

            t2 = self.pf.read_row_group(
                rgi, columns=["ms2_mzs", "ms2_normalized_intensities"])
            import pyarrow as pa
            t2 = t2.filter(pa.array(need))
            keep_rows = np.flatnonzero(need)
            row_pos = {int(r): i for i, r in enumerate(keep_rows)}
            mz_list = t2.column("ms2_mzs").to_pylist()
            it_list = t2.column("ms2_normalized_intensities").to_pylist()

            prepped: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
            row_fp: dict[int, int] = {}

            def get_prepped(li: int):
                got = prepped.get(li)
                if got is None:
                    p = row_pos.get(li)
                    if p is None:
                        got = (np.empty(0, np.float32),) * 4
                    else:
                        lmz, lit = prep_peaks(mz_list[p], it_list[p], cfg)
                        if len(lmz) >= cfg["min_peaks"]:
                            lnl = neutral_loss_peaks(pmz_all[li], lmz, lit)
                            got = (lmz, lit, lnl[0], lnl[1])
                        else:
                            got = (np.empty(0, np.float32),) * 4
                        row_fp[li] = spectrum_fp(mz_list[p], it_list[p])
                    prepped[li] = got
                return got

            n_hit = 0
            for qi, (lo, hi) in windows.items():
                for pos in range(lo, hi):
                    li = idx_sorted[pos]
                    lmz, lit, lnl_mz, lnl_it = get_prepped(li)
                    if len(lmz) < cfg["min_peaks"]:
                        continue
                    c = cosine_sorted(q_mz[qi], q_it[qi], lmz, lit, tol_mz)
                    nl = cosine_sorted(q_nl[qi][0], q_nl[qi][1], lnl_mz, lnl_it, tol_mz)
                    k = (q_mol[qi], keys_all[li])
                    acc = best.get(k)
                    if acc is None:
                        acc = [-1.0, -1.0, -1.0]   # cos(excl self), nl(excl self), cos(self)
                        best[k] = acc
                    if row_fp.get(li, 0) == int(q_fp[qi]):
                        # 同一张谱：单独记账，不计入主分数
                        if c > acc[2]:
                            acc[2] = float(c)
                    else:
                        if c > acc[0]:
                            acc[0] = float(c)
                        if nl > acc[1]:
                            acc[1] = float(nl)
                    n_hit += 1
            if verbose:
                log.info("  rg %2d/%d: mass_window_hits=%d", rgi + 1, n_rg, n_hit)

        if not best:
            return pd.DataFrame(columns=["molecule_id", "cand_key", "f_cos",
                                         "f_nl", "f_cos_self"])

        rows = [{"molecule_id": m, "cand_key": k,
                 "f_cos": v[0], "f_nl": v[1], "f_cos_self": v[2]}
                for (m, k), v in best.items()]
        return pd.DataFrame(rows)


def add_lib_hit(df: pd.DataFrame, cfg: Optional[dict] = None) -> pd.DataFrame:
    cfg = {**DEFAULT_CFG, **(cfg or {})}
    out = df.copy()
    out["f_lib_hit"] = (out["f_cos"] >= cfg["hit_threshold"]).astype(np.int8)
    return out


def search_scores(
    cand_tbl: pd.DataFrame,
    scanner: LibraryScanner,
    queries: pd.DataFrame,
    cfg: Optional[dict] = None,
    exclude_keys: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """对候选打分，缺失通道填 NaN（契约要求：不挑行）。"""
    hits = scanner.run(queries, cfg=cfg, exclude_keys=exclude_keys)
    hits = add_lib_hit(hits, cfg)
    out = cand_tbl[["molecule_id", "cand_key"]].merge(
        hits, on=["molecule_id", "cand_key"], how="left")
    for c in ("f_cos", "f_nl", "f_lib_hit"):
        if c not in out:
            out[c] = np.nan
    return out[["molecule_id", "cand_key", "f_cos", "f_nl", "f_lib_hit"]]


if __name__ == "__main__":
    print("cfg:", DEFAULT_CFG)
