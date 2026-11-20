"""A 负责：数据加载与泄露检查。

官方 schema（docs/00 §3，已核实）：

test.parquet 12 列
    molecule_id, spectrum_id, ms2_mzs, ms2_normalized_intensities,
    base_peak_intensity, adduct, ionization_mode, instrument_type,
    precursor_mz, collision_energy_ev, collision_energy_orig,
    collision_energy_orig_units

train.parquet 另加 8 列
    normalized_smiles, inchikey, inchikey14, molecular_formula,
    ingest_lib, adduct_orig, precursor_error_ppm, num_peaks

**最关键的一条事实**：可见的 test.parquet 取自训练集，rerun 时会被替换。
在它上面算分必然虚高。`check_leak()` 就是用来亲手确认这件事的。
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
from typing import Optional

import numpy as np
import pandas as pd

import common as C

log = logging.getLogger("casmi.load")

# 原始列名 -> 我们的内部列名
RENAME = {
    "ms2_mzs": "peaks_mz",
    "ms2_normalized_intensities": "peaks_int",
    "ionization_mode": "polarity",       # "positive" / "negative"
    "instrument_type": "instrument",
    "normalized_smiles": "smiles",
}

POLARITY_MAP = {"positive": 1, "negative": -1}


def _ensure_rdkit_cols() -> None:
    """parquet 里的 list 列读进来是 numpy array，统一成 list[float]。"""
    pass


def load_split(path: str, with_labels: bool = True) -> pd.DataFrame:
    """读 train/test parquet，规整成内部 schema。

    返回列：
        molecule_id, spectrum_id, adduct, polarity(int8), instrument,
        precursor_mz(float32), ce_ev(list[float32]), ce_orig(str),
        base_peak_intensity(float32), peaks_mz(list), peaks_int(list),
        num_peaks(int32)
      train 另加：smiles, inchikey14_raw, molecular_formula, ingest_lib,
                adduct_orig, precursor_error_ppm

    注意 `collision_energy_ev` 是 **list**（多能量合并的谱），
    统一成 Python list[float]，取不到就空 list。
    """
    df = pd.read_parquet(path)
    missing = {"molecule_id", "spectrum_id", "ms2_mzs", "precursor_mz"} - set(df.columns)
    if missing:
        raise ValueError(f"{path} 缺少必需列: {missing}")

    out = pd.DataFrame(index=df.index)
    out["molecule_id"] = df["molecule_id"].astype("string")
    out["spectrum_id"] = df["spectrum_id"].astype("string")
    out["adduct"] = df["adduct"].map(C.normalize_adduct).astype("string")
    out["polarity"] = (
        df["ionization_mode"].map(POLARITY_MAP).fillna(0).astype("int8")
        if "ionization_mode" in df else 0
    )
    out["instrument"] = df.get("instrument_type", pd.Series([None] * len(df))).astype("string")
    out["precursor_mz"] = df["precursor_mz"].astype("float32")
    out["base_peak_intensity"] = (
        df["base_peak_intensity"].astype("float32") if "base_peak_intensity" in df else np.nan
    )

    if "collision_energy_ev" in df:
        out["ce_ev"] = df["collision_energy_ev"].map(
            lambda v: [float(x) for x in (v if isinstance(v, (list, tuple, np.ndarray)) else [v])
                       if x is not None and x == x]
        )
    else:
        out["ce_ev"] = [[] for _ in range(len(df))]
    out["ce_orig"] = df.get("collision_energy_orig", pd.Series([None] * len(df))).astype("string")

    out["peaks_mz"] = df["ms2_mzs"].map(lambda v: np.asarray(v, dtype=np.float32))
    out["peaks_int"] = df["ms2_normalized_intensities"].map(
        lambda v: np.asarray(v, dtype=np.float32)
    )
    out["num_peaks"] = (
        df["num_peaks"].astype("int32") if "num_peaks" in df
        else out["peaks_mz"].map(len).astype("int32")
    )

    if with_labels:
        if "normalized_smiles" in df:
            out["smiles"] = df["normalized_smiles"].astype("string")
        if "inchikey14" in df:
            out["ik14_raw"] = df["inchikey14"].astype("string")
        if "molecular_formula" in df:
            out["molecular_formula"] = df["molecular_formula"].astype("string")
        if "ingest_lib" in df:
            out["ingest_lib"] = df["ingest_lib"].astype("string")
        if "adduct_orig" in df:
            out["adduct_orig"] = df["adduct_orig"].astype("string")
        if "precursor_error_ppm" in df:
            out["precursor_error_ppm"] = df["precursor_error_ppm"].astype("float32")

    return out.reset_index(drop=True)


# ---------------------------------------------------------------- 泄露检查

def spectrum_fingerprint(mz, intensity, ndigits: int = 3) -> str:
    """谱指纹：排序后的 m/z 四舍五入 + 强度粗量化，用于跨文件比对同一张谱。"""
    if mz is None or len(mz) == 0:
        return ""
    m = np.round(np.asarray(mz, dtype=np.float64), ndigits)
    i = np.round(np.asarray(intensity, dtype=np.float64) * 100).astype(np.int32)
    h = hashlib.blake2b(digest_size=8)
    h.update(np.ascontiguousarray(m).tobytes())
    h.update(np.ascontiguousarray(i).tobytes())
    return h.hexdigest()


def check_leak(test: pd.DataFrame, train: pd.DataFrame) -> dict:
    """确认「可见 test 取自训练集」这条事实。

    三项检查：
      1. spectrum_id 重叠
      2. 谱指纹重叠
      3. molecule_id 重叠（可见 test 的匿名 id 不应出现在 train，但结构可能重复）

    **必须第一天亲手跑一次**，不要只信文档。
    """
    tsid = set(test["spectrum_id"].dropna())
    trsid = set(train["spectrum_id"].dropna())
    overlap_sid = tsid & trsid

    test["_fp"] = [spectrum_fingerprint(m, i) for m, i in zip(test["peaks_mz"], test["peaks_int"])]
    train["_fp"] = [spectrum_fingerprint(m, i) for m, i in zip(train["peaks_mz"], train["peaks_int"])]
    tfp, trfp = set(test["_fp"]) - {""}, set(train["_fp"]) - {""}
    overlap_fp = tfp & trfp

    res = {
        "n_test_spectra": len(test),
        "n_test_molecules": int(test["molecule_id"].nunique()),
        "n_train_spectra": len(train),
        "spectrum_id_overlap": len(overlap_sid),
        "spectrum_id_overlap_frac": len(overlap_sid) / max(len(tsid), 1),
        "fingerprint_overlap": len(overlap_fp),
        "fingerprint_overlap_frac": len(overlap_fp) / max(len(tfp), 1),
    }
    test.drop(columns="_fp", inplace=True, errors="ignore")
    train.drop(columns="_fp", inplace=True, errors="ignore")
    return res


def assert_visible_test_is_train_sample(res: dict, min_frac: float = 0.5) -> None:
    """如果可见 test 真的取自 train，指纹重叠率应该很高（接近 1）。

    **这条断言是给队友看的**：任何人想在可见 test 上调参，先看这个数字。
    """
    if res["fingerprint_overlap_frac"] < min_frac:
        log.warning(
            "可见 test 与 train 的指纹重叠只有 %.1f%%，低于预期 %.0f%%。"
            "先确认你读的是不是官方 test.parquet。",
            res["fingerprint_overlap_frac"] * 100, min_frac * 100,
        )
    else:
        log.warning(
            "确认：可见 test.parquet 有 %.1f%% 的谱能在 train 里找到同一张谱。"
            "**在它上面算分是作弊式的虚高，本地验证请从 train 切分。**",
            res["fingerprint_overlap_frac"] * 100,
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default=os.path.join(C.RAW_DIR, "train.parquet"))
    ap.add_argument("--test", default=os.path.join(C.RAW_DIR, "test.parquet"))
    ap.add_argument("--check-leak", action="store_true")
    ap.add_argument("--stats", action="store_true")
    a = ap.parse_args()

    if a.check_leak:
        if not (os.path.exists(a.train) and os.path.exists(a.test)):
            raise SystemExit(f"找不到数据：{a.train} / {a.test}，先下载")
        tr = load_split(a.train, with_labels=True)
        te = load_split(a.test, with_labels=False)
        res = check_leak(te, tr)
        for k, v in res.items():
            print(f"  {k}: {v}")
        assert_visible_test_is_train_sample(res)

    if a.stats:
        tr = load_split(a.train, with_labels=True)
        print(tr.groupby("ingest_lib").size().sort_values(ascending=False))
        print("\nadduct 分布（test 应为 10 种）:")
        print(tr["adduct"].value_counts().head(15))
        print("\n每分子谱数:", tr.groupby("molecule_id").size().describe())


if __name__ == "__main__":
    main()
