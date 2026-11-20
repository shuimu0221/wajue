"""把外部结构库（COCONUT / PubChem / NPAtlas）转成候选池。

**设计目标：不管你下到的是什么格式，这个脚本都能吃。**

自动识别列名（大小写不敏感）：
    SMILES : smiles / SMILES / canonical_smiles / isomeric_smiles / structure / sdf
    质量   : exact_mass / monoisotopic_mass / exact_molecular_weight / mass /
             molecular_weight / mw / mz / precursor_mz
    分子式 : molecular_formula / formula / mol_formula / molecularformula

质量优先级：**给定质量 > 从分子式算 > 丢弃**。
key 优先级：**给定 InChIKey 前 14 位 > 用 rdkit 算 > 用 canonical SMILES 兜底**。

支持格式：.csv / .tsv / .txt(制表符) / .parquet / .sdf / .smi

用法:
    python code/a_ingest_external.py k_data/COCONUT.csv --name coconut
    python code/a_ingest_external.py k_data/pubchem.parquet --name pubchem --max-mz 1300
    python code/a_ingest_external.py k_data/ --name npatlas          # 整个目录
    python code/a_ingest_external.py --report                        # 只看现有池

关键过滤（**别省**）：
    * 质量窗口：默认 100–1300 Da。测试集单同位素质量 157–1159 Da，
      中位 348。库外的结构（比如分子量 5000 的多肽）纯属浪费候选位。
    * 去重：按 cand_key。
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import chem
import common as C

SMILES_KEYS = ["smiles", "canonical_smiles", "isomeric_smiles", "smiles_std",
               "structure", "mol_smiles", "can_smiles"]
MASS_KEYS = ["exact_mass", "monoisotopic_mass", "exact_molecular_weight",
             "monoisotopic_mw", "mass", "molecular_weight", "mol_weight",
             "mw", "mz", "precursor_mz", "neutral_mass", "exactmass"]
FORMULA_KEYS = ["molecular_formula", "formula", "mol_formula",
                "molecularformula", "molform"]
INCHIKEY_KEYS = ["inchikey14", "inchikey", "inchi_key", "ik14"]


def _find_col(cols, candidates):
    """大小写/下划线不敏感地找列名。返回第一个命中的原始列名。"""
    norm = {str(c).strip().lower().replace(" ", "_").replace("-", "_"): c for c in cols}
    for k in candidates:
        if k in norm:
            return norm[k]
    # 退一步：子串匹配
    for k in candidates:
        for nk, orig in norm.items():
            if k in nk:
                return orig
    return None


def read_any(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    print(f"[read] {os.path.basename(path)} ({ext})")
    if ext == ".parquet":
        return pd.read_parquet(path)
    if ext == ".sdf":
        from rdkit import Chem
        rows, supp = [], Chem.ForwardSDMolSupplier(path, sanitize=False)
        n = 0
        for m in supp:
            if m is None:
                continue
            try:
                smi = Chem.MolToSmiles(m)
            except Exception:
                continue
            rows.append({"smiles": smi})
            n += 1
            if n % 50000 == 0:
                print(f"   ... {n:,}", flush=True)
        return pd.DataFrame(rows)
    if ext in (".smi", ".smiles"):
        return pd.read_csv(path, sep=None, engine="python", header=None,
                           names=["smiles", "name"] if _has_two_cols(path) else ["smiles"])
    # csv / tsv / txt
    sep = "\t" if ext in (".tsv", ".txt") else ","
    try:
        df = pd.read_csv(path, sep=sep, low_memory=False, on_bad_lines="skip")
        if df.shape[1] == 1:      # 分隔符猜错了
            df = pd.read_csv(path, sep=None, engine="python", low_memory=False,
                             on_bad_lines="skip")
        return df
    except Exception as e:
        raise SystemExit(f"读不了 {path}: {e}")


def _has_two_cols(path: str) -> bool:
    with open(path, encoding="utf-8", errors="replace") as f:
        line = f.readline()
    return len(line.split()) >= 2


def normalize(df: pd.DataFrame, name: str, source: str,
              min_mz: float, max_mz: float) -> pd.DataFrame:
    cols = list(df.columns)
    c_smi = _find_col(cols, SMILES_KEYS)
    c_mass = _find_col(cols, MASS_KEYS)
    c_form = _find_col(cols, FORMULA_KEYS)
    c_ik = _find_col(cols, INCHIKEY_KEYS)

    print(f"  识别: smiles={c_smi!r} mass={c_mass!r} formula={c_form!r} ik={c_ik!r}")
    if c_smi is None:
        raise SystemExit(f"{name}: 找不到 SMILES 列（实际列: {cols[:15]}）")
    if c_mass is None and c_form is None and c_ik is None:
        raise SystemExit(f"{name}: 找不到质量/分子式/InChIKey 中的任何一个")

    out = pd.DataFrame(index=df.index)
    out["smiles"] = df[c_smi].astype("string")
    out = out[out["smiles"].notna() & (out["smiles"].str.len() > 2)]

    if c_form is not None:
        out["molecular_formula"] = df.loc[out.index, c_form].astype("string")
    else:
        out["molecular_formula"] = pd.NA

    # 质量：先看给定列
    mz = pd.Series(np.nan, index=out.index, dtype="float64")
    if c_mass is not None:
        mz = pd.to_numeric(df.loc[out.index, c_mass], errors="coerce")
        print(f"  给定质量: {mz.notna().sum():,} 个可用")
    # 缺失的从分子式补
    need = mz.isna()
    if need.any() and c_form is not None:
        calc = pd.Series(chem.mass_vector(out.loc[need, "molecular_formula"]),
                         index=out.index[need], dtype="float64")
        mz = mz.fillna(calc)
        print(f"  分子式补算: {calc.notna().sum():,} 个")
    out["mz_neutral"] = mz

    # key
    if c_ik is not None:
        out["cand_key"] = df.loc[out.index, c_ik].astype("string").str.slice(0, 14)
    else:
        out["cand_key"] = pd.NA

    out["source"] = source
    before = len(out)

    out = out.dropna(subset=["mz_neutral"])
    inrange = out["mz_neutral"].between(min_mz, max_mz)
    print(f"  质量窗口 {min_mz:.0f}–{max_mz:.0f} Da: {inrange.sum():,}/{before:,}")
    out = out[inrange]

    print(f"[{name}] {len(out):,} 个结构可用")
    return out


def fill_keys(df: pd.DataFrame) -> pd.DataFrame:
    """用 rdkit 补 cand_key（tautomer 规范化后 InChIKey14，与评测同口径）。"""
    from rdkit import Chem, RDLogger
    from rdkit.Chem.MolStandardize import rdMolStandardize
    RDLogger.DisableLog("rdApp.*")

    need = df["cand_key"].isna() | (df["cand_key"].astype(str).str.len() != 14)
    print(f"[key] 需要计算: {need.sum():,}/{len(df):,}")
    if not need.any():
        return df

    te = rdMolStandardize.TautomerEnumerator()
    keys, bad = [], 0
    sub = df.loc[need, "smiles"]
    for i, smi in enumerate(sub, 1):
        try:
            m = Chem.MolFromSmiles(smi)
            if m is None:
                keys.append(pd.NA)
                bad += 1
            else:
                k = Chem.MolToInchiKey(te.Canonicalize(m))
                keys.append(k[:14] if k else pd.NA)
                if not k:
                    bad += 1
        except Exception:
            keys.append(pd.NA)
            bad += 1
        if i % 50000 == 0:
            print(f"   ... {i:,}/{len(sub):,}", flush=True)
    df.loc[need, "cand_key"] = keys
    print(f"[key] 解析失败 {bad:,}")
    return df


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?", help="文件或目录")
    ap.add_argument("--name", default="external")
    ap.add_argument("--source", default=None, help="source 标签（默认=name）")
    ap.add_argument("--min-mz", type=float, default=100.0)
    ap.add_argument("--max-mz", type=float, default=1300.0)
    ap.add_argument("--out", default=os.path.join(C.CAND_DIR, "pool_external.parquet"))
    ap.add_argument("--append", action="store_true", help="并入已有 pool_external")
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()

    if a.report:
        for f in sorted(glob.glob(os.path.join(C.CAND_DIR, "*.parquet"))):
            try:
                d = pd.read_parquet(f, columns=["cand_key"])
                print(f"  {os.path.basename(f):<32} {len(d):>9,} 结构")
            except Exception:
                print(f"  {os.path.basename(f):<32} (读不了)")
        return 0

    if not a.path:
        raise SystemExit("需要 path。用 --report 看现有池。")

    paths = ([a.path] if os.path.isfile(a.path)
             else sorted(sum((glob.glob(os.path.join(a.path, e))
                              for e in ("*.csv", "*.tsv", "*.txt", "*.parquet", "*.sdf")), [])))
    if not paths:
        raise SystemExit(f"{a.path} 下没有可读文件")

    parts = []
    for p in paths:
        raw = read_any(p)
        d = normalize(raw, os.path.basename(p), a.source or a.name,
                      a.min_mz, a.max_mz)
        if len(d):
            parts.append(d)
    if not parts:
        raise SystemExit("所有文件都没有可用结构")

    pool = pd.concat(parts, ignore_index=True)
    pool = fill_keys(pool)
    pool = pool.dropna(subset=["cand_key"])
    before = len(pool)
    pool = pool.drop_duplicates("cand_key").reset_index(drop=True)
    print(f"[dedupe] {before:,} -> {len(pool):,}")

    if a.append and os.path.exists(a.out):
        old = pd.read_parquet(a.out)
        pool = pd.concat([old, pool], ignore_index=True)
        pool = pool.drop_duplicates("cand_key", keep="first")
        print(f"[append] + {len(old):,} 已存在 -> {len(pool):,}")

    pool = pool.sort_values("mz_neutral").reset_index(drop=True)
    cols = ["cand_key", "smiles", "molecular_formula", "mz_neutral", "source"]
    pool[cols].to_parquet(a.out, index=False)
    print(f"\nwritten {a.out}  ({len(pool):,} 结构, "
          f"{pool.mz_neutral.min():.0f}–{pool.mz_neutral.max():.0f} Da)")
    print(pool["source"].value_counts().to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
