"""A 负责：提交校验与拼装。

**格式以官方 Evaluation 页原文为准**（docs/00 §4）：

    molecule_id,smiles
    m_0014ef,CC1=CC(=O)C=CC1=O;OC(=O)c1ccccc1O
    m_004d06,NCCc1ccc(O)cc1;CC(=O)Nc1ccc(O)cc1

拒绝条件（原文列出，`validate_sub` 必须全部覆盖）：
  * 缺 `molecule_id` 或 `smiles` 列
  * 文件为空
  * 任一列含 null
  * `molecule_id` 重复
  * 任一分子超过 25 个分号分隔的猜测
"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

import common as C

log = logging.getLogger("casmi.submit")

REQUIRED_COLS = ["molecule_id", "smiles"]


def validate_sub(sub: pd.DataFrame, expected_ids: Optional[set] = None, k: int = C.TOPK) -> list[str]:
    """返回错误列表；空列表 = 通过。**每次提交前必跑。**"""
    errs: list[str] = []

    if list(sub.columns) != REQUIRED_COLS:
        errs.append(f"列名必须精确为 {REQUIRED_COLS}，实际 {list(sub.columns)}")
        if "molecule_id" not in sub.columns or "smiles" not in sub.columns:
            return errs

    if len(sub) == 0:
        errs.append("提交为空")

    if sub[REQUIRED_COLS].isna().any().any():
        n_null = int(sub[REQUIRED_COLS].isna().any(axis=1).sum())
        errs.append(f"{n_null} 行含 null（官方直接拒收）")

    if sub["molecule_id"].duplicated().any():
        dups = sub.loc[sub["molecule_id"].duplicated(), "molecule_id"].head(5).tolist()
        errs.append(f"molecule_id 重复：{dups}")

    if expected_ids is not None:
        got, want = set(sub["molecule_id"]), set(expected_ids)
        if got != want:
            miss, extra = want - got, got - want
            errs.append(f"id 集合不一致：缺 {len(miss)} 个（如 {list(miss)[:3]}），多 {len(extra)} 个")

    try:
        from rdkit import Chem
        rdkit_ok = True
    except ImportError:
        rdkit_ok = False
        log.warning("rdkit 未安装，跳过 SMILES 可解析性检查")

    n_over, n_bad, n_empty = 0, 0, 0
    for val in sub["smiles"]:
        if not isinstance(val, str):
            n_empty += 1
            continue
        items = [s.strip() for s in val.split(";")]
        if not items or any(not s for s in items):
            n_empty += 1
            continue
        if len(items) > k:
            n_over += 1
        if rdkit_ok:
            for s in items:
                if Chem.MolFromSmiles(s) is None:
                    n_bad += 1
                    break

    if n_empty:
        errs.append(f"{n_empty} 行含空串或非字符串")
    if n_over:
        errs.append(f"{n_over} 行超过 {k} 个 SMILES（官方直接拒收）")
    if n_bad:
        errs.append(f"{n_bad} 行含不可解析 SMILES")

    return errs


def assemble_submission(
    scores: pd.DataFrame,
    topk: int = C.TOPK,
    key_col: str = "cand_key",
    smiles_col: str = "smiles",
) -> pd.DataFrame:
    """分数长表 -> 提交表。

    scores: [molecule_id, <key_col>, <smiles_col>, score]

    规则（docs/00 §4）：
      * 按 score 降序
      * 同一 molecule_id 内按 cand_key 去重——同一个 InChIKey14 出现两次，
        第二个永远不可能得分，还白占一格
      * 最多 topk 个
      * cand_key 为空的行丢弃（无法判等）
      * **每个 molecule_id 恰好一行**（官方硬要求）
    """
    df = scores.dropna(subset=[key_col]).copy()
    df = df.sort_values(["molecule_id", "score"], ascending=[True, False], kind="mergesort")
    df = df.drop_duplicates(["molecule_id", key_col], keep="first")
    df = df.groupby("molecule_id", sort=False).head(topk)

    out = (
        df.groupby("molecule_id", sort=False)[smiles_col]
        .apply(lambda s: ";".join(s.tolist()))
        .reset_index()
    )
    out.columns = REQUIRED_COLS
    return out


def save(sub: pd.DataFrame, path: str) -> None:
    sub.to_csv(path, index=False)
    log.info("written %s (%d rows)", path, len(sub))


# ---------------------------------------------------------------- 自检

def _selftest() -> bool:
    ok = True

    def check(cond, msg):
        nonlocal ok
        if not cond:
            ok = False
            print(f"FAIL {msg}")
        else:
            print(f"ok   {msg}")

    scores = pd.DataFrame(
        {
            "molecule_id": ["m1"] * 4 + ["m2"],
            "cand_key": ["K1", "K2", "K2", "K3", "K9"],
            "smiles": ["CCO", "CCN", "CCN", "CCC", "c1ccccc1"],
            "score": [0.9, 0.8, 0.7, 0.6, 0.5],
        }
    )
    sub = assemble_submission(scores, topk=3)
    check(len(sub) == 2, "two rows out")
    check(list(sub.columns) == REQUIRED_COLS, f"columns == {REQUIRED_COLS}")
    m1 = sub.loc[sub.molecule_id == "m1", "smiles"].iloc[0]
    check(m1 == "CCO;CCN;CCC", f"dedupe + order -> {m1}")

    # rdkit 未安装时 validate_sub 会跳过可解析性检查，但这些结构都合法
    check(validate_sub(sub) == [], "valid submission passes")

    dup = pd.concat([sub, sub.iloc[[0]]], ignore_index=True)
    check(any("重复" in e for e in validate_sub(dup)), "detects duplicate id")

    over = pd.DataFrame({"molecule_id": ["m1"], "smiles": [";".join(["C"] * 26)]})
    check(any("超过" in e for e in validate_sub(over)), "detects >25 guesses")

    empty = pd.DataFrame({"molecule_id": ["m1"], "smiles": ["CCO;"]})
    check(any("空串" in e for e in validate_sub(empty)), "detects empty item")

    nul = pd.DataFrame({"molecule_id": ["m1", "m2"], "smiles": ["CCO", None]})
    check(any("null" in e for e in validate_sub(nul)), "detects null")

    nofile = pd.DataFrame({"molecule_id": []})
    check(any("列名" in e for e in validate_sub(nofile)) or validate_sub(nofile), "detects bad cols")

    print("\nselftest:", "PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    import sys

    if "--selftest" in sys.argv:
        raise SystemExit(0 if _selftest() else 1)
    print("用法: python code/a_submit.py --selftest")
