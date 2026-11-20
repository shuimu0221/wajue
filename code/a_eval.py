"""A 负责：评测、折划分、泄漏断言。

判分规则见 docs/00-赛题理解与判分细节.md §3；验证协议见 docs/03-验证协议.md。

关键约定
--------
* 候选/提交/标签一律用 **cand_key**（= metric key）比对，**不要用原始 SMILES 字符串**。
* `metric_key()` 依赖固定版本的 RDKit；若本机版本不一致，请用 A 预先缓存的
  `data/interim/ik14_map.parquet`（raw_ik14 -> metric_key），不要自己重算。
"""

from __future__ import annotations

import argparse
import logging
from typing import Callable, Iterable, Optional, Sequence

import numpy as np
import pandas as pd

import common as C

log = logging.getLogger("casmi.eval")

# ---------------------------------------------------------------- metric key

_KEY_CACHE: dict[str, Optional[str]] = {}


def metric_key(smiles: str, canon_cache: Optional[dict] = None) -> Optional[str]:
    """SMILES -> metric key（tautomer canonicalization 后 InChIKey 前 14 位）。

    与判分器一致的三步：
      1. 解析；2. RDKit TautomerEnumerator 规范化；3. InChIKey[:14]

    本机 RDKit 版本必须等于 C.RDKIT_METRIC_VERSION，否则结果的 1.58% 会漂移。
    """
    if smiles is None:
        return None
    if canon_cache is not None and smiles in canon_cache:
        return canon_cache[smiles]
    if smiles in _KEY_CACHE:
        return _KEY_CACHE[smiles]

    try:
        from rdkit import Chem, RDLogger
        from rdkit.Chem.MolStandardize import rdMolStandardize

        RDLogger.DisableLog("rdApp.*")
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            _KEY_CACHE[smiles] = None
            return None
        te = rdMolStandardize.TautomerEnumerator()
        mol = te.Canonicalize(mol)
        ik = Chem.MolToInchiKey(mol)
        key = ik[:14] if ik else None
    except ImportError:
        raise RuntimeError(
            "rdkit 未安装。先 pip install rdkit==%s，或改用缓存的 ik14_map" % C.RDKIT_METRIC_VERSION
        )
    _KEY_CACHE[smiles] = key
    if canon_cache is not None:
        canon_cache[smiles] = key
    return key


def heavy_formula(smiles: str) -> Optional[str]:
    """重原子组成串。**这是我们的预筛手段，不是官方判等规则。**

    官方规则只有一条（docs/00 §4）：tautomer 规范化后比 InChIKey14。
    不过 InChIKey14 天然编码了骨架，所以分子式不符的候选可以被安全地提前剔除，
    纯属加速手段。不要把它当成"官方会判否"来推理。
    """
    try:
        from rdkit import Chem
        from rdkit.Chem.rdMolDescriptors import CalcMolFormula

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        # 只保留重原子计数，避免氢数口径不一致
        f = CalcMolFormula(mol)
        parts = []
        for el, n in __import__("re").findall(r"([A-Z][a-z]?)(\d*)", f):
            if el and el != "H":
                parts.append(el + (n if n else "1"))
        return "".join(sorted(parts))
    except ImportError:
        return None


# ---------------------------------------------------------------- 核心指标

def mrr_from_lists(
    pred: Sequence[Sequence[str]],
    truth: Sequence[str],
    k: int = C.TOPK,
) -> np.ndarray:
    """纯数学部分：每个分子的 reciprocal rank（未命中 = 0）。

    pred[i] 已按置信度降序、且已去重、且元素是 metric key。
    注意「只取第一个命中位置」，同时防御性地截断到 k。
    """
    assert len(pred) == len(truth), f"长度不一致 {len(pred)} vs {len(truth)}"
    out = np.zeros(len(pred), dtype=np.float64)
    for i, (lst, t) in enumerate(zip(pred, truth)):
        if t is None:
            continue
        for rank, key in enumerate(lst[:k], start=1):
            if key == t:
                out[i] = 1.0 / rank
                break
    return out


def recall_at(pred: Sequence[Sequence[str]], truth: Sequence[str], k: int) -> float:
    hits = sum(1 for lst, t in zip(pred, truth) if t is not None and t in set(lst[:k]))
    n = sum(1 for t in truth if t is not None)
    return hits / n if n else 0.0


def local_eval(
    sub: pd.DataFrame,
    truth: pd.DataFrame,
    folds: Optional[pd.DataFrame] = None,
    k: int = C.TOPK,
    key_of: Callable[[str], Optional[str]] = metric_key,
) -> dict:
    """严格 MRR@25 + 分层诊断。

    sub   : columns [molecule_id, smiles]（smiles 是 ';' 分隔字符串或 list）
    truth : columns [molecule_id, smiles]（唯一正确答案）
    folds : columns [molecule_id, fold, sim_type]（可选，用于分层）

    返回 dict: mrr / n / recall@{1,5,25} / per_fold / per_sim_type / dup_rate
    """
    truth_map = dict(zip(truth["molecule_id"], truth["smiles"]))
    truth_key_map = {m: key_of(s) for m, s in truth_map.items()}

    pred_keys: list[list[str]] = []
    mol_ids: list[str] = []
    dup_rate_num = 0
    pred_col = "smiles" if "smiles" in sub.columns else "smiles_list"
    for mid, lst in zip(sub["molecule_id"], sub[pred_col]):
        items = lst.split(";") if isinstance(lst, str) else list(lst or [])
        keys, seen = [], set()
        for s in items:
            if not s:
                continue
            kk = key_of(s.strip())
            if kk is None or kk in seen:
                dup_rate_num += 1
                continue
            seen.add(kk)
            keys.append(kk)
        pred_keys.append(keys[:k])
        mol_ids.append(mid)

    missing = [m for m in truth_map if m not in set(mol_ids)]
    if missing:
        log.warning("%d 个分子没有提交，按 0 分处理", len(missing))

    rr = mrr_from_lists(pred_keys, [truth_key_map[m] for m in mol_ids])
    keys_arr = np.array([truth_key_map[m] for m in mol_ids], dtype=object)

    res = {
        "mrr": float(rr.mean()) if len(rr) else 0.0,
        "n": int(len(rr)),
        "recall@1": recall_at(pred_keys, list(keys_arr), 1),
        "recall@5": recall_at(pred_keys, list(keys_arr), 5),
        "recall@25": recall_at(pred_keys, list(keys_arr), 25),
        "mrr_no_none": float(rr[np.array([x is not None for x in keys_arr])].mean())
        if any(x is not None for x in keys_arr)
        else 0.0,
        "dup_or_unparseable": int(dup_rate_num),
    }

    if folds is not None and "sim_type" in folds:
        fm = dict(zip(folds["molecule_id"], folds["sim_type"]))
        types = np.array([fm.get(m, -1) for m in mol_ids])
        res["per_sim_type"] = {}
        for t in sorted(set(types.tolist())):
            m = types == t
            if m.sum():
                res["per_sim_type"][int(t)] = {
                    "mrr": float(rr[m].mean()),
                    "n": int(m.sum()),
                    "recall@25": recall_at(
                        [p for p, keep in zip(pred_keys, m) if keep],
                        [x for x, keep in zip(keys_arr, m) if keep],
                        25,
                    ),
                }

    if folds is not None and "fold" in folds:
        fm = dict(zip(folds["molecule_id"], folds["fold"]))
        fv = np.array([fm.get(m, -1) for m in mol_ids])
        res["per_fold"] = {
            int(f): float(rr[fv == f].mean()) for f in sorted(set(fv.tolist())) if (fv == f).sum()
        }
        if len(res["per_fold"]) > 1:
            res["fold_std"] = float(np.std(list(res["per_fold"].values())))

    return res


# ---------------------------------------------------------------- 折划分

def get_folds(molecules: Iterable[str], scheme: str = "mimic") -> pd.DataFrame:
    """返回 folds.parquet schema: molecule_id, fold, sim_type, excluded_from_index。

    scheme: "random" | "scaffold" | "mimic"
    """
    raise NotImplementedError(
        "A 实现：random=debug；scaffold=Murcko 骨架分折；mimic=按 1/2/3 模拟类别裁剪候选池"
    )


def assert_no_leak(index, holdout_keys: Iterable[str]) -> None:
    """断言 holdout 的结构已从谱库索引中排除，且精确谱匹配返回空。

    失败**必须抛异常**——这是全队最容易犯、最晚发现的错误。
    """
    raise NotImplementedError("A 实现：见 docs/03-验证协议.md §2.3")


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

    # 纯数学部分：不依赖 rdkit
    rr = mrr_from_lists([["A"], ["B", "A"], ["Z"] * 25, ["Z"] * 9 + ["A"]], ["A", "A", "A", "A"])
    check(abs(rr[0] - 1.0) < 1e-9, "rank1 = 1.0")
    check(abs(rr[1] - 0.5) < 1e-9, "rank2 = 0.5")
    check(rr[2] == 0.0, "未命中 = 0")
    check(abs(rr[3] - 0.1) < 1e-9, "rank10 = 0.1")

    # 截断到 k=25：第 26 位不算命中
    check(mrr_from_lists([["Z"] * 25 + ["A"]], ["A"])[0] == 0.0, "rank26 不计分")

    # 只取第一个命中
    check(mrr_from_lists([["A", "A"]], ["A"])[0] == 1.0, "只取第一个命中")

    check(abs(recall_at([["B", "A"], ["C"]], ["A", "A"], 5) - 0.5) < 1e-9, "recall@5")

    # 端到端（用 identity key，绕过 rdkit）
    # m1: 正确答案排第 1 -> 1.0 ; m2: 排第 2 -> 0.5 ; m3: 未命中 -> 0.0
    sub = pd.DataFrame(
        {
            "molecule_id": ["m1", "m2", "m3"],
            "smiles_list": ["K1;K2", "K9;K2", "K7;K8"],
        }
    )
    truth = pd.DataFrame({"molecule_id": ["m1", "m2", "m3"], "smiles": ["K1", "K2", "K3"]})
    folds = pd.DataFrame(
        {"molecule_id": ["m1", "m2", "m3"], "fold": [0, 0, 1], "sim_type": [1, 2, 3]}
    )
    r = local_eval(sub, truth, folds, key_of=lambda s: s)
    check(abs(r["mrr"] - (1.0 + 0.5 + 0.0) / 3) < 1e-9, f"end-to-end mrr={r['mrr']:.4f}")
    check(abs(r["recall@1"] - 1 / 3) < 1e-9, f"recall@1={r['recall@1']:.4f}")
    check(abs(r["recall@5"] - 2 / 3) < 1e-9, f"recall@5={r['recall@5']:.4f}")
    check(r["per_sim_type"][3]["mrr"] == 0.0, "per_sim_type split")
    check(abs(r["per_sim_type"][2]["mrr"] - 0.5) < 1e-9, "per_sim_type class2 = 0.5")

    # 去重：同一行内重复 key 只算首次
    sub2 = pd.DataFrame({"molecule_id": ["m1"], "smiles_list": ["K5;K5;K1"]})
    r2 = local_eval(sub2, truth.iloc[[0]], None, key_of=lambda s: s)
    check(abs(r2["mrr"] - 1 / 2) < 1e-9, "行内去重后 rank 从 1 变 2")

    print("\nselftest:", "PASS" if ok else "FAIL")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--sub", help="submission csv")
    ap.add_argument("--truth", help="truth csv (molecule_id, smiles)")
    ap.add_argument("--folds", default=None)
    a = ap.parse_args()
    if a.selftest:
        raise SystemExit(0 if _selftest() else 1)
    if not (a.sub and a.truth):
        ap.error("需要 --sub 和 --truth，或用 --selftest")
    sub = pd.read_csv(a.sub)
    truth = pd.read_csv(a.truth)
    folds = pd.read_parquet(a.folds) if a.folds else None
    print(local_eval(sub, truth, folds))


if __name__ == "__main__":
    main()
