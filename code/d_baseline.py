"""第一条端到端基线：检索分 + 质量名次 -> 排序 -> 提交 -> 评估。

**为什么这个基线有意义**
真实测试集里 class 3 大约占一半，那些分子在库里找不到。所以"候选召回"是硬上限，
而"排序"决定能不能拿分。这个基线只做两件事：
    1. 用检索通道的 cosine 排
    2. 同分时用质量偏差排

结果是后续所有改动的对照点（B2_prev）。

**评估口径**：直接用 cand_key（InChIKey14）比对，不需要 rdkit。
`a_eval.local_eval` 支持传 `key_of`，这里传恒等函数。
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import a_eval
import a_submit
import common as C


def score_baseline(cand: pd.DataFrame, feat: pd.DataFrame,
                   w_cos: float = 1.0, w_nl: float = 0.5,
                   w_mass: float = 0.1, mass_scale: float = 2.0) -> pd.DataFrame:
    """线性组合基线。

    权重来自 2026-11-20 的消融实测（`experiments.md`）：

        cos only          0.8603
        cos + nl 0.3      0.8691
        cos + nl 0.5      0.8715   <- 最优
        质量项任何形式     +0.0012 以内（噪声级）

    **质量项要用连续衰减，不要用「分子内 rank 均匀铺开」**：
    候选池每分子中位 79 个，rank/n 会把 0–1 摊得极开，反而放大噪声
    （实测 rank 形式 w=0.2 时 MRR 从 0.8603 掉到 0.8337）。
    """
    df = cand[["molecule_id", "cand_key", "smiles", "ppm", "rank_mass"]].merge(
        feat, on=["molecule_id", "cand_key"], how="left")

    df["f_cos"] = df["f_cos"].fillna(0.0)
    df["f_nl"] = df["f_nl"].fillna(0.0)
    # 质量证据：指数衰减。只是轻微打破平局，不该主导排序。
    df["f_mass"] = np.exp(-df["ppm"].abs() / mass_scale)
    df["f_rank"] = 1.0 - (df["rank_mass"] - 1) / df.groupby("molecule_id")[
        "rank_mass"].transform("max").clip(lower=1)

    df["score"] = (w_cos * df["f_cos"] + w_nl * df["f_nl"] + w_mass * df["f_mass"])
    return df


def evaluate(df: pd.DataFrame, truth: pd.DataFrame, tag: str = "") -> dict:
    """评估。

    注意：`assemble_submission` 输出的是真实 SMILES，而评测 key 是 InChIKey14。
    这里把 smiles 换成本行对应的 cand_key 再交给 local_eval（key_of=恒等），
    等价于「用 metric key 判等」，且不需要 rdkit。
    """
    ev = df.copy()
    ev["smiles"] = ev["cand_key"]
    sub = a_submit.assemble_submission(ev, topk=C.TOPK)
    errs = a_submit.validate_sub(sub)

    tr = truth[["molecule_id", "cand_key"]].rename(columns={"cand_key": "smiles"})
    res = a_eval.local_eval(sub, tr, None, key_of=lambda s: s)
    return {"tag": tag, **res, "n_errors": len(errs)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cand", default=os.path.join(C.CAND_DIR, "candidates.parquet"))
    ap.add_argument("--feat", default=os.path.join(C.FEAT_DIR, "feat_search.parquet"))
    ap.add_argument("--w-cos", type=float, default=1.0)
    ap.add_argument("--w-nl", type=float, default=0.3)
    ap.add_argument("--w-mass", type=float, default=0.2)
    ap.add_argument("--out", default=os.path.join(C.OUT_DIR, "baseline_submission.csv"))
    a = ap.parse_args()

    cand = pd.read_parquet(a.cand)
    feat = pd.read_parquet(a.feat)
    truth = pd.read_parquet(os.path.join(C.INTERIM_DIR, "dev_labels.parquet"))

    # ---------------- 对照 1：随机（按质量偏差）
    rnd = cand.copy()
    rnd["score"] = -rnd["ppm"].abs()
    r1 = evaluate(rnd, truth, "B0_mass_only")

    # ---------------- 对照 2：只按候选顺序（无信息）
    rnd2 = cand.copy()
    rnd2["score"] = -rnd2["rank_mass"]
    r2 = evaluate(rnd2, truth, "B0_mass_rank")

    # ---------------- 主基线
    df = score_baseline(cand, feat, a.w_cos, a.w_nl, a.w_mass)
    r3 = evaluate(df, truth, "B1_search+mass")

    rows = []
    for r in (r1, r2, r3):
        rows.append({
            "tag": r["tag"], "MRR@25": round(r["mrr"], 4),
            "R@1": round(r["recall@1"], 4), "R@5": round(r["recall@5"], 4),
            "R@25": round(r["recall@25"], 4), "n": r["n"],
            "errors": r["n_errors"],
        })
    print("\n" + "=" * 72)
    print(pd.DataFrame(rows).to_string(index=False))
    print("=" * 72)
    print("\n注：R@25 = '答案在提交列表里'的比例。本开发集真值已被排除，")
    print("    所以这里测的是**排序能力**，不是召回能力。")

    sub = a_submit.assemble_submission(df, topk=C.TOPK)
    a_submit.save(sub, a.out)
    print(f"\n提交样例前 3 行：")
    print(sub.head(3).to_string(index=False, max_colwidth=70))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
