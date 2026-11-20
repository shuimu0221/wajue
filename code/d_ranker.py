"""D 负责：LightGBM ranker（W2 · P1）。

为什么现在做这个
----------------
W1 失败分析（docs/04）的结论：MRR 缺口 0.1270 **全部来自排序**，
其中 33% 的错排分子「首位分 − 答案分 < 0.05」，是**平局问题**。
线性加权分不开平局，ranker 可以（它能用候选数、通道一致性等交互特征）。

**不需要 rdkit**：标签直接用 train 自带的 `inchikey14`，特征全在
`candidates_with_truth.parquet` + `feat_search_with_truth.parquet` 里。

评估协议
--------
* 按 molecule_id 做 5 折 GroupKFold（**分子级**，绝不按行随机）
* 输出 OOF 的 MRR@25，与线性基线 0.8727 对比
* 同时报告 R@1（这才是 ranker 该改善的指标）

⚠️ 只有 400 个分子，全是 class 1。**这个数字不代表榜分**，
只说明「在排序层面还有多少空间」。
"""

from __future__ import annotations

import argparse
import os
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

KEY = ["molecule_id", "cand_key"]


# ---------------------------------------------------------------- 特征

def build_features(cand: pd.DataFrame, feat: pd.DataFrame,
                   truth: pd.DataFrame) -> pd.DataFrame:
    """构造候选级特征。**不得使用任何依赖真值的特征**（会泄漏）。"""
    df = cand[KEY + ["smiles", "molecular_formula", "ppm", "rank_mass"]].merge(
        feat, on=KEY, how="left")

    df["f_cos"] = df["f_cos"].fillna(0.0)
    df["f_nl"] = df["f_nl"].fillna(0.0)
    df["f_cos_self"] = df["f_cos_self"].fillna(0.0)
    df["ppm_abs"] = df["ppm"].abs()
    df["had_lib_hit"] = df["f_lib_hit"].fillna(0).astype(np.int8)

    g = df.groupby("molecule_id")
    # 分子内上下文：候选池有多大、这个候选处在什么位置、通道是否一致
    df["n_cand"] = g["cand_key"].transform("size")
    df["cos_rank"] = g["f_cos"].rank(ascending=False, method="min")
    df["cos_gap_top"] = g["f_cos"].transform("max") - df["f_cos"]
    df["cos_z"] = (df["f_cos"] - g["f_cos"].transform("mean")) / (
        g["f_cos"].transform("std").replace(0, np.nan))
    df["nl_rank"] = g["f_nl"].rank(ascending=False, method="min")
    # 通道一致性：cos 和 nl 同时排前面 = 强信号
    df["rank_agree"] = -(df["cos_rank"] - df["nl_rank"]).abs()
    # 同分子式的候选有几个（同式异构体是主要竞争者）
    df["n_same_formula"] = g["molecular_formula"].transform(
        lambda s: s.map(s.value_counts()))
    # 该分子式在整个候选池里的稀有度
    df["formula_freq_pool"] = df["molecular_formula"].map(
        df["molecular_formula"].value_counts())

    df["cos_x_nl"] = df["f_cos"] * df["f_nl"]
    df["cos_minus_nl"] = df["f_cos"] - df["f_nl"]

    return df


FEATURES = [
    # ⚠️ 绝对不要放 f_cos_self：
    # 它表示「该候选的参考谱 == 查询谱本身」。真实测试集的谱不在库里，
    # 这个特征恒为 0；放进来看似能把 MRR 提到 1.0000，那是纯粹的自泄漏
    # （实测重要度 94.9%，其它特征全被压成 0）。
    "f_cos", "f_nl", "ppm_abs", "rank_mass", "had_lib_hit",
    "n_cand", "cos_rank", "cos_gap_top", "cos_z", "nl_rank", "rank_agree",
    "n_same_formula", "formula_freq_pool", "cos_x_nl", "cos_minus_nl",
]

# 明确列出禁止入模的列，供断言用
FORBIDDEN = ["f_cos_self", "label"]


def label_of(df: pd.DataFrame, truth: pd.DataFrame) -> np.ndarray:
    tmap = dict(zip(truth["molecule_id"], truth["cand_key"]))
    return (df[KEY].apply(lambda r: tmap.get(r["molecule_id"]) == r["cand_key"],
                          axis=1).to_numpy().astype(np.int8))


# ---------------------------------------------------------------- 评估

def mrr_at(df: pd.DataFrame, score: np.ndarray, truth: pd.DataFrame,
           k: int = 25) -> dict:
    d = df[KEY].copy()
    d["score"] = score
    tmap = dict(zip(truth["molecule_id"], truth["cand_key"]))
    rr, r1, r5 = [], 0, 0
    for mol, g in d.groupby("molecule_id", sort=False):
        top = g.nlargest(k, "score")["cand_key"].to_numpy()
        t = tmap.get(mol)
        hit = np.flatnonzero(top == t)
        if len(hit):
            r = 1.0 / (hit[0] + 1)
            rr.append(r)
            r1 += hit[0] == 0
            r5 += hit[0] < 5
        else:
            rr.append(0.0)
    n = len(tmap)
    return {"mrr": float(np.mean(rr)), "r1": r1 / n, "r5": r5 / n, "n": n}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cand", default=os.path.join(C.CAND_DIR,
                                                   "candidates_with_truth.parquet"))
    ap.add_argument("--feat", default=os.path.join(C.FEAT_DIR,
                                                   "feat_search_with_truth.parquet"))
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=C.SEED)
    a = ap.parse_args()

    cand = pd.read_parquet(a.cand)
    feat = pd.read_parquet(a.feat)
    truth = pd.read_parquet(os.path.join(C.INTERIM_DIR, "dev_labels.parquet"))

    df = build_features(cand, feat, truth)
    df["label"] = label_of(df, truth)
    print(f"[data] {len(df):,} 候选 / {df.molecule_id.nunique()} 分子, "
          f"正样本 {int(df.label.sum())}")

    # 线性基线（同一份特征，做对照）
    lin = df["f_cos"] + 0.5 * df["f_nl"] + 0.1 * np.exp(-df["ppm_abs"] / 2.0)
    base = mrr_at(df, lin.to_numpy(), truth)
    print(f"[baseline 线性] MRR {base['mrr']:.4f}  R@1 {base['r1']:.3f}  "
          f"R@5 {base['r5']:.3f}")

    # GroupKFold OOF
    groups = df["molecule_id"].to_numpy()
    oof = np.zeros(len(df))
    gkf = GroupKFold(n_splits=a.folds)
    params = dict(
        objective="lambdarank", metric="ndcg", ndcg_eval_at=[1, 5],
        learning_rate=0.05, num_leaves=15, min_data_in_leaf=30,
        feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
        lambda_l2=1.0, verbose=-1, seed=a.seed,
    )

    imps = []
    for f, (tr_i, va_i) in enumerate(gkf.split(df, groups=groups), 1):
        tr = df.iloc[tr_i]
        # lambdarank 要求按 group 连续排列，并给出 group sizes
        tr = tr.sort_values("molecule_id", kind="mergesort")
        gs = tr.groupby("molecule_id", sort=False).size().to_numpy()
        ds = lgb.Dataset(tr[FEATURES], label=tr["label"], group=gs,
                         free_raw_data=False)
        booster = lgb.train(params, ds, num_boost_round=200)
        oof[va_i] = booster.predict(df.iloc[va_i][FEATURES])
        imps.append(pd.Series(booster.feature_importance("gain"), index=FEATURES))
        print(f"  fold {f}: trained on {len(tr):,} rows / {len(gs)} molecules")

    res = mrr_at(df, oof, truth)
    print("\n" + "=" * 60)
    print(f"{'':<16}{'MRR@25':>10}{'R@1':>10}{'R@5':>10}")
    print(f"{'线性基线':<16}{base['mrr']:>10.4f}{base['r1']:>10.3f}{base['r5']:>10.3f}")
    print(f"{'LightGBM OOF':<16}{res['mrr']:>10.4f}{res['r1']:>10.3f}{res['r5']:>10.3f}")
    d = res["mrr"] - base["mrr"]
    print(f"{'Δ':<16}{d:>+10.4f}{res['r1']-base['r1']:>+10.3f}"
          f"{res['r5']-base['r5']:>+10.3f}")
    print("=" * 60)
    print("噪声阈值：±0.02（400 分子，标准误 ≈ 0.5/√400）")
    print("结论：" + ("ranker 有效" if d > 0.02 else
                   "**未超过噪声，不要上线**"))

    print("\n=== 特征重要度（gain，5 折平均）===")
    imp = pd.concat(imps, axis=1).mean(axis=1).sort_values(ascending=False)
    print((imp / imp.sum() * 100).round(1).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
