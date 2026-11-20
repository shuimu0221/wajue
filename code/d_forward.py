"""D 负责：前向模型（候选会碎成什么）。W3 开工。

只对 top-K（100 左右）候选算——全量算一定超时。
目标明确：**治同分子式异构体**。评估时专门看"同式错排"那批分子。
"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

import common as C

log = logging.getLogger("casmi.fwd")


def forward_scores(
    cand_tbl: pd.DataFrame,
    spectra: pd.DataFrame,
    top_k: int = 100,
    backend: str = "metfrag",
) -> pd.DataFrame:
    """返回 [molecule_id, cand_key, f_fwd, f_fwd_npeaks]。

    backend: metfrag | cfmid | iceberg | custom
    **能吃现成就不自己训**——W3 时间只够接一个。
    """
    raise NotImplementedError("D 实现")


def same_formula_error_report(pred: pd.DataFrame, truth: pd.DataFrame, cand: pd.DataFrame) -> dict:
    """同式错排专项评估：
    正确答案在候选池里、但排在第 1 位的候选与它 heavy_formula 相同 → 典型错排。
    前向模型是否把这类分子的 rank 提前了？
    """
    raise NotImplementedError("D 实现")


if __name__ == "__main__":
    print("backend options: metfrag | cfmid | iceberg | custom")
