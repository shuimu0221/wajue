"""B 负责：class 3 骨架编辑生成。

W3 才开工。前提：W2 末确认 `cand_in_pool` 是主要瓶颈。
如果候选召回已经够高，**就不要做这条**，把时间投到 D 的融合上。
"""

from __future__ import annotations

import logging

import pandas as pd

log = logging.getLogger("casmi.gen")

# 小改操作集。每个操作接收 RDKit Mol，返回 list[Mol]（可能为空）。
DEFAULT_OPS = [
    "add_CH2",          # 同系物
    "sub_CH2",
    "add_O",            # 羟基化 / 醚化
    "remove_O",
    "move_OH",          # 羟基位置移动
    "add_glycosyl",     # 加糖基（天然产物常见）
    "remove_glycosyl",
    "methylate",        # -OH -> -OCH3
    "demethylate",
    "shift_double_bond",
]


def generate(cand_tbl: pd.DataFrame, mz_neutral: dict, max_per_mol: int = 200, ops=None):
    """以库命中为骨架生成变体。

    实现顺序建议：
      1. 取每个分子 top-1 检索命中作为骨架（可选 top-3）
      2. 逐个 op 生成，立即做分子式/ppm 硬过滤
      3. 用 metric_key 去重（含与已有候选去重）
      4. 用 D 的 ranker 打分，**只往尾部插**

    返回 candidates 的 schema，source="generated"。
    """
    raise NotImplementedError("B 实现：见 docs/roles/B-检索通道.md W3")


def validate_generated(df: pd.DataFrame, mz_neutral: dict, existing_keys: set) -> pd.DataFrame:
    """生成结果的硬过滤流水线。任何一条不过就丢弃（错猜白占位置）。"""
    raise NotImplementedError
