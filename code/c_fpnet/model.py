"""C 负责：指纹模型（谱 -> 2048 位指纹）。

docs/roles/C-指纹模型.md

GPU 预算纪律：Kaggle 每周 30h，W1 只能花 ≤6h。
不要一上来跑 5 折 × 多 seed。
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

import common as C

log = logging.getLogger("casmi.fpnet")

N_BITS = 2048


def build_fp_targets(smiles: str) -> Optional[np.ndarray]:
    """SMILES -> 2048 位 Morgan 指纹（训练目标）。"""
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=N_BITS)
    return np.array(fp, dtype=np.float32)


def fp_logits(spectra: pd.DataFrame, ckpt: str, batch_size: int = 512) -> np.ndarray:
    """推理：返回 (n_spectra, 2048) 概率。

    输入预处理（与训练必须完全一致）：
      * 峰数截断到 C.MAX_PEAKS，按强度取 top-k 后按 m/z 升序
      * 强度 sqrt 变换 + 归一化到 1
      * 标量特征：precursor_mz、adduct(embedding)、polarity、collision_energy
    """
    raise NotImplementedError("C 实现")


def aggregate_logits(logits: np.ndarray, spec_ids, method: str = "mean") -> dict:
    """多谱聚合（中位 3 张谱，这是隐藏增益点）。

    method: mean | ce_weighted | attention
    推荐先做 mean 拿基线，再做候选层聚合（先每谱算候选分，再聚合）。
    """
    raise NotImplementedError("C 实现")


def fp_scores(cand_tbl: pd.DataFrame, logits_by_mol: dict, mol_fps: dict) -> pd.DataFrame:
    """候选打分：logits (n,2048) · 候选指纹 (n,2048)。

    返回 [molecule_id, cand_key, f_fp_dot, f_fp_cos, f_fp_nbits_hit]
    """
    raise NotImplementedError("C 实现")


def loss_weight_per_molecule(n_spectra: np.ndarray) -> np.ndarray:
    """按 1/n_spectra 加权，否则 20 张谱的分子权重是 2 张的 10 倍。"""
    return 1.0 / np.maximum(n_spectra, 1)


if __name__ == "__main__":
    import sys

    sys.path.insert(0, C.CODE_DIR)
