"""化学基础工具：分子式解析、单同位素精确质量、中性质量推算。

**故意不依赖 RDKit**——本机还没装。分子式到质量这一步纯算术，
用 train 自带的 `molecular_formula` 列就够，不必为它引入化学库。

RDKit 真正不可替代的地方只有两处：
    1. metric_key（InChIKey14，评测口径）
    2. 结构编辑/指纹（B、C 的工作）
见 docs/01 §2。
"""

from __future__ import annotations

import re
from typing import Optional

# 单同位素质量（monoisotopic）。NIST 常用值。
MONO: dict[str, float] = {
    "H": 1.00782503207, "D": 2.0141017778, "C": 12.0, "N": 14.0030740048,
    "O": 15.9949146196, "S": 31.97207100, "P": 30.97376163,
    "F": 18.99840322, "Cl": 34.96885268, "Br": 78.9183371, "I": 126.904473,
    "B": 11.0093054, "Si": 27.9769265325, "Se": 79.9165213,
    "Na": 22.9897692809, "K": 38.96370668, "Li": 7.01600455,
    "Fe": 55.9349375, "Zn": 63.9291422, "Mg": 23.9850417, "Ca": 39.9625909,
    "Cu": 62.9295975, "Mn": 54.9380451, "Co": 58.9331950, "Ni": 57.9353429,
    "As": 74.9215965, "Al": 26.98153863, "Sn": 119.9021966, "Sb": 120.9038180,
}

_TOKEN = re.compile(r"([A-Z][a-z]?)(\d*)")


def parse_formula(formula: str) -> Optional[dict[str, int]]:
    """解析 Hill 式分子式。无法解析返回 None。

    支持 ``C18H24N4O2``、``C6H12O6``、``Cl2`` 等。
    不处理电荷、同位素标记、括号（train 的公式都是扁平的）。
    """
    if not isinstance(formula, str) or not formula:
        return None
    out: dict[str, int] = {}
    pos = 0
    for m in _TOKEN.finditer(formula):
        if m.start() != pos:          # 中间有无法识别的内容
            return None
        el, cnt = m.group(1), m.group(2)
        if el not in MONO:
            return None
        out[el] = out.get(el, 0) + (int(cnt) if cnt else 1)
        pos = m.end()
    if pos != len(formula) or not out:
        return None
    return out


def formula_mass(formula: str) -> Optional[float]:
    """分子式 -> 单同位素中性质量。"""
    d = parse_formula(formula)
    if d is None:
        return None
    return sum(MONO[el] * n for el, n in d.items())


def n_heavy(formula: str) -> int:
    d = parse_formula(formula)
    return 0 if d is None else sum(n for el, n in d.items() if el != "H")


def mass_vector(formulas) -> "list[float | None]":
    """批量版，避免反复解析同一公式。"""
    cache: dict[str, Optional[float]] = {}
    out = []
    for f in formulas:
        if f not in cache:
            cache[f] = formula_mass(f)
        out.append(cache[f])
    return out


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

    d = parse_formula("C18H24N4O2")
    check(d == {"C": 18, "H": 24, "N": 4, "O": 2}, f"parse C18H24N4O2 -> {d}")
    d2 = parse_formula("C6H12O6")
    check(d2 == {"C": 6, "H": 12, "O": 6}, "parse C6H12O6")
    d3 = parse_formula("CCl4")
    check(d3 == {"C": 1, "Cl": 4}, f"parse CCl4 -> {d3}")
    check(parse_formula("nonsense") is None, "reject nonsense")
    check(parse_formula("C18Xx4") is None, "reject unknown element")

    # 葡萄糖 180.06339
    m = formula_mass("C6H12O6")
    check(m is not None and abs(m - 180.06339) < 1e-4, f"glucose mass -> {m:.5f}")
    # 苯丙氨酸 C9H11NO2 = 165.07898
    m2 = formula_mass("C9H11NO2")
    check(m2 is not None and abs(m2 - 165.07898) < 1e-4, f"Phe mass -> {m2:.5f}")
    check(n_heavy("C18H24N4O2") == 24, "n_heavy excludes H")

    # 与 common 的加合物表联动：C6H12O6 + [M+H]+ = 180.06339 + 1.00728 = 181.07066
    # 注意是加**质子**质量（1.007276），不是氢原子质量（1.007825）——
    # 质谱测的是离子 m/z，电子质量已经扣掉。
    import common as C
    mz = m + C.ADDUCT_DELTAS["[M+H]+"]
    check(abs(mz - 181.07066) < 1e-4, f"[M+H]+ m/z -> {mz:.5f}")

    print("\nselftest:", "PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    import sys

    if "--selftest" in sys.argv:
        raise SystemExit(0 if _selftest() else 1)
    print("用法: python code/chem.py --selftest")
