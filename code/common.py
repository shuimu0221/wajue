"""公共配置与常量。全队共用，改动需通知。

契约见 docs/02-接口契约.md。不改签名、不加全局可变状态。
"""

from __future__ import annotations

import logging
import os
import re
from typing import Optional

LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
log = logging.getLogger("casmi")

# ---------------- 路径 ----------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
INTERIM_DIR = os.path.join(DATA_DIR, "interim")
CAND_DIR = os.path.join(DATA_DIR, "candidates")
FEAT_DIR = os.path.join(DATA_DIR, "features")
SPLIT_DIR = os.path.join(DATA_DIR, "splits")
OUT_DIR = os.path.join(ROOT, "output")
CODE_DIR = os.path.join(ROOT, "code")

for _d in (RAW_DIR, INTERIM_DIR, CAND_DIR, FEAT_DIR, SPLIT_DIR, OUT_DIR):
    os.makedirs(_d, exist_ok=True)

# 数据实际落点可能不在 data/raw（例如手动下载到 k_data/）。
# 按顺序找第一个存在的，避免把大文件搬来搬去。
_DATA_SEARCH = [RAW_DIR, os.path.join(ROOT, "k_data")]


def find_data_file(name: str, required: bool = True) -> str | None:
    """在候选目录里找数据文件，返回第一个存在的绝对路径。"""
    for d in _DATA_SEARCH:
        p = os.path.join(d, name)
        if os.path.exists(p):
            return p
    if required:
        raise FileNotFoundError(
            f"找不到 {name}，已查找: {_DATA_SEARCH}\n"
            f"把官方数据放到 {RAW_DIR} 或 {os.path.join(ROOT, 'k_data')}"
        )
    return None

SEED = 2026
N_FOLDS = 5
TOPK = 25
PPM_DEFAULT = 10.0
MAX_CAND_PER_MOL = 2000
MAX_PEAKS = 128

# 判分所用 RDKit 版本（docs/00 §3.1）。本地必须一致，否则 metric key 会漂移。
RDKIT_METRIC_VERSION = "2026.03.3"


# ---------------- 加合物表 ----------------
# 键是规整后的字符串，值是「加合离子相对中性分子的质量偏移」，单位 Da（已除以电荷数）。
# 二聚体（2M 开头）另需把中性质量折半，见 is_dimer / neutral_mass_from_precursor。
PROTON = 1.007276466
ELECTRON = 0.00054858

H2O = 18.010565
FORMIC = 46.00548  # CH2O2

# 测试集**只**用这 10 种加合物（data-description 原文，已核对）。
# 训练集还有更多（二聚体、乙腈加合等），建库时要能处理，但候选池打分只关心这 10 种。
TEST_ADDUCTS: frozenset[str] = frozenset({
    "[M+H]+", "[M+NH4]+", "[M-H2O+H]+", "[M-2H2O+H]+", "[M+Na]+",
    "[M+K]+", "[M-H]-", "[M-H2O-H]-", "[M+CH2O2-H]-", "[M+Cl]-",
})

ADDUCT_DELTAS: dict[str, float] = {
    # ---- 测试集 10 种 ----
    "[M+H]+": PROTON,
    "[M+NH4]+": 18.033823,
    "[M-H2O+H]+": PROTON - H2O,            # -17.00329
    "[M-2H2O+H]+": PROTON - 2 * H2O,       # -35.01385
    "[M+Na]+": 22.989218,
    "[M+K]+": 38.963158,
    "[M-H]-": -PROTON,
    "[M-H2O-H]-": -PROTON - H2O,           # -19.01784
    "[M+CH2O2-H]-": FORMIC - PROTON,       # 44.99820（甲酸加合）
    "[M+Cl]-": 34.969402,
    # ---- 测试集之外的常见写法（训练集里会出现）----
    "[M+H-H2O]+": PROTON - H2O,            # 同上，另一种写法
    "[M+H-2H2O]+": PROTON - 2 * H2O,
    "[M-H-H2O]-": -PROTON - H2O,
    "[M+HCOO]-": FORMIC - PROTON,          # 甲酸的旧名
    "[M+CH3COO]-": 59.013851,
    "[M+ACN+H]+": PROTON + 41.026549,
    "[M+2H]2+": PROTON,
    "[M+2Na]2+": 22.989218,
    "[M-2H]2-": -PROTON,
    # 二聚体（质量需折半，用 is_dimer 判断；**测试集不含**）
    "[2M+H]+": PROTON,
    "[2M+Na]+": 22.989218,
    "[2M-H]-": -PROTON,
    "[2M+HCOO]-": FORMIC - PROTON,
}

_CHARGE_RE = re.compile(r"\](\d*)([+-])$")


def charge_of(adduct: str) -> int:
    """返回电荷数的**绝对值符号化**结果：'[M+2H]2+' -> +2, '[M-H]-' -> -1。"""
    m = _CHARGE_RE.search(adduct or "")
    if not m:
        return 1
    n = int(m.group(1)) if m.group(1) else 1
    return n if m.group(2) == "+" else -n


def is_dimer(adduct: str) -> bool:
    return (adduct or "").lstrip("[").startswith("2M")


def normalize_adduct(adduct: str) -> str:
    """规整字符串：去空格、统一括号。未知加合物原样返回（并告警）。"""
    if not isinstance(adduct, str):
        return ""
    a = adduct.strip().replace(" ", "")
    a = a.replace("〔", "[").replace("〕", "]")
    if a and a not in ADDUCT_DELTAS and a != "":
        log.debug("unknown adduct: %s", a)
    return a


def neutral_mass_from_precursor(precursor_mz: float, adduct: str) -> Optional[float]:
    """由加合物 m/z 推中性精确质量。未知加合物返回 None（调用方应回退到质量窗口放宽）。

    m/z = (M * n_dimer + delta) / |charge|  =>  M = (mz*|z| - delta) / n_dimer
    """
    a = normalize_adduct(adduct)
    if a not in ADDUCT_DELTAS:
        return None
    z = abs(charge_of(a))
    n = 2 if is_dimer(a) else 1
    if z == 0 or precursor_mz is None:
        return None
    return (float(precursor_mz) * z - ADDUCT_DELTAS[a]) / n


# ---------------- 碰撞能脏值 ----------------
_CE_RE = re.compile(r"[-+]?\d*\.?\d+")


def parse_ce(value) -> Optional[float]:
    """解析碰撞能。数据里混着 60 / 60.0 / [60.0] / '60.0 eV' 等写法。

    取第一个数值；解析不出返回 None（不要填 0，0 和缺失语义不同）。
    """
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return None if value != value else float(value)  # NaN check
    s = str(value).strip()
    if not s or s.lower() in {"nan", "none", "na", "n/a"}:
        return None
    m = _CE_RE.search(s)
    if not m:
        return None
    try:
        return float(m.group())
    except ValueError:
        return None


# ---------------- 峰数据 ----------------
def normalize_intensity(intensity) -> "list[float]":
    """归一化到最大峰 = 1。空则返回空列表。"""
    if intensity is None or len(intensity) == 0:
        return []
    mx = max(intensity)
    if mx <= 0:
        return [0.0] * len(intensity)
    return [float(x) / mx for x in intensity]


def sqrt_intensity(intensity) -> "list[float]":
    return [float(x) ** 0.5 for x in intensity]


def dedupe_peaks(mz, intensity, tol: float = 1e-4):
    """同一 m/z（容差内）的峰强度相加，返回按 m/z 升序的结果。"""
    pairs = sorted(zip(mz, intensity))
    out_mz, out_int = [], []
    for m, i in pairs:
        if out_mz and abs(m - out_mz[-1]) <= tol:
            out_int[-1] += i
        else:
            out_mz.append(float(m))
            out_int.append(float(i))
    return out_mz, out_int


def limit_peaks(mz, intensity, k: int = MAX_PEAKS):
    """保留强度最高的 k 个峰，再按 m/z 升序返回。"""
    if len(mz) <= k:
        return list(mz), list(intensity)
    idx = sorted(range(len(intensity)), key=lambda i: intensity[i], reverse=True)[:k]
    idx.sort()
    return [float(mz[i]) for i in idx], [float(intensity[i]) for i in idx]


# ---------------- 质量工具 ----------------
def ppm_error(theoretical: float, observed: float) -> float:
    if theoretical == 0:
        return float("inf")
    return (observed - theoretical) / theoretical * 1e6


def within_ppm(a: float, b: float, ppm: float = PPM_DEFAULT) -> bool:
    if a is None or b is None:
        return False
    return abs(a - b) <= abs(b) * ppm * 1e-6 + 1e-6


# ---------------- 自检 ----------------
def _selfcheck():
    ok = True

    def check(cond, msg):
        nonlocal ok
        if not cond:
            ok = False
            log.error("FAIL %s", msg)
        else:
            log.info("ok   %s", msg)

    check(parse_ce(60) == 60.0, "parse_ce(60)")
    check(parse_ce("60.0") == 60.0, "parse_ce('60.0')")
    check(parse_ce("[60.0]") == 60.0, "parse_ce('[60.0]')")
    check(parse_ce("35.0 eV") == 35.0, "parse_ce('35.0 eV')")
    check(parse_ce(None) is None, "parse_ce(None)")
    check(parse_ce(float("nan")) is None, "parse_ce(nan)")

    # 全部 10 种测试加合物都还原到同一中性质量 180.0639（葡萄糖 C6H12O6）
    M = 180.0639
    cases = {
        "[M+H]+": M + PROTON,
        "[M+NH4]+": M + 18.033823,
        "[M-H2O+H]+": M + PROTON - H2O,
        "[M-2H2O+H]+": M + PROTON - 2 * H2O,
        "[M+Na]+": M + 22.989218,
        "[M+K]+": M + 38.963158,
        "[M-H]-": M - PROTON,
        "[M-H2O-H]-": M - PROTON - H2O,
        "[M+CH2O2-H]-": M + FORMIC - PROTON,
        "[M+Cl]-": M + 34.969402,
    }
    for a, mz in cases.items():
        got = neutral_mass_from_precursor(mz, a)
        check(got is not None and abs(got - M) < 0.01, f"neutral_mass {a} -> {got}")
    check(TEST_ADDUCTS == frozenset(cases), "TEST_ADDUCTS 与自检用例一致")
    # 二聚体：[2M+H]+ 的 m/z ≈ 2M + 1.0073
    m4 = neutral_mass_from_precursor(2 * M + PROTON, "[2M+H]+")
    check(m4 is not None and abs(m4 - M) < 0.01, f"neutral_mass dimer -> {m4}")
    check(charge_of("[M+2H]2+") == 2 and charge_of("[M-H]-") == -1, "charge_of")

    check(is_dimer("[2M+H]+") and not is_dimer("[M+H]+"), "is_dimer")
    check(normalize_intensity([2, 4, 8]) == [0.25, 0.5, 1.0], "normalize_intensity")
    check(within_ppm(180.0639, 180.0640, 10), "within_ppm")
    check(not within_ppm(180.0639, 180.10, 10), "within_ppm negative")

    try:
        import rdkit  # noqa: F401
        log.info("ok   rdkit %s (metric target %s)", rdkit.__version__, RDKIT_METRIC_VERSION)
    except ImportError:
        log.warning("rdkit NOT installed — metric key 无法计算，先跑 pip install rdkit==2026.3.3")

    log.info("selfcheck: %s", "PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if _selfcheck() else 1)
