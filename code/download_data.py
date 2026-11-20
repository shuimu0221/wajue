"""下载 CASMI 2026 比赛数据。

**不依赖 `kaggle` 包**——直接用 httpx 打 Kaggle API v1（HTTP Basic 认证）。
这样本机不需要 pip install，只需要一个凭证文件。

前置条件（两件都必须做，否则 403）
--------------------------------
1. 在浏览器里**接受比赛规则**（只需一次，绑定账号）：
   https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/rules
2. 拿到 API token：
   https://www.kaggle.com/settings -> API -> "Create New Token"
   会下载 kaggle.json，放到 `~/.kaggle/kaggle.json`
   （或设环境变量 KAGGLE_USERNAME / KAGGLE_KEY）

用法
----
    python code/download_data.py --check       # 只检查凭证与权限，不下载
    python code/download_data.py --list        # 列出文件与大小
    python code/download_data.py               # 下载到 data/raw/ 并校验
    python code/download_data.py --force       # 已存在也重新下载

下载后会自动：
    * 解压
    * 用 pyarrow 读 parquet schema，对照官方 12/18 列清单打印差异
    * 提示下一步该跑什么
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import zipfile
from pathlib import Path

import httpx

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

SLUG = "enveda-CASMI26-molecule-id-mass-spectra"
API = "https://www.kaggle.com/api/v1"

# 官方 data-description 页给出的列清单（docs/00 §3），用于下载后核对
EXPECTED_TEST_COLS = [
    "molecule_id", "spectrum_id", "ms2_mzs", "ms2_normalized_intensities",
    "base_peak_intensity", "adduct", "ionization_mode", "instrument_type",
    "precursor_mz", "collision_energy_ev", "collision_energy_orig",
    "collision_energy_orig_units",
]
EXPECTED_TRAIN_EXTRA = [
    "normalized_smiles", "inchikey", "inchikey14", "molecular_formula",
    "ingest_lib", "adduct_orig", "precursor_error_ppm", "num_peaks",
]


def _read_cred_file(p: Path) -> dict | None:
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        raise SystemExit(f"读不了 {p}: {e}")
    return d if isinstance(d, dict) else None


def load_creds() -> tuple[str | None, str]:
    """按以下顺序找凭证，返回 (username | None, token)。

    支持两种形态：
      * 旧式：``{"username": "xxx", "key": "32位hex"}``  -> HTTP Basic (user, key)
      * 新式：``{"key": "KGAT_..."}`` 或 ``{"token": "KGAT_..."}``
        -> 单 token，没有 username，按 Bearer 发送（见 build_client）
    """
    env_key = os.environ.get("KAGGLE_KEY") or os.environ.get("KAGGLE_TOKEN")
    env_user = os.environ.get("KAGGLE_USERNAME")
    if env_key:
        return env_user, env_key

    candidates: list[Path] = [Path.home() / ".kaggle" / "kaggle.json"]
    if os.environ.get("KAGGLE_CONFIG_DIR"):
        candidates.insert(0, Path(os.environ["KAGGLE_CONFIG_DIR"]) / "kaggle.json")
    candidates += [Path.cwd() / "kaggle.json",
                   Path(__file__).resolve().parent.parent / "kaggle.json"]

    for p in candidates:
        if not p.is_file():
            continue
        d = _read_cred_file(p) or {}
        tok = d.get("key") or d.get("token")
        user = d.get("username")
        if tok:
            print(f"[creds] 使用 {p}（{'单 token' if not user else '用户 ' + user}）")
            return user, str(tok).strip()

    raise SystemExit(
        "找不到 Kaggle 凭证。二选一：\n"
        "  A) 浏览器打开 https://www.kaggle.com/settings -> API -> Create New Token，\n"
        "     把下载的 kaggle.json 放到 "
        f"{Path.home() / '.kaggle' / 'kaggle.json'}\n"
        "  B) 设环境变量 KAGGLE_KEY（旧式再配 KAGGLE_USERNAME）\n"
    )


def build_client(user: str | None, token: str) -> httpx.Client:
    """旧式凭证 -> HTTP Basic；新式单 token（KGAT_）-> Bearer。"""
    common = dict(
        timeout=httpx.Timeout(60.0, read=600.0),
        follow_redirects=True,
        headers={"User-Agent": "casmi26-download/1.0"},
    )
    if user:
        return httpx.Client(auth=(user, token), **common)
    return httpx.Client(headers={**common.pop("headers"),
                                 "Authorization": f"Bearer {token}"}, **common)


def explain_status(code: int, body: str) -> str:
    if code == 401:
        return ("401 未认证：凭证错了或失效。去 https://www.kaggle.com/settings 重新生成 token。")
    if code == 403:
        return (
            "403 禁止访问：**多半是还没接受比赛规则**。\n"
            f"  去 {API.replace('/api/v1','')}/competitions/{SLUG}/rules 点 'I Understand and Accept'。\n"
            "  （也可能账号未成年/地区受限，见规则 §1）"
        )
    if code == 404:
        return f"404 找不到：slug 或文件名不对。slug={SLUG}"
    return f"{code}: {body[:400]}"


def list_files(c: httpx.Client) -> list[dict]:
    r = c.get(f"{API}/competitions/data/list/{SLUG}")
    if r.status_code != 200:
        raise SystemExit(explain_status(r.status_code, r.text))
    files = r.json()
    print(f"[list] {len(files)} 个文件：")
    for f in files:
        size = f.get("totalBytes") or 0
        print(f"  {f.get('name','?'):<45} {size/1024**3:8.2f} GB  ({size:,} B)")
    total = sum((f.get("totalBytes") or 0) for f in files)
    print(f"  {'合计':<45} {total/1024**3:8.2f} GB")
    return files


def stream_download(c: httpx.Client, url: str, dest: Path) -> None:
    """流式下载，带进度与断点续传（Kaggle 下载 URL 不支持 Range，这里只做刷新显示）。"""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    t0 = time.time()
    done = 0
    with c.stream("GET", url) as r:
        if r.status_code != 200:
            raise SystemExit(explain_status(r.status_code, r.read().decode("utf-8", "replace")))
        total = int(r.headers.get("Content-Length") or 0)
        with open(tmp, "wb") as fh:
            for chunk in r.iter_bytes(1024 * 256):
                fh.write(chunk)
                done += len(chunk)
                if total:
                    pct = done / total * 100
                    mb = done / 1024**2
                    sp = mb / max(time.time() - t0, 1e-6)
                    print(f"\r  {dest.name}: {pct:5.1f}%  {mb:8.1f} MB  {sp:6.1f} MB/s",
                          end="", flush=True)
    print()
    tmp.replace(dest)


def verify_parquet(path: Path, expected_full: list[str], label: str) -> None:
    """核对列名。官方页面写的列必须全在，多出来的列只提示不报错。"""
    try:
        import pyarrow.parquet as pq
    except ImportError:
        print(f"[verify] pyarrow 未安装，跳过 {path.name} 的 schema 核对")
        return
    pf = pq.ParquetFile(path)
    cols = list(pf.schema_arrow.names)
    missing = [c for c in expected_full if c not in cols]
    extra = [c for c in cols if c not in expected_full]
    print(f"[verify] {label}: {pf.metadata.num_rows:,} 行 × {len(cols)} 列")
    if missing:
        print(f"  !! 官方清单里缺这些列: {missing}")
    else:
        print("  ok 官方列清单全部存在")
    if extra:
        print(f"  (另有未在官方清单中的列: {extra})")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="只检查凭证与权限")
    ap.add_argument("--list", action="store_true", help="列出文件与大小")
    ap.add_argument("--force", action="store_true", help="已存在也重新下载")
    ap.add_argument("--dest", default=C.RAW_DIR, help=f"下载目录（默认 {C.RAW_DIR}）")
    a = ap.parse_args()

    user, key = load_creds()
    dest = Path(a.dest)
    dest.mkdir(parents=True, exist_ok=True)

    with client(user, key) as c:
        files = list_files(c)

        if a.check or a.list:
            print("\n[ok] 凭证有效，能看到文件清单。")
            if a.list:
                return 0

        if a.check:
            return 0

        # 优先整包下载，省去逐个文件
        zip_path = dest / f"{SLUG}.zip"
        all_present = all((dest / f["name"]).exists() for f in files)
        if all_present and not a.force:
            print("[skip] 所有文件已存在。要重新下载加 --force")
        else:
            print(f"\n[download] 整包 -> {zip_path}")
            stream_download(c, f"{API}/competitions/data/download-all/{SLUG}", zip_path)

            print(f"[unzip] {zip_path}")
            with zipfile.ZipFile(zip_path) as z:
                names = z.namelist()
                print(f"  {len(names)} 个条目: {names}")
                z.extractall(dest)
            try:
                zip_path.unlink()
                print("  已删除 zip")
            except OSError:
                pass

    # ---- 校验 ----
    print()
    tr = dest / "train.parquet"
    te = dest / "test.parquet"
    if te.is_file():
        verify_parquet(te, EXPECTED_TEST_COLS, "test.parquet")
    if tr.is_file():
        verify_parquet(tr, EXPECTED_TEST_COLS + EXPECTED_TRAIN_EXTRA, "train.parquet")

    print(
        "\n下一步（**重要**，先确认 test 真的是 train 的子集）：\n"
        "  python code/a_load.py --stats\n"
        "  python code/a_load.py --check-leak\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
