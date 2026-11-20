"""下载 Kaggle Dataset 到 k_data/external/。

用法:
    python code/dl_dataset.py pri2si17/coconut-natural-products-2026-09 coconut
    python code/dl_dataset.py shankhasuvromullick/npatlas npatlas
"""

from __future__ import annotations

import json
import os
import sys
import time
import zipfile

import httpx

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "k_data", "external")


def main() -> int:
    if len(sys.argv) < 2:
        raise SystemExit("用法: python code/dl_dataset.py <owner/slug> [tag]")
    ref = sys.argv[1]
    tag = sys.argv[2] if len(sys.argv) > 2 else ref.split("/")[-1]

    cred_path = os.path.join(ROOT, "kaggle.json")
    if not os.path.exists(cred_path):
        raise SystemExit(f"找不到 {cred_path}")
    d = json.load(open(cred_path, encoding="utf-8"))
    user, key = d["username"], d["key"]

    os.makedirs(OUT, exist_ok=True)
    zip_path = os.path.join(OUT, f"{tag}.zip")
    dest = os.path.join(OUT, tag)

    print(f"[dl] {ref} -> {zip_path}", flush=True)
    t0 = time.time()
    done = 0
    with httpx.Client(auth=(user, key),
                      timeout=httpx.Timeout(60.0, read=1800.0),
                      follow_redirects=True,
                      headers={"User-Agent": "casmi26/1.0"}) as c:
        with c.stream("GET", f"https://www.kaggle.com/api/v1/datasets/download/{ref}") as r:
            if r.status_code != 200:
                body = r.read()[:300]
                raise SystemExit(f"HTTP {r.status_code}: {body!r}")
            total = int(r.headers.get("content-length") or 0)
            with open(zip_path, "wb") as f:
                for chunk in r.iter_bytes(1024 * 512):
                    f.write(chunk)
                    done += len(chunk)
                    if total:
                        print(f"\r  {done/total*100:5.1f}%  {done/1024**2:8.1f} MB",
                              end="", flush=True)
    print(f"\n[done] {done/1024**2:.1f} MB in {time.time()-t0:.0f}s")

    try:
        with zipfile.ZipFile(zip_path) as z:
            names = z.namelist()
            print(f"[zip] {len(names)} members:")
            for n in names[:20]:
                print(f"   {n}")
            z.extractall(dest)
        print(f"[extract] -> {dest}")
    except Exception as e:
        print(f"[warn] unzip failed ({e})，原样保留 {zip_path}")
        return 1

    for root, _, files in os.walk(dest):
        for f in files:
            p = os.path.join(root, f)
            print(f"   {p}  {os.path.getsize(p)/1024**2:8.1f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
