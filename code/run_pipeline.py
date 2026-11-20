"""主流程。**每人只加开关，不写 if 张三的模型，不加代码分支。**

用法
----
    python code/run_pipeline.py --smoke                    # 50 分子，5 分钟
    python code/run_pipeline.py --stage all --budget 9h    # 正式推理
    python code/run_pipeline.py --stage submit             # 只拼提交
    python code/run_pipeline.py --stage eval               # 只算 CV

降级链（W4 定稿，顺序来自消融表：先砍 ΔMRR 最小的）
-----------------------------------------------------
    全量 → 砍前向模型(只算 top-20) → 砍生成 → 指纹只算 top-500 → 纯检索
    **任何降级路径都必须先给所有分子产出 25 个兜底候选。**
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import common as C
import a_candidates
import a_eval
import a_submit

log = logging.getLogger("casmi.pipeline")


def stage_build(args):
    """A：构建候选池。"""
    log.info("[stage] build candidates")
    # df = a_candidates.build_candidates(...)
    raise NotImplementedError("等 A 的 build_candidates 落地")


def stage_search(args):
    """B：检索通道特征。"""
    log.info("[stage] search features")
    raise NotImplementedError("等 B 的 search_scores 落地")


def stage_fp(args):
    """C：指纹通道特征。"""
    log.info("[stage] fp features")
    raise NotImplementedError("等 C 的 fp_scores 落地")


def stage_fwd(args):
    """D：前向模型特征（仅 top-K）。"""
    log.info("[stage] forward features (top_k=%d)", args.fwd_topk)
    raise NotImplementedError("等 D 的 forward_scores 落地")


def stage_rank(args):
    """D：融合重排 -> 分数长表。"""
    log.info("[stage] rank")
    raise NotImplementedError("等 D 的 predict_ranker 落地")


def stage_submit(args):
    """A：拼提交 + 校验。"""
    log.info("[stage] submit")
    raise NotImplementedError("等 D 的分数长表落地")


STAGES = {
    "build": stage_build,
    "search": stage_search,
    "fp": stage_fp,
    "fwd": stage_fwd,
    "rank": stage_rank,
    "submit": stage_submit,
}
ORDER = ["build", "search", "fp", "fwd", "rank", "submit"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=["all", "eval"] + ORDER)
    ap.add_argument("--smoke", action="store_true", help="50 个分子，5 分钟内跑完")
    ap.add_argument("--budget", default=None, help="推理时间预算，如 9h")
    ap.add_argument("--fwd-topk", type=int, default=100)
    ap.add_argument("--out", default=os.path.join(C.OUT_DIR, "submission.csv"))
    args = ap.parse_args()

    t0 = time.time()
    todo = ORDER if args.stage == "all" else [args.stage]
    for name in todo:
        if name == "eval":
            continue
        t = time.time()
        STAGES[name](args)
        log.info("[done] %s %.1fs", name, time.time() - t)

    log.info("total %.1fs", time.time() - t0)


if __name__ == "__main__":
    main()
