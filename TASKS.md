# TASKS · 看板

> 规则：**卡片没交付 = 没做**。每人每天同步时只改自己段落的状态。
> 状态：`TODO` / `WIP` / `BLOCKED` / `DONE`
> W1 只有 **3 天**（11/20–11/22），今天是 11/20，下面「今日必做」必须今天做完。

## 今日必做（2026-11-20）

| 任务 | 负责 | 状态 |
|---|---|---|
| 四人接受比赛规则 + **合并成一队**（12/07 后合不了） | 全体 | TODO |
| 数据下载到 `k_data/`（train 3.0GB / test 4.8MB） | A | **DONE** |
| `rdkit==2026.3.3` 安装（**仍缺，阻塞 metric key 与指纹**） | A | TODO |
| **`check_leak`：可见 test 100% 来自 train（已实测）** | A | **DONE** |
| 自检全绿（`common` / `chem` / `a_eval` / `a_submit`） | A | **DONE** |
| 官方页面原文存进 `_research/` | A | **DONE** |

---

## W1（11/20–11/22，只有 3 天）

### A · 基建 / 验证 / 集成
| ID | 任务 | 状态 |
|---|---|---|
| A-1 | `common.py`：路径、**10 种测试加合物表**、峰处理 | **DONE** |
| A-2 | `a_load.load_split()`：按官方 12/18 列 schema 加载 | **DONE** |
| A-3 | `_leak.py`：谱指纹全量比对 → **命中率 100%** | **DONE** |
| A-4 | `local_eval()` MRR@25（数学部分 + 分层诊断） | **DONE** |
| A-5 | `validate_sub()` + `assemble_submission()`（列名 `smiles`） | **DONE** |
| A-6 | `chem.py`：分子式→单同位素质量（不依赖 rdkit） | **DONE** |
| A-7 | `_recover_labels.py`：恢复 400 个分子的真值（一致性 400/400） | **DONE** |
| A-8 | `build_candidates()` 质量分桶 + ik14 去重 + 真值排除 | **DONE** |
| A-9 | `derives neutral mass`：400/400 在 5 ppm 内 | **DONE** |
| **A-10** | **接入 PubChem + COCONUT 结构池（class 2/3 的唯一出路）** | **TODO · P0** |
| A-11 | `get_folds(scheme="mimic")` + `assert_no_leak()` | TODO |
| **A-12** | **失败分析表** → 见 [docs/04](docs/04-W1失败分析.md) | **DONE** |

### B · 检索通道
| ID | 任务 | 状态 |
|---|---|---|
| B-1 | 环境（matchms 非必需，已用纯 numpy 实现） | **DONE** |
| B-2 | `LibraryScanner`：流式扫描 21 个 row group，284s | **DONE** |
| B-3 | `search_scores()`：cos + 中性丢失 + `f_cos_self` | **DONE** |
| B-4 | **同谱排除**（保留同化合物其他谱，模拟 class 1） | **DONE** |
| **B-5** | **analog 传播（近邻结构指纹外推）** | **TODO · P2** |
| B-6 | 前处理网格（相对强度下限、每 50 Da top-6、去同位素） | TODO |
| B-7 | `f_entropy` 通道（需 ms_entropy） | TODO |

### C · 指纹模型
| ID | 任务 | 状态 |
|---|---|---|
| C-1 | `IterableDataset` + 谱定长裁剪，分子级 shuffle | TODO |
| C-2 | FPNet 基线 + `1/n_spectra` 加权损失 | TODO |
| C-3 | 训练脚本 | TODO |
| C-4 | `fp_logits()` 推理 | TODO |
| **C-0** | **先装 rdkit（没有它无法生成指纹标签）** | **BLOCKED** |

### D · 重排 / 融合
| ID | 任务 | 状态 |
|---|---|---|
| D-1 | `rank_normalize` / `weighted_sum` / `merge_features` | **DONE** |
| D-2 | **`d_baseline.py`：第一条端到端基线 MRR 0.8727** | **DONE** |
| D-3 | 通道消融 → cos+nl 最优，rank 型质量特征有害 | **DONE** |
| **D-4** | **LightGBM ranker**（33% 错排是平局，线性分不开） | **TODO · P1** |
| D-5 | `predict_ranker()` → 分数长表 | TODO |
| D-6 | 特征重要度报告 | TODO |

**W1 验收**：✅ 可信 CV 基线（0.8727，class 1 条件）+ ✅ 端到端管线 + ✅ 失败分析表

---

## W2（11/23–11/29）— 已提前启动

| ID | 任务 | 负责 | 状态 |
|---|---|---|---|
| **A-10** | **外部结构池（P0 关键路径）** | A | **DONE** |
| A-10a | Kaggle Dataset 下载器 `dl_dataset.py` | A | **DONE** |
| A-10b | 通用入库器 `a_ingest_external.py`（自动识别列名） | A | **DONE** |
| A-10c | COCONUT 入库：738,807 → **470,226** 唯一结构 | A | **DONE** |
| A-10d | NPAtlas 入库：36,454 → **32,550** 唯一结构 | A | **DONE** |
| A-10e | 合并池 **720,158** 结构（COCONUT ∩ train 仅 5.4%） | A | **DONE** |
| A-10f | `_make_np_dev.py`：天然产物验证集（390 分子） | A | **DONE** |
| A-10g | `derive_neutral_masses` 多簇聚类修复 | A | **DONE** |
| A-10h | 官方同款前体一致性清洗 | A | **DONE** |
| A-11 | `get_folds(scheme="mimic")` + `assert_no_leak()` | A | TODO |
| A-13 | `merge_features()` 统一 schema | A | **DONE** |
| A-14 | 通道消融脚本 | A | **DONE** |
| **A-15** | **实现按谱证据分流（class1→小池 / class2→大池）** | A | **TODO · 下一优先** |
| **D-7** | LightGBM ranker | D | **DONE · 不上线**（+0.0074 未超噪声） |
| D-7a | 修掉 `f_cos_self` 自泄漏 | D | **DONE** |

### 本轮四条硬结论

| # | 结论 | 证据 |
|---|---|---|
| 1 | **COCONUT 有净价值**：class 2 从 **0 → 0.2193** | 148 分子模拟，真值只来自 COCONUT |
| 2 | **但不能无差别合并**：class 1 从 0.7581 → **0.7111**（−0.047） | 候选数 33→69 |
| 3 | **分子式注释没用**：oracle 实验 0.2153 vs 0.2193 | 窗口内分子式本就不重复 |
| 4 | **class 2 的瓶颈是同分异构体** | 只能靠结构打分模型解 |

### 期望总分（class1 用 train 池 0.7581，class2 用 COCONUT 0.2193）

| class1 | class2 | class3 | 期望 MRR |
|---|---|---|---|
| 40% | 20% | 40% | 0.3471 |
| 50% | 20% | 30% | 0.4229 |
| **50%** | **30%** | **20%** | **0.4448** |

榜单中位 0.440。

---

## W3（11/30–12/06）

| ID | 任务 | 负责 | 状态 |
|---|---|---|---|
| A-17 | `run_pipeline.py --budget 9h` 降级链 | A | TODO |
| A-18 | `internet=off` 完整跑一遍 | A | TODO |
| B-12 | `generate_edits()` class 3 骨架编辑 | B | TODO |
| B-13 | 生成候选只往尾部插 + `mrr_by_sim_type[3]` 评估 | B | TODO |
| C-8 | 推理加速 | C | TODO |
| C-9 | 预训练骨干（DreaMS / GeMS） | C | TODO |
| D-10 | `forward_scores()` top-100 候选 | D | TODO |
| D-11 | 同式错排专用评估 | D | TODO |
| D-12 | OOF 上搜融合权重并冻结 | D | TODO |
| **全体** | **12/07 前确认队伍最终状态** | 全体 | TODO |

---

## W4（12/07–12/14）

| ID | 任务 | 负责 | 状态 |
|---|---|---|---|
| A-19 | 冻结代码，只允许改 bug 和超时 | A | TODO |
| A-20 | 提交记录表 | A | TODO |
| D-13 | 降级链每档 `--smoke` 验证 | D | TODO |
| D-14 | 多 seed 复跑，选 CV 最稳的两个 | D | TODO |
| 全体 | **12/14 最终提交**（UTC 23:59） | 全体 | TODO |

---

## 阻塞登记

| 日期 | 谁阻塞在谁 | 原因 | 解决 |
|---|---|---|---|
| 2026-11-20 | C、D(部分) | **rdkit 未安装**，无法算 metric key / 生成指纹标签 | A 装 `rdkit==2026.3.3`；D 的基线已绕过（直接用 ik14 比对） |
| 2026-11-20 | 全体 | **缺 PubChem / COCONUT 结构池** → class 2/3 召回恒为 0 | A-10，W2 头号任务 |
