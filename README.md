# wajue · CASMI 2026 分子鉴定

4 人队参加 [CASMI 2026 · Molecule ID from Mass Spectra](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra)。
任务：从 LC-MS/MS 谱预测 2D 结构，每分子 ≤25 个 SMILES，**MRR@25** 计分。

> 所有事实已于 **2026-11-20** 从官方页面核实，原文落盘在 [_research/](_research/)，
> 出处见 [docs/00](docs/00-赛题理解与判分细节.md)。

## 先读这个

**可见的 `test.parquet` 取自训练集，rerun 时会被替换。**
在它上面算任何分数都是虚高的。本地验证只能从 `train.parquet` 切分。

```bash
python code/a_load.py --check-leak   # 亲手确认这件事，不要只信文档
```

## 当前进度（2026-11-20）

| 阶段 | 状态 | 结果 |
|---|---|---|
| 数据与 schema | ✅ | train 2,539,608 行 / test 1,213 行，与官方 12/18 列清单完全一致 |
| 泄漏核查 | ✅ | **可见 test 的谱 100% 出现在 train 里**（指纹全量比对） |
| 中性质量推算 | ✅ | 10 种加合物，400/400 在 5 ppm 内（修了多谱多簇的 bug） |
| 结构池 | ✅ | train 275,301 + **COCONUT 470,226** + NPAtlas 32,550 → **合并 720,158** |
| 谱库检索 | ✅ | cos + 中性丢失，流式扫 21 个 row group，284s |
| **合成空间基线** | ✅ | **MRR@25 0.8727**（R@1 78.2%）— 化学空间不匹配，**不能当榜分** |
| **天然产物基线** | ✅ | **MRR@25 0.7581**（R@1 70.4%）— 更接近真实测试集 |
| **class 2 模拟** | ✅ | **0.2193**（无 COCONUT 时为 **0.0000**） |
| LightGBM ranker | ⚠️ | +0.0074，**未超噪声，不上线** |
| 按谱证据分流 | ✅ | `build_candidates_routed()` 已实现，阈值待校准 |

### 四条硬结论

1. **COCONUT 有净价值**：class 2 从 **0 → 0.2193**（148 分子模拟）
2. **但不能无差别合并**：class 1 从 0.7581 → **0.7111**（−0.047，候选数 33→69）
3. **分子式注释没用**：oracle 实验 0.2153 vs 0.2193，零收益
4. **class 2 的瓶颈是同分异构体**，只能靠结构打分模型（指纹/前向）

### 期望总分

| class1 | class2 | class3 | 期望 MRR |
|---|---|---|---|
| 40% | 20% | 40% | 0.3471 |
| 50% | 20% | 30% | 0.4229 |
| **50%** | **30%** | **20%** | **0.4448** |

榜单中位 **0.440**。

详细数据见 [experiments.md](experiments.md) · 失败分析见 [docs/04](docs/04-W1失败分析.md)

## 外部数据

已下载入库（`k_data/external/`）：

| 库 | 原始 | 入库 | 命令 |
|---|---|---|---|
| COCONUT csv_lite 2026-09 | 738,807 | 470,226 | `python code/a_ingest_external.py <csv> --name coconut` |
| NPAtlas 2024-09 | 36,454 | 32,550 | 同上，`--name npatlas` |

下载器：`python code/dl_dataset.py <owner/slug> [tag]`（需要根目录 `kaggle.json`）

## 入口

| 文件 | 给谁看 |
|---|---|
| **[PLAN.md](PLAN.md)** | 所有人先读（目标校准、纪律、分工、里程碑） |
| **[docs/00](docs/00-赛题理解与判分细节.md)** | 判分规则的硬事实 + 已订正的错误假设 |
| **[docs/02](docs/02-接口契约.md)** | 四个人并行工作的前提 |
| **[docs/03](docs/03-验证协议.md)** | 为什么 CV 会虚高、怎么防 |
| **[TASKS.md](TASKS.md)** | 看板，每天同步时改状态 |
| [docs/roles/](docs/roles/) | 按角色的四周任务清单 |
| [experiments.md](experiments.md) | 实验日志，每次必记 |

## 快速开始

```bash
cd wajue
pip install -r requirements.txt
python code/common.py              # 环境 + 加合物表自检
python code/a_eval.py --selftest    # 评测实现自检
python code/a_submit.py --selftest  # 提交格式自检
```

## 现实目标（按真实榜单校准）

榜单 5,591 队，中位 **0.440**，p90 **0.492**，榜首 **0.552**。分数压缩在 0.44–0.50。

| 目标 | 分数 |
|---|---|
| 能交上去 | 0.45 |
| 像样 | 0.49 |
| 有竞争力 | 0.52 |
| 争奖 | 0.55+ |

## 目录

```
code/           实现（每人只改自己的文件，见接口契约）
data/           raw → interim → candidates → features（git 忽略）
_research/      官方页面原文（已抓取）
notebooks/      探索用，不要成为唯一实现
output/         提交文件
experiments.md  实验日志
```

## 五条纪律

1. 可见 `test.parquet` 是训练集样本，不能用来算分。
2. 本地 CV 是唯一可信信号，公开榜只用来确认没崩。
3. 所有外部数据提前挂成 Kaggle Dataset（推理断网 9 小时）。
4. 验证集按 `cand_key` 排除，不按 `ingest_lib` 过滤。
5. 提交列名是 `molecule_id,smiles`。
