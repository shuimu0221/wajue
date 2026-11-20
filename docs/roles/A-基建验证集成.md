# A · 基建 / 验证 / 集成

**角色定位**：队长。全队唯一掌握「分数是否可信」的人。你的产出决定另外三个人是不是在白干。

**你负责的文件**：`code/common.py`、`code/a_load.py`、`code/a_eval.py`、
`code/a_candidates.py`、`code/a_submit.py`、`code/run_pipeline.py`

> 已完成（W1 前）：`common.py`（10 种测试加合物 + 峰处理，自检 PASS）、
> `a_load.py`（官方 schema 加载 + 泄露检查）、`a_eval.py`（MRR@25 数学部分 + 分层，自检 PASS）、
> `a_submit.py`（`molecule_id,smiles` 格式 + 5 条官方拒收条件，自检 PASS）、
> `d_ranker.py` 的 `rank_normalize` / `weighted_sum`。

---

## W1（11/20–11/22，只有 3 天）

### 今天（11/20）必须做完
- [ ] **报销名 + 组队合并**（12/07 之后合不了，这条只有你们自己能做）
- [ ] 配 `~/.kaggle/kaggle.json`，下载 `train/test/sample_submission`
- [ ] `pip install "rdkit==2026.3.3"`，版本号记进 `experiments.md`
- [ ] **`python code/a_load.py --check-leak`**——亲手确认可见 test 来自 train

### 本周交付

- [ ] **`get_folds(scheme="mimic")` + `assert_no_leak()`**
  - 见 [03-验证协议](../03-验证协议.md)
  - `mimic` 的 10/38/52 是**我们的假设**，不是官方数字，做成可调参数
  - **断言不过就抛异常，不许静默继续**
- [ ] **`build_candidates()`**
  - 候选源：`train.parquet` 的 27.5 万结构 + **COCONUT** + PubChem 过滤子集
  - 官方明确说 class 2 是「结构在 PubChem 或 COCONUT 里」→ **这两个库必须覆盖**
  - 质量分桶 parquet，pyarrow row-group 统计做范围裁剪，**别全量进内存**
  - 按 `cand_key` 去重；**测试集 10 种加合物**全部要能推中性质量
  - `max_per_mol=2000`
- [ ] `run_pipeline.py --smoke`：50 分子、5 分钟端到端
- [ ] **失败分析表**（W1 最有价值的产出）

### W1 验收
1. `a_load.py --stats` 打印出 train 的库分布、加合物分布、每分子谱数
2. `a_load.py --check-leak` 确认指纹重叠
3. 从 **train 切分**建出可信 CV 基线（**不是**在可见 test 上算）
4. **失败分析**：`cand_in_pool` 多少？候选缺失 vs 同式错排各占多少失分？

> 第 4 条决定 W2 三个人往哪儿投。

---

## W2：集成与度量

- [ ] `merge_features()` 统一 schema，缺值写 NaN 不丢行
- [ ] 每通道 ΔMRR 对比脚本
- [ ] 候选池版本管理：行数 + `candidates_hash()` 记进 `experiments.md`
- [ ] 分阶段耗时 profiler，外推到隐藏集（~1,500 谱 / ~400 分子）
- [ ] 每通道错例分析模板（哪些分子掉分、答案排第几）

---

## W3：预算与降级链

- [ ] `run_pipeline.py --budget 9h`
  - 降级顺序：前向模型 → 生成 → 指纹全量 → 只留检索
  - **降级链必须先产出兜底答案**（所有分子 25 个候选），再做贵计算
- [ ] `internet=off` 的 Kaggle notebook 完整跑一遍
- [ ] 提交队列与记录表

---

## W4：稳定与收口

- [ ] 冻结代码，只允许改 bug 和超时
- [ ] 两次最终提交：选多 seed CV 最稳的两版
- [ ] 复盘文档

---

## 常见错误

| ❌ | ✅ |
|---|---|
| 在可见 test.parquet 上算分 | 从 train 切分，可见 test 只用来确认 schema |
| 用榜分选方案 | 只用来确认没崩（~124 分子，噪声 ±0.004/分子） |
| 按 `ingest_lib` 做 holdout | 按 `cand_key` 排除 + 断言（同化合物跨库重复） |
| 信 `inchikey14` 列就是评测 key | 用 Kaggle 镜像重算并缓存 |
| 特征表用 SMILES 当键 | 一律 `cand_key` |
| 提交列名写 `smiles_list` | 官方是 **`smiles`** |
| 候选池重建了没通知 | 记 hash + 群里说 |
