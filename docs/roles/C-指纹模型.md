# C · 指纹模型

**角色定位**：纯 ML，GPU 最密集。任务单一但吃资源：**把谱变成指纹**。

**你负责的文件**：`code/c_fpnet/`（`dataset.py`、`model.py`、`train.py`、`infer.py`）

**依赖**：A 的 `spectra.parquet`。**不需要等候选池就能开工。**

---

## W1（11/20–11/22）

### 本周交付

- [ ] **数据加载**：~2.5M 行**不要一次性进内存**
  - `IterableDataset` + 谱定长裁剪（top-128 峰，按强度）
  - **分子级 shuffle**（按 `molecule_id`），不是谱级
  - 强度 sqrt 变换；基峰已经是 1.0，不用再归一
- [ ] **模型（FPNet 基线）**
  - 输入：`[mz, intensity_sqrt]` + padding mask
  - 拼标量：`precursor_mz`、`adduct` embedding（**10 种测试加合物**必含）、
    `polarity`、`collision_energy_ev`（**list，取 mean/max 或整个喂进去**）
  - 骨干：1D CNN 或小 Transformer（**先小**）
  - 输出：2048 位概率（sigmoid）
  - 损失：**按分子加权 `1/n_spectra`**，否则 20 张谱的分子权重是 2 张的 10 倍
- [ ] **训练脚本**：单卡跑通、断点续训、日志
- [ ] **推理脚本**：`fp_logits(spectra, ckpt) -> (n, 2048)`

### GPU 预算纪律
Kaggle 每周 **30 小时**。W1 只花 **≤6 小时**：
- 先在 10% 数据上验证不崩、能收敛
- 跑一个 epoch 看 loss 曲线
- **不要一上来跑 5 折 × 多 seed**

### W1 验收
1. checkpoint 能加载并推理
2. 单谱指纹的**逐位 AUC**（held-out 谱，不需要候选池）
3. 有 loss 曲线和耗时报

---

## W2：候选打分与多谱聚合

- [ ] **`fp_scores()`**：候选分 = 指纹位与 logit 的点积
  - 至少三种：`f_fp_dot` / `f_fp_cos` / `f_fp_nbits_hit`
  - 写进 `feat_fp.parquet`，主键 `(molecule_id, cand_key)`
- [ ] **多谱聚合**（中位 3 张谱，隐藏增益点）
  - v1：logit 平均
  - v2：按 `collision_energy_ev` 分组 attention
  - v3：**候选层聚合**（先每谱算候选分再聚合，通常更好）
- [ ] **预训练骨干**：官方点名 **DreaMS / GeMS** 和 MassIVE
  - DreaMS 在 Zenodo 有预训练权重（`zenodo.org/records/10997887`）
  - 收益大但风险高：**先在小型验证上确认推理不慢于预算**

### W2 验收
- [ ] `mimic` 折上指纹通道 ΔMRR
- [ ] 候选打分耗时：2000 候选 × N 分子，外推到 ~400 分子是否超预算

---

## W3：效率与质量

- [ ] 推理加速：半精度、batch、**向量化点积**（2000×2048 不该是瓶颈）
- [ ] 指纹位重要性分析：几乎从不出现的位可降权
- [ ] 互补性检查：指纹模型在 class 2 / class 3 上分别贡献多少

---

## W4：冻结

- [ ] 权重备份进 Dataset，确认 `internet=off` 能加载
- [ ] 只修 bug

---

## 常见错误

| ❌ | ✅ |
|---|---|
| 谱级 shuffle | 分子级 |
| 不按分子数加权损失 | `1/n_spectra` |
| W1 就烧完 30h GPU | W1 ≤6h |
| 只看总 MRR | 必须拆 class 2 / class 3 |
| 种子不可复现 | 固定 seed，记进 `experiments.md` |
| 把 `collision_energy_ev` 当标量 | 它是 **list** |
