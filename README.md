# RelationFusion

无融合真值的红外/可见光图像融合（infrared and visible image fusion）研究与训练代码，模型版本 `relation_v3`。

核心思想是把"融合"重新表述为"关系建模"：不直接学习"两张图 → 一张图"的黑盒映射，而是先显式分解两个模态之间**共有 / 私有 / 未解释**的关系，再依据证据决策加权融合。由于融合任务不存在标准答案，训练全部依靠自造的三层监督信号（干净参考重建、保守弱目标、已知因子合成视图），不使用任何融合 ground-truth。

## 方法结构

前向管线（`nets/fusion.py` 的 `RelationFusion`）：

1. **SharedEncoder** — Restormer 风格通道注意力 Transformer，红外/可见光共享参数、分别编码；
2. **RelationFactorization** — 关系分解：共有内容 + 各模态私有内容 + 支持度图 + 未解释残差，隶属度 sigmoid 混合；
3. **CandidateReliability / SourceTrust** — 三个候选的逐像素质量图与源图可信度；
4. **RelationDecision** — 光谱描述子 + MDA 空间注意力产生证据，Beta 证据理论决策输出融合权重与不确定度；
5. **JointInteraction** — 对未解释残差做跨通道注意力，产生联合分支；
6. **assemble + ImageDecoder** — 按质量/贡献度门控加权四条分支，解码为融合亮度图。

训练采用五阶段课程（warmup → relation → base → joint → 可选 calibration），每个参数组独立 AdamW 与绝对轮次余弦调度；监督信号由退化增强（干净参考）、自然图像弱目标、受控合成视图（已知因子真值）三层构成，并通过系统的梯度分权设计保证各概念只从自己的监督信号学习。

## 环境

- Python 3.12 + PyTorch（开发环境为 torch 2.5.1 + cu124）
- 训练默认使用 CUDA；`--device cpu` 可运行但很慢；`--check-data` 无需 CUDA

## 数据准备

数据集（MSRS / LLVIP）请自行下载，按如下结构放置：

```
data/MSRS/train/     # 训练集：1083 对
  ir/    xxx.png ...          # 红外
  vis/   xxx.png ...          # 可见光（目录名 vis 或 vi 均可，文件名与 ir 对应）
data/MSRS/test/      # 测试集：361 对
data/LLVIP/test/     # 测试集：347 对
```

两个目录中的图像按**文件名主干**配对，配对数量不匹配会直接报错。

## 使用

```bash
# 数据配对冒烟检查（无需 CUDA）
python train.py --check-data

# 训练（参数在 train.py 顶部的 TRAIN_PARAMETERS / MODEL_PARAMETERS 中修改，也可用同名命令行参数覆盖）
python train.py
python train.py --output runs/relation_v3_new

# 断点续训（恢复优化器/调度器/RNG/数据加载进度）
python train.py --resume runs/relation_v3/checkpoints/latest.pt

# 推理融合 + 指标评估
python test.py --checkpoint runs/relation_v3/checkpoints/latest.pt \
  --data-root data/MSRS/test --output results/relation_v3 --metrics
```

`--metrics` 输出十项融合指标（EN / SD / SF / AG / MI / CC / SCD / VIF / Qabf / SSIM）到 `<output>/metrics.csv`，结果同时保存灰度图（`gray/`）与伪彩色图（`rgb/`）。

## 代码结构

```
train.py / test.py   训练与推理入口（用户参数为文件顶部常量，同时生成 argparse 参数）
nets/                模型：fusion(总装) blocks(编码器/解码器) decomposition(关系分解)
                     verification(质量校验) decision(证据决策) interaction(联合交互)
                     spectral_gate / spatial_attention / beta_decision(决策子件)
utils/               dataset(配对数据) augment(退化增强) controlled_views(受控合成视图)
                     targets(目标构造) losses(损失) engine(训练引擎) settings(配置校验)
                     checkpoint(检查点) metrics(指标) operators/image/numerics/runtime
```

## 说明

- 检查点为完整快照（模型/优化器/调度器/RNG/数据加载状态 + 训练溯源），加载时严格校验一致性；`--joint-mode auto` 依据检查点溯源自动选择推理分支。
- 训练不划分验证集、不按指标挑选检查点；参考教师（SourceTrust 冻结）通过受控视图上的程序化审计启用。
- 本仓库不含数据集与训练产物；受版面所限权重与融合结果不入库。

## License

TBD
