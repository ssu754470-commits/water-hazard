# 已运行的 U-Net baseline

这是现有本地实验的归档，迁入仓库时没有重新训练、重新评估或改动训练算法。训练入口及其依赖按当前已有代码复制；其中断点恢复功能在运行期间补齐，主训练使用的是新初始化权重，不是旧 NaN 模型续训。

## 已完成实验

- 数据：HD10K IROS2022 的 liquid_dirts，4000 对图片/分割掩码。solid_dirts 是检测框，不纳入本分割 baseline。
- 3600 张训练、400 张验证，随机划分 seed=0；模型 seed=42。隐藏的 macOS 元数据文件被过滤，掩码统一为二值前景。
- U-Net，RGB 输入，2 类，转置卷积上采样；缩放 0.5（320×240），batch=1。
- 修复后使用 FP32 + AdamW，初始学习率 1e-4，完成 30 轮。完整配置见 [run-config.json](run-config.json)。
- 2026-10-14 13:00（北京时间）完成；最后一次验证 Dice **0.678323**，第 30 轮平均训练 loss **0.002572**。
- 模型导出时通过有限值与单张真实图像推理检查。数值有效不代表分割质量充分。

## 结果与限制

[results/training-summary.log](results/training-summary.log) 来自既有日志，保留验证结果和每轮摘要；[run-manifest.json](results/run-manifest.json) 记录来源、权重路径、大小与 SHA256。

第 15 轮曾单独评估：整体 Dice 0.6121；有污渍图片平均 Dice 0.2543，IoU 0.2061；污渍像素 precision 0.9859，recall 0.2268；200 张空掩码中 6 张存在误检。详见 [原始指标](results/epoch15-validation.json) 和 [预测对比](results/epoch15-predictions.png)。**这些详细指标属于第 15 轮，不是第 30 轮。** 第 30 轮仅保留原日志指标，未在本次归档中补做评估。

验证集一半无污渍，整体 Dice 会掩盖前景漏检；随机视频帧划分还可能共享相近场景，不能当作独立测试集性能。后续调阈值/微调产生的 liquid_improved、liquid_final 等实验不混入本 baseline。

## 文件与权重

代码来源：milesial/Pytorch-UNet，保留 [GPL-3.0 LICENSE](LICENSE) 和 [上游 README](UPSTREAM_README.md)。本地适配包括 HD10K 数据读取、数值检查、FP32 AdamW、每轮调度和完整状态保存/恢复。

旧 AMP/RMSprop 运行第 1 轮即出现 NaN，旧 checkpoints_hd10k 不可用，不能作为 baseline 成功结果。有效权重是 checkpoints_hd10k_fixed_20261014 的第 30 轮，已导出为 MODEL_VALID.pth。

遵循仓库大型模型使用外部存储的约定，124 MB 权重和原始数据不进 Git。当前只有本地位置（未上传公共下载链接），详见 manifest；其他机器需另行取得权重后才能推理。

## 运行入口（仅供后续使用，本次未执行）

```powershell
cd projects/water-hazard/baselines/unet
python -m pip install -r requirements.txt
python train_hd10k.py --data-root C:/program1/HD10K_IROS2022 --epochs 30
python train_hd10k.py --data-root C:/program1/HD10K_IROS2022 --epochs 30 --resume checkpoints_hd10k_fixed_20261014/training_state.pth
python predict.py --model MODEL_VALID.pth --input image.jpg --scale 0.5
```

训练状态恢复与仅载入模型权重 `--load` 不同；`--resume` 恢复模型、优化器、调度器、scaler 和步数，但不保证随机数状态逐步重放。requirements.txt 是导入依赖列表，environment-observed.txt 是归档时环境快照，不声称是训练开始时的锁定环境。

根目录复制的三个辅助脚本保留本地原始路径与目录假设，属于实验来源文件；不要直接从归档位置运行它们。训练与预测的正式入口是上面的 train_hd10k.py、predict.py。

完整实验记录见 [2026-10-14 U-Net baseline](../../docs/experiments/2026-10-14-unet-baseline.md)，包含失败经过、配置、轮次对应指标和未解决的问题。
