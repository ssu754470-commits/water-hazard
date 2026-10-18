# Water Hazard

已完成 HD10K 液体污渍分割的 U-Net baseline，现将既有代码和实验结果归档到此项目。

- [U-Net baseline：代码、配置、运行说明与结果](baselines/unet/README.md)
- [本次 baseline 实验记录（含失败与不足）](docs/experiments/2026-10-14-unet-baseline.md)
- 状态：修复后的 FP32/AdamW 实验完成 30 轮；最终日志验证 Dice 0.6783。
- 已知问题：前景漏检；随机视频帧验证不代表独立测试性能。
- 权重保留本地，仓库记录路径和 SHA256；原始数据与大型模型未纳入 Git。

本次仅整理已有实验，没有重新训练。后续改进实验与 baseline 分开记录。

