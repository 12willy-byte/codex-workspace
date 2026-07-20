# Architecture

```text
FrameSource -> VisionAnalyzer -> CameraCalibration -> Temporal/Multi-camera Fusion
                                                        |
                                                Pool BEV/Occupancy
                                                        |
                                                Future Risk Predictor
                                                        |
                                              Independent Supervisor
                                                        |
                              Evidence <- EventService -> AlarmSink
```

核心模块通过协议隔离。当前学习型视觉部分可选，世界模型、风险预测和监督器仍是可复现的确定性工程基线，不代表真实数据准确率。没有明确全局身份时，跨摄像头轨迹保持隔离。

逐摄像头标定必须使用验证过的 `CameraCalibrationArtifact`。它绑定源帧摘要、图像尺寸、控制点、Homography、水域 ROI、重投影误差、版本和标定人，并把摘要传入运行配置。详细流程见 [CAMERA_CALIBRATION.md](CAMERA_CALIBRATION.md)。

学习型时序预测通过固定特征协议注入 `WorldModelPipeline`。模型摘要、序列质量、输出范围和外部批准均失败关闭；未经验证的模型不能凭预测独立触发报警。详细契约见 [TEMPORAL_MODEL.md](TEMPORAL_MODEL.md)。

现有本地基线包括 RTSP 重连、检测/跟踪/Pose、水域 ROI、证据录像、SQLite 事件与审计、连续报警资格门。尚待实现的生产适配器包括 PostgreSQL 多节点存储、MQTT/IP 广播、集中身份与真实外部不可变审计。报警输出必须幂等，人工确认与证据处置必须保留审计记录。
