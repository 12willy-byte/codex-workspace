# Architecture

```text
FrameSource -> VisionAnalyzer -> RiskEngine -> EventService -> AlarmSink
                                      |             |
                                  assessment     event store
```

核心模块通过 `ports.py` 中的协议隔离。当前风险引擎是确定性基线，目的是先验证事件闭环；视觉推理结果必须按同一 `track_id` 连续输入，达到阈值与确认帧数后才生成事件。

后续适配器：RTSP 重连、检测/跟踪/Pose/水面 ROI、PostgreSQL、MQTT/IP 广播。报警输出需幂等，人工确认与解除操作必须保留审计记录。
