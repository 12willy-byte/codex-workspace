# AquaGuard AI v4.0

基于现有泳池视频监控系统的防溺水辅助预警工程基线。

> 安全边界：当前版本用于软件架构验证和算法接入，不是经过真实泳池验证的生命安全产品，不能替代持证救生员、现场巡查或法定安全设施。

## 当前已落地

- FastAPI 健康检查、风险评估和事件查询接口
- 可独立测试的多特征风险融合逻辑
- 连续帧确认、报警去抖和事件状态模型
- 摄像头、视觉模型、报警输出的协议接口
- 单元测试、容器文件和 CI 基础配置

真实 RTSP、YOLO/Pose、持久化数据库与 IP 音箱驱动尚未接入；参见 [`docs/ROADMAP.md`](docs/ROADMAP.md)。

项目范围、真实进度和统一里程碑以 [`docs/MASTER_PLAN.md`](docs/MASTER_PLAN.md) 为唯一依据。

## 快速运行

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
uvicorn aquaguard.main:app --reload
pytest
```

API 文档：`http://127.0.0.1:8000/docs`
