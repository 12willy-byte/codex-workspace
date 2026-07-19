# AquaGuard AI v4.0

基于现有泳池视频监控系统的防溺水辅助预警工程基线。

> 安全边界：当前版本用于软件架构验证和算法接入，不是经过真实泳池验证的生命安全产品，不能替代持证救生员、现场巡查或法定安全设施。

## 当前已落地

- FastAPI 健康检查、风险评估和事件查询接口
- 可独立测试的多特征风险融合逻辑
- 连续帧确认、报警去抖和事件状态模型
- 摄像头、视觉模型、报警输出的协议接口
- 多摄像头近时刻融合、Pool BEV、未来风险与独立安全监督器
- 报警前后证据窗口、可恢复索引、完整性校验与留存清理
- 多摄像头运行生命周期和校验严格的配置装配
- 单元测试、容器文件和 CI 基础配置

真实 RTSP 现场验证、YOLO/Pose、持久化数据库与 IP 音箱驱动尚未接入；参见 [`docs/ROADMAP.md`](docs/ROADMAP.md)。

摄像头默认禁用。启用摄像头必须显式提供视频源、九参数图像到泳池坐标 Homography，以及生产帧分析器。缺少生产分析器时应用会明确拒绝启动，不会用脚本或空检测冒充现场保护。

当前可选 `ultralytics_tracking` 适配器只输出人员检测、本地跟踪 ID 和锚点运动，不输出头部入水、姿态、挣扎或溺水概率。模型必须是本地文件，系统不会隐式联网下载权重。

`ultralytics_pose_tracking` 会在同一个跟踪结果中关联 COCO 17 点 Pose，额外输出躯干垂直度和关键点遮挡程度。它仍不会在没有水域标定时推断头部入水，也不会把手腕运动直接等同于挣扎。

可为每个 Pose 摄像头配置图像坐标 `water_roi` 多边形。系统只输出 `head_in_water_region` 和 `water_relation_confidence` 供审计；头部位于水域投影范围不代表已经沉水，因此不会自动改写 `head_submerged`。

Pose 适配器还输出跨帧 `wrist_motion` 及其置信度；关键点遮挡会中断历史，快速腕动不会直接写成 `struggle`。离场轨迹状态按可配置 TTL 清理，避免长期运行内存无限增长。

运行状态 API 会分别报告 `tracking_only`、`pose_baseline` 或未来的 `validated_assistive_alerting`。视频线程处于运行状态不代表已具备防溺水报警能力。

主应用的报警资格门只允许 `validated_assistive_alerting` 创建对外报警事件和证据窗口。未验证摄像头仍可返回风险评估，但 API 会给出 `alarm_eligible=false` 和明确抑制原因。

所有评估都会记录输入特征、风险结果、报警资格、抑制原因和关联事件 ID。默认使用有界的进程内审核快照，服务重启后会丢失。

设置 `AQUAGUARD_AUDIT_DATABASE_PATH` 后，评估审核改用 SQLite 事务存储并可在重启后恢复；支持按摄像头、抑制原因和数量过滤。未设置时保持有界内存模式。

设置 `AQUAGUARD_EVENT_DATABASE_PATH` 后，正式报警事件、处理状态和冷却依据会写入 SQLite，并可在服务重启后恢复。数据库事务可防止同一设备上多个服务实例在冷却期内重复登记；事件登记与证据文件写入仍不是单一原子事务，因此系统提供独立的一致性核对接口。

`GET /api/v1/system/evidence-consistency` 提供只读一致性报告，区分待采集、待持久化、已存储、缺失、校验失败和无对应事件的孤立证据。报告只诊断，不会自动删除或伪造修复生命安全事件记录。

证据处置支持把 `ready` 片段写盘，以及对缺失、孤立和完整性失败进行带操作人和原因的审计记录；失败尝试同样留下结果。处置不会删除证据或补造事件。写 API 默认关闭，只有设置 `AQUAGUARD_REMEDIATION_ENABLED=true` 才开放；在用户认证和角色权限完成前，不应在生产环境启用。设置 `AQUAGUARD_REMEDIATION_DATABASE_PATH` 可将处置审计保存到 SQLite 并在重启后恢复。

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
