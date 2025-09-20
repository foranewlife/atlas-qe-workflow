# Repository Guidelines

## Structure & Roles
- Core: `aqflow/core/` — `eos_controller.py` (build tasks), `process_orchestrator.py` (state-based scheduler), `resources_simple.py` (resource tools), `task_creation.py` (inputs), `task_processing.py` (proc utils)。
- Software adapters: `aqflow/software/` — `atlas.py`, `qe.py`（命令与输入生成）。
- CLI: `aqflow`（统一入口）；旧代码在 `archive/legacy/`。
- Config/Examples: `config/resources.yaml`, `examples/*`；Outputs: `results/`, Logs: `logs/`。

## Run & Logs
- Minimal run（建议加超时防卡）：
  - `timeout 60s aqflow eos examples/test_qe_small/test_qe_small.yaml`
- 状态持久化：`results/tasks_simple.json`；失败会打印 `job.out` 尾部（最多 2000 字）。

## Task Service（看板 + 本地运行）
- 启动服务（提供看板和本地运行 API）
  - `aqflow server [--host 127.0.0.1 --port 8765]`
- 在当前目录运行 atlas/qe（自动启动服务如未运行）
  - `aqflow atlas` 或 `aqflow qe`
- 看板
  - 浏览器打开 `http://127.0.0.1:8765/dashboard`（每 2 秒自动刷新）

## Resource & Scheduler Config（config/resources.yaml）
- resources: 每资源配置
  - `name`, `type: local|remote`, `cores`（总容量），远端含 `host`, `user`, `workdir`
  - `software.<sw>`: `path`, `cores`（单任务核数），`mpi`，`env`（OMP 等）
  - 可选 `transfer.pull_all: true` 回传远端任务目录全部文件
- scheduler: `max_parallel`, `poll_interval`；timeouts: `default`（软超时秒）；policy: `prefer.atlas: local`, `prefer.qe_multicore: remote`
- 核心使用位置：在对应资源的 `software.<sw>.cores` 设置（QE >1 自动走 MPI）。

## Coding & Conventions
- Python 3.11+；Black(88)，4 空格；类型标注优先；命名：`snake_case`/`CapWords`/`UPPER_CASE`。
- 可执行放 `scripts/`，复用逻辑放 `aqflow/`；不要把输出写到 repo 根。

## Dev Tips
- 远端失败定位：日志包含 `ssh/scp` rc 与命令；回收失败会打印错误；启用 `--log-level DEBUG` 查看详细命令。
- 常用目录：QE/ATLAS 输出默认回收 `job.out`；如需全量回收，在资源下开启 `transfer.pull_all`。
