# Repository Guidelines

## Structure & Roles
- Core: `aqflow/core/` — `eos.py`（任务与输入生成）、`executor.py`（极简执行/调度）、`tasks.py`（输入生成）。
- Software adapters: `aqflow/software/` — `atlas.py`, `qe.py`（输入生成）。
- CLI: `aqflow`（统一入口）。
- Config/Examples: `config/resources.yaml`, `examples/*`；Outputs: `results/`, Logs: `logs/`。

## Run & Logs
- Minimal run（建议加超时防卡）：
  - `timeout 60s aqflow eos examples/test_qe_small/test_qe_small.yaml`
- 状态持久化：`aqflow/board.json`（单文件）；全局软链：`~/.aqflow/boards/`。
- 日志：`logs/aqflow.log`（文件），终端默认只显示 WARNING+。

## Dashboard（无服务）与本地提交
- 终端看板：`aqflow board`（聚合 `~/.aqflow/boards/*.json`，默认仅显示 running）。
- 直接将当前目录提交给调度器：`aqflow atlas` 或 `aqflow qe`（读取 `config/resources.yaml`）。

## Resource & Scheduler Config（config/resources.yaml）
- resources: 每资源配置
  - `name`, `type: local|remote`, `cores`（总容量），远端含 `host`, `user`, `workdir`
  - `software.<sw>`: `path`, `cores`（单任务核数），`mpi`，`env`（OMP 等）
  - 可选 `transfer.pull_all: true` 回传远端任务目录全部文件
- scheduler: `max_parallel`, `poll_interval`；timeouts: `default`（软超时秒）
- 核心使用位置：在对应资源的 `software.<sw>.cores` 设置（QE >1 自动走 MPI）。

## Coding & Conventions
- Python 3.11+；Black(88)，4 空格；类型标注优先；命名：`snake_case`/`CapWords`/`UPPER_CASE`。
- 可执行放 `scripts/`，复用逻辑放 `aqflow/`；不要把输出写到 repo 根。

## Dev Tips
- 远端失败定位：日志包含 `ssh/scp` rc 与命令；回收失败会打印错误；如需详细命令打印请查看 `logs/aqflow.log`。
- 常用目录：QE/ATLAS 输出默认回收 `job.out`；如需全量回收，在资源下开启 `transfer.pull_all`。
