# ATLAS-QE 材料计算工作流

## 项目愿景

构建基于配置驱动的自动化材料计算引擎，支持从收敛性测试到EOS分析的工作流。以最小状态机执行模型为核心，减少系统复杂度。

## CLAUD提示词（代码审查/重构原则）
一、 函数设计与复杂度优化（核心：可读性/可维护/可测试）
- 单一职责：超出单一职责的函数必须拆分为专注的子函数，并给出拆分依据；
- 扁平化：检查圈复杂度与缩进深度，使用卫语句（guard clauses）或策略模式降低嵌套；
- 长度限制：超出约定行数的函数需识别核心步骤并提取为独立函数；
- 纯度评估：标注并讨论副作用（I/O、外部状态修改等）的必要性；
- 可选：引入不可变数据/高阶函数以降低状态管理复杂度。

二、 副作用管理与明确约定（核心：可预测性/可测试性）
- 副作用清单：识别并分类所有副作用，明确边界；
- 逻辑分离：将纯业务逻辑与副作用隔离（如：先算后写）；
- 不可变输入：函数不得修改入参（必要时返回新值）；
- 依赖显式化：消除对全局状态的隐式依赖（使用依赖注入）；
- 替代实现：提供无副作用或副作用集中管理的方案。

三、 防御性编程与契约设计（核心：鲁棒与简洁）
- 契约优先：参数类型提示 + 入口校验（可结合 Pydantic）；
- 断言/验证：在函数开头明确参数假设，内部基于已验证参数编写；
- 错误处理：针对边界 I/O/网络操作，提供清晰、可诊断的异常信息；
- 并发安全：识别并处理竞态条件（文件写入/多进程并发等）；
- 边界覆盖：为关键边缘情形补充处理逻辑与测试。

四、 代码质量与工程化（核心：专业级质量）
- 命名与可读性：优化命名、解释性变量、消除魔法数字；
- 规范：遵循 PEP 8（行长、导入顺序、命名一致性等）；
- 安全：避免命令注入/不安全反序列化/路径穿越等；
- 性能：避免重复计算与不必要 I/O，合理选择数据结构；
- 资源：文件句柄/进程/连接的正确释放与关闭。

### 完整工作流架构
```
收敛性测试 → 结构优化 → EOS分析
           ↓
    配置驱动 + 轻量状态机
```

### 核心理念
- **模块化**：各阶段可独立运行或联合执行
- **智能缓存**：基于参数指纹避免重复计算
- **增量研究**：新参数集不影响已有计算
- **灵活查询**：任意参数组合的对比分析

## 当前实现特性（已完成）
- **多结构支持**：每个结构独立的体积参数和平衡体积
- **模板系统**：基于真实计算文件的参数替换机制
- **参数组合枚举**：明确的方法-结构适用性配置
- **配置解析**：YAML驱动的参数空间定义

## 目标功能（规划中）
- **收敛性测试**：QE参数(ecut_wfc, k_points)和ATLAS参数(gap)自动优化
- **结构优化**：QE variable cell relaxation获得平衡结构
- **智能缓存**：基于参数指纹的计算复用和相似参数集检测
- **任务调度**：本机/服务器的轻量调度（无常驻服务）
- **实时监控**：计算进度追踪和故障诊断
- **查询分析**：多维参数查询和科学对比构建

## 配置文件结构

### 赝势集定义
全局共享的赝势集，避免重复定义：

```yaml
pseudopotential_sets:
  gaas_lda_lpp_density:
    Ga: "ga.lda.lpp.lpp_density.UPF"
    As: "as.lda.lpp.lpp_density.UPF"
  gaas_lda_nlpp:
    Ga: "ga.lda.nlpp.UPF"
    As: "as.lda.nlpp.UPF"
  mg_lda:
    Mg: "mg.lda.upf"
```

### 结构定义
每个结构包含独立的体积参数和物理属性：

```yaml
structures:
  - name: "gaas_zincblende"
    elements: ["Ga", "As"]
    file: "POSCAR"
    description: "GaAs zincblende structure"
    volume_range: [0.8, 1.1]
    volume_points: 15

  - name: "mg_fcc"
    elements: ["Mg"]
    structure_type: "fcc"
    lattice_parameter: 4.54     # Å
    description: "Mg FCC structure"
    volume_range: [0.7, 1.3]
    volume_points: 13
```

### 参数组合配置
每个组合通过引用赝势集，指定方法参数：

```yaml
parameter_combinations:
  - name: "atlas_kedf801_gaas_lda_lpp_density"
    software: "atlas"
    template_file: "atlas.in.template"
    applies_to_structures: ["gaas_zincblende", "gaas_wurtzite"]
    pseudopotential_set: "gaas_lda_lpp_density"
    template_substitutions:
      KEDF: 801

  - name: "qe_ecut60_gaas_lda_lpp_density"
    software: "qe"
    template_file: "job.in.template"
    applies_to_structures: ["gaas_zincblende", "gaas_wurtzite"]
    pseudopotential_set: "gaas_lda_lpp_density"
    template_substitutions: {}
```

### 模板文件系统
通用化模板，程序自动生成结构相关内容：

```
# atlas.in.template
# CELLFILE, ELEMENTS, ppfile由程序自动生成
iSYSTEM = ATLAS_JOB
GAP = 0.20
KEDF = {KEDF}  # 唯一的方法参数占位符
# ... 其他固定参数

# job.in.template
&SYSTEM
   ecutwfc = 60  # 固定参数
   ecutrho = 240
   # nat, ntyp由程序自动设置
/
# ATOMIC_SPECIES, CELL_PARAMETERS, ATOMIC_POSITIONS由程序自动生成
```

## 工作流执行逻辑

```
对于每个结构:
  ├── 筛选适用的参数组合 (applies_to_structures)
  ├── 对于每个适用组合:
  │   ├── 从pseudopotential_sets获取赝势文件
  │   ├── 基于template_file和elements自动生成完整输入文件
  │   ├── 根据volume_range创建体积系列
  │   ├── 执行EOS计算 (volume_points个点)
  │   └── 拟合EOS结果
  └── 生成结构级分析
```

### 程序自动生成内容
- **ATLAS**: CELLFILE, ELEMENTS, ppfile行
- **QE**: ATOMIC_SPECIES, nat, ntyp, K_POINTS, CELL_PARAMETERS, ATOMIC_POSITIONS
- **通用**: 基于structure元素和pseudopotential_set的正确匹配

## 输出目录结构

```
results/
├── gaas_zincblende/
│   ├── atlas_kedf801_lpp_lpp_density/
│   │   ├── 0.8/ → atlas.in, atlas.out
│   │   ├── 0.82143/ → ...
│   │   ├── EOSRes.json
│   │   └── EOS.png
│   └── qe_ecut60_lpp_lpp_density/
├── mg_fcc/
│   ├── atlas_kedf801_mg/
│   └── qe_ecut60_mg/
└── comprehensive_analysis/
    ├── structure_comparison.png
    └── method_comparison.png
```

## 技术规范

### 计算资源
- **本机**：8核，ATLAS单核，QE多核MPI
- **并行策略**：结构×组合×体积点的三维并行

### 软件依赖
- **Python 3.11+**, **pymatgen** (EOS拟合)
- **uv** 包管理

## 项目进度跟踪

### 开发历程
- **2024-09**: 初始实现，参数组合枚举和模板系统 ✅
- **当前状态**: 配置驱动的输入文件生成系统完成，缺少计算执行引擎
- **下一阶段**: 建立ATLAS/QE软件调用和任务监控系统

### 架构演进
```
阶段1: 单体脚本 → 阶段2: 模块化引擎 → 阶段3: 智能工作流
[当前位置: 阶段2完成，向阶段3迁移]
```

### 关键组件状态
- **配置系统** ✅ YAML解析、参数空间定义
- **模板引擎** ✅ 输入文件生成、赝势匹配
- **目录管理** ✅ 结构化输出目录
- **软件执行** ✅ ATLAS/QE调用接口框架
- **任务调度** ✅ 基础任务管理和队列
- **结果收集** ✅ 能量提取和状态跟踪
- **状态追踪** ✅ 单文件状态（aqflow/board.json）
- **监控面板** ✅ 命令行查询工具
- **模块化架构** ✅ 完整的 aqflow/ 模块组织

### 当前可用功能
- **完整输入文件生成**: 支持多结构、多方法的输入文件自动创建
- **任务状态管理**: 单文件（aqflow/board.json）持久化任务状态
- **执行引擎框架**: ATLAS/QE软件调用接口（需配置实际可执行文件路径）
- **进度监控**: 实时任务状态查询和统计
- **模块化设计**: 清晰的代码组织，易于扩展和维护

### 技术债务（已解决）
- ~~`run_eos_workflow.py` 单体文件需模块化拆分~~ ✅ 已重构
- ~~缺少软件执行的进程管理和错误处理~~ ✅ 已实现
- ~~配置系统需扩展支持软件路径和执行参数~~ ✅ 框架已建立

### 使用示例（Quick Start）

```bash
# 1) 运行 EOS 示例（读取 config/resources.yaml 软件路径）
aqflow eos examples/test_qe_small/test_qe_small.yaml

# 2) 在 atlas/qe 目录下直接提交单任务
aqflow atlas   # 或 aqflow qe

# 3) 查询任务状态（默认仅显示 running）
aqflow board
# 可选：过滤/分组
# aqflow board --filter status:failed --group-by resource
```

### 当前交付成果

#### 1. 可运行的模块化工作流系统
- **输入文件生成**: 131个任务的完整输入文件自动创建
- **任务状态管理**: 单文件持久化，支持任务状态跟踪
- **进度监控**: 实时查询工具，按结构、状态、组合多维度查询

#### 2. 测试脚本和工具
- `aqflow eos`: 主工作流执行器（统一 CLI）
- `scripts/query_tasks.py`: 任务状态查询和监控工具
- 完整的131个计算任务自动生成和管理

#### 3. 技术架构成果
- **模块化设计**: 8个核心模块，清晰的接口定义
- **可扩展性**: 新软件、新方法可快速集成
- **数据持久化**: 任务状态和结果的完整记录

## 完整系统架构（目标设计）

### 1. 收敛性测试与结构优化

#### 收敛性测试协议
- **QE参数**：ecut_wfc (40→120 Ry), k_points (6→12), ecut_rho (自动4x)
- **ATLAS参数**：gap (0.15→0.30 Å)
- **收敛标准**：能量 1e-3 eV, 力 1e-2 eV/Å

#### 结构处理
- **标准结构**：fcc, bcc, diamond + 晶格参数
- **自定义结构**：POSCAR文件输入
- **QE优化**：variable cell relaxation获得平衡结构

### 2. 参数集管理与智能调度

#### 任务与状态
- **任务状态**：created/queued/running/succeeded/failed/timeout
- **状态持久化**：aqflow/board.json（工作目录），~/.aqflow/boards（聚合查看）

#### 资源调度
- **本机**：ATLAS单核，QE多核MPI
- **6101服务器**：大规模并行计算
- **动态负载均衡**：基于参数集类型优化分配

### 3. 数据管理与存储

#### 目录结构
```
data/{system}/
├── convergence_tests/          # 收敛性测试
│   ├── qe_ecut_convergence/
│   └── atlas_gap_convergence/
├── structure_optimization/     # 结构优化
│   ├── initial_structures/
│   └── qe_relax_calculations/
├── eos_calculations/          # EOS分析
│   ├── atlas_fcc_kedf701/
│   └── qe_fcc_ncpp_ecut80/
└── analysis/                  # 综合分析
    ├── convergence_summary/
    ├── structure_comparison/
    └── eos_comparisons/
```

### 4. 分析与可视化系统

#### 查询引擎
- **参数查询**：按软件、结构、参数范围灵活查询
- **对比构建**：自动发现科学合理的参数集组合
- **数据聚合**：多参数集数据的统一处理

#### 可视化模块
- **单体EOS**：individual参数集的EOS曲线 (基于plot_eos.py)
- **对比分析**：结构效应、方法对比、参数敏感性
- **矩阵图**：体积模量热力图、误差分布
- **报告生成**：HTML综合报告

## 验收标准

### 核心功能
- [ ] **完整工作流**：收敛测试 → 结构优化 → EOS分析一键完成
- [ ] **参数集管理**：唯一指纹生成，增量计算检测
- [ ] **智能缓存**：相同参数集100%复用
- [ ] **灵活查询**：多维参数查询，科学对比自动构建
- [ ] **可视化输出**：单体EOS图，对比分析图

### 性能指标
- **响应时间**：配置解析<30秒，查询响应<1秒
- **并行效率**：资源利用率>80%，故障恢复率>90%
- **易用性**：新用户15分钟内完成首次计算

### 当前实现状态（重构完成）
- [x] **配置解析** - YAML驱动的参数空间定义
- [x] **模板处理** - 输入文件自动生成
- [x] **目录管理** - 结构化输出组织
- [x] **软件执行** - ATLAS/QE调用框架已建立
- [x] **任务调度** - 任务管理和状态跟踪系统
- [x] **结果收集** - 基础的能量提取和状态记录
- [x] **状态追踪** - SQLite数据库缓存系统
- [x] **查询分析** - 命令行查询工具和数据导出
- [x] **主执行器** - 替换 run_eos_workflow_modular.py 为 run_distributed_workflow.py

### 下一阶段开发计划
1. **软件路径配置**: 集成实际ATLAS/QE可执行文件配置
2. **并行执行优化**: 多进程任务调度和资源管理
3. **EOS拟合集成**: 集成现有plot_eos.py功能
4. **Web监控界面**: 实时进度监控仪表板
5. **收敛性测试**: 自动参数优化模块

## 项目目录结构规范

### 目录分类原则
**禁止在项目主目录生成工作文件**，所有文件必须按功能分类存放：

```
atlas-qe-workflow/
├── aqflow/                 # 核心源代码模块
│   ├── core/              # 核心功能模块
│   ├── calculators/       # 计算器接口
│   └── utils/             # 工具函数
├── scripts/               # 可执行脚本和工具
│   ├── run_eos_workflow_modular.py
│   └── query_tasks.py
├── examples/              # 示例配置和模板
├── tests/                 # 单元测试
├── config/                # 配置文件
├── docs/                  # 项目文档
├── results/               # 计算结果输出（git忽略）
├── archive/               # 废弃文件存档
└── data/                  # 输入数据文件
```

### 文件分类规则
- **可执行脚本**: 放入`scripts/`目录
- **核心模块**: 放入 `aqflow/` 目录，按功能分组
- **配置文件**: 放入`config/`或`examples/`目录
- **测试文件**: 放入`tests/`目录
- **文档**: 放入`docs/`目录
- **临时/结果文件**: 放入`results/`目录（会被git忽略）
- **废弃文件**: 移入`archive/`目录

### 开发规范
1. **新功能开发**: 在 `aqflow/` 中创建模块；CLI 放 `aqflow/scripts/`
2. **配置管理**: 示例配置放`examples/`，系统配置放`config/`
3. **测试驱动**: 每个新模块都应有对应的测试文件
4. **文档同步**: 重要功能变更需更新CLAUDE.md和相关文档
