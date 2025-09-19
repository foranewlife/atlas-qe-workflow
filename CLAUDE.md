# ATLAS-QE 材料计算工作流

构建**多结构多方法EOS分析引擎**，支持真实模板文件的参数组合枚举和智能缓存。

## 核心特性
- **多结构支持**：每个结构独立的体积参数和平衡体积
- **模板系统**：基于真实计算文件的参数替换机制
- **参数组合枚举**：明确的方法-结构适用性配置
- **智能缓存**：基于参数指纹避免重复计算

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
- **现有资产**：plot_eos.py
- **uv** 包管理

### 使用示例

```bash
# 执行多结构EOS研究
python run_eos_workflow.py examples/gaas_eos_study/gaas_eos_study.yaml

# 预期计算量：3个结构 × 6个组合 × 15个体积点 = 270个任务
```