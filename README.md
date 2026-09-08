# School Zone Matrix

从学校、小区和官方学区描述生成“学校 × 小区”概率矩阵与0/1关系矩阵。核心方法直接预测小区对应哪些学校，社区/居委会信息是可选特征，不是前置条件。

## 方法

1. 固定行政区、招生年份和学段。
2. 对全部学校与全部小区做笛卡尔积，不使用 Gold 缩小候选全集。
3. 从官方描述提取小区点名、边界、行政范围和共享规则特征；可以通过额外特征表加入道路、地址、OSM或经过审计的社区信息。
4. 用训练关系为每所学校建立正负空间锚点，计算最近距离、k近邻投票和候选竞争特征。
5. 使用逻辑回归与梯度提升模型进行多标签关系预测。
6. 在外层测试折之外做四折合并校准，选择模型、阈值和每个小区的最大候选数。
7. 先冻结全部预测及 SHA-256，再由独立命令读取 Gold 评价。

龙华区监督开发实验的五折汇总为 **85.97% 精确率、86.09% 召回率**。这不是全新盲测；详细限制见 [龙华结果](docs/longhua_result.md)。

## 安装

```bash
python -m venv .venv
python -m pip install -e ".[dev]"
```

## 输入文件

`communities.csv`：

| 必需字段 | 含义 |
|---|---|
| `community_id` | 稳定的小区ID |
| `community_name` | 小区名称 |
| `lon`, `lat` | WGS84经纬度；也可直接提供投影坐标 `x`, `y` |

可选提供 `group_id` 和 `outer_fold`。缺失时，程序会把同名或100米内的小区归入同一空间组，再稳定分成五折。

`schools.csv`：`school_id, school_name`。

`descriptions.csv`：`school_id, description`。同一学校允许多行，准备阶段会合并。

`labels.csv`：

| 字段 | 含义 |
|---|---|
| `community_id`, `school_id` | 关系主键 |
| `label` | 1属于、0不属于 |
| `observed` | 是否为可靠可评价标签；未知关系必须为false |

`extra_features.csv` 可用同一关系主键加入已经独立审计的数值特征。字段名不得包含 `gold`、`truth`、`label`、`prediction` 或 `probability`。

## 完整流程

先生成完整候选矩阵：

```bash
school-zone-matrix prepare \
  --communities data/communities.csv \
  --schools data/schools.csv \
  --descriptions data/descriptions.csv \
  --extra-features data/extra_features.csv \
  --output data/candidates.csv
```

科研评价分成两个命令。第一步会读取训练折标签，但在任何特征构造和校准前清空整折测试标签，只生成并冻结预测：

```bash
school-zone-matrix crossfit-predict \
  --candidates data/candidates.csv \
  --labels data/labels.csv \
  --output-dir runs/district_2026
```

预测冻结后，第二步才评分：

```bash
school-zone-matrix evaluate \
  --predictions runs/district_2026/predictions.csv \
  --labels data/labels.csv \
  --output-dir runs/district_2026/evaluation
```

在正式冻结预测前，可先只读检查 Gold 是否存在“整所学校无正例”、校园尺度 Polygon 或楼栋规则与小区中心点粒度冲突：

```bash
school-zone-matrix audit-labels \
  --labels data/labels.csv \
  --schools data/schools.csv \
  --descriptions data/descriptions.csv \
  --geometry-stats data/school_geometry_stats.csv \
  --output-dir runs/district_2026/gold_audit
```

该命令只生成审核报告，不修改标签。用法和判定边界见 [Gold可观测性审计](docs/gold_observability.md)。

生产模型使用全部可靠训练关系：

```bash
school-zone-matrix train-release \
  --candidates data/candidates.csv \
  --labels data/labels.csv \
  --model models/district_2026.joblib
```

为新小区重新执行 `prepare` 后预测：

```bash
school-zone-matrix predict \
  --model models/district_2026.joblib \
  --candidates data/new_candidates.csv \
  --output-dir outputs/new_prediction
```

输出包括：

- `relations.csv`：每条“小区—学校”关系的概率和0/1结果；
- `probability_matrix.csv`：学校为行、小区为列的概率矩阵；
- `relation_matrix.csv`：学校为行、小区为列的0/1矩阵。

## 示例与测试

```bash
python scripts/make_example_data.py
python -m pytest
```

合成数据会写入 `examples/synthetic/`，不包含真实小区、学校或 Gold。
测试配置关闭了 pytest 的临时目录插件，以避开部分 Windows 目录的访问权限问题；本项目测试不依赖该插件。

## 科研边界

官方学区地图及其派生关系只能用于冻结后的评价，不能参与候选生成、训练折之外的锚点构造、阈值选择或错误修复。使用历史或当年训练标签属于监督学习，应与零标签方法分开报告。完整规则见 [Gold隔离协议](docs/research_protocol.md)。
