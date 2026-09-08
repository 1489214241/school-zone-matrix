# Gold 可观测性审计

学校学区 Polygon 与住宅中心点都来自官方或可信数据，也不等于每条“小区—学校”关系都能被可靠判断。常见的两种问题是：

- 地图返回的 Polygon 只有校园或一栋建筑大小，而官网文字列出多个外部住宅；
- 官网按楼栋、座号或部队范围划分，但住宅目录只有整个小区的一个中心点。

这两种情况下，中心点在 Polygon 外不能直接证明关系为负。先运行只读审计：

```bash
school-zone-matrix audit-labels \
  --labels data/labels.csv \
  --schools data/schools.csv \
  --descriptions data/descriptions.csv \
  --geometry-stats data/school_geometry_stats.csv \
  --output-dir runs/district_2026/gold_audit
```

`school_geometry_stats.csv` 至少包含 `school_id,polygon_area_m2`。面积阈值只是筛查参数，不能单独作为删标签依据。

输出会列出每所学校的正、负、U数量，以及小 Polygon、楼栋规则和零正例提示。命令不修改 `labels.csv`。研究者应回到官方来源核验：无法由当前数据粒度判断的关系记 U；取得楼栋坐标或官方房屋编码后再生成楼栋级 0/1。

可观测性规则和掩码必须在预测前冻结。看过测试误差后做的调整只能作为事后敏感性分析，不能替换主实验成绩。
