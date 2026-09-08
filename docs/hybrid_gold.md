# 官方住宅字段优先的 Hybrid Gold

同一所学校的官方地图记录可能同时提供学区 Polygon 和住宅字段。两者冲突时，已经完成实体消歧的官方住宅点名是硬正例：关系标签为1，Polygon、中心点距离和模型结果都不能把它改成0。

Polygon仍可补充住宅字段没有逐个列出的关系。构建顺序为：

1. 从质量合格的基础空间Gold保留正关系；
2. 加入已经解析并完成实体消歧的官方住宅正关系；
3. 对冲突边以官方住宅正关系覆盖基础0或未知状态；
4. 只将至少具有一个正关系的小区纳入完整二值Gold；
5. 对纳入的小区，以全部正关系的补集生成0；没有任何正证据的小区整体排除，不生成全0行。

输入的 `official_positive_edges.csv` 至少包含 `community_id,school_id`，可增加 `evidence` 字段。该文件必须由官方住宅字段完成实体链接后生成；命令不会把任意字符串子串自动当成可靠实体。

```bash
school-zone-matrix build-hybrid-gold \
  --base-labels data/polygon_labels.csv \
  --official-positive-edges data/official_residence_positive_edges.csv \
  --output-dir runs/district_2026/hybrid_gold
```

输出包括完整二值标签、正关系证据、被官方点名覆盖的旧0/未知关系、排除小区和汇总报告。新版Gold应在下一次实验前冻结；看过旧实验错误后构建的版本不能用于追溯包装旧预测的盲测成绩。
