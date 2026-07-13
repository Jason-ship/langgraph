# NovelFactory — 评分系统 (v6.3 VerdictEngine)

## VerdictEngine 调用顺序

```
1. 程序化分析（纯代码，毫秒级）
   ├── ai_style_sensor: 8维AI味检测 (N-gram/句长波动/词汇多样性/模板化/标点节奏/对白比例/感官词/语义平滑)
   ├── old_reader_sensor: 老书虫视角评分
   └── cross_chapter_sensor: 跨章一致性检测 (角色声音/文风/节奏/情节连贯/伏笔)
2. 知情辩论（LLM，注入程序化结果）
   ├── editor_review → reader_review → debate_router → 单轮收敛
   └── 产出: DebateReport (issues/strengths/severity_weight/transcript)
3. 四维 LLM 评分（1次）
   ├── 文学性(30) + 结构(25) + 角色(20) + 节奏(15) = 100
   └── 含跨章一致性维度(0-100)
4. 融合计算 → 校准 → 决议
```

## 融合公式

```
final_score = quality_score × 0.40
            + programmatic_normalized × 0.30
            + cross_chapter_consistency × 0.20
            - debate_penalty × 0.10
```

## 三路决策

| final_score | level | 路由 | 说明 |
|-------------|-------|------|------|
| ≥75 | PASS | __exit_for_chapter__ | 通过 |
| ≥55 | REFINE | chapter_refiner | 需润色 |
| <55 | REWRITE | chapter_writer | 重写 |
| 严重毒点 + 未用尽重写 | REWRITE | chapter_writer | 毒点强制重写 |
| 次数用尽 | PASS | __exit_for_chapter__ | 防死循环 |

## 题材阈值 (GENRE_THRESHOLDS)

| 题材 | quality 阈值 | composite 阈值 | AI味阈值 | 特点 |
|------|:-----------:|:--------------:|:--------:|------|
| 玄幻 | 85 | 0.65 | 0.30 | 设定严谨性 |
| 仙侠 | 88 | 0.65 | 0.25 | 文风古韵最严格 |
| 都市 | 80 | 0.55 | 0.35 | 节奏快爽点密 |
| 系统流 | 75 | 0.50 | 0.50 | 固定提示语极多 |
| 无敌流 | 72 | 0.45 | 0.55 | 最不需要文学性 |
| 爽文 | 75 | 0.50 | 0.50 | 文学性要求最低 |
| 悬疑灵异 | 85 | 0.60 | 0.30 | 气氛悬念最重要 |
| default | 85 | 0.65 | 0.30 | 通用标准 |

## 校准 (CalibrationModule)

- LLM `quality_score` 虚高 >90 → 降低权重
- 程序化分过低 <0.5 → 降低程序化权重
- 短文本 → LLM 权重提升至 0.8
- 严重毒点 → 分数封顶 50
