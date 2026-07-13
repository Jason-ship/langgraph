# 写作指南知识库使用指南

## 一、系统架构

```
写作指南知识库
     │
     ├─ PostgreSQL: writing_guides 表（结构化数据）
     └─ Milvus: writing_guides Collection（向量检索）
           └─ Embedding: Qwen3-Embedding（复用现有配置）
```

---

## 二、数据格式规范

每条写作指南必须包含以下字段：

| 字段 | 类型 | 必须 | 说明 |
|------|------|------|------|
| `title` | string | ✅ | 指南标题，如"打脸爽点的三层递进写法" |
| `content` | string | ✅ | 指南正文内容，建议 200-800 字 |
| `source` | string | ✅ | 来源：`manual`（手动）/ `ai_analysis` / `template` |
| `source_url` | string | ❌ | 原文链接（如有） |
| `tags` | string[] | ✅ | 标签数组，用于分类检索 |
| `genre` | string | ❌ | 题材：`都市` / `玄幻` / `科幻` / `轻小说` |
| `quality_score` | float | ✅ | 质量评分 0-1，默认 0.5，≥0.7 才参与检索 |
| `chapter_ref` | string | ❌ | 参考章节（如来源是小说分析） |
| `guide_type` | string | ✅ | 类型：`technique`（写作技巧）/ `analysis`（小说分析） |

---

## 三、标签体系

### 3.1 按功能分类

| 标签 | 说明 | 适用场景 |
|------|------|----------|
| `爽点` | 爽点设计与节奏 | 打脸/装逼/逆袭/升级章节 |
| `打脸` | 打脸场景写法 | 反派嚣张→主角爆发→众人震惊 |
| `装逼` | 装逼场景写法 | 隐藏实力→不经意展露→旁人震惊 |
| `逆袭` | 逆境反转写法 | 低谷→反转→高潮 |
| `升级` | 修炼突破写法 | 瓶颈→顿悟→力量可视化 |
| `感情` | 感情戏写法 | 误会→解结→升华 |
| `悬念` | 章节钩子写法 | 结尾留悬念 |
| `人物` | 人物塑造技巧 | 对话性格化/反派设计 |
| `对话` | 对话写法 | 人物语言差异化 |
| `反派` | 反派塑造 | 让读者期待打脸 |
| `文笔` | 文笔表达 | 感官描写/画面感 |
| `感官` | 五感描写 | 场景细节 |
| `节奏` | 节奏控制 | 爽点密度/情绪曲线 |
| `情绪` | 情绪节奏 | 情绪曲线设计 |
| `毒点` | 避坑指南 | 毒点规避 |
| `避坑` | 常见错误 | 圣母/战力崩坏等 |
| `开篇` | 开篇写法 | 第一章/黄金三章 |
| `章节结构` | 章节内部结构 | 段落安排 |
| `写作技巧` | 综合技巧 | 综合类 |
| `写作指南` | 综合指南 | 综合类 |

### 3.2 按题材分类

| 题材 | 说明 |
|------|------|
| `都市` | 都市言情/都市修真 |
| `玄幻` | 玄幻奇幻 |
| `科幻` | 科幻未来 |
| `轻小说` | 轻小说/二次元 |
| `null` | 通用（不区分题材） |

---

## 四、导入文档格式（飞书 Markdown）

飞书文档导入时，按以下格式组织内容：

```markdown
# 指南标题

> **标签**: 爽点, 打脸, 都市
> **题材**: 都市
> **质量评分**: 0.8
> **来源**: manual
> **类型**: technique

## 正文内容

这里是写作指南的正文内容...

建议按照以下结构组织：
1. 核心概念
2. 具体写法/技巧
3. 示例片段
4. 注意事项
```

### 导入脚本示例

```python
from novelfactory.state.writing_guide_store import get_guide_store

store = get_guide_store()

# 添加一条指南
guide_id = store.add_guide(
    title="打脸爽点的三层递进写法",
    content="""打脸是网文最核心的爽点类型之一...

写法要点：
1. 铺垫仇恨值：反派先嚣张...
2. 制造反差期待：主角表面示弱...
3. 爆发打脸：主角一鸣惊人...

每层至少 200 字铺垫，爆发要干脆利落。""",
    source="manual",
    tags=["爽点", "打脸", "都市"],
    genre="都市",
    quality_score=0.8,
    guide_type="technique",
)

print(f"已添加指南: {guide_id}")
```

---

## 五、检索接口

### 5.1 基础检索

```python
from novelfactory.state.writing_guide_store import get_guide_store

store = get_guide_store()

results = store.search(
    query="打脸爽点写法",
    top_k=5,
    genre="都市",
    min_quality=0.7,
)

for guide in results["guides"]:
    print(f"标题: {guide['title']}")
    print(f"内容: {guide['content'][:200]}...")
    print(f"标签: {guide['tags']}")
    print("---")
```

### 5.2 按标签检索

```python
results = store.search(
    query="",
    top_k=10,
    tags=["爽点", "节奏"],
    min_quality=0.7,
)
```

---

## 六、评分阈值

| 字段 | 通过线 | 说明 |
|------|--------|------|
| `quality_score` | ≥ 0.7 | 低于此值不参与检索 |
| AI味指数 | ≤ 0.3 | AI味合格线 |
| 老书虫分 | ≥ 70 | 老书虫视角合格线 |
| 综合指标 | ≥ 0.7 | 老书虫分/100 × (1 − AI味指数) |

---

## 七、数据库维护

### 7.1 初始化数据库

```bash
# 本地
cd d:/NovelFactory/novelfactory/src/novelfactory
python -m scripts.migrate_writing_guides_db

# 服务器
cd /home/jason/novelfactory
python -m scripts.migrate_writing_guides_db
```

### 7.2 查看指南总数

```python
from novelfactory.state.writing_guide_store import get_guide_store

store = get_guide_store()
print(f"当前指南总数: {store.count()}")
```

### 7.3 批量导入（从 JSON）

```python
import json
from novelfactory.state.writing_guide_store import get_guide_store

store = get_guide_store()

with open("guides.json", "r", encoding="utf-8") as f:
    guides = json.load(f)

for guide in guides:
    store.add_guide(**guide)

print(f"导入完成，当前总数: {store.count()}")
```

---

## 八、飞书文档抓取规范

### 8.1 抓取流程

1. 在飞书文档中按上述格式撰写写作指南
2. 导出为 Markdown
3. 运行导入脚本解析并入库

### 8.2 质量要求

- 每条指南至少 200 字
- 内容需要有具体的写作建议，而非泛泛的理论
- 最好包含可操作的技巧或示例
- 标签需从上述标签体系中选择
- 题材需准确填写

### 8.3 禁止内容

- 涉及抄袭/洗稿的内容
- 与平台规则冲突的内容
- 过于主观、无实际参考价值的评论

---

## 九、FAQ

**Q: 指南质量评分如何确定？**
A: 初始默认为 0.5，由运营根据内容质量调整。高于 0.7 的才会被检索使用。

**Q: 同一类型的指南可以有多条吗？**
A: 可以，且建议多条从不同角度覆盖同一类型。

**Q: 如何删除低质量指南？**
A: 直接从 PostgreSQL 删除即可：`DELETE FROM writing_guides WHERE quality_score < 0.5;`
```

---

## 十、集成到写作流程

写作时，指南知识库**只在评审阶段被调用**：

```
ChapterWriter（自由创作）→ ChapterReviewer（四维评审）
                                 ↓
                          AI味检测（8维统计）
                                 ↓
                          老书虫评审（毒点+爽点）
                                 ↓
                          指南检索（根据问题类型）
                                 ↓
                          综合评分 → 通过/不通过
```

Refiner 会收到：
- `ai_style_fix`：AI味方面的具体建议
- `lao_shu_chong_fix`：老书虫视角的具体建议
- `guide_references`：检索到的相关指南（最多 3 条）
- `toxic_points`：检测到的毒点
- `shuangdian_points`：检测到的爽点
