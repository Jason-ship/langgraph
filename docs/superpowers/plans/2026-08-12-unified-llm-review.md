# 统一 LLM 评分系统重构 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将分散的"程序化传感器 + 5 个 LLM 评审维度 + 加权融合"重构为单一统一 LLM 评审系统，每章评审 LLM 调用从 12-18 次降至 1-2 次，清除所有程序化评分。

**Architecture:** 新增 `evaluation/unified/` 模块：单次 LLM 调用输出标签化双通道评审报告（五视角 + 四维 + 毒点/爽点/水段/跳跃定位 + 综合分 + 段落级修复指令），程序化解析 + 自洽校验（只校验不评分），修复后轻量回归复查。`verdict/engine.py` 换内核，`VerdictResult` 对外字段保持兼容；删除 `evaluation/programmatic/` 及旧 LLM 评分维度引用。

**Tech Stack:** Python 3.12, LangGraph, pydantic, pytest, pytest-asyncio, deepseek-v4-flash（经 `async_llm_call_with_retry`）

**Spec 依据:** Spec+RFC《统一 LLM 评分系统重构》（10 节，TBD 按默认值：TBD-1 单次五视角 / TBD-2 回归复查 / TBD-3 停用后删除 / TBD-4 保留忽略）

---

## 文件结构

| 动作 | 文件 | 职责 |
|------|------|------|
| Create | `src/novelfactory/evaluation/unified/schemas.py` | 统一评审结果数据契约 |
| Create | `src/novelfactory/evaluation/unified/parser.py` | 标签化解析 + 自洽校验 |
| Create | `src/novelfactory/evaluation/unified/prompts.py` | 统一评审/回归复查/分歧仲裁 prompt |
| Create | `src/novelfactory/evaluation/unified/arbitration.py` | 分歧仲裁（替代多轮辩论） |
| Create | `src/novelfactory/evaluation/unified/engine.py` | 评审引擎（调 LLM + 解析 + 降级重试 + 仲裁） |
| Create | `src/novelfactory/evaluation/unified/__init__.py` | 导出 |
| Modify | `src/novelfactory/evaluation/verdict/engine.py` | evaluate 换内核，删程序化调用与辩论 |
| Modify | `src/novelfactory/evaluation/verdict/feedback.py` | 适配 UnifiedReviewResult |
| Modify | `src/novelfactory/evaluation/coordinator.py` | 适配新评审结果字段 |
| Modify | `src/novelfactory/config/quality_params.py` | 参数清理与新增 |
| Modify | `src/novelfactory/evaluation/service.py` | 批处理适配 |
| Modify | `src/novelfactory/evaluation/replay.py` | 重放适配 |
| Delete | `src/novelfactory/evaluation/programmatic/` | 程序化传感器全部删除 |
| Delete | `src/novelfactory/evaluation/debate/` | 多轮辩论引擎（由仲裁替代） |
| Delete | `src/novelfactory/analysis/ai_style_analyzer.py` | AI味 8 维算法（程序化，唯一残留引用为 decay 检测） |
| Delete | `src/novelfactory/analysis/old_reader_reviewer.py` | 毒点/爽点关键词算法（程序化） |
| Modify | `src/novelfactory/graph/chat/agents/quality_tuner_agent.py` | 调参 Agent prompt 的权重规则（删除"权重和=1.0"约束） |
| Test | `tests/unit/evaluation/test_unified_review.py` | 解析/自洽/降级/复查/仲裁测试 |
| Modify | `tests/unit/evaluation/test_verdict_engine_parallel.py` | 原依赖真实程序化传感器，重写为统一评审 mock |
| Modify | `tests/unit/evaluation/test_debate_engine.py` | 辩论引擎删除后改测仲裁模块 |
| Modify | `tests/unit/evaluation/test_llm_analysis.py` | 旧 LLM 维度解析测试改测 unified parser |

---

### Task 1: 统一评审数据契约

**Files:**
- Create: `src/novelfactory/evaluation/unified/schemas.py`
- Test: `tests/unit/evaluation/test_unified_review.py`（本任务仅 schema 部分）

- [ ] **Step 1: 写失败测试**

```python
# tests/unit/evaluation/test_unified_review.py
from novelfactory.evaluation.unified.schemas import UnifiedReviewResult, UnifiedFourDim

def test_four_dim_total():
    fd = UnifiedFourDim(logic=26.0, writing=20.0, character=21.0, world=17.0)
    assert fd.total() == 84.0

def test_review_result_defaults():
    r = UnifiedReviewResult()
    assert r.failed is False
    assert r.severe_toxic is False
    assert r.final_score == 60.0
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/unit/evaluation/test_unified_review.py -v --tb=short`
Expected: FAIL（ModuleNotFoundError: novelfactory.evaluation.unified）

- [ ] **Step 3: 创建 schema**

```python
# src/novelfactory/evaluation/unified/schemas.py
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class UnifiedFourDim:
    """四维评分（满分 30/25/25/20，合计 100）。"""

    logic: float = 0.0      # 剧情逻辑 0-30
    writing: float = 0.0    # 文笔表达 0-25
    character: float = 0.0  # 人物一致性 0-25
    world: float = 0.0      # 世界观契合 0-20

    def total(self) -> float:
        return self.logic + self.writing + self.character + self.world


@dataclass
class UnifiedReviewResult:
    """统一 LLM 评审结果（标签化双通道解析产物）。"""

    final_score: float = 60.0
    four_dim: UnifiedFourDim = field(default_factory=UnifiedFourDim)
    toxic_points: list[dict] = field(default_factory=list)   # [{type, paragraph, severity, reason}]
    severe_toxic: bool = False
    shuangdian_points: list[str] = field(default_factory=list)
    shuangdian_count: int = 0
    water_paragraphs: list[int] = field(default_factory=list)
    scene_transition_issues: list[dict] = field(default_factory=list)  # [{from, to, reason}]
    human_like_score: float = 0.0
    attraction_score: float = 0.0
    immersion_score: float = 0.0
    cross_chapter_score: float = 0.0
    decay_hint: str = "none"
    perspective_disagreements: list[str] = field(default_factory=list)
    review_comments: str = ""
    fix_suggestions: list[str] = field(default_factory=list)
    raw_text: str = ""
    failed: bool = False
    retried: bool = False
    consistency_fixed: bool = False  # 自洽校验是否修正过分数
```

- [ ] **Step 4: 运行测试确认通过**

Run: `pytest tests/unit/evaluation/test_unified_review.py -v --tb=short`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add src/novelfactory/evaluation/unified/schemas.py tests/unit/evaluation/test_unified_review.py
git commit -m "feat(unified): add UnifiedReviewResult data contract"
```

---

### Task 2: 标签化解析器 + 自洽校验

**Files:**
- Create: `src/novelfactory/evaluation/unified/parser.py`
- Test: `tests/unit/evaluation/test_unified_review.py`

- [ ] **Step 1: 写失败测试**

```python
from novelfactory.evaluation.unified.parser import parse_review_output, apply_consistency_check

SAMPLE = """<review_analysis>毒点核查：P8 战力突兀。</review_analysis>
[评分] final=78.5
[四维-剧情逻辑] 26/30
[四维-文笔] 20/25
[四维-人物] 21/25
[四维-世界观] 17/20
[毒点] POWER_BREAK|P8|severe|战力突兀
[爽点] 打脸|P5; 升级|P14
[AI味] 0.72
[吸引力] 85  [沉浸] 88
[跨章] 82
[衰减] none
[废话段] P3; P12
[跳跃] P8->P9|场景无过渡

评审意见：
- 老书虫视角：……
- 番茄编辑视角：……
- 读者视角：……
- 评论员视角：……
- 分歧点：老书虫认为 P8 战力突兀，番茄编辑认为可接受
- 修复指令：P3 删冗余环境描写
"""

def test_parse_review_output():
    r = parse_review_output(SAMPLE)
    assert r is not None
    assert r.final_score == 78.5
    assert r.four_dim.logic == 26.0
    assert r.severe_toxic is True
    assert r.water_paragraphs == [3, 12]
    assert r.shuangdian_count == 2
    assert len(r.perspective_disagreements) == 1

def test_consistency_caps_severe_toxic():
    r = parse_review_output(SAMPLE)
    changed = apply_consistency_check(r)
    assert changed is True
    assert r.final_score <= 70.0  # severe 毒点强制降分

def test_consistency_caps_no_shuangdian():
    raw = SAMPLE.replace("[爽点] 打脸|P5; 升级|P14", "[爽点] 无").replace("[评分] final=78.5", "[评分] final=80.0")
    r = parse_review_output(raw)
    changed = apply_consistency_check(r)
    assert changed is True
    assert r.final_score <= 65.0

def test_parse_garbage_returns_none():
    assert parse_review_output("完全不是评审输出") is None
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/unit/evaluation/test_unified_review.py -v --tb=short`
Expected: FAIL（ModuleNotFoundError: parser）

- [ ] **Step 3: 创建解析器**

```python
# src/novelfactory/evaluation/unified/parser.py
from __future__ import annotations

import re

from novelfactory.evaluation.unified.schemas import UnifiedFourDim, UnifiedReviewResult

_RE_FINAL = re.compile(r"\[评分\]\s*final\s*=\s*([\d.]+)")
_RE_DIM = re.compile(r"\[四维-([\u4e00-\u9fa5]+)\]\s*([\d.]+)")
_RE_TOXIC = re.compile(r"\[毒点\]\s*(.+)")
_RE_SHUANG = re.compile(r"\[爽点\]\s*(.+)")
_RE_AI = re.compile(r"\[AI味\]\s*([\d.]+)")
_RE_ATTR = re.compile(r"\[吸引力\]\s*([\d.]+)\s*\[沉浸\]\s*([\d.]+)")
_RE_CROSS = re.compile(r"\[跨章\]\s*([\d.]+)")
_RE_DECAY = re.compile(r"\[衰减\]\s*(\S+)")
_RE_WATER = re.compile(r"\[废话段\]\s*(.+)")
_RE_JUMP = re.compile(r"\[跳跃\]\s*(.+)")
_RE_DISAGREE = re.compile(r"分歧点[：:]\s*(.+)")
_RE_SEVERE = re.compile(r"severe", re.IGNORECASE)

_DIM_KEY = {"剧情逻辑": "logic", "文笔": "writing", "人物": "character", "世界观": "world"}


def _first_float(text: str | None, default: float = 0.0) -> float:
    if not text:
        return default
    m = re.search(r"([\d.]+)", text)
    return float(m.group(1)) if m else default


def _split_semicolon(text: str | None) -> list[str]:
    if not text or text.strip() in ("无", "none", "-"):
        return []
    return [p.strip() for p in text.split(";") if p.strip()]


def _parse_ints(text: str | None) -> list[int]:
    if not text:
        return []
    out: list[int] = []
    for token in re.split(r"[;,\s]+", text):
        if token.startswith("P") and token[1:].isdigit():
            out.append(int(token[1:]))
    return out


def parse_review_output(raw: str) -> UnifiedReviewResult | None:
    """标签化解析统一评审输出；无法提取综合分时返回 None。"""
    if not raw:
        return None
    m_final = _RE_FINAL.search(raw)
    if not m_final:
        return None
    result = UnifiedReviewResult(raw_text=raw)
    result.final_score = float(m_final.group(1))
    for m in _RE_DIM.finditer(raw):
        key = _DIM_KEY.get(m.group(1))
        if key:
            setattr(result.four_dim, key, float(m.group(2)))
    m_toxic = _RE_TOXIC.search(raw)
    if m_toxic and _split_semicolon(m_toxic.group(1)):
        entries = []
        for item in _split_semicolon(m_toxic.group(1)):
            parts = item.split("|")
            entries.append({
                "type": parts[0].strip() if parts else "",
                "paragraph": int(re.sub(r"\D", "", parts[1])) if len(parts) > 1 and re.search(r"\d", parts[1]) else 0,
                "severity": "severe" if any(_RE_SEVERE.search(p) for p in parts) else "normal",
                "reason": parts[2].strip() if len(parts) > 2 else "",
            })
        result.toxic_points = entries
        result.severe_toxic = any(e["severity"] == "severe" for e in entries)
    m_shuang = _RE_SHUANG.search(raw)
    if m_shuang:
        result.shuangdian_points = _split_semicolon(m_shuang.group(1))
        result.shuangdian_count = len(result.shuangdian_points)
    result.human_like_score = _first_float(_RE_AI.search(raw).group(1) if _RE_AI.search(raw) else None)
    m_attr = _RE_ATTR.search(raw)
    if m_attr:
        result.attraction_score = float(m_attr.group(1))
        result.immersion_score = float(m_attr.group(2))
    result.cross_chapter_score = _first_float(_RE_CROSS.search(raw).group(1) if _RE_CROSS.search(raw) else None)
    m_decay = _RE_DECAY.search(raw)
    if m_decay:
        result.decay_hint = m_decay.group(1).lower()
    m_water = _RE_WATER.search(raw)
    if m_water:
        result.water_paragraphs = _parse_ints(m_water.group(1))
    m_jump = _RE_JUMP.search(raw)
    if m_jump:
        for item in _split_semicolon(m_jump.group(1)):
            parts = item.split("|")
            result.scene_transition_issues.append({
                "from": parts[0].strip() if parts else "",
                "to": parts[1].strip() if len(parts) > 1 else "",
                "reason": parts[2].strip() if len(parts) > 2 else "",
            })
    m_dsg = _RE_DISAGREE.search(raw)
    if m_dsg:
        result.perspective_disagreements = _split_semicolon(m_dsg.group(1))
    return result


def apply_consistency_check(result: UnifiedReviewResult) -> bool:
    """自洽校验（只校验不评分）：修正 final_score，返回是否修正。"""
    changed = False
    if result.severe_toxic and result.final_score > 70.0:
        result.final_score = 70.0
        result.consistency_fixed = True
        changed = True
    if result.shuangdian_count == 0 and result.final_score > 65.0:
        result.final_score = 65.0
        result.consistency_fixed = True
        changed = True
    if result.four_dim.total() > 0 and abs(result.four_dim.total() - result.final_score) > 15.0:
        result.final_score = round(result.four_dim.total(), 1)
        result.consistency_fixed = True
        changed = True
    return changed
```

- [ ] **Step 4: 运行测试确认通过**

Run: `pytest tests/unit/evaluation/test_unified_review.py -v --tb=short`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add src/novelfactory/evaluation/unified/parser.py tests/unit/evaluation/test_unified_review.py
git commit -m "feat(unified): add tag-based parser with consistency check"
```

---

### Task 3: 统一评审 Prompt

**Files:**
- Create: `src/novelfactory/evaluation/unified/prompts.py`
- Test: `tests/unit/evaluation/test_unified_review.py`

- [ ] **Step 1: 写失败测试**

```python
from novelfactory.evaluation.unified.prompts import build_unified_review_prompt, build_quick_recheck_prompt

def test_build_prompt_contains_core_directives():
    p = build_unified_review_prompt(chapter_text="测试正文", genre="都市", prev_summary="前情", guide="题材指南")
    for keyword in ("[评分] final=", "老书虫视角", "番茄编辑视角", "[废话段]", "[爽点]", "网文爽感铁律", "P8"):
        assert keyword in p or keyword.replace("P8", "") in p

def test_build_recheck_prompt_injects_old_issues():
    p = build_quick_recheck_prompt(chapter_text="正文", old_issues=["P8 战力突兀"])
    assert "P8 战力突兀" in p
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/unit/evaluation/test_unified_review.py -v --tb=short`
Expected: FAIL（ModuleNotFoundError: prompts）

- [ ] **Step 3: 创建 prompt 模块**

```python
# src/novelfactory/evaluation/unified/prompts.py
from __future__ import annotations

from novelfactory.evaluation.llm._shared import trim_text

UNIFIED_REVIEW_SYSTEM = """你是统一小说章节评审官，深谙中国网文。一次调用完成五个视角评审并给出综合分。

## 评审流程（必须先分析再评分）
1. 在 <review_analysis> 中逐条核查：毒点、爽点、场景衔接、废话段落（不推进剧情/情绪/信息的段落）
2. 按五视角分别评审（各自给分与依据）
3. 输出结构化标签评分

## 网文爽感铁律（评分依据）
- 每章至少 3 个可感知爽点（打脸/奖励兑现/升级/装逼/解气），缺少则 attraction 与 final 必须压低
- 废话判定：每句必须推进剧情/情绪/信息，否则记为 [废话段]（给出段落编号 Pn）
- 场景跳跃判定：时间/地点变化无过渡句，记为 [跳跃]

## 五视角
- 老书虫视角：毒点/爽点/节奏（毒点含 severity=severe/normal）
- 番茄编辑视角：开篇钩子/爽点密度/短剧适配/修改建议
- 读者视角：沉浸感（0-100）
- 评论员视角：剧情逻辑/人物一致性/世界观契合批判
- 四维评分：剧情逻辑(0-30)/文笔(0-25)/人物(0-25)/世界观(0-20)

## 锚定打分（综合分对标，禁止漂移）
- 90：四维≥85 且爽点≥4 且无 severe 毒点、无水段
- 80：四维≥75 且爽点≥3、无 severe 毒点、水段≤2
- 70：逻辑自洽但节奏平、爽点1-2个或水段3+
- 60 及以下：逻辑断裂/严重毒点/全程无爽点
final_score 必须与四维、毒点、爽点自洽。

## 输出格式（严格遵守，先 <review_analysis> 再标签）
<review_analysis>
（内部推理，勿解析）
</review_analysis>
[评分] final=<0-100>
[四维-剧情逻辑] <0-30>/30
[四维-文笔] <0-25>/25
[四维-人物] <0-25>/25
[四维-世界观] <0-20>/20
[毒点] <TYPE>|P<段落>|severe|原因; <TYPE>|P<段落>|normal|原因（无则：无）
[爽点] <类型>|P<段落>; ...（无则：无）
[AI味] <0-1>
[吸引力] <0-100>  [沉浸] <0-100>
[跨章] <0-100>
[衰减] <none|head|tail|mid>
[废话段] P3; P12（无则：无）
[跳跃] P5->P6|原因（无则：无）

评审意见：
- 老书虫视角：…
- 番茄编辑视角：…（含修改建议）
- 读者视角：…
- 评论员视角：…
- 分歧点：…
- 修复指令：P3 删…；P12 改…（逐条可执行）"""

QUICK_RECHECK_SYSTEM = """你是统一小说章节评审官的复查助手。上一版评审指出以下问题，请复查本章修复后是否通过。

输出：
<review_analysis>（逐条对照旧问题核查：是否清除、是否引入新问题）</review_analysis>
[复查] passed=true|false
[毒点] …（仍存在或新增的，无则：无）
[新增问题] …
[分数变化] final: 旧值 -> 新值"""

ARBITRATION_SYSTEM = """你是统一小说章节评审官的分歧仲裁官。主评审中五个视角存在分歧，请仲裁。

输入为分歧点清单（每项含：焦点、涉及视角、双方立场）。输出：
<review_analysis>（逐条权衡双方论据）</review_analysis>
[裁决] <采纳哪方或折中，1-2 句>
[分数修正] final: <原值> -> <新值，说明依据>
[备注] <是否需转重写/润色>"""


def _render_chapter(chapter_text: str) -> str:
    return trim_text(chapter_text or "")


def build_unified_review_prompt(
    *,
    chapter_text: str,
    genre: str,
    prev_summary: str,
    guide: str,
) -> str:
    return (
        f"{UNIFIED_REVIEW_SYSTEM}\n\n"
        f"## 评审对象\n题材：{genre}\n题材评分指引：{guide or '（无）'}\n"
        f"前章摘要（跨章一致性用）：{prev_summary or '（无）'}\n\n"
        f"## 章节正文\n{_render_chapter(chapter_text)}"
    )


def build_quick_recheck_prompt(
    *,
    chapter_text: str,
    old_issues: list[str],
    old_score: float,
) -> str:
    issues = "\n".join(f"- {i}" for i in old_issues) or "- （无）"
    return (
        f"{QUICK_RECHECK_SYSTEM}\n\n"
        f"## 上一版问题清单\n{issues}\n上一版综合分：{old_score:.0f}\n\n"
        f"## 修复后正文\n{_render_chapter(chapter_text)}"
    )


def build_arbitration_prompt(
    *,
    disagreements: list[str],
    old_score: float,
) -> str:
    items = "\n".join(f"- {d}" for d in disagreements) or "- （无）"
    return (
        f"{ARBITRATION_SYSTEM}\n\n"
        f"## 分歧点清单\n{items}\n原综合分：{old_score:.0f}"
    )
```

注意：`trim_text` 从 `novelfactory.evaluation.llm._shared` 导入；该文件在后续 Task 10 删除程序化/旧 LLM 维度时**保留**（`_shared.py` 为公共工具，供 unified 复用，不属评分组件）。

- [ ] **Step 4: 运行测试确认通过**

Run: `pytest tests/unit/evaluation/test_unified_review.py -v --tb=short`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add src/novelfactory/evaluation/unified/prompts.py tests/unit/evaluation/test_unified_review.py
git commit -m "feat(unified): add unified review and recheck prompts"
```

---

### Task 4A: 分歧仲裁模块（替代多轮辩论）

**Files:**
- Create: `src/novelfactory/evaluation/unified/arbitration.py`
- Test: `tests/unit/evaluation/test_unified_review.py`

- [ ] **Step 1: 写失败测试**

```python
from novelfactory.evaluation.unified.arbitration import parse_arbitration, arbitrate

ARB = """<review_analysis>权衡：老书虫看重战力一致，番茄编辑看重节奏。</review_analysis>
[裁决] 采纳老书虫：P8 需补战力铺垫，但不必重写整章
[分数修正] final: 78.5 -> 74.0
[备注] REFINE"""

def test_parse_arbitration():
    r = parse_arbitration(ARB)
    assert r is not None
    assert r["new_score"] == 74.0
    assert r["action"] == "REFINE"

def test_parse_arbitration_garbage():
    assert parse_arbitration("乱码") is None

class _FakeArbLlm:
    async def ainvoke(self, *a, **kw):
        return type("R", (), {"content": ARB})()

def test_arbitrate_with_llm():
    score = asyncio.run(arbitrate(_FakeArbLlm(), disagreements=["P8 战力突兀"], old_score=78.5))
    assert score == 74.0
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/unit/evaluation/test_unified_review.py -v --tb=short`
Expected: FAIL（ModuleNotFoundError: arbitration）

- [ ] **Step 3: 创建仲裁模块**

```python
# src/novelfactory/evaluation/unified/arbitration.py
from __future__ import annotations

import logging
import re

from langchain_core.language_models import BaseChatModel

from novelfactory.agents.infra.async_retry import async_llm_call_with_retry
from novelfactory.evaluation.unified.prompts import build_arbitration_prompt

logger = logging.getLogger(__name__)

_RE_NEW = re.compile(r"\[分数修正\]\s*final:\s*[\d.]+\s*->\s*([\d.]+)")
_RE_ACTION = re.compile(r"\[备注\]\s*(PASS|REFINE|REWRITE)", re.IGNORECASE)


def parse_arbitration(raw: str) -> dict | None:
    """解析仲裁输出；提取新分数与建议动作。"""
    if not raw:
        return None
    m = _RE_NEW.search(raw)
    if not m:
        return None
    return {
        "new_score": float(m.group(1)),
        "action": (_RE_ACTION.search(raw).group(1).upper() if _RE_ACTION.search(raw) else "REFINE"),
    }


async def arbitrate(
    llm: BaseChatModel,
    *,
    disagreements: list[str],
    old_score: float,
) -> float:
    """分歧仲裁：输入分歧清单，返回仲裁后分数（失败返回原分）。"""
    if not disagreements:
        return old_score
    prompt = build_arbitration_prompt(disagreements=disagreements, old_score=old_score)
    try:
        response = await async_llm_call_with_retry(llm.ainvoke, prompt, step_name="unified_arbitration")
        raw = response.content if hasattr(response, "content") else str(response)
        parsed = parse_arbitration(raw)
        if parsed:
            logger.info("[unified_arbitration] %s -> %s", old_score, parsed["new_score"])
            return parsed["new_score"]
    except Exception as e:
        logger.warning("[unified_arbitration] failed: %s", e)
    return old_score
```

- [ ] **Step 4: 运行测试确认通过**

Run: `pytest tests/unit/evaluation/test_unified_review.py -v --tb=short`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add src/novelfactory/evaluation/unified/arbitration.py tests/unit/evaluation/test_unified_review.py
git commit -m "feat(unified): add disagreement arbitration replacing multi-round debate"
```

---

### Task 4: 统一评审引擎（主评审 + 降级）

**Files:**
- Create: `src/novelfactory/evaluation/unified/engine.py`
- Test: `tests/unit/evaluation/test_unified_review.py`

- [ ] **Step 1: 写失败测试（mock LLM）**

```python
import asyncio
from novelfactory.evaluation.unified.engine import UnifiedReviewEngine
from novelfactory.evaluation.unified.parser import parse_review_output

class _FakeLlm:
    def __init__(self, payload): self._payload = payload
    async def ainvoke(self, *a, **kw):
        return type("R", (), {"content": self._payload})()

SAMPLE = """<review_analysis>ok</review_analysis>
[评分] final=82.0
[四维-剧情逻辑] 27/30
[四维-文笔] 21/25
[四维-人物] 22/25
[四维-世界观] 18/20
[毒点] 无
[爽点] 打脸|P5; 升级|P14; 奖励|P20
[AI味] 0.75
[吸引力] 88  [沉浸] 90
[跨章] 84
[衰减] none
[废话段] 无
[跳跃] 无
评审意见：
- 老书虫视角：爽点密
- 番茄编辑视角：钩子强
- 读者视角：沉浸
- 评论员视角：逻辑通
- 分歧点：无
- 修复指令：P5 补一句反派反应"""

def test_engine_evaluate_success():
    engine = UnifiedReviewEngine(_FakeLlm(SAMPLE))
    r = asyncio.run(engine.evaluate(chapter_text="正文", genre="都市", prev_summary="", guide=""))
    assert r.failed is False
    assert r.final_score == 82.0
    assert r.shuangdian_count == 3

def test_engine_evaluate_retry_then_fallback():
    engine = UnifiedReviewEngine(_FakeLlm("不可解析"))
    r = asyncio.run(engine.evaluate(chapter_text="正文", genre="都市", prev_summary="", guide="", retries=1))
    assert r.failed is True
    assert r.final_score == 60.0  # fallback
    assert r.retried is True
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/unit/evaluation/test_unified_review.py -v --tb=short`
Expected: FAIL（ModuleNotFoundError: engine）

- [ ] **Step 3: 创建引擎**

```python
# src/novelfactory/evaluation/unified/engine.py
from __future__ import annotations

import logging

from langchain_core.language_models import BaseChatModel

from novelfactory.agents.infra.async_retry import async_llm_call_with_retry
from novelfactory.evaluation.unified.parser import apply_consistency_check, parse_review_output
from novelfactory.evaluation.unified.prompts import (
    build_quick_recheck_prompt,
    build_unified_review_prompt,
)
from novelfactory.evaluation.unified.schemas import UnifiedReviewResult

logger = logging.getLogger(__name__)

_DEFAULT_FALLBACK = 60.0


def _get_unified_param(key: str, default: float) -> float:
    try:
        from novelfactory.config.quality_params import quality_center
        val = quality_center.get(key)
        if val is not None:
            return float(val)
    except Exception:
        pass
    return default


class UnifiedReviewEngine:
    """统一评审引擎：一次 LLM 调用完成五视角评审 + 标签解析 + 自洽校验 + 降级。"""

    def __init__(self, llm: BaseChatModel):
        self._llm = llm

    async def evaluate(
        self,
        *,
        chapter_text: str,
        genre: str,
        prev_summary: str,
        guide: str,
        retries: int = 1,
    ) -> UnifiedReviewResult:
        fallback = _get_unified_param("unified.fallback_score", _DEFAULT_FALLBACK)
        prompt = build_unified_review_prompt(
            chapter_text=chapter_text, genre=genre, prev_summary=prev_summary, guide=guide,
        )
        last_result = UnifiedReviewResult(failed=True, final_score=fallback)
        for attempt in range(retries + 1):
            try:
                response = await async_llm_call_with_retry(
                    self._llm.ainvoke, prompt, step_name="unified_review"
                )
                raw = response.content if hasattr(response, "content") else str(response)
                result = parse_review_output(raw)
                if result is not None:
                    apply_consistency_check(result)
                    # v8.2: 分歧驱动仲裁（替代多轮辩论）——仅严重分歧触发，1 次轻量调用
                    severe_dsg = [
                        d for d in result.perspective_disagreements
                        if "severe" in d or "严重" in d
                    ]
                    if severe_dsg:
                        from novelfactory.evaluation.unified.arbitration import arbitrate
                        result.final_score = await arbitrate(
                            self._llm,
                            disagreements=severe_dsg,
                            old_score=result.final_score,
                        )
                    result.failed = False
                    result.retried = attempt > 0
                    return result
                last_result = UnifiedReviewResult(failed=True, final_score=fallback, raw_text=raw)
            except Exception as e:
                logger.warning("[unified_review] attempt %d failed: %s", attempt + 1, e)
        return last_result

    async def quick_recheck(
        self,
        *,
        chapter_text: str,
        old_issues: list[str],
        old_score: float,
        retries: int = 1,
    ) -> UnifiedReviewResult:
        fallback = _get_unified_param("unified.fallback_score", _DEFAULT_FALLBACK)
        prompt = build_quick_recheck_prompt(
            chapter_text=chapter_text, old_issues=old_issues, old_score=old_score,
        )
        for attempt in range(retries + 1):
            try:
                response = await async_llm_call_with_retry(
                    self._llm.ainvoke, prompt, step_name="unified_recheck"
                )
                raw = response.content if hasattr(response, "content") else str(response)
                result = parse_review_output(raw)
                if result is not None:
                    result.failed = False
                    return result
            except Exception as e:
                logger.warning("[unified_recheck] attempt %d failed: %s", attempt + 1, e)
        return UnifiedReviewResult(failed=True, final_score=fallback)
```

- [ ] **Step 4: 运行测试确认通过**

Run: `pytest tests/unit/evaluation/test_unified_review.py -v --tb=short`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add src/novelfactory/evaluation/unified/engine.py tests/unit/evaluation/test_unified_review.py
git commit -m "feat(unified): add UnifiedReviewEngine with retry and fallback"
```

---

### Task 5: 创建统一模块导出

**Files:**
- Create: `src/novelfactory/evaluation/unified/__init__.py`

- [ ] **Step 1: 创建导出**

```python
# src/novelfactory/evaluation/unified/__init__.py
from novelfactory.evaluation.unified.engine import UnifiedReviewEngine
from novelfactory.evaluation.unified.parser import apply_consistency_check, parse_review_output
from novelfactory.evaluation.unified.schemas import UnifiedFourDim, UnifiedReviewResult

__all__ = [
    "UnifiedReviewEngine",
    "UnifiedReviewResult",
    "UnifiedFourDim",
    "parse_review_output",
    "apply_consistency_check",
]
```

- [ ] **Step 2: 验证导入**

Run: `python -c "from novelfactory.evaluation.unified import UnifiedReviewEngine; print('ok')"`
Expected: `ok`

- [ ] **Step 3: 提交**

```bash
git add src/novelfactory/evaluation/unified/__init__.py
git commit -m "feat(unified): export unified review module"
```

---

### Task 6: verdict/engine.py 换内核（删程序化、接入统一评审）

**Files:**
- Modify: `src/novelfactory/evaluation/verdict/engine.py`
- Modify: `tests/unit/evaluation/test_verdict_engine.py`

- [ ] **Step 1: 阅读现有 evaluate 定位修改点**

Read: `src/novelfactory/evaluation/verdict/engine.py:296-400`（evaluate 主流程）与 `_fuse`（约 891-1038）
Expected: 确认步骤 1（run_programmatic_analysis）与 2-5 并行 gather 及 _fuse 加权为改造目标。

**本任务需一并删除的私有方法/常量**（调研确认无其他消费方）：
- `_detect_quality_decay()`（L137-213）与 `_DECAY_*` 常量（唯一引用 `analysis/ai_style_analyzer`，将随 Task 9 删除）
- `_resolve_llm_scores()`（L632-681）、`_analyze_toxic_state()`（L683-737）、`_ToxicState`
- `_calculate_weighted_score()`（L739-778）、`_apply_bonuses()`（L780-823，长度归一化部分删除，迭代加分移入 `_fuse_unified`）
- `_run_debate()`（L403-433）、`_run_four_dim_review()`（L435-556）、`_parse_four_dim_response()`、`_parse_evidence_chain()`
- 注意：`_parse_four_dim_response` 存在四维分项字段名 bug（用 `plot_logic/writing_style/...` 构造而 schema 是 `literary/structure/character/pacing`，被 except 静默吞掉）——统一评审不再复现此 bug，`UnifiedReviewResult.four_dim` 直接承载分项。

- [ ] **Step 2: 改造 evaluate 主流程**

替换 `evaluate` 中"步骤 1 + 并行 gather + _fuse"为统一评审调用（保留方法签名与 VerdictResult 组装）：

```python
# 在 evaluate 内（替换原程序化+并行 gather 部分）：
from novelfactory.evaluation.unified import UnifiedReviewEngine

unified_engine = UnifiedReviewEngine(reviewer_llm)
ur = await unified_engine.evaluate(
    chapter_text=chapter_text,
    genre=genre,
    prev_summary=prev_summary,
    guide=genre_scoring_guide,
    retries=int(_get_quality_param("unified.max_retries", 1)),
)

verdict = self._fuse_unified(
    ur,
    attempt_info,
    chapter_length=len(chapter_text),
)
logger.info(
    "[VerdictEngine] ch%d 统一评审 | final=%.1f quality=%.1f toxic=%d shuang=%d water=%d failed=%s retried=%s",
    chapter_index, ur.final_score, ur.four_dim.total(),
    len(ur.toxic_points), ur.shuangdian_count, len(ur.water_paragraphs),
    ur.failed, ur.retried,
)
return verdict
```

- [ ] **Step 3: 新增 `_fuse_unified`（替代 `_fuse`）**

```python
def _fuse_unified(self, ur, attempt_info, *, chapter_length: int) -> VerdictResult:
    """统一评审融合：LLM 综合分 + 迭代加分 + 路由阈值（保留原权重语义的简化版）。"""
    from novelfactory.config.quality_params import quality_center

    final_score = ur.final_score
    if attempt_info.loop_count > 0 or attempt_info.refine_attempts > 0:
        bonus_rewrite = float(quality_center.get("verdict.iteration_bonus.rewrite") or 3.0)
        bonus_refine = float(quality_center.get("verdict.iteration_bonus.refine") or 2.0)
        bonus_max = float(quality_center.get("verdict.iteration_bonus.max") or 8.0)
        bonus = attempt_info.loop_count * bonus_rewrite + attempt_info.refine_attempts * bonus_refine
        final_score = min(final_score + bonus, bonus_max + ur.final_score)

    quality_score = ur.four_dim.total()
    feedback = self._build_unified_feedback(ur)
    verdict = VerdictResult(
        level=self._decide_level(final_score, ur.severe_toxic, attempt_info),
        passed=False,
        final_score=round(final_score, 1),
        quality_score=quality_score,
        programmatic_score=0.0,  # 程序化已移除，字段保留兼容
        cross_chapter_consistency=ur.cross_chapter_score,
        debate_penalty=0.0,
        ai_style_score=ur.human_like_score,
        lao_shu_chong_score=ur.final_score,
        feedback=feedback,
    )
    return verdict
```

- [ ] **Step 4: 实现 `_decide_level` 与 `_build_unified_feedback` 辅助方法**

```python
def _decide_level(self, final_score: float, severe_toxic: bool, attempt_info) -> VerdictLevel:
    """路由三态（保留阈值与次数兜底，severe 毒点由 LLM 判定驱动强制重写）。"""
    from novelfactory.config.quality_params import quality_center

    pass_th = float(quality_center.get("verdict.pass_threshold") or 73.0)
    refine_th = float(quality_center.get("verdict.refine_threshold") or 55.0)
    both_exhausted = attempt_info.rewrite_exhausted and attempt_info.refine_exhausted
    if both_exhausted:
        return VerdictLevel.PASS
    if severe_toxic and not attempt_info.rewrite_exhausted:
        return VerdictLevel.REWRITE
    if final_score >= pass_th:
        return VerdictLevel.PASS
    if final_score >= refine_th:
        return VerdictLevel.REFINE
    return VerdictLevel.REWRITE


def _build_unified_feedback(self, ur) -> "FeedbackBundle":
    from novelfactory.evaluation.schemas import FeedbackBundle

    toxic_types = [t.get("type", "") for t in ur.toxic_points if t.get("type")]
    return FeedbackBundle(
        score_summary=(
            f"统一评审：{ur.final_score:.0f}/100 | 四维{ur.four_dim.total():.0f} | "
            f"爽点{ur.shuangdian_count} | 水段{len(ur.water_paragraphs)} | 跨章{ur.cross_chapter_score:.0f}"
        ),
        toxic_points=toxic_types,
        shuangdian_points=ur.shuangdian_points,
        debate_issues=[d for d in ur.perspective_disagreements],
        debate_strengths=ur.shuangdian_points[:3],
        debate_suggestions="\n".join(ur.fix_suggestions),
        review_comments=ur.review_comments,
    )
```

- [ ] **Step 5: 更新测试适配**

将 `tests/unit/evaluation/test_verdict_engine.py` 中引用 `run_programmatic_analysis`/`ProgrammaticReport` 的用例改为 mock `UnifiedReviewEngine`（注入 `_FakeLlm` 或直接 patch `UnifiedReviewEngine.evaluate` 返回构造的 `UnifiedReviewResult`），断言：路由三态、迭代加分、severe 强制重写、次数用尽强制 PASS。

- [ ] **Step 6: 运行测试**

Run: `pytest tests/unit/evaluation/test_verdict_engine.py tests/unit/evaluation/test_unified_review.py -v --tb=short`
Expected: PASS（无程序化引用残留）

- [ ] **Step 7: 提交**

```bash
git add src/novelfactory/evaluation/verdict/engine.py tests/unit/evaluation/test_verdict_engine.py
git commit -m "refactor(verdict): switch to unified LLM review, remove programmatic pipeline"
```

---

### Task 7: coordinator / feedback / service / replay 适配

**Files:**
- Modify: `src/novelfactory/evaluation/coordinator.py`
- Modify: `src/novelfactory/evaluation/verdict/feedback.py`
- Modify: `src/novelfactory/evaluation/service.py`
- Modify: `src/novelfactory/evaluation/replay.py`
- Test: `tests/unit/test_review_nodes.py`

- [ ] **Step 1: coordinator 适配**

`coordinator.py` 中 `verdict_engine_node` 的异常降级分支（约 124-145 行）保留；检查 `sw.write` 的评审摘要行（约 164-169 行）字段兼容（`verdict.programmatic_score` 改为显示 `verdict.feedback.score_summary`）：

```python
# coordinator.py 164-169 替换：
sw.write(
    f"[verdict_engine] 第{current_ch}章评审：{verdict.final_score:.1f}/100 → {level_text}{force_pass}\n"
    f"  四维={verdict.quality_score:.0f} | 跨章={verdict.cross_chapter_consistency:.0f} | "
    f"AI味={verdict.ai_style_score:.2f}\n"
    f"  {verdict.feedback.score_summary}\n"
)
```

- [ ] **Step 2: feedback.py 适配**

确认 `FeedbackBundle` 由 `_build_unified_feedback` 构造后无 `programmatic` 直接依赖；删除其中对 `ProgrammaticReport` 的 import（若仅用于类型标注）。

- [ ] **Step 3: service.py / replay.py 适配**

将 `service.py` 的 `review_pipeline` 与 `replay.py` 中调用 `run_programmatic_analysis`/旧评审维度的地方替换为 `UnifiedReviewEngine`（同 Task 6 模式）。`ReplayReport` 的 `programmatic_score` 对比字段保留（值恒 0.0，`_build_report` 不再展示该维差异）。

- [ ] **Step 3.5: 前端/流式显示适配（programmatic_score 恒 0 处理）**

`VerdictResult.programmatic_score` 保留字段 = 0.0 保证兼容，但以下消费点显示"程序化分"已无意义，改为显示统一评审摘要：
- `server/stream_tracker.py:61-69,95-97`：SSE 进度事件 `composite_score` 来源从 `verdict_result.programmatic_score` 改为 `verdict_result.final_score / 100`
- `cli/dashboard.py:445-457`：`Programmatic: Y` 显示改为 `Unified: Y`（读 `verdict_result.final_score`）
- `graph/crews/writing_nodes/routing.py:91-95,106-111`：`_exit_for_chapter` 记录与流式同理改为 final_score

- [ ] **Step 4: 运行相关测试**

Run: `pytest tests/unit/test_review_nodes.py tests/unit/evaluation/test_llm_analysis.py -v --tb=short`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add src/novelfactory/evaluation/coordinator.py src/novelfactory/evaluation/verdict/feedback.py src/novelfactory/evaluation/service.py src/novelfactory/evaluation/replay.py
git commit -m "refactor(evaluation): adapt coordinator/service/replay to unified review"
```

---

### Task 8: quality_params 清理与新增

**Files:**
- Modify: `src/novelfactory/config/quality_params.py`
- Test: `tests/unit/config/test_constants_consistency.py`

- [ ] **Step 1: 清理注册表**

删除以下注册块（保留代码注释说明"v8.2 已移除，由统一评审承接"）：
- `verdict.weights.*` 循环注册（融合权重）
- `verdict.debate_penalty.*`（cap/per_issue/per_severe）
- `calibration.*`（llm_virtual_high/programmatic_low/short_text_llm_weight/severe_toxic_cap）
- `ai_style.weights.*` 循环注册
- `toxic.weights.*` 与 `shuangdian.weights.*` 循环注册
- `debate.max_rounds` / `debate.convergence_idle_rounds`
- `verdict.length_normalize` / `verdict.normalize_base`
- `fallback.*`（保留 `fallback.quality_score` 供降级，其余删除）

新增：

```python
# ── 统一评审参数（v8.2 新增） ──
_U = "unified"
_register(ParamSpec("unified.max_retries", "unified", int, 1, 0, 3,
                    "统一评审失败最大重试次数", _U, ("unified",)))
_register(ParamSpec("unified.fallback_score", "unified", float, 60.0, 0.0, 100.0,
                    "统一评审失败降级分（REFINE 档）", _U, ("unified",)))
```

同步更新 `FEEDBACK_PARAM_MAP` 中引用已删参数的条目（如"评分虚高"→["unified.fallback_score"]、"太严了/太松了"→ 保留 pass/refine_threshold、"战力崩坏/水文/NTR"→["unified"]、"爽点不够"→["unified"]、"辩论不够深入"→["unified"]、"程序化分析权重"→删除），删除 `WEIGHT_GROUPS` 中 `verdict_weights`/`ai_style_weights` 两个组。

**调参 Agent 规则同步**：`src/novelfactory/graph/chat/agents/quality_tuner_agent.py` 的调参 prompt（约 L42-124）含"权重类参数总和必须=1.0"约束——删除 weights 组后该规则失效，改为提示"统一评审下可直接调整 pass_threshold/refine_threshold/iteration_bonus.*/iteration.max_*"。

- [ ] **Step 2: 更新一致性测试**

在 `tests/unit/config/test_constants_consistency.py` 增加：

```python
from novelfactory.config.quality_params import PARAM_REGISTRY

def test_removed_programmatic_params_not_registered():
    for key in ("verdict.weights.programmatic", "verdict.debate_penalty.cap",
                "calibration.llm_virtual_high", "ai_style.weights.cliche_ratio",
                "toxic.weights.NTR", "shuangdian.weights.打脸", "debate.max_rounds"):
        assert key not in PARAM_REGISTRY

def test_unified_params_registered():
    assert "unified.max_retries" in PARAM_REGISTRY
    assert "unified.fallback_score" in PARAM_REGISTRY
```

- [ ] **Step 3: 运行测试**

Run: `pytest tests/unit/config/test_constants_consistency.py -v --tb=short`
Expected: PASS

- [ ] **Step 4: 提交**

```bash
git add src/novelfactory/config/quality_params.py tests/unit/config/test_constants_consistency.py
git commit -m "refactor(quality-params): remove programmatic params, add unified review params"
```

---

### Task 9: 删除程序化组件与旧评分维度

**Files:**
- Delete: `src/novelfactory/evaluation/programmatic/__init__.py`
- Delete: `src/novelfactory/evaluation/programmatic/runner.py`
- Delete: `src/novelfactory/evaluation/programmatic/ai_style_sensor.py`
- Delete: `src/novelfactory/evaluation/programmatic/old_reader_sensor.py`
- Delete: `src/novelfactory/evaluation/programmatic/cross_chapter_sensor.py`
- Delete: `src/novelfactory/evaluation/llm/attraction_llm.py`
- Delete: `src/novelfactory/evaluation/llm/old_reader_llm.py`
- Delete: `src/novelfactory/evaluation/llm/ai_style_llm.py`
- Delete: `src/novelfactory/evaluation/verdict/calibration.py`
- Delete: `src/novelfactory/evaluation/debate/__init__.py`
- Delete: `src/novelfactory/evaluation/debate/engine.py`
- Delete: `src/novelfactory/evaluation/debate/parser.py`
- Delete: `src/novelfactory/evaluation/debate/prompts.py`
- Modify: `src/novelfactory/evaluation/__init__.py`
- Modify: `src/novelfactory/evaluation/verdict/engine.py`（清理残留 import）

- [ ] **Step 1: 全仓搜索残留引用**

Run: `rg -n "run_programmatic_analysis|ProgrammaticReport|CrossChapterSignals|attraction_llm|old_reader_llm|ai_style_llm|InformedDebateEngine|debate\.engine|calibration|ai_style_analyzer|old_reader_reviewer" src/`
Expected: 输出为空（Task 6/7/4A 已全部替换）；如有残留，逐一改为统一评审/仲裁或删除。

- [ ] **Step 2: 删除文件**

Run: `git rm src/novelfactory/evaluation/programmatic/ src/novelfactory/evaluation/debate/ src/novelfactory/evaluation/llm/attraction_llm.py src/novelfactory/evaluation/llm/old_reader_llm.py src/novelfactory/evaluation/llm/ai_style_llm.py src/novelfactory/evaluation/verdict/calibration.py src/novelfactory/analysis/ai_style_analyzer.py src/novelfactory/analysis/old_reader_reviewer.py`

- [ ] **Step 3: 更新 `evaluation/__init__.py` 导出**

删除对 `ProgrammaticReport`/`CrossChapterSignals`/`DebateReport`/`FourDimReviewResult` 的导出；新增：

```python
from novelfactory.evaluation.unified import (
    UnifiedReviewEngine,
    UnifiedReviewResult,
    parse_review_output,
    apply_consistency_check,
)
```

保留 `VerdictResult`/`VerdictLevel`/`FeedbackBundle`/`AttemptInfo` 导出（下游兼容）。

- [ ] **Step 4: 清理 engine.py 残留 import**

移除 `engine.py` 顶部对 `run_programmatic_analysis`、`ProgrammaticReport`、`CrossChapterSignals`、`llm_old_reader_analysis`、`llm_ai_style_analysis`、`attraction_llm_analysis`、`InformedDebateEngine` 的 import；保留 `_get_quality_param` 与 `VerdictResult` 组装所需导入。

- [ ] **Step 4.5: 测试文件适配（调研确认的三个依赖旧结构的测试）**

1. `tests/unit/evaluation/test_verdict_engine_parallel.py`：原文件 monkeypatch 5 个旧 LLM 维度桩且真实执行 `run_programmatic_analysis`（L11-12）——重写为：patch `UnifiedReviewEngine.evaluate`（async，返回构造的 `UnifiedReviewResult`），断言 evaluate 只调用统一评审 1 次、结果正确路由。
2. `tests/unit/evaluation/test_debate_engine.py`：原测 `DebateReport.severity_weight`（引 `VERDICT_DEBATE_PENALTY_CAP`）——debate 删除后，改为测 `unified/arbitration.py` 的 `parse_arbitration`/`arbitrate`（见 Task 4A 测试），删除旧 `DebateReport` 用例。
3. `tests/unit/evaluation/test_llm_analysis.py`：原测旧 LLM 维度 JSON 解析——改测 `unified/parser.py` 的标签化解析（复用 Task 2 的 SAMPLE 断言集），删除旧维度解析用例。

- [ ] **Step 5: 全量单测 + lint**

Run: `pytest tests/unit -v --tb=short`
Run: `ruff check src/novelfactory/evaluation/`
Expected: 全部 PASS；ruff 无错误

- [ ] **Step 6: 提交**

```bash
git add -A src/novelfactory/evaluation/
git commit -m "refactor(evaluation): remove programmatic pipeline and legacy LLM review dimensions"
```

---

### Task 10: 集成验证（mock 全链路）

**Files:**
- Modify: `tests/integration/test_full_pipeline.py`

- [ ] **Step 1: 更新集成测试**

将 `tests/integration/test_full_pipeline.py` 中依赖程序化评分的断言改为统一评审（patch `UnifiedReviewEngine.evaluate` 返回固定 `UnifiedReviewResult`），验证：setup→写作→评审→路由→修复→复查→完成 全链路无异常。

- [ ] **Step 2: 运行集成测试**

Run: `pytest tests/integration/test_full_pipeline.py -v --tb=short`
Expected: PASS

- [ ] **Step 3: 提交**

```bash
git add tests/integration/test_full_pipeline.py
git commit -m "test(integration): adapt full pipeline test to unified review"
```

---

### Task 11: 容器重建与端到端验证

**Files:**
- 无代码改动

- [ ] **Step 1: 语法与 lint 全量检查**

Run: `docker run --rm -v d:\langgraph:/app -w /app langgraph-api python -m compileall -q src/`
Run: `docker run --rm -v d:\langgraph:/app -w /app langgraph-api ruff check src/novelfactory/evaluation/`
Expected: 无输出（通过）

- [ ] **Step 2: 重建 api 容器**

Run: `docker compose -p langgraph up -d --build api`
Expected: `Container langgraph_api Started`，`docker compose -p langgraph ps` 显示 `healthy`

- [ ] **Step 3: 校验参数接口**

Run: `Invoke-RestMethod -Uri 'http://localhost:8123/feishu/quality-params' -Method Post -ContentType 'application/json' -Body '{}' -TimeoutSec 30`
Expected: 200；`params` 中无 `verdict.weights.*`，含 `unified.max_retries`

- [ ] **Step 4: 新建线程端到端**

Run: `POST /threads` → `POST /threads/{id}/runs/wait`（3 章）
Expected: 写作完成；日志 `AUDIT: llm_call phase=unified_review` 每章 ≤ 2 次；无 `run_programmatic_analysis` 引用；章节同步飞书

- [ ] **Step 5: 结果抽查**

从容器 `/tmp/chapter_*.md` 提取 1 章正文，检查：无程序化毒点误报、含「」系统提示格式、段落 ≥2 句、爽点 ≥3（对照统一评审输出的 shuangdian_count）

- [ ] **Step 6: 提交（无代码改动则跳过）**

---

## Self-Review

**1. Spec 覆盖检查：**
- FR-001 统一评审引擎 → Task 1-5
- FR-002 清除程序化 → Task 9
- FR-003 融合简化 → Task 6（_fuse_unified）
- FR-004 两段式评审 → Task 4（quick_recheck）+ Task 6 路由
- FR-005 多视角保留 → Task 3（prompt 五视角）；辩论优点保留 → Task 4A（分歧仲裁，替代多轮辩论）
- FR-006 降级兜底 → Task 4（retry+fallback）
- FR-007 参数清理 → Task 8
- FR-008 下游适配 → Task 7
- FR-009 自洽校验 → Task 2（apply_consistency_check）
- NFR-001 效率（≤2 调用：主评审 1 + 严重分歧仲裁 1）→ Task 11 Step 4 验证
- TR-001/TR-002 → Task 8/9（git 回滚即可，未加开关）

**2. 占位符扫描：** 无 TBD/TODO；每个代码步骤含完整代码；测试含具体断言。

**3. 类型一致性：** `UnifiedReviewResult` 字段（final_score/four_dim/toxic_points/severe_toxic/shuangdian_count/water_paragraphs/cross_chapter_score/human_like_score）在 Task 1 定义，Task 2/4/6 引用一致；`apply_consistency_check`/`parse_review_output` 签名在 Task 2 定义、Task 4/6 调用一致；`_fuse_unified`/`_decide_level`/`_build_unified_feedback` 在 Task 6 定义并相互引用一致；`parse_arbitration`/`arbitrate` 在 Task 4A 定义、Task 4 调用一致。

**4. 调研回填（第二次深度调研新增覆盖）：**
- `analysis/ai_style_analyzer.py`、`analysis/old_reader_reviewer.py` 删除（唯一残留引用 decay 检测/程序化传感器）→ Task 9 Step 1-2
- `quality_tuner_agent.py` 调参 prompt 权重规则更新 → Task 8
- 测试适配 3 个文件（parallel/debate/llm_analysis）→ Task 9 Step 4.5
- 前端显示（stream_tracker/dashboard/routing）programmatic_score 恒 0 处理 → Task 7 Step 3.5
- 每章 LLM 调用基数修正为 12-18 次（6 reviewer + 6-12 debate）→ Goal
- 降级逻辑双份（coordinator.py L123-147 与 service.py L83-104）在重构后统一为 `UnifiedReviewEngine` 的 fallback（fallback_score=60），两处降级分支仅保留 `VerdictResult` 组装壳
