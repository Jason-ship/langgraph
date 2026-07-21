"""跨章一致性传感器 — 纯代码，零 LLM。

只产出客观数据（"句长+40%"），不做判断（"声音漂移"）。
判断交给 LLM（注入四维评分 prompt 和辩论 prompt）。

检测维度：
    1. 角色声音信号 — 句长统计对比
    2. 节奏信号 — 对话/动作/描写密度对比
    3. 文风信号 — 词汇丰富度对比
    4. 伏笔信号 — 暗示性语句/未解释元素检测
    5. 情节连贯信号 — 关键词重叠率
"""

from __future__ import annotations

import logging
import re
from collections import Counter

from novelfactory.evaluation.schemas import CrossChapterSignals

logger = logging.getLogger(__name__)

# ========== 常量 ==========
_MIN_TEXT_LENGTH = 100  # 最短检测文本长度
_MIN_SENTENCE_LENGTH = 2  # 最短有效句长
_MIN_SENTENCE_COUNT = 3  # 最少句数

# 句子分隔符
_SENTENCE_SPLIT_PATTERN = r"[。！？\n]+"

# ── v7.3: ItemStateTracker 用的预编译正则（避免方法内 import re 的作用域问题）──
_DESTROY_RE = re.compile(
    r"碎|毁|断|裂|崩|焚|熔|爆|烂|破|废|灭|化|"
    r"劈|折|斩|切|砍|撕|扯|砸|碾|压|两半|齑粉|灰烬|虚无"
)
_LOST_RE = re.compile(r"脱手|掉落|丢失|遗失|不见|失落|被夺|被抢|被偷")

# 对话匹配（中文引号）
_DIALOGUE_PATTERN = r'[""\'\']([^""\'\']{3,})[""\'\']'

# 动作描写关键词（简化版）
_ACTION_KEYWORDS = {
    "冲",
    "跑",
    "跳",
    "闪",
    "劈",
    "斩",
    "刺",
    "挥",
    "踢",
    "打",
    "抓",
    "推",
    "拉",
    "撞",
    "跌",
    "翻",
    "射",
    "挡",
    "躲",
    "攻",
}

# 暗示性语句模式（潜在伏笔）
_FORESHADOWING_PATTERNS = [
    r"似乎.{0,20}",
    r"仿佛.{0,20}",
    r"隐约.{0,20}",
    r"不知道为什么.{0,20}",
    r"莫名.{0,20}",
    r"神秘.{0,20}",
    r"未知.{0,20}",
    r"隐藏.{0,20}",
    r"暗中.{0,20}",
    r"殊不知.{0,20}",
]

# 停用词（用于关键词提取）
_STOPWORDS = {
    "的",
    "了",
    "是",
    "在",
    "有",
    "和",
    "就",
    "不",
    "都",
    "一",
    "上",
    "也",
    "到",
    "说",
    "要",
    "去",
    "会",
    "着",
    "看",
    "想",
    "这",
    "那",
    "他",
    "她",
    "它",
    "我",
    "你",
    "们",
    "个",
    "来",
    "里",
    "下",
    "以",
    "于",
    "为",
    "而",
    "但",
    "却",
    "又",
    "还",
    "只",
    "才",
    "便",
    "被",
    "把",
    "向",
    "从",
    "对",
    "给",
    "让",
}


class CrossChapterSensor:
    """跨章一致性传感器 — 纯代码，零 LLM。

    只产出客观数据，不做判断。判断交给 LLM。
    """

    def analyze(
        self,
        chapter_text: str,
        chapter_index: int,
        prev_chapters_summary: str = "",
        character_setting: str = "",
        story_outline: str = "",
        tracker: ItemStateTracker | None = None,
    ) -> CrossChapterSignals:
        """执行跨章一致性信号采集。

        Args:
            chapter_text: 当前章节文本
            chapter_index: 当前章节序号
            prev_chapters_summary: 前文摘要（已有 context_builder 产出）
            character_setting: 角色设定
            story_outline: 故事大纲

        Returns:
            CrossChapterSignals — 客观信号数据
        """
        has_prev = bool(
            prev_chapters_summary and len(prev_chapters_summary.strip()) > 50
        )

        if not chapter_text or len(chapter_text.strip()) < _MIN_TEXT_LENGTH:
            return CrossChapterSignals(
                has_prev_context=has_prev,
                chapter_index=chapter_index,
            )

        # 1. 句长信号
        cur_avg_sent_len = self._avg_sentence_length(chapter_text)
        prev_avg_sent_len = (
            self._avg_sentence_length(prev_chapters_summary) if has_prev else 0.0
        )
        sent_delta = self._calc_delta(cur_avg_sent_len, prev_avg_sent_len)

        # 2. 节奏信号
        cur_dialogue_density = self._dialogue_density(chapter_text)
        prev_dialogue_density = (
            self._dialogue_density(prev_chapters_summary) if has_prev else 0.0
        )
        dialogue_delta = self._calc_delta(cur_dialogue_density, prev_dialogue_density)

        cur_action_density = self._action_density(chapter_text)
        prev_action_density = (
            self._action_density(prev_chapters_summary) if has_prev else 0.0
        )

        # 3. 文风信号
        cur_vocab_richness = self._vocab_richness(chapter_text)
        prev_vocab_richness = (
            self._vocab_richness(prev_chapters_summary) if has_prev else 0.0
        )
        style_delta = self._calc_delta(cur_vocab_richness, prev_vocab_richness)

        # 4. 伏笔信号
        new_foreshadowing = self._detect_foreshadowing(chapter_text)
        unexplained = self._find_unexplained(prev_chapters_summary) if has_prev else []

        # 5. 情节连贯信号
        setting_keywords = (
            self._extract_keywords(prev_chapters_summary, top_n=15) if has_prev else []
        )
        chapter_keywords = self._extract_keywords(chapter_text, top_n=15)
        overlap = self._keyword_overlap(setting_keywords, chapter_keywords)

        signals = CrossChapterSignals(
            cur_avg_sentence_length=cur_avg_sent_len,
            prev_avg_sentence_length=prev_avg_sent_len,
            sentence_length_delta=sent_delta,
            cur_dialogue_density=cur_dialogue_density,
            prev_dialogue_density=prev_dialogue_density,
            dialogue_density_delta=dialogue_delta,
            cur_action_density=cur_action_density,
            prev_action_density=prev_action_density,
            cur_vocab_richness=cur_vocab_richness,
            prev_vocab_richness=prev_vocab_richness,
            style_drift_delta=style_delta,
            potential_new_foreshadowing=new_foreshadowing,
            unexplained_elements=unexplained,
            setting_keywords=setting_keywords,
            chapter_keywords=chapter_keywords,
            keyword_overlap_ratio=overlap,
            has_prev_context=has_prev,
            chapter_index=chapter_index,
        )

        # v7.3: 物品状态追踪（纯代码，零 LLM）
        # 每次调用创建独立实例，避免跨请求/项目的状态污染
        item_tracker = tracker if tracker is not None else ItemStateTracker()
        item_issues = item_tracker.update_from_chapter(chapter_text)
        if item_issues:
            signals.item_state_issues = item_issues
            logger.info(
                "[ItemStateTracker] 第%d章 发现%d个物品状态问题",
                chapter_index,
                len(item_issues),
            )

        return signals

    def _avg_sentence_length(self, text: str) -> float:
        """计算平均句长（字符数）。"""
        if not text:
            return 0.0
        sentences = re.split(_SENTENCE_SPLIT_PATTERN, text)
        sentences = [
            s.strip() for s in sentences if s.strip() and len(s) > _MIN_SENTENCE_LENGTH
        ]
        if not sentences:
            return 0.0
        return sum(len(s) for s in sentences) / len(sentences)

    def _calc_delta(self, cur: float, prev: float) -> float:
        """计算变化百分比。"""
        if prev == 0:
            return 0.0
        return (cur - prev) / prev

    def _dialogue_density(self, text: str) -> float:
        """计算对话密度（对话字符数/总字符数）。"""
        if not text:
            return 0.0
        total = len(text)
        if total == 0:
            return 0.0
        dialogues = re.findall(_DIALOGUE_PATTERN, text)
        dialogue_chars = sum(len(d) for d in dialogues)
        return dialogue_chars / total

    def _action_density(self, text: str) -> float:
        """计算动作描写密度（动作关键词数/总词数）。"""
        if not text:
            return 0.0
        words = [text[i : i + 2] for i in range(len(text) - 1)]
        if not words:
            return 0.0
        action_count = sum(1 for w in words if any(k in w for k in _ACTION_KEYWORDS))
        return action_count / len(words)

    def _vocab_richness(self, text: str) -> float:
        """计算词汇丰富度（unique/total）。"""
        if not text:
            return 0.0
        words = [text[i : i + 2] for i in range(len(text) - 1)]
        words = [
            w
            for w in words
            if not all(c in " \t\n\r，。！？；：、''（）()【】[]《》<>" for c in w)
        ]
        if not words:
            return 0.0
        return len(set(words)) / len(words)

    def _detect_foreshadowing(self, text: str) -> list[str]:
        """检测潜在伏笔（暗示性语句）。"""
        results: list[str] = []
        for pattern in _FORESHADOWING_PATTERNS:
            matches = re.findall(pattern, text)
            for m in matches[:2]:
                m_stripped = m.strip()
                if len(m_stripped) > 3:
                    results.append(m_stripped[:30])
        return results[:5]

    def _find_unexplained(self, prev_summary: str) -> list[str]:
        """从前文摘要中提取未解释元素。"""
        results: list[str] = []
        for pattern in _FORESHADOWING_PATTERNS:
            matches = re.findall(pattern, prev_summary)
            for m in matches[:2]:
                m_stripped = m.strip()
                if len(m_stripped) > 3:
                    results.append(m_stripped[:30])
        return results[:5]

    def _extract_keywords(self, text: str, top_n: int = 15) -> list[str]:
        """提取关键词（简化版：2-gram 频率排序）。"""
        if not text:
            return []
        words = [text[i : i + 2] for i in range(len(text) - 1)]
        words = [
            w
            for w in words
            if w not in _STOPWORDS
            and not all(c in " \t\n\r，。！？；：、''（）()【】[]《》<>" for c in w)
        ]
        if not words:
            return []
        counter = Counter(words)
        return [w for w, _ in counter.most_common(top_n)]

    def _keyword_overlap(self, keywords_a: list[str], keywords_b: list[str]) -> float:
        """计算关键词重叠率（Jaccard 系数）。"""
        if not keywords_a or not keywords_b:
            return 0.0
        set_a = set(keywords_a)
        set_b = set(keywords_b)
        intersection = set_a & set_b
        union = set_a | set_b
        return len(intersection) / len(union) if union else 0.0


# ═══════════════════════════════════════════════════════════════════════════════
#  v7.3: 跨章物品状态追踪器
#  参考 SCORE (UC Berkeley 2025) — Item Status Tracking 从 0 提升到 76-98。
#  跟踪关键物品的三态（active/lost/destroyed），跨章节检测状态不一致。
# ═══════════════════════════════════════════════════════════════════════════════


class ItemStateTracker:
    """跨章关键物品状态追踪。

    纯代码实现，零 LLM。只在当前章节文本中做模式匹配检测状态变化。
    核心三态: active → lost → destroyed（不可逆回 active）。
    """

    # 状态常量
    ACTIVE = "active"
    LOST = "lost"
    DESTROYED = "destroyed"

    # 状态转移合法性规则
    _VALID_TRANSITIONS: dict[str, list[str]] = {
        "active": ["lost", "destroyed"],
        "lost": ["active", "destroyed"],  # lost→active 需要解释，但技术上可能
        "destroyed": [],  # 不可逆
    }

    def __init__(self, initial_items: dict[str, str] | None = None):
        self._states: dict[str, str] = initial_items or {}

    @property
    def states(self) -> dict[str, str]:
        """获取当前物品状态快照。"""
        return dict(self._states)

    def update_from_chapter(self, chapter_text: str) -> list[str]:
        """解析本章文本，更新物品状态，返回检测到的不一致问题。

        检测方法：对每个已注册的物品名，在文本中搜索该物品名+附近的关键词。
        对于新物品的初次注册，由调用方通过 set_state() 手动设置。

        Args:
            chapter_text: 当前章节文本

        Returns:
            list[str] — 检测到的不一致问题描述
        """
        if not chapter_text:
            return []

        issues: list[str] = []

        # 对每个已注册的物品，检查本章是否有状态变化
        for item in list(self._states.keys()):
            if item not in chapter_text:
                continue

            old_state = self._states[item]

            # 检测 destroyed → 重新 active
            if old_state == self.DESTROYED:
                # 用 find + 切片替代复杂正则
                idx = chapter_text.find(item)
                if idx >= 0:
                    # 检查物品名后附近是否有"使用"类动词
                    after = chapter_text[idx + len(item) : idx + len(item) + 10]
                    # 检查物品名前附近是否有"使用"类动词
                    before = chapter_text[max(0, idx - 10) : idx]
                    surrounding = before + after
                    for kw in (
                        "使用",
                        "拿出",
                        "提起",
                        "握住",
                        "拔出",
                        "举起",
                        "拿起",
                        "祭出",
                        "放出",
                        "催动",
                        "持",
                        "握",
                    ):
                        if kw in surrounding:
                            issues.append(
                                f"物品「{item}」之前已 destroyed，本章却有使用它的描述（严重状态冲突）"
                            )
                            break

            # 检测 active/lost → destroyed
            if old_state in (self.ACTIVE, self.LOST):
                ctx = self._get_item_context(chapter_text, item)
                if ctx and _DESTROY_RE.search(ctx):
                    self._states[item] = self.DESTROYED

            # 检测 active → lost
            if old_state == self.ACTIVE:
                ctx = self._get_item_context(chapter_text, item)
                if ctx and _LOST_RE.search(ctx):
                    self._states[item] = self.LOST

        return issues

    @staticmethod
    def _get_item_context(text: str, item: str, window: int = 20) -> str:
        """获取物品名附近的上下文（前后各 window 字符）。"""
        idx = text.find(item)
        if idx == -1:
            return ""
        start = max(0, idx - window)
        end = min(len(text), idx + len(item) + window)
        return text[start:end]

    def set_state(self, item: str, state: str, force: bool = False) -> bool:
        """手动设置物品状态。

        Args:
            item: 物品名
            state: 目标状态（active/lost/destroyed）
            force: 是否强制设置（跳过合法性检查）

        Returns:
            bool — 是否设置成功
        """
        if state not in (self.ACTIVE, self.LOST, self.DESTROYED):
            return False
        if force:
            self._states[item] = state
            return True

        old_state = self._states.get(item)
        if old_state is None:
            self._states[item] = state
            return True

        allowed = self._VALID_TRANSITIONS.get(old_state, [])
        if state in allowed:
            self._states[item] = state
            return True

        return False

    def get_state(self, item: str) -> str | None:
        """查询物品当前状态。"""
        return self._states.get(item)


# ═══════════════════════════════════════════════════════════════════════════════
#  v7.3: 情感一致性感知检索器
#  参考 SCORE (UC Berkeley 2025) 的情感分析 σ(e) 过滤策略。
#  在相似章节检索后加入情感一致性过滤，防止检索到情感弧线不一致的章节。
# ═══════════════════════════════════════════════════════════════════════════════


class SentimentConsistencyFilter:
    """情感一致性过滤器 — 对检索结果做情感维度后过滤。

    用法:
        filter = SentimentConsistencyFilter()
        filtered = filter.filter(similar_chapters, query_sentiment=0.7)
    """

    def __init__(self, sentiment_threshold: float = 0.3):
        """
        Args:
            sentiment_threshold: 情感评分最大差异阈值（0-1）。越小越严格。
        """
        self._threshold = sentiment_threshold

    def estimate_sentiment(self, text: str) -> float:
        """快速估计文本情感分（0=负面, 0.5=中性, 1=正面）。

        纯代码实现，零 LLM。基于情感词典的正/负面词计数。
        """
        if not text or len(text) < 20:
            return 0.5

        # 正面词（网文常见）
        positive = {
            "喜",
            "笑",
            "胜",
            "赢",
            "成",
            "功",
            "破",
            "突",
            "进",
            "升",
            "强",
            "获",
            "得",
            "好",
            "妙",
            "绝",
            "赞",
            "美",
            "乐",
            "欢",
            "得意",
            "兴奋",
            "激动",
            "痛快",
            "畅快",
            "欣喜",
            "满足",
            "扬眉吐气",
            "大获全胜",
            "旗开得胜",
            "春风得意",
        }
        # 负面词
        negative = {
            "悲",
            "哀",
            "伤",
            "痛",
            "苦",
            "惨",
            "败",
            "输",
            "死",
            "亡",
            "危",
            "险",
            "惧",
            "怕",
            "慌",
            "乱",
            "怒",
            "恨",
            "怨",
            "愁",
            "绝望",
            "恐惧",
            "愤怒",
            "悲伤",
            "痛苦",
            "沮丧",
            "焦虑",
            "岌岌可危",
            "九死一生",
            "绝境",
            "末路",
        }

        pos_count = sum(1 for w in positive if w in text)
        neg_count = sum(1 for w in negative if w in text)
        total = pos_count + neg_count

        if total == 0:
            return 0.5
        return pos_count / total

    def filter(
        self,
        chapters: list[dict],
        query_sentiment: float | None = None,
        query_text: str = "",
    ) -> list[dict]:
        """对检索结果做情感一致性过滤。

        Args:
            chapters: 检索到的章节列表（每项至少含 "chapter_text" / "content" / "summary" 字段）
            query_sentiment: 目标情感分。为 None 时从 query_text 估计。
            query_text: 用于估计目标情感分的文本（仅当 query_sentiment 为 None 时使用）

        Returns:
            过滤后的章节列表，按情感相似度排序（最相似在前）
        """
        if not chapters:
            return []

        if query_sentiment is None:
            query_sentiment = self.estimate_sentiment(query_text) if query_text else 0.5

        scored: list[tuple[dict, float]] = []
        for ch in chapters:
            text = ch.get("chapter_text", "") or ch.get("content", "") or ch.get("summary", "")
            if not text:
                continue
            ch_sentiment = self.estimate_sentiment(text[:1000])
            diff = abs(ch_sentiment - query_sentiment)
            if diff <= self._threshold:
                scored.append((ch, diff))

        scored.sort(key=lambda x: x[1])
        return [s[0] for s in scored]
