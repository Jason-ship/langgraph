"""情感一致性过滤器 — RAG 检索结果后过滤（v8.2 从 programmatic 迁移）。

纯代码实现，零 LLM、零评分职责：基于情感词典估计章节情感分，
过滤与查询情感差异过大的检索结果。
"""

from __future__ import annotations


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
