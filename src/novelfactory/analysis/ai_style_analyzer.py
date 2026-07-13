"""
AI味检测模块

基于 8 维统计特征综合判定文本的"机械感"程度。
完全程序化，无需 LLM 调用，毫秒级响应。

检测维度：
1. N-gram 重复率（3-gram, 4-gram）—— 权重 25%
2. 句长波动 —— 权重 20%
3. 词汇多样性（Unique/Total）—— 权重 18%
4. 模板化表达比例 —— 权重 15%
5. 标点节奏变异系数 —— 权重 10%
6. 对白比例 —— 权重 7%
7. 情绪/感官词密度 —— 权重 3%
8. 语义平滑度 —— 权重 2%

使用方式：
    from novelfactory.analysis.ai_style_analyzer import analyze_ai_style
    result = analyze_ai_style("这是一段待检测的小说文本...")
"""

from __future__ import annotations

import logging
import math
import re
from typing import TypedDict

logger = logging.getLogger(__name__)

# ========== 数值常量（提取自各评分函数，避免魔法数字） ==========
# 句长波动
_MIN_SENTENCE_LENGTH = 2  # 最短有效句长
_MIN_SENTENCE_COUNT = 3  # 最少句数阈值
_STD_DEV_AI_THRESHOLD = 8  # 句长标准差 ≤ 此值 → 极AI
_STD_DEV_HUMAN_THRESHOLD = 20  # 句长标准差 ≥ 此值 → 人类
_STD_DEV_RANGE = 12  # 句长标准差归一化范围
# 词汇多样性
_DIVERSITY_AI_THRESHOLD = 0.3  # 词汇多样性 ≤ 此值 → 极AI
_DIVERSITY_HUMAN_THRESHOLD = 0.7  # 词汇多样性 ≥ 此值 → 人类
_DIVERSITY_RANGE = 0.4  # 词汇多样性归一化范围
# 模板化表达
_CLICHE_AMPLIFICATION = 5  # 模板命中率放大系数
# 标点节奏
_MIN_PUNCTUATION_COUNT = 3  # 最少标点数量
_CV_AI_THRESHOLD = 0.3  # 变异系数 ≤ 此值 → 极AI
_CV_HUMAN_THRESHOLD = 0.8  # 变异系数 ≥ 此值 → 人类
_CV_RANGE = 0.5  # 变异系数归一化范围
# 对白比例
_DIALOGUE_LOWER = 0.25  # 对白比例正常下限
_DIALOGUE_UPPER = 0.45  # 对白比例正常上限
_DIALOGUE_LOW_RANGE = 0.25  # 对白过少归一化除数
_DIALOGUE_HIGH_RANGE = 0.55  # 对白过多归一化除数
# 感官/情绪词密度
_SENSORY_DEFAULT = 0.5  # 无词可判时默认返回
_SENSORY_LOWER = 0.02  # 密度正常下限
_SENSORY_UPPER = 0.08  # 密度正常上限
_SENSORY_LOW_RANGE = 0.02  # 密度过低归一化除数
_SENSORY_HIGH_RANGE = 0.12  # 密度过高归一化除数
# 语义平滑度
_SEMANTIC_MIN_SENT_LEN = 5  # 最短句长
_SEMANTIC_MIN_COUNT = 3  # 最少句数
_SEMANTIC_AI_THRESHOLD = 0.92  # 相似度 ≥ 此值 → 极AI
_SEMANTIC_HUMAN_THRESHOLD = 0.80  # 相似度 ≤ 此值 → 人类
_SEMANTIC_RANGE = 0.12  # 归一化范围
# N-gram 权重
_NGRAM_3_WEIGHT = 0.6
_NGRAM_4_WEIGHT = 0.4
# 主函数
_MIN_TEXT_LENGTH = 50  # 最短检测文本长度（字符）
# 问题告警阈值
_ISSUE_REPETITION = 0.4
_ISSUE_CLICHE = 0.3
_ISSUE_SENTENCE_VARIANCE = 0.5
_ISSUE_LEXICAL = 0.4
_ISSUE_PUNCTUATION = 0.5
_ISSUE_DIALOGUE = 0.3
_ISSUE_SENSORY = 0.3
_ISSUE_SEMANTIC = 0.5
# 质量标准
_QUALITY_PASS = 0.3

# ========== 模板化表达词表 ==========
CLICHE_PATTERNS = [
    # 连接词/过渡词
    "就在这时",
    "说时迟那时快",
    "只见那",
    "只见",
    "而此时",
    "然而就在",
    "却在这时",
    "正在这时",
    "不由得",
    "不由得一",
    "不由得",
    # 泛化形容词
    "精彩的",
    "美妙的",
    "无与伦比的",
    "极其",
    "相当的",
    "颇为",
    "十分",
    "非常之",
    "异常的",
    "惊人的",
    # 情绪词堆砌
    "勃然大怒",
    "大惊失色",
    "喜出望外",
    "欣喜若狂",
    "怒不可遏",
    "顿时",
    "瞬间",
    "刹那",
    "眨眼间",
    # 套路描写
    "嘴角微微上扬",
    "嘴角勾起一抹",
    "眉头紧锁",
    "深吸一口气",
    "眼中闪过",
    "眸中闪过",
    "目光一凝",
    "双手紧握",
    "浑身一颤",
    "心头一跳",
    "心中一凛",
    "心头一沉",
    "不敢置信",
    "难以置信",
    "一脸震惊",
    "满脸惊愕",
    "冷哼一声",
    "仰天大笑",
    "放声大笑",
    "一道寒光",
    "寒光一闪",
    "刀光剑影",
    "剑芒",
]

# ========== 情绪/感官词表 ==========
SENSORY_WORDS = {
    "视觉": [
        "看见",
        "看到",
        "映入眼帘",
        "注视",
        "凝视",
        "视野",
        "光线",
        "颜色",
        "红色",
        "金色",
        "白色",
        "漆黑",
    ],
    "听觉": [
        "听到",
        "听见",
        "回荡",
        "回响",
        "轰鸣",
        "轰鸣声",
        "声音",
        "喧哗",
        "寂静",
        "沉默",
        "尖锐",
        "沙哑",
    ],
    "嗅觉": [
        "闻到",
        "嗅到",
        "弥漫",
        "充斥",
        "芳香",
        "恶臭",
        "血腥味",
        "泥土味",
        "清新的",
    ],
    "触觉": [
        "感到",
        "感受",
        "触碰",
        "抚摸",
        "刺痛",
        "冰凉",
        "炽热",
        "温暖",
        "刺痛",
        "灼烧",
    ],
    "味觉": ["品尝", "入口", "甘甜", "苦涩", "辛辣", "酸涩"],
}

EMOTION_WORDS = [
    "愤怒",
    "高兴",
    "悲伤",
    "恐惧",
    "惊讶",
    "平静",
    "紧张",
    "兴奋",
    "狂喜",
    "绝望",
    "不安",
    "满足",
    "羞耻",
    "嫉妒",
    "崇拜",
    "厌恶",
    "心疼",
    "憋屈",
    "爽快",
    "舒畅",
    "压抑",
    "难受",
    "委屈",
    "心酸",
]

ALL_SENSORY_EMOTION = set()
for words in SENSORY_WORDS.values():
    ALL_SENSORY_EMOTION.update(words)
ALL_SENSORY_EMOTION.update(EMOTION_WORDS)

# ========== 维度权重 ==========
AI_WEIGHTS = {
    "repetition_ngram": 0.25,
    "sentence_length_variance": 0.20,
    "lexical_diversity": 0.18,
    "cliche_ratio": 0.15,
    "punctuation_rhythm": 0.10,
    "dialogue_ratio": 0.07,
    "sensory_emotion_density": 0.03,
    "semantic_smoothness": 0.02,
}


# ========== 结果类型 ==========
class AIStyleMetrics(TypedDict):
    repetition_ngram: float  # [0, 1] 越高越 AI
    sentence_length_variance: float
    lexical_diversity: float  # [0, 1] 越低越 AI（词汇单一）
    cliche_ratio: float  # [0, 1] 越高越 AI
    punctuation_rhythm: float  # [0, 1] 越高越 AI
    dialogue_ratio: float  # [0, 1] 偏离 0.3-0.5 区间越 AI
    sensory_emotion_density: float  # [0, 1] 过低像说明书
    semantic_smoothness: float  # [0, 1] 越高越 AI


class AIStyleResult(TypedDict):
    ai_style_score: float  # 0-1，越低越好，≤0.3合格
    metrics: AIStyleMetrics
    issues: list[str]  # 发现的具体问题列表
    details: dict  # 各维度详细数据


# ========== 分词工具 ==========
_jieba_available = False
try:
    import jieba

    jieba.initialize()
    _jieba_available = True
except ImportError:
    logger.warning("jieba 未安装，AI味检测将使用字符级分析作为降级方案")

    def _char_tokenize(text: str) -> list[str]:
        """降级方案：按字符分词（每2个字符作为一个词）"""
        return [text[i : i + 2] for i in range(len(text) - 1)]


def _tokenize(text: str) -> list[str]:
    """分词：优先用 jieba，降级用字符级"""
    if _jieba_available:
        return list(jieba.cut(text, cut_all=False))
    return _char_tokenize(text)


# ========== N-gram 重复率 ==========
def _compute_ngram_repetition(text: str, n: int = 3) -> float:
    """
    计算 N-gram 重复率。
    重复率 = 重复的 N-gram 数 / 总 N-gram 数
    AI 写作倾向于在局部使用相同的高频词汇组合，导致重复率偏高。
    """
    words = _tokenize(text)
    # 过滤停用词和单字符
    words = [w for w in words if len(w) >= 1 and not w.isspace()]
    if len(words) < n:
        return 0.0

    ngrams = []
    for i in range(len(words) - n + 1):
        ngram = " ".join(words[i : i + n])
        ngrams.append(ngram)

    if not ngrams:
        return 0.0

    # 计算重复率
    unique_ngrams = len(set(ngrams))
    repetition_ratio = 1.0 - (unique_ngrams / len(ngrams))
    return min(1.0, repetition_ratio)


def _repetition_ngram_score(text: str) -> float:
    """综合 3-gram 和 4-gram"""
    r3 = _compute_ngram_repetition(text, 3)
    r4 = _compute_ngram_repetition(text, 4)
    return r3 * _NGRAM_3_WEIGHT + r4 * _NGRAM_4_WEIGHT


# ========== 句长波动 ==========
def _sentence_length_variance_score(text: str) -> float:
    """
    计算句长标准差。AI 写作句子长度趋同，标准差偏小。
    人类写作句长变化大，标准差较大。
    返回 0-1，值越高越 AI。
    """
    # 简单按句号/感叹号/问号/换行分句
    sentences = re.split(r"[。！？\n]+", text)
    sentences = [
        s.strip() for s in sentences if s.strip() and len(s) > _MIN_SENTENCE_LENGTH
    ]

    if len(sentences) < _MIN_SENTENCE_COUNT:
        return 0.0

    lengths = [len(s) for s in sentences]
    mean_len = sum(lengths) / len(lengths)

    if mean_len == 0:
        return 0.0

    variance = sum((ln - mean_len) ** 2 for ln in lengths) / len(lengths)
    std_dev = math.sqrt(variance)

    # 正常人类写作句长标准差约 15-25
    # AI 写作标准差约 5-12
    # 归一化：std_dev <= 8 → 1.0（极AI），std_dev >= 20 → 0.0（人类）
    if std_dev <= _STD_DEV_AI_THRESHOLD:
        return 1.0
    if std_dev >= _STD_DEV_HUMAN_THRESHOLD:
        return 0.0
    return 1.0 - (std_dev - _STD_DEV_AI_THRESHOLD) / _STD_DEV_RANGE


# ========== 词汇多样性 ==========
def _lexical_diversity_score(text: str) -> float:
    """
    计算词汇多样性 = unique_words / total_words。
    AI 写作倾向于重复使用相同的高频词，同义词替换有限。
    返回 0-1，值越高越 AI。
    """
    words = _tokenize(text)
    # 过滤停用词
    words = [w.strip() for w in words if w.strip() and len(w.strip()) > 1]
    if not words:
        return 0.0

    unique_count = len(set(words))
    total_count = len(words)
    diversity = unique_count / total_count if total_count > 0 else 0.0

    # diversity <= 0.3 → 1.0（极AI），diversity >= 0.7 → 0.0（人类）
    if diversity <= _DIVERSITY_AI_THRESHOLD:
        return 1.0
    if diversity >= _DIVERSITY_HUMAN_THRESHOLD:
        return 0.0
    # 反向：词汇多样性低 = AI味高
    return 1.0 - (diversity - _DIVERSITY_AI_THRESHOLD) / _DIVERSITY_RANGE


# ========== 标点节奏变异系数 ==========
def _punctuation_rhythm_score(text: str) -> float:
    """
    计算标点间隔的变异系数(CV)。
    AI 写作标点分布过于均匀，CV 偏小。
    返回 0-1，值越高越 AI。
    """
    # 找到所有标点位置
    punctuation_positions = []
    for i, char in enumerate(text):
        if char in "，。！？；：、":
            punctuation_positions.append(i)

    if len(punctuation_positions) < _MIN_PUNCTUATION_COUNT:
        return 0.0

    # 计算相邻标点间隔
    intervals = []
    for i in range(1, len(punctuation_positions)):
        interval = punctuation_positions[i] - punctuation_positions[i - 1]
        intervals.append(interval)

    if not intervals:
        return 0.0

    mean_interval = sum(intervals) / len(intervals)
    if mean_interval == 0:
        return 0.0

    variance = sum((iv - mean_interval) ** 2 for iv in intervals) / len(intervals)
    std_dev = math.sqrt(variance)
    cv = std_dev / mean_interval  # 变异系数

    # CV <= 0.3 → 极AI，CV >= 0.8 → 人类
    if cv <= _CV_AI_THRESHOLD:
        return 1.0
    if cv >= _CV_HUMAN_THRESHOLD:
        return 0.0
    return 1.0 - (cv - _CV_AI_THRESHOLD) / _CV_RANGE


# ========== 对白比例 ==========
def _dialogue_ratio_score(text: str) -> float:
    """
    计算对白字数 / 总字数。
    全对白或无对白的极端结构都是 AI 特征。
    正常范围：0.25-0.45 之间越正常。
    返回 0-1，值越高越 AI。
    """
    total_chars = len(text)
    if total_chars == 0:
        return 0.0

    # 匹配引号内的对白（包括中文引号）
    dialogues = re.findall(r'[""\'\']([^""\'\']{3,})[""\'\']', text)
    dialogue_chars = sum(len(d) for d in dialogues)

    ratio = dialogue_chars / total_chars

    # 正常范围 _DIALOGUE_LOWER-_DIALOGUE_UPPER → 0.0
    # 偏离越多 → 越 AI
    if _DIALOGUE_LOWER <= ratio <= _DIALOGUE_UPPER:
        return 0.0
    if ratio < _DIALOGUE_LOWER:
        # 对白过少
        return min(1.0, (_DIALOGUE_LOWER - ratio) / _DIALOGUE_LOW_RANGE)
    # 对白过多
    return min(1.0, (ratio - _DIALOGUE_UPPER) / _DIALOGUE_HIGH_RANGE)


# ========== 情绪/感官词密度 ==========
def _sensory_emotion_density_score(text: str) -> float:
    """
    计算情绪词/感官词密度。
    过低像说明书（AI味），过高堆砌情绪（也不好）。
    正常范围：0.02-0.08 之间。
    返回 0-1，值越高越 AI。
    """
    words = _tokenize(text)
    total_words = len(words)
    if total_words == 0:
        return _SENSORY_DEFAULT  # 无法判断，默认中位

    match_count = sum(1 for w in words if w in ALL_SENSORY_EMOTION)
    density = match_count / total_words

    # 正常密度 _SENSORY_LOWER-_SENSORY_UPPER → 0.0（无AI味）
    # 过低或过高 → 越 AI
    if _SENSORY_LOWER <= density <= _SENSORY_UPPER:
        return 0.0
    if density < _SENSORY_LOWER:
        return min(1.0, (_SENSORY_LOWER - density) / _SENSORY_LOW_RANGE)
    return min(1.0, (density - _SENSORY_UPPER) / _SENSORY_HIGH_RANGE)


# ========== 语义平滑度 ==========
_semantic_available = False
_embedding_model = None


def _init_embedding() -> bool | None:
    """延迟加载 embedding 模型。

    优先级：
      1. 远程 API (OpenAI 兼容, 如 SiliconFlow)
      2. 本地 HuggingFace 模型 (需要 langchain-community)
    """
    global _semantic_available, _embedding_model
    if _semantic_available:
        return True

    from novelfactory.config.settings import settings

    # 方案 1: 远程 API (langchain-openai 已安装)
    if settings.EMBEDDING_BASE_URL:
        try:
            from langchain_openai import OpenAIEmbeddings

            _embedding_model = OpenAIEmbeddings(
                model=settings.EMBEDDING_MODEL,
                openai_api_key=settings.EMBEDDING_API_KEY,
                openai_api_base=settings.EMBEDDING_BASE_URL,
            )
            _semantic_available = True
            logger.info(
                "语义平滑度检测: 远程 Embedding 加载成功 (%s)",
                settings.EMBEDDING_BASE_URL,
            )
            return True
        except Exception as e:
            logger.warning(f"语义平滑度检测: 远程 Embedding 加载失败: {e}")

    # 方案 2: 本地 HuggingFace 模型 (需要 langchain-community)
    try:
        from langchain_community.embeddings import HuggingFaceBgeEmbeddings

        _embedding_model = HuggingFaceBgeEmbeddings(
            model_name=settings.EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        _semantic_available = True
        return True
    except ImportError:
        logger.warning("langchain_community 未安装，跳过语义平滑度检测")
        return False
    except Exception as e:
        logger.warning(f"Embedding 模型加载失败: {e}")
        return False


def _semantic_smoothness_score(text: str) -> float:
    """
    计算相邻句的语义相似度均值。
    AI 写作句间衔接过于顺滑，相邻句 embedding 相似度偏高。
    返回 0-1，值越高越 AI。
    """
    if not _init_embedding():
        return 0.0  # 降级：不检测

    # 分句
    sentences = re.split(r"[。！？\n]+", text)
    sentences = [
        s.strip() for s in sentences if s.strip() and len(s) > _SEMANTIC_MIN_SENT_LEN
    ]
    if len(sentences) < _SEMANTIC_MIN_COUNT:
        return 0.0

    # v5.4-fix: 先展平为字符串列表再调用 embed_documents（之前传入 tuple 列表导致类型错误）
    try:
        flat_texts = [
            s for pair in zip(sentences[:-1], sentences[1:], strict=False) for s in pair
        ]
        if not flat_texts:
            return 0.0
        raw_embeddings = _embedding_model.embed_documents(flat_texts)
        # 重组为句子对 embedding: [(emb_s0, emb_s1), (emb_s0, emb_s1), ...]
        pair_embeddings = [
            (raw_embeddings[i], raw_embeddings[i + 1])
            for i in range(0, len(raw_embeddings), 2)
        ]
        # 计算每对的余弦相似度（embedding 已归一化，点积=余弦）
        similarities = []
        for emb_a, emb_b in pair_embeddings:
            sim = sum(a * b for a, b in zip(emb_a, emb_b, strict=False))
            similarities.append(sim)

        if not similarities:
            return 0.0

        mean_sim = sum(similarities) / len(similarities)

        # mean_sim >= _SEMANTIC_AI_THRESHOLD → 极AI，mean_sim <= _SEMANTIC_HUMAN_THRESHOLD → 人类
        if mean_sim >= _SEMANTIC_AI_THRESHOLD:
            return 1.0
        if mean_sim <= _SEMANTIC_HUMAN_THRESHOLD:
            return 0.0
        return (mean_sim - _SEMANTIC_HUMAN_THRESHOLD) / _SEMANTIC_RANGE

    except Exception as e:
        logger.warning(f"语义平滑度检测失败: {e}")
        return 0.0


# ========== 主函数 ==========
# ── 题材感知豁免模板词表 ──
_GENRE_CLICHE_ALLOWLIST: dict[str, set[str]] = {
    # 系统流："叮""恭喜宿主"等系统提示不算 AI 味
    "系统流": {"恭喜宿主", "叮", "系统提示", "宿主", "签到成功", "任务完成", "抽奖中"},
    # 爽文：套路化打脸描写不算 AI 味
    "爽文": {
        "全场震惊",
        "倒吸一口凉气",
        "脸色大变",
        "不敢置信",
        "脸色铁青",
        "哑口无言",
        "全场寂静",
        "落针可闻",
        "嘴角微微上扬",
        "冷哼一声",
        "不屑一顾",
        "众人惊呆了",
    },
    # 无敌流：碾压式描写不算 AI 味
    "无敌流": {
        "一招",
        "秒杀",
        "碾压",
        "不堪一击",
        "蝼蚁",
        "全场震惊",
        "脸色大变",
        "膜拜",
    },
    # 重生：预知未来相关不算 AI 味
    "重生": {
        "上一世",
        "前世",
        "重生前",
        "重活一世",
        "这一世",
        "既然老天给我",
        "绝不会",
    },
    # 科幻/末世：科技类固定描写不算 AI 味
    "末世": {"末日", "丧尸", "异能觉醒", "生存", "基地"},
    "科幻": {"星际", "虫族", "机甲", "战舰", "能量", "量子"},
    # 游戏：游戏数据不算 AI 味
    "游戏": {"等级", "经验值", "装备", "技能", "MP", "HP", "攻击力"},
    # 悬疑灵异：氛围描写不算 AI 味
    "悬疑灵异": {"阴森", "诡异", "毛骨悚然", "背后一凉", "不对劲"},
}


def _get_cliche_allowlist(genre: str | None) -> set[str]:
    """按题材获取 AI 味豁免模板词表。"""
    if not genre:
        return set()
    from novelfactory.config.constants import resolve_genre

    resolved = resolve_genre(genre)
    return _GENRE_CLICHE_ALLOWLIST.get(resolved, set())


def _cliche_ratio_score(text: str, genre: str | None = None) -> float:
    """
    计算模板化表达命中比例。
    支持题材感知豁免：特定题材中常见的套路表达不计入 AI 味。

    返回 0-1，值越高越 AI。
    """
    text_lower = text

    # 获取当前题材的豁免词表
    allowlist = _get_cliche_allowlist(genre)

    hit_count = 0
    for pattern in CLICHE_PATTERNS:
        if pattern in text_lower:
            if pattern in allowlist:
                continue  # 豁免：不视为 AI 味
            hit_count += 1

    # hit_rate = 命中词数 / 总词表数
    hit_rate = hit_count / len(CLICHE_PATTERNS)
    return min(1.0, hit_rate * _CLICHE_AMPLIFICATION)


def analyze_ai_style(text: str, genre: str | None = None) -> AIStyleResult:
    """
    分析文本的 AI 味综合评分。

    Args:
        text: 待检测的中文小说文本
        genre: 题材名称（如 "爽文"、"系统流"），用于题材感知豁免

    Returns:
        AIStyleResult，包含：
        - ai_style_score: 0-1，越低越好，≤0.3合格
        - metrics: 8维统计特征详情
        - issues: 发现的具体问题列表
        - details: 各维度原始数值
    """
    if not text or len(text.strip()) < _MIN_TEXT_LENGTH:
        return AIStyleResult(
            ai_style_score=0.0,
            metrics=AIStyleMetrics(
                repetition_ngram=0.0,
                sentence_length_variance=0.0,
                lexical_diversity=0.0,
                cliche_ratio=0.0,
                punctuation_rhythm=0.0,
                dialogue_ratio=0.0,
                sensory_emotion_density=0.0,
                semantic_smoothness=0.0,
            ),
            issues=["文本过短，无法准确检测"],
            details={"is_short_text": True},
        )

    # 逐维度计算
    metrics_raw = {
        "repetition_ngram": _repetition_ngram_score(text),
        "sentence_length_variance": _sentence_length_variance_score(text),
        "lexical_diversity": _lexical_diversity_score(text),
        "cliche_ratio": _cliche_ratio_score(text, genre=genre),
        "punctuation_rhythm": _punctuation_rhythm_score(text),
        "dialogue_ratio": _dialogue_ratio_score(text),
        "sensory_emotion_density": _sensory_emotion_density_score(text),
        "semantic_smoothness": _semantic_smoothness_score(text),
    }

    # 加权求和得到 AI 味指数
    ai_style_score = sum(metrics_raw[key] * AI_WEIGHTS[key] for key in AI_WEIGHTS)

    # 归一化到 0-1
    ai_style_score = min(1.0, max(0.0, ai_style_score))

    # 生成问题列表
    issues = []
    if metrics_raw["repetition_ngram"] > _ISSUE_REPETITION:
        issues.append(
            f"【N-gram重复】偏高({metrics_raw['repetition_ngram']:.2f})，建议减少连续重复描写，增加句式变化"
        )
    if metrics_raw["cliche_ratio"] > _ISSUE_CLICHE:
        issues.append(
            '【模板化表达】检测到套路化表达较多，如"就在这时"、"不由得"等，建议替换为更自然的描写'
        )
    if metrics_raw["sentence_length_variance"] > _ISSUE_SENTENCE_VARIANCE:
        issues.append("【句长波动】句子长度过于趋同，缺乏长短变化，建议穿插长短句")
    if metrics_raw["lexical_diversity"] > _ISSUE_LEXICAL:
        issues.append("【词汇多样性】词汇重复率偏高，建议增加同义词替换和多样化表达")
    if metrics_raw["punctuation_rhythm"] > _ISSUE_PUNCTUATION:
        issues.append("【标点节奏】标点使用过于规律，建议增加标点变化打破节奏")
    if metrics_raw["dialogue_ratio"] > _ISSUE_DIALOGUE:
        issues.append("【对白比例】对白比例异常，需检查是否过少或过多")
    if metrics_raw["sensory_emotion_density"] > _ISSUE_SENSORY:
        issues.append("【感官描写】情绪/感官词密度异常，建议增加五感描写或降低堆砌")
    if metrics_raw["semantic_smoothness"] > _ISSUE_SEMANTIC:
        issues.append(
            "【语义平滑度】句间衔接过于顺滑，缺乏思维断层，建议增加场景切换或情绪跳跃"
        )

    # 转换指标为标准格式
    metrics = AIStyleMetrics(
        repetition_ngram=round(metrics_raw["repetition_ngram"], 4),
        sentence_length_variance=round(metrics_raw["sentence_length_variance"], 4),
        lexical_diversity=round(metrics_raw["lexical_diversity"], 4),
        cliche_ratio=round(metrics_raw["cliche_ratio"], 4),
        punctuation_rhythm=round(metrics_raw["punctuation_rhythm"], 4),
        dialogue_ratio=round(metrics_raw["dialogue_ratio"], 4),
        sensory_emotion_density=round(metrics_raw["sensory_emotion_density"], 4),
        semantic_smoothness=round(metrics_raw["semantic_smoothness"], 4),
    )

    return AIStyleResult(
        ai_style_score=round(ai_style_score, 4),
        metrics=metrics,
        issues=issues,
        details={
            "word_count": len(_tokenize(text)),
            "sentence_count": len(re.split(r"[。！？\n]+", text)),
            "jieba_available": _jieba_available,
            "semantic_available": _semantic_available,
            "weights_used": AI_WEIGHTS,
        },
    )


# ========== 命令行测试 ==========
if __name__ == "__main__":
    test_texts = [
        # AI 味明显的文本
        "就在这时，只见那主角不由得深吸一口气，眼中闪过一丝精光，嘴角微微上扬，说时迟那时快，只见那反派顿时勃然大怒，不敢置信地看着他。主角冷哼一声，不由得一惊，精彩的表演，无与伦比的力量，令在场所有人都大惊失色，精彩极了。",
        # 相对自然的文本
        '老张推开门，屋里烟雾缭绕。他皱了皱眉，把窗户推开一条缝。\n\n"你又抽烟。"他说。\n\n对面的人没吭声，手指夹着的烟已经燃到了滤嘴。他叹了口气，在旁边坐下。沉默了好一会儿，那人才开口：\n\n"我有件事想跟你说。"\n\n他没有追问，只是等着。窗外的风灌进来，带着一股泥土的气息。',
    ]

    for i, text in enumerate(test_texts):
        print(f"\n{'=' * 40}")
        print(f"测试文本 {i + 1}:")
        result = analyze_ai_style(text)
        print(f"AI味指数: {result['ai_style_score']:.4f}")
        print(
            f"是否合格(≤{_QUALITY_PASS}): {'✅' if result['ai_style_score'] <= _QUALITY_PASS else '❌'}"
        )
        print("问题列表:")
        for issue in result["issues"]:
            print(f"  - {issue}")
        print(f"各维度: {result['metrics']}")
