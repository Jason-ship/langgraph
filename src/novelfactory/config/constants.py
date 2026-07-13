"""NovelFactory centralized constants.

所有跨文件引用或在多处被引用的魔法数字集中于此。
模块级私有常量（`_MAX_RETRIES`、`_BASE_DELAY` 等）仍保留在原模块中定义，
通过 `from novelfactory.config.constants import ...` 导入以获得唯一真实来源。

版本：v1.0.0
创建日期：2026-06-23（Phase 4.1 常量统一化重构）
"""

# ═══════════════════════════════════════════════════════════════════════════════
# 评分阈值（v5.0 双条件评分系统）
# 来源: graph/crews/writing_crew.py + analysis/quality_scorer.py
# ═══════════════════════════════════════════════════════════════════════════════

QUALITY_SCORE_THRESHOLD = (
    85.0  # 四维评分通过线（爽文版：降低文学性权重，侧重节奏/爽点）
)
COMPOSITE_THRESHOLD = 0.65  # 综合指标通过线（爽文版降低门槛）
AI_STYLE_THRESHOLD = 0.4  # AI味指数合格线（爽文版更宽松，≤0.4为合格）
LAO_SHU_THRESHOLD = 65.0  # 老书虫评分合格线（爽文版：节奏快、爽点密更重要）

# writing_crew.py 内部使用的细化阈值
EXCELLENT_THRESHOLD = 85  # 优秀线（quality_score ≥ 85 → 通过候选）
GOOD_THRESHOLD = 75  # 良好线
POOR_THRESHOLD = 55  # 差线（< 55 → 需重写）
COMPOSITE_OK_THRESHOLD = 0.65  # 综合指标通过（与 COMPOSITE_THRESHOLD 同步）
COMPOSITE_RELAXED_THRESHOLD = 0.5  # 宽松通过线（评分器故障时启用）
SCORER_FAULT_THRESHOLD = 0.1  # 评分器故障判定阈值

# ═══════════════════════════════════════════════════════════════════════════════
# 重试与超时
# 来源: agents/infra/retry.py
# ═══════════════════════════════════════════════════════════════════════════════

MAX_RETRIES = 3  # 最大重试次数
BASE_DELAY = 2  # 基础退避延迟（秒），指数增长：2, 4, 8
TIMEOUT_EXTRACT = 120  # 提取/评分类任务超时（秒）
TIMEOUT_SHORT = 180  # 短文本生成超时（秒）
TIMEOUT_LONG = 300  # 长章节生成超时（秒）
DEFAULT_TIMEOUT = 900  # 兜底超时（秒）

RETRY_ON_HTTP_STATUS = {500, 502, 503, 504}  # 可重试 HTTP 状态码
NO_RETRY_ON_HTTP_STATUS = {400, 401, 403}  # 不重试 HTTP 状态码
RETRY_IMMEDIATE_HTTP_STATUS = 429  # 立即重试（限流）

# ═══════════════════════════════════════════════════════════════════════════════
# 业务常量
# 来源: 多个文件
# ═══════════════════════════════════════════════════════════════════════════════

MAX_REWRITE_ATTEMPTS = 5  # 最大重写次数（writing_crew 循环守卫）
CHAPTER_MIN_WORD_COUNT = 1500  # 章节最少字数（config/settings.py）
DRAFT_PREVIEW_LENGTH = 500  # 草稿预览截断长度
GUIDANCE_MAX_LENGTH = 2000  # 人工指导最大长度
MIN_CHAPTER_TEXT_LENGTH = 500  # 章节文本最短判定长度

# ═══════════════════════════════════════════════════════════════════════════════
# Crew Supervisor
# 来源: crews/supervisor.py
# ═══════════════════════════════════════════════════════════════════════════════

MAX_SUPERVISOR_ITERATIONS = 20  # Supervisor 循环安全上限

# ═══════════════════════════════════════════════════════════════════════════════
# 图构建常量（来源: graph/new_builder.py + graph/routing.py + graph/nodes/supervisor.py）
# ═══════════════════════════════════════════════════════════════════════════════

RECURSION_LIMIT = 5000  # 根图递归限制
SUBGRAPH_RECURSION_LIMIT = 200  # 子图递归限制（writing/media/sync/setup）
FALLBACK_TARGET_CHAPTERS = 1000  # 默认目标章节数（300万字）
MIN_DISPLAY_TEXT_LENGTH = 20  # 最小显示文本长度
DISPLAY_TEXT_MAX_LENGTH = 2000  # 最大显示文本长度
COMPRESS_KEEP_RECENT = 5  # 压缩保留最近消息数（Supervisor 消息压缩）
COMPRESS_KEEP_RECENT_CHAPTERS = 50  # 压缩保留最近章节数（novel_state reducer 默认值）
COMPRESS_OLD_TRUNC_LEN = 50  # 旧章节字符串摘要截断长度
COST_ROUND_DIGITS = 4  # 费用估算小数精度
ENDGAME_CHAPTERS_REMAINING = 10  # 终局剩余章节阈值
FORESHADOWING_HIGH_PRIORITY = 7  # 伏笔高优先级阈值
MAX_MESSAGES = 100  # Supervisor 消息压缩上限

# ── Phase 字符串常量（与 NovelFactoryState.current_phase Literal 对齐）──
PHASE_SETUP = "setup"
PHASE_WRITING = "writing"
PHASE_MEDIA = "media"
PHASE_SYNC = "sync"
PHASE_DONE = "done"

ALL_PHASES: tuple[str, ...] = (
    PHASE_SETUP,
    PHASE_WRITING,
    PHASE_MEDIA,
    PHASE_SYNC,
    PHASE_DONE,
)

# phase_labels 映射表（来源: graph/nodes/supervisor.py）
PHASE_LABELS: dict[str, str] = {
    PHASE_SETUP: "开始项目设定",
    PHASE_WRITING: "进入写作阶段",
    PHASE_MEDIA: "进入媒体生成",
    PHASE_SYNC: "进入同步阶段",
    PHASE_DONE: "全部章节完成",
}

# ═══════════════════════════════════════════════════════════════════════════════
# Writing Crew 细化常量（来源: graph/crews/writing_crew.py + writing_nodes/routing.py）
# ═══════════════════════════════════════════════════════════════════════════════

REFINE_MAX_ATTEMPTS: dict[str, int] = {
    "high": 1,  # 80-89: fast pass after 1 refine
    "mid": 2,  # 60-79: allow 2 refine attempts
}

# ═══════════════════════════════════════════════════════════════════════════════
# RAG 缓存
# 来源: agents/infra/rag_cache.py
# ═══════════════════════════════════════════════════════════════════════════════

RAG_CACHE_DEFAULT_MAXSIZE = 500  # 默认缓存容量
RAG_CACHE_DEFAULT_TTL = 3600.0  # 默认过期时间（秒，1 小时）
RAG_CACHE_KEY_TRIM = 16  # SHA256 哈希摘要截断长度

# ═══════════════════════════════════════════════════════════════════════════════
# Embedding 维度
# 来源: store/embedding.py + store/milvus_store.py + store/guide_store.py
# ═══════════════════════════════════════════════════════════════════════════════

EMBEDDING_DIMS_DEFAULT = 1024  # Qwen3-Embedding 默认维度

# ═══════════════════════════════════════════════════════════════════════════════
# 飞书集成常量
# 来源: integrations/feishu/feishu_toolkit.py + feishu_api.py + event_handler.py
# ═══════════════════════════════════════════════════════════════════════════════

FEISHU_LARK_TIMEOUT = 30  # lark-cli 默认超时（秒）
FEISHU_LARK_DOC_TIMEOUT = 60  # 文档操作超时（秒）
FEISHU_MAX_DOC_CHARS = 50000  # 文档单次上传最大字符数
FEISHU_HTTPX_TIMEOUT = 30.0  # httpx 客户端超时（秒）
FEISHU_REQUESTS_TIMEOUT = 15  # 消息发送超时（秒）

# ═══════════════════════════════════════════════════════════════════════════════
# 日志轮转
# 来源: agents/infra/logger.py
# ═══════════════════════════════════════════════════════════════════════════════

LOG_MAX_BYTES = 10 * 1024 * 1024  # 日志文件最大字节数（10 MB）
LOG_BACKUP_COUNT = 5  # 备份文件数量

# ═══════════════════════════════════════════════════════════════════════════════
# 熔断器配置
# 来源: agents/infra/circuit_breaker.py
# ═══════════════════════════════════════════════════════════════════════════════

CIRCUIT_BREAKER_CONFIG: dict[str, dict] = {
    "matrix": {"max_failures": 20, "cooldown_seconds": 30, "timeout_seconds": 60.0},
    "ark": {"max_failures": 20, "cooldown_seconds": 30, "half_open_max_calls": 2},
    "deepseek": {"max_failures": 20, "cooldown_seconds": 30, "half_open_max_calls": 2},
    "siliconflow": {
        "max_failures": 10,
        "cooldown_seconds": 60,
        "half_open_max_calls": 1,
    },
}

# ═══════════════════════════════════════════════════════════════════════════════
# 题材感知评分阈值（v5.5+）
# 来源: analysis/quality_scorer.py + graph/crews/writing_nodes/routing.py
# ═══════════════════════════════════════════════════════════════════════════════

GENRE_THRESHOLDS: dict[str, dict] = {
    # ── 男频 · 按世界观划分 ──
    "玄幻": {
        "quality_score": 85,
        "composite": 0.65,
        "ai_style": 0.30,
        "lao_shu": 65,
        "description": "架空世界+修炼体系+宗门势力，重设定严谨性和升级逻辑",
        "themes": ["东方玄幻", "异世大陆", "王朝争霸", "高武世界"],
    },
    "仙侠": {
        "quality_score": 88,
        "composite": 0.65,
        "ai_style": 0.25,
        "lao_shu": 65,
        "description": "修真体系+渡劫飞升，文风需有古韵，对AI味最严格",
        "themes": ["修真文明", "幻想修仙", "古典仙侠", "神话修真", "现代修真"],
    },
    "奇幻": {
        "quality_score": 82,
        "composite": 0.60,
        "ai_style": 0.35,
        "lao_shu": 62,
        "description": "西方背景+魔法体系，翻译腔和固定描写容忍度中高",
        "themes": ["剑与魔法", "史诗奇幻", "神秘幻想", "历史神话"],
    },
    "武侠": {
        "quality_score": 85,
        "composite": 0.60,
        "ai_style": 0.30,
        "lao_shu": 65,
        "description": "江湖世界+门派纷争+侠义精神，传统文风要求",
        "themes": ["传统武侠", "武侠幻想", "国术无双", "古武未来"],
    },
    "都市": {
        "quality_score": 80,
        "composite": 0.55,
        "ai_style": 0.35,
        "lao_shu": 60,
        "description": "现实社会背景，代入感第一，节奏快爽点密集",
        "themes": ["都市生活", "娱乐明星", "商战职场", "都市异能", "青春校园"],
    },
    "历史": {
        "quality_score": 88,
        "composite": 0.65,
        "ai_style": 0.25,
        "lao_shu": 70,
        "description": "真实/架空历史背景，需严谨考据，毒点最敏感",
        "themes": ["架空历史", "秦汉三国", "两宋元明", "清史民国", "历史传记"],
    },
    "科幻": {
        "quality_score": 85,
        "composite": 0.60,
        "ai_style": 0.30,
        "lao_shu": 65,
        "description": "科技推理+未来世界，逻辑性要求高",
        "themes": ["星际文明", "时空穿梭", "末世危机", "超级科技", "进化变异"],
    },
    "悬疑灵异": {
        "quality_score": 85,
        "composite": 0.60,
        "ai_style": 0.30,
        "lao_shu": 60,
        "description": "气氛和悬念最重要，模板化表达影响恐怖感",
        "themes": ["侦探推理", "灵异探险", "规则怪谈", "惊悚微恐"],
    },
    "游戏": {
        "quality_score": 78,
        "composite": 0.55,
        "ai_style": 0.40,
        "lao_shu": 55,
        "description": "虚拟网游/电竞，系统数据频繁输出，固定表达多",
        "themes": ["电子竞技", "游戏异界", "游戏生涯", "游戏主播"],
    },
    "军事": {
        "quality_score": 88,
        "composite": 0.65,
        "ai_style": 0.25,
        "lao_shu": 70,
        "description": "军旅/战争/谍战背景，严谨性要求最高",
        "themes": ["战争幻想", "军旅生涯", "谍战特工", "抗战烽火"],
    },
    # ── 男频 · 按剧情模式划分 ──
    "系统流": {
        "quality_score": 75,
        "composite": 0.50,
        "ai_style": 0.50,
        "lao_shu": 55,
        "description": "固定'叮''恭喜宿主'等系统提示极多，AI味必须大幅放松",
        "themes": ["签到系统", "抽奖系统", "任务系统", "面板流"],
    },
    "重生": {
        "quality_score": 78,
        "composite": 0.55,
        "ai_style": 0.45,
        "lao_shu": 58,
        "description": "回到过去弥补遗憾，节奏快打脸密集，爽文特性强",
        "themes": ["重生复仇", "重生商战", "重生恋情"],
    },
    "穿越": {
        "quality_score": 80,
        "composite": 0.55,
        "ai_style": 0.40,
        "lao_shu": 60,
        "description": "文化差异降维打击是核心爽点",
        "themes": ["穿越古代", "穿越异界", "穿越历史"],
    },
    "无敌流": {
        "quality_score": 72,
        "composite": 0.45,
        "ai_style": 0.55,
        "lao_shu": 50,
        "description": "开局即无敌全程碾压，不需憋屈铺垫，最不需要文学性",
        "themes": ["开局无敌", "满级大佬", "扮猪吃虎"],
    },
    "种田": {
        "quality_score": 82,
        "composite": 0.58,
        "ai_style": 0.32,
        "lao_shu": 62,
        "description": "慢节奏生活经营，文笔细腻度有要求",
        "themes": ["农业建设", "家族经营", "领地发展"],
    },
    "末世": {
        "quality_score": 80,
        "composite": 0.55,
        "ai_style": 0.38,
        "lao_shu": 60,
        "description": "末日求生+异能觉醒，生存压力驱动节奏",
        "themes": ["末日求生", "丧尸危机", "异能觉醒", "废土重建"],
    },
    # ── 女频 · 按题材划分 ──
    "现代言情": {
        "quality_score": 82,
        "composite": 0.55,
        "ai_style": 0.35,
        "lao_shu": 60,
        "description": "现代都市情感故事，情绪拉扯最重要",
        "themes": ["甜宠", "豪门", "先婚后爱", "久别重逢", "娱乐圈"],
    },
    "古代言情": {
        "quality_score": 85,
        "composite": 0.58,
        "ai_style": 0.30,
        "lao_shu": 62,
        "description": "古代社会背景，文风有古韵要求",
        "themes": ["宫斗宅斗", "架空权谋", "江湖言情", "种田言情"],
    },
    "幻想言情": {
        "quality_score": 80,
        "composite": 0.55,
        "ai_style": 0.38,
        "lao_shu": 58,
        "description": "修仙/玄幻/异能等幻想+言情，想象力可放飞",
        "themes": ["仙侠言情", "玄幻言情", "异能言情"],
    },
    "耽美": {
        "quality_score": 85,
        "composite": 0.58,
        "ai_style": 0.30,
        "lao_shu": 60,
        "description": "纯爱/耽美，情感细腻度要求高",
        "themes": ["现代耽美", "古代耽美", "幻想耽美"],
    },
    # ── 混合/泛类 ──
    "爽文": {
        "quality_score": 75,
        "composite": 0.50,
        "ai_style": 0.50,
        "lao_shu": 55,
        "description": "一切为爽服务，黄金三章+10章单元弧，对文学性要求最低",
        "themes": ["打脸", "逆袭", "装逼", "扮猪吃虎"],
    },
    "脑洞": {
        "quality_score": 78,
        "composite": 0.50,
        "ai_style": 0.45,
        "lao_shu": 58,
        "description": "创意新颖>文笔精致，允许非常规表达",
        "themes": ["都市脑洞", "玄幻脑洞", "悬疑脑洞", "规则类"],
    },
    "同人": {
        "quality_score": 78,
        "composite": 0.50,
        "ai_style": 0.42,
        "lao_shu": 55,
        "description": "基于已有作品再创作，核心受众对文笔容忍度高",
        "themes": ["动漫衍生", "游戏同人", "小说同人"],
    },
    "二次元": {
        "quality_score": 76,
        "composite": 0.50,
        "ai_style": 0.45,
        "lao_shu": 55,
        "description": "猫娘/中二/吐槽等二次元表达，允许大量非常规句式",
        "themes": ["原生幻想", "衍生同人", "搞笑吐槽"],
    },
    "default": {
        "quality_score": 85,
        "composite": 0.65,
        "ai_style": 0.30,
        "lao_shu": 65,
        "description": "通用标准 — 未匹配题材时使用",
        "themes": [],
    },
}

# 题材匹配关键词（按优先级排序，用于 genre 字段自动解析）
_GENRE_KEYWORD_MAP: list[tuple[list[str], str]] = [
    (["系统流", "系统文", "签到系统", "抽奖系统", "任务系统", "面板流"], "系统流"),
    (["无敌流", "开局无敌", "满级大佬"], "无敌流"),
    (["种田", "农家", "农业", "领地经营", "家族经营"], "种田"),
    (["末世", "末日", "废土", "丧尸"], "末世"),
    (["重生", "重生复仇", "重生商战"], "重生"),
    (["穿越", "穿越古代", "穿越异界", "穿越历史"], "穿越"),
    (["玄幻", "东方玄幻", "高武世界", "异世大陆", "王朝争霸"], "玄幻"),
    (["仙侠", "修真", "修仙", "古典仙侠", "洪荒"], "仙侠"),
    (["奇幻", "魔法", "剑与魔法", "史诗奇幻"], "奇幻"),
    (["武侠", "江湖", "侠义", "国术"], "武侠"),
    (["都市", "现代都市", "都市生活", "娱乐明星", "商战"], "都市"),
    (["历史", "架空历史", "穿越历史", "三国", "唐宋", "明清"], "历史"),
    (["科幻", "星际", "太空", "机甲", "未来世界", "人工智能"], "科幻"),
    (["悬疑", "灵异", "推理", "侦探", "怪谈", "惊悚"], "悬疑灵异"),
    (["游戏", "电竞", "网游"], "游戏"),
    (["军事", "军旅", "谍战", "战争"], "军事"),
    (["现代言情", "现言", "甜宠", "豪门", "先婚后爱"], "现代言情"),
    (["古代言情", "古言", "宫斗", "宅斗", "权谋"], "古代言情"),
    (["幻想言情", "幻言", "仙侠言情", "玄幻言情"], "幻想言情"),
    (["耽美", "纯爱", "BL"], "耽美"),
    (["同人", "衍生", "二次创作"], "同人"),
    (["二次元", "中二", "动漫"], "二次元"),
    (["爽文", "爽", "打脸", "逆袭", "装逼", "扮猪吃虎"], "爽文"),
    (["脑洞"], "脑洞"),
]


def resolve_genre(genre: str | None) -> str:
    """根据 genre 字符串自动解析为 GENRE_THRESHOLDS 中的标准题材名。

    匹配优先级：
      1. 精确匹配 genre 键名
      2. 关键词匹配（按关键词列表遍历）
      3. 兜底返回 "default"
    """
    if not genre:
        return "default"
    genre_lower = genre.lower()

    # 1. 精确匹配
    if genre in GENRE_THRESHOLDS:
        return genre

    # 2. 关键词匹配
    for keywords, target in _GENRE_KEYWORD_MAP:
        for kw in keywords:
            if kw in genre or kw in genre_lower:
                return target

    return "default"


def get_genre_threshold(genre: str | None, key: str, default: float = 0.0) -> float:
    """获取指定题材的单个阈值，未匹配返回 default。

    Args:
        genre: 题材名称（如 "爽文"、"玄幻"），支持自动解析
        key: 阈值键名（如 "quality_score"、"composite"）
        default: 未找到时的默认值
    """
    resolved = resolve_genre(genre)
    thresholds = GENRE_THRESHOLDS.get(resolved, GENRE_THRESHOLDS["default"])
    return float(thresholds.get(key, default))


def get_genre_thresholds(genre: str | None) -> dict:
    """获取指定题材的完整阈值字典，未匹配返回 default。"""
    resolved = resolve_genre(genre)
    return GENRE_THRESHOLDS.get(resolved, GENRE_THRESHOLDS["default"])


# ═══════════════════════════════════════════════════════════════════════════════
# DeepSeek Flash 定价（CNY / 百万 tokens）
# 来源: agents/infra/retry.py
# ═══════════════════════════════════════════════════════════════════════════════

TOKENS_PER_MILLION = 1_000_000.0
DEEPSEEK_PROMPT_TOKEN_RATE = 0.5  # ¥0.5 / 1M input tokens
DEEPSEEK_COMPLETION_TOKEN_RATE = 2.0  # ¥2.0 / 1M output tokens

# ═══════════════════════════════════════════════════════════════════════════════
# Fallback 降级默认值 — v5.7 P2-fix: 消除 writing_crew.py 和 quality_panel.py
# 中的重复定义。来源: writing_crew._quality_panel_integrated 降级路径。
# ═══════════════════════════════════════════════════════════════════════════════

FALLBACK_QUALITY_SCORE = 85.0  # 降级默认四维评分
FALLBACK_COMPOSITE_SCORE = 0.7  # 降级默认综合指标
FALLBACK_AI_STYLE_SCORE = 0.3  # 降级默认 AI 味指数
FALLBACK_LAO_SHU_SCORE = 70.0  # 降级默认老书虫评分

FALLBACK_DEGRADE_RESULT: dict = {
    "crew_result": {
        "review_result": {
            "quality_score": FALLBACK_QUALITY_SCORE,
            "programmatic_score": FALLBACK_COMPOSITE_SCORE,
            "passed": True,
            "review_comments": "降级通过",
        }
    },
    "quality_score": FALLBACK_QUALITY_SCORE,
    "programmatic_score": FALLBACK_COMPOSITE_SCORE,
    "ai_style_score": FALLBACK_AI_STYLE_SCORE,
    "lao_shu_chong_score": FALLBACK_LAO_SHU_SCORE,
    "passed": True,
    "toxic_points": [],
    "shuangdian_points": [],
    "ai_style_fix": "",
    "lao_shu_chong_fix": "",
}

# ═══════════════════════════════════════════════════════════════════════════════
# Pipeline 常量（来源: pipeline/scale_manager.py + phase2_manager.py + phase3_manager.py）
# ═══════════════════════════════════════════════════════════════════════════════

# ── 分层大纲 (OutlineManager) ──
OUTLINE_DEFAULT_WORD_COUNT = 3000  # 章默认字数目标

# ── 滑动上下文窗口 (SlidingContextWindow) ──
CONTEXT_KEY_EVENTS_MAX = 50  # 高重要性事件最大返回数
CONTEXT_KEY_EVENT_IMPORTANCE_THRESHOLD = 5  # 高重要性事件阈值
CONTEXT_RECENT_EVENTS_COUNT = 5  # 最近事件返回数
CONTEXT_PREV_SUMMARIES_COUNT = 3  # 前情提要素材数
CONTEXT_LAYER1_COUNT = 5  # 紧邻上下文前 N 章
CONTEXT_EVENT_DEFAULT_IMPORTANCE = 5  # 事件默认重要性
CONTEXT_EVENT_EXTRACT_INPUT_MAX = 4000  # 事件提取输入文本截断

# ── 一致性审计 (ConsistencyAuditor) ──
AUDIT_INTERVAL = 10  # 每 N 章全量审计
AUDIT_FULL_WINDOW = 10  # 全量审计检查最近 N 章
AUDIT_FULL_SETTING_MAX = 3000  # 全量审计设定文本截断
AUDIT_FULL_CHAPTERS_MAX = 8000  # 全量审计章节文本截断
AUDIT_SCORE_LIMIT = 10  # 审计评分趋势返回条数
AUDIT_MAX_CRITICAL_DISPLAY = 3  # 严重问题最多显示条数
AUDIT_MAX_MAJOR_DISPLAY = 3  # 重要问题最多显示条数
AUDIT_TREND_DISPLAY_COUNT = 5  # 趋势显示条数
AUDIT_MIN_TREND_POINTS = 2  # 趋势显示最少数据点

# ── 伏笔管理 (ForeshadowingManager) ──
FORESHADOW_AHEAD = 5  # 伏笔到期前提醒窗口（章）
FORESHADOW_INPUT_MAX = 4000  # 伏笔提取输入文本截断
FORESHADOW_DEFAULT_PRIORITY = 5  # 伏笔默认优先级
FORESHADOW_MAX_OVERDUE_DISPLAY = 5  # 过期伏笔最多显示条数
FORESHADOW_MAX_UPCOMING_DISPLAY = 5  # 即将到期伏笔最多显示条数

# ── 节奏控制 (PacingController) ──
PACING_INPUT_MAX = 3000  # 节奏分析输入文本截断
PACING_CLIMAX_STREAK = 4  # 连续高潮阈值（章）
PACING_SLOW_STREAK = 6  # 连续平淡阈值（章）
PACING_INTENSITY_DIFF = 5  # 强度波动过大阈值
PACING_LOW_INTENSITY = 4  # 低强度判定阈值
PACING_TREND_WINDOW = 10  # 节奏趋势分析窗口（章）
PACING_MIN_SAMPLES = 3  # 节奏分析最少样本数

# ── Phase3 质量衰减检测 ──
QUALITY_DECAY_MIN_SAMPLES = 3  # 质量衰减检测最少样本数
QUALITY_DECAY_THRESHOLD = 5.0  # 质量衰减判定阈值（均值差）
QUALITY_DECAY_RECENT_LIMIT = 10  # 质量衰减检测取最近 N 章

# ═══════════════════════════════════════════════════════════════════════════════
# Node-level RetryPolicy 常量（v5.7 P0-fix: 从 graph/checkpointer.py 迁移至此，
# 消除 agents/infra/ → graph/checkpointer → graph/new_builder 循环导入）
# 来源: graph/checkpointer.py (original) — 移至此处切断循环
# ═══════════════════════════════════════════════════════════════════════════════

from langgraph.types import RetryPolicy  # noqa: E402

DEFAULT_RETRY = RetryPolicy(
    max_attempts=3,
    retry_on=lambda exc: isinstance(exc, TimeoutError | ConnectionError | RuntimeError),
    initial_interval=1.0,
    max_interval=60.0,
    jitter=True,
)
"""通用节点重试策略 — 网络错误和运行时异常时重试 3 次。"""

WRITER_RETRY = RetryPolicy(
    max_attempts=5,
    retry_on=lambda exc: True,
    initial_interval=2.0,
    max_interval=120.0,
    jitter=True,
)
"""写作节点重试策略 — LLM 调用全部重试（最高 5 次）。"""

REVIEWER_RETRY = RetryPolicy(
    max_attempts=3,
    retry_on=lambda exc: True,
    initial_interval=1.0,
    jitter=True,
)
"""审查节点重试策略 — LLM 审查重试 3 次。"""


# ═══════════════════════════════════════════════════════════════════════════════
# v6.3 评分融合配置（evaluation/ 模块）
# 来源: evaluation/verdict/engine.py + evaluation/verdict/calibration.py
# ═══════════════════════════════════════════════════════════════════════════════

# 融合权重
# v7.1: 新增 llm_old_reader + llm_human_like 两个 LLM 语义分析维度。
# 减少程序化权重，增加 LLM 语义权重，让评分更灵活、更智能。
VERDICT_WEIGHTS: dict[str, float] = {
    "quality": 0.25,  # v7.1: 四维 LLM 评分权重继续下调
    "programmatic": 0.30,  # v7.1: 程序化分析权重下调（让位给LLM语义分析）
    "llm_old_reader": 0.10,  # v7.1: NEW — LLM 老书虫语义评分
    "llm_human_like": 0.05,  # v7.1: NEW — LLM AI味语义评分
    "cross_chapter": 0.20,  # 跨章一致性权重
    "debate_penalty": 0.10,  # 辩论惩罚权重（扣分）
}

# v7.0: 迭代次数宽松加分 — 重写/润色多次后逐渐放宽评分，防止死循环
# 让 final_score 随迭代次数逐步提升，而非次数用尽时一刀切强通过
VERDICT_ITERATION_BONUS_REWRITE = 3.0  # 每次重写加 3 分
VERDICT_ITERATION_BONUS_REFINE = 2.0  # 每次润色加 2 分
VERDICT_ITERATION_BONUS_MAX = 10.0  # 封顶 10 分

# v7.3: 长度归一化（Log Length Penalty）
# 参考 Lost in Stories (微软, 2026) 消除 Verbosity Bias 的思路：
#   论文 CED 使用线性归一化（e*10000/w），用于模型间错误密度比较。
#   本项目改用对数归一化（score /= log2(len/3000+1)），因为：
#     (a) 评分是抽象质量分（0-100），不是错误计数
#     (b) 对数衰减更温和，避免对超长章节过度惩罚
#     (c) 3000 字基准 = 中文章节的典型长度
VERDICT_LENGTH_NORMALIZE = True  # 是否启用长度归一化
VERDICT_NORMALIZE_BASE = 3000  # 基准字数（中文字符）

# 决策阈值
VERDICT_PASS_THRESHOLD = 75.0  # 融合分通过线
VERDICT_REFINE_THRESHOLD = 55.0  # 融合分润色/重写分界

# 辩论惩罚
# v7.6-fix: 降低辩论惩罚力度 — 辩论本质是对章节的深入分析，发现问题是正常的
# 不应过度惩罚。之前 PER_ISSUE=5/CAP=30 导致辩论惩罚占比过高（~30%总分）。
VERDICT_DEBATE_PENALTY_CAP = 18.0  # 辩论惩罚上限（原30.0 → 18.0）
VERDICT_DEBATE_PENALTY_PER_ISSUE = 3.0  # 每个问题扣分（原5.0 → 3.0）
VERDICT_DEBATE_PENALTY_PER_SEVERE = 6.0  # 严重问题额外扣分（原10.0 → 6.0）

# 校准触发条件
CALIBRATION_LLM_VIRTUAL_HIGH = 90.0  # LLM 虚高阈值
CALIBRATION_PROGRAMMATIC_LOW = 0.5  # 程序化分过低阈值
CALIBRATION_SHORT_TEXT_LLM_WEIGHT = 0.8  # 短文本 LLM 权重
CALIBRATION_SEVERE_TOXIC_CAP = 50.0  # 严重毒点分数封顶
