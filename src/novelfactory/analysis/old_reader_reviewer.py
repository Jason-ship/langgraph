"""
老书虫评审模块

从毒点检测和爽点兑现两个方向评估章节。

使用方式：
    from novelfactory.analysis.old_reader_reviewer import review_as_old_reader
    result = review_as_old_reader(chapter_text, context={"outline": "...", "genre": "都市"})
"""

from __future__ import annotations

import logging
from typing import TypedDict

logger = logging.getLogger(__name__)


# ========== 模式匹配常量 ==========
_PATTERN_GROUP_LEN_2 = 2  # 双关键词模式组（keywords1, keywords2）
_PATTERN_GROUP_LEN_3 = 3  # 三关键词模式组（keywords1, keywords2, exclude_words）

# ========== 评分阈值常量 ==========
_VERDICT_EXCELLENT = 85  # 优秀评分阈值（≥85 且无毒点→优秀）
_VERDICT_READABLE = 70  # 可读评分阈值（≥70 → 可读）
_VERDICT_REWRITE = 50  # 需大改评分阈值（≥50 → 需大改）

# ========== 文本长度常量 ==========
_MIN_TEXT_LENGTH = 100  # 最小评审文本长度（低于此长度跳过评审）


# ========== 毒点配置 ==========
TOXIC_POINTS = {
    "NTR": {
        "weight": 50,
        "description": "绿帽/NTR，主角女人被夺走",
        "patterns": [
            # 关键词组合
            (
                ["他的女人", "女主", "他的女友", "他的女朋友", "他最爱的女人"],
                ["被人", "被抢", "被夺", "被侮辱", "被玷污"],
            ),
            (
                ["女友", "女朋友", "老婆", "妻子", "女人"],
                ["背叛", "失身", "失贞", "离开了他", "跟了别人"],
            ),
            (["主角的女人", "主角的女人", "主角的女人"], ["被", "的", ""]),
        ],
        "severity": "extreme",
    },
    "SHENGMU": {
        "weight": 30,
        "description": "圣母/舔狗，无底线原谅敌人",
        "patterns": [
            (
                ["原谅", "算了", "放过", "宽恕", "不追究"],
                ["仇人", "敌人", "仇敌", "曾经伤害过"],
            ),
            (["圣母", "心软"], ["", ""]),
            (["无底线", "无条件"], ["原谅", "宽恕", "容忍"]),
        ],
        "severity": "high",
    },
    "PROTAGONIST_STUPID": {
        "weight": 25,
        "description": "主角智商下线，行为愚蠢",
        "patterns": [
            (["明知", "明明知道"], ["还是", "偏要", "还是去", "依然"]),
            (["愚蠢", "太傻", "太天真"], ["决定", "选择", "相信"]),
            (["这么明显的", "任何人都看得出来"], ["他就是", "他居然不知道"]),
        ],
        "severity": "high",
    },
    "NUE_ZHU": {
        "weight": 25,
        "description": "虐主，长期憋屈压抑",
        "patterns": [
            (["憋屈", "压抑", "受气"], ["整整", "一直", "永远"]),
            (["被欺负", "被打压", "被羞辱"], ["不敢还手", "不敢反抗", "默默忍受"]),
            (["主角的亲人", "主角的家人", "父母"], ["被欺负", "被侮辱", "被害"]),
        ],
        "severity": "high",
    },
    "POWER_BREAK": {
        "weight": 20,
        "description": "战力崩坏，力量体系前后矛盾",
        "patterns": [
            # 需要与前后文对比，这里只做初步检测
            (["上一秒", "刚才", "前面"], ["天下无敌", "无人能敌", "最强"], []),
            (
                ["下一秒", "然后", "接着", "结果"],
                ["被轻易", "不堪一击", "一招", "秒了"],
            ),
            (
                ["力量体系", "战力", "实力"],
                ["突然", "毫无征兆"],
                ["暴涨", "暴涨", "翻倍"],
            ),
        ],
        "severity": "medium",
    },
    "ANTAGONIST_STUPID": {
        "weight": 15,
        "description": "反派降智，工具化无脑行事",
        "patterns": [
            (["反派", "敌人"], ["愚蠢", "脑子不好", "智商"], [""]),
            (["反派", "大反派", "Boss"], ["无脑", "不考虑", "不思考"], [""]),
            (["反派明明", "敌人明明"], ["可以", "有能力"], ["偏要", "就是不用"]),
        ],
        "severity": "medium",
    },
    "CHARACTER_DEATH": {
        "weight": 15,
        "description": "重要角色死亡（强行赚眼泪）",
        "patterns": [
            (["为了救主角", "为主角挡", "代替主角"], ["死了", "牺牲", "倒下", "离世"]),
            (["亲人", "兄弟", "挚友", "爱人"], ["死", "牺牲", "去世", "离去"]),
            (["他死了", "她死了"], ["", ""]),
        ],
        "severity": "medium",
    },
    "WATER_CONTENT": {
        "weight": 15,
        "description": "水文/凑字数，无意义重复",
        "patterns": [
            (["同样的场景", "同样的描写", "再次", "又是"], ["", ""]),
            (["他深深地", "他缓缓地", "他轻轻地"], ["吸了一口气", "说", "走", "看"]),
        ],
        "severity": "medium",
    },
    "SENTIMENTAL_TORTURE": {
        "weight": 10,
        "description": "无意义煽情",
        "patterns": [
            (
                ["哭得", "泣不成声", "泪流满面", "泪如雨下"],
                ["哭了很久", "哭个不停", "哭了整整"],
            ),
        ],
        "severity": "low",
    },
    "MORAL_WRONG": {
        "weight": 10,
        "description": "三观不正情节",
        "patterns": [
            (["杀人", "抢夺", "偷盗"], ["是应该的", "是对的", "天经地义", "理所当然"]),
        ],
        "severity": "low",
    },
}


# ========== 爽点配置 ==========
SHUANGDIAN_POINTS = {
    "打脸": {
        "weight": 1.0,
        "description": "反派嚣张→主角爆发→众人震惊",
        "patterns": [
            (["嚣张", "得意", "趾高气扬", "不可一世"], ["", ""]),
            (["冷哼", "淡淡地", "不屑地"], ["说道", "说", ""]),
            (["众人", "所有人", "在场的"], ["震惊", "惊呆了", "目瞪口呆", "不敢相信"]),
        ],
        "sub_patterns": [
            "不屑一顾",
            "众人惊呆",
            "脸色大变",
            "脸色铁青",
            "哑口无言",
            "不敢置信",
            "倒吸一口凉气",
            "全场寂静",
            "落针可闻",
        ],
    },
    "装逼": {
        "weight": 0.9,
        "description": "隐藏实力被发现，旁人惊愕膜拜",
        "patterns": [
            (["隐藏", "隐藏实力", "不为人知"], ["身份", "背景", "实力"], []),
            (["原来他", "没想到他", "竟然是"], ["", ""]),
            (["众人", "所有人"], ["膜拜", "跪下", "震惊", "惊呼"], []),
        ],
        "sub_patterns": [
            "膜拜",
            "跪下",
            "不可思议",
            "刮目相看",
            "肃然起敬",
        ],
    },
    "逆袭": {
        "weight": 0.85,
        "description": "低谷→反转→高潮",
        "patterns": [
            (["绝境", "困境", "低谷"], ["反转", "逆袭", "翻盘"], []),
            (["绝地反击", "绝境逢生", "绝处逢生"], ["", ""]),
            (["原本以为", "本以为"], ["必输", "必死", "必败"], ["结果", "没想到"]),
        ],
        "sub_patterns": [
            "绝地反击",
            "逆风翻盘",
            "化险为夷",
            "置之死地",
            "峰回路转",
        ],
    },
    "升级": {
        "weight": 0.8,
        "description": "修炼→突破→力量提升",
        "patterns": [
            (["突破", "进阶", "晋升"], ["", ""]),
            (["修为", "境界", "实力"], ["提升", "大涨", "暴涨", "突破"], []),
            (["从", "从一阶"], ["到", "升至", "突破至"], []),
        ],
        "sub_patterns": [
            "突破成功",
            "顺利突破",
            "境界提升",
            "修为大涨",
            "力量暴涨",
        ],
    },
    "感情": {
        "weight": 0.75,
        "description": "误会→解结→关系升华",
        "patterns": [
            (["误会", "误解"], ["解开", "消除", "解除"], []),
            (["坦白", "坦诚"], ["心意", "感情"], []),
            (["感情", "关系"], ["升温", "更进一步", "更进一步"], []),
        ],
        "sub_patterns": [
            "误会解除",
            "重归于好",
            "感情升温",
            "心结解开",
            "坦诚相待",
        ],
    },
    "悬念": {
        "weight": 0.7,
        "description": "章节结尾留钩子",
        "patterns": [
            (["就在这时", "就在此时", "就在此时刻"], ["", ""]),
            (["突然", "忽然"], ["有人", "他", "一个声音"], []),
            (["下一章", "故事远未结束"], ["", ""]),
        ],
        "sub_patterns": [
            "欲知后事如何",
            "请看下回",
            "故事远未结束",
            "但这只是开始",
            "而这，仅仅是",
            "而这一切的背后",
        ],
    },
}


# ========== 结果类型 ==========
class OldReaderResult(TypedDict):
    lao_shu_chong_score: float  # 0-100，越高越好，≥70合格
    toxic_points: list[str]  # 检测到的毒点类型列表
    toxic_details: list[dict]  # 每个毒点的详情
    shuangdian_points: list[str]  # 检测到的爽点类型列表
    shuangdian_details: list[dict]  # 每个爽点的详情
    issues: list[str]  # 给修改环节的问题描述
    verdict: str  # 总体评价：优秀/可读/需改/弃书


def _check_patterns(text: str, patterns: list, match_type: str = "any") -> list[tuple]:
    """
    检测文本中匹配的模式。

    Args:
        text: 待检测文本
        patterns: 模式列表，每个元素是([关键词1, 关键词2], [关联词1], [排除词])
        match_type: "any"=任一关键词组匹配即可，"all"=所有关键词组都匹配

    Returns:
        [(匹配文本, 匹配类型), ...]
    """
    results = []
    text_lower = text

    for pattern_group in patterns:
        # 解析模式
        if len(pattern_group) == _PATTERN_GROUP_LEN_2:
            keywords1, keywords2 = pattern_group
            exclude_words = []
        elif len(pattern_group) == _PATTERN_GROUP_LEN_3:
            keywords1, keywords2, exclude_words = pattern_group
        else:
            keywords1, keywords2, exclude_words = pattern_group[0], pattern_group[1], []

        # 两组关键词必须同时存在
        found1 = any(kw in text_lower for kw in keywords1 if kw)
        found2 = any(kw in text_lower for kw in keywords2 if kw)
        excluded = any(ex in text_lower for ex in exclude_words if ex)

        if excluded:
            continue

        if match_type == "all":
            if found1 and found2:
                results.append(
                    (f"{keywords1}...{keywords2}", keywords1[0] if keywords1 else "")
                )
        elif found1 and keywords1 and keywords2:
            results.append(
                (f"{keywords1}...{keywords2}", keywords1[0] if keywords1 else "")
            )
        elif found1 and not keywords2:
            results.append((keywords1[0], keywords1[0]))

    return results


def _detect_toxic_points(text: str) -> tuple[list[str], list[dict]]:
    """检测毒点"""
    detected = []
    details = []

    for toxic_type, config in TOXIC_POINTS.items():
        matches = _check_patterns(text, config["patterns"])
        if matches:
            detected.append(toxic_type)
            details.append(
                {
                    "type": toxic_type,
                    "weight": config["weight"],
                    "description": config["description"],
                    "severity": config["severity"],
                    "matches": [m[0] for m in matches[:3]],  # 最多记录3个
                }
            )

    return detected, details


def _detect_shuangdian(text: str) -> tuple[list[str], list[dict]]:
    """检测爽点"""
    detected = []
    details = []

    # 检查主模式
    for shuangdian_type, config in SHUANGDIAN_POINTS.items():
        matches = _check_patterns(text, config["patterns"])

        # 检查子模式（辅助关键词）
        sub_matches = []
        for sub in config.get("sub_patterns", []):
            if sub in text:
                sub_matches.append(sub)

        if matches or sub_matches:
            detected.append(shuangdian_type)
            details.append(
                {
                    "type": shuangdian_type,
                    "weight": config["weight"],
                    "description": config["description"],
                    "pattern_matches": [m[0] for m in matches[:3]],
                    "sub_matches": sub_matches[:3],
                }
            )

    return detected, details


def _calculate_lao_shu_chong_score(
    toxic_details: list[dict],
    shuangdian_details: list[dict],
    genre: str | None = None,
) -> float:
    """计算老书虫综合评分（v5.5: 支持题材感知权重调整）

    v5.1.1: 增加基础分下限(60分), 防止常见网文用词过度扣分。
    v5.5: 根据题材调整毒点权重和爽点加成，使评分更贴合不同题材的读者期望。
    """
    # 基础分 100
    score = 100.0

    # 解析题材
    resolved = _resolve_genre({"genre": genre}) if genre else None

    # 获取题材感知毒点权重调整
    toxic_adjust = _GENRE_TOXIC_WEIGHT_ADJUST.get(resolved, {})

    # 扣除毒点分数（应用题材权重调整）
    for toxic in toxic_details:
        weight = toxic["weight"]
        ttype = toxic["type"]
        # 应用题材权重系数
        if ttype in toxic_adjust:
            weight = weight * toxic_adjust[ttype]
        score -= weight

    # 获取题材感知爽点权重调整
    shuangdian_adjust = _GENRE_SHUANGDIAN_BONUS.get(resolved, {})

    # 爽点加分（应用题材权重调整）
    shuangdian_bonus = 0.0
    for shuangdian in shuangdian_details:
        bonus_multiplier = shuangdian_adjust.get(shuangdian["type"], 1.0)
        shuangdian_bonus += shuangdian["weight"] * 5 * bonus_multiplier

    # 爽点加分上限（题材感知）
    bonus_cap = 25.0 if resolved in ("爽文", "无敌流", "系统流") else 20.0
    shuangdian_bonus = min(bonus_cap, shuangdian_bonus)
    score += shuangdian_bonus

    # v5.1.1: 基础分下限
    severe_toxics = {"NTR", "NUE_ZHU", "MORAL_WRONG", "SHENGMU"}
    has_severe_toxic = any(t["type"] in severe_toxics for t in toxic_details)
    if not has_severe_toxic:
        # 爽文/无敌流类题材基础分下限更高（套路被接受度高）
        base_floor = 65.0 if resolved in ("爽文", "无敌流", "系统流") else 60.0
        score = max(base_floor, score)

    # v6.2 FIX (R5): 绝对下限 — 非严重毒点时不低于 30 分
    # 评分器不稳定性可能导致合法文本被扣到极低分（0.0），
    # 这会连锁触发 R2 评分校准 + R3 重写死循环。
    if not has_severe_toxic:
        score = max(30.0, score)

    return max(0.0, min(100.0, score))


def _generate_verdict(score: float, toxic_points: list[str]) -> str:
    """生成总体评价"""
    if score >= _VERDICT_EXCELLENT and not toxic_points:
        return "优秀"
    if score >= _VERDICT_READABLE:
        return "可读"
    if score >= _VERDICT_REWRITE:
        return "需大改"
    if toxic_points and any(t in ["NTR", "SHENGMU", "NUE_ZHU"] for t in toxic_points):
        return "警告：严重毒点"
    return "弃书风险"


# ── 题材感知权重调整 ──
_GENRE_TOXIC_WEIGHT_ADJUST: dict[str, dict[str, float]] = {
    # 降低某些题材中特定毒点的扣分权重
    "爽文": {
        "ANTAGONIST_STUPID": 0.3,  # 反派降智在爽文中是常见套路，权重降 70%
        "WATER_CONTENT": 0.5,  # 水文扣分减半
        "NUE_ZHU": 0.4,  # 速虐速爽模式不扣那么多
    },
    "系统流": {
        "WATER_CONTENT": 0.4,
        "ANTAGONIST_STUPID": 0.4,
    },
    "无敌流": {
        "ANTAGONIST_STUPID": 0.2,  # 无敌流反派就是用来秒的
        "WATER_CONTENT": 0.5,
        "NUE_ZHU": 0.3,
    },
    "重生": {
        "ANTAGONIST_STUPID": 0.5,
    },
    "脑洞": {
        "POWER_BREAK": 0.5,  # 脑洞文战力崩坏容忍度高
    },
    "悬疑灵异": {
        "CHARACTER_DEATH": 0.3,  # 悬疑文死人正常
    },
    "历史": {
        "MORAL_WRONG": 1.5,  # 历史文三观审查更严（权重提高 50%）
    },
    "仙侠": {
        "CHARACTER_DEATH": 0.5,
    },
    # v7.6: 都市亲情治愈 — SHENGMU(原谅)是核心情感，不是毒点
    "都市": {
        "SHENGMU": 0.1,  # 权重降至 10%（原本 30），保留感知但不严重扣分
    },
}

# 爽点加分题材调整（某些题材中特定爽点权重更高）
_GENRE_SHUANGDIAN_BONUS: dict[str, dict[str, float]] = {
    "爽文": {"打脸": 1.5, "装逼": 1.3, "逆袭": 1.4},
    "系统流": {"升级": 1.4, "装逼": 1.2},
    "无敌流": {"装逼": 1.5, "打脸": 1.4},
    "重生": {"打脸": 1.3, "逆袭": 1.3, "悬念": 1.2},
    "都市": {"打脸": 1.2, "装逼": 1.2},
    "现代言情": {"感情": 1.5, "悬念": 1.2},
}


def _resolve_genre(context: dict | None) -> str | None:
    """从 context 中解析 genre。"""
    if not context:
        return None
    genre = context.get("genre")
    if not genre:
        return None
    # 使用 constants 中的 resolve_genre 统一解析
    try:
        from novelfactory.config.constants import resolve_genre

        return resolve_genre(genre)
    except (ImportError, ValueError):
        return genre


def review_as_old_reader(
    text: str,
    context: dict | None = None,
) -> OldReaderResult:
    """
    老书虫视角评审。

    Args:
        text: 待评审的章节文本
        context: 额外上下文（大纲、题材等）

    Returns:
        OldReaderResult，包含毒点列表、爽点列表、综合评分等
    """
    if not text or len(text.strip()) < _MIN_TEXT_LENGTH:
        logger.warning(
            "[老书虫] 文本过短 (%d chars)，跳过评审", len(text) if text else 0
        )
        return OldReaderResult(
            lao_shu_chong_score=50.0,
            toxic_points=[],
            toxic_details=[],
            shuangdian_points=[],
            shuangdian_details=[],
            issues=["文本过短，无法准确评审"],
            verdict="无法评判",
        )

    # 检测毒点
    toxic_points, toxic_details = _detect_toxic_points(text)

    # 检测爽点
    shuangdian_points, shuangdian_details = _detect_shuangdian(text)

    # 计算评分（v5.5: 传入 genre 实现题材自适应）
    genre = (context or {}).get("genre", "")
    score = _calculate_lao_shu_chong_score(
        toxic_details, shuangdian_details, genre=genre
    )
    verdict = _generate_verdict(score, toxic_points)

    logger.info(
        "[老书虫] 文本=%d字 毒点=%d(%s) 爽点=%d(%s) 评分=%.1f 评价=%s",
        len(text),
        len(toxic_points),
        ",".join(toxic_points) if toxic_points else "无",
        len(shuangdian_points),
        ",".join(shuangdian_points) if shuangdian_points else "无",
        score,
        verdict,
    )

    # 生成修改建议
    issues = []
    for toxic in toxic_details:
        suggestions = {
            "NTR": "删除涉及 NTR 情节的内容，改用其他方式推进剧情",
            "SHENGMU": "主角的善良要有锋芒，不要无底线原谅敌人",
            "PROTAGONIST_STUPID": "主角的决策需要有充分理由，不能明知故犯",
            "NUE_ZHU": "适当给主角正向反馈，不要让主角长期憋屈",
            "POWER_BREAK": "严格维护战力体系，不要出现战力崩塌的情节",
            "ANTAGONIST_STUPID": "反派要有自己的逻辑和底线，不能工具化",
            "CHARACTER_DEATH": "重要角色死亡需要充分的情感铺垫",
            "WATER_CONTENT": "删除重复的水文段落，精简节奏",
            "SENTIMENTAL_TORTURE": "控制煽情段落长度，不要过度渲染悲伤",
            "MORAL_WRONG": "调整价值观表达，确保主角行为符合基本道德",
        }
        issues.append(
            f"【{toxic['description']}】{suggestions.get(toxic['type'], '请重新审视该情节')}"
        )

    # 爽点建议
    if not shuangdian_points:
        issues.append("【爽点缺失】本章缺少明显爽点，建议增加打脸/装逼/悬念等元素")

    return OldReaderResult(
        lao_shu_chong_score=round(score, 2),
        toxic_points=toxic_points,
        toxic_details=toxic_details,
        shuangdian_points=shuangdian_points,
        shuangdian_details=shuangdian_details,
        issues=issues,
        verdict=verdict,
    )


# ========== 命令行测试 ==========
if __name__ == "__main__":
    test_texts = [
        {
            "name": "有严重毒点的文本",
            "text": """
就在这时，只见那主角的女人被人夺走，他的女友背叛了他，跟了别人。
主角明明知道这是陷阱，还是愚蠢地走了进去。
反派得意洋洋地站在他面前，主角却无脑地选择相信他。
主角的亲人被侮辱，主角却无底线地选择原谅。
整章压抑憋屈，主角一直受气不敢还手，读者看得难受。
            """,
        },
        {
            "name": "有爽点的文本",
            "text": """
反派嚣张地站在那里，不可一世。主角不屑一顾，冷哼一声。
主角展露真正实力，众人惊呆了，脸色大变，倒吸一口凉气。
原来他竟然是隐藏实力的绝世高手，众人膜拜。
主角从一阶突破至九阶，境界提升，实力暴涨。
就在此时，他收到了一封信——故事远未结束。
            """,
        },
        {
            "name": "正常文本",
            "text": """
老张推开门，屋里烟雾缭绕。他皱了皱眉，把窗户推开一条缝。
\"你又抽烟。\"他说。
对面的人没吭声，手指夹着的烟已经燃到了滤嘴。
            """,
        },
    ]

    for t in test_texts:
        print(f"\n{'=' * 40}")
        print(f"测试：{t['name']}")
        result = review_as_old_reader(t["text"])
        print(f"老书虫评分: {result['lao_shu_chong_score']}")
        print(f"毒点: {result['toxic_points']}")
        print(f"爽点: {result['shuangdian_points']}")
        print(f"评价: {result['verdict']}")
        print("建议:")
        for issue in result["issues"]:
            print(f"  - {issue}")
