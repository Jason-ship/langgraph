"""Codec Engine 工具函数 — 零LLM模块的 @tool 包装。

所有工具遵循项目标准模式：
@tool + _get_instance 懒加载 + try/except + logger.error + json.dumps 返回

本模块将场景分割 / Expert Index / 情绪曲线的规则逻辑完整内联到 @tool 函数体内，
不依赖外部类实例，每个 tool 自包含全部关键词库和检测逻辑。

使用方式：
    tools = get_codec_tools()
    agent = create_react_agent(llm, tools=tools, prompt=...)
"""

from __future__ import annotations

import json
import logging
import re
import threading
from typing import Any

from langchain_core.tools import tool

logger = logging.getLogger(__name__)

# ── 模块级单例（懒加载占位 — 本模块工具无需外部存储）─────────────────────
_SENTINEL = threading.Lock()


def _get_codec_instance() -> object:
    """懒加载占位单例（本模块工具无需外部存储连接）。

    遵循项目 _get_store/_get_milvus 命名惯例，保持接口一致。
    返回一个标记对象表示模块已就绪。
    """
    global _SENTINEL  # noqa: PLW0602 — 用于触发模块级初始化
    _SENTINEL.acquire()
    _SENTINEL.release()
    return object()


# ══════════════════════════════════════════════════════════════════════════
# 模块级关键词库 — 完整内联自现有独立类
# ══════════════════════════════════════════════════════════════════════════

# ── Scene Splitter 常量 ────────────────────────────────────────────────

TIME_KEYWORDS: list[str] = [
    "第二天",
    "第三日",
    "次日",
    "翌日",
    "隔天",
    "第二天一早",
    "第二天清晨",
    "第二天早上",
    "第二天上午",
    "第二天中午",
    "第二天下午",
    "第二天傍晚",
    "第二天晚上",
    "第二天夜里",
    "三日后",
    "五日后",
    "七日后",
    "数日后",
    "几日后",
    "十日后",
    "半月后",
    "一个月后",
    "一年后",
    "三年后",
    "三天后",
    "五天后",
    "七天后",
    "数天后",
    "几天后",
    "与此同时",
    "就在这时",
    "正在这时",
    "就在此时",
    "片刻后",
    "片刻之后",
    "过了一会儿",
    "不多时",
    "不多久",
    "不久",
    "不久之后",
    "不多时后",
    "黄昏",
    "傍晚",
    "深夜",
    "黎明",
    "清晨",
    "拂晓",
    "破晓",
    "日落时分",
    "夕阳西下",
    "夜幕降临",
    "天色渐暗",
    "天亮了",
    "天黑了",
    "天蒙蒙亮",
    "这天",
    "这天晚上",
    "这天下午",
    "这天上午",
    "那天",
    "那天晚上",
    "那天下午",
    "那天上午",
    "一日",
    "这一日",
    "那一日",
    "晚上",
    "夜间",
    "半夜",
    "午夜",
    "子夜",
    "上午",
    "中午",
    "下午",
    "早晨",
    "早上",
    "从此",
    "从此以后",
    "自此",
    "打那以后",
    "转眼间",
    "转瞬",
    "一转眼",
    "一瞬间",
    "春去秋来",
    "寒来暑往",
    "光阴似箭",
    "时光飞逝",
    "多年后",
    "多年以后",
    "许多年后",
    "小时候",
    "那时候",
    "那时",
    "当年",
    "突然",
    "忽然",
    "猛然间",
    "骤然间",
    "这时",
    "此时",
    "此刻",
    "当下",
]

LOCATION_KEYWORDS: list[str] = [
    "来到",
    "走进",
    "走入",
    "踏入",
    "进入",
    "跑进",
    "冲进",
    "离开",
    "走出",
    "离去",
    "告别",
    "辞别",
    "回到",
    "返回",
    "归来",
    "回来",
    "前往",
    "赶往",
    "赶赴",
    "奔赴",
    "飞往",
    "驶向",
    "穿过",
    "越过",
    "跨过",
    "翻过",
    "到达",
    "抵达",
    "赶到",
    "出现在",
    "现身于",
    "现身",
    "在",
    "位于",
    "地处",
    "远处",
    "近处",
    "前方",
    "背后",
    "楼上",
    "楼下",
    "屋内",
    "屋外",
    "门外",
    "城中",
    "城外",
    "村口",
    "街头",
    "街角",
    "山脚下",
    "山顶上",
    "河岸边",
    "海边",
    "大厅",
    "房间",
    "院子",
    "花园",
    "书房",
    "客栈",
    "酒楼",
    "茶馆",
    "庙宇",
    "宫殿",
    "森林",
    "山谷",
    "河流",
    "湖泊",
    "大海",
    "密室",
    "地牢",
    "塔顶",
    "阁楼",
]

SCENE_BREAK_MARKERS: list[str] = [
    "***",
    "——",
    "……",
    "◆",
    "◇",
    "□",
    "★",
    "☆",
    "○",
    "●",
    "△",
    "▲",
]

# 常用中文姓氏表（百家姓前 120+，覆盖 90%+ 常见姓）
_COMMON_SURNAMES: set[str] = {
    "赵",
    "钱",
    "孙",
    "李",
    "周",
    "吴",
    "郑",
    "王",
    "冯",
    "陈",
    "褚",
    "卫",
    "蒋",
    "沈",
    "韩",
    "杨",
    "朱",
    "秦",
    "尤",
    "许",
    "何",
    "吕",
    "施",
    "张",
    "孔",
    "曹",
    "严",
    "华",
    "金",
    "魏",
    "陶",
    "姜",
    "戚",
    "谢",
    "邹",
    "喻",
    "柏",
    "水",
    "窦",
    "章",
    "云",
    "苏",
    "潘",
    "葛",
    "奚",
    "范",
    "彭",
    "郎",
    "鲁",
    "韦",
    "昌",
    "马",
    "苗",
    "凤",
    "花",
    "方",
    "俞",
    "任",
    "袁",
    "柳",
    "酆",
    "鲍",
    "史",
    "唐",
    "费",
    "廉",
    "岑",
    "薛",
    "雷",
    "贺",
    "倪",
    "汤",
    "滕",
    "殷",
    "罗",
    "毕",
    "郝",
    "邬",
    "安",
    "常",
    "乐",
    "于",
    "时",
    "傅",
    "皮",
    "卞",
    "齐",
    "康",
    "伍",
    "余",
    "元",
    "卜",
    "顾",
    "孟",
    "平",
    "黄",
    "和",
    "穆",
    "萧",
    "尹",
    "姚",
    "邵",
    "湛",
    "汪",
    "祁",
    "毛",
    "禹",
    "狄",
    "米",
    "贝",
    "明",
    "臧",
    "计",
    "伏",
    "成",
    "戴",
    "谈",
    "宋",
    "茅",
    "庞",
    "熊",
    "纪",
    "舒",
    "屈",
    "项",
    "祝",
    "董",
    "梁",
    "杜",
    "阮",
    "蓝",
    "闵",
    "席",
    "季",
    "麻",
    "强",
    "贾",
    "路",
    "娄",
    "危",
    "江",
    "童",
    "颜",
    "郭",
    "梅",
    "盛",
    "林",
    "刁",
    "钟",
    "徐",
    "邱",
    "骆",
    "高",
    "夏",
    "蔡",
    "田",
    "樊",
    "胡",
    "凌",
    "霍",
    "虞",
    "万",
    "支",
    "柯",
    "昝",
    "管",
    "卢",
    "莫",
    "经",
    "房",
    "裘",
    "缪",
    "干",
    "解",
    "应",
    "宗",
    "丁",
    "宣",
    "贲",
    "邓",
    "郁",
    "单",
    "杭",
    "洪",
    "包",
    "诸",
    "左",
    "石",
    "崔",
    "吉",
    "钮",
    "龚",
    "程",
    "嵇",
    "邢",
    "滑",
    "裴",
    "陆",
    "荣",
    "翁",
    "荀",
    "羊",
    "於",
    "惠",
    "甄",
    "曲",
    "家",
    "封",
    "芮",
    "羿",
    "储",
    "靳",
    "汲",
    "邴",
    "糜",
    "松",
    "井",
    "段",
    "富",
    "巫",
    "乌",
    "焦",
    "巴",
    "弓",
    "牧",
    "隗",
    "山",
    "谷",
    "车",
    "侯",
    "宓",
    "蓬",
    "全",
    "郗",
    "班",
    "仰",
    "秋",
    "仲",
    "伊",
    "宫",
    "宁",
    "仇",
    "栾",
    "暴",
    "甘",
    "钭",
    "厉",
    "戎",
    "祖",
    "武",
    "符",
    "刘",
    "景",
    "詹",
    "束",
    "龙",
    "叶",
    "幸",
    "司",
    "韶",
    "郜",
    "黎",
    "蓟",
    "薄",
    "印",
    "宿",
    "白",
    "怀",
    "蒲",
    "邰",
    "从",
    "鄂",
    "索",
    "咸",
    "籍",
    "赖",
    "卓",
    "蔺",
    "屠",
    "蒙",
    "池",
    "乔",
    "阴",
    "鬱",
    "胥",
    "能",
    "苍",
    "双",
    "闻",
    "莘",
    "党",
    "翟",
    "谭",
    "贡",
    "劳",
    "逄",
    "姬",
    "申",
    "扶",
    "堵",
    "冉",
    "宰",
    "郦",
    "雍",
    "郤",
    "璩",
    "桑",
    "桂",
    "濮",
    "牛",
    "寿",
    "通",
    "边",
    "扈",
    "燕",
    "冀",
    "郏",
    "浦",
    "尚",
    "农",
    "温",
    "别",
    "庄",
    "晏",
    "柴",
    "瞿",
    "阎",
    "充",
    "慕",
    "连",
    "茹",
    "习",
    "宦",
    "艾",
    "鱼",
    "容",
    "向",
    "古",
    "易",
    "慎",
    "戈",
    "廖",
    "庾",
    "终",
    "暨",
    "居",
    "衡",
    "步",
    "都",
    "耿",
    "满",
    "弘",
    "匡",
    "国",
    "文",
    "寇",
    "广",
    "禄",
    "阙",
    "东",
    "欧",
    "殳",
    "沃",
    "利",
    "蔚",
    "越",
    "夔",
    "隆",
    "师",
    "巩",
    "厍",
    "聂",
    "晁",
    "勾",
    "敖",
    "融",
    "冷",
    "訾",
    "辛",
    "阚",
    "那",
    "简",
    "饶",
    "空",
    "曾",
    "毋",
    "沙",
    "乜",
    "养",
    "鞠",
    "须",
    "丰",
    "巢",
    "关",
    "蒯",
    "相",
    "查",
    "后",
    "荆",
    "红",
    "游",
    "竺",
    "权",
    "逯",
    "盖",
    "益",
    "桓",
    "公",
    "万俟",
    "司马",
    "上官",
    "欧阳",
    "夏侯",
    "诸葛",
    "闻人",
    "东方",
    "赫连",
    "皇甫",
    "尉迟",
    "公羊",
    "澹台",
    "公冶",
    "宗政",
    "濮阳",
    "淳于",
    "单于",
    "太叔",
    "申屠",
    "公孙",
    "仲孙",
    "轩辕",
    "令狐",
    "钟离",
    "宇文",
    "长孙",
    "慕容",
    "鲜于",
    "闾丘",
    "司徒",
    "司空",
    "亓官",
    "司寇",
    "仉",
    "督",
    "子车",
    "颛孙",
    "端木",
    "巫马",
    "公西",
    "漆雕",
    "乐正",
    "壤驷",
    "公良",
    "拓跋",
    "夹谷",
    "宰父",
    "谷梁",
    "晋",
    "楚",
    "闫",
    "法",
    "汝",
    "鄢",
    "涂",
    "钦",
    "段干",
    "百里",
    "东郭",
    "南门",
    "呼延",
    "归",
    "海",
    "羊舌",
    "微生",
    "岳",
    "帅",
    "缑",
    "亢",
    "况",
    "后",
    "有",
    "琴",
    "梁丘",
    "左丘",
    "东门",
    "西门",
    "商",
    "牟",
    "佘",
    "佴",
    "伯",
    "赏",
    "南宫",
    "墨",
    "哈",
    "谯",
    "笪",
    "年",
    "爱",
    "阳",
    "佟",
    "第五",
    "言",
    "福",
}

_DOUBLE_SURNAMES: set[str] = {s for s in _COMMON_SURNAMES if len(s) >= 2}
_SINGLE_SURNAMES: set[str] = {s for s in _COMMON_SURNAMES if len(s) == 1}

# ── Expert Index 常量 ──────────────────────────────────────────────────

STATIC_VERB_FALSE_POSITIVES: set[str] = {
    "总是",
    "但是",
    "可是",
    "还是",
    "就是",
    "也是",
    "更是",
    "而是",
    "便是",
    "算是",
    "只有",
    "所有",
    "没有",
    "还有",
    "含有",
    "拥有",
    "具有",
    "未有",
    "只有",
}

SPECIFIC_SUBJECT_MARKERS: set[str] = {
    "我",
    "你",
    "他",
    "她",
    "它",
    "我们",
    "你们",
    "他们",
    "她们",
    "它们",
    "咱",
    "咱们",
    "您",
    "某人",
    "谁",
    "大家",
    "诸位",
}

GENERIC_SUBJECT_MARKERS: set[str] = {
    "人们",
    "人类",
    "世人",
    "众人",
    "所有人",
    "任何人",
    "每个人",
    "大家伙",
    "凡是",
    "任何",
    "所有",
    "每个",
    "人人",
    "大伙",
    "大家",
    "群众",
    "大众",
}

DYNAMIC_VERBS: set[str] = {
    # 位移/运动
    "走",
    "跑",
    "跳",
    "飞",
    "游",
    "爬",
    "奔",
    "冲",
    "追",
    "赶",
    "逃",
    "溜",
    "逛",
    "遛",
    "移",
    "挪",
    "滚",
    "翻",
    "跨",
    "踩",
    "踏",
    "登",
    "跃",
    "蹦",
    # 手部动作
    "拿",
    "放",
    "拉",
    "推",
    "抱",
    "举",
    "提",
    "扛",
    "抬",
    "搬",
    "扔",
    "丢",
    "抛",
    "接",
    "递",
    "抓",
    "握",
    "捏",
    "掐",
    "拧",
    "扯",
    "撕",
    "掰",
    "敲",
    "打",
    "拍",
    "摸",
    "碰",
    "撞",
    "扶",
    "撑",
    "托",
    "端",
    "捧",
    "拾",
    "捡",
    "掏",
    "插",
    "拔",
    "扣",
    "按",
    "压",
    "抽",
    "甩",
    "挥",
    "摇",
    "摆",
    "抖",
    "擦",
    "抹",
    "扫",
    "洗",
    "刷",
    "拖",
    "戳",
    "捅",
    "砸",
    "劈",
    "砍",
    "切",
    "割",
    "削",
    "刺",
    "扎",
    # 口部/面部
    "说",
    "讲",
    "问",
    "答",
    "叫",
    "喊",
    "吼",
    "嚷",
    "唱",
    "读",
    "念",
    "骂",
    "夸",
    "叹",
    "哭",
    "笑",
    "吃",
    "喝",
    "咬",
    "嚼",
    "咽",
    "吞",
    "吐",
    "舔",
    "吸",
    "吹",
    "吻",
    "尝",
    # 眼部/感知
    "看",
    "望",
    "盯",
    "瞪",
    "瞄",
    "瞥",
    "瞅",
    "瞧",
    "观",
    "察",
    "见",
    "听",
    "闻",
    "嗅",
    "觉",
    "感",
    # 腿部/身体
    "坐",
    "站",
    "躺",
    "蹲",
    "跪",
    "趴",
    "靠",
    "弯",
    "伸",
    "缩",
    "转",
    "躲",
    "藏",
    "避",
    "踢",
    "踹",
    "迈",
    "退",
    "进",
    "出",
    # 操作/动作
    "做",
    "干",
    "搞",
    "弄",
    "写",
    "画",
    "记",
    "抄",
    "贴",
    "挂",
    "盖",
    "包",
    "装",
    "拆",
    "开",
    "关",
    "锁",
    "系",
    "解",
    "穿",
    "戴",
    "脱",
    "换",
    "背",
    "披",
    "套",
    "试",
    "用",
    "使",
    "操作",
    "处理",
    "整理",
    "收拾",
    "打扫",
    "布置",
    "准备",
    "制作",
    "建造",
    "修建",
    "种植",
    "养殖",
    "烹饪",
    "煮",
    "炒",
    "烧",
    "烤",
    "蒸",
    "炖",
    "煎",
    "炸",
    # 社交/交互
    "告诉",
    "通知",
    "汇报",
    "报告",
    "解释",
    "介绍",
    "邀请",
    "请求",
    "命令",
    "指示",
    "吩咐",
    "叮嘱",
    "嘱咐",
    "安慰",
    "鼓励",
    "表扬",
    "批评",
    "责备",
    "指责",
    "嘲笑",
    "讽刺",
    "挖苦",
    "拥抱",
    "握手",
    "敬礼",
    "鞠躬",
    "跪拜",
    "磕头",
    # 心理活动（外显动作版）
    "思考",
    "思索",
    "回忆",
    "回想",
    "想象",
    "幻想",
    "琢磨",
    "盘算",
    "算计",
    "计划",
    "打算",
    "决定",
    "选择",
    "挑选",
    "寻找",
    "搜索",
    "探查",
    "检查",
    "检验",
    "验证",
    "测试",
    "完成",
    "开始",
    "结束",
    "继续",
    "进行",
    "使用",
    "利用",
    "采取",
    "实施",
    "执行",
    "实现",
    "达到",
    "获得",
    "失去",
    "放弃",
    "坚持",
    "努力",
    "尝试",
    "避免",
    "防止",
    "确保",
    "保证",
    "允许",
    "禁止",
    "同意",
    "拒绝",
    "承认",
    "否认",
    "证明",
    "表明",
    "显示",
    "反映",
    "体现",
    # 战斗/冲突
    "打斗",
    "搏斗",
    "战斗",
    "厮杀",
    "攻击",
    "进攻",
    "防守",
    "防御",
    "躲避",
    "闪避",
    "反击",
    "还击",
    "刺杀",
    "暗杀",
    "谋杀",
    "杀害",
    "杀死",
    "消灭",
    "摧毁",
    "破坏",
    "保护",
    "守卫",
    "救援",
    "拯救",
    "救助",
    "释放",
    "发射",
    "投掷",
    "射击",
    "挥砍",
    "刺击",
    # 其他常见动词
    "买",
    "卖",
    "租",
    "借",
    "还",
    "给",
    "送",
    "取",
    "交",
    "收",
    "付",
    "花",
    "省",
    "赚",
    "赔",
    "欠",
    "带",
    "领",
    "陪",
    "跟",
    "随",
    "找",
    "寻",
    "求",
    "要",
    "讨",
    "抢",
    "夺",
    "偷",
    "盗",
    "骗",
    "欺",
    "瞒",
    "哄",
    "劝",
    "阻",
    "拦",
    "挡",
    "堵",
    "塞",
    "填",
    "补",
    "修",
    "改",
    "换",
    "变",
    "化",
    "生",
    "死",
    "活",
    "存",
    "亡",
    "灭",
    "燃",
    "烧",
    "炸",
    "裂",
    "碎",
    "破",
    "断",
    "折",
    "塌",
    "倒",
    "翻",
    "倾",
    "覆",
    "涌",
    "冒",
    "喷",
    "射",
    "流",
    "滴",
    "淌",
    "渗",
    "浸",
    "泡",
    "淹",
    "浮",
    "沉",
    "飘",
    "扬",
    "散",
    "聚",
    "集",
    "合",
    "分",
    "离",
    "隔",
    "连",
    "通",
    "串",
    "绕",
    "缠",
    "绑",
    "捆",
    "封",
    "闭",
    "拧",
}

STATIVE_VERBS: set[str] = {
    "是",
    "有",
    "存在",
    "在于",
    "显得",
    "好像",
    "仿佛",
    "似乎",
    "如同",
    "好比",
    "犹如",
    "宛若",
    "恰似",
    "属于",
    "位于",
    "包含",
    "包括",
    "具备",
    "拥有",
    "具有",
    "缺乏",
    "缺少",
    "充满",
    "遍布",
    "弥漫",
    "笼罩",
    "覆盖",
    "布满",
    "保持",
    "维持",
    "处于",
    "面临",
    "面对",
    "立于",
    "横亘",
    "坐落",
    "矗立",
    "伫立",
    "屹立",
    "挺立",
    "构成",
    "组成",
    "形成",
    "等于",
    "意味",
    "代表",
    "标志",
    "象征",
    "表示",
    "表明",
    "说明",
    "反映",
    "体现",
    "对应",
    "匹配",
    "符合",
    "适合",
    "适应",
    "喜欢",
    "喜爱",
    "热爱",
    "讨厌",
    "厌恶",
    "憎恨",
    "害怕",
    "恐惧",
    "担心",
    "担忧",
    "期望",
    "期待",
    "渴望",
    "向往",
    "希望",
    "想要",
    "需要",
    "愿意",
    "情愿",
    "能够",
    "可以",
    "值得",
    "源自",
    "来自",
    "来源",
    "出自",
    "产自",
    "生长于",
    "生活在",
    "居住在",
}

EPISODIC_MARKERS: set[str] = {
    "突然",
    "忽然",
    "猛地",
    "猛然",
    "骤然",
    "倏然",
    "猝然",
    "陡然",
    "顿然",
    "忽地",
    "蓦地",
    "蓦然",
    "冷不丁",
    "冷不防",
    "这时",
    "那时",
    "那天",
    "此刻",
    "此时",
    "当时",
    "就在",
    "正在这时",
    "正在此时",
    "就在这时",
    "就在此时",
    "说时迟那时快",
    "一刹那",
    "一瞬间",
    "一转眼",
    "一转眼间",
    "转瞬间",
    "转眼间",
    "顷刻间",
    "霎时间",
    "眨眼间",
    "弹指间",
    "须臾间",
    "片刻间",
    "很快",
    "飞快地",
    "迅速",
    "瞬间",
    "即刻",
    "立即",
    "立刻",
    "随即",
    "旋即",
    "便",
    "就",
}

HABITUAL_MARKERS: set[str] = {
    "总是",
    "经常",
    "常常",
    "时常",
    "往往",
    "通常",
    "平日",
    "平时",
    "平常",
    "素日",
    "向来",
    "一贯",
    "一直",
    "始终",
    "每天",
    "每日",
    "每夜",
    "每回",
    "每次",
    "每个",
    "天天",
    "年年",
    "月月",
    "周周",
    "夜夜",
    "日复一日",
    "年复一年",
    "偶尔",
    "间或",
    "有时",
    "有时候",
    "时不时",
    "不时",
    "反复",
    "一再",
    "再三",
    "屡次",
    "屡屡",
    "频频",
    "接连",
    "连续",
    "持续",
    "不断",
    "不停",
    "经常性",
    "习惯性",
    "周期性",
    "定期",
    "照例",
    "按例",
    "如常",
    "照常",
    "照旧",
    "依旧",
    "依然",
}

PASSIVE_MARKERS: set[str] = {
    "被",
    "让",
    "给",
    "叫",
    "受",
    "遭",
    "挨",
    "蒙",
    "受到",
    "遭到",
    "遭受",
    "蒙受",
    "为",
    "被...所",
    "为...所",
    "被...给",
}

PAST_MARKERS: set[str] = {
    "已经",
    "早已",
    "早就",
    "曾经",
    "曾",
    "已",
    "业已",
    "都已",
    "均已",
    "已然",
    "了",
    "过",
    "早已",
    "早已经",
    "已曾",
    "事先",
    "预先",
    "提前",
    "此前",
    "之前",
    "以前",
    "原先",
    "原来",
    "本来",
    "方才",
    "刚才",
    "刚刚",
    "过去",
    "往日",
    "昔日",
    "从前",
    "当年",
    "当时",
}

CURRENT_START_MARKERS: set[str] = {
    "正在",
    "现在",
    "今日",
    "此刻",
    "此时",
    "当下",
    "目前",
    "如今",
    "现今",
    "而今",
    "时下",
    "眼下",
    "现如今",
    "这会儿",
    "这时候",
    "今天",
    "今",
    "本",
    "将",
    "会",
    "要",
    "即将",
    "将要",
    "快要",
    "就快要",
    "打算",
    "准备",
    "计划",
    "想要",
}

CURRENT_END_MARKERS: set[str] = {
    "着",
    "在",
    "正在",
    "目前",
    "至今",
    "仍然",
    "依旧",
    "依然",
    "仍旧",
    "还",
    "还在",
    "不停",
    "不断",
    "一直",
    "始终",
    "持续",
    "持续着",
    "维持着",
    "保持着",
}

FUTURE_END_MARKERS: set[str] = {
    "将",
    "会",
    "要",
    "即将",
    "将要",
    "快要",
    "就快要",
    "马上",
    "立刻",
    "立即",
    "很快",
    "迟早",
    "总有一天",
    "有朝一日",
    "打算",
    "准备",
    "计划",
    "想要",
    "希望",
    "期望",
    "期待",
    "决心",
    "决定",
    "预定",
    "拟",
    "拟将",
    "行将",
    "即",
}

IMPACTFUL_MARKERS: set[str] = {
    "导致",
    "使得",
    "致使",
    "引起",
    "引发",
    "触发",
    "诱发",
    "带来",
    "造成",
    "酿成",
    "促成",
    "催生",
    "激起",
    "激发",
    "从此",
    "从此以后",
    "自此",
    "自那以后",
    "从那以后",
    "打那以后",
    "此后",
    "此后不久",
    "再也",
    "永远",
    "永",
    "永久",
    "永世",
    "永生",
    "永无止境",
    "再也不能",
    "再也不会",
    "从此不再",
    "从此再没",
    "再没有",
    "再也没有",
    "无法挽回",
    "无可挽回",
    "不可逆转",
    "覆水难收",
    "一辈子",
    "一生",
    "余生",
    "终生",
    "终身",
}

RESOLVED_MARKERS: set[str] = {
    "结束了",
    "完成了",
    "已解决",
    "已结束",
    "做完了",
    "搞定了",
    "办妥了",
    "完结了",
    "落幕了",
    "闭幕了",
    "收场了",
    "过去了",
    "消散了",
    "消失了",
    "不见了",
    "终结了",
    "终止了",
    "停止了",
    "完结",
    "告终",
    "了结",
    "了断",
    "平息",
    "平复",
    "烟消云散",
    "化为乌有",
    "尘埃落定",
    "一笔勾销",
    "到此为止",
}

# ── Emotion Arc 常量 ───────────────────────────────────────────────────

NEGATION_WORDS: frozenset[str] = frozenset(
    {"不", "没", "别", "无", "未", "莫", "勿", "休"}
)

POSITIVE_EMOTIONS: dict[str, float] = {
    # 基础正向（32词）
    "高兴": 0.8,
    "开心": 0.8,
    "快乐": 0.9,
    "喜悦": 0.8,
    "兴奋": 0.9,
    "激动": 0.7,
    "感动": 0.6,
    "欣慰": 0.5,
    "满意": 0.6,
    "满足": 0.5,
    "幸福": 0.9,
    "甜蜜": 0.7,
    "喜欢": 0.6,
    "爱": 0.7,
    "疼爱": 0.6,
    "关爱": 0.5,
    "希望": 0.4,
    "期待": 0.3,
    "惊喜": 0.8,
    "轻松": 0.4,
    "放心": 0.3,
    "骄傲": 0.5,
    "自豪": 0.6,
    "感激": 0.6,
    "佩服": 0.4,
    "同情": 0.2,
    "羡慕": 0.4,
    "平静": 0.1,
    "温暖": 0.5,
    "舒畅": 0.5,
    "痛快": 0.6,
    "爽": 0.5,
    # 扩展正向（34词）
    "欢快": 0.7,
    "愉悦": 0.7,
    "欣喜": 0.7,
    "称心": 0.5,
    "得意": 0.5,
    "振奋": 0.8,
    "鼓舞": 0.7,
    "欢喜": 0.7,
    "悠然": 0.3,
    "自在": 0.4,
    "惬意": 0.6,
    "舒适": 0.4,
    "安宁": 0.3,
    "祥和": 0.4,
    "融洽": 0.5,
    "和睦": 0.5,
    "亲切": 0.5,
    "友好": 0.5,
    "热情": 0.6,
    "灿烂": 0.6,
    "浪漫": 0.6,
    "甜美": 0.6,
    "爽快": 0.5,
    "明朗": 0.5,
    "开怀": 0.7,
    "畅快": 0.6,
    "雀跃": 0.8,
    "陶醉": 0.7,
    "温柔": 0.5,
    "宽容": 0.4,
    "舒心": 0.5,
    "庆幸": 0.5,
    "知足": 0.4,
    "动人": 0.5,
}

NEGATIVE_EMOTIONS: dict[str, float] = {
    # 基础负向（36词）
    "悲伤": -0.7,
    "伤心": -0.7,
    "难过": -0.6,
    "痛苦": -0.8,
    "愤怒": -0.8,
    "生气": -0.6,
    "恼火": -0.7,
    "恨": -0.9,
    "恐惧": -0.8,
    "害怕": -0.7,
    "担忧": -0.4,
    "焦虑": -0.5,
    "紧张": -0.4,
    "不安": -0.5,
    "失望": -0.6,
    "绝望": -0.9,
    "沮丧": -0.6,
    "郁闷": -0.5,
    "寂寞": -0.4,
    "孤独": -0.5,
    "羞愧": -0.4,
    "尴尬": -0.3,
    "后悔": -0.5,
    "遗憾": -0.4,
    "厌恶": -0.7,
    "恶心": -0.6,
    "嫉妒": -0.5,
    "仇恨": -0.9,
    "烦恼": -0.4,
    "无聊": -0.3,
    "疲惫": -0.3,
    "失落": -0.5,
    "惊慌": -0.6,
    "惊恐": -0.8,
    "悲哀": -0.7,
    "忧愁": -0.5,
    # 扩展负向（34词）
    "悲痛": -0.8,
    "哀伤": -0.7,
    "忧伤": -0.6,
    "凄惨": -0.8,
    "凄凉": -0.6,
    "悲凉": -0.6,
    "辛酸": -0.5,
    "痛心": -0.7,
    "痛恨": -0.8,
    "憎恨": -0.8,
    "憎恶": -0.7,
    "怨恨": -0.7,
    "恼怒": -0.7,
    "暴躁": -0.6,
    "恐慌": -0.7,
    "畏惧": -0.6,
    "胆怯": -0.4,
    "愧疚": -0.5,
    "内疚": -0.5,
    "懊悔": -0.6,
    "懊恼": -0.5,
    "焦躁": -0.5,
    "烦闷": -0.4,
    "压抑": -0.5,
    "抑郁": -0.7,
    "消沉": -0.5,
    "空虚": -0.4,
    "茫然": -0.4,
    "迷惘": -0.4,
    "困惑": -0.3,
    "纠结": -0.3,
    "煎熬": -0.6,
    "辛劳": -0.3,
    "悲痛欲绝": -0.9,
}

HIGH_AROUSAL: dict[str, float] = {
    "愤怒": 0.9,
    "兴奋": 0.9,
    "激动": 0.9,
    "惊恐": 0.9,
    "狂喜": 0.9,
    "暴怒": 0.9,
    "恐惧": 0.8,
    "紧张": 0.8,
    "惊喜": 0.8,
    "震撼": 0.8,
    "惊慌": 0.8,
    "热烈": 0.7,
    "激烈": 0.7,
    "绝望": 0.7,
    "悲痛": 0.7,
    "震惊": 0.9,
}

LOW_AROUSAL: dict[str, float] = {
    "平静": 0.1,
    "放松": 0.2,
    "轻松": 0.2,
    "安宁": 0.1,
    "无聊": 0.2,
    "疲惫": 0.2,
    "困倦": 0.1,
    "宁静": 0.1,
    "温和": 0.2,
    "冷淡": 0.2,
    "淡漠": 0.1,
    "懒散": 0.2,
}

_SENTENCE_PATTERN = re.compile(r"[^。！？\n]+[。！？\n]")


# ══════════════════════════════════════════════════════════════════════════
# 内部辅助函数 — Scene Splitter
# ══════════════════════════════════════════════════════════════════════════


def _split_sentences(text: str) -> list[str]:
    """将文本分割为句子列表（合并过短句子）。"""
    parts = re.split(r"(?<=[。！？!?\n\r])", text)
    sentences: list[str] = [p.strip() for p in parts if p.strip()]
    merged: list[str] = []
    buf = ""
    for s in sentences:
        if len(buf) + len(s) < 15 and buf:
            buf += s
        else:
            if buf:
                merged.append(buf)
            buf = s
    if buf:
        merged.append(buf)
    return merged if merged else sentences


def _extract_entities(text_block: str) -> tuple[list[str], list[str]]:
    """使用规则回退提取人物/地点实体。"""
    persons: set[str] = set()
    locations: set[str] = set()

    # ── 人物: 姓 + 1~2 字名 ──
    for m in re.finditer(
        r"[\u4e00-\u9fff]{2,4}(?=[\u3000-\u303f\s，。！？、；：）\)」』”’\n\r\t\u4e00-\u9fff])",
        text_block,
    ):
        name = m.group()
        if 2 <= len(name) <= 4:
            matched = False
            for ds in _DOUBLE_SURNAMES:
                if name.startswith(ds):
                    given = name[len(ds) :]
                    if 1 <= len(given) <= 2:
                        persons.add(name)
                        matched = True
                    break
            if not matched and name[0] in _SINGLE_SURNAMES:
                given = name[1:]
                if 1 <= len(given) <= 3:
                    persons.add(name)

    # ── 地点: 关键词触发 + 后缀检测 ──
    loc_suffixes = (
        r"(?:市|城|镇|村|县|区|山|河|湖|海|江|岛|殿|厅|室|房|"
        r"楼|院|园|宫|庙|塔|阁|洞|谷|峰|岭|崖|滩|湾|港|关|门|"
        r"口|街|路|巷|大道|广场|公园|花园|市场|客栈|酒楼|茶馆|"
        r"宫殿|森林|山谷|河流|湖泊|大海|密室|地牢|塔顶|阁楼|"
        r"大殿|正厅|后院|前厅|书房|卧室|客厅|厨房|院子|花园|"
        r"走廊|门口|窗外|屋内|屋外|楼上|楼下|城中|城外|村口|"
        r"街头|街角|山脚|山顶|河边|海岸|岸边|半山腰|山脚下|"
        r"河岸边|山顶上)"
    )
    loc_prefixes = [
        r"(?:在|来到|回到|走进|踏入|进入|前往|赶往|抵达|到达|离开|走出|穿过|越过)\s*",
        r"(?:位于|地处|出现在|现身于)\s*",
    ]
    for prefix in loc_prefixes:
        pattern = prefix + r"([\u4e00-\u9fff]{2,8}" + loc_suffixes + r")"
        for m in re.finditer(pattern, text_block):
            locations.add(m.group(1))

    # 后缀直接匹配
    for m in re.finditer(r"([\u4e00-\u9fff]{2,8})" + loc_suffixes, text_block):
        locations.add(m.group())

    return sorted(persons), sorted(locations)


def _extract_scene_entities(text_block: str) -> tuple[list[str], str, str]:
    """提取场景的角色/地点/时间标记。"""
    chars, locs = _extract_entities(text_block)
    location = locs[0] if locs else ""
    time_marker = ""
    earliest_pos = len(text_block)
    for kw in TIME_KEYWORDS:
        pos = text_block.find(kw)
        if pos != -1 and pos < earliest_pos:
            earliest_pos = pos
            time_marker = kw
    return chars, location, time_marker


def _detect_entity_density(sentences: list[str], window_size: int = 3) -> list[float]:
    """计算每个句子的实体密度（滑动窗口内的唯一实体数 / 窗口大小）。"""
    n = len(sentences)
    if n == 0:
        return []
    sent_entities: list[tuple[set[str], set[str]]] = []
    for sent in sentences:
        persons, locs = _extract_entities(sent)
        sent_entities.append((set(persons), set(locs)))
    densities: list[float] = []
    half = window_size // 2
    for i in range(n):
        start = max(0, i - half)
        end = min(n, i + half + 1)
        window_persons: set[str] = set()
        window_locs: set[str] = set()
        for j in range(start, end):
            p, loc_set = sent_entities[j]
            window_persons.update(p)
            window_locs.update(loc_set)
        unique_count = len(window_persons) + len(window_locs)
        densities.append(unique_count / (end - start))
    return densities


def _find_density_shifts(
    densities: list[float], shift_threshold: float = 1.8
) -> list[int]:
    """找实体密度突变点（基于差分 + z-score 阈值）。"""
    if len(densities) < 3:
        return []
    diffs: list[float] = [
        densities[i + 1] - densities[i] for i in range(len(densities) - 1)
    ]
    if not diffs:
        return []
    mean = sum(diffs) / len(diffs)
    variance = sum((d - mean) ** 2 for d in diffs) / len(diffs)
    std = variance**0.5
    if std < 1e-6:
        return []
    threshold = shift_threshold * std
    shifts: list[int] = []
    for i, d in enumerate(diffs):
        if abs(d) > threshold:
            shifts.append(i)
    return shifts


def _match_keywords(
    sentences: list[str],
) -> list[dict[str, Any]]:
    """匹配每句的时间/地点关键词。"""
    results: list[dict[str, Any]] = []
    for i, sent in enumerate(sentences):
        time_kw: list[str] = []
        loc_kw: list[str] = []
        for marker in SCENE_BREAK_MARKERS:
            if marker in sent:
                time_kw.append(marker)
        for kw in TIME_KEYWORDS:
            if kw in sent:
                time_kw.append(kw)
        for kw in LOCATION_KEYWORDS:
            if kw in sent:
                loc_kw.append(kw)
        if time_kw or loc_kw:
            results.append(
                {
                    "sentence_index": i,
                    "time_kw": time_kw,
                    "loc_kw": loc_kw,
                }
            )
    return results


def _check_coref_continuity(
    sentences: list[str],
    boundary: int,
    window: int = 3,
) -> bool:
    """检查边界前后文本的实体连续性。

    若前后窗口内的实体重叠率 > 50%，则认为连续（返回 True），该弱边界应被丢弃。
    """
    start_before = max(0, boundary - window)
    end_before = boundary + 1
    start_after = boundary + 1
    end_after = min(len(sentences), boundary + 1 + window)

    text_before = "".join(sentences[start_before:end_before])
    text_after = "".join(sentences[start_after:end_after])

    if not text_before or not text_after:
        return False

    persons_before, locs_before = _extract_entities(text_before)
    persons_after, locs_after = _extract_entities(text_after)

    set_before = set(persons_before) | set(locs_before)
    set_after = set(persons_after) | set(locs_after)

    if not set_before or not set_after:
        return False

    overlap = len(set_before & set_after)
    min_count = min(len(set_before), len(set_after))
    if min_count == 0:
        return False
    return (overlap / min_count) > 0.5


def _merge_boundaries(
    shifts: list[int],
    keyword_matches: list[dict[str, Any]],
    sentences: list[str],
    window_size: int = 3,
) -> list[int]:
    """融合突变点 + 关键词 → 最终边界。"""
    num_sentences = len(sentences)
    if num_sentences < 2:
        return []

    kw_sentence_set: set[int] = {m["sentence_index"] for m in keyword_matches}
    kw_boundaries: set[int] = set()
    for sidx in kw_sentence_set:
        if sidx < num_sentences - 1:
            kw_boundaries.add(sidx)

    strong: set[int] = set()
    weak_shifts: set[int] = set()
    weak_kw: set[int] = set()

    for s in shifts:
        if any(abs(s - k) <= 1 for k in kw_sentence_set):
            strong.add(s)
        else:
            weak_shifts.add(s)

    for k in kw_boundaries:
        if k not in strong and not any(abs(k - s) <= 1 for s in shifts):
            weak_kw.add(k)

    weak_boundaries: set[int] = set()
    win = max(1, window_size)

    for b in weak_shifts:
        if not _check_coref_continuity(sentences, b, win):
            weak_boundaries.add(b)

    for b in weak_kw:
        if not _check_coref_continuity(sentences, b, win):
            weak_boundaries.add(b)

    all_boundaries: set[int] = strong | weak_boundaries
    sorted_b = sorted(all_boundaries)
    final: list[int] = []
    for b in sorted_b:
        if b < 0 or b >= num_sentences - 1:
            continue
        if not final or b - final[-1] >= window_size:
            final.append(b)
    return final


def _cut_by_boundaries(
    sentences: list[str],
    boundaries: list[int],
    original_text: str,
) -> list[str]:
    """按边界索引将句子列表切分为场景文本块。"""
    if not boundaries:
        return [original_text]
    blocks: list[str] = []
    prev = 0
    for b in boundaries:
        block = "".join(sentences[prev:b])
        if block.strip():
            blocks.append(block.strip())
        prev = b
    rest = "".join(sentences[prev:])
    if rest.strip():
        blocks.append(rest.strip())
    return blocks if blocks else [original_text]


def _finalize_scenes(text_blocks: list[str]) -> list[dict[str, Any]]:
    """将文本块转为场景字典列表。"""
    scenes: list[dict[str, Any]] = []
    char_offset = 0
    for idx, block in enumerate(text_blocks):
        if not block.strip():
            continue
        chars, loc, time_ = _extract_scene_entities(block)
        scenes.append(
            {
                "scene_id": idx,
                "start_char": char_offset,
                "end_char": char_offset + len(block),
                "characters": chars,
                "location": loc,
                "time_marker": time_,
                "text": block,
            }
        )
        char_offset += len(block)
    return scenes


def _pre_split_by_markers(text: str) -> list[str]:
    """按显式场景分隔标记预分割文本。"""
    escaped = [re.escape(m) for m in SCENE_BREAK_MARKERS]
    pattern = (
        r"(?:^|\n)\s*(?:" + "|".join(escaped) + r")\s*(?:\n|$)"
        r"|(?:\n\s*[-–—]{3,}\s*\n)"
        r"|(?:\n\s*\*{3,}\s*\n)"
        r"|(?:\n\s*·{3,}\s*\n)"
    )
    parts = re.split(pattern, text.strip())
    return [p.strip() for p in parts if p.strip()]


# ══════════════════════════════════════════════════════════════════════════
# 内部辅助函数 — Expert Index
# ══════════════════════════════════════════════════════════════════════════


def _is_stative_false_positive(text: str, stative_verb: str) -> bool:
    """判断静态动词匹配是否为误判（属于常见非动词组合词的一部分）。"""
    if len(stative_verb) == 1:
        for fp in STATIC_VERB_FALSE_POSITIVES:
            if stative_verb in fp and fp in text:
                return True
    return False


def _has_true_stative(text: str) -> bool:
    """检查文本中是否包含真正的静态动词（排除常见误判）。"""
    for sv in STATIVE_VERBS:
        if sv in text:
            if not _is_stative_false_positive(text, sv):
                return True
    return False


def _has_passive_marker(text: str) -> bool:
    """检查是否有被动标记。"""
    for marker in ("受到", "遭到", "遭受", "蒙受"):
        if marker in text:
            return True
    if "为" in text and "所" in text:
        idx_wei = text.index("为")
        idx_suo = text.index("所")
        if idx_wei < idx_suo and idx_suo - idx_wei <= 10:
            return True
    if "被" in text and "给" in text:
        idx_bei = text.index("被")
        idx_gei = text.index("给")
        if idx_bei < idx_gei and idx_gei - idx_bei <= 10:
            return True
    for marker in ("被", "让", "给", "叫", "遭", "挨"):
        if marker in text:
            return True
    return False


def _detect_genericity(text: str) -> str:
    """检测通指性：specific / generic。"""
    stripped = text.strip()
    if not stripped:
        return "specific"

    # 取第一个"词"（1-2个字符）
    first_two = stripped[:2]
    if first_two in SPECIFIC_SUBJECT_MARKERS:
        return "specific"
    if first_two in GENERIC_SUBJECT_MARKERS:
        return "generic"
    first_char = stripped[0]
    if first_char in SPECIFIC_SUBJECT_MARKERS:
        return "specific"
    if first_char in GENERIC_SUBJECT_MARKERS:
        return "generic"

    if any(stripped.startswith(m) for m in ("凡是", "任何", "所有", "每个")):
        return "generic"
    if any(
        stripped.startswith(m) for m in ("人们", "人类", "世人", "大家", "人人", "众人")
    ):
        return "generic"

    return "specific"


def _detect_eventivity(text: str) -> str:
    """检测事件性：dynamic / stative。"""
    text_stripped = text.strip()
    if not text_stripped:
        return "dynamic"
    has_dynamic = any(dv in text for dv in DYNAMIC_VERBS)
    has_stative = _has_true_stative(text)
    if has_dynamic:
        return "dynamic"
    if has_stative:
        return "stative"
    return "dynamic"


def _detect_boundedness(text: str) -> str:
    """检测有界性：episodic / habitual / static。"""
    for marker in (
        "总是",
        "经常",
        "常常",
        "通常",
        "每天",
        "每日",
        "每次",
        "偶尔",
        "有时",
    ):
        if marker in text:
            return "habitual"
    for marker in HABITUAL_MARKERS:
        if marker in text:
            return "habitual"
    if _has_true_stative(text):
        return "static"
    for marker in EPISODIC_MARKERS:
        if marker in text:
            return "episodic"
    return "episodic"


def _detect_initiativity(text: str) -> str:
    """检测主动性：initiate / receive。"""
    if _has_passive_marker(text):
        return "receive"
    return "initiate"


def _detect_time_start(text: str) -> str:
    """检测起始时间：past / current。"""
    for marker in PAST_MARKERS:
        if marker in text:
            return "past"
    for marker in CURRENT_START_MARKERS:
        if marker in text:
            return "current"
    return "past"


def _detect_time_end(text: str) -> str:
    """检测结束时间：current / future。"""
    for marker in ("将", "会", "要", "即将", "将会", "将要"):
        if marker in text:
            return "future"
    for marker in FUTURE_END_MARKERS:
        if marker in text:
            return "future"
    for marker in CURRENT_END_MARKERS:
        if marker in text:
            return "current"
    if "了" in text or "过" in text:
        return "current"
    return "current"


def _detect_impact(text: str) -> str:
    """检测持续影响：impactful / resolved。"""
    for marker in RESOLVED_MARKERS:
        if marker in text:
            return "resolved"
    for marker in IMPACTFUL_MARKERS:
        if marker in text:
            return "impactful"
    if "结束" in text and ("了" in text or "过" in text):
        return "resolved"
    return "impactful"


def _expert_index_to_onehot(
    genericity: str,
    eventivity: str,
    boundedness: str,
    initiativity: str,
    time_start: str,
    time_end: str,
    impact: str,
) -> list[int]:
    """转换为15维 one-hot 编码向量。"""
    onehot: list[int] = []

    # genericity (2维)
    onehot.extend([1, 0] if genericity == "specific" else [0, 1])
    # eventivity (2维)
    onehot.extend([1, 0] if eventivity == "dynamic" else [0, 1])
    # boundedness (3维)
    if boundedness == "episodic":
        onehot.extend([1, 0, 0])
    elif boundedness == "habitual":
        onehot.extend([0, 1, 0])
    else:
        onehot.extend([0, 0, 1])
    # initiativity (2维)
    onehot.extend([1, 0] if initiativity == "initiate" else [0, 1])
    # time_start (2维)
    onehot.extend([1, 0] if time_start == "past" else [0, 1])
    # time_end (2维)
    onehot.extend([1, 0] if time_end == "current" else [0, 1])
    # impact (2维)
    onehot.extend([1, 0] if impact == "impactful" else [0, 1])

    return onehot


# ══════════════════════════════════════════════════════════════════════════
# 内部辅助函数 — Emotion Arc
# ══════════════════════════════════════════════════════════════════════════


def _split_sentences_with_offset(text: str) -> list[tuple[str, int]]:
    """将文本分割为句子，返回 (句子文本, 起始偏移量) 列表。"""
    sentences: list[tuple[str, int]] = []
    for m in _SENTENCE_PATTERN.finditer(text):
        raw = m.group().strip()
        if raw:
            sentences.append((raw, m.start()))
    if sentences:
        last_end = sentences[-1][1] + len(sentences[-1][0])
    else:
        last_end = 0
    remainder = text[last_end:].strip()
    if remainder:
        sentences.append((remainder, last_end))
    return sentences


def _score_sentence(sentence: str) -> tuple[float, float]:
    """计算单句的情感极性(valence)和唤起度(arousal)。"""
    # ── Valence ──
    valence_contribs: list[float] = []

    for word, val in POSITIVE_EMOTIONS.items():
        if word in sentence:
            if _has_negation_in_sentence(sentence, word):
                valence_contribs.append(-val * 0.7)
            else:
                valence_contribs.append(val)

    for word, val in NEGATIVE_EMOTIONS.items():
        if word in sentence:
            if _has_negation_in_sentence(sentence, word):
                valence_contribs.append(-val * 0.7)
            else:
                valence_contribs.append(val)

    if valence_contribs:
        valence = sum(valence_contribs) / len(valence_contribs)
        valence = max(-1.0, min(1.0, valence))
    else:
        valence = 0.0

    # ── Arousal ──
    high_vals: list[float] = []
    low_vals: list[float] = []

    for word, val in HIGH_AROUSAL.items():
        if word in sentence:
            high_vals.append(val)
    for word, val in LOW_AROUSAL.items():
        if word in sentence:
            low_vals.append(val)

    if high_vals and low_vals:
        arousal = (
            sum(high_vals) / len(high_vals) + (1.0 - sum(low_vals) / len(low_vals))
        ) / 2.0
    elif high_vals:
        arousal = sum(high_vals) / len(high_vals)
    elif low_vals:
        arousal = 1.0 - sum(low_vals) / len(low_vals)
    else:
        arousal = 0.5

    arousal = max(0.0, min(1.0, arousal))
    return valence, arousal


def _has_negation_in_sentence(sentence: str, word: str) -> bool:
    """检查情感词前是否有否定词（前3字符范围内）。"""
    idx = sentence.find(word)
    if idx < 0:
        return False
    start = max(0, idx - 3)
    prefix = sentence[start:idx]
    for neg in NEGATION_WORDS:
        if neg in prefix:
            return True
    return False


def _smooth_gaussian(values: list[float], sigma: float = 1.5) -> list[float]:
    """高斯滤波平滑序列（可选 numpy/scipy）。"""
    if len(values) < 3:
        return list(values)
    try:
        import numpy as np
        from scipy.ndimage import gaussian_filter1d

        arr = np.array(values, dtype=np.float64)
        smoothed = gaussian_filter1d(arr, sigma=sigma)
        return smoothed.tolist()
    except ImportError:
        # 无 scipy 时用简单移动平均回退
        window = max(1, int(sigma * 2))
        result: list[float] = []
        half = window // 2
        for i in range(len(values)):
            left = max(0, i - half)
            right = min(len(values), i + half + 1)
            result.append(sum(values[left:right]) / (right - left))
        return result


def _find_local_minimum(values: list[float], center: int, total_length: int) -> int:
    """在指定中心点附近寻找局部极小值。"""
    if total_length < 10:
        return min(max(0, center), total_length - 1)
    window = max(2, int(total_length * 0.10))
    left = max(0, center - window)
    right = min(total_length - 1, center + window)
    if right - left < 2:
        return min(max(0, center), total_length - 1)
    segment = values[left : right + 1]
    min_idx = segment.index(min(segment))
    return left + min_idx


def _detect_narrative_stages(
    valence: list[float],
) -> list[tuple[int, str]]:
    """检测叙事弧阶段（Freytag金字塔四阶段）。"""
    n = len(valence)
    if n == 0:
        return [(0, "exposition")]
    if n == 1:
        return [(0, "exposition")]

    # 找到全局最高点
    climax_idx = valence.index(max(valence))

    # 确保高潮不在序列两端
    if climax_idx < int(n * 0.15):
        climax_idx = int(n * 0.60)
    elif climax_idx > int(n * 0.90):
        climax_idx = int(n * 0.75)

    # 计算默认边界（Freytag金字塔）
    boundaries_raw = [0, int(n * 0.25), int(n * 0.60), int(n * 0.80), n]

    # 根据实际高潮位置调整边界
    expected_climax_center = int(n * 0.65)
    displacement = climax_idx - expected_climax_center

    if abs(displacement) > int(n * 0.08):
        boundaries_raw[2] = max(
            boundaries_raw[1] + 2,
            min(n - 4, climax_idx - int(n * 0.05)),
        )
        boundaries_raw[3] = min(n, boundaries_raw[2] + int(n * 0.20))

    # 在边界附近寻找局部极小值进行微调
    adjusted_boundaries = [0]
    for i in range(1, len(boundaries_raw) - 1):
        candidate = _find_local_minimum(valence, boundaries_raw[i], n)
        adjusted_boundaries.append(candidate)
    adjusted_boundaries.append(n)

    # 确保单调递增
    for i in range(1, len(adjusted_boundaries)):
        if adjusted_boundaries[i] <= adjusted_boundaries[i - 1]:
            adjusted_boundaries[i] = min(n - 1, adjusted_boundaries[i - 1] + 1)

    stages: list[tuple[int, str]] = [
        (adjusted_boundaries[0], "exposition"),
        (adjusted_boundaries[1], "rising"),
        (adjusted_boundaries[2], "climax"),
        (adjusted_boundaries[3], "falling"),
    ]
    return stages


# ══════════════════════════════════════════════════════════════════════════
# @tool 定义
# ══════════════════════════════════════════════════════════════════════════


@tool
def find_scene_boundaries(text: str) -> str:
    """检测文本中的场景边界，返回场景分割结果列表。

    基于实体密度突变+时间/地点关键词的规则分割，零LLM依赖。
    全部逻辑内联自 SceneSplitter 类。

    Args:
        text: 原始小说文本

    Returns:
        JSON字符串: [{scene_id, start_char, end_char, characters, location, time_marker, text}]
    """
    if not text or not text.strip():
        return json.dumps([], ensure_ascii=False)

    try:
        # 1. 用分隔标记预分割（显式分隔优先）
        pre_split_scenes = _pre_split_by_markers(text)
        if len(pre_split_scenes) > 1:
            scenes = _finalize_scenes(pre_split_scenes)
            return json.dumps(scenes, ensure_ascii=False, default=str)

        # 2. 无显式标记 — 执行完整算法
        sentences = _split_sentences(text)

        if len(sentences) < 2:
            chars, loc, time_ = _extract_scene_entities(text)
            scene = {
                "scene_id": 0,
                "start_char": 0,
                "end_char": len(text),
                "characters": chars,
                "location": loc,
                "time_marker": time_,
                "text": text,
            }
            return json.dumps([scene], ensure_ascii=False, default=str)

        # 3. 检测密度突变 + 关键词
        densities = _detect_entity_density(sentences)
        shifts = _find_density_shifts(densities)
        keyword_matches = _match_keywords(sentences)

        # 4. 融合 → 最终边界
        boundaries = _merge_boundaries(shifts, keyword_matches, sentences)

        # 5. 若仍无边界 → 单个场景
        if not boundaries:
            chars, loc, time_ = _extract_scene_entities(text)
            scene = {
                "scene_id": 0,
                "start_char": 0,
                "end_char": len(text),
                "characters": chars,
                "location": loc,
                "time_marker": time_,
                "text": text,
            }
            return json.dumps([scene], ensure_ascii=False, default=str)

        # 6. 按边界切割为场景块 → finalize
        scene_texts = _cut_by_boundaries(sentences, boundaries, text)
        scenes = _finalize_scenes(scene_texts)
        return json.dumps(scenes, ensure_ascii=False, default=str)

    except Exception as e:
        logger.error("[codec_tools] find_scene_boundaries error: %s", e)
        return json.dumps({"error": str(e)}, ensure_ascii=False)


@tool
def compute_expert_index(texts: list[str]) -> str:
    """计算句子列表的7维语言学特征，返回15维one-hot编码。

    全部规则内联自 ExpertIndexExtractor 类，零LLM依赖。
    参考: Beyond LLMs ACL 2025 §3.3 — Expert Index

    Args:
        texts: 句子列表

    Returns:
        JSON字符串: [{genericity, eventivity, boundedness, initiativity,
                      time_start, time_end, impact, onehot}]
    """
    if not texts:
        return json.dumps([], ensure_ascii=False)

    try:
        result: list[dict[str, Any]] = []
        for t in texts:
            if not t.strip():
                continue

            # 7维检测
            genericity = _detect_genericity(t)
            eventivity = _detect_eventivity(t)
            boundedness = _detect_boundedness(t)
            initiativity = _detect_initiativity(t)
            time_start = _detect_time_start(t)
            time_end = _detect_time_end(t)
            impact = _detect_impact(t)

            # 15维 one-hot
            onehot = _expert_index_to_onehot(
                genericity,
                eventivity,
                boundedness,
                initiativity,
                time_start,
                time_end,
                impact,
            )

            result.append(
                {
                    "genericity": genericity,
                    "eventivity": eventivity,
                    "boundedness": boundedness,
                    "initiativity": initiativity,
                    "time_start": time_start,
                    "time_end": time_end,
                    "impact": impact,
                    "onehot": onehot,
                }
            )

        return json.dumps(result, ensure_ascii=False, default=str)

    except Exception as e:
        logger.error("[codec_tools] compute_expert_index error: %s", e)
        return json.dumps({"error": str(e)}, ensure_ascii=False)


@tool
def extract_emotion_arc(
    text: str, window_size: int = 7, gaussian_sigma: float = 1.5
) -> str:
    """从文本中提取情绪曲线和叙事弧阶段。

    使用情感词典+滑动窗口分析，零LLM依赖。
    全部逻辑内联自 EmotionArcExtractor 类。

    算法:
    1. 将文本按句分割
    2. 对每句计算情感极性(valence, [-1.0, 1.0])和唤起度(arousal, [0.0, 1.0])
    3. 高斯滤波平滑
    4. 基于曲线形态划分叙事弧阶段(铺垫→推进→峰值→回落)

    Args:
        text: 输入文本（中文小说/章节文本）
        window_size: 滑动窗口大小(句子数)，默认7（当前实现中句粒度为1，通过高斯滤波实现等效平滑）
        gaussian_sigma: 高斯滤波平滑标准差，默认1.5，控制平滑程度

    Returns:
        JSON字符串: {valence_sequence, arousal_sequence, window_positions, stages}
    """
    # 参数验证
    window_size = max(1, window_size)
    gaussian_sigma = max(0.1, gaussian_sigma)

    if not text or not text.strip():
        return json.dumps(
            {
                "valence_sequence": [],
                "arousal_sequence": [],
                "window_positions": [],
                "stages": [],
            },
            ensure_ascii=False,
        )

    try:
        # 1. 按句分割
        sentences = _split_sentences_with_offset(text)
        if not sentences:
            return json.dumps(
                {
                    "valence_sequence": [],
                    "arousal_sequence": [],
                    "window_positions": [],
                    "stages": [],
                },
                ensure_ascii=False,
            )

        # 2. 逐句评分
        valence_list: list[float] = []
        arousal_list: list[float] = []
        positions: list[int] = []

        for sent_text, offset in sentences:
            sent_text = sent_text.strip()
            if not sent_text:
                continue
            v, a = _score_sentence(sent_text)
            valence_list.append(v)
            arousal_list.append(a)
            positions.append(offset)

        if not valence_list:
            return json.dumps(
                {
                    "valence_sequence": [],
                    "arousal_sequence": [],
                    "window_positions": [],
                    "stages": [],
                },
                ensure_ascii=False,
            )

        # 3. 高斯平滑
        valence_arr = _smooth_gaussian(valence_list, gaussian_sigma)
        arousal_arr = _smooth_gaussian(arousal_list, gaussian_sigma)

        # 4. 检测叙事弧阶段
        stages = _detect_narrative_stages(valence_arr)

        result = {
            "valence_sequence": valence_arr,
            "arousal_sequence": arousal_arr,
            "window_positions": positions,
            "stages": [(pos, stage) for pos, stage in stages],
        }
        return json.dumps(result, ensure_ascii=False, default=str)

    except Exception as e:
        logger.error("[codec_tools] extract_emotion_arc error: %s", e)
        return json.dumps({"error": str(e)}, ensure_ascii=False)


# ── 工具集导出 ──────────────────────────────────────────────────────────────


def get_codec_tools() -> list:
    """获取所有Codec工具函数列表，可直接传入 create_react_agent。"""
    return [
        find_scene_boundaries,
        compute_expert_index,
        extract_emotion_arc,
    ]
    return {"label": "situation", "confidence": 0.3}


# ── Tool: STAC规则分类 ─────────────────────────────────────────────────


def _rule_stac_classify(text: str) -> dict[str, Any]:
    """单句STAC规则分类 — 基于动词关键词匹配。

    简单规则:
    - 含"是/有/在/充满/显得/似乎"等静态动词 → Situation
    - 含"需要/必须/应该/想要/打算"等情态动词 → Task
    - 含"走/跑/说/做/打/说/问/引/攻"等动态动词 → Action
    - 含"导致/引起/使得/造成/变成/成为"等结果动词 → Consequence
    - 默认 → Situation
    """
    situation_verbs = {
        "是",
        "有",
        "在",
        "充满",
        "显得",
        "似乎",
        "位于",
        "处于",
        "乃",
        "即",
    }
    task_verbs = {
        "需要",
        "必须",
        "应该",
        "想要",
        "打算",
        "计划",
        "试图",
        "争取",
        "宜",
        "当",
        "须",
    }
    action_verbs = {
        "走",
        "跑",
        "说",
        "做",
        "打",
        "看",
        "拿",
        "放",
        "给",
        "来",
        "去",
        "问",
        "引",
        "攻",
        "守",
        "杀",
        "战",
        "追",
        "逃",
        "立",
        "坐",
        "起",
        "入",
        "出",
        "至",
        "从",
        "率",
        "领",
        "召",
        "聚",
        "使",
        "令",
        "告",
    }
    consequence_verbs = {
        "导致",
        "引起",
        "使得",
        "造成",
        "变成",
        "成为",
        "改变",
        "迫使",
        "促使",
        "引发",
        "遂",
        "乃",
        "于是",
        "因此",
    }

    for v in consequence_verbs:
        if v in text:
            return {"text": text[:100], "label": "consequence", "confidence": 0.7}
    for v in task_verbs:
        if v in text:
            return {"text": text[:100], "label": "task", "confidence": 0.7}
    for v in action_verbs:
        if v in text:
            return {"text": text[:100], "label": "action", "confidence": 0.7}
    for v in situation_verbs:
        if v in text:
            return {"text": text[:100], "label": "situation", "confidence": 0.6}

    return {"text": text[:100], "label": "situation", "confidence": 0.3}


@tool
def apply_rule_stac(texts: list[str]) -> str:
    """对句子列表应用规则版STAC分类。

    基于简单动词关键词匹配的叙事功能四分类。
    不调用LLM。

    Args:
        texts: 句子列表

    Returns:
        JSON字符串, [{text, label, confidence}]
    """
    try:
        results = [_rule_stac_classify(t) for t in texts]
        return json.dumps(results, ensure_ascii=False)
    except Exception as e:
        logger.error("[apply_rule_stac] error: %s", e)
        return json.dumps(
            [{"text": t[:100], "label": "situation", "confidence": 0.3} for t in texts],
            ensure_ascii=False,
        )
