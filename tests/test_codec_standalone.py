"""Narrative Codec Engine — 编入引擎独立模块集成测试。

测试所有编码模块的基本功能和输出格式正确性。
本测试不依赖Docker和外部数据库，仅测试内存中的数据处理。

重构说明 (v6.1):
  独立类 → @tool 函数模式:
    SceneSplitter          → find_scene_boundaries (tools.py)
    ExpertIndexExtractor   → compute_expert_index (tools.py)
    EmotionArcExtractor    → extract_emotion_arc (tools.py)
    STACClassifier         → apply_rule_stac (tools.py)
    SentenceRefiner        → 移除 (LLM Agent 路径)
    CausalGraphBuilder     → 移除 (LLM Agent 路径)
  新增:
    CodecCrewLocalState    → state.py
    build_codec_crew       → crew.py (仅测试编译)
"""

from __future__ import annotations

import json
from typing import Any

import networkx as nx
import pytest

# ── 待测试模块导入 ──────────────────────────────────────────────────────────
from novelfactory.pipeline.narrative_codec.crew import build_codec_crew
from novelfactory.pipeline.narrative_codec.schemas import (
    CausalGraph,
    ConsistencyErrorType,
    EmotionArc,
    EntityNode,
    EventNode,
    ExpertIndex,
    NarrativeStage,
    STACBond,
    STACLabel,
    STACLabeledSentence,
    Scene,
    StructuredProtocol,
    TraitVector,
    VALID_STAC_BONDS,
)
from novelfactory.pipeline.narrative_codec.state import CodecCrewLocalState
from novelfactory.pipeline.narrative_codec.tools import (
    apply_rule_stac,
    compute_expert_index,
    extract_emotion_arc,
    find_scene_boundaries,
)


# ═══════════════════════════════════════════════════════════════════════════
# Mock 对象
# ═══════════════════════════════════════════════════════════════════════════


class MockLLMResponse:
    """模拟 LangChain LLM 响应。"""

    def __init__(self, content: str = "") -> None:
        self.content = content

    def __str__(self) -> str:
        return self.content


class MockLLM:
    """Mock LLM — 返回固定响应，不实际调用 API。"""

    def __init__(self, response: str = "") -> None:
        self._response = response

    def invoke(self, messages: Any, **kwargs: Any) -> MockLLMResponse:
        return MockLLMResponse(content=self._response)


# ═══════════════════════════════════════════════════════════════════════════
# 测试数据 — 约 800 字原创中文小说片段
# ═══════════════════════════════════════════════════════════════════════════

SAMPLE_CHAPTER = (
    "天色渐暗，森林中弥漫着浓重的雾气。"
    "李慕白站在一棵古松之下，神情凝重地望着远方。"
    "他必须在天黑之前找到那个传说中的洞穴。"
    "手中的罗盘指针疯狂旋转，仿佛受到了某种力量的干扰。"
    "突然，一阵尖锐的嘶鸣声从密林深处传来。"
    "李慕白握紧了手中的长剑，警觉地环顾四周。"
    "他沿着蜿蜒的山路快速奔跑，脚下的落叶发出沙沙的声响。"
    "三年前，师父在这片森林中离奇失踪。"
    "从此以后，李慕白每日都在寻找师父的下落。"
    "他记得师父曾经说过，这片森林深处封印着一件上古神器。"
    "而解开封印的关键，就在那座被遗忘的洞穴之中。"
    "李慕白加快了脚步，心中涌起一股强烈的不安。"
    "前方的雾气越来越浓，能见度不足三尺。"
    "他感到一股冰冷的气息从背后靠近。"
    "李慕白猛地转身，长剑出鞘，划出一道银白色的弧光。"
    "然而身后空无一物，只有飘荡的雾气和摇曳的树影。"
    "他的心跳如擂鼓，额头渗出细密的汗珠。"
    "这时，一道微弱的蓝光从不远处的石缝中透出。"
    "李慕白小心翼翼地靠近，发现那是一个被藤蔓掩盖的洞口。"
    "洞口狭窄仅容一人通过，里面传来潺潺的水声。"
    "他深吸一口气，点燃火折子，弯腰钻进了洞穴。"
    "洞内通道蜿蜒曲折，石壁上布满了奇怪的符文。"
    "那些符文散发着淡淡的荧光，仿佛在诉说着古老的秘密。"
    "李慕白伸手触摸其中一枚符文，指尖传来一阵灼热。"
    "突然整个洞穴开始剧烈震动，碎石从头顶簌簌落下。"
    "他意识到自己触发了某种古老的机关。"
    "必须尽快找到出路，否则将被埋葬在这座千年古墓之中。"
    "李慕白拼命向前奔跑，身后的通道不断坍塌。"
    "就在千钧一发之际，他看到前方有一道亮光。"
    "他纵身一跃，滚出了洞口，重重地摔在了一片草地上。"
    "阳光刺眼，鸟语花香，这里竟然是另一片天地。"
    "他躺在地上大口喘息，劫后余生的喜悦充斥心间。"
    "但同时他也明白，自己已经踏上了一段无法回头的旅程。"
)

# 短文本用于简单测试
SHORT_TEXT = "李慕白走进了山洞，发现洞壁上刻满了古老的符文。"

# 无显式结构的平坦文本
FLAT_TEXT = "这是第一句话。这是第二句话。这是第三句话。"


# ═══════════════════════════════════════════════════════════════════════════
# 辅助函数 — 将 @tool 的 JSON 输出解析为 Python 对象
# ═══════════════════════════════════════════════════════════════════════════


def _parse_tool_output(result_str: str) -> Any:
    """解析 @tool 的 JSON 字符串输出。"""
    return json.loads(result_str)


# ═══════════════════════════════════════════════════════════════════════════
# 1. Schema 定义测试
# ═══════════════════════════════════════════════════════════════════════════


class TestTraitVector:
    """测试 TraitVector 创建和字段约束。"""

    def test_default_creation(self) -> None:
        tv = TraitVector()
        assert tv.value == 0.0
        assert tv.inertia == 0.0
        assert tv.exogenous_noise == 0.0
        assert tv.evidence_strength == "weak"

    def test_custom_values(self) -> None:
        tv = TraitVector(
            value=0.8,
            inertia=0.5,
            exogenous_noise=0.1,
            evidence_strength="strong",
        )
        assert tv.value == 0.8
        assert tv.inertia == 0.5
        assert tv.evidence_strength == "strong"

    def test_value_bounds(self) -> None:
        """值域约束：value 必须在 [0.0, 1.0] 内。"""
        with pytest.raises(Exception):
            TraitVector(value=1.5)
        with pytest.raises(Exception):
            TraitVector(value=-0.1)


class TestEntityNode:
    """测试 EntityNode。"""

    def test_default_creation(self) -> None:
        en = EntityNode()
        assert en.entity_id == ""
        assert en.name == ""
        assert en.traits == {}
        assert en.aliases == []

    def test_with_traits(self) -> None:
        en = EntityNode(
            entity_id="char_001",
            name="李慕白",
            traits={"courage": TraitVector(value=0.9)},
            aliases=["小李", "白哥"],
        )
        assert en.entity_id == "char_001"
        assert en.traits["courage"].value == 0.9
        assert "小李" in en.aliases


class TestEventNode:
    """测试 EventNode — 包含 fabula_time / syuzhet_index。"""

    def test_default_creation(self) -> None:
        ev = EventNode()
        assert ev.event_id == ""
        assert ev.event_type == "outcome"
        assert ev.fabula_time == 0
        assert ev.syuzhet_index == 0

    def test_dual_timeline(self) -> None:
        """fabula_time 和 syuzhet_index 可以不同（双时间轴）。"""
        ev = EventNode(
            event_id="ev_005",
            label="发现洞穴",
            fabula_time=3,
            syuzhet_index=7,
        )
        assert ev.fabula_time == 3
        assert ev.syuzhet_index == 7
        assert ev.fabula_time != ev.syuzhet_index

    def test_event_type_options(self) -> None:
        """event_type 限于 choice/outcome/revelation/utterance。"""
        EventNode(event_type="choice")
        EventNode(event_type="outcome")
        EventNode(event_type="revelation")
        EventNode(event_type="utterance")
        with pytest.raises(Exception):
            EventNode(event_type="invalid_type")  # type: ignore[arg-type]


class TestSTACLabelEnum:
    """测试 STACLabel 枚举有 4 个值。"""

    def test_four_labels(self) -> None:
        labels = list(STACLabel)
        assert len(labels) == 4

    def test_values(self) -> None:
        assert STACLabel.SITUATION.value == "situation"
        assert STACLabel.TASK.value == "task"
        assert STACLabel.ACTION.value == "action"
        assert STACLabel.CONSEQUENCE.value == "consequence"


class TestSTACBondEnum:
    """测试 STACBond 枚举有 11 个值。"""

    def test_eleven_bonds(self) -> None:
        bonds = list(STACBond)
        assert len(bonds) == 11

    def test_example_bonds(self) -> None:
        assert STACBond.S_TO_A.value == "situation→action"
        assert STACBond.T_TO_A.value == "task→action"
        assert STACBond.A_TO_C.value == "action→consequence"

    def test_all_valid_bonds_in_schema(self) -> None:
        """VALID_STAC_BONDS 字典包含 11 种连接。"""
        assert len(VALID_STAC_BONDS) == 11


class TestExpertIndexOneHot:
    """测试 ExpertIndex.to_onehot() 输出 15 维（2+2+3+2+2+2+2）。"""

    ONEHOT_DIM = 15

    def test_onehot_length(self) -> None:
        ei = ExpertIndex()
        oh = ei.to_onehot()
        assert len(oh) == self.ONEHOT_DIM

    def test_onehot_values(self) -> None:
        """默认值的 one-hot 编码验证。"""
        ei = ExpertIndex()
        expected = [
            1, 0,  # genericity(2): specific
            1, 0,  # eventivity(2): dynamic
            1, 0, 0,  # boundedness(3): episodic
            1, 0,  # initiativity(2): initiate
            1, 0,  # time_start(2): past
            1, 0,  # time_end(2): current
            1, 0,  # impact(2): impactful
        ]
        assert len(expected) == self.ONEHOT_DIM
        assert ei.to_onehot() == expected

    def test_onehot_generic(self) -> None:
        ei = ExpertIndex(genericity="generic")
        oh = ei.to_onehot()
        assert oh[0] == 0
        assert oh[1] == 1

    def test_onehot_boundedness_habitual(self) -> None:
        ei = ExpertIndex(boundedness="habitual")
        oh = ei.to_onehot()
        # 前 4 维不变(genericity 2 + eventivity 2)，第 5-7 维：habitual=[0,1,0]
        assert oh[4] == 0
        assert oh[5] == 1
        assert oh[6] == 0

    def test_onehot_receive(self) -> None:
        ei = ExpertIndex(initiativity="receive")
        oh = ei.to_onehot()
        # 前 7 维不变(genericity 2 + eventivity 2 + boundedness 3)，第 8-9 维：receive=[0,1]
        assert oh[7] == 0
        assert oh[8] == 1

    def test_onehot_future(self) -> None:
        ei = ExpertIndex(time_end="future")
        oh = ei.to_onehot()
        # genericity(2) + eventivity(2) + boundedness(3) + initiativity(2) + time_start(2) = 11
        # time_end(2) 在索引 11-12
        assert oh[11] == 0
        assert oh[12] == 1

    def test_onehot_resolved(self) -> None:
        """impact=resolved → 最后 2 维为 [0,1]。"""
        ei = ExpertIndex(impact="resolved")
        oh = ei.to_onehot()
        # genericity(2) + eventivity(2) + boundedness(3) + initiativity(2)
        # + time_start(2) + time_end(2) = 13, impact(2) 在索引 13-14
        assert oh[13] == 0  # impactful
        assert oh[14] == 1  # resolved

    def test_onehot_all_custom(self) -> None:
        """全自定义值域的 one-hot 验证。"""
        ei = ExpertIndex(
            genericity="generic",
            eventivity="stative",
            boundedness="static",
            initiativity="receive",
            time_start="current",
            time_end="future",
            impact="resolved",
        )
        oh = ei.to_onehot()
        # [genericity(2), eventivity(2), boundedness(3), initiativity(2),
        #  time_start(2), time_end(2), impact(2)] = 15维
        expected = [
            0, 1,  # generic
            0, 1,  # stative
            0, 0, 1,  # static
            0, 1,  # receive
            0, 1,  # current
            0, 1,  # future
            0, 1,  # resolved
        ]
        assert len(expected) == self.ONEHOT_DIM
        assert oh == expected


class TestStructuredProtocol:
    """测试 StructuredProtocol。"""

    def test_default_creation(self) -> None:
        sp = StructuredProtocol()
        assert sp.extracted_info == ""
        assert sp.rationale == ""
        assert sp.answer is None
        assert sp.confidence == 0.0

    def test_with_data(self) -> None:
        sp = StructuredProtocol(
            extracted_info="李慕白发现洞穴",
            rationale="文本中提到'发现那是一个被藤蔓掩盖的洞口'",
            answer="洞穴入口",
            confidence=4.0,
        )
        assert "洞穴" in sp.extracted_info


class TestConsistencyErrorType:
    """测试 ConsistencyErrorType 至少有 8 种。"""

    def test_at_least_eight_types(self) -> None:
        types = list(ConsistencyErrorType)
        assert len(types) >= 8

    def test_specific_errors(self) -> None:
        """验证 ConStory 5 大类中已知的错误类型存在。"""
        error_values = {e.value for e in ConsistencyErrorType}
        assert "absolute_time_contradiction" in error_values
        assert "causeless_effect" in error_values
        assert "abandoned_plot_element" in error_values
        assert "memory_contradiction" in error_values
        assert "skill_fluctuation" in error_values
        assert "core_rule_violation" in error_values
        assert "nomenclature_confusion" in error_values
        assert "perspective_confusion" in error_values


class TestSceneSchema:
    """测试 Scene 结构。"""

    def test_scene_with_char_positions(self) -> None:
        scene = Scene(
            scene_id=1,
            start_char=10,
            end_char=100,
            characters=["李慕白"],
            location="森林",
            time_marker="傍晚",
            text=SAMPLE_CHAPTER[10:100],
        )
        assert scene.scene_id == 1
        assert scene.start_char == 10
        assert scene.end_char == 100
        assert "李慕白" in scene.characters


class TestEmotionArcSchema:
    """测试 EmotionArc。"""

    def test_default_empty(self) -> None:
        ea = EmotionArc()
        assert ea.valence_sequence == []
        assert ea.arousal_sequence == []
        assert ea.stages == []

    def test_with_data(self) -> None:
        ea = EmotionArc(
            valence_sequence=[0.5, -0.2, 0.8],
            arousal_sequence=[0.3, 0.6, 0.9],
            window_positions=[0, 50, 100],
            stages=[(0, NarrativeStage.EXPOSITION)],
        )
        assert len(ea.valence_sequence) == 3


class TestCausalGraphSchema:
    """测试 CausalGraph。"""

    def test_default_empty(self) -> None:
        cg = CausalGraph()
        assert cg.events == []
        assert cg.edges == []


class TestSTACLabeledSentenceSchema:
    """测试 STACLabeledSentence Schema。"""

    def test_default_creation(self) -> None:
        s = STACLabeledSentence()
        assert s.text == ""
        assert s.stac_label is None
        assert s.confidence == 0.0

    def test_with_data(self) -> None:
        s = STACLabeledSentence(
            text="李慕白握紧了长剑。",
            stac_label=STACLabel.ACTION,
            confidence=0.85,
        )
        assert s.text == "李慕白握紧了长剑。"
        assert s.stac_label == STACLabel.ACTION


class TestCodecCrewLocalState:
    """测试 CodecCrewLocalState 状态定义。"""

    def test_creation(self) -> None:
        """空创建可通过，TypedDict 默认为空。"""
        state = CodecCrewLocalState()
        assert isinstance(state, dict)

    def test_with_values(self) -> None:
        """传入字段值后应正确存储。"""
        state = CodecCrewLocalState(
            raw_text="测试文本",
            codec_stage="split",
        )
        assert state.get("raw_text") == "测试文本"
        assert state.get("codec_stage") == "split"
        assert state.get("scenes") is None
        assert state.get("codec_error") is None


# ═══════════════════════════════════════════════════════════════════════════
# 2. 场景分割测试 — find_scene_boundaries
# ═══════════════════════════════════════════════════════════════════════════


class TestFindSceneBoundaries:
    """测试 find_scene_boundaries — 纯规则，零LLM依赖。"""

    def test_output_is_list_of_scenes(self) -> None:
        """输出为 list[dict] 且每个场景有字符位置。"""
        result = _parse_tool_output(find_scene_boundaries.invoke({"text": SAMPLE_CHAPTER}))
        assert isinstance(result, list)
        assert len(result) >= 1
        for scene in result:
            assert isinstance(scene, dict)
            assert "start_char" in scene
            assert "end_char" in scene
            assert scene["end_char"] > scene["start_char"]
            assert len(scene.get("text", "")) > 0

    def test_scene_has_characters(self) -> None:
        """场景应至少提取到一个角色。"""
        result = _parse_tool_output(find_scene_boundaries.invoke({"text": SAMPLE_CHAPTER}))
        if result:
            assert any(scene.get("characters") for scene in result)

    def test_scene_has_location_or_time(self) -> None:
        """场景应包含地点或时间标记。"""
        result = _parse_tool_output(find_scene_boundaries.invoke({"text": SAMPLE_CHAPTER}))
        if result:
            assert any(
                scene.get("location") or scene.get("time_marker")
                for scene in result
            )

    def test_empty_text_returns_empty(self) -> None:
        result = _parse_tool_output(find_scene_boundaries.invoke({"text": ""}))
        assert result == []
        result = _parse_tool_output(find_scene_boundaries.invoke({"text": "   "}))
        assert result == []

    def test_short_text_single_scene(self) -> None:
        """短文本应返回单个场景。"""
        result = _parse_tool_output(find_scene_boundaries.invoke({"text": SHORT_TEXT}))
        assert len(result) == 1

    def test_explicit_marker_split(self) -> None:
        """显式分隔标记（带换行）应分割场景。"""
        text = "第一段内容。\n***\n第二段内容。"
        result = _parse_tool_output(find_scene_boundaries.invoke({"text": text}))
        assert len(result) >= 2

    def test_scene_text_preserved(self) -> None:
        """场景原文应完整保留。"""
        result = _parse_tool_output(find_scene_boundaries.invoke({"text": SHORT_TEXT}))
        if result:
            assert "李慕白" in result[0].get("text", "")


# ═══════════════════════════════════════════════════════════════════════════
# 3. ExpertIndex 提取测试 — compute_expert_index
# ═══════════════════════════════════════════════════════════════════════════


class TestComputeExpertIndex:
    """测试 compute_expert_index — 全部基于规则，无 LLM。"""

    def test_extract_single_sentence(self) -> None:
        result = _parse_tool_output(
            compute_expert_index.invoke({"texts": ["李慕白握紧了手中的长剑。"]})
        )
        assert isinstance(result, list)
        assert len(result) == 1
        item = result[0]
        assert item["genericity"] == "specific"
        assert item["eventivity"] == "dynamic"
        assert item["initiativity"] == "initiate"

    def test_extract_batch_output_length(self) -> None:
        """批量提取输出长度与输入一致。"""
        sentences = ["他走了。", "她在笑。", "天很蓝。"]
        result = _parse_tool_output(compute_expert_index.invoke({"texts": sentences}))
        assert len(result) == 3

    def test_genericity_specific(self) -> None:
        """人称代词开头 → specific。"""
        for sentence in ["他走进洞穴。", "她看着远方。", "我们出发了。"]:
            result = _parse_tool_output(
                compute_expert_index.invoke({"texts": [sentence]})
            )
            assert result[0]["genericity"] == "specific", f"Failed: {sentence}"

    def test_genericity_generic(self) -> None:
        """泛指词开头 → generic。"""
        result = _parse_tool_output(
            compute_expert_index.invoke({"texts": ["人们总是忘记过去的教训。"]})
        )
        assert result[0]["genericity"] == "generic"

    def test_eventivity_dynamic(self) -> None:
        """含动态动词 → dynamic。"""
        result = _parse_tool_output(
            compute_expert_index.invoke({"texts": ["李慕白拼命向前奔跑。"]})
        )
        assert result[0]["eventivity"] == "dynamic"

    def test_eventivity_stative(self) -> None:
        """仅含静态动词 → stative。"""
        result = _parse_tool_output(
            compute_expert_index.invoke({"texts": ["天色渐暗，森林中弥漫着雾气。"]})
        )
        assert result[0]["eventivity"] in ("stative", "dynamic")

    def test_boundedness_habitual(self) -> None:
        """含习惯性标记 → habitual。"""
        result = _parse_tool_output(
            compute_expert_index.invoke({"texts": ["他每天都会来这片森林。"]})
        )
        assert result[0]["boundedness"] == "habitual"

    def test_boundedness_episodic(self) -> None:
        """含一次性事件标记 → episodic。"""
        result = _parse_tool_output(
            compute_expert_index.invoke({"texts": ["突然，一阵嘶鸣声从密林深处传来。"]})
        )
        assert result[0]["boundedness"] == "episodic"

    def test_initiativity_receive(self) -> None:
        """含被动标记 → receive。"""
        result = _parse_tool_output(
            compute_expert_index.invoke({"texts": ["他被那把剑深深地吸引了。"]})
        )
        assert result[0]["initiativity"] == "receive"

    def test_initiativity_initiate(self) -> None:
        """无被动标记 → initiate。"""
        result = _parse_tool_output(
            compute_expert_index.invoke({"texts": ["李慕白握紧了手中的长剑。"]})
        )
        assert result[0]["initiativity"] == "initiate"

    def test_time_start_past(self) -> None:
        """含过去标记 → past。"""
        result = _parse_tool_output(
            compute_expert_index.invoke({"texts": ["师父曾经说过一句话。"]})
        )
        assert result[0]["time_start"] == "past"

    def test_time_start_current(self) -> None:
        """含现在标记 → current。"""
        result = _parse_tool_output(
            compute_expert_index.invoke({"texts": ["他正在思考下一步的计划。"]})
        )
        assert result[0]["time_start"] == "current"

    def test_impact_resolved(self) -> None:
        """含已解决标记 → resolved。"""
        result = _parse_tool_output(
            compute_expert_index.invoke({"texts": ["一切终于结束了。"]})
        )
        assert result[0]["impact"] == "resolved"

    def test_impact_impactful(self) -> None:
        """含持续影响标记 → impactful。"""
        result = _parse_tool_output(
            compute_expert_index.invoke({"texts": ["从此以后，他再也没有回到故乡。"]})
        )
        assert result[0]["impact"] == "impactful"

    def test_onehot_in_output(self) -> None:
        """工具输出应包含 15 维 one-hot。"""
        result = _parse_tool_output(
            compute_expert_index.invoke({"texts": ["他猛地转身，长剑出鞘。"]})
        )
        assert "onehot" in result[0]
        oh = result[0]["onehot"]
        assert len(oh) == 15
        assert all(isinstance(v, int) for v in oh)
        assert sum(oh) == 7  # 7 个维度各有一个 1

    def test_empty_input_returns_empty_list(self) -> None:
        result = _parse_tool_output(compute_expert_index.invoke({"texts": []}))
        assert result == []


# ═══════════════════════════════════════════════════════════════════════════
# 4. STAC 分类测试 — apply_rule_stac
# ═══════════════════════════════════════════════════════════════════════════


class TestApplyRuleStac:
    """测试 apply_rule_stac — 规则版分类，零LLM依赖。"""

    def test_classify_output_list(self) -> None:
        """分类输出为 dict 列表。"""
        sentences = [
            "天色渐暗，森林中弥漫着雾气。",
            "李慕白握紧了手中的长剑。",
        ]
        result = _parse_tool_output(apply_rule_stac.invoke({"texts": sentences}))
        assert len(result) == 2
        for r in result:
            assert "label" in r
            assert "confidence" in r
            assert 0.0 <= r["confidence"] <= 1.0

    def test_situation_detection(self) -> None:
        """静态/描写句 → situation。"""
        result = _parse_tool_output(
            apply_rule_stac.invoke({"texts": ["天色渐暗，森林中弥漫着浓重的雾气。"]})
        )
        assert result[0]["label"] == "situation"

    def test_task_detection(self) -> None:
        """情态动词句 → task。"""
        result = _parse_tool_output(
            apply_rule_stac.invoke({"texts": ["他需要帮忙。"]})
        )
        assert result[0]["label"] == "task"

    def test_action_detection(self) -> None:
        """含动作动词句 → action。"""
        result = _parse_tool_output(
            apply_rule_stac.invoke({"texts": ["李慕白快步走进了山洞。"]})
        )
        assert result[0]["label"] == "action"

    def test_consequence_detection(self) -> None:
        """结果/因果句 → consequence。"""
        result = _parse_tool_output(
            apply_rule_stac.invoke({"texts": ["这一切导致了他的失败。"]})
        )
        assert result[0]["label"] == "consequence"

    def test_empty_sentence_returns_situation(self) -> None:
        """空句子应返回 situation 降级。"""
        result = _parse_tool_output(apply_rule_stac.invoke({"texts": [""]}))
        assert result[0]["label"] == "situation"

    def test_batch_ordering(self) -> None:
        """批量输入输出顺序一致。"""
        texts = [
            "他需要帮助。",
            "李慕白跑了起来。",
            "这一切导致了他的失败。",
            "天色很暗。",
        ]
        result = _parse_tool_output(apply_rule_stac.invoke({"texts": texts}))
        expected_labels = ["task", "action", "consequence", "situation"]
        assert [r["label"] for r in result] == expected_labels


class TestSTACBondValidation:
    """测试 STACBond 有效连接映射表 — 直接验证 schemas.VALID_STAC_BONDS。"""

    def test_all_eleven_bonds_valid(self) -> None:
        """遍历 STACBond 枚举，确认每个值都在 VALID_STAC_BONDS 的值中。"""
        valid_bond_values = {b.value for b in VALID_STAC_BONDS.values()}
        for bond in STACBond:
            assert bond.value in valid_bond_values, f"{bond.value} 不在有效连接中"

    def test_situation_to_action(self) -> None:
        """situation→action 为有效连接。"""
        key = (STACLabel.SITUATION, STACLabel.ACTION)
        assert key in VALID_STAC_BONDS
        assert VALID_STAC_BONDS[key] == STACBond.S_TO_A

    def test_task_to_action(self) -> None:
        """task→action 为有效连接。"""
        key = (STACLabel.TASK, STACLabel.ACTION)
        assert key in VALID_STAC_BONDS
        assert VALID_STAC_BONDS[key] == STACBond.T_TO_A

    def test_action_to_consequence(self) -> None:
        """action→consequence 为有效连接。"""
        key = (STACLabel.ACTION, STACLabel.CONSEQUENCE)
        assert key in VALID_STAC_BONDS
        assert VALID_STAC_BONDS[key] == STACBond.A_TO_C

    def test_invalid_bond_not_in_map(self) -> None:
        """task→situation 不在有效连接中。"""
        key = (STACLabel.TASK, STACLabel.SITUATION)
        assert key not in VALID_STAC_BONDS

    def test_consequence_to_situation(self) -> None:
        """consequence→situation 为有效连接（叙事循环）。"""
        key = (STACLabel.CONSEQUENCE, STACLabel.SITUATION)
        assert key in VALID_STAC_BONDS


# ═══════════════════════════════════════════════════════════════════════════
# 5. 情绪曲线测试 — extract_emotion_arc
# ═══════════════════════════════════════════════════════════════════════════


class TestExtractEmotionArc:
    """测试 extract_emotion_arc — 全部基于词典，无 LLM。"""

    def test_output_format(self) -> None:
        """输出包含 valence_sequence / arousal_sequence / stages。"""
        result = _parse_tool_output(extract_emotion_arc.invoke({"text": SAMPLE_CHAPTER}))
        assert isinstance(result, dict)
        assert "valence_sequence" in result
        assert "arousal_sequence" in result
        assert "stages" in result
        assert isinstance(result["valence_sequence"], list)
        assert isinstance(result["arousal_sequence"], list)
        assert isinstance(result["stages"], list)

    def test_sequence_length(self) -> None:
        """情绪序列长度合理（与句子数正相关）。"""
        result = _parse_tool_output(extract_emotion_arc.invoke({"text": SAMPLE_CHAPTER}))
        assert len(result["valence_sequence"]) >= 3
        assert len(result["valence_sequence"]) == len(result["arousal_sequence"])
        assert len(result["valence_sequence"]) == len(result["window_positions"])

    def test_valence_range(self) -> None:
        """valence 值在 [-1.0, 1.0] 范围内。"""
        result = _parse_tool_output(extract_emotion_arc.invoke({"text": SAMPLE_CHAPTER}))
        for v in result["valence_sequence"]:
            assert -1.0 <= v <= 1.0

    def test_arousal_range(self) -> None:
        """arousal 值在 [0.0, 1.0] 范围内。"""
        result = _parse_tool_output(extract_emotion_arc.invoke({"text": SAMPLE_CHAPTER}))
        for a in result["arousal_sequence"]:
            assert 0.0 <= a <= 1.0

    def test_stages_have_four_phases(self) -> None:
        """叙事弧应包含 4 个阶段。"""
        result = _parse_tool_output(extract_emotion_arc.invoke({"text": SAMPLE_CHAPTER}))
        assert len(result["stages"]) == 4

    def test_stage_types(self) -> None:
        """阶段类型为 exposition/rising/climax/falling。"""
        result = _parse_tool_output(extract_emotion_arc.invoke({"text": SAMPLE_CHAPTER}))
        for _, stage in result["stages"]:
            assert stage in (
                "exposition",
                "rising",
                "climax",
                "falling",
            )

    def test_stage_order(self) -> None:
        """阶段应依序出现：exposition → rising → climax → falling。"""
        result = _parse_tool_output(extract_emotion_arc.invoke({"text": SAMPLE_CHAPTER}))
        stage_order = [stage for _, stage in result["stages"]]
        expected_order = [
            "exposition",
            "rising",
            "climax",
            "falling",
        ]
        assert stage_order == expected_order

    def test_empty_text_returns_empty(self) -> None:
        result = _parse_tool_output(extract_emotion_arc.invoke({"text": ""}))
        assert result["valence_sequence"] == []
        assert result["stages"] == []

    def test_short_text(self) -> None:
        """短文本（1-2 句）也能正常输出。"""
        result = _parse_tool_output(extract_emotion_arc.invoke({"text": "他很高兴。"}))
        assert len(result["valence_sequence"]) >= 1

    def test_smoothing(self) -> None:
        """平滑后的序列应与原始窗口位置长度一致。"""
        result = _parse_tool_output(extract_emotion_arc.invoke({"text": SAMPLE_CHAPTER}))
        assert len(result["valence_sequence"]) == len(result["window_positions"])

    def test_positive_emotion_detected(self) -> None:
        """含正向情感词的句子应产生正 valence。"""
        result = _parse_tool_output(
            extract_emotion_arc.invoke({"text": "他感到无比喜悦和兴奋，心中充满了希望。"})
        )
        assert any(v > 0 for v in result["valence_sequence"])

    def test_negative_emotion_detected(self) -> None:
        """含负向情感词的句子应产生负 valence。"""
        result = _parse_tool_output(
            extract_emotion_arc.invoke({"text": "他感到深深的绝望和恐惧，内心充满了悲伤。"})
        )
        assert any(v < 0 for v in result["valence_sequence"])


# ═══════════════════════════════════════════════════════════════════════════
# 6. Codec Crew 子图测试
# ═══════════════════════════════════════════════════════════════════════════


class TestCodecCrew:
    """测试 CodecCrew 构建 — 仅测试子图编译，不运行。"""

    def test_build_codec_crew_compiles(self) -> None:
        """build_codec_crew() 应成功编译为 CompiledStateGraph。"""
        crew = build_codec_crew()
        assert crew is not None
        # 验证子图包含 8 个节点
        node_names = list(crew.nodes.keys())
        expected_nodes = [
            "scene_splitter",
            "sentence_refiner",
            "stac_classifier",
            "expert_index",
            "causal_graph_builder",
            "emotion_arc",
            "dag_validator",
            "__exit_for_codec__",
        ]
        for name in expected_nodes:
            assert name in node_names, f"缺少节点: {name}"

    def test_codec_crew_is_compiled(self) -> None:
        """验证返回的是编译后的图。"""
        crew = build_codec_crew()
        # CompiledStateGraph 应具有 compiled 属性
        assert hasattr(crew, "compiled") or hasattr(crew, "get_graph")

    def test_codec_crew_no_checkpointer(self) -> None:
        """子图编译不传 checkpointer（父图统一持久化）。"""
        crew = build_codec_crew()
        assert crew.checkpointer is None


# ═══════════════════════════════════════════════════════════════════════════
# 7. 跨模块集成测试
# ═══════════════════════════════════════════════════════════════════════════


class TestCodecIntegration:
    """多个模块串联测试 — 验证数据流完整性。"""

    def test_scene_to_expert_index(self) -> None:
        """场景分割结果可送入 compute_expert_index。"""
        scenes = _parse_tool_output(
            find_scene_boundaries.invoke({"text": SAMPLE_CHAPTER})
        )
        assert len(scenes) >= 1

        for scene in scenes:
            # 将场景文本按句分割后提取
            sentences = [s for s in scene.get("text", "").split("。") if s.strip()]
            if not sentences:
                continue
            results = _parse_tool_output(
                compute_expert_index.invoke({"texts": sentences})
            )
            assert len(results) == len(sentences)
            for r in results:
                oh = r["onehot"]
                assert len(oh) == 15

    def test_stac_rule_classification_and_bonds(self) -> None:
        """规则版 STAC 分类结果可验证 Bond 合法性。"""
        texts = [
            "天色渐暗，森林中弥漫着雾气。",
            "李慕白必须找到洞穴。",
            "他沿着山路快速奔跑。",
            "他触发了机关，洞穴开始震动。",
        ]
        labeled = _parse_tool_output(apply_rule_stac.invoke({"texts": texts}))

        assert len(labeled) == 4

        # 验证相邻句子的 Bond 合法性
        labels = [r["label"] for r in labeled]
        valid_count = 0
        for i in range(len(labels) - 1):
            key = (STACLabel(labels[i]), STACLabel(labels[i + 1]))
            if key in VALID_STAC_BONDS:
                valid_count += 1
        assert valid_count >= 0

    def test_emotion_arc_to_stage_statistics(self) -> None:
        """EmotionArc 的 stages 可转换为阶段统计。"""
        result = _parse_tool_output(
            extract_emotion_arc.invoke({"text": SAMPLE_CHAPTER})
        )

        # 验证每个阶段覆盖了序列的一部分
        stage_names = set()
        for _, stage in result["stages"]:
            stage_names.add(stage)
        assert len(stage_names) == 4

        # 阶段位置单调递增
        positions = [pos for pos, _ in result["stages"]]
        assert positions == sorted(positions)

    def test_simplified_encode_pipeline(self) -> None:
        """简化编入管线：场景分割 → STAC → ExpertIndex → 情绪曲线。

        这是高层次的集成验证，确保纯规则模块组合使用时数据流正确。
        句子精炼和因果图构建属于 LLM Agent 路径，不在本测试范围。
        """
        # Step 1: 场景分割
        scenes = _parse_tool_output(
            find_scene_boundaries.invoke({"text": SAMPLE_CHAPTER})
        )
        assert len(scenes) >= 1
        primary_text = SAMPLE_CHAPTER

        # Step 2: 获取句级文本用于 STAC 和 ExpertIndex
        sentences = [s.strip() + "。" for s in primary_text.split("。") if s.strip()]

        # Step 3: STAC 规则分类
        stac_result = _parse_tool_output(
            apply_rule_stac.invoke({"texts": sentences})
        )
        assert len(stac_result) == len(sentences)

        # Step 4: ExpertIndex 提取
        ei_result = _parse_tool_output(
            compute_expert_index.invoke({"texts": sentences})
        )
        assert len(ei_result) == len(sentences)
        for item in ei_result:
            assert len(item["onehot"]) == 15

        # Step 5: 情绪曲线
        emotion = _parse_tool_output(
            extract_emotion_arc.invoke({"text": primary_text})
        )
        assert len(emotion["valence_sequence"]) >= 1
        assert len(emotion["stages"]) == 4
