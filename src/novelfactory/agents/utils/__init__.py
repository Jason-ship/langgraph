"""Agent utility helpers.

v5.8 之前: 包含 structured.py（bind_structured 异常保护 + invoke_structured_or_freetext）。
v5.8 之后: structured.py 已移除，结构化输出改为 JSON 解析 + validate_json_output，
          LLM 调用改为 async_llm_call_with_retry + 文本提取。
          本目录保留为包结构占位，当前无活跃模块。
"""
