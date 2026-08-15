import os

for name in ("DEEPSEEK_API_KEY", "ARK_API_KEY", "SILICONFLOW_API_KEY", "OPENAI_API_KEY"):
    v = os.getenv(name, "")
    print(f"{name}: len={len(v)} head={v[:6]!r} tail={v[-4:]!r}")
