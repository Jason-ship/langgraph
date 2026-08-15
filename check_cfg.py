from novelfactory.config.settings import settings

print("settings.DEEPSEEK_API_KEY:", len(settings.DEEPSEEK_API_KEY), repr(settings.DEEPSEEK_API_KEY[:6]))
print("settings.ARK_API_KEY:", len(settings.ARK_API_KEY), repr(settings.ARK_API_KEY[:6]))

from novelfactory.config.llm import _get_deepseek_config, _get_siliconflow_config

c = _get_deepseek_config()
print("deepseek cfg:", c["api_key"][:6], c["base_url"])
s = _get_siliconflow_config()
print("sf cfg:", (s["api_key"][:6] if s["api_key"] else "EMPTY"), s["base_url"])
