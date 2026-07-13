import json, os, subprocess

# Initialize lark-cli config at container startup
app_id = os.environ.get("LARK_APP_ID", "")
app_secret = os.environ.get("LARK_APP_SECRET", "")
if app_id and app_secret:
    cfg = {
        "apps": [
            {
                "appId": app_id,
                "appSecret": app_secret,
                "brand": "feishu",
                "defaultAs": "bot",
                "users": [
                    {
                        "userOpenId": os.environ.get("FEISHU_USER_OPEN_ID", ""),
                        "userName": "NovelFactory",
                    }
                ],
            }
        ]
    }
    lark_config_dir = os.environ.get(
        "LARK_CONFIG_DIR", os.path.expanduser("~/.lark-cli")
    )
    os.makedirs(lark_config_dir, exist_ok=True)
    with open(os.path.join(lark_config_dir, "config.json"), "w") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)
    print("lark-cli config initialized")
