"""Application settings for NovelFactory LangGraph.

Supports multiple configuration sources (highest to lowest priority):
  1. Environment variables
  2. .env file
  3. Default values

Tech stack: DeepSeek V4 Flash (via 火山引擎 Coding Plan) — replaced MiniMax 2026-06-14,
switched to ark.cn-beijing.volces.com 2026-06-15.
"""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings

# ── Environment variable overrides ─────────────────────────────────────────
# Referenced from TradingAgents' default_config.py _ENV_OVERRIDES pattern.

_ENV_OVERRIDES: dict[str, str] = {
    "NOVELFACTORY_APP_VERSION": "APP_VERSION",
    "NOVELFACTORY_LOG_LEVEL": "LOG_LEVEL",
    "NOVELFACTORY_CHECKPOINT_TYPE": "CHECKPOINT_TYPE",
    "NOVELFACTORY_STORAGE_TYPE": "STORAGE_TYPE",
    "NOVELFACTORY_QUOTA_CHECK_BEFORE_CALL": "QUOTA_CHECK_BEFORE_CALL",
    "NOVELFACTORY_MAX_RETRIES": "MAX_RETRIES",
    "NOVELFACTORY_CHAPTER_MIN_WORD_COUNT": "CHAPTER_MIN_WORD_COUNT",
    "NOVELFACTORY_CHAPTER_TARGET_WORD_COUNT": "CHAPTER_TARGET_WORD_COUNT",
    "NOVELFACTORY_QUOTA_THRESHOLD": "QUOTA_THRESHOLD",
}

_BOOL_TRUE = ("true", "1", "yes", "on")
_BOOL_FALSE = ("false", "0", "no", "off")


def _coerce_env(value: str, reference: object) -> object:
    """Coerce env-var string to the type of the existing default value."""
    if isinstance(reference, bool):
        normalized = value.strip().lower()
        if normalized in _BOOL_TRUE:
            return True
        if normalized in _BOOL_FALSE:
            return False
        raise ValueError(
            f"expected a boolean ({'/'.join(_BOOL_TRUE + _BOOL_FALSE)}), got {value!r}"
        )
    if isinstance(reference, int) and not isinstance(reference, bool):
        return int(value)
    if isinstance(reference, float):
        return float(value)
    return value


class Settings(BaseSettings):
    """Application settings."""

    # ── Version (唯一源) ──────────────────────────────────────────────────────
    APP_VERSION: str = Field(
        default="8.0.0",
        validation_alias="NOVELFACTORY_VERSION",
        description="应用版本号，通过 NOVELFACTORY_VERSION 环境变量或 APP_VERSION 覆盖",
    )

    # ── LLM (ARK API — 权威来源) ─────────────────────────────────────────
    ARK_API_KEY: str = Field(default="", description="火山引擎方舟 API Key")
    ARK_BASE_URL: str = Field(
        default="https://ark.cn-beijing.volces.com/api/coding/v3",
        description="火山引擎 Coding Plan OpenAI 兼容端点",
    )

    # ── Database ───────────────────────────────────────────────────────────────
    DATABASE_URL: str = Field(default="", description="PostgreSQL连接URL（完整格式）")
    DB_HOST: str = Field(default="localhost", description="PostgreSQL主机")
    DB_PORT: int = Field(default=5432, description="PostgreSQL端口")
    DB_NAME: str = Field(default="novelfactory", description="数据库名")
    DB_USER: str = Field(default="noveluser", description="数据库用户")
    DB_PASSWORD: str = Field(default="novelpass2024", description="数据库密码（Docker默认密码，生产环境请修改）")

    # ── Neo4j ──────────────────────────────────────────────────────────────────
    NEO4J_HOST: str = Field(default="localhost", description="Neo4j主机")
    NEO4J_PORT: int = Field(default=7687, description="Neo4j Bolt端口")
    NEO4J_USER: str = Field(default="neo4j", description="Neo4j用户名")
    NEO4J_PASSWORD: str = Field(default="novelgraph2024", description="Neo4j密码（Docker默认密码，生产环境请修改）")

    # ── Milvus ─────────────────────────────────────────────────────────────────
    MILVUS_HOST: str = Field(default="localhost", description="Milvus主机")
    MILVUS_PORT: int = Field(default=19530, description="Milvus端口")

    @property
    def MILVUS_URI(self) -> str:  # noqa: N802
        """Milvus gRPC URI, 拼接 host 和 port."""
        return f"http://{self.MILVUS_HOST}:{self.MILVUS_PORT}"

    # ── Embedding ──────────────────────────────────────────────────────────────
    EMBEDDING_MODEL: str = Field(
        default="Qwen/Qwen3-Embedding-0.6B",
        description="Embedding 模型名称（HuggingFace 或 SiliconFlow 模型名）",
    )
    EMBEDDING_BASE_URL: str = Field(
        default="",
        description="Embedding API 基础地址（为空则使用本地加载的 HuggingFace 模型）",
    )
    EMBEDDING_API_KEY: str = Field(
        default="",
        description="Embedding API 密钥（远程 API 时必填）",
    )
    EMBEDDING_DIMS: int = Field(
        default=768,
        description="Embedding 向量维度",
    )

    # ── Redis ─────────────────────────────────────────────────────────────────
    REDIS_URL: str = Field(
        default="", description="Redis连接URL（完整格式，优先于组件配置）"
    )
    REDIS_HOST: str = Field(default="localhost", description="Redis主机")
    REDIS_PORT: int = Field(default=6379, description="Redis端口")
    REDIS_DB: int = Field(default=0, description="Redis数据库编号")
    REDIS_PASSWORD: str = Field(default="novelredis2024", description="Redis密码（Docker默认密码，生产环境请修改）")

    # ── Timeouts ───────────────────────────────────────────────────────────────
    LLM_REQUEST_TIMEOUT: float = Field(
        default=300.0,  # 5分钟 — 长文本生成(3000字+)需要更长的超时时间
        description="LLM API 请求超时(秒)",
    )
    DB_CONNECT_TIMEOUT: int = Field(default=10, description="数据库连接超时(秒)")
    DB_STATEMENT_TIMEOUT: int = Field(default=30000, description="SQL语句超时(毫秒)")
    DB_POOL_MIN_SIZE: int = Field(default=2, description="数据库连接池最小连接数")
    DB_POOL_MAX_SIZE: int = Field(default=10, description="数据库连接池最大连接数")

    # ── LangSmith ──────────────────────────────────────────────────────────────
    LANGSMITH_API_KEY: str = Field(default="", description="LangSmith API密钥")
    LANGSMITH_TRACING: bool = Field(default=False, description="是否启用LangSmith追踪")
    LANGSMITH_PROJECT: str = Field(
        default="novelfactory", description="LangSmith项目名"
    )
    LANGSMITH_ENDPOINT: str = Field(
        default="",
        description="LangSmith API端点 (EU: https://eu.api.smith.langchain.com)",
    )
    LANGSMITH_WORKSPACE_ID: str = Field(default="", description="LangSmith工作区ID")
    LANGSMITH_ORG_ID: str = Field(default="", description="LangSmith组织ID")

    # ── Storage ──────────────────────────────────────────────────────────────
    STORAGE_TYPE: str = Field(
        default="postgres", description="存储类型: postgres/local/feishu/s3"
    )
    STORAGE_PATH: str = Field(default="~/.novelfactory", description="本地存储路径")

    BASE_URL: str = Field(default="http://localhost:8123", description="应用访问地址")

    # ── Checkpoint ─────────────────────────────────────────────────────────────
    CHECKPOINT_TYPE: str = Field(
        default="postgres",
        description="检查点类型: memory/postgres/redis/sqlite",
    )

    # ── Logging ────────────────────────────────────────────────────────────────
    LOG_LEVEL: str = Field(default="INFO", description="日志级别")
    LOG_FORMAT: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="日志格式",
    )

    # ── Retry ─────────────────────────────────────────────────────────────────
    MAX_RETRIES: int = Field(default=3, description="最大重试次数")
    RETRY_BACKOFF_FACTOR: float = Field(default=2.0, description="重试退避因子")
    RETRY_MAX_INTERVAL: float = Field(default=60.0, description="最大重试间隔(秒)")

    # ── Business ──────────────────────────────────────────────────────────────
    QUOTA_THRESHOLD: float = Field(default=5.0, description="配额警告阈值(%)")
    QUOTA_CHECK_BEFORE_CALL: bool = Field(
        default=False,
        description="是否在每次 LLM 调用前检查配额（启用严格预算控制）",
    )
    QUOTA_CHECK_INTERVAL_SECONDS: float = Field(
        default=60.0,
        description="配额检查最小间隔(秒)",
    )
    CHAPTER_MIN_WORD_COUNT: int = Field(default=1500, description="章节最少字数")
    CHAPTER_TARGET_WORD_COUNT: int = Field(default=3000, description="章节目标字数")
    GRADE_C_THRESHOLD: int = Field(default=60, description="C级及格分数")
    GRADE_B_THRESHOLD: int = Field(default=75, description="B级良好分数")
    GRADE_A_THRESHOLD: int = Field(default=90, description="A级优秀分数")
    MAX_KEEP_CHAPTERS: int = Field(default=100, description="内存中保留的最大章节数")
    COMPRESS_KEEP_RECENT: int = Field(default=50, description="压缩时保留的最近章节数")

    # ── Feishu ─────────────────────────────────────────────────────────────────
    LARK_APP_ID: str = Field(default="", description="飞书应用 ID")
    LARK_APP_SECRET: str = Field(default="", description="飞书应用密钥")
    LARK_PROXY_HOST: str = Field(
        default="172.28.0.1",
        description="tools-proxy 宿主机地址（Docker bridge 网关）",
    )
    LARK_PROXY_PORT: int = Field(
        default=5004,
        description="tools-proxy 服务端口",
    )
    LARK_PROXY_ENABLED: bool = Field(
        default=True,
        description="是否通过 tools-proxy HTTP 调用 lark-cli（关闭则回退 subprocess）",
    )
    FEISHU_USER_OPEN_ID: str = Field(
        default="", description="飞书用户 Open ID（接收通知）"
    )
    FEISHU_CHAT_ID: str = Field(
        default="", description="飞书群聊 Chat ID（接收群通知，oc_ 前缀）"
    )
    FEISHU_ROOT_FOLDER: str = Field(default="", description="飞书根目录 Folder Token")
    FEISHU_VERIFICATION_TOKEN: str = Field(default="", description="飞书回调验证 Token")
    FEISHU_ENCRYPT_KEY: str = Field(default="", description="飞书回调加密 Key")

    # ── Channels (Feishu WebSocket) ────────────────────────────────────────────
    CHANNELS_ENABLED: bool = Field(
        default=True, description="是否启用渠道层（FeishuChannel WebSocket）"
    )

    # ── CORS ─────────────────────────────────────────────────────────────────
    CORS_ALLOWED_ORIGINS: str = Field(
        default="*", description="CORS 允许的源（逗号分隔）"
    )

    # ── Server ───────────────────────────────────────────────────────────────
    HOST: str = Field(default="0.0.0.0", description="服务器监听地址")
    PORT: int = Field(default=8000, description="服务器监听端口")

    # ── Project ──────────────────────────────────────────────────────────────
    PROJECT_NAME: str = Field(default="江寻录", description="项目名称")

    # ── Pricing ──────────────────────────────────────────────────────────────
    DEEPSEEK_PRICE_INPUT: float = Field(
        default=0.5, description="DeepSeek 输入价格（每百万 token）"
    )
    DEEPSEEK_PRICE_OUTPUT: float = Field(
        default=2.0, description="DeepSeek 输出价格（每百万 token）"
    )
    DEEPSEEK_PRICE_DEFAULT_INPUT: float = Field(
        default=0.5, description="默认输入价格（每百万 token）"
    )
    DEEPSEEK_PRICE_DEFAULT_OUTPUT: float = Field(
        default=2.0, description="默认输出价格（每百万 token）"
    )

    # ── Monitoring & Alerting ──────────────────────────────────────────────────
    AUDIT_LOG_ENABLED: bool = Field(default=False, description="是否启用审计日志")
    AUDIT_LOG_DIR: str = Field(
        default="~/.novelfactory/audit", description="审计日志目录"
    )
    FEISHU_ALERT_WEBHOOK: str = Field(default="", description="飞书告警 Webhook URL")
    ALERT_MIN_LEVEL: str = Field(default="warning", description="最小告警级别")
    NOVELFACTORY_ENV: str = Field(
        default="development", description="运行环境: development/staging/production"
    )
    NOVELFACTORY_LOG_PATH: str = Field(
        default="", description="日志文件路径（空=stdout）"
    )

    # ── Properties ──────────────────────────────────────────────────────────────

    @property
    def lark_proxy_url(self) -> str:
        return f"http://{self.LARK_PROXY_HOST}:{self.LARK_PROXY_PORT}"

    @property
    def database_url(self) -> str:
        if self.DATABASE_URL:
            return self.DATABASE_URL
        return (
            f"postgresql://{self.DB_USER}:{self.DB_PASSWORD}"
            f"@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"
        )

    @property
    def redis_url(self) -> str:
        if self.REDIS_PASSWORD:
            return f"redis://:{self.REDIS_PASSWORD}@{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"
        return f"redis://{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"

    @property
    def storage_path_expanded(self) -> str:
        import os

        return os.path.expanduser(self.STORAGE_PATH)

    @property
    def is_production(self) -> bool:
        return self.NOVELFACTORY_ENV == "production"

    @property
    def checkpoint_config(self) -> dict:
        return {
            "type": self.CHECKPOINT_TYPE,
            "database_url": self.database_url
            if self.CHECKPOINT_TYPE == "postgres"
            else None,
            "redis_url": self.redis_url if self.CHECKPOINT_TYPE == "redis" else None,
        }

    def model_post_init(self, __context: object) -> None:
        """Apply NOVELFACTORY_* env-var overrides after pydantic-settings init."""
        import os

        for env_var, key in _ENV_OVERRIDES.items():
            raw = os.environ.get(env_var)
            if raw is None or raw == "":
                continue
            current = getattr(self, key, None)
            try:
                coerced = _coerce_env(raw, current)
                setattr(self, key, coerced)
            except ValueError as exc:
                raise ValueError(f"Invalid value for {env_var}: {exc}") from exc

        # v6.1: 启动时输出配置摘要
        self._log_effective_config()

    def _log_effective_config(self) -> None:
        """启动时输出所有生效配置项，含来源标注。"""
        import logging

        _logger = logging.getLogger(__name__)
        _logger.info("=== Effective Config ===")
        for field_name in type(self).model_fields:
            value = getattr(self, field_name)
            # 掩码敏感字段
            if any(
                kw in field_name.lower()
                for kw in ["key", "secret", "password", "token"]
            ):
                value = f"{str(value)[:4]}...{str(value)[-4:]}" if value else ""
            source = (
                "ENV_OVERRIDE"
                if field_name in _ENV_OVERRIDES.values()
                else "default/env"
            )
            _logger.info("  %s = %s (%s)", field_name, value, source)
        _logger.info("========================")

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": True,
        "extra": "ignore",
    }


settings = Settings()
