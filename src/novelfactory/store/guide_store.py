"""
写作指南检索模块

负责从 PostgreSQL + Milvus 中检索写作指南。

使用方式：
    from novelfactory.state.writing_guide_store import WritingGuideStore
    store = WritingGuideStore()
    results = store.search(query="打脸爽点写法", top_k=5, genre="都市")
"""

from __future__ import annotations

import logging
from typing import Any, TypedDict
from uuid import uuid4

from novelfactory.config.constants import EMBEDDING_DIMS_DEFAULT

logger = logging.getLogger(__name__)

# WritingGuide 字段数（id, title, content, source, source_url, tags, genre, quality_score, chapter_ref, guide_type）
_WRITING_GUIDE_COLUMN_COUNT = 10


# ========== 类型定义 ==========
class WritingGuide(TypedDict):
    id: str
    title: str
    content: str
    source: str
    source_url: str | None
    tags: list[str]
    genre: str | None
    quality_score: float
    chapter_ref: str | None
    guide_type: str


class WritingGuideResult(TypedDict):
    guides: list[WritingGuide]
    total: int
    query: str


# ========== 存储模块 ==========
class WritingGuideStore:
    """写作指南的存储和检索"""

    COLLECTION_NAME = "writing_guides"
    EMBEDDING_DIM = EMBEDDING_DIMS_DEFAULT  # 复用 Qwen3-Embedding 维度

    def __init__(self) -> None:
        self._milvus_client: Any = None
        self._collection: Any = None
        self._embedding_model: Any = None
        self._initialized = False

    def _get_pg_connection(self):
        """从 DatabaseManager 单例获取短连接，使用后应通过 ``with`` 归还。"""
        from novelfactory.config.database import DatabaseManager

        db = DatabaseManager.get_instance()
        return db.get_connection()

    def _init_milvus(self) -> None:
        """初始化 Milvus 连接"""
        if self._milvus_client is not None:
            return

        try:
            from pymilvus import DataType, MilvusClient

            from novelfactory.config.settings import settings

            self._milvus_client = MilvusClient(
                uri=settings.MILVUS_URI, db_name="default"
            )
            logger.info(f"WritingGuideStore: Milvus 连接成功 ({settings.MILVUS_URI})")

            # 确保 collection 存在且 schema 正确
            existing = self._milvus_client.list_collections()
            if self.COLLECTION_NAME in existing:
                # 检查旧 schema (auto_id=True) 并迁移
                info = self._milvus_client.describe_collection(self.COLLECTION_NAME)
                if info.get("auto_id"):
                    logger.warning(
                        f"WritingGuideStore: 检测到旧版 auto_id collection, 正在重建 "
                        f"'{self.COLLECTION_NAME}'..."
                    )
                    self._milvus_client.drop_collection(self.COLLECTION_NAME)
                    existing.remove(self.COLLECTION_NAME)

            if self.COLLECTION_NAME not in existing:
                # 创建 collection: VARCHAR 主键, 不 auto_id, 动态字段已默认启用
                schema = MilvusClient.create_schema(
                    auto_id=False,
                    enable_dynamic_field=True,
                )
                schema.add_field(
                    field_name="id",
                    datatype=DataType.VARCHAR,
                    max_length=64,
                    is_primary=True,
                )
                schema.add_field(
                    field_name="vector",
                    datatype=DataType.FLOAT_VECTOR,
                    dim=self.EMBEDDING_DIM,
                )
                index_params = self._milvus_client.prepare_index_params()
                index_params.add_index(
                    field_name="vector", metric_type="IP", index_type="FLAT"
                )

                self._milvus_client.create_collection(
                    collection_name=self.COLLECTION_NAME,
                    schema=schema,
                    index_params=index_params,
                )
                logger.info(
                    f"WritingGuideStore: 创建 Milvus Collection '{self.COLLECTION_NAME}' "
                    f"(VARCHAR pk, dim={self.EMBEDDING_DIM})"
                )

            # 加载集合到查询节点，避免搜索时报 Model does not exist
            self._milvus_client.load_collection(self.COLLECTION_NAME)
            logger.info(
                f"WritingGuideStore: Milvus Collection '{self.COLLECTION_NAME}' 就绪"
            )

        except ImportError:
            logger.warning("WritingGuideStore: pymilvus 未安装，使用 PG-only 模式")
            self._milvus_client = None
        except Exception as e:
            logger.warning(f"WritingGuideStore: Milvus 连接失败: {e}")
            self._milvus_client = None

    def _init_embedding(self) -> None:
        """延迟加载 embedding 模型。

        优先级：
          1. 远程 API (OpenAI 兼容, 如 SiliconFlow) — 通过 EMBEDDING_BASE_URL 配置
          2. 本地 HuggingFace 模型 — 通过 langchain_community
        """
        if self._embedding_model is not None:
            return

        from novelfactory.config.settings import settings

        # 方案 1: 远程 API (langchain-openai 已安装)
        if settings.EMBEDDING_BASE_URL:
            try:
                from langchain_openai import OpenAIEmbeddings

                self._embedding_model = OpenAIEmbeddings(
                    model=settings.EMBEDDING_MODEL,
                    openai_api_key=settings.EMBEDDING_API_KEY,
                    openai_api_base=settings.EMBEDDING_BASE_URL,
                )
                logger.info(
                    "WritingGuideStore: 远程 Embedding 加载成功 (%s)",
                    settings.EMBEDDING_BASE_URL,
                )
                return
            except Exception as e:
                logger.warning(
                    f"WritingGuideStore: 远程 Embedding 加载失败，降级到本地: {e}"
                )

        # 方案 2: 本地 HuggingFace 模型 (需要 langchain-community)
        try:
            from langchain_community.embeddings import HuggingFaceBgeEmbeddings

            self._embedding_model = HuggingFaceBgeEmbeddings(
                model_name=settings.EMBEDDING_MODEL,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )
            logger.info("WritingGuideStore: 本地 Embedding 模型加载成功")
        except ImportError:
            logger.warning(
                "WritingGuideStore: langchain_community 未安装，Embedding 功能不可用，"
                "将降级到 PG-only 检索模式"
            )
            self._embedding_model = None
        except Exception as e:
            logger.warning(f"WritingGuideStore: 本地 Embedding 模型加载失败: {e}")
            self._embedding_model = None

    def initialize(self) -> None:
        """显式初始化所有连接（PG 连接按需获取，不再预初始化）"""
        self._init_milvus()
        self._init_embedding()
        self._initialized = True

    # ========== 检索方法 ==========

    def search(
        self,
        query: str,
        top_k: int = 5,
        genre: str | None = None,
        guide_type: str | None = None,
        min_quality: float = 0.6,
        tags: list[str] | None = None,
    ) -> WritingGuideResult:
        """
        检索写作指南。

        Args:
            query: 检索关键词/语义query
            top_k: 返回数量
            genre: 按题材过滤
            guide_type: 按类型过滤（technique/analysis）
            min_quality: 最低质量分数
            tags: 按标签过滤

        Returns:
            WritingGuideResult
        """
        if not self._initialized:
            self.initialize()

        # 优先使用 Milvus 向量检索
        if self._milvus_client and self._embedding_model:
            return self._search_with_milvus(
                query, top_k, genre, guide_type, min_quality, tags
            )
        # 降级：只用 PG 全文检索
        return self._search_with_pg_only(
            query, top_k, genre, guide_type, min_quality, tags
        )

    def _search_with_milvus(
        self,
        query: str,
        top_k: int,
        genre: str | None,
        guide_type: str | None,
        min_quality: float,
        tags: list[str] | None,
    ) -> WritingGuideResult:
        """Milvus 向量检索 + PG 过滤"""
        assert self._embedding_model is not None, "embedding_model not initialized"
        assert self._milvus_client is not None, "milvus_client not initialized"
        try:
            # 1. 生成 query embedding
            query_vec = self._embedding_model.embed_query(query)
            query_vec = [float(v) for v in query_vec]

            # 2. 先从 PG 筛选候选 IDs
            candidate_ids = self._get_candidate_ids(
                genre, guide_type, min_quality, tags, limit=100
            )

            if not candidate_ids:
                return WritingGuideResult(guides=[], total=0, query=query)

            # 3. Milvus 向量检索
            results = self._milvus_client.search(
                collection_name=self.COLLECTION_NAME,
                data=[query_vec],
                limit=top_k,
                output_fields=["id", "pg_id"],
            )

            # 4. 从 PG 获取详情
            guides = []
            for hit in results[0]:
                pg_id = hit.get("entity", {}).get("pg_id")
                if pg_id:
                    guide = self._get_guide_by_pg_id(str(pg_id))
                    if guide is not None:
                        raw: dict[str, Any] = dict(guide)
                        raw["_score"] = hit.get("distance", 0)
                        guides.append(raw)

            return WritingGuideResult(
                guides=guides[:top_k],  # type: ignore[typeddict-item]
                total=len(guides),
                query=query,
            )

        except Exception as e:
            logger.warning(f"Milvus 检索失败，降级到 PG: {e}")
            return self._search_with_pg_only(
                query, top_k, genre, guide_type, min_quality, tags
            )

    def _search_with_pg_only(
        self,
        query: str,
        top_k: int,
        genre: str | None,
        guide_type: str | None,
        min_quality: float,
        tags: list[str] | None,
    ) -> WritingGuideResult:
        """PostgreSQL 全文检索（降级方案）"""
        sql = """
            SELECT id, title, content, source, source_url, tags, genre,
                   quality_score, chapter_ref, guide_type
            FROM writing_guides
            WHERE 1=1
        """
        params: list[Any] = []
        param_idx = 1

        if query:
            sql += " AND (title ILIKE %s OR content ILIKE %s)"
            params.append(f"%{query}%")
            params.append(f"%{query}%")
            param_idx += 2

        if genre:
            sql += " AND genre = %s"
            params.append(genre)
            param_idx += 1

        if guide_type:
            sql += " AND guide_type = %s"
            params.append(guide_type)
            param_idx += 1

        sql += " AND quality_score >= %s"
        params.append(min_quality)
        param_idx += 1

        if tags:
            sql += " AND tags && %s"
            params.append(tags)

        sql += f" ORDER BY quality_score DESC LIMIT {top_k}"

        try:
            logger.debug(
                f"PG search: about to execute, sql={sql[:80]}, params={params}"
            )
            conn = self._get_pg_connection()
            if conn is None:
                return WritingGuideResult(guides=[], total=0, query=query)
            with conn:
                with conn.cursor() as cur:
                    cur.execute(sql, params)
                    rows = cur.fetchall()
                # conn.__exit__ 提交 + 归还连接
            logger.debug(f"PG search returned {len(rows)} rows")
            guides = []
            for row in rows:
                if len(row) < _WRITING_GUIDE_COLUMN_COUNT:
                    logger.warning(
                        f"PG row has only {len(row)} columns, expected {_WRITING_GUIDE_COLUMN_COUNT}: {row}"
                    )
                    continue
                guides.append(
                    WritingGuide(
                        id=str(row[0]),
                        title=row[1],
                        content=row[2],
                        source=row[3],
                        source_url=row[4],
                        tags=row[5] or [],
                        genre=row[6],
                        quality_score=row[7],
                        chapter_ref=row[8],
                        guide_type=row[9],
                    )
                )
            return WritingGuideResult(guides=guides, total=len(guides), query=query)
        except Exception as e:
            import traceback

            logger.error(f"PG 检索失败: {e}")
            logger.error(f"PG 检索 traceback: {traceback.format_exc()}")
            return WritingGuideResult(guides=[], total=0, query=query)

    def _get_candidate_ids(
        self,
        genre: str | None,
        guide_type: str | None,
        min_quality: float,
        tags: list[str] | None,
        limit: int = 100,
    ) -> list[str]:
        """从 PG 获取候选 IDs 用于 Milvus 检索"""
        sql = "SELECT id FROM writing_guides WHERE quality_score >= %s"
        params: list[Any] = [min_quality]

        if genre:
            sql += " AND genre = %s"
            params.append(genre)
        if guide_type:
            sql += " AND guide_type = %s"
            params.append(guide_type)
        if tags:
            sql += " AND tags && %s"
            params.append(tags)

        sql += f" LIMIT {limit}"

        try:
            conn = self._get_pg_connection()
            if conn is None:
                return []
            with conn:
                with conn.cursor() as cur:
                    cur.execute(sql, params)
                    return [str(row[0]) for row in cur.fetchall()]
        except Exception as e:
            logger.error(f"候选 ID 查询失败: {e}")
            return []

    def _get_guide_by_pg_id(self, pg_id: str) -> WritingGuide | None:
        """从 PG 获取单条指南详情"""
        sql = """
            SELECT id, title, content, source, source_url, tags, genre,
                   quality_score, chapter_ref, guide_type
            FROM writing_guides WHERE id = %s
        """

        try:
            conn = self._get_pg_connection()
            if conn is None:
                return None
            with conn:
                with conn.cursor() as cur:
                    cur.execute(sql, (pg_id,))
                    row = cur.fetchone()
                    if row:
                        return WritingGuide(
                            id=str(row[0]),
                            title=row[1],
                            content=row[2],
                            source=row[3],
                            source_url=row[4],
                            tags=row[5] or [],
                            genre=row[6],
                            quality_score=row[7],
                            chapter_ref=row[8],
                            guide_type=row[9],
                        )
        except Exception as e:
            logger.error(f"PG 单条查询失败: {e}")
        return None

    # ========== 写入方法 ==========

    def add_guide(
        self,
        title: str,
        content: str,
        source: str = "manual",
        source_url: str | None = None,
        tags: list[str] | None = None,
        genre: str | None = None,
        quality_score: float = 0.5,
        chapter_ref: str | None = None,
        guide_type: str = "technique",
    ) -> str:
        """
        添加一条写作指南到数据库。
        会同时写入 PostgreSQL 和 Milvus。

        Returns:
            新记录的 ID
        """
        if not self._initialized:
            self.initialize()

        guide_id = str(uuid4())

        conn = self._get_pg_connection()
        if conn is not None:
            sql = """
                INSERT INTO writing_guides
                (id, title, content, source, source_url, tags, genre,
                 quality_score, chapter_ref, guide_type)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id
            """
            try:
                with conn:
                    with conn.cursor() as cur:
                        cur.execute(
                            sql,
                            (
                                guide_id,
                                title,
                                content,
                                source,
                                source_url,
                                tags or [],
                                genre,
                                quality_score,
                                chapter_ref,
                                guide_type,
                            ),
                        )
                        result = cur.fetchone()
                        if result:
                            guide_id = str(result[0])
                    # conn.__exit__ 提交 + 归还连接
                logger.info(f"WritingGuideStore: 写入 PG 成功, id={guide_id}")
            except Exception as e:
                logger.error(f"WritingGuideStore: 写入 PG 失败: {e}")
                raise

        # 写入 Milvus（如果可用）
        if self._milvus_client and self._embedding_model and guide_id:
            assert self._embedding_model is not None  # type narrowing for mypy
            assert self._milvus_client is not None
            try:
                search_text = f"{title}\n{content}"
                vec = self._embedding_model.embed_query(search_text)
                vec = [float(v) for v in vec]

                self._milvus_client.insert(
                    collection_name=self.COLLECTION_NAME,
                    data=[
                        {
                            "id": guide_id,
                            "title": title,
                            "content": content[:500],  # 截断
                            "pg_id": guide_id,
                            "vector": vec,
                        }
                    ],
                )
                logger.info(f"WritingGuideStore: 写入 Milvus 成功, id={guide_id}")
            except Exception as e:
                logger.warning(f"WritingGuideStore: 写入 Milvus 失败: {e}")

        return guide_id

    def count(self) -> int:
        """统计指南总数"""
        try:
            conn = self._get_pg_connection()
            if conn is None:
                return 0
            with conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT COUNT(*) FROM writing_guides")
                    return cur.fetchone()[0]
        except Exception as e:
            logger.warning(f"WritingGuideStore count failed: {e}")
            return 0

    def is_connected(self) -> bool:
        """Whether the guide store is initialized and ready."""
        return self._initialized

    def close(self) -> None:
        """Release Milvus client resources (PG uses short connections via DatabaseManager)."""
        if self._milvus_client is not None:
            try:
                self._milvus_client.close()
            except Exception:
                pass
            self._milvus_client = None
        self._initialized = False


# ========== 单例实例 ==========
_guide_store: WritingGuideStore | None = None


def get_guide_store() -> WritingGuideStore:
    """获取写作指南存储单例"""
    global _guide_store
    if _guide_store is None:
        _guide_store = WritingGuideStore()
    return _guide_store


# ========== 命令行测试 ==========
if __name__ == "__main__":
    store = WritingGuideStore()
    store.initialize()

    print(f"\n当前指南总数: {store.count()}")

    results = store.search(query="打脸爽点写法", top_k=3)
    print("\n检索 '打脸爽点写法':")
    for guide in results["guides"]:
        print(f"  [{guide['title']}] score={guide.get('_score', 'N/A')}")
        print(f"    {guide['content'][:100]}...")
