# NovelFactory v6.1 — Makefile
# =============================================================================

.PHONY: help dev build up down restart prod health db lint typecheck format test
.PHONY: validate security deps audit logs sync clean

help:  ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ── Development ────────────────────────────────────────────────────────
dev:  ## 启动开发服务器 (原生 uvicorn, 需要 WSL 或 Git Bash)
	python -m uvicorn novelfactory.server.app:app --host 0.0.0.0 --port 8000 --reload

# ── Docker ─────────────────────────────────────────────────────────────
build:  ## 构建 Docker 镜像
	docker compose build api

up:  ## 启动所有服务
	docker compose -p langgraph up -d

down:  ## 停止所有服务
	docker compose -p langgraph down

restart:  down up  ## 重启所有服务

prod:  ## Start with production overrides
	docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d

# ── Health ─────────────────────────────────────────────────────────────
health:  ## 运行所有服务的健康检查
	@echo "=== PostgreSQL ==="
	@docker compose -p langgraph exec postgres pg_isready -U noveluser -d novelfactory 2>/dev/null || echo "PG: not running"
	@echo "=== Redis ==="
	@docker compose -p langgraph exec redis redis-cli -a $$REDIS_PASSWORD ping 2>/dev/null || echo "Redis: not running"
	@echo "=== API ==="
	@curl -s http://localhost:8123/health | python -m json.tool 2>/dev/null || echo "API: not running"
	@echo "=== Neo4j ==="
	@curl -s http://localhost:7474 2>/dev/null || echo "Neo4j: not running"
	@echo "=== Milvus ==="
	@curl -s http://localhost:9091/healthz 2>/dev/null || echo "Milvus: not running"

db:  ## 连接到 PostgreSQL
	docker compose -p langgraph exec postgres psql -U noveluser -d novelfactory

# ── Code Quality ───────────────────────────────────────────────────────
lint:  ## 运行 ruff 代码检查
	ruff check src/ --select E,F,W,I,N,UP --ignore E501

typecheck:  ## 运行 mypy 类型检查
	mypy src/ --ignore-missing-imports --check-untyped-defs

validate: lint typecheck  ## 运行完整代码质量检查（lint + typecheck）

format:  ## 使用 ruff 自动格式化
	ruff format src/

security:  ## 运行 ruff 安全检查
	ruff check src/ --select S

# ── Dependency Management ─────────────────────────────────────────────
deps:  ## 审计依赖项已知漏洞
	uv pip freeze 2>/dev/null | pip-audit -r /dev/stdin 2>&1 || echo "pip-audit 不可用，请安装：pip install pip-audit"

# ── Full Audit ─────────────────────────────────────────────────────────
audit: validate security deps  ## 完整代码审计（lint + typecheck + security + deps）

# ── Testing ────────────────────────────────────────────────────────────
test:  ## 运行所有测试
	python -m pytest tests/ -v --tb=short

# ── Lifecycle ──────────────────────────────────────────────────────────
logs:  ## 查看 API 日志
	docker compose -p langgraph logs -f api

hot-reload:  ## 热重载 Python 代码到运行中的容器（无需构建镜像）
	@CONTAINER=langgraph_api; \
	echo "=== Hot-reloading to $$CONTAINER ==="; \
	docker cp src/novelfactory/server/app.py $$CONTAINER:/app/src/novelfactory/server/app.py; \
	docker cp src/novelfactory/server/routes/. $$CONTAINER:/app/src/novelfactory/server/routes/; \
	docker cp src/novelfactory/server/streaming.py $$CONTAINER:/app/src/novelfactory/server/streaming.py; \
	docker restart $$CONTAINER; \
	echo "=== Waiting for container to be ready ==="; \
	sleep 5; \
	curl -s http://localhost:8123/health && echo

clean:  ## 清理 __pycache__ 和临时文件
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name '*.pyc' -delete 2>/dev/null || true
	rm -rf .pytest_cache .mypy_cache .ruff_cache 2>/dev/null || true
