# CLAUDE.md

## 项目概览

Huobao Drama 是一个 AI 短剧/视频生产工具，使用 TypeScript 全栈实现。

## 目录结构

```
backend/   — Hono + Drizzle ORM + Mastra（AI 智能体）+ better-sqlite3
frontend/  — Nuxt 3 + Vue 3 + TypeScript（纯 CSS，无 UI 框架）
configs/   — config.yaml 配置文件
data/      — SQLite 数据库 + 静态资源文件
skills/    — 智能体 SKILL.md 定义
```

## 常用命令

### 后端（`backend/`）
- `npm run dev` — 使用 tsx watch 启动开发服务，默认端口 `5679`
- `npm start` — 启动生产服务
- `npm run typecheck` — TypeScript 类型检查

### 前端（`frontend/`）
- `npm run dev` — 启动 Nuxt 开发服务，端口 `3013`，代理 `/api` 到 `5679`
- `npm run build` — 生产构建

## 架构说明

### 后端
- **HTTP**：Hono 框架，包含 CORS 和 logger 中间件
- **数据库**：Drizzle ORM + better-sqlite3，WAL 模式，schema 在 `src/db/schema.ts`
- **AI 智能体**：Mastra 框架 + AI SDK，支持兼容 OpenAI 协议的服务商
- **智能体类型**：`script_rewriter`、`extractor`、`storyboard_breaker`
- **SSE 流式响应**：通过 Hono `streamSSE` 返回智能体对话响应
- **文件存储**：本地文件系统，默认位于 `data/static/`

### 前端
- **框架**：Nuxt 3 + Vue 3 + TypeScript
- **路由**：Nuxt 文件路由，页面位于 `frontend/app/pages`
- **状态/API**：组合式函数位于 `frontend/app/composables`
- **API 客户端**：统一请求客户端在 `frontend/app/composables/useApi.ts`
- **样式**：纯 CSS + CSS variables，暗色主题

## 数据库

SQLite 默认路径是 `data/huobao_drama.db`，也可以通过 `DB_PATH` 覆盖。
表结构匹配既有 GORM 创建的表结构。
自动启用 WAL 模式。无需迁移，当前代码直接读取既有 DB。

## 关键配置

- `configs/config.yaml` 是环境私有的本地配置文件；当前后端运行时不会把它作为 `PORT` 或 `DB_PATH` 的真值。
- AI 服务配置保存在数据库 `ai_service_configs` 表。
- 智能体配置保存在数据库 `agent_configs` 表。

## 环境隔离

- 编辑运行时配置、AI 服务商设置、数据库内容、生成素材或 ComfyUI 工作流前，先阅读 `docs/environment-isolation.md`。
- `configs/config.yaml`、`.env*`、SQLite DB 文件、生成素材和真实 ComfyUI 工作流文件都必须保持环境私有。
- 从另一台机器或容器运行/编辑前，先确认该环境的 `PORT`、`DB_PATH`、静态资源路径，以及设置页面中的 AI/智能体配置。
