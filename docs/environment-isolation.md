# 环境隔离说明

本项目会把运行时状态写入本地文件和 SQLite 表。除非已经明确做过迁移或导出审查，否则每个环境都要当成彼此独立的工作空间。

## 目标

- 本机开发、Docker、测试和生产环境的数据相互独立。
- 避免误复用 AI 服务端点、API Key、ComfyUI 工作流、生成素材或数据库记录。
- 明确哪些文件是随仓库共享的模板，哪些文件是环境私有的运行时状态。

## 受版本控制的内容与环境私有内容

受版本控制的内容用于定义项目本身：

- `backend/`、`frontend/`、`skills/`
- `configs/config.example.yaml`
- `Dockerfile`、`docker-compose.yml`
- `docs/`

环境私有文件不能当成跨环境共享的项目真值：

- `configs/config.yaml`
- `.env`、`.env.local`
- `data/huobao_drama.db`、`data/huobao_drama.db-wal`、`data/huobao_drama.db-shm`
- `data/static/`、`data/storage/`
- 生成的视频、图片、音频、字幕和临时导出文件
- 后续真实 ComfyUI 工作流 JSON，凡是包含真实端点 URL、模型路径、输出路径、seed、凭据或机器特定节点设置的，都属于环境私有内容

当前 `.gitignore` 已经排除了这些本地状态文件。从其他环境编辑本仓库时，不要删除这些 ignore 规则。

## 运行时配置

当前后端会直接读取这些环境变量：

| 变量 | 当前行为 | 隔离规则 |
|---|---|---|
| `PORT` | 后端 HTTP 端口，默认 `5679`。 | 每个环境单独设置，避免端口冲突。 |
| `DB_PATH` | SQLite 数据库路径，默认 `data/huobao_drama.db`。 | 每个环境使用独立数据库文件。 |
| `STORAGE_PATH` | 部分存储工具会读取该值，默认 `data/static`。 | 在静态 URL 服务路径一起改造前，不要随意指向任意自定义目录。 |

注意：`configs/config.yaml` 是环境私有配置文件，但当前后端入口没有把它作为 `PORT`、`DB_PATH` 或 `STORAGE_PATH` 的运行时真值。如果某个环境需要通过 YAML 驱动运行时配置，必须先在代码里接入，并文档化配置优先级。

因为文件 URL 会以 `static/...` 形式返回，后端又从项目 `data/` 目录服务 `/static/*`，所以当前最稳妥的生成素材隔离方式是使用独立 checkout、独立容器 volume，或保持 `data/static` 结构的数据目录。不要把 `STORAGE_PATH` 指向随机绝对路径后，就假设浏览器仍然能正常访问素材。

## SQLite 状态

SQLite 不只是业务数据，也保存运行时配置：

- `ai_service_configs`：服务商、Base URL、API Key、模型列表、优先级、启用状态
- `agent_configs`：智能体模型、提示词、temperature、token 限制、启用状态
- 短剧、剧集、角色、场景、分镜、图片、视频、音频等业务记录
- `local_path`、`image_url`、`video_url`、`tts_audio_url` 等生成文件引用

规则：

- 不要把 `data/huobao_drama.db*` 当作普通同步文件在环境之间随手复制。
- 在 UI 里修改设置前，先确认后端进程实际使用的是预期的 `DB_PATH`。
- 测试新服务商、新模型或 ComfyUI 桥接前，使用临时数据库，或先备份当前 DB 文件。
- 如果需要种子数据，导出经过审查且不含凭据的数据夹具，不要直接复制工作数据库。

## AI 服务与智能体设置

设置页面填写的值会保存到 SQLite，因此都是当前环境的本地状态，包括：

- API Key
- Base URL
- 模型名称
- 服务商优先级和启用状态
- 智能体提示词和模型覆盖配置

不要把 Web UI 里的状态当成仓库级配置。如果另一个环境需要同样的逻辑设置，应在那个环境中用适合当地环境的端点、密钥和模型重新配置。

## ComfyUI 工作流边界

ComfyUI 支持应该分成两层保存和说明：

- 模板层：可提交到仓库的文档或示例工作流，用来描述必填输入、占位符名称、预期输出和适配器假设。
- 环境层：真实 ComfyUI 工作流 JSON、服务地址、节点 ID、模型名称、checkpoint 路径、LoRA 路径、输出目录和回调 URL。

如果真实工作流包含以下内容，不要直接提交：

- 绝对本地路径
- 局域网 IP 或机器名
- 不同环境会变化的 API 端点
- 凭据或签名 URL
- 不是所有环境都可用的服务商特定模型 ID
- 指向另一个环境共享 `data/static` 目录的输出路径

后续新增 ComfyUI 集成时，优先采用这类命名：

- `configs/comfyui/workflows/<name>.example.json`：模板
- `docs/comfyui/<name>.md`：字段映射和环境说明
- 被 `.gitignore` 排除的本地文件，或外部密钥/配置系统：保存真实环境值

## 凭据

不要把 API Key、token、ComfyUI 凭据、服务商凭据或个人服务密码写入仓库文件。

需要长期复用的本机外部系统凭据时，优先查本机 Codex service account 和 Keychain 配置。不要从旧聊天、截图或复制来的数据库中回捞明文凭据。

## 跨环境编辑检查清单

在另一个环境编辑或运行本项目之前：

1. 确认 checkout 路径和分支。
2. 确认预期的后端端口。
3. 确认运行中后端实际使用的 `DB_PATH`。
4. 确认生成素材会落在预期的 `data/static` 边界内。
5. 确认 `configs/config.yaml` 只是本地文件，还是已经被代码实际读取。
6. 调用 AI 服务商前，先检查设置页面中的 AI 和智能体配置。
7. 修改服务商、智能体或 ComfyUI 相关设置前，先备份 SQLite 文件。
8. 真实 ComfyUI 工作流在清理成 example 前，不要提交到 git。

## 安全的本机开发示例

尝试不应影响默认本机数据库的改动时，显式指定数据库路径：

```bash
cd backend
DB_PATH=../data/scratch/huobao_drama.db PORT=5679 npm run dev
```

生成素材方面，在静态服务路径和 `STORAGE_PATH` 还没有一起做成完整可配置之前，优先使用独立 checkout 或独立容器 volume。
