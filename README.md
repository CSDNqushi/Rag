# rag_project_example

一个最小可运行的本地 RAG 示例项目，包含：
- 基于 Chroma 的向量检索问答（RAG）
- 基于文件的对话历史存储（`chat_history/`）
- 从用户发言中抽取“长期记忆”（name / relation / observation），并写入向量库（可选写入本地 JSON 图谱）
- Streamlit 前端页面用于问答与记忆系统验证

## 项目说明

### 主要能力
- RAG 问答：从本地 Chroma 向量库检索相关文档片段，拼接上下文后让模型回答。
- 对话历史：按 `session_id` 落盘到本地文件（JSON）。
- 长期记忆：
  - 从新增用户发言（HumanMessage）抽取结构化记忆（`name/relation/observation`）
  - 去重与更新：同一会话内按 `session_id + name + relation` 做主键 upsert
  - 存储：默认写入 Chroma 独立 collection（`memory`），可选写入 `memory_graph/<session_id>.json`
- 记忆验证台：支持对话输入、记忆查询、记忆管理，并返回结构化结果（回答 + 引用记忆）。

### 代码入口
- RAG 问答页面：[app.qa.py]
- 记忆系统验证页面：[app.memory.py](基于app.qa改进，使用时请运行这个py文件)
- RAG 链路：[rag.py]
- 对话历史存储：[file_history_store.py]
- 记忆抽取与存储：[memory_service.py]

## 技术选型

### 前端
- Streamlit：快速构建交互式页面（对话输入、查询、管理）。

### LLM / Embeddings
- Tongyi（Qwen）：对话模型（`ChatTongyi`）
- DashScope Embeddings：向量化模型（`DashScopeEmbeddings`）

### RAG 向量数据库
- Chroma（本地持久化）：文档向量库 + 记忆向量库（独立 collection）

### 编排框架
- LangChain（core/community/chroma）：Prompt、Runnable、消息历史、向量库封装。

## 安装步骤（Windows / PowerShell）

### 1) 创建虚拟环境
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -V
```

### 2) 安装依赖
本仓库未内置 `requirements.txt`，你可以按实际需要安装（建议在虚拟环境中执行）：
```powershell
pip install -U streamlit langchain langchain-core langchain-community langchain-chroma chromadb dashscope
```

### 3) 配置环境变量（重要）
出于安全原因，README **不会**写入真实的 API Key。请在你自己的本机环境中设置。

```powershell
$env:DASHSCOPE_API_KEY="YOUR_DASHSCOPE_API_KEY"#请注意是通义气模型的 API Key

```


如果你希望把配置写入本地文件，请使用 `.env`（不要提交到仓库）：
```env
DASHSCOPE_API_KEY=YOUR_DASHSCOPE_API_KEY
```

### 4) 配置项目参数
参数集中在 [config_data.py]，包括：
- RAG collection、落盘目录
- 记忆后端选择（`memory_backend=chroma|graph|both|none`）
- 默认会话 `session_id`

## 运行方式

### 运行 RAG 问答页面
```powershell
.\.venv\Scripts\streamlit.exe run .\app.qa.py
```

### 运行记忆系统验证页面
```powershell
.\.venv\Scripts\streamlit.exe run .\app.memory.py
```

## 记忆系统验证要点
- 在“对话输入”里发一段包含个人信息/偏好的句子（例如：姓名、居住地、喜好等）
- 页面会：
  1) 触发记忆抽取并写入（默认写入 Chroma 的 `memory` collection）
  2) 以本轮输入做 query 检索 top-k 记忆作为引用
  3) 返回结构化结果：`answer + citations`
- 在“记忆管理”里可查看、删除、清空、覆盖编辑记忆

## 目录结构（运行后会自动生成的文件夹）
- `chroma_db/`：Chroma 本地数据库（RAG 默认落盘目录）
- `chat_history/`：对话历史文件（按 `session_id` 分文件）
- `memory_graph/`：本地 JSON 图谱（当 `memory_backend` 含 `graph` 时生成）
- `_tmp_memory_chroma/`、`_tmp_memory_graph/`：自测脚本产生的临时目录

## 安全提示
- 不要把任何真实密钥写进 README、代码或提交到仓库。
- 建议使用环境变量或 `.env` 文件，并把 `.env` 加入忽略列表（本项目已包含 `.gitignore` 规则）。

## 附：在 README 中展示 Markdown 代码块的示例
下面是一个“在 README 里展示代码块语法本身”的示例（使用 ```markdown）：

```markdown
```powershell
$env:DASHSCOPE_API_KEY="YOUR_DASHSCOPE_API_KEY"
```
```

