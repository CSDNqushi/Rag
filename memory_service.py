import json
import os
import re
import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterable, Optional

import config_data as config

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from langchain_chroma import Chroma
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_community.embeddings import DashScopeEmbeddings


def _utc_now_iso() -> str:
    """
    生成 UTC 时间的 ISO 字符串。

    作用：
    - 统一 created_at/updated_at 的时间格式，便于多端/多进程消费
    - 避免本地时区带来的歧义
    """
    return datetime.now(timezone.utc).isoformat()


def _normalize_key_part(value: str) -> str:
    """
    将 name/relation 这类“主键字段”做一个轻量规范化，减少重复。

    规范化策略（可按你们业务再调整）：
    - 去除首尾空白
    - 折叠连续空白为单个空格
    - 统一小写（对中文基本无影响，对英文可减少重复）
    """
    if value is None:
        return ""
    value = str(value).strip()
    # 把 value 里“连续的空白字符”都替换成“一个普通空格
    value = re.sub(r"\s+", " ", value)
    return value.lower()


def _stable_memory_id(session_id: str, name: str, relation: str) -> str:
    """
    为“同一会话内，同一 name+relation”的记忆生成稳定的 ID。

    为什么要做成稳定 ID：
    - Chroma 向量库里以 id 作为主键最稳妥
    - 我们可以通过 delete + add_texts(ids=...) 达到“更新”的效果
    - 方便后续做“查看/回放/删除某条记忆”
    """
    raw = f"{session_id}||{_normalize_key_part(name)}||{_normalize_key_part(relation)}"
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    return f"mem_{digest}"


def _extract_json_candidate(text: str) -> Optional[str]:
    """
    从模型输出中尽量“捞出”一个 JSON（数组或对象）的字符串片段。

    背景：
    - 大模型即使被要求“只输出 JSON”，也偶尔会夹带解释、markdown code fence 等
    - 生产上解析需要具备一定容错性，避免一次异常就中断主业务流程
    """
    if not text:
        return None

    # 常见场景：```json ... ``` 这种 code fence
    fenced = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, flags=re.IGNORECASE)
    if fenced:
        return fenced.group(1).strip()

    # 再尝试：优先截取第一个 JSON 数组
    start = text.find("[")
    end = text.rfind("]")
    if 0 <= start < end:
        return text[start : end + 1].strip()

    # 再尝试：截取第一个 JSON 对象
    start = text.find("{")
    end = text.rfind("}")
    if 0 <= start < end:
        return text[start : end + 1].strip()

    return None


def _safe_load_memories(model_output_text: str) -> list[dict[str, str]]:
    """
    将模型输出解析为记忆列表（list[dict]），并做字段规整。

    期望输出格式（推荐）：
    [
      {"name": "...", "relation": "...", "observation": "..."},
      ...
    ]

    兼容输出格式（容错）：
    {"memories":[...]} / {"items":[...]} 等。
    """
    candidate = _extract_json_candidate(model_output_text) or model_output_text
    try:
        data = json.loads(candidate)
    except Exception:
        return []

    if isinstance(data, dict):
        for key in ("memories", "items", "data", "result"):
            if key in data and isinstance(data[key], list):
                data = data[key]
                break

    if not isinstance(data, list):
        return []

    cleaned: list[dict[str, str]] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "")).strip()
        relation = str(item.get("relation", "")).strip()
        observation = str(item.get("observation", "")).strip()
        if not name or not relation or not observation:
            continue
        cleaned.append({"name": name, "relation": relation, "observation": observation})

    return cleaned


@dataclass
class MemoryUpsertResult:
    """
    单条记忆写入/更新后的结果。

    作用：
    - 便于调用方统计：新增了几条、更新了几条、跳过了几条
    - 便于后续扩展：比如把 result 回写到日志/指标/审计
    """

    memory_id: str
    action: str  # "created" | "updated" | "skipped"


class PromptMemoryExtractor:
    """
    基于提示词工程的“记忆抽取器”。

    职责：
    - 输入：用户本轮发言（可附带近期对话历史）
    - 输出：结构化记忆列表（name / relation / observation）

    注意：
    - 该类只做“抽取”，不做“去重/存储”
    - 这样可以独立替换：规则抽取、模型微调、函数调用等
    """

    def __init__(
        self,
        model: Optional[Any] = None,
        max_history_messages: int = 20,
        max_items_per_turn: int = 20,
    ):
        self.model = model or ChatTongyi(model=config.chat_model_name)
        self.max_history_messages = max_history_messages
        self.max_items_per_turn = max_items_per_turn

        # 关键点：强约束输出“纯 JSON”，并定义字段含义与粒度。
        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "你是一个信息抽取系统。你的任务是从用户发言中抽取“可长期记忆”的事实。\n"
                    "只输出 JSON，不要输出任何解释、markdown、代码块。\n"
                    "输出必须是 JSON 数组，数组元素是对象，且只包含 3 个字段：\n"
                    '- name: 实体名称（人名/地点/组织/物品/宠物等，尽量用用户原话）\n'
                    '- relation: 关系/属性（例如：居住地/工作单位/家人/喜好/生日/联系方式/拥有/厌恶等）\n'
                    "- observation: 具体事实描述（完整句子，必要时补充上下文，但不要编造）\n"
                    "抽取规则：\n"
                    "1) 只抽取明确陈述的事实，绝不猜测。\n"
                    "2) 优先抽取对后续对话有用的稳定信息（身份、偏好、习惯、关系、地点、时间等）。\n"
                    "3) 如果没有可记忆信息，输出空数组 []。\n"
                    f"4) 最多输出 {max_items_per_turn} 条。\n",
                ),
                (
                    "system",
                    "我会提供最近的对话历史（可能为空），用于消歧和补全指代，但不要把历史里未被用户确认的猜测写成事实：",
                ),
                MessagesPlaceholder(variable_name="history"),
                ("human", "用户本轮发言：{user_text}"),
            ]
        )

        # 组成一个可直接 invoke 的链条：prompt -> model -> string
        self.chain = self.prompt | self.model | StrOutputParser()

    def extract(self, user_text: str, history: list[BaseMessage]) -> list[dict[str, str]]:
        """
        从用户发言抽取结构化记忆。

        输入：
        - user_text: 用户本轮发言文本
        - history: 最近若干条对话消息（BaseMessage），用于指代消解（例如“我老婆”“我家”）

        输出：
        - list[{"name":..., "relation":..., "observation":...}]
        """
        recent_history = history[-self.max_history_messages :] if history else []
        output_text = self.chain.invoke({"user_text": user_text, "history": recent_history})
        memories = _safe_load_memories(output_text)
        return memories[: self.max_items_per_turn]


class ChromaMemoryStore:
    """
    记忆存储：写入 Chroma 向量数据库（RAG 可检索）。

    存储模型：
    - 每条记忆作为一条向量文档
    - 以稳定 memory_id 作为 Chroma 的 ids 主键，便于 upsert
    - metadata 保留结构化字段 + 审计字段（created_at/updated_at/source 等）
    """

    def __init__(
        self,
        embedding: Optional[Any] = None,
        collection_name: Optional[str] = None,#Chroma 的 collection 名（可以理解成“表名/集合名”）。
        persist_directory: Optional[str] = None,#Chroma 本地持久化目录
    ):
        self.embedding = embedding or DashScopeEmbeddings(model=config.embedding_model_name)
        self.collection_name = collection_name or getattr(config, "memory_collection_name", "memory")
        self.persist_directory = persist_directory or getattr(config, "memory_persist_directory", config.persist_directory)

        self.vector_store = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embedding,
            persist_directory=self.persist_directory,
        )

    def _get_existing(self, memory_id: str) -> Optional[dict[str, Any]]:
        """
        从 Chroma 中读取一条既有记忆。

        返回值说明：
        - None：不存在
        - dict：包含 "metadatas"/"documents"/"ids" 等
        """
        try:
            existing = self.vector_store.get(ids=[memory_id])
        except Exception:
            return None
        # - 第一次判断：防 existing 整体就是空/None
        # - 第二次判断：防 existing 有壳但内容为空（尤其是 ids=[] ）
        if not existing:
            return None
        ids = existing.get("ids") or []
        if not ids:
            return None
        return existing

    def upsert_memory(
        self,
        session_id: str,
        name: str,
        relation: str,
        observation: str,
        *,
        source: str,
        extracted_from: Optional[dict[str, Any]] = None,
    ) -> MemoryUpsertResult:
        """
        对单条记忆做“去重 + 更新 + 写入”。

        去重主键：
        - session_id + name + relation（规范化后）

        更新策略（可按业务调整）：
        - 若已存在：尝试合并 observation（避免丢失历史事实），并刷新 updated_at
        - 若不存在：写入 created_at/updated_at
        """
        memory_id = _stable_memory_id(session_id, name, relation)
        now = _utc_now_iso()

        existing = self._get_existing(memory_id)

        created_at = now
        old_observation = ""
        old_metadata: dict[str, Any] = {}
        if existing:
            action = "updated"
            metadatas = existing.get("metadatas") or []
            old_metadata = metadatas[0] if metadatas else {}
            created_at = old_metadata.get("created_at", now)

            documents = existing.get("documents") or []
            if documents:
                old_observation = str(documents[0] or "")
        else:
            action = "created"

        # 合并 observation：尽量避免重复，同时保留新信息
        merged_observation = observation.strip()
        if old_observation:
            old_obs = old_observation.strip()
            # 如果“新内容”是“旧内容”的子串（旧的已经包含新信息），那就用旧的即可：
            if merged_observation and merged_observation in old_obs:
                merged_observation = old_obs
            # 如果“旧内容”是“新内容”的子串（新内容更完整），
            # 这里 pass 表示什么都不做，保持 merged_observation 还是“新内容”（用更完整的）
            elif old_obs and old_obs in merged_observation:
                pass
            # 新旧都非空、互不包含、且不完全相等，说明两者是“不同信息”，则合并：
            elif merged_observation and old_obs and merged_observation != old_obs:
                merged_observation = f"{old_obs}\n{merged_observation}"

        metadata: dict[str, Any] = {
            "session_id": session_id,
            "name": name,
            "relation": relation,
            "source": source,
            "created_at": created_at,
            "updated_at": now,
        }
        if extracted_from:
            # 注意：metadata 里只放必要信息，避免存入隐私/敏感大字段。
            # 这样未来你在检索/排查时能知道：这条记忆来自用户消息（而不是文件上传、系统规则、人工标注等）。
            metadata["extracted_from_type"] = extracted_from.get("type")

        # Chroma（LangChain 封装）没有显式 upsert：最稳妥方式是 delete + add_texts(ids=...)
        # - 若不存在，delete 不影响
        # - 若存在，delete 后再 add，实现“更新”
        try:
            self.vector_store.delete(ids=[memory_id])
        except Exception:
            # delete 失败不应阻断主业务，继续尝试 add
            pass

        # 文本内容建议做“可检索友好”的表达：把结构字段组合成自然语言
        # 例：张三｜居住地：上海
        memory_text = f"{name}｜{relation}：{merged_observation}"
        # 调用 self.vector_store.add_texts(...) 写入向量库
        self.vector_store.add_texts(
            texts=[memory_text],
            # 元数据（结构字段 + created_at/updated_at/source 等），用于过滤/展示/审计
            metadatas=[metadata],
            # 这条记忆的主键 ID
            # （我们前面用 session_id+name+relation 算出来的稳定 ID），用于“同一条记忆更新时覆盖同一个 id”
            ids=[memory_id],
        )

        return MemoryUpsertResult(memory_id=memory_id, action=action)


class JsonGraphMemoryStore:
    """
    记忆存储：本地 JSON“知识图谱”（轻量版）。

    为什么用 JSON 文件模拟 KG：
    - 你们代码库目前没有 Neo4j/NetworkX 等依赖
    - 先把“数据结构 + 去重更新 + 元数据”跑通，后续可无缝替换为真正的图数据库

    存储结构（每个 session 一个文件）：
    {
      "session_id": "...",
      "updated_at": "...",
      "nodes": { "<name>": {"name": "...", "created_at": "...", "updated_at": "..."} },
      "edges": {
         "<edge_id>": {"src": "<name>", "relation": "...", "observation": "...", "created_at": "...", "updated_at": "..."}
      }
    }
    """

    def __init__(self, base_path: Optional[str] = None):
        self.base_path = base_path or getattr(config, "memory_graph_path", "./memory_graph")
        os.makedirs(self.base_path, exist_ok=True)

    def _graph_path(self, session_id: str) -> str:
        return os.path.join(self.base_path, f"{session_id}.json")

    def _load_graph(self, session_id: str) -> dict[str, Any]:
        path = self._graph_path(session_id)
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            return {"session_id": session_id, "updated_at": _utc_now_iso(), "nodes": {}, "edges": {}}

    def _save_graph(self, session_id: str, graph: dict[str, Any]) -> None:
        path = self._graph_path(session_id)
        with open(path, "w", encoding="utf-8") as f:
            #：indent=2，输出带缩进的“漂亮格式”（pretty print），每层缩进 2 个空格，
            json.dump(graph, f, ensure_ascii=False, indent=2)

    def upsert_memory(
        self,
        session_id: str,
        name: str,
        relation: str,
        observation: str,
        *,
        source: str,
    ) -> MemoryUpsertResult:
        memory_id = _stable_memory_id(session_id, name, relation)
        now = _utc_now_iso()

        graph = self._load_graph(session_id)
        nodes = graph.setdefault("nodes", {})
        edges = graph.setdefault("edges", {})

        # upsert node
        node = nodes.get(name)
        if not node:
            nodes[name] = {"name": name, "created_at": now, "updated_at": now}
        else:
            node["updated_at"] = now

        # upsert edge（以 memory_id 作为边的唯一键）
        edge = edges.get(memory_id)
        if not edge:
            action = "created"
            edges[memory_id] = {
                "src": name,
                "relation": relation,
                "observation": observation,
                "source": source,
                "created_at": now,
                "updated_at": now,
            }
        else:
            action = "updated"
            # 简单合并：同 Chroma 的策略，避免重复
            old_obs = str(edge.get("observation", "")).strip()
            new_obs = str(observation).strip()
            if new_obs and new_obs in old_obs:
                edge["observation"] = old_obs
            elif old_obs and old_obs in new_obs:
                edge["observation"] = new_obs
            elif old_obs and new_obs and old_obs != new_obs:
                edge["observation"] = f"{old_obs}\n{new_obs}"
            edge["updated_at"] = now

        graph["updated_at"] = now
        self._save_graph(session_id, graph)

        return MemoryUpsertResult(memory_id=memory_id, action=action)


class MemoryService:
    """
    记忆系统总入口（编排层）。

    执行逻辑（从“用户发言”到“写入记忆库”）：
    1) 过滤出用户发言（HumanMessage）
    2) 使用 PromptMemoryExtractor 抽取结构化记忆：name/relation/observation
    3) 对每条记忆做去重与更新（主键：session_id+name+relation）
    4) 将记忆写入：
       - 向量库（Chroma，用于 RAG 检索）
       - 或/和 图谱（本地 JSON，便于后续替换为图数据库）

    设计目的：
    - 将“抽取”“存储”“编排”分层，降低耦合，方便后续迭代
    """

    def __init__(
        self,
        extractor: Optional[PromptMemoryExtractor] = None,
        chroma_store: Optional[ChromaMemoryStore] = None,
        graph_store: Optional[JsonGraphMemoryStore] = None,
    ):
        self.extractor = extractor or PromptMemoryExtractor(
            max_history_messages=getattr(config, "memory_max_history_messages", 20),
            max_items_per_turn=getattr(config, "memory_max_items_per_turn", 20),
        )
        self.chroma_store = chroma_store or ChromaMemoryStore()
        self.graph_store = graph_store or JsonGraphMemoryStore()

    def _enabled_backend(self) -> str:
        """
        读取配置决定写入后端。

        支持值：
        - "chroma"：只写向量库
        - "graph"：只写图谱 JSON
        - "both"：两边都写
        - "none"：不写（仅抽取/或完全关闭）
        """
        return getattr(config, "memory_backend", "chroma")

    def process_new_messages(
        self,
        *,#这是“强制关键字参数”语法： * 后面的参数调用时必须写成 session_id=... 这种形式，避免位置传参搞错顺序
        session_id: str,
        new_messages: Iterable[BaseMessage],
        all_messages: list[BaseMessage],
        source: str = "chat_history",
    ) -> list[MemoryUpsertResult]:
        """
        处理“新增的消息”并将记忆写入存储。

        为什么输入需要同时有 new_messages 和 all_messages：
        - new_messages：告诉我们本次到底新增了哪些（避免重复处理）
        - all_messages：提供上下文给抽取器，用于指代消解（例如“她/他/我们家”）
        """
        if not getattr(config, "memory_enabled", True):#配置里关掉记忆功能就直接返回空结果，不做任何抽取/写入。
            return []

        results: list[MemoryUpsertResult] = []#收集每条写入/更新后的结果
        backend = self._enabled_backend()#读取 config.memory_backend

        # 仅从“用户发言”里抽取记忆（AI 回复通常不应当进入长期记忆）
        human_texts: list[str] = []
        for msg in new_messages:
            if isinstance(msg, HumanMessage):
                human_texts.append(str(getattr(msg, "content", "") or ""))

        if not human_texts:
            return []

        # 一次性合并本轮用户发言，减少模型调用次数
        user_text = "\n".join([t for t in human_texts if t.strip()])
        if not user_text.strip():
            return []

        memories = self.extractor.extract(user_text=user_text, history=all_messages)
        #extract() 会用提示词让模型输出 JSON，并解析成：
        # [{ "name": "...", "relation": "...", "observation": "..." }, ...]

        #逐条写入：去重 + 更新 + 存储
        for mem in memories:
            name = mem["name"]
            relation = mem["relation"]
            observation = mem["observation"]

            if backend in ("chroma", "both"):
                res = self.chroma_store.upsert_memory(
                    session_id=session_id,
                    name=name,
                    relation=relation,
                    observation=observation,
                    source=source,
                    extracted_from={"type": "human_message"},
                )
                results.append(res)

            if backend in ("graph", "both"):
                res = self.graph_store.upsert_memory(
                    session_id=session_id,
                    name=name,
                    relation=relation,
                    observation=observation,
                    source=source,
                )
                results.append(res)

        return results


_MEMORY_SERVICE_SINGLETON: Optional[MemoryService] = None


def get_memory_service() -> MemoryService:
    """
    获取 MemoryService 单例（懒加载）。

    作用：
    - 避免在 import 阶段就初始化模型/向量库（可能引发网络/IO）
    - 多次调用 add_messages 时复用同一实例，减少开销
    """
    global _MEMORY_SERVICE_SINGLETON
    if _MEMORY_SERVICE_SINGLETON is None:
        _MEMORY_SERVICE_SINGLETON = MemoryService()
    return _MEMORY_SERVICE_SINGLETON
