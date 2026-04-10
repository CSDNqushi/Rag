"""
一个“本地自测脚本”，用于验证记忆去重与更新逻辑是否可运行。

重要说明：
1) 该脚本不会调用任何外部 LLM（避免你本地没有配置 API KEY 时失败）
2) 该脚本使用 langchain_core 的 FakeEmbeddings + 本地 Chroma 持久化目录
3) 该脚本会在项目目录下创建临时文件夹：
   - ./_tmp_memory_chroma
   - ./_tmp_memory_graph
"""

import os
from typing import Any

from langchain_core.embeddings.fake import FakeEmbeddings

from memory_service import (
    ChromaMemoryStore,
    JsonGraphMemoryStore,
    MemoryService,
    PromptMemoryExtractor,
)


class FakeExtractor(PromptMemoryExtractor):
    """
    伪造的抽取器：用固定规则返回记忆，模拟“模型输出的结构化记忆”。

    作用：
    - 在不依赖外部模型的情况下，验证：
      - session_id+name+relation 的去重是否生效
      - 更新时 observation 的合并逻辑是否符合预期
      - Chroma/Graph 两种后端都能写入
    """

    def __init__(self):
        pass

    def extract(self, user_text: str, history: list[Any]) -> list[dict[str, str]]:
        if "浦东" in user_text:
            return [{"name": "张三", "relation": "居住地", "observation": "上海浦东"}]
        return [{"name": "张三", "relation": "居住地", "observation": "上海"}]


def main() -> None:
    # 临时目录（避免污染你正式的 rag/memory collection）
    chroma_dir = "./_tmp_memory_chroma"
    graph_dir = "./_tmp_memory_graph"
    os.makedirs(chroma_dir, exist_ok=True)
    os.makedirs(graph_dir, exist_ok=True)

    chroma_store = ChromaMemoryStore(
        embedding=FakeEmbeddings(size=8),
        collection_name="memory_selftest",
        persist_directory=chroma_dir,
    )
    graph_store = JsonGraphMemoryStore(base_path=graph_dir)

    service = MemoryService(
        extractor=FakeExtractor(),
        chroma_store=chroma_store,
        graph_store=graph_store,
    )

    session_id = "selftest_session"

    # 第一次写入：应为 created
    res1 = service.process_new_messages(
        session_id=session_id,
        new_messages=[],
        all_messages=[],
        source="selftest",
    )
    assert res1 == [], "没有新消息时，应不产生写入结果"

    # 模拟“新增的用户发言”
    from langchain_core.messages import HumanMessage

    res2 = service.process_new_messages(
        session_id=session_id,
        new_messages=[HumanMessage(content="我叫张三，我住在上海。")],
        all_messages=[HumanMessage(content="我叫张三，我住在上海。")],
        source="selftest",
    )
    assert res2, "第一次写入应产生结果"

    # 第二次写入（同一 name+relation）：应为 updated，并合并 observation
    res3 = service.process_new_messages(
        session_id=session_id,
        new_messages=[HumanMessage(content="补充：我住在上海浦东。")],
        all_messages=[
            HumanMessage(content="我叫张三，我住在上海。"),
            HumanMessage(content="补充：我住在上海浦东。"),
        ],
        source="selftest",
    )
    assert res3, "第二次写入应产生结果"

    # 验证 Chroma 中的文档是否包含“上海”和“上海浦东”（合并策略：旧+新）
    # 由于我们以稳定 memory_id 做主键，这里直接 get 全量（只会有 1 条）
    stored = chroma_store.vector_store.get()
    assert stored.get("ids"), "Chroma 中应存在至少一条记忆"
    doc = (stored.get("documents") or [""])[0]
    assert "张三" in doc and "居住地" in doc, "文档文本应包含 name/relation"
    assert "上海" in doc, "应包含旧 observation"
    assert "浦东" in doc, "应包含新 observation"

    print("selftest OK")


if __name__ == "__main__":
    main()

