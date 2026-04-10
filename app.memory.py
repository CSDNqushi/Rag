import streamlit as st

import config_data as config
from memory_service import ChromaMemoryStore, get_memory_service

from datetime import datetime, timezone

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_models.tongyi import ChatTongyi


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _default_session_id() -> str:
    return (
        (config.session_config or {})
        .get("configurable", {})
        .get("session_id", "user001")
    )


def _get_store() -> ChromaMemoryStore:
    if "memory_store" not in st.session_state:
        st.session_state["memory_store"] = ChromaMemoryStore()
    return st.session_state["memory_store"]


def _search_memories(store: ChromaMemoryStore, session_id: str, query: str, k: int) -> list[dict]:
    results = store.vector_store.similarity_search_with_score(
        query=query,
        k=k,
        filter={"session_id": session_id},
    )
    items: list[dict] = []
    for doc, score in results:
        items.append(
            {
                "id": getattr(doc, "id", None),
                "text": doc.page_content,
                "metadata": doc.metadata,
                "score": score,
            }
        )
    return items


def _answer_with_memories(question: str, memories: list[dict]) -> str:
    model = ChatTongyi(model=config.chat_model_name)
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "你是一个对话助手。请优先依据“可用记忆”回答用户问题，并在信息不足时明确说明。",
            ),
            ("human", "用户问题：{question}\n\n可用记忆：\n{memories_text}"),
        ]
    )

    memories_text = "无"
    if memories:
        lines = []
        for m in memories:
            meta = m.get("metadata") or {}
            lines.append(
                f"- id={m.get('id')} | name={meta.get('name')} | relation={meta.get('relation')} | text={m.get('text')}"
            )
        memories_text = "\n".join(lines)

    return (prompt | model | StrOutputParser()).invoke(
        {"question": question, "memories_text": memories_text}
    )


st.title("记忆系统验证台")
st.divider()

if "session_id" not in st.session_state:
    st.session_state["session_id"] = _default_session_id()

if "chat_messages" not in st.session_state:
    st.session_state["chat_messages"] = []

if "lc_messages" not in st.session_state:
    st.session_state["lc_messages"] = []

store = _get_store()
memory_service = get_memory_service()

with st.sidebar:
    st.subheader("会话")
    st.session_state["session_id"] = st.text_input(
        "session_id",
        value=st.session_state["session_id"],
    )
    top_k = st.number_input("引用记忆条数 k", min_value=1, max_value=50, value=5, step=1)
    st.write(f"记忆库 collection：{store.collection_name}")
    st.write(f"落盘目录：{store.persist_directory}")

tab_chat, tab_query, tab_manage = st.tabs(["对话输入", "记忆查询", "记忆管理"])

with tab_chat:
    st.subheader("对话输入")

    for m in st.session_state["chat_messages"]:
        st.chat_message(m["role"]).write(m["content"])

    user_input = st.chat_input("输入内容（将触发记忆抽取 + 可选引用记忆回答）")
    if user_input:
        session_id = st.session_state["session_id"]

        st.chat_message("user").write(user_input)
        st.session_state["chat_messages"].append({"role": "user", "content": user_input})
        st.session_state["lc_messages"].append(HumanMessage(content=user_input))

        try:
            memory_service.process_new_messages(
                session_id=session_id,
                new_messages=[HumanMessage(content=user_input)],
                all_messages=list(st.session_state["lc_messages"]),
                source="app.memory.chat",
            )
        except Exception as e:
            st.error(f"记忆抽取/写入失败：{e}")

        citations: list[dict] = []
        try:
            citations = _search_memories(
                store, session_id=session_id, query=user_input, k=int(top_k)
            )
        except Exception as e:
            st.error(f"记忆检索失败：{e}")

        try:
            answer = _answer_with_memories(user_input, citations)
        except Exception as e:
            answer = f"回答生成失败：{e}"

        st.chat_message("assistant").write(answer)
        st.session_state["chat_messages"].append({"role": "assistant", "content": answer})
        st.session_state["lc_messages"].append(AIMessage(content=answer))

        st.subheader("结构化结果")
        st.json({"answer": answer, "citations": citations})

with tab_query:
    st.subheader("记忆查询")

    session_id = st.session_state["session_id"]
    query = st.text_input("查询文本", value="")
    k = st.number_input("k", min_value=1, max_value=50, value=10, step=1)

    if st.button("查询", type="primary") and query.strip():
        try:
            items = _search_memories(store, session_id=session_id, query=query, k=int(k))
            st.json({"query": query, "session_id": session_id, "results": items})
        except Exception as e:
            st.error(f"查询失败：{e}")

with tab_manage:
    st.subheader("记忆管理")
    session_id = st.session_state["session_id"]

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("刷新列表", type="primary"):
            st.session_state["manage_refresh"] = True
    with col_b:
        if st.button("清空本会话记忆", type="secondary"):
            try:
                store.vector_store.delete(where={"session_id": session_id})
                st.session_state["manage_refresh"] = True
            except Exception as e:
                st.error(f"清空失败：{e}")

    if st.session_state.get("manage_refresh") or True:
        try:
            data = store.vector_store.get(where={"session_id": session_id}, limit=500)
        except Exception as e:
            st.error(f"读取列表失败：{e}")
            data = {"ids": [], "documents": [], "metadatas": []}
        ids = data.get("ids") or []
        docs = data.get("documents") or []
        metas = data.get("metadatas") or []

        memories = []
        for i in range(min(len(ids), len(docs), len(metas))):
            memories.append({"id": ids[i], "text": docs[i], "metadata": metas[i]})

        st.write(f"当前会话记忆数：{len(memories)}")
        st.json(memories)

        if memories:
            st.divider()
            st.subheader("删除/编辑")
            selected_id = st.selectbox("选择记忆 id", options=[m["id"] for m in memories])
            selected = next((m for m in memories if m["id"] == selected_id), None) or {}

            action_col1, action_col2 = st.columns(2)
            with action_col1:
                if st.button("删除选中记忆"):
                    store.vector_store.delete(ids=[selected_id])
                    st.session_state["manage_refresh"] = True
                    st.rerun()

            with action_col2:
                st.write("")

            meta = selected.get("metadata") or {}
            name = st.text_input("name", value=str(meta.get("name", "")))
            relation = st.text_input("relation", value=str(meta.get("relation", "")))
            old_text = str(selected.get("text", "") or "")
            old_obs = old_text
            if "：" in old_text:
                old_obs = old_text.split("：", 1)[1]
            elif ":" in old_text:
                old_obs = old_text.split(":", 1)[1]
            observation = st.text_area("observation（将覆盖写入）", value=old_obs.strip())

            if st.button("保存修改（覆盖写入）"):
                created_at = meta.get("created_at")
                new_meta = dict(meta)
                if created_at is not None:
                    new_meta["created_at"] = created_at
                new_meta["updated_at"] = _utc_now_iso()
                new_meta["name"] = name
                new_meta["relation"] = relation
                new_meta["session_id"] = session_id
                new_meta["source"] = "app.memory.manage"

                try:
                    memory_text = f"{name}｜{relation}：{observation.strip()}"
                    store.vector_store.delete(ids=[selected_id])
                    store.vector_store.add_texts(
                        texts=[memory_text],
                        metadatas=[new_meta],
                        ids=[selected_id],
                    )
                    st.session_state["manage_refresh"] = True
                    st.rerun()
                except Exception as e:
                    st.error(f"保存失败：{e}")
