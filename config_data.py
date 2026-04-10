md5_path="./md5.text"

#Chroma
collection_name="rag"
persist_directory="./chroma_db"

# =========================
# Memory（长期记忆）相关配置
# =========================
# 是否开启“从用户发言抽取记忆并写入存储”的能力
memory_enabled=True

# 记忆写入后端：
# - "chroma"：写入向量库（RAG 可检索）
# - "graph"：写入本地 JSON 知识图谱（轻量版）
# - "both"：两边都写
# - "none"：不写入（仅抽取/或完全关闭）
memory_backend="chroma"

# Chroma 记忆库（与 rag collection 分开，避免混淆）
memory_collection_name="memory"
memory_persist_directory=persist_directory

# 图谱 JSON 的落盘目录（每个 session 一个文件）
memory_graph_path="./memory_graph"

# 抽取时最多带入多少条历史消息用于消歧（太大可能增加成本/时延）
memory_max_history_messages=20

# 单轮最多抽取多少条记忆（防止模型输出过长）
memory_max_items_per_turn=20

#spliter
chunk_size=500
chunk_overlap=20
separator=["\n","\n\n",".","!","?",""," "]
max_split_char_number=1000

#
similarity_threshold=1

#
embedding_model_name="text-embedding-v4"
chat_model_name="qwen3-max"

#session id配置
session_config={
    "configurable":{
            "session_id":"user001"
    }
}
