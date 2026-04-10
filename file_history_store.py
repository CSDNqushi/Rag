import json
import os
from typing import Sequence
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, HumanMessage, message_to_dict, messages_from_dict

# 说明：
# - file_history_store.py 本身负责“对话历史落盘”
# - 新需求要求：在记录用户发言时，额外抽取结构化记忆并写入记忆库（向量库/图谱）
# - 为了尽量不侵入现有 RAG 链路，我们把触发点放在 add_messages()：
#   只要消息被写入历史文件，就会尝试从“新增的人类消息”里抽取记忆
from memory_service import get_memory_service


def get_history(session_id):
    return FileChatMessageHistory(session_id, "./chat_history")


class FileChatMessageHistory(BaseChatMessageHistory):
    def __init__(self, session_id, storage_path):
        self.session_id = session_id        # 会话id
        self.storage_path = storage_path    # 不同会话id的存储文件，所在的文件夹路径
        # 完整的文件路径
        self.file_path = os.path.join(self.storage_path, self.session_id)

        # 确保文件夹是存在的
        os.makedirs(os.path.dirname(self.file_path), exist_ok=True)

    def add_messages(self, messages: Sequence[BaseMessage]) ->None:
        """
        将本次新增消息写入本地历史文件。

        额外能力（新需求）：
        - 从新增消息中识别“用户发言”（HumanMessage）
        - 用提示词工程抽取结构化记忆（name / relation / observation）
        - 对记忆去重与更新（同一 session 内，以 name+relation 作为主键）
        - 将记忆写入向量数据库（Chroma）或本地 Graph JSON（由 config 决定）

        设计原则：
        - 记忆抽取/写入属于“副作用”，任何异常都不应影响主链路（历史落盘 + RAG 对话）
        - 因此这里对记忆逻辑做 try/except 包裹
        """

        # Sequence 序列：类似 list、tuple
        # 1) 读取已有的消息列表（从文件反序列化回来）
        all_messages = list(self.messages)

        # 2) 合并新增消息
        all_messages.extend(messages)

        # 将数据同步写入到本地文件中
        # 类对象写入文件 -&gt; 一堆二进制
        # 为了方便，可以将BaseMessage消息转为字典（借助json模块以json字符串写入文件）
        # 官方message_to_dict：单个消息对象（BaseMessage类实例） -&gt; 字典
        # new_messages = []
        # for message in all_messages:
        #     d = message_to_dict(message)
        #     new_messages.append(d)

        new_messages = [message_to_dict(message) for message in all_messages]
        # 将数据写入文件
        with open(self.file_path, "w", encoding="utf-8") as f:
            json.dump(new_messages, f)

        # 3) 触发记忆抽取与写入（仅处理“本次新增的用户消息”）
        # 注意：messages 里可能包含 AIMessage（模型回复），也可能一次包含多条
        # 我们只从 HumanMessage 抽取长期记忆，避免把模型输出当事实写入记忆库。
        try:
            # 快速过滤：若本次新增没有 HumanMessage，则不做任何记忆处理
            has_human = any(isinstance(m, HumanMessage) for m in messages)
            if not has_human:
                return

            memory_service = get_memory_service()
            memory_service.process_new_messages(
                session_id=self.session_id,
                new_messages=messages,
                all_messages=all_messages,
                source="file_history_store.add_messages",
            )
        except Exception:
            # 记忆模块异常不应影响主业务链路，这里选择静默吞掉
            # 如果你们需要排障，可在此处接入 logging，并注意不要打印敏感信息（手机号/地址等）。
            return

    @property       # @property装饰器将messages方法变成成员属性用
    def messages(self) -> list[BaseMessage]:
        # 当前文件内： list[字典]
        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                messages_data = json.load(f)    # 返回值就是：list[字典]
                return messages_from_dict(messages_data)
        except FileNotFoundError:
            return []

    def clear(self) -> None:
        with open(self.file_path, "w", encoding="utf-8") as f:
            json.dump([], f)
