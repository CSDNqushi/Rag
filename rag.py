
from vector_stores import VectorStoreService
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder 
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.runnables import RunnablePassthrough,RunnableWithMessageHistory,RunnableLambda
from langchain_core.documents import Document
import config_data as config
from langchain_core.output_parsers import StrOutputParser
from file_history_store import get_history

def print_prompt(prompt):
    print("prompt:")
    print(prompt)
    print("="*20)
    return prompt

class RagService(object):

    def __init__(self):
        self.vector_service=VectorStoreService(
            DashScopeEmbeddings(model=config.embedding_model_name)
            )
        self.prompt_template=ChatPromptTemplate.from_messages(
            [
                ("system","你是一个专业的问答机器人"
                "请根据提供的上下文回答问题。参考资料{context}。"),
                ("system","并且我提供用户对话历史记录，如下："),
                MessagesPlaceholder(variable_name="history"),
                ("human","请回答用户提问{input}"),
            ]
            )
        self.chat_model=ChatTongyi(model=config.chat_model_name)
        self.chain=self.__get_chain()
    def __get_chain(self):
        retriever=self.vector_service.get_retriever()

        def format_documents(docs:list[Document]):
            if not docs:
                return "无相关参考资料"
            formatted_str=""
            for doc in docs:
                formatted_str += f"文档片段：{doc.page_content}\n文档元数据：{doc.metadata}\n\n"

            return formatted_str
        def format_for_retriever(value:dict)->str:
            return value["input"]
        def format_for_prompt_template(value):
            new_value={}
            new_value["context"]=value["context"]
            new_value["input"]=value["input"]["input"]
            new_value["history"]=value["input"]["history"]
            return new_value
        chain=(
            {
                "input":RunnablePassthrough(),
                "context":RunnableLambda(format_for_retriever)|retriever|RunnableLambda(lambda x: print(type(x), x) or x)|format_documents,
            }
            |RunnableLambda(lambda x: print(type(x), x) or x)
            |RunnableLambda(format_for_prompt_template)
            |self.prompt_template
            # |print_prompt
            |self.chat_model
            |StrOutputParser()
        )

        conversation_chain=RunnableWithMessageHistory(
                chain,
                get_history,
                input_message_key="input",
                history_messages_key="history",
        )
        
        return conversation_chain


if __name__ == "__main__":

    result=RagService().chain.invoke({"input":"给出纯棉材质（春季衬衫、T恤、休闲裤）的洗涤养护"}, config=session_config)
    # print(result)
    print(getattr(result, "content", result))
