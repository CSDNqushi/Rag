from langchain_chroma import Chroma
import config_data as config



class VectorStoreService(object):
    def __init__(self,embedding):
        self.embedding=embedding

        self.vector_store=Chroma(
            collection_name=config.collection_name,#数据库表名
            embedding_function=self.embedding,
            persist_directory=config.persist_directory #数据库本地存储文件夹
        )

    def get_retriever(self):
        return self.vector_store.as_retriever(search_kwargs={"k":config.similarity_threshold})

if __name__=="__main__":
    from langchain_community.embeddings import DashScopeEmbeddings
    vector_store=VectorStoreService(embedding=DashScopeEmbeddings(model="text-embedding-v4"))
    retriever=vector_store.get_retriever()
    res=retriever.invoke("体重200斤，尺码推荐")
    print(res)