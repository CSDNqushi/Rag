# 知识库
import os
import config_data as config
import hashlib
from langchain_chroma import Chroma
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from datetime import datetime

def check_md5(md5_str:str):
    """
    检查md5字符串是否存在
    """
    if not os.path.exists(config.md5_path):
        open(config.md5_path, 'w',encoding="utf-8").close()
        return False
    else:
        for line in open(config.md5_path, 'r',encoding="utf-8").readlines():
            line=line.strip()
            if line==md5_str:
                return True

    return False

def save_md5(md5_str:str):
    open(config.md5_path, 'a',encoding="utf-8").write(md5_str+"\n")
    pass

def get_string_md5(input_str:str,encoding="utf-8"):
    return hashlib.md5(input_str.encode(encoding)).hexdigest()
    

class KnowledgeBaseService(object):

    def __init__(self):
        #文件夹不存在则创建
        os.makedirs(config.persist_directory, exist_ok=True)
        self.chroma=Chroma(
            collection_name=config.collection_name,#数据库表名
            embedding_function=DashScopeEmbeddings(model="text-embedding-v4"),
            persist_directory=config.persist_directory #数据库本地存储文件夹
        )
        self.spliter=RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,#文本段分割最大长度
            chunk_overlap=config.chunk_overlap,#文本段分割重叠长度
            separators=config.separator, #文本段分割分隔符
            length_function=len #文本段分割长度函数
        )
    
    def upload_by_str(self,data,file_name):
        """
        上传字符串
        :param data: 字符串
        :param file_name: 文件名
        :return: None
        """
        md5_max=get_string_md5(data)
        # 检查文件是否存在
        if check_md5(md5_max):
            print("文件已存在")
            return "文件已存在"
        # 检查字符串长度是否超过最大分割字符数
        if len(data)>config.max_split_char_number:
            knowledge_chunk:list[str]=self.spliter.split_text(data)
        else:
            knowledge_chunk=[data]
        
        metadata={
            "source": file_name,
            "create_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "operator": "ljn"
        }
        self.chroma.add_texts(
            knowledge_chunk,
            metadatas=[metadata for _ in knowledge_chunk]
        )
        
        save_md5(md5_max)
        return "上传成功"

if __name__ == "__main__":
    kb = KnowledgeBaseService()
    r=kb.upload_by_str("何意味1", "testfile")
    print(r)
