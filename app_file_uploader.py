import streamlit as st
from knowledge_base import KnowledgeBaseService
import time

st.title("知识库更新服务")

uploader_file=st.file_uploader(
    "请上传文件",
    type=["txt","pdf","docx"],
    accept_multiple_files=False,
)


if "service" not in st.session_state:
    st.session_state["service"]=KnowledgeBaseService()

if uploader_file is not None:
    file_name=uploader_file.name
    file_type=uploader_file.type
    file_size=uploader_file.size/1024

    st.subheader(f"文件信息:{file_name}")
    st.write(f"文件类型:{file_type}|大小:{file_size:.2f}KB")

    txt=uploader_file.getvalue().decode("utf-8")
    # st.write(txt)
    with st.spinner("上传中..."):
        time.sleep(1)
        result=st.session_state["service"].upload_by_str(txt,file_name)
        st.write(result)