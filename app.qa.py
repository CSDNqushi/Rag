import streamlit as st
import time
from rag import RagService
import config_data as config

st.title("智能客服")
st.divider()



if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": "你是一个问答机器人，请根据提供的上下文回答问题。"},
        {"role": "assistant", "content": "你好，有什么可以帮到你。"}
    ]
    st.session_state["rag_service"]=RagService()

for message in st.session_state.messages:
    st.chat_message(message["role"]).write(message["content"])

prompt = st.chat_input("请输入问题：")

if prompt:

    st.chat_message("user").write(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.spinner("思考中..."):
        res_stream=st.session_state["rag_service"].chain.stream({"input":prompt}, config=config.session_config)
        assistant_text=st.chat_message("assistant").write_stream(res_stream)or ""
        st.session_state.messages.append({"role": "assistant", "content": assistant_text})
