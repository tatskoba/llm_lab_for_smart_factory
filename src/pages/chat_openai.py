#   Chat with ChatGPT
#   このサンプルコードは、LangChainの古いバージョンを使用しています。

import time
import os
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferWindowMemory
from langchain_openai import AzureChatOpenAI
import openai

# ツールの設定ページのURL
TOOL_SETTING_URL = os.environ['TOOL_SETTING_URL']

# Azure OpenAI ChatGPT チェーンのロード
def load_azure_openai_chain():

    api_version = os.getenv("AZURE_OPENAI_API_VERSION")
    deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")

    llm = AzureChatOpenAI(
        api_version=api_version, 
        azure_deployment=deployment_name
    )

    # メモリを初期化
    if 'azure_buffer_memory' not in st.session_state:
        st.session_state.azure_buffer_memory = ConversationBufferWindowMemory( 
            k=st.session_state.azure_buffer_memory_k, 
            return_messages=True
        )

    # チェーンを初期化
    if 'azure_openai_chain' not in st.session_state:
        st.session_state.azure_openai_chain = ConversationChain(
            memory=st.session_state.azure_buffer_memory,
            llm=llm
        )

# Azure OpenAI ChatGPTのチャット画面
def chat_azure_openai():

    # チャット履歴を初期化
    if "azure_openai_messages" not in st.session_state:
        st.session_state["azure_openai_messages"] = [
            {"role": "system", "content": "あなたは、人々が情報を見つけるのを助け、韻を踏んで応答する AI アシスタントです。 ユーザーが答えがわからない質問をした場合は、その旨を伝えます。"},
            {"role": "assistant", "content": "なんでも質問してください"}]

    # チャット履歴クリアボタン
    cols = st.columns([40,60])
    with cols[0]:
        pass
    with cols[1]:
        if st.button("チャット履歴クリア"):
            st.session_state.azure_openai_messages = []
            st.session_state.azure_buffer_memory.clear()

    # アプリの再実行でチャット履歴にメッセージを表示
    for message in st.session_state.azure_openai_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # チェーンのロード（OpenAI社の
    load_azure_openai_chain()

    # ユーザの質問を取得
    if user_input := st.chat_input("質問を入力してください。"):
        # チャット履歴にユーザのメッセージを追加
        st.session_state.azure_openai_messages.append({"role": "user", "content": user_input})

        # ユーザのメッセージを表示
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""

            # ポップアップメッセージを表示
            with st.spinner('考えています ...'):
                
                # チェーンのメモリを更新
                st.session_state.azure_openai_chain.memory = st.session_state.azure_buffer_memory

                # チェーンを呼び出して応答を取得
                assistant_response = st.session_state.azure_openai_chain.invoke(st.session_state["azure_openai_messages"])

            # 数秒遅れさせてメッセージを表示（句点で区切って出力）
            for chunk in assistant_response["response"].split("。"):
                if len(full_response) > 0:
                    full_response += "。"
                full_response += chunk
                time.sleep(0.5)

                # ブリンクするカーソルを追加表示（タイプしている感を出すため）
                message_placeholder.markdown(full_response + "▌")

            message_placeholder.markdown(full_response)
        st.session_state.azure_openai_messages.append({"role": "assistant", "content": full_response})

# OpenAI ChatGPT チェーンのロード
def load_openai_chain():

    llm = ChatOpenAI(
        model=st.session_state.openai_model_name,             # 「ツールの設定」で設定したモデル名を使用
        #openai_api_key=st.session_state["OPENAI_API_KEY"],   # 環境変数ではなくセッション変数を使う場合はこちらを有効にする
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
    )

    # メモリを初期化
    if 'buffer_memory' not in st.session_state:
        st.session_state.buffer_memory = ConversationBufferWindowMemory( 
            k=st.session_state.openai_memory_buffer_k, 
            return_messages=True
        )

    # チェーンを初期化
    if 'openai_chain' not in st.session_state:
        st.session_state.openai_chain = ConversationChain(
            llm=llm,
            memory=st.session_state.buffer_memory,
            verbose=True
        )

# OpenAI ChatGPTのチャット画面
def chat_openai():

    # チャット履歴を初期化
    if "openai_messages" not in st.session_state:
        st.session_state["openai_messages"] = [
            {"role": "system", "content": "あなたは、人々が情報を見つけるのを助け、韻を踏んで応答する AI アシスタントです。 ユーザーが答えがわからない質問をした場合は、その旨を伝えます。"},
            {"role": "assistant", "content": "なんでも質問してください"}]

    # チャット履歴クリアボタン
    cols = st.columns([40,60])
    with cols[0]:
        pass
    with cols[1]:
        if st.button("チャット履歴クリア"):
            st.session_state.openai_messages = []
            st.session_state.buffer_memory.clear()

    # アプリの再実行でチャット履歴にメッセージを表示
    for message in st.session_state.openai_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # チェーンのロード（OpenAI社の
    load_openai_chain()

    # ユーザの質問を取得
    if user_input := st.chat_input("質問を入力してください。"):
        # チャット履歴にユーザのメッセージを追加
        st.session_state.openai_messages.append({"role": "user", "content": user_input})

        # ユーザのメッセージを表示
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""

            # ポップアップメッセージを表示
            with st.spinner('考えています ...'):
                
                # チェーンのメモリを更新
                st.session_state.openai_chain.memory = st.session_state.buffer_memory

                # チェーンを呼び出して応答を取得
                assistant_response = st.session_state.openai_chain.invoke(input=user_input)

            # 数秒遅れさせてメッセージを表示（句点で区切って出力）
            for chunk in assistant_response["response"].split("。"):
                if len(full_response) > 0:
                    full_response += "。"
                full_response += chunk
                time.sleep(0.5)

                # ブリンクするカーソルを追加表示（タイプしている感を出すため）
                message_placeholder.markdown(full_response + "▌")

            message_placeholder.markdown(full_response)
        st.session_state.openai_messages.append({"role": "assistant", "content": full_response})
    
#-----------------------------------
# Chat with ChatGPTのメイン処理

if st.session_state.get("openai_api_key_configured"):

    if st.session_state.selected_chatgpt_service == "OpenAI ChatGPT":
        st.markdown("#### ChatGPTへの質問（OpenAI ChatGPT）")
        chat_openai()
    elif st.session_state.selected_chatgpt_service == "Azure OpenAI":
        st.markdown("#### ChatGPTへの質問（Azure OpenAI ChatGPT）")
        chat_azure_openai()

else:
    st.error("アプリケーション立上げ時は「[ツールの設定](" + TOOL_SETTING_URL + ")」でAPI Keyを設定してください。")    