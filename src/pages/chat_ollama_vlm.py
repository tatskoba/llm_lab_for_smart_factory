import os
from PIL import Image
import streamlit as st
from streamlit_chat import message
from components.ollama_vlm_chat import ollama_vlm_chat
from components.prompt_library import list_prompt_files, get_prompt_library, set_prompt
# ツールの設定ページのURL
TOOL_SETTING_URL = os.environ['TOOL_SETTING_URL']

# メッセージを表示する
def display_messages():
    if "ollama_vlm_messages" not in st.session_state:
        st.session_state["ollama_vlm_messages"] = []
    
    for i, (msg, is_user) in enumerate(st.session_state["ollama_vlm_messages"]):
        message(msg, is_user=is_user, key=str(i))
    st.session_state["thinking_spinner"] = st.empty()

# 画像ファイルを読み込んで保存する
def image_read_and_save_file():
    # VLMクラスのインスタンスを作成
    st.session_state["ollama_vlm_assistant"] = ollama_vlm_chat()
    st.session_state["user_input"] = ""

# 画像についての質問を処理する
def image_process_input():
    if st.session_state["user_input"] and len(st.session_state["user_input"].strip()) > 0:
        user_text = st.session_state["user_input"].strip()

        # ユーザー質問に情報を追加する（必要に応じて）
        ask_text = user_text + "。日本語で回答してください。"

        with st.session_state["thinking_spinner"], st.spinner(f"解析中..."):
            # LLMに質問を投げる
            agent_text = st.session_state["ollama_vlm_assistant"].analyze_image_by_api(ask_text)
            
        st.session_state["ollama_vlm_messages"].append((user_text, True))
        st.session_state["ollama_vlm_messages"].append((agent_text, False))

        # 最後に入力をクリアする
        st.session_state["user_input"] = ""

# 対話履歴をクリアする
def clear_messages():
    st.session_state["ollama_vlm_messages"] = []


def chat_ollama_vlm():

    st.markdown("#### 画像について質問する")

    with st.expander("条件設定", expanded=True):

        cols = st.columns([45,15,20,20])
        with cols[0]:
                # 画像ファイルをアップロードする
                uploaded_image_file = st.file_uploader(
                    "Upload image file",
                    type=["jpg", "jpeg", "png"],
                    key="uploaded_image_file",
                    on_change=image_read_and_save_file,
                    label_visibility="collapsed",
                    accept_multiple_files=False,
                )
                model_name = st.session_state.ollama_vlm_model_name
        with cols[1]:
            pass
        with cols[2]:
            st.write("　")
            st.write("　")
            st.button("対話履歴をクリア", on_click=clear_messages)
        with cols[3]: 
            o_vlm_dialog_window_size_list = [200, 300, 400, 500, 600]
            st.selectbox(
                '対話ウィンドウの長さ',
                tuple(o_vlm_dialog_window_size_list),         
                key="o_vlm_dialog_window_size"
            )
            if "o_vlm_dialog_window_size" not in st.session_state:
                st.session_state["o_vlm_dialog_window_size"] = 400

    st.session_state["ingestion_spinner"] = st.empty()

    st.markdown("##### 対話履歴")
    
    cols = st.columns([45,55])
    with cols[0]:
        if "uploaded_image_file" in st.session_state and st.session_state.uploaded_image_file is not None:
            # 画像を取得して表示する
            st.image(Image.open(st.session_state["uploaded_image_file"]))
        else:
            st.write("画像ファイルをアップロードしてください")
    with cols[1].container(height=st.session_state.o_vlm_dialog_window_size ):
        # メッセージを表示する
        display_messages() 

    # ユーザーの入力を処理する（OSS LLMとの対話）
    if st.session_state.uploaded_image_file is not None:
        cols = st.columns([50,50])
        with cols[0]:
            # 入力欄を表示
            st.text_area(
                "VLM（" + model_name + "）への質問", 
                key="user_input", 
                height=110,
                max_chars=200,
            )
            st.button("質問する", on_click=image_process_input)

        with cols[1]:

            # プロンプトライブラリを取得してユーザ入力欄に設定する
            files = list_prompt_files()
            if len(files) > 0:
                selected_prompt_library_file = st.selectbox("プロンプトライブラリ", files, index=None)
            if selected_prompt_library_file != "":
                prompt_list = get_prompt_library(selected_prompt_library_file)
                if len(prompt_list) > 0:
                    st.selectbox("プロンプトライブラリ", prompt_list, index=None, key="selected_prompt", on_change=set_prompt)

#------------------------------------
# ollamaモデルとの質問メイン処理

# モデルが利用できる場合は質問を受け付ける
if "ollama_availability" in st.session_state and st.session_state.ollama_availability == True:
    chat_ollama_vlm()
else:
    st.error("アプリケーション立上げ時は「[ツールの設定](" + TOOL_SETTING_URL + ")」へアクセスしてください。")
    

