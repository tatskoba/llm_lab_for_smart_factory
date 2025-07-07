import os
import streamlit as st

# プロンプトライブラリファイルのパス
prompt_library_path = "./data/prompt_library/"

# プロンプトライブラリのファイル一覧を取得して返す
def list_prompt_files():
    files = os.listdir(prompt_library_path)
    return files

# プロンプトライブラリの選択をしてpronpt_libraryにセットする
def get_prompt_library(filename):
    if filename == "" or filename is None:
        return []
    prompt_library = []
    with open(prompt_library_path + filename, "r") as f:
        prompt_library = f.readlines()
    return prompt_library

# プロンプトの選択を入力欄にセットする
def set_prompt():
    st.session_state["user_input"] = st.session_state["selected_prompt"]

