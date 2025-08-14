import os
from PIL import Image
import streamlit as st
from streamlit_chat import message
from components.ollama_vlm_chat import ollama_vlm_chat
# ツールの設定ページのURL
TOOL_SETTING_URL = os.environ['TOOL_SETTING_URL']

# メッセージを表示する
def display_messages():
    if "ollama_ocr_messages" not in st.session_state:
        st.session_state["ollama_ocr_messages"] = []
    
    for i, (msg, is_user) in enumerate(st.session_state["ollama_ocr_messages"]):
        message(msg, is_user=is_user, key=str(i))
    st.session_state["thinking_spinner"] = st.empty()

# 画像ファイルを読み込んで保存する
def image_read_and_save_file():
    # VLMクラスのインスタンスを作成
    st.session_state["ollama_ocr_assistant"] = ollama_vlm_chat()
    st.session_state["user_input"] = ""

# 画像についての質問を処理する
def image_process_input():
    if st.session_state["user_input"] and len(st.session_state["user_input"].strip()) > 0:
        user_text = st.session_state["user_input"].strip()

        with st.session_state["thinking_spinner"], st.spinner(f"解析中..."):
            # LLMに質問を投げる
            agent_text = st.session_state["ollama_ocr_assistant"].analyze_image_by_api(user_text, "ocr")

        st.session_state["ollama_ocr_messages"].append((user_text, True))
        st.session_state["ollama_ocr_messages"].append((agent_text, False))

        # 最後に入力をクリアする
        st.session_state["user_input"] = ""

# 対話履歴をクリアする
def clear_messages():
    st.session_state["ollama_ocr_messages"] = []

# OCRプロンプトを設定する
def set_ocr_prompt():
    prompt_simple = """画像のテキストをすべて抽出してください。"""

    prompt_text = """ 画像内のすべてのテキストを、日本語で見たまま正確に抽出してください。  
- 要約、言い換え、補完、または欠落したテキストの推測を絶対に行わないでください。
- 出力は以下のフォーマットを正確に再現してください:
- ヘッダー: 画像内に存在する場合のみ使用してください。
- 箇条書き、番号付きリスト: 画像内の形式をそのまま保持してください。
- スペース、句読点、改行: 画像内の配置をそのまま維持してください。
- テキストが不明瞭または一部しか見えない場合でも、推測せずに、見える範囲で可能な限り正確に抽出してください。
- 無関係または繰り返しのように見えるテキストであっても、すべて含めてください。
- テキストの順序を変更せず、画像内で見える順序をそのまま維持してください。
        """
    prompt_markdown = """ 画像内のすべてのテキストを、日本語で見たまま正確に抽出してください。  
- 要約、言い換え、補完、または欠落したテキストの推測を絶対に行わないでください。
- 出力はMarkdown形式でフォーマットしてください。
- ヘッダー: 画像内に存在する場合のみ使用してください。
- 箇条書き、番号付きリスト: 画像内の形式をそのまま保持してください。
- スペース、句読点、改行: 画像内の配置をそのまま維持してください。
- テキストが不明瞭または一部しか見えない場合でも、推測せずに、見える範囲で可能な限り正確に抽出してください。
- 無関係または繰り返しのように見えるテキストであっても、すべて含めてください。
- テキストの順序を変更せず、画像内で見える順序をそのまま維持してください。
        """    
    if "o_ocr_output_format" in st.session_state:
        if st.session_state["o_ocr_output_format"] == "テキスト（シンプル）":
            st.session_state["user_input"] = prompt_simple
        elif st.session_state["o_ocr_output_format"] == "テキスト":
            st.session_state["user_input"] = prompt_text
        elif st.session_state["o_ocr_output_format"] == "Markdown":
            st.session_state["user_input"] = prompt_markdown


def ollama_ocr():

    st.markdown("#### 画像からテキストを読み取る (OCR)")

    with st.expander("条件設定", expanded=True):

        cols = st.columns([45,20,15,20])
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
                model_name = st.session_state.ollama_ocr_model_name
        with cols[1]:
            pass
        with cols[2]:
            st.write("　")
            st.write("　")
            st.button("出力結果をクリア", on_click=clear_messages)
        with cols[3]: 
            o_ocr_dialog_window_size_list = [200, 300, 400, 500, 600]
            st.selectbox(
                '出力結果ウィンドウの長さ',
                tuple(o_ocr_dialog_window_size_list),
                key="o_ocr_dialog_window_size"
            )
            if "o_ocr_dialog_window_size" not in st.session_state:
                st.session_state["o_ocr_dialog_window_size"] = 400

    st.session_state["ingestion_spinner"] = st.empty()

    st.markdown("##### OCR実行")
    
    cols = st.columns([45,55])
    with cols[0]:
        if "uploaded_image_file" in st.session_state and st.session_state.uploaded_image_file is not None:
            # 画像を取得して表示する
            st.image(Image.open(st.session_state["uploaded_image_file"]))
        else:
            st.write("画像ファイルをアップロードしてください")
    with cols[1].container(height=st.session_state.o_ocr_dialog_window_size):
        # メッセージを表示する
        display_messages() 

    # ユーザーの入力を処理する（OSS LLMとの対話）
    if st.session_state.uploaded_image_file is not None:
        cols = st.columns([50,50])
        with cols[0]:
            # 入力欄を表示
            st.text_area(
                "VLM（" + model_name + "）を使う", 
                key="user_input", 
                height=150,
                max_chars=1000,
            )
            st.button("OCRを実行", on_click=image_process_input)

        with cols[1]:

            # 出力フォーマットを選択し、OCR用プロンプトを設定する
            o_ocr_output_format = ["テキスト（シンプル）", "テキスト", "Markdown"]
            st.selectbox(
                '出力フォーマット',
                tuple(o_ocr_output_format),
                index = None,
                key="o_ocr_output_format",
                on_change=set_ocr_prompt
            )
            if "o_ocr_output_format" not in st.session_state:
                st.session_state["o_ocr_output_format"] = "テキスト（シンプル）"

#------------------------------------
# ollamaモデルとの質問メイン処理

# モデルが利用できる場合は質問を受け付ける
if "ollama_availability" in st.session_state and st.session_state.ollama_availability == True:
    ollama_ocr()
else:
    st.error("アプリケーション立上げ時は「[ツールの設定](" + TOOL_SETTING_URL + ")」へアクセスしてください。")
    

