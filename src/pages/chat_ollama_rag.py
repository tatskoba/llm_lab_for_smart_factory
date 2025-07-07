import os
import streamlit as st
from streamlit_chat import message
from components.rag import RAG
from components.prompt_library import list_prompt_files, get_prompt_library, set_prompt

# ツールの設定ページのURL
TOOL_SETTING_URL = os.environ.get('TOOL_SETTING_URL')
MODEL_FOLDER = os.environ.get('MODEL_FOLDER')
VECTOR_DB_FOLDER = os.environ.get('VECTOR_DB_FOLDER')

# RAGを準備する
def prepare_rag(file_type):

    # ベクトルDBのフォルダが存在しなければリターンする
    st.session_state.rag_ready = 0
    if file_type == "CSV":
        persist_directory = VECTOR_DB_FOLDER + "vector_db_csv"
    elif file_type == "Excel":    
        persist_directory = VECTOR_DB_FOLDER + "vector_db_excel"
    elif file_type == "PDF":
        persist_directory = VECTOR_DB_FOLDER + "vector_db_pdf"
    if os.path.exists(persist_directory) == False:
         st.session_state.rag_ready = -1
         return 

    # RAGクラスのインスタンスを作成
    st.session_state["assistant"] = RAG()
    st.session_state["user_input"] = ""

    with st.session_state["ingestion_spinner"], st.spinner(f"ベクトルDBを読み込み中..."):         
        
        # ベクトルDBをロードする
        st.session_state["assistant"].load_vector_db(file_type)

        # プロンプトテンプレートを設定する
        st.session_state["assistant"].set_prompt_template()

        # Chainを作る
        st.session_state["assistant"].make_chain() 

    st.session_state.rag_ready = 1

    st.session_state.vector_db_file_type = file_type

    # RAGに使用しているLLMモデル名を取得
    st.session_state.ollama_rag_llm = st.session_state.ollama_llm_model_name    

# メッセージを表示する
def display_messages():
    if "ollama_llm_messages" not in st.session_state:
        st.session_state["ollama_llm_messages"] = []
    
    for i, (msg, is_user) in enumerate(st.session_state["ollama_llm_messages"]):
        message(msg, is_user=is_user, key=str(i))
    st.session_state["thinking_spinner"] = st.empty()


# システムログを表示する
def display_system_log():
    # システムログが存在しない場合は初期化する   
    if "ollama_llm_system_log" not in st.session_state:
        st.session_state["ollama_llm_system_log"] = []

    # システムログを表示する
    for log in st.session_state["ollama_llm_system_log"]:
        txt = log.split("\n")
        for i in range(len(txt)):
            s = f"<p style='font-size:12px;'>{txt[i]}</p>"
            st.markdown(s, unsafe_allow_html=True)             


# 質問を処理する
def process_input():
    if st.session_state["user_input"] and len(st.session_state["user_input"].strip()) > 0:
        user_text = st.session_state["user_input"].strip()

        # ユーザー質問に情報を追加する（必要に応じて）
        ask_text = user_text + "。回答は日本語で作成してください。"

        with st.session_state["thinking_spinner"], st.spinner(f"解析中..."):
            # ask(user_text)でLLMに質問を投げる
            agent_text, chunk_info = st.session_state["assistant"].ask(ask_text)

        # 回答テキストがあれば、メッセージに追加する
        st.session_state["ollama_llm_messages"].append((user_text, True))
        st.session_state["ollama_llm_messages"].append((agent_text, False))

        # チャンク情報があれば、システムログに追加する
        if chunk_info:
            st.session_state["ollama_llm_system_log"].append("★★★★★★★★★★★★★★★★★★★★★★★★")
            st.session_state["ollama_llm_system_log"].append("■ 質問：" + user_text)
            st.session_state["ollama_llm_system_log"].append(chunk_info)

        # 最後に入力をクリアする
        st.session_state["user_input"] = ""

# 対話履歴をクリアする
def clear_messages():
    st.session_state["ollama_llm_messages"] = []
    st.session_state["ollama_llm_system_log"] = []

def chat_ollama_rag():

    st.markdown("#### ファイルについて質問する")
    
    with st.expander("条件設定", expanded=True):

        cols = st.columns([20,80])
        with cols[0]:
            st.write("LLMモデル：" + st.session_state.ollama_llm_model_name)
        with cols[1]:
            st.write("Embedding APIの設定：" + st.session_state["embeddings_api"])
        
        cols = st.columns([25,33,22,20])
        with cols[0]:
            # ファイル種類を選択する
            vector_db_file_type_list = ("CSV", "Excel", "PDF")
            if "vector_db_file_type" not in st.session_state:
                st.session_state.vector_db_file_type = "CSV"
                index = 0
            else:
                index = vector_db_file_type_list.index(st.session_state.vector_db_file_type) if st.session_state.vector_db_file_type in vector_db_file_type_list else 0
            selected_file_type = st.selectbox(
                "ベクトルDBを選択する", 
                options=vector_db_file_type_list,  
                index = index
            )
            if "vector_db_file_type" in st.session_state and "rag_ready" in st.session_state and st.session_state.rag_ready == 1:            
                st.write(st.session_state.vector_db_file_type + "が設定されています")
        with cols[1]:
            st.write("　")
            st.write("　")
            if st.session_state.embeddings_api == "OpenAI Embedding API" and st.session_state.get("OPENAI_API_KEY") is None:
                st.error("OpenAI Embedding APIを選択していますので、OpenAIのAPI keyを設定してください")
            elif st.session_state.embeddings_api == "intfloat/multilingual-e5-large" \
                and os.path.exists(MODEL_FOLDER + "multilingual-e5-large") == False:
                st.error("選択されたEmbeddingモデルはまだローカルにダウンロードされていません。")
            else:
                st.button("設定条件でRAGをセットアップ", on_click=prepare_rag, args=(selected_file_type,))
                st.markdown("ツールの設定の変更、および、ベクトルDBを変更した場合は再ロードしてください")
        with cols[2]:
            st.write("　")
            st.write("　")
            st.button("対話履歴をクリア", on_click=clear_messages)
        with cols[3]: 
            o_llm_dialog_window_size_list = [200, 300, 400, 500, 600]
            st.selectbox(
                '対話ウィンドウの長さ',
                tuple(o_llm_dialog_window_size_list),
                key="o_llm_dialog_window_size"
            )
            if "o_llm_dialog_window_size" not in st.session_state:
                st.session_state.o_llm_dialog_window_size = 400

    st.session_state["ingestion_spinner"] = st.empty()

    # ユーザーの入力を処理する（OSS LLMとの対話）
    if "rag_ready" in st.session_state and st.session_state.rag_ready == 1:

        st.markdown("##### LLMとの対話")

        cols = st.columns([40,60])
        with cols[0]:
            st.text_area(
                "LLM（" + st.session_state.ollama_llm_model_name + "）への質問", 
                key="user_input", 
                height=110,
                max_chars=1000,
            )
            subcols = st.columns([70,30])
            with subcols[0]:
                pass

            with subcols[1]:
                st.button("質問する", on_click=process_input)

            # プロンプトライブラリを取得してユーザ入力欄に設定する
            files = list_prompt_files()
            if len(files) > 0:
                selected_prompt_library_file = st.selectbox("プロンプトライブラリ", files, index=None)
            if selected_prompt_library_file != "":
                prompt_list = get_prompt_library(selected_prompt_library_file)
                if len(prompt_list) > 0:
                    st.selectbox("プロンプト一覧", prompt_list, index=None, key="selected_prompt", on_change=set_prompt)

        with cols[1]:
            st.markdown("対話履歴")
            with st.container(height=st.session_state.o_llm_dialog_window_size): 
                # メッセージを表示する
                display_messages()

            st.markdown("システムメッセージ")
            with st.container(height=400):            
                # システムログを表示する
                display_system_log()

    else:
        if st.session_state.rag_ready == -1:
            st.error("ベクトルDBが存在しません。データ準備ページでベクトルDBを作成してください")
        elif st.session_state.rag_ready == -2:
            st.error("ツールの設定で選択したLLMモデルを反映させるため、ベクトルDBを設定してください")
        else:
            st.error("ベクトルDBを設定してください")

#------------------------------------
# ollamaモデルとの質問メイン処理

# モデルが利用できる場合は質問を受け付ける
if "ollama_availability" in st.session_state and st.session_state.ollama_availability == True:
    if "rag_ready" not in st.session_state:
        st.session_state.rag_ready = 0

    if "ollama_rag_llm" in st.session_state and \
        st.session_state.ollama_rag_llm != st.session_state.ollama_llm_model_name:
        st.session_state.rag_ready = -2    # LLMが変更になったため、RAGインスタンスの再作成が必要

    if  st.session_state.chunk_overlap >= st.session_state.rag_chunk_size:
        st.error("RAGのText splitter設定でチャンクサイズが重複サイズ以上に設定されています。ツールの設定で変更してください")
    else:
        # RAGページを表示する
        chat_ollama_rag()
else:
    st.error("アプリケーション立上げ時は「[ツールの設定](" + TOOL_SETTING_URL + ")」へアクセスしてください。")

