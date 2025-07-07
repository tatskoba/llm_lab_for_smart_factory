import os
import numpy as np
import pandas as pd
import shutil
from torch import embedding
import streamlit as st
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import CSVLoader
from langchain_community.document_loaders import UnstructuredExcelLoader
from langchain.vectorstores.utils import filter_complex_metadata
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
#from langchain.vectorstores.faiss import FAISS
import chromadb

# プロセッサー指定
PROCESSOR = os.environ.get('PROCESSOR')

# フォルダの設定
DATA_FOLDER = os.environ.get('DATA_FOLDER')
VECTOR_DB_FOLDER = os.environ.get('VECTOR_DB_FOLDER')
MODEL_FOLDER = os.environ.get('MODEL_FOLDER')

# ツールの設定ページのURL
TOOL_SETTING_URL = os.environ['TOOL_SETTING_URL']

# ベクトルDBを作成する
def make_vector_db(file_type, file_list):

    # ファイルリストが空の場合は何もしない
    if len(file_list) == 0:
        return

    with st.spinner("ベクトルDBを作成中..."):

        # Chromadbのキャッシュをクリアする
        chromadb.api.client.SharedSystemClient.clear_system_cache()

        # 選択されたファイルのみベクトル化する
        documents = []
        for i, row in file_list.iterrows():    
            if row["ベクトルDBに追加"]:
                file_name = os.path.join(DATA_FOLDER, file_type, row["ファイル名"])
                if file_type == "pdf":
                    loader = PyPDFLoader(file_name)
                elif file_type == "csv":                  
                    loader = CSVLoader(file_name,
                                        csv_args={
                                            'delimiter': ',',
                                            'quotechar': '"',
                                            #"fieldnames": ["column1", "column2", "column3"]   # カラム指定ができる
                                        })
                elif file_type == "excel":
                    loader = UnstructuredExcelLoader(file_name, mode="elements")

                documents.extend(loader.load())

        # テキストをチャンクに分割
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=st.session_state.rag_chunk_size, 
            chunk_overlap=st.session_state.chunk_overlap
        )    
        chunked_documents = text_splitter.split_documents(documents)

        # pdfは複雑なメタデータをフィルタリング
        if file_type == "pdf" or file_type == "excel":
            chunked_documents = filter_complex_metadata(chunked_documents)

        # コレクション名とベクトルDBの保存先を設定
        collection_name = file_type + "_collection"
        persist_directory = os.path.join(VECTOR_DB_FOLDER, "vector_db_" + file_type)

        # ベクトルDBの保存先フォルダを削除（Embedding APIによって次元数の互換性がないため）
        if os.path.isdir(persist_directory):
            shutil.rmtree(persist_directory)

        # ベクトルDBのクライアントを作成
        client = chromadb.PersistentClient(path=persist_directory)

        # 選択されたembedding APIを選択
        if st.session_state.embeddings_api == "intfloat/multilingual-e5-large":
            # ローカルに保存したモデルを指定
            embeddings = HuggingFaceEmbeddings(
                model_name=MODEL_FOLDER + "multilingual-e5-large",
                model_kwargs = {'device': PROCESSOR},   # GPU or CPU
                #encode_kwargs = {'normalize_embeddings': True}            
            )
        elif st.session_state.embeddings_api == "OpenAI Embedding API":
            # OpenAI Embedding APIを指定
            embeddings = OpenAIEmbeddings()

        # ベクトルDBの作成
        vector_db = Chroma(
                    collection_name=collection_name,
                    embedding_function=embeddings,
                    client=client
        )
        vector_db.add_documents(
                    documents=chunked_documents, 
                    embedding=embeddings 
        )


# ベクトルDBの確認
def confirm_vector_db(vectordb_file_type, disp_type):

    # コレクション名とベクトルDBの保存先を設定
    collection_name = vectordb_file_type + "_collection"
    persist_directory = os.path.join(VECTOR_DB_FOLDER, "vector_db_" + vectordb_file_type)

    # ベクトルDBのクライアントを作成
    client = chromadb.PersistentClient(path=persist_directory)
    collection = client.get_collection(name=collection_name)

    # チャンク数を取得して表示
    st.write("チャンク数：" + str(collection.count()))
    
    # 画面に表示
    if disp_type == "json":
        vector_db_in_json = collection.get()  # すべて取得
        st.json(vector_db_in_json)
    elif disp_type == "シンプルテーブル":
        vector_db_in_json = collection.get(include=["documents", "metadatas"])  # documentsとmetadatasのみ取得        
        docs = pd.DataFrame(vector_db_in_json["documents"])
        matadatas = pd.json_normalize(vector_db_in_json["metadatas"])

        if vectordb_file_type == "pdf":
            matadatas["page"] += 1   # ページ番号を1から始まるようにする
            vector_db_table = pd.concat([docs, matadatas[["page", "source"]]], axis=1)   # ドキュメントとメタデータを結合         
            vector_db_table.columns = ['チャンクテキスト', 'ページ', 'ファイル名']
        elif vectordb_file_type == "csv":
            matadatas["row"] += 1   # 行番号を1から始まるようにする
            vector_db_table = pd.concat([docs, matadatas], axis=1)   # ドキュメントとメタデータを結合         
            vector_db_table.columns = ['チャンクテキスト', '行', 'ファイル名']
        elif vectordb_file_type == "excel":
            vector_db_table = pd.concat([docs, matadatas[["page_name", "filename"]]], axis=1)   # ドキュメントとメタデータを結合            
            vector_db_table.columns = ['チャンクテキスト', 'シート名', 'ファイル名']
        vector_db_table.index = np.arange(1, len(vector_db_table)+1)   # 1から始まるインデックスにする   
        st.write(vector_db_table)


# ベクトルDBの準備
def prep_vector_db():

    st.markdown("#### RAG データ準備")

    # ベクトルDB表示の変数を初期化
    display_vector_db = False
    click_button = False
    disp_type = "json"

    cols = st.columns([20,50,30])
    with cols[0]:
        # データソース選択
        st.radio("ファイルタイプ選択", ("pdf", "csv", "excel"), key="vectordb_file_type",  args=[1, 0, 0], index=0,)

    with cols[1]:
        # ファイルリスト表示
        if st.session_state.vectordb_file_type == "pdf":
            st.write("PDFファイル一覧")
            file_list = os.listdir(DATA_FOLDER + "/pdf")
            df = pd.DataFrame(file_list, columns=["ファイル名"])
            df.insert(0, "No.", range(1, len(df) + 1))
            df["ベクトルDBに追加"] = True
            pdf_file_list = st.data_editor(df, hide_index=True, height=300)

        elif st.session_state.vectordb_file_type == "csv":
            st.write("CSVファイル一覧")
            file_list = os.listdir(DATA_FOLDER + "/csv")
            df = pd.DataFrame(file_list, columns=["ファイル名"])
            df.insert(0, "No.", range(1, len(df) + 1))
            df["ベクトルDBに追加"] = True
            csv_file_list = st.data_editor(df, hide_index=True, height=300)

        elif st.session_state.vectordb_file_type == "excel":
            st.write("Excelファイル一覧")
            file_list = os.listdir(DATA_FOLDER + "/excel")
            df = pd.DataFrame(file_list, columns=["ファイル名"])
            df.insert(0, "No.", range(1, len(df) + 1))
            df["ベクトルDBに追加"] = True
            excel_file_list = st.data_editor(df, hide_index=True, height=300)

    with cols[2]:
        st.write("　")
        st.write("　")

        if st.session_state.embeddings_api == "OpenAI Embedding API" and st.session_state.get("OPENAI_API_KEY") is None:
            st.error("Embedding APIにOpenAI Embedding APIを選択した場合は、OpenAIのAPI keyを設定してください")
        else:
            if st.session_state.vectordb_file_type == "pdf":
                if len(pdf_file_list) > 0:
                    st.button("ベクトルDBを作成", on_click=make_vector_db, args=("pdf", pdf_file_list))
                if os.path.isdir(os.path.join(VECTOR_DB_FOLDER, "vector_db_pdf")):
                    display_vector_db = True
            elif st.session_state.vectordb_file_type == "csv":
                if len(csv_file_list) > 0:
                    st.button("ベクトルDBを作成", on_click=make_vector_db, args=("csv", csv_file_list))
                if os.path.isdir(os.path.join(VECTOR_DB_FOLDER, "vector_db_csv")):
                    display_vector_db = True
            elif st.session_state.vectordb_file_type == "excel":
                if len(excel_file_list) > 0:
                    st.button("ベクトルDBを作成", on_click=make_vector_db, args=("excel", excel_file_list))
                if os.path.isdir(os.path.join(VECTOR_DB_FOLDER, "vector_db_excel")):
                    display_vector_db = True
            st.write("　")

            if display_vector_db:
                click_button = st.button("ベクトルDBを確認")
                subcols = st.columns([10, 90])
                with subcols[0]:
                    pass
                with subcols[1]:
                    disp_type = st.radio("出力タイプの選択", ("json", "シンプルテーブル"), horizontal=True)

    if click_button:
        with st.container(border=True):
            confirm_vector_db(st.session_state.vectordb_file_type, disp_type)

    # ollama RAGでの利用の前にrag_readyをFalseに設定しておく
    st.session_state.rag_ready = False

#---------------------------------
# ベクトルDB準備のメイン処理
if "prep_vector_db" in st.session_state:
    prep_vector_db()
else:
    st.error("アプリケーション立上げ時は「[ツールの設定](" + TOOL_SETTING_URL + ")」へアクセスしてください。")
