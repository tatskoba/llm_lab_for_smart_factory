import os
import streamlit as st
from langchain_community.chat_models import ChatOllama
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain.prompts import PromptTemplate
import chromadb


# プロセッサー指定
PROCESSOR = os.environ.get('PROCESSOR')

# フォルダの設定
DATA_FOLDER = os.environ.get('DATA_FOLDER')
VECTOR_DB_FOLDER = os.environ.get('VECTOR_DB_FOLDER')
MODEL_FOLDER = os.environ.get('MODEL_FOLDER')

class RAG():
    def __init__(self):
        # クラス変数を初期化
        self.clear()

        # ollaamaシステムプロンプトを取得
        ollama_system_prompt = self.read_ollama_system_prompt()

        # OllamaモデルAPIを設定する
        self.model = ChatOllama(
            model=st.session_state.ollama_llm_model_name,            # Ollamaのモデル名
            base_url="http://ollama:11434/",                         # OllamaのAPIのURL
            temperature=float(st.session_state.ollama_temperature),  # Ollamaの温度設定
            system=ollama_system_prompt,                            # Ollamaのシステムプロンプト
            # その他のパラメータはデフォルト値
            # 追加で設定する場合はAPIマニュアルを参照のこと https://api.python.langchain.com/en/latest/chat_models/langchain_community.chat_models.ollama.ChatOllama.html
        )

        # システムプロンプトを読み込む
        with open(DATA_FOLDER + "ollama_config/ollama_system_prompt.txt") as f:
            self.system_prompt = f.read()

    # ollamaシステムプロンプトをファイルから読み込む
    def read_ollama_system_prompt(self):
        ollama_system_prompt = ""
        read_file = DATA_FOLDER + "ollama_config/ollama_system_prompt.txt"
        with open(read_file) as f:
            ollama_system_prompt = f.read()
        return ollama_system_prompt

    # ベクトルDBをロードする
    def load_vector_db(self, file_type):

        # Chromadbのキャッシュをクリアする
        chromadb.api.client.SharedSystemClient.clear_system_cache()

        # コレクション名とベクトルDBの保存先を設定
        if file_type == "PDF":
            collection_name = "pdf_collection"
            persist_directory = VECTOR_DB_FOLDER + "vector_db_pdf"
        elif file_type == "CSV":
            collection_name = "csv_collection"
            persist_directory = VECTOR_DB_FOLDER + "vector_db_csv"
        elif file_type == "Excel":
            collection_name = "excel_collection"        
            persist_directory = VECTOR_DB_FOLDER + "vector_db_excel"
        else:
            collection_name = None
            persist_directory = None

        if collection_name == None:
            st.write("エラー: コレクションが設定されていません")
            return
        
        # ベクトルDBのクライアントを作成
        client = chromadb.PersistentClient(path=persist_directory)

        # 選択されたembedding APIをを選択
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

        # persistされたベクトルDBをロードする
        vector_db = Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            client=client,
        )

        # ベクトルDBからretriever作成
        self.retriever = vector_db.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "k": st.session_state.sim_search_top_k,
                "score_threshold": st.session_state.score_threshold,
            },
        )

    # プロンプトテンプレートを設定
    def set_prompt_template(self):

        # LangChainのRAGプロンプトテンプレート
        # 参照先：https://smith.langchain.com/hub/rlm/rag-prompt
        # 英語でのシステムプロンプト例
        #     You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
        prompt_text = self.system_prompt + """
            Question: {question} 
            Context: {context} 
            Answer:
            """
        self.prompt = PromptTemplate.from_template(
            prompt_text
        )
            
    # Chainを作る
    def make_chain(self):

        self.chain = ({"context": self.retriever, "question": RunnablePassthrough()}
                          | self.prompt
                          | self.model
                          | StrOutputParser())

    # 質問を投げる
    def ask(self, query: str):
        if not self.chain:
            return "ベクトルDBを選択してください。"

        # ユーザー質問の回答を取得
        ans_result = self.chain.invoke(query)

        # ベクトルDBから質問の回答（チャンク）を取得
        ret_text = self.retriever.invoke(query)

        # チャンク情報を取得
        chunk_info = "■ 取得されたチャンク数：" + str(len(ret_text)) + "\n"
        for i in range(len(ret_text)):
            # チャンクのテキストを取得
            chunk_info += "＜チャンク " + str(i+1) +"＞ ---------------------------------------------\n"
            chunk_info += "【メタデータ】\n"
            chunk_info += str(ret_text[i].metadata) + "\n"
            chunk_info += "【チャンクテキスト】\n"
            chunk_info += str(ret_text[i].page_content) + "\n\n"

        # デバッグ：チャンクされたドキュメントを保存
        with open(VECTOR_DB_FOLDER + "retriever_output.txt", "w") as f:
            # 取得されたチャンク情報をファイルに書き込む
            f.write(chunk_info)

        return ans_result, chunk_info

    # メモリをクリアする
    def clear(self):
        self.vector_store = None
        self.retriever = None
        self.chain = None
        self.prompt = None
        self.model = None