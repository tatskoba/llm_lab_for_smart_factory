import os
import streamlit as st

# システム設定
st.set_page_config(
     page_title="LLM Lab for Smart Factory",
     page_icon=":mag_right:",
     layout="wide",
     initial_sidebar_state="expanded"
)

# メインページ
if __name__ == "__main__":

     # アプリへの初回アクセスでフラグを立てる
     st.session_state.init_access_app = True

     # 環境変数を設定
     os.environ['DATA_FOLDER'] = "/workspaces/llm_lab/data/"             # データフォルダの設定
     os.environ['MODEL_FOLDER'] = "/workspaces/llm_lab/models/"          # モデルフォルダの設定
     os.environ['VECTOR_DB_FOLDER'] = "/workspaces/llm_lab/vector_db/"   # ベクターDBフォルダの設定
     os.environ['MYSQL_TABLE_NAME'] = "time_series_data"                 # MySQLテーブル名の設定

     # プロダクト情報の表示
     disp_prod_company = """
          <div style="position: fixed; bottom: 0; padding: 18px;  color: gray">
               <div style="font-size: 14px; ">AI Collab Worker</div>
               <div style="font-size: 15px; font-weight: bold""> LLM Lab for Smart Factory</div>
               <div style="font-size: 14px;">version 0.0.6</div>
               <div style="font-size: 14px; ">© LangCloud Technologies LLC</div>
          </div>
       """
     st.sidebar.markdown(disp_prod_company, unsafe_allow_html=True)

     # Streamlitの右上のハンバーガーメニューとDeployボタンを非表にする
     st.markdown(
          """
          <style>
               #MainMenu {visibility: hidden;}
               .stAppDeployButton {display:none;}
          </style>
          """, unsafe_allow_html=True
     )

     # サイドバーの設定
     # アイコンのリストはこちら  https://fonts.google.com/icons
     #pg = st.navigation([setting, prep_vector_db, chat_ollama_rag, prep_vision_data, classify_image, chat_vlm, chat_ollama_vlm, chat_openai, faq])
     
     pages = {
          "初期設定" : [
               st.Page(page="pages/setting.py", title="ツールの設定", icon=":material/settings_applications:")
          ],
          "ローカルRAG": [
               st.Page(page="pages/prep_vector_db.py", title="データ準備", icon=":material/folder:"),
               st.Page(page="pages/chat_ollama_rag.py", title="RAG (ollama)", icon=":material/psychology:"), 
          ],
          "ローカル画像AI": [
               st.Page(page="pages/prep_vision_data.py", title="データ準備", icon=":material/folder:"),
               st.Page(page="pages/classify_image.py", title="画像分類", icon=":material/label_important:"),
               st.Page(page="pages/chat_ollama_vlm.py", title="画像質問回答 (ollama)", icon=":material/psychology:"), 
               st.Page(page="pages/chat_vlm.py", title="画像質問回答 (個別VLM)", icon=":material/hub:"),
               st.Page(page="pages/ollama_ocr.py", title="OCR (ollama)", icon=":material/psychology:"), 
          ],
          "オンライン生成AI": [
               st.Page(page="pages/chat_openai.py", title="ChatGPTとの対話", icon=":material/open_with:"),
          ],
          "ソフトウェアについて": [
               st.Page(page="pages/faq.py", title="情報", icon=":material/help:")
          ]
     }
     pg = st.navigation(pages)
     pg.run()

