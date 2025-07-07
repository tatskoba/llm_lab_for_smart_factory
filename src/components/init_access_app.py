import streamlit as st

def init_access_app():
    # アプリに初めてアクセスする場合、トップページ（設定ページ）にまずはリダイレクトする
    if "init_access_app" not in st.session_state:
        if st.session_state.init_access_app != True:
            st.session_state.init_access_app = True
            st.write("最初にトップページにアクセスしてください")
            st.switch_page("setting.py")
