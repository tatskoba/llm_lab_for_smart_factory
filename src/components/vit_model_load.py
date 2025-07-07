import os
import torch
import gc
import streamlit as st
from transformers import AutoImageProcessor, AutoModelForImageClassification

# プロセッサー指定
PROCESSOR = os.environ.get('PROCESSOR')

# ローカルに保存されたVLMモデルのモデル名とパス
model_path_name = {
    "Swin Transformer V2":"/workspaces/llm_lab/models/swinv2",
    "Swin Transformer V2 Finetuning":"/workspaces/llm_lab/models/swinv2_finetuning",    
}

def vit_model_load(selected_model_name: str): 
       
    with st.spinner(selected_model_name + "モデルをロード中..."):
        # ローカルに保存したモデルのパスを取得
        model_local_path = model_path_name[selected_model_name]       
        
        # もしファイルのパスが存在すれば、モデルをロード
        if  os.path.exists(model_local_path):

            # モデルをメモリから削除
            if "vit_model" in st.session_state:
                del st.session_state.vit_model
            if "vit_tokenizer" in st.session_state:
                del st.session_state.vit_tokenizer

            # ガベージコレクションで変数を削除してメモリを解放
            gc.collect()

            if PROCESSOR == "gpu":
                # GPUメモリの解放
                torch.cuda.empty_cache()

            if selected_model_name == "Swin Transformer V2" or selected_model_name == "Swin Transformer V2 Finetuning":
                st.session_state.vit_tokenizer = AutoImageProcessor.from_pretrained(model_local_path)
                st.session_state.vit_model = AutoModelForImageClassification.from_pretrained(model_local_path)
            else:
                st.error("選択されたモデルはまだローカルにダウンロードされていません。")
                return False
        else:
            st.error("選択されたモデルはまだローカルにダウンロードされていません。")
            return False

    # モデルがロードされたら、モデル名とステータスをセット
    st.session_state.vit_model_name = selected_model_name
    st.session_state.vit_model_availability = True

    return True
