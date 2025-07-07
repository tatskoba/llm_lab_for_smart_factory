import os
import torch
import gc
import streamlit as st
from transformers import AutoModelForCausalLM 
from transformers import LlavaForConditionalGeneration   # for llava-calm2-siglip
from transformers import AutoProcessor
from transformers import BitsAndBytesConfig

# プロセッサー指定
PROCESSOR = os.environ.get('PROCESSOR')

# フォルダの設定
MODEL_FOLDER = os.environ.get('MODEL_FOLDER')

# ローカルに保存されたVLMモデルのモデル名とパス
model_path_name = {
    #"Phi-4-multimodal-instruct" : MODEL_FOLDER + "phi4m",  # preparing for v0.0.5
    "Phi-3.5-vision-instruct" : MODEL_FOLDER + "phi35v",
    "Phi-3.5-vision-finetuning-lora" : MODEL_FOLDER + "phi35v_finetuning_lora",
    "llava-calm2-siglip" : MODEL_FOLDER + "llava-calm2-siglip",
}

def vlm_model_load(selected_model_name: str): 

    with st.spinner(selected_model_name + "モデルをロード中..."):
        # ローカルに保存したモデルのパスを取得
        model_local_path = model_path_name[selected_model_name]
                 
        # もしファイルのパスが存在すれば、モデルをロード
        if  os.path.exists(model_local_path):

            # モデルをメモリから削除
            if "vlm_model" in st.session_state:
                del st.session_state.vlm_model
            if "vlm_tokenizer" in st.session_state:
                del st.session_state.vlm_tokenizer

            # ガベージコレクションで変数を削除してメモリを解放
            gc.collect()

            if PROCESSOR == "gpu":
                # GPUメモリの解放
                torch.cuda.empty_cache()

            if selected_model_name == "llava-calm2-siglip":
                st.session_state.vlm_model = LlavaForConditionalGeneration.from_pretrained(
                    model_local_path,
                    device_map=PROCESSOR,
                    torch_dtype=torch.bfloat16,
                    trust_remote_code=True,                    
                  # _attn_implementation='flash_attention_2',   # only for GPU
                    _attn_implementation='eager',  # only for CPU
                )
            else:
                # Note: set _attn_implementation='eager' if you don't have flash_attn installed
                st.session_state.vlm_model = AutoModelForCausalLM.from_pretrained(
                    model_local_path,
                    device_map=PROCESSOR,
                    torch_dtype="auto",
                    trust_remote_code=True,
                    # _attn_implementation='flash_attention_2',   # only for GPU
                    _attn_implementation='eager',  # only for CPU
                    use_cache=True,
                )

            # トークナイザーの読み込み
            st.session_state.vlm_tokenizer = AutoProcessor.from_pretrained(
                model_local_path,
                trust_remote_code=True,
                num_crops=16,
            )
                        
            #st.session_state.vlm_model
            st.session_state.vlm_model.to(PROCESSOR)
        else:
            st.error("選択されたモデルはまだローカルにダウンロードされていません。")
            return False

    # モデルがロードされたら、モデル名とステータスをセット
    st.session_state.vlm_model_name = selected_model_name
    st.session_state.vision_model_availability = True

    return True
