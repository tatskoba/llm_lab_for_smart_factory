from PIL import Image
import torch
import gc
import os
import streamlit as st
from streamlit_chat import message
from components.prompt_library import list_prompt_files, get_prompt_library, set_prompt

# プロセッサー指定
PROCESSOR = os.environ.get('PROCESSOR')

# ツールの設定ページのURL
TOOL_SETTING_URL = os.environ['TOOL_SETTING_URL']

# メッセージを表示する
def display_messages():
    if "vlm_messages" not in st.session_state:
        st.session_state["vlm_messages"] = []
    
    for i, (msg, is_user) in enumerate(st.session_state["vlm_messages"]):
        message(msg, is_user=is_user, key=str(i))
    st.session_state["vlm_thinking_spinner"] = st.empty()

# 画像についての質問を処理する
def image_process_input(prompt, image_num, images):

    if st.session_state["user_input"] and len(st.session_state["user_input"].strip()) > 0:

        ask_text = st.session_state["user_input"].strip()

        with st.session_state["vlm_thinking_spinner"], st.spinner(f"解析中..."):
            # LLMに質問を投げる
            # LLaVa-Calm2-SigLipモデルの場合
            if st.session_state.vlm_model_name == "llava-calm2-siglip":
                # プロンプトの設定
                prompt = "USER: <image>" + ask_text + "\nASSISTANT: "

                agent_text = chat_vlm("1枚", images, prompt)

            # phi3v/phi3.5vモデルの場合
            else:
                # プロンプトの設定
                user_prompt = '<|user|>\n'
                assistant_prompt = '<|assistant|>\n'
                prompt_suffix = "<|end|>\n"

                # 画像が１枚の時
                if image_num == "1枚":
                    prompt = f"{user_prompt}<|image_1|>\n" + ask_text + f"{prompt_suffix}{assistant_prompt}"
                    agent_text = chat_vlm(image_num, images, prompt)
                # 画像が２枚の時
                elif image_num == "2枚":
                    prompt = f"{user_prompt}<|image_1|>\n<|image_2|>\n" + ask_text + f"?{prompt_suffix}{assistant_prompt}"
                    agent_text = chat_vlm(image_num, images, prompt)

        st.session_state["vlm_messages"].append((ask_text, True))
        st.session_state["vlm_messages"].append((agent_text, False))

        # 最後に入力をクリアする
        st.session_state["user_input"] = ""

# 対話履歴をクリアする
def clear_messages():
    st.session_state["vlm_messages"] = []

# プロンプトライブラリの選択を入力欄にセットする
def set_prompt_library():
    st.session_state["user_input"] = st.session_state["prompt_library"]

def chat_vlm(image_num, image_file, prompt):

    if image_num == "1枚":
        # 1. Analyze the image with Phi-3-Vision
        images = Image.open(image_file)
        #st.write(image_file)
    elif image_num == "2枚":
        image_1 = Image.open(image_file[0])
        image_2 = Image.open(image_file[1])
        images = [image_1, image_2]
        #st.write(images)

    # ガベージコレクションで変数を削除してメモリを解放
    gc.collect()

    if PROCESSOR == "gpu":
        # GPUメモリの解放
        torch.cuda.empty_cache()

    with st.spinner("画像を解析中..."):

        generation_args = { 
            "max_new_tokens": int(st.session_state.vlm_new_max_tokens), 
            "temperature": float(st.session_state.vlm_temperature), 
            "do_sample": bool(st.session_state.vlm_do_sample), 
        }

        # LLaVa-Calm2-SigLipモデルの場合はtorch.bfloat16を使用
        if st.session_state.vlm_model_name == "llava-calm2-siglip":
            inputs = st.session_state.vlm_tokenizer(prompt, images, return_tensors="pt").to(PROCESSOR, torch.bfloat16)
            generate_ids = st.session_state.vlm_model.generate(
                **inputs,
                **generation_args
            )

            response = st.session_state.vlm_tokenizer.tokenizer.decode(
                generate_ids[0][:-1],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )

            return response.split("ASSISTANT:")[-1]

        # phi3v/phi3.5vモデルの場合はtorch.float32を使用
        else:
            inputs = st.session_state.vlm_tokenizer(prompt, images, return_tensors="pt").to(PROCESSOR)

            generate_ids = st.session_state.vlm_model.generate(
                **inputs, 
                eos_token_id=st.session_state.vlm_tokenizer.tokenizer.eos_token_id,
                **generation_args
            )

            generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]

            response = st.session_state.vlm_tokenizer.batch_decode(
                generate_ids, 
                skip_special_tokens=True, 
                clean_up_tokenization_spaces=False
            )[0]

            return response

def chat_vim():

    st.markdown("#### 画像について質問する")

    with st.expander("設定", expanded=True):
        cols = st.columns([80,20])
        with cols[0]:
            st.write("画像ファイル選択")
            # lava-calm2-siglipモデルの場合
            if st.session_state.vlm_model_name == "llava-calm2-siglip":
                # ファイルをアップロードする
                uploaded_image_file = st.file_uploader("画像ファイル", type=["jpg", "jpeg", "png"])
                uploaded_image_file_2 = None
                image_num = "1枚"
            # phi3v/phi3.5vモデルの場合        
            else:
                image_num = st.radio(label_visibility="collapsed", label="アップロードする画像ファイルの枚数", options=("1枚", "2枚"),  index=0, horizontal=True,)
                sub_cols = st.columns([50,50])
                with sub_cols[0]:
                    # ファイルをアップロードする
                    uploaded_image_file = st.file_uploader("画像ファイル", type=["jpg", "jpeg", "png"])
                with sub_cols[1]:
                    uploaded_image_file_2 = None
                    if image_num == "2枚":
                        uploaded_image_file_2 = st.file_uploader("2枚目の画像ファイル", type=["jpg", "jpeg", "png"])
        with cols[1]:
            vlm_dialog_window_size_list = [200, 300, 400, 500, 600]
            st.selectbox(
                '対話ウィンドウの長さ',
                tuple(vlm_dialog_window_size_list),
                key="vlm_dialog_window_size"
            )
            if "vlm_dialog_window_size" not in st.session_state:
                st.session_state["vlm_dialog_window_size"] = 400
            st.button("対話履歴をクリア", on_click=clear_messages)

    st.markdown("###### 対話履歴")
    cols = st.columns([30, 70])
    with cols[0]:
        if uploaded_image_file is not None:
            image_data = Image.open(uploaded_image_file)
            st.image(image_data)
            images = uploaded_image_file
            if uploaded_image_file_2 is not None:
                image_data_2 = Image.open(uploaded_image_file_2)
                st.image(image_data_2)
                images = [uploaded_image_file, uploaded_image_file_2]
        else:
            st.write("画像ファイルをアップロードしてください")

    with cols[1].container(height=st.session_state.vlm_dialog_window_size):
        # メッセージを表示する
        display_messages()

    # ユーザーの入力を処理する（OSS LLMとの対話）
    if uploaded_image_file is not None:
        cols = st.columns([50,50])
        with cols[0]:
            # 入力欄を表示
            user_question = st.text_area(
                "VLM（" + st.session_state.vlm_model_name + "）への質問", 
                key="user_input", 
                height=160,
                max_chars=1000,
            )
            st.button("質問する", on_click=image_process_input, args=(user_question, image_num, images)) 

        with cols[1]:
            # プロンプトライブラリを取得してユーザ入力欄に設定する
            files = list_prompt_files()
            if len(files) > 0:
                selected_prompt_library_file = st.selectbox("プロンプトライブラリ", files, index=None)
            if selected_prompt_library_file != "":
                prompt_list = get_prompt_library(selected_prompt_library_file)
                if len(prompt_list) > 0:
                    st.selectbox("プロンプト一覧", prompt_list, index=None, key="selected_prompt", on_change=set_prompt)

#------------------------------------
# 時系列データについての質問のメイン処理

if "vision_model_availability" in st.session_state:
    chat_vim()
else:
    st.error("アプリケーション立上げ時は「[ツールの設定](" + TOOL_SETTING_URL + ")」でモデルをロードしてください。")


