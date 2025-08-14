import os
import glob
import streamlit as st
from components.ollama_llm_list import get_ollama_llm_list
from components.vlm_model_load import vlm_model_load
from components.vit_model_load import vit_model_load

DATA_FOLDER = os.environ.get('DATA_FOLDER')
MODEL_FOLDER = os.environ.get('MODEL_FOLDER')

# 画像データ準備のフラグを設定
st.session_state["prep_vision_data"] = True

# ベクトルDBの準備のフラグを設定
st.session_state["prep_vector_db"] = False

# OpenAI API Keyの設定
def set_openai_api_key(api_key: str):
    os.environ["OPENAI_API_KEY"] = api_key  # 環境変数にセットする
    st.session_state["OPENAI_API_KEY"] = api_key    # セッション変数にもセットしておく

    st.session_state["openai_api_key_configured"] = True
    print('OPENAI API keyの設定が成功しました!')

# Azure OpenAI API Keyの設定
def set_azure_openai_api_key(api_key: str):
    os.environ["AZURE_OPENAI_API_KEY"] = api_key  # 環境変数にセットする
    st.session_state["AZURE_OPENAI_API_KEY"] = api_key    # セッション変数にもセットしておく

    st.session_state["azure_openai_api_key_configured"] = True
    print('AZURE OPENAI API keyの設定が成功しました!')

# プロンプトテンプレートデータをファイルから読み込む
def read_prompt_templete():
    st.session_state.edit_prompt_template_file = DATA_FOLDER + "prompt_template/" + st.session_state.selected_prompt_template_file
    with open(st.session_state.edit_prompt_template_file) as f:
        prompt_text = f.read()
        st.session_state["prompt_template_input"] = prompt_text

# プロンプトテンプレートをファイル保存する
def save_prompt_template():
    if st.session_state.selected_prompt_template_file is not None:
        with open(DATA_FOLDER + "prompt_template/" + st.session_state.selected_prompt_template_file, "w") as f:
            f.write(st.session_state.prompt_template_input)
    else:
        st.error("ファイルが選択されていません。")

# ollamaシステムプロンプトをファイルから読み込む
def read_ollama_system_prompt():
    system_prompt = ""
    read_file = DATA_FOLDER + "ollama_config/ollama_system_prompt.txt"
    with open(read_file) as f:
        system_prompt = f.read()
    return system_prompt        

# ollamaシステムプロンプトをファイル保存する
def save_ollama_system_prompt(value):
    save_file = DATA_FOLDER + "ollama_config/ollama_system_prompt.txt"
    with open(save_file, "w") as f:
        f.write(value)


#-----------------------------------
# システム設定のメイン処理

# 画面表示
st.markdown(
    "#### ツールの設定"
)

st.markdown(
    "##### 1. 画像分類モデル（Vision Transformer）の設定"
)

# 画像分類モデルの設定
st.markdown("- モデルの選択とロード")    

# 利用するモデルを選択する
cols = st.columns([4, 32, 40, 24])
with cols[0]:
    pass
with cols[1]:
    vit_model_name_list = [
        "Swin Transformer V2", 
        "Swin Transformer V2 Finetuning"
    ]
    if "selected_vit_model_name" not in st.session_state:
        index = 0
    else:
        index = vit_model_name_list.index(st.session_state.selected_vit_model_name) if st.session_state.selected_vit_model_name in vit_model_name_list else 0
    
    st.session_state.selected_vit_model_name = st.selectbox(
        '画像分類モデル',
        tuple(vit_model_name_list),
        index=index
    )
with cols[2]:
    st.write("  ")
    st.write("  ")
    if st.button('モデルをロードする', key="load_vit_model"):
        # 画像分類モデルの読み込み
        vit_model_load(st.session_state.selected_vit_model_name)
with cols[3]:
    pass

cols = st.columns([4, 40, 56])
with cols[0]:
    pass
with cols[1]:
    if "vit_model_name" not in st.session_state:
        st.error("モデルをロードしてください。")
    else:
        if st.session_state.selected_vit_model_name == st.session_state.vit_model_name:
            st.write("選択されたモデルはロードされています。")
        else:
            st.error("選択されたモデルはロードされていません。")
with cols[2]:
    pass

st.divider()

st.markdown(
    "##### 2. VLM（Vision Language Model）の設定"
)

# VLMモデルの設定
st.markdown("- モデルの選択とロード")    

# 利用するモデルを選択する
cols = st.columns([4, 32, 40, 24])
with cols[0]:
    pass
with cols[1]:
    vlm_model_name_list = [
        "Phi-3.5-vision-instruct",
        "Phi-3.5-vision-finetuning-lora",
        "llava-calm2-siglip",
    ]
    if "selected_vlm_model_name" not in st.session_state:
        index = 0
    else:
        index = vlm_model_name_list.index(st.session_state.selected_vlm_model_name) if st.session_state.selected_vlm_model_name in vlm_model_name_list else 0
    
    st.session_state.selected_vlm_model_name = st.selectbox(
        'VLMモデル',
        tuple(vlm_model_name_list),
        index=index
    )
with cols[2]:
    st.write("  ")
    st.write("  ")
    if st.button('モデルをロードする', key="load_vlm_model"):
        # VLMモデルの読み込み
        vlm_model_load(st.session_state.selected_vlm_model_name)
with cols[3]:
    pass

cols = st.columns([4, 40, 56])
with cols[0]:
    pass
with cols[1]:
    if "vlm_model_name" not in st.session_state:
        st.error("モデルをロードしてください。")
    else:
        if st.session_state.selected_vlm_model_name == st.session_state.vlm_model_name:
            st.write("選択されたモデルはロードされています。")
        else:
            st.error("選択されたモデルはロードされていません。")
with cols[2]:
    pass

if "vlm_model_name" in st.session_state:
    # VLMモデルのパラメータ設定
    st.markdown("- VLMモデルのパラメータ設定")
    cols = st.columns([4, 32, 32, 32])
    with cols[0]:
        pass
    with cols[1]:
        vlm_new_max_tokens_list = [200, 400, 600, 800, 1000]
        if "vlm_new_max_tokens" not in st.session_state:
            st.session_state.vlm_new_max_tokens = '400'
        index = vlm_new_max_tokens_list.index(st.session_state.vlm_new_max_tokens) if st.session_state.vlm_new_max_tokens in vlm_new_max_tokens_list else 0
        st.session_state.vlm_new_max_tokens = st.selectbox(
            'new_max_tokens',
            tuple(vlm_new_max_tokens_list),
            key="vlm_param1",
            index=index
        )
    with cols[2]:
        vlm_temperature_list = [0, 0.2, 0.4, 0.6, 0.8, 1]
        if "vlm_temperature" not in st.session_state:
            st.session_state.vlm_temperature = '0.8'
        index = vlm_temperature_list.index(st.session_state.vlm_temperature) if st.session_state.vlm_temperature in vlm_temperature_list else 0
        st.session_state.vlm_temperature = st.selectbox(
            'temperature',
            tuple(vlm_temperature_list),
            key="vlm_param2",
            index=index
        )
    with cols[3]:
        vlm_do_sample_list = [True, False]
        if "vlm_do_sample" not in st.session_state:
            st.session_state.vlm_do_sample = False
        index = vlm_do_sample_list.index(st.session_state.vlm_do_sample) if st.session_state.vlm_do_sample in vlm_do_sample_list else 0
        st.session_state.vlm_do_sample = st.selectbox(
            'do_sample',
            tuple(vlm_do_sample_list),
            key="vlm_param3",
            index=index
        )

st.divider()

st.markdown(
    "##### 3. ollamaの設定"
)

# ollamaコンテナの接続およびLLMモデルの取得状況を確認    
ollama_llm_list, ollama_vlm_list, ollama_ocr_list = get_ollama_llm_list()
if len(ollama_llm_list) > 0 and len(ollama_vlm_list) > 0 and len(ollama_ocr_list) > 0:
    st.session_state.ollama_availability = True
    st.markdown("- モデルの選択（モデルはダウンロード済み）")
else:
    st.session_state.ollama_availability = False
    st.error("- ollamaコンテナが利用できない、または、モデルが利用できません。モデルをダウンロードしてください。")
    st.session_state.ollama_llm_model_name = ""

if st.session_state.ollama_availability == True:
    # 利用可能なollamaモデルを確認して表示
    cols = st.columns([4, 32, 32, 32])
    with cols[0]:
        pass
    with cols[1]:
        # デフォルトのollama LLMモデルの設定
        if "ollama_llm_model_name" not in st.session_state:
            if "phi4:14b" in ollama_llm_list:
                st.session_state.ollama_llm_model_name = "phi4:14b"                
                index = ollama_llm_list.index(st.session_state.ollama_llm_model_name)
            else:
                index = 0
        else:
            if "ollama_param0" in st.session_state:
                if st.session_state.ollama_llm_model_name != st.session_state.ollama_param0:
                    index = ollama_llm_list.index(st.session_state.ollama_param0)
                else:
                    index = ollama_llm_list.index(st.session_state.ollama_llm_model_name)           
            else:
                index = ollama_llm_list.index(st.session_state.ollama_llm_model_name)       
        st.session_state.ollama_llm_model_name = st.selectbox(
            "LLMの選択",
            tuple(ollama_llm_list),
            index=index,
            key="ollama_param0"
        )
    with cols[2]:
        # デフォルトのollama Vision LMモデルの設定
        if "ollama_vlm_model_name" not in st.session_state:
            st.session_state.ollama_vlm_model_name = "gemma3:12b"
            index = ollama_vlm_list.index(st.session_state.ollama_vlm_model_name)
        else:
            if "ollama_param1" in st.session_state:
                if st.session_state.ollama_vlm_model_name != st.session_state.ollama_param1:
                    index = ollama_vlm_list.index(st.session_state.ollama_param1)
                else:
                    index = ollama_vlm_list.index(st.session_state.ollama_vlm_model_name)
            else:
                index = ollama_vlm_list.index(st.session_state.ollama_vlm_model_name)
        st.session_state.ollama_vlm_model_name = st.selectbox(
            "VLMの選択",
            tuple(ollama_vlm_list),
            index=index,
            key="ollama_param1"
        )
    with cols[3]:
        # デフォルトのOCR用ollama Vision LMモデルの設定
        if "ollama_ocr_model_name" not in st.session_state:
            st.session_state.ollama_ocr_model_name = "gemma3:12b"
            index = ollama_ocr_list.index(st.session_state.ollama_ocr_model_name)
        else:
            if "ollama_param9" in st.session_state:
                if st.session_state.ollama_ocr_model_name != st.session_state.ollama_param9:
                    index = ollama_ocr_list.index(st.session_state.ollama_param9)
                else:
                    index = ollama_ocr_list.index(st.session_state.ollama_ocr_model_name)
            else:
                index = ollama_ocr_list.index(st.session_state.ollama_ocr_model_name)
        st.session_state.ollama_ocr_model_name = st.selectbox(
            "OCR用VLMの選択",
            tuple(ollama_ocr_list),
            index=index,
            key="ollama_param9"
        )

    # RAGのモデル設定
    st.markdown("- RAGのモデル設定")    
    cols = st.columns([4, 32, 42, 22])
    with cols[0]:
        pass
    with cols[1]:
        ollama_temperature_list = [0, 0.2, 0.4, 0.6, 0.8, 1]
        if "ollama_temperature" not in st.session_state:
            st.session_state.ollama_temperature = 0.8
            index = ollama_temperature_list.index(st.session_state.ollama_temperature)
        else:
            if "ollama_param2" in st.session_state:
                if st.session_state.ollama_temperature != st.session_state.ollama_param2:
                    index = ollama_temperature_list.index(st.session_state.ollama_param2)
                else:
                    index = ollama_temperature_list.index(st.session_state.ollama_temperature)
            else:
                index = ollama_temperature_list.index(st.session_state.ollama_temperature)
        st.session_state.ollama_temperature = st.selectbox(
            'temperature',
            tuple(ollama_temperature_list),
            index=index,
            key="ollama_param2"
        )
    with cols[2]:
        # ollamaシステムプロンプトをファイルから読み込む
        system_prompt = read_ollama_system_prompt()

        # ollamaシステムプロンプト設定欄
        input = st.text_area(
            "RAGシステムプロンプト",
            placeholder="入力してください",
            value=system_prompt,
            key="ollama_param3", 
            height=125,
            max_chars=1000,
        )
        st.button("変更を反映", key="ollama_system", on_click=save_ollama_system_prompt, args=(input,))
    with cols[3]:
        pass

    st.write("- Embedding APIの選択")

    cols = st.columns([4, 32, 32, 32])
    with cols[0]:
        pass 
    with cols[1]:
        # デフォルトのEmbedding APIの設定
        embeddings_list = ["intfloat/multilingual-e5-large", "OpenAI Embedding API"]
        if "embeddings_api" not in st.session_state:
            st.session_state.embeddings_api = "intfloat/multilingual-e5-large"
            index = embeddings_list.index(st.session_state.embeddings_api)
        else:
            if "ollama_param4" in st.session_state:
                if st.session_state.embeddings_api != st.session_state.ollama_param4:
                    index = embeddings_list.index(st.session_state.ollama_param4)
                else:
                    index = embeddings_list.index(st.session_state.embeddings_api)
            else:
                index = embeddings_list.index(st.session_state.embeddings_api)
        st.session_state.embeddings_api = st.selectbox(
            "",
            tuple(embeddings_list),
            index=index,
            key="ollama_param4", 
            label_visibility="collapsed",
        )

        # multilingual-e5-largeの場合、モデルファイルが存在するか確認
        if st.session_state["embeddings_api"] == "intfloat/multilingual-e5-large" \
            and os.path.exists(MODEL_FOLDER + "multilingual-e5-large") == False:
            st.error("選択されたモデルはまだローカルにダウンロードされていません。")

        # Embeddingモデルを設定したのでベクトルDBの準備のフラグを設定
        st.session_state.prep_vector_db = True

    with cols[2]:
        pass
    with cols[3]:
        pass

    st.markdown("- RAGのText splitter設定（Chromaのための文書前処理）") 
    cols = st.columns([4, 32, 32, 32])
    with cols[0]:
        pass
    with cols[1]:
        # デフォルトのText splitterの設定
        # チャンクサイズの設定
        rag_chunk_size_list = [128, 256, 512, 1024]
        if "rag_chunk_size" not in st.session_state:
            st.session_state.rag_chunk_size = 512
            index = rag_chunk_size_list.index(st.session_state.rag_chunk_size)
        else:
            if "ollama_param5" in st.session_state:
                if st.session_state.rag_chunk_size != st.session_state.ollama_param5:
                    index = rag_chunk_size_list.index(st.session_state.ollama_param5)
                else:
                    index = rag_chunk_size_list.index(st.session_state.rag_chunk_size)
            else:
                index = rag_chunk_size_list.index(st.session_state.rag_chunk_size)
        st.session_state.rag_chunk_size = st.selectbox(
            'チャンクサイズ',
            tuple(rag_chunk_size_list),         
            index=index,
            key="ollama_param5"
        )
    with cols[2]:
        # チャンクの重複の設定
        chunk_overlap_list = [0, 64, 128, 256, 512]
        if "chunk_overlap" not in st.session_state:
            st.session_state.chunk_overlap = 256
            index = chunk_overlap_list.index(st.session_state.chunk_overlap)
        else:
            if "ollama_param6" in st.session_state:
                if st.session_state.chunk_overlap != st.session_state.ollama_param6:
                    index = chunk_overlap_list.index(st.session_state.ollama_param6)
                else:
                    index = chunk_overlap_list.index(st.session_state.chunk_overlap) 
            else:
                index = chunk_overlap_list.index(st.session_state.chunk_overlap)
        prev_value = st.session_state.chunk_overlap
        st.session_state.chunk_overlap = st.selectbox(
            'チャンクの重複',
            tuple(chunk_overlap_list), 
            index=index,
            key="ollama_param6",
            help="チャンクサイズよりも小さな値を設定してください（チャンクサイズ以上の値は選択できません）",
        )
        if  st.session_state.chunk_overlap >= st.session_state.rag_chunk_size:
            st.error("チャンクの重複はチャンクサイズよりも小さな値を設定してください")
    with cols[3]:
        pass

    # 類似文書検索（Vector Store）設定
    st.markdown("- 類似文書検索（Vector Store）設定") 
    cols = st.columns([4, 32, 32, 32])
    with cols[0]:
        pass
    with cols[1]:
        # 類似文書検索数（top_k）の設定（＝検索で取得するチャンク数）
        sim_search_top_k_list = [1, 3, 5, 10, 15, 30]
        if "sim_search_top_k" not in st.session_state:
            st.session_state.sim_search_top_k = 3
            index = sim_search_top_k_list.index(st.session_state.sim_search_top_k)
        else:
            if "ollama_param7" in st.session_state:
                if st.session_state.sim_search_top_k != st.session_state.ollama_param7:
                    index = sim_search_top_k_list.index(st.session_state.ollama_param7)
                else:
                    index = sim_search_top_k_list.index(st.session_state.sim_search_top_k)
            else:
                index = sim_search_top_k_list.index(st.session_state.sim_search_top_k)
        st.session_state.sim_search_top_k = st.selectbox(
            '取得する上位のチャンク数',
            tuple(sim_search_top_k_list),
            index=index,
            key="ollama_param7"            
        )
    with cols[2]:
        # 類似スコアの閾値の設定
        score_threshold_list = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
        if "score_threshold" not in st.session_state:
            st.session_state.score_threshold = 0.3
            index = score_threshold_list.index(st.session_state.score_threshold)
        else:
            if "ollama_param8" in st.session_state:
                if st.session_state.score_threshold != st.session_state.ollama_param8:
                    index = score_threshold_list.index(st.session_state.ollama_param8)
                else:
                    index = score_threshold_list.index(st.session_state.score_threshold)
            else:
                index = score_threshold_list.index(st.session_state.score_threshold)
        st.session_state.score_threshold = st.selectbox(
            '取得する類似スコアの閾値',
            tuple(score_threshold_list),
            index=index,
            key="ollama_param8"
        )       
    with cols[3]:
        pass

st.divider()

st.markdown(
    "##### 4. クラウドサービスの選択"
)

# クラウドサービスの選択（OpenAI ChatGPT or Azure OpenAI）
cols = st.columns([25, 75])
with cols[0]:
    selected_chatgpt_service_list = ["OpenAI ChatGPT", "Azure OpenAI"]
    if "selected_chatgpt_service" not in st.session_state:
        st.session_state.selected_chatgpt_service = "OpenAI ChatGPT"
        index = selected_chatgpt_service_list.index(st.session_state.selected_chatgpt_service)
    else:
        if "selected_chatgpt" in st.session_state:
            if st.session_state.selected_chatgpt_service != st.session_state.selected_chatgpt:
                index = selected_chatgpt_service_list.index(st.session_state.selected_chatgpt)
            else:
                index = selected_chatgpt_service_list.index(st.session_state.selected_chatgpt_service)
        else:
            index = selected_chatgpt_service_list.index(st.session_state.selected_chatgpt_service)
    st.session_state.selected_chatgpt_service = st.selectbox(
        '',
        tuple(selected_chatgpt_service_list),
        key="selected_chatgpt",
        index=index,
        label_visibility ="collapsed",
    )
with cols[1]:
    pass

# OpenAI ChatGPTの設定
if st.session_state.selected_chatgpt_service == "OpenAI ChatGPT":

    if "OPENAI_API_KEY" not in st.session_state:
        api_key = os.environ.get('OPENAI_API_KEY')

        if api_key is None:
            st.session_state.OPENAI_API_KEY = ""
            st.session_state.openai_api_key_configured = False
        else:
            st.session_state.OPENAI_API_KEY = api_key
            st.session_state.openai_api_key_configured = True

    cols = st.columns([45,5,25,25])
    with cols[0]:
        st.markdown("- OpenAI API keyの設定")
        openai_api_key_input = st.text_input(
            "[OpenAI API key](https://platform.openai.com/account/api-keys) ",
            type="password",
            placeholder="OpenAI API keyを設定してください (sk-...)",
            help="API keyはこちらから取得できます： https://platform.openai.com/account/api-keys.", 
            value=st.session_state.get("OPENAI_API_KEY", ""),
        )
    with cols[1]:
        pass
    with cols[2]:
        st.markdown("- OpenAIモデルの選択")
        openai_model_name_list = ["gpt-4o-mini", "gpt-4o"]
        if "openai_model_name" not in st.session_state:
            st.session_state.openai_model_name = "gpt-4o-mini"
            index = openai_model_name_list.index(st.session_state.openai_model_name)
        else:
            if "openai_model_selection" in st.session_state:
                if st.session_state.openai_model_name != st.session_state.openai_model_selection:
                    index = openai_model_name_list.index(st.session_state.openai_model_selection)
                else:
                    index = openai_model_name_list.index(st.session_state.openai_model_name)
            else:
                index = openai_model_name_list.index(st.session_state.openai_model_name)
        st.session_state.openai_model_name = st.selectbox(
            'OpenAIモデル',
            tuple(openai_model_name_list),
            key="openai_model_selection",
            index=index,
            help="OpenAIモデル一覧： https://platform.openai.com/docs/models", 
        )

        # OpenAIの場合、モデルが変更された場合は、チェーンを再ロードする
        if 'openai_chain' in st.session_state:
            del st.session_state["openai_chain"]
    with cols[3]:
        pass

    cols = st.columns([20,80])
    with cols[0]:
        st.button("API Keyを設定する", key="openai_key", on_click=set_openai_api_key, args=(openai_api_key_input,))
    with cols[1]:
        st.write("　")
        if not st.session_state.get("openai_api_key_configured"):
            st.error("OpenAI API keyが設定されていません。")
        else:
            st.markdown("OpenAI API Keyは設定されています。")

    cols = st.columns([20,80])
    with cols[0]:
        st.markdown("- メモリバッファ数")
        openai_memory_buffer_k_list = [1, 2, 3, 4, 5, 10]
        if "openai_memory_buffer_k" not in st.session_state:
            st.session_state.openai_memory_buffer_k = 1
            index = openai_memory_buffer_k_list.index(st.session_state.openai_memory_buffer_k)
        else:
            if "openai_memory_buffer_k_selection" in st.session_state:            
                if st.session_state.openai_memory_buffer_k != st.session_state.openai_memory_buffer_k_selection:
                    index = openai_memory_buffer_k_list.index(st.session_state.openai_memory_buffer_k_selection)
                else:
                    index = openai_memory_buffer_k_list.index(st.session_state.openai_memory_buffer_k)
            else:
                index = openai_memory_buffer_k_list.index(st.session_state.openai_memory_buffer_k)
        st.session_state.openai_memory_buffer_k = st.selectbox(
            '',
            tuple(openai_memory_buffer_k_list),
            key="openai_memory_buffer_k_selection",
            index=index,
            label_visibility="collapsed",
        )
    with cols[1]:
        pass

# Azure OpenAIの設定
elif st.session_state.selected_chatgpt_service == "Azure OpenAI":

    if "AZURE_OPENAI_API_KEY" not in st.session_state:
        api_key = os.environ.get('AZURE_OPENAI_API_KEY')

        if api_key is None:
            st.session_state.AZURE_OPENAI_API_KEY = ""
            st.session_state.azure_openai_api_key_configured = False
        else:
            st.session_state.AZURE_OPENAI_API_KEY = api_key
            st.session_state.azure_openai_api_key_configured = True

    cols = st.columns([45,5,40,10])
    with cols[0]:
        st.markdown("- Azure OpenAI API Keyの設定")
        azure_openai_api_key_input = st.text_input(
            "[Azure OpenAI API key](https://portal.azure.com/#browse/resourcegroups) ",
            type="password",
            placeholder="Azure OpenAI API keyを設定してください",
            help="API keyはAzureリソースで設定・取得できます： https://portal.azure.com/#browse/resourcegroups", 
            value=st.session_state.get("AZURE_OPENAI_API_KEY", ""),
        )
    with cols[1]:
        pass
    with cols[2]:
        deployment_name = os.environ.get('AZURE_OPENAI_DEPLOYMENT_NAME')
        st.markdown("- Azure OpenAI デプロイモデル名：" + deployment_name)
        
        # モデルが変更された場合は、チェーンを再ロードする
        if 'azure_openai_chain' in st.session_state:    
            del st.session_state["azure_openai_chain"]
    with cols[3]:
        pass

    cols = st.columns([20,80])
    with cols[0]:
        st.button("API Keyを設定する", key="azure_openai", on_click=set_azure_openai_api_key, args=(azure_openai_api_key_input,))
    with cols[1]:
        st.write("　")
        if not st.session_state.get("azure_openai_api_key_configured"):
            st.error("Azure OpenAI API keyが設定されていません。")
        else:
            st.markdown("Azure OpenAI API Keyは設定されています。")

    cols = st.columns([20,80])
    with cols[0]:
        st.markdown("- メモリバッファ数")
        azure_buffer_memory_k_list = [1, 2, 3, 4, 5, 10]
        if "azure_buffer_memory_k" not in st.session_state:
            st.session_state.azure_buffer_memory_k = 1
            index = azure_buffer_memory_k_list.index(st.session_state.azure_buffer_memory_k)
        else:
            if "azure_buffer_memory_k_selection" in st.session_state:            
                if st.session_state.azure_buffer_memory_k != st.session_state.azure_buffer_memory_k_selection:
                    index = azure_buffer_memory_k_list.index(st.session_state.azure_buffer_memory_k_selection)
                else:
                    index = azure_buffer_memory_k_list.index(st.session_state.azure_buffer_memory_k)
            else:
                index = azure_buffer_memory_k_list.index(st.session_state.azure_buffer_memory_k)
        st.session_state.azure_buffer_memory_k = st.selectbox(
            '',
            tuple(azure_buffer_memory_k_list),
            key="azure_buffer_memory_k_selection",
            index=index,
            label_visibility="collapsed",
        )
    with cols[1]:
        pass