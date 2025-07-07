from PIL import Image
import os
import streamlit as st

# ツールの設定ページのURL
TOOL_SETTING_URL = os.environ['TOOL_SETTING_URL']

# 画像ファイルの分類
def classify_image(uploaded_image_file):

    # 画像を開く
    image = Image.open(uploaded_image_file).convert("RGB")
    
    # 画像をトークナイズ
    inputs = st.session_state.vit_tokenizer(images=image, return_tensors="pt")
    
    # モデルに入力を渡す
    outputs = st.session_state.vit_model(**inputs)
    logits = outputs.logits

    # 予測クラスを取得
    predicted_class_idx = logits.argmax(-1).item()

    # 予測クラスをラベルに変換   
    vit_prediction = st.session_state.vit_model.config.id2label[predicted_class_idx]

    st.write("予測結果:", vit_prediction)
    
def classify_image_main():

    st.markdown("#### Vision Transformerによる画像分類")

    cols = st.columns([50, 50])
    with cols[0]:
        # ファイルをアップロードする
        uploaded_image_file = st.file_uploader("画像ファイル選択", type=["jpg", "jpeg", "png"])

        st.divider()  

        # ViTによる画像分類
        if uploaded_image_file is not None:
            classify_image(uploaded_image_file)

    with cols[1]:
        if uploaded_image_file is not None:
            image_data = Image.open(uploaded_image_file)
            st.image(image_data)

#---------------------------------
# 画像分類のメイン処理
if "vit_model_availability" in st.session_state:
    classify_image_main()            
else:
    st.error("アプリケーション立上げ時は「[ツールの設定](" + TOOL_SETTING_URL + ")」でモデルをロードしてください。")
