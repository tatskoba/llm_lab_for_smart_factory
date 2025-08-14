import requests
import streamlit as st

# 対応可能なモデル名をリストで定義
ollama_llm_lists = ["phi4:14b", "phi3.5:3.8b", "llama3.2:3b", "gpt-oss:20b"]
ollama_vlm_lists = ["llava-phi3:3.8b","gemma3:12b","llama3.2-vision:11b"]
ollama_ocr_lists = ["gemma3:12b", "llama3.2-vision:11b"]

def get_ollama_llm_list():
    # ollamaのモデル一覧を取得
    url = "http://ollama:11434/api/tags"
    headers = {
        "Authorization": "Bearer TOKEN123",
        "Content-Type": "application/json"
    }
    response2 = requests.get(url, headers=headers)
    #st.write(response2.json())
    
    res_json = response2.json()

    # 利用可能なモデル名を取得
    llm_available_models = []
    for key in res_json['models']:
        if key['name'] in ollama_llm_lists:
            llm_available_models.append(key['name'])
    vlm_available_models = []
    for key in res_json['models']:
        if key['name'] in ollama_vlm_lists:
            vlm_available_models.append(key['name'])
    ocr_available_models = []
    for key in res_json['models']:
        if key['name'] in ollama_ocr_lists:
            ocr_available_models.append(key['name'])

    return llm_available_models, vlm_available_models, ocr_available_models