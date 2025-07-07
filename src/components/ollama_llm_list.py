import requests
import streamlit as st

# 対応可能なモデル名をリストで定義
ollama_llm_lists = ["phi4:latest", "phi3.5:latest", "phi3:latest", "llama3.2:latest"]
ollama_vlm_lists = ["llava:latest", "llava-phi3:latest","gemma3:12b","llama3.2-vision:latest"]

def ollama_llm_list():
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

    return llm_available_models, vlm_available_models