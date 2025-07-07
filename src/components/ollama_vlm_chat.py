import os
import requests
import base64
import json
from pathlib import Path
import tempfile
import streamlit as st

class ollama_vlm_chat():
    def __init__(self):
        st.session_state.encoded_image_data = None

    # 画像をBase64エンコード
    def encode_image_to_base64(self):
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            st.markdown("## Original image file")
            fp = Path(tmp_file.name)
            fp.write_bytes(st.session_state.uploaded_image_file.getvalue())
        
        with open(tmp_file.name, "rb") as image_file:
            encoded_img = base64.b64encode(image_file.read()).decode('utf-8')
            os.remove(tmp_file.name)
        return encoded_img
    
    # 画像をAPIで解析
    def analyze_image_by_api(self, prompt):
    
        # 画像をBase64エンコード
        if st.session_state.uploaded_image_file is not None:
            st.session_state.encoded_image_data = self.encode_image_to_base64()
        else:
            return "Error: No image file uploaded"

        data = {
            'model': st.session_state.ollama_vlm_model_name,
            'prompt': prompt,
            'images': [st.session_state.encoded_image_data],
        }
  
        try:
            # Dockerのollama APIを呼び出す
            response = requests.post('http://ollama:11434/api/generate',
                                     headers={'Content-Type': 'application/json'},
                                     json=data,
                                     stream=True)

            if response.status_code == 200:
                full_response = ''
                for line in response.iter_lines():
                    if line:
                        json_response = json.loads(line)
                        if 'response' in json_response:
                            full_response += json_response['response']
                return full_response
            else:
                return f"Error: {response.status_code} - {response.text}"
        except requests.ConnectionError as e:
            return f"ConnectionError: {e}"
        