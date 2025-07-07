# tools/azure_openai_api_example.py: call Azure OpenAI API 
# reference: https://learn.microsoft.com/ja-jp/azure/ai-services/openai/concepts/advanced-prompt-engineering

import os
import openai

####################################################
# Please set Azure OpenAI API key to the environment variable

def check_azure_openai_api_key():
    if os.environ.get('AZURE_OPENAI_API_KEY') is None:
        print("Please set the Zzure OpenAI API key to the environment variable: OPENAI_API_KEY")
        exit()
####################################################

if __name__ == "__main__":

    # set your API info
    check_azure_openai_api_key()
    openai.api_type = "azure"
    openai.api_base = os.getenv("AZURE_OPENAI_ENDPOINT")
    openai.api_version = os.getenv("AZURE_OPENAI_API_VERSION")
    DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")

    response = openai.chat.completions.create(
        model=DEPLOYMENT_NAME,
        messages = [
            {"role":"system","content":"あなたは、人々が情報を見つけるのを助け、韻を踏んで応答する AI アシスタントです。 ユーザーが答えがわからない質問をした場合は、その旨を伝えます。"},
            {"role":"user","content":"こんにちは！"}],
        temperature=0.7,
        max_tokens=800
    )

    print(response.choices[0].message.content)