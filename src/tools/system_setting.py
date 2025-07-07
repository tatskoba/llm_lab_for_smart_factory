import os

def setup_env_data():
    
    # Set the system settings
    # data folder
    os.environ['DATA_FOLDER'] = "/workspaces/llm_lab/data/"      
    # model folder
    os.environ['MODEL_FOLDER'] = "/workspaces/llm_lab/models/"          

if __name__ == "__main__":
    setup_env_data()