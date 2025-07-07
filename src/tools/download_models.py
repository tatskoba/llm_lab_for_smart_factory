import os
import readchar
from huggingface_hub import snapshot_download
from system_setting import setup_env_data

main_menu = '''
Press keys on keyboard to download the model!
    1: intfloat/multilingual-e5-large
    2: microsoft/Phi-3.5-vision-instruct
    3: cyberagent/llava-calm2-siglip
    4: microsoft/swinv2-tiny-patch4-window8-256
    5: ds4sd/SmolDocling-256M-preview
    ctrl+c: exit the tool
'''

def show_menu():
    print("\033[H\033[J",end='') 
    print(main_menu)

if __name__ == "__main__":

    # Get the system settings
    setup_env_data()
    MODEL_FOLDER = os.environ.get('MODEL_FOLDER')

    try:
        # Show the main menu
        show_menu()

        while True:
            key = readchar.readchar()
            key = key.lower()
            if key in('12345'):
                if key == '1':
                    model_card = "intfloat/multilingual-e5-large"
                    local_path = MODEL_FOLDER + "multilingual-e5-large"
                elif key == '2':
                    model_card = "microsoft/Phi-3.5-vision-instruct"
                    local_path = MODEL_FOLDER + "phi35v"
                elif key == '3':
                    model_card = "cyberagent/llava-calm2-siglip"
                    local_path = MODEL_FOLDER + "llava-calm2-siglip"
                elif key == '4':
                    model_card = "microsoft/swinv2-tiny-patch4-window8-256"
                    local_path = MODEL_FOLDER + "swinv2"
                elif key == '5':
                    model_card = "ds4sd/SmolDocling-256M-preview"
                    local_path = MODEL_FOLDER + "smoldocling"
                print("***********************")

                # Create the local directory if it does not exist
                if not os.path.isdir(local_path):
                    os.makedirs(local_path)
                    print("Downloading the model: " + model_card + " ...")
                else:
                    print("The specified model may exist already. Please confirm it.")
                    input("Press Enter to continue...")

                # Dwonload the model
                download_path = snapshot_download(
                    repo_id=model_card,
                    local_dir=local_path,
                )
                print("The model downloading is done!")
                print(" - Model name:", model_card)
                print(" - Downloaded path:", download_path)
                
                show_menu()

            # Exit the tool
            elif key == readchar.key.CTRL_C:
                break
    finally:
        print("\n Bye")