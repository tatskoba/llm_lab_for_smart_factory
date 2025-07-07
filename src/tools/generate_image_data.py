# 時系列データ異常検知用サンプルデータ生成スクリプト
#
# データセット： 工程_パラメータ_OK/NGラベルの時系列データ画像
#    工程：stage1, stage2, stage3
#    各工程にパラメータが2種類、OK/NGが2種類の時系列データがある
#
# データセットに含まれるフォルダ構造は以下の通り。
# time_series_data
#    time_series_data.csv ・・・　データリストファイル
#    time_series_train.jsonl ・・・　phi3 visionのLoRAファインチューニング用訓練データjsonlファイル
#    time_series_test.jsonl ・・・　phi3 visionのLoRAファインチューニング用テストデータjsonlファイル
#    time_series_val.jsonl ・・・　phi3 visionのLoRAファインチューニング用評価データjsonlファイル
#    images  ・・・　画像データファイル
#       train
#          stage_1_param_1_ok  ・・・　この階層がラベルを示す
#          stage_1_param_1_ng
#          stage_1_param_2_ok
#          stage_1_param_2_ng
#          stage_2_param_1_ok
#          stage_2_param_1_ng
#          stage_2_param_2_ok
#          stage_2_param_2_ng
#          stage_3_param_1_ok
#          stage_3_param_1_ng
#          stage_3_param_2_ok
#          stage_3_param_2_ng
#       test
#          stage1_param1_ok
#           ... 
#       val
#          stage1_param1_ok
#           ... 
#   csv　・・・　時系列データのCSVファイル
#       train
#         ... (上と同じ)
#       test
#         ... (上と同じ)
#       val
#         ... (上と同じ)

import os
import random
import datetime
import matplotlib.pyplot as plt
import japanize_matplotlib 
import glob

from system_setting import setup_env_data

NUM_OF_SERIAL = 35    # シリアル番号最大数
#NUM_OF_SERIAL = 16
NUM_OF_TRAIN_SPLIT = 0.8   # 学習セットの分割のための数
#NUM_OF_TRAIN_SPLIT = 0.6
NUM_OF_TEST_SPLIT = 0.1   # テストセットの分割のための数
#NUM_OF_TEST_SPLIT = 0.2
FILE_NUM_PER_ID = 5       # phi3 vision LoRAファインチューニング用の一つのIDあたりの画像ファイル数
#FILE_NUM_PER_ID = 4

# グラフデータを生成する関数
def generate_data(stage, param, status):
    x = range(0, 21, 1)

    if stage == 1 and param == 1:
        # 1: 工程1 パラメータ1
        y = [0, 1, 0, 1, 0, 1, 1, 4, 9, 15, 15, 15, 15, 15, 15, 1, 0, 1, 0, 1,0]
        if status == "NG":
            y[12] = y[12] - 6
            y[13] = y[13] - 6
            desc = "時系列データは異常です。途中で値が大きく低下しています。"
        else:
            desc = "時系列データは正常です。"
    elif stage == 1 and param == 2:
        # 2: 工程1 パラメータ2
        y = [0, 1, 3, 2, 4, 5, 4, 6, 5, 4, 6, 4, 4, 7, 8, 6, 5, 6, 7, 6, 6]
        if status == "NG":
            for i in range(12, 20):
                y[i] = y[i] + 9
            desc = "時系列データは異常です。途中で値が大きく上昇しています。"
        else:
            desc = "時系列データは正常です。"
    elif stage == 2 and param == 1:
        # 3: 工程2 パラメータ1
        y = [0, 5, 12, 6, -1, -6, -13, -5, 1, 5, 12, 6, -1, -6, -13, -4, 1, 6, 13, 6, -1]
        if status == "NG":
            y[17] = 0 
            y[18] = 0 
            y[19] = 0 
            y[20] = 0
            desc = "時系列データは異常です。途中から値が0になっています。"
        else:
            desc = "時系列データは正常です。"
    elif stage == 2 and param == 2:
        # 4: 工程2 パラメータ2
        y = [6, -1, -6, -13, -5, 1, 5, 12, 6, -1, -6, -13, -4, 1, 6, 13, 6, -1, -6, -13, -5]
        if status == "NG":
            y[12] = 25
            desc = "時系列データは異常です。途中で値が突然大きく上昇しています。"   
        else:
            desc = "時系列データは正常です。"
    elif stage == 3 and param == 1:
        # 5: 工程3 パラメータ1
        y = [0, -1, -2, -3, -2, 14, 17, 15, 16, 15, 16, 14, 15, 13, 9, 3, 1, 0, 1, 4, 1]
        if status == "NG":
            y[10] = -11
            y[11] = 23
            y[12] = -10
            desc = "時系列データは異常です。途中で値が大きく変動しています。"
        else:
            desc = "時系列データは正常です。"
    elif stage == 3 and param == 2:
        # 6: 工程3 パラメータ2
        y = [0, -1, -4, -3, -2, 7, 15, 18, 18, 19, 19, 19, 19, 19, 19, 13, 10, 9, 4, 1, 0]
        if status == "NG":
            y[10] = -5
            y[11] = -6
            desc = "時系列データは異常です。途中で値が大きく低下しています。"
        else:
            desc = "時系列データは正常です。"

    # 最大プラスマイナス2の一葉乱数を加える
    y = list(map(lambda x: x+random.uniform(-1, 1), y))

    return x, y, desc


# データファイルを生成
def generate_datafile(DATA_FOLDER):
    if not os.path.isdir(DATA_FOLDER + "time_series_data"):
        os.makedirs(DATA_FOLDER + "time_series_data")
    if not os.path.isdir(DATA_FOLDER + "time_series_data/images"):
        os.makedirs(DATA_FOLDER + "time_series_data/images")
    if not os.path.isdir(DATA_FOLDER + "time_series_data/csv"):
        os.makedirs(DATA_FOLDER + "time_series_data/csv")

    # データリストファイル名
    data_list_file = DATA_FOLDER + "time_series_data/time_series_data.csv"

    # データリストファイルのヘッダーの定義
    def_data_list_file_header = "ml_type,judge,class,description,date_time,serial,stage,parameter,csv_file,image_file"

    # データリストファイルを作成
    with open(data_list_file, "w") as f0:
        # ヘッダーを書き込む
        f0.write(def_data_list_file_header + "\n")

        # 日時の初期値を設定
        date_time = datetime.datetime(2024, 12, 1, 8, 0)

        # データの説明を保存する辞書を初期化
        img_file_info = {}

        # グラフデータを生成  
        for serial in range(1, NUM_OF_SERIAL+1):  # 各ラベルで複数のグラフを作成
            for stage in range(1, 4):
                for param in range(1, 3):
                    for status in ["OK", "NG"]:

                        # 製造時刻
                        date_time = date_time + datetime.timedelta(minutes=1)
                        dt = date_time.strftime('%Y/%m/%d %H:%M:%S')

                        # 時系列データとデータ説明
                        x, y, desc = generate_data(stage, param, status)

                        # DPIとピクセル数を指定してfigureを生成
                        fig = plt.figure(figsize=(10, 8), dpi=50)
                        ax = fig.add_subplot(1,1,1)
                        ax.plot(x, y, '-b')
                        ax.set_xlabel('時間',fontsize=24)
                        ax.set_ylabel('サンプル値',fontsize=24)

                        # 目盛りの位置を固定
                        ax.set_xticks(range(0, 25, 5))
                        ax.set_yticks(range(-20, 40, 10))

                        # 目盛りラベルの設定
                        ax.set_xticklabels(ax.get_xticks(), fontsize=18,) #rotation=45)
                        ax.set_yticklabels(ax.get_yticks(), fontsize=18)

                        # グラフの範囲を設定
                        ax.set_xlim(0,20)
                        ax.set_ylim(-20, 30)

                        # 補助線をグリッド表示
                        ax.vlines([5, 10, 15], -20, 30, linestyles='dashed') 
                        ax.hlines([-10, 0, 10, 20, 30], 0, 20, linestyles='dashed') 

                        title = "工程" + str(stage) + " パラメータ" + str(param)
                        ax.set_title(title,fontsize=30)
                        #ax.legend(fontsize=24) 

                        plt.subplots_adjust(left=0.15, right=0.9, bottom=0.15, top=0.9)

                        # 保存フォルダを分ける（train, test, validation）
                        if serial < ((NUM_OF_SERIAL+1) * NUM_OF_TRAIN_SPLIT):  # trainデータとして保存
                            ml_type = "train"
                        elif serial >= ((NUM_OF_SERIAL+1) * NUM_OF_TRAIN_SPLIT) and serial < ((NUM_OF_SERIAL+1) * (NUM_OF_TRAIN_SPLIT+NUM_OF_TEST_SPLIT)):  # testデータとして保存
                            ml_type = "test"
                        else: # validationデータとして保存
                            ml_type = "val"

                        # 画像フォルダのパス
                        class_name =  "stage_" + str(stage) + "_param_" + str(param) + "_" + status
                        image_save_folder = DATA_FOLDER + "time_series_data/images/" + ml_type + "/" + class_name

                        if not os.path.isdir(image_save_folder):
                            os.makedirs(image_save_folder)

                        # 画像ファイルを出力
                        image_fname = "serial_" + str(serial).zfill(2) + "_stage_" + str(stage) + "_param_" + str(param) + "_" + status + ".jpg"
                        plt.savefig(image_save_folder + "/" + image_fname)
                        plt.clf()
                        plt.close()

                        # データ説明を保存（キー：画像ファイル名）
                        img_file_info[image_fname] = {"serial": serial, "stage": stage, "param": param, "status": status, "desc": desc}

                        # 時系列データCSVフォルダのパス
                        csv_save_folder = DATA_FOLDER + "time_series_data/csv/" + ml_type + "/stage_" + str(stage) + "_param_" + str(param) + "_" + status

                        if not os.path.isdir(csv_save_folder):
                            os.makedirs(csv_save_folder)                       

                        # 時系列データCSVファイルを出力
                        csv_fname = "serial_" + str(serial).zfill(2) + "_stage_" + str(stage) + "_param_" + str(param) + "_" + status + ".csv"
                        with open(csv_save_folder + "/" + csv_fname, "w") as f_data_file:
                            f_data_file.write("x,y\n")
                            for i in range(0,len(x)):
                                f_data_file.write(str(x[i]) + "," + str(y[i]) + "\n" )

                         # 一覧ファイルにデータ追加
                        f0.write(ml_type + "," + status + "," + class_name + "," + desc + "," + dt + "," + str(serial).zfill(2) + "," + str(stage) + "," + str(param) + "," + csv_fname + "," + image_fname + "\n")  

    # データの説明を返す
    return img_file_info


# phi3 visionのLoRAファインチューニング用jsonlファイルを作成
# 作成するフォーマットは、phi-3 CookBookのサンプルスクリプトが出力するデータフォーマットに従う
# https://github.com/microsoft/Phi-3CookBook/blob/main/code/04.Finetuning/vision_finetuning/convert_ucf101.py
# 以下は、サンプルスクリプトのデータフォーマットの一部（1学習単位）
'''
{"id": "test-0000000339", "source": "ucf101", "conversations": [{"images": ["test/ApplyLipstick/v_ApplyLipstick_g14_c01.0.jpg", "test/ApplyLipstick/v_ApplyLipstick_g14_c01.1.jpg", "test/ApplyLipstick/v_ApplyLipstick_g14_c01.2.jpg", "test/ApplyLipstick/v_ApplyLipstick_g14_c01.3.jpg", "test/ApplyLipstick/v_ApplyLipstick_g14_c01.4.jpg", "test/ApplyLipstick/v_ApplyLipstick_g14_c01.5.jpg", "test/ApplyLipstick/v_ApplyLipstick_g14_c01.6.jpg", "test/ApplyLipstick/v_ApplyLipstick_g14_c01.7.jpg"], "user": "Classify the video into one of the following classes: ApplyEyeMakeup, ApplyLipstick, Archery, BabyCrawling, BalanceBeam, BandMarching, BaseballPitch, Basketball, BasketballDunk, BenchPress.", "assistant": "ApplyLipstick"}]}
'''
def make_phi3v_lora_jsonl(DATA_FOLDER, img_file_info):

    # フォルダ定義
    folder_def = ["train", "test", "val"]
    learning_data = DATA_FOLDER + "time_series_data"

    # 学習タイプ別フォルダ３つを個別に処理（train, test, val）
    for folder in folder_def:

        # 複数画像ファイルセットIDの初期値を設定（学習タイプ別にIDを作成）
        id = 1

        # 学習タイプ別にLoRAファインチューニング用jsonlファイルを作成
        fname = learning_data + "/time_series_" + folder + ".jsonl"

        with open(fname, "w") as f:

            # ラベル別のパスリストを取得（ソートしたリスト）
            label_list = sorted(glob.glob(learning_data + "/images/" + folder + "/*"))
            #print(label_list)

            # ラベル毎にその下の画像ファイル一覧を取得し、複数画像ファイルをセットにしてIDを付与し、セット単位でjsonlファイルに書き込む
            for path in label_list:
                images = glob.glob(path + "/*.jpg")
                class_name = os.path.basename(path)
                files = sorted(list(map(lambda x: os.path.basename(x), images)))
                image_file_set = []
                for file in files:
                    image_file_set.append(folder + "/" + class_name + "/" + file)

                # 画像ファイルをセットにしてIDを付与して、jsonlファイルに書き込む
                #print(str(len(image_file_set)), image_file_set)
                for i in range(0, len(image_file_set), FILE_NUM_PER_ID):
                    # jsolファイルへの書き込むデータを作成
                    id_text_data = "{\"id\": \"" + folder + "-" + str(id).zfill(10) + "\", \"source\": \"time_series_data\", \"conversations\": [{\"images\": ["       
                    
                    for j in range(i, min(i+FILE_NUM_PER_ID-1,len(image_file_set)-1)):
                        if j != i:
                            id_text_data += ","
                        id_text_data += "\"" + image_file_set[j] + "\""

                    # ファイルセットの最初のファイルのデータの説明を取得
                    file1_name = os.path.basename(image_file_set[i])
                    reason = img_file_info[file1_name]["desc"]

                    # パターン１（カテゴリー）
                    id_text_data += "], \"user\": \"この時系列データは次のどのカテゴリーにあてはまりますか: stage_1_param_1_NG, stage_1_param_1_OK, stage_1_param_2_NG, stage_1_param_2_OK, stage_2_param_1_NG, stage_2_param_1_OK, stage_2_param_2_NG, stage_2_param_2_OK, stage_3_param_1_NG, stage_3_param_1_OK, stage_3_param_2_NG, stage_3_param_2_OK\", \"assistant\": \"" + class_name + "\"}]}\n"

                    # パターン２（カテゴリーと理由）
                    #id_text_data += "], \"user\": \"この時系列データは次のどのカテゴリーにあてはまりますか。カテゴリーとその理由を教えてください : stage_1_param_1_NG, stage_1_param_1_OK, stage_1_param_2_NG, stage_1_param_2_OK, stage_2_param_1_NG, stage_2_param_1_OK, stage_2_param_2_NG, stage_2_param_2_OK, stage_3_param_1_NG, stage_3_param_1_OK, stage_3_param_2_NG, stage_3_param_2_OK\", \"assistant\": \"カテゴリ : " + class_name + "、理由: " + reason + "\"}]}\n"

                    # jsonlファイルに書き込む
                    f.write(id_text_data)

                    # IDをインクリメント
                    id += 1

#------------------------
if __name__ == "__main__":

    # set the system settings
    setup_env_data()
    DATA_FOLDER = os.environ['DATA_FOLDER']

    # データファイルを生成
    img_file_info = generate_datafile(DATA_FOLDER)

    # phi3 visionのLoRAファインチューニング用jsonlファイルを作成
    make_phi3v_lora_jsonl(DATA_FOLDER, img_file_info)


