import os
import time
import glob
from altair import RowColboolean
import pandas as pd
import matplotlib.pyplot as plt
import japanize_matplotlib
from torch import values_copy 
from sqlalchemy import text
import streamlit as st
from st_aggrid import AgGrid, GridUpdateMode
from st_aggrid.grid_options_builder import GridOptionsBuilder

# データ保存フォルダの設定
DATA_FOLDER = os.environ.get('DATA_FOLDER')

# MySQLテーブル名の設定
MYSQL_TABLE_NAME = os.environ['MYSQL_TABLE_NAME']

# ツールの設定ページのURL
TOOL_SETTING_URL = os.environ['TOOL_SETTING_URL']

# 編集データを表に反映
def reflect_edit_data_to_table():

    # 編集中データをテーブルに反映
    st.session_state.grid_table["selected_rows"]["ml_type"] = st.session_state.data_ml_type
    st.session_state.grid_table["selected_rows"]["judge"] = st.session_state.data_judge
    st.session_state.grid_table["selected_rows"]["class"] = st.session_state.data_class
    st.session_state.grid_table["selected_rows"]["description"] = st.session_state.data_description

    # 編集対象データで、st.session_state.data_list_dfのデータを更新
    idx = st.session_state.grid_table["selected_rows"].index[0]
    clm = st.session_state.grid_table["selected_rows"].columns.get_loc("ml_type")
    st.session_state.data_list_df.iat[int(idx),int(clm)] = st.session_state.data_ml_type
    clm = st.session_state.grid_table["selected_rows"].columns.get_loc("judge")
    st.session_state.data_list_df.iat[int(idx),int(clm)] = st.session_state.data_judge
    clm = st.session_state.grid_table["selected_rows"].columns.get_loc("class")
    st.session_state.data_list_df.iat[int(idx),int(clm)] = st.session_state.data_class
    clm = st.session_state.grid_table["selected_rows"].columns.get_loc("description")
    st.session_state.data_list_df.iat[int(idx),int(clm)] = st.session_state.data_description
    
# 指定されたファイル名のファイルが存在するか確認する
def check_file_exist():
    st.write("check_file_exist")
    if os.path.isfile(DATA_FOLDER + "/" + st.session_state.selected_data_folder + "/" + st.session_state.inputed_save_data_list_file):
        st.session_state.new_file_exist = True
    else:
        st.session_state.new_file_exist = False
        st.session_state.selected_save_data_list_file = st.session_state.inputed_save_data_list_file

# 読み込むCSVファイル選択メニューを表示する
def disp_read_csv_data_list():

    if st.session_state.selected_data_folder is not None:
        files = glob.glob(DATA_FOLDER + "/" + st.session_state.selected_data_folder + "/*.csv")
        if len(files) > 0:
            data_list_files = []
            for filepath in files:
                filename = os.path.basename(filepath)
                data_list_files.append(filename)

            if "selected_read_data_list_file" not in st.session_state:
                st.session_state.selected_read_data_list_file = None

            option = st.selectbox(
                "読み込むファイルを選択してください",
                tuple(data_list_files),
                key="selected_read_data_list_file",
                on_change=read_data_from_csv,
            )
        else:
            st.error("データリストファイルが存在しません。")
            return False
    else:
        st.error("データフォルダが選択されていません。")
        return False
    return True

# 保存するCSVファイル選択メニューを表示する
def disp_save_data_to_csv():

    st.radio("", ("既存ファイル", "新規ファイル"), key="how_to_save_csv", horizontal=True, args=[1, 0])

    if st.session_state.how_to_save_csv == "既存ファイル":
        files = glob.glob(DATA_FOLDER + "/" + st.session_state.selected_data_folder + "/*.csv")
        if len(files) > 0:
            data_list_files = []
            for filepath in files:
                filename = os.path.basename(filepath)
                data_list_files.append(filename)

            if "selected_save_data_list_file" not in st.session_state:
                st.session_state.selected_save_data_list_file = None

            option = st.selectbox(
                "保存するファイルを選択してください",
                tuple(data_list_files),
                key="selected_save_data_list_file",
            )
        else:
            st.error("データリストファイルが存在しません")
    elif st.session_state.how_to_save_csv == "新規ファイル":
        new_fname = st.text_input(
                "保存する新規ファイル名を入力してください", 
                value="",
                key="inputed_save_data_list_file",
                on_change=check_file_exist,
            )
        if "new_file_exist" in st.session_state and st.session_state.new_file_exist:
            st.error("そのファイルはすでに存在しています。")
            st.session_state.selected_save_data_list_file = None
        else:
            st.session_state.selected_save_data_list_file = new_fname


# データの編集結果をCSVファイルに保存する
def save_edit_data_to_csv():
    if len(st.session_state.selected_save_data_list_file) > 0:

        # 編集データを表に反映
        reflect_edit_data_to_table()
        
        # CSVファイルを上書き保存する
        st.session_state.data_list_df.to_csv(
            DATA_FOLDER + st.session_state.selected_data_folder + "/" + st.session_state.selected_save_data_list_file, 
            index=False, 
            mode='w'
        )

# データの編集結果をMySQLに保存する
def save_edit_data_to_mysql():
        
    # 編集データを表に反映
    reflect_edit_data_to_table()

    # dada_list_dfをMySQLに保存する
    with st.spinner('MySQLに保存中 ...'):

        # 一度に保存するデータ数
        MAX_COUNT = 50  
        
        # データベースdemo_dbに接続
        conn = st.connection("mysql", type="sql")

        count = 0
        import_data = ""
        for index, row in st.session_state.data_list_df.iterrows():
                
            # MAX＿COUNT行の単位で、データベースにデータを保存
            if len(import_data) > 0:
                import_data += ','     
            import_data += '("%s", "%s", "%s", "%s", "%s", "%s", "%s", "%s", "%s", "%s")' \
                % (row["ml_type"], \
                    row["judge"], \
                    row["class"], \
                    row["description"], \
                    row["date_time"], \
                    row["serial"], \
                    row["stage"], \
                    row["parameter"], \
                    row["csv_file"], \
                    row["image_file"]
                )

            if count == MAX_COUNT:
                table_format = "ml_type,judge,class,description,date_time,serial,stage,parameter,csv_file,image_file"
                sql = 'INSERT INTO ' + MYSQL_TABLE_NAME + ' (' + table_format + ') VALUES ' + import_data \
                    + " ON DUPLICATE KEY UPDATE ml_type = VALUES(ml_type), judge = VALUES(judge), class = VALUES(class), description = VALUES(description);"                   

                with conn.session as s:
                    s.execute(text(sql), params=dict())
                    s.commit()

                count = 1
                import_data = ""           

            count += 1

    # CSVまたは1MySQLからデータを再読み込み
    if st.session_state.read_data_source == "CSV":
        read_data_from_csv()
    elif st.session_state.read_data_source == "MySQL":
        read_data_from_mysql()

# MySQLからデータを読み込む
def read_data_from_mysql():
        
    # データベースdemo_dbに接続
    conn = st.connection("mysql", type="sql")

    # データベースからデータを取得
    sql = "SELECT * from " + MYSQL_TABLE_NAME + ";"
    st.session_state.data_list_df  = conn.query(sql, ttl=0)
    st.session_state.selected_items = None  # 選択されたデータを初期化

# CSVファイルからデータを読み込む
def read_data_from_csv():
    st.session_state.data_list_df = pd.read_csv(DATA_FOLDER + "/" + st.session_state.selected_data_folder + "/" + st.session_state.selected_read_data_list_file)
    st.session_state.selected_items = None

# 画像ファイルパスの取得
def get_image_file_path():

    basename = os.path.splitext(os.path.basename(st.session_state.grid_table["selected_rows"]["csv_file"][0]))[0]

    image_file = basename + ".jpg"

    file_path = os.path.join(DATA_FOLDER, st.session_state.selected_data_folder, "images", st.session_state.grid_table["selected_rows"]["ml_type"][0], st.session_state.grid_table["selected_rows"]["class"][0], image_file)

    return file_path

# 画像ファイルを作成
def make_image_file(fig):

    # 画像ファイルパスの取得
    file_path = get_image_file_path()

    # 画像ファイル出力
    fig.savefig(file_path)

# 選択中のデータファイルの画像ファイルの存在確認（ファイルのパスを返す）
def check_image_file():

    # 画像ファイルパスの取得
    file_path = get_image_file_path()

    if os.path.isfile(file_path):
        return file_path
    else:
        return ""

# 画像データを作成して表示する
def display_image_data(df):

    # X、Yデータを取得
    x = df.iloc[:,0]
    y = df.iloc[:,1]

    # matplotlibでグラフ画像を作成する
    fig = plt.figure(
        figsize=(float(st.session_state.image_fig_size_x), float(st.session_state.image_fig_size_y)),
        dpi=int(st.session_state.image_dpi)
    )
    ax = fig.add_subplot(1,1,1)
    ax.set_xlabel(df.columns[0],fontsize=18)
    ax.set_ylabel(df.columns[1],fontsize=18)     
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=16,) #rotation=45)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=16)
    ax.plot(x,y)

    # 画像を画面表示する
    st.pyplot(fig)

    return fig

# データの編集を表示する
def prap_data():

    cols = st.columns([45,55])
    with cols[0]: 
        if st.session_state.read_data_source == "CSV" and st.session_state.selected_read_data_list_file is None:
            st.error("プルダウンメニューからファイルを選択してください")
            return
    with cols[1]:
        pass

    st.write("データを選択してください")

    # streamlit-aggridでデータフレームを表示
    gd = GridOptionsBuilder.from_dataframe(st.session_state.data_list_df)
    gd.configure_selection(pre_selected_rows=st.session_state.selected_items)
    gridoptions = gd.build()
    st.session_state.grid_table = AgGrid(st.session_state.data_list_df, height=300, gridOptions=gridoptions,
                    update_mode=GridUpdateMode.SELECTION_CHANGED)

    # 画像データfigの初期化
    st.session_state.fig = None

    # 選択されたデータの編集
    if "selected_items" in st.session_state and st.session_state.grid_table["selected_rows"] is not None:

        if "ml_type" not in st.session_state.grid_table["selected_rows"]:
            st.error("データフォーマットが異なるため、このCSVファイルは編集できません。")
            return

        st.write("データを編集してください。")
        cols = st.columns([43,2,53,2])
        with cols[0]:

            # MLタイプの編集
            if st.session_state.grid_table["selected_rows"] is not None:
                txt = st.session_state.grid_table["selected_rows"]["ml_type"][0]
            else:
                if "data_ml_type" in st.session_state:
                    txt =  st.session_state.data_ml_type
                else:
                    txt = ""
            st.text_input(
                "MLタイプ（ml_type）",
                value=txt,
                key="data_ml_type",
            )

            # 判定の編集
            if st.session_state.grid_table["selected_rows"] is not None:
                txt = st.session_state.grid_table["selected_rows"]["judge"][0]
            else:
                if "data_judge" in st.session_state:
                    txt =  st.session_state.data_judge
                else:
                    txt = ""
            st.text_input(
                "判定（judge）",
                value=txt,
                key="data_judge",
            )

            # クラスの編集
            if st.session_state.grid_table["selected_rows"] is not None:
                txt = st.session_state.grid_table["selected_rows"]["class"][0]
            else:
                if "data_class" in st.session_state:
                    txt =  st.session_state["data_class"]
                else:
                    txt = ""
            st.text_input(
                "クラス（class）",
                value=txt,
                key="data_class",
            )

            # データ説明の編集
            if st.session_state.grid_table["selected_rows"] is not None:
                txt = st.session_state.grid_table["selected_rows"]["description"][0]
            else:
                if "data_description" in st.session_state:
                    txt =  st.session_state.data_description
                else:
                    txt =  ""
            st.text_area(
                "データ説明（description）",
                value=txt,
                height = 140,
                key="data_description",
            )

            if "data_list_df" in st.session_state:
                subcols = st.columns([70,30])
                with subcols[0]:
                    # データの保存
                    if st.session_state.save_data_source == "CSV":
                        if st.session_state.selected_save_data_list_file is not None:
                            ret = st.button("編集データを表に反映して保存", on_click=save_edit_data_to_csv)
                            if not ret and len(st.session_state.selected_save_data_list_file) == 0:
                                st.error("保存ファイルを指定してください")
                    elif st.session_state.save_data_source == "MySQL":
                        st.button("編集データを表に反映して保存", on_click=save_edit_data_to_mysql)                    
                with subcols[1]:
                    pass
        with cols[1]:
            pass
        with cols[2]:
            # CSVファイルの選択されたデータリストファイル名を取得
            if st.session_state.read_data_source == "CSV":
                basename = os.path.splitext(os.path.basename(st.session_state.selected_read_data_list_file))[0]
            # MySQLであれば１つだけ存在するテーブル名（MYSQL_TABLE_NAME）を指定する
            elif st.session_state.read_data_source == "MySQL":
                basename = MYSQL_TABLE_NAME

            if st.session_state.grid_table["selected_rows"] is not None:
                st.session_state.csv_data_path = os.path.join(DATA_FOLDER, basename, "csv", st.session_state.grid_table["selected_rows"]["ml_type"][0], st.session_state.grid_table["selected_rows"]["class"][0], st.session_state.grid_table["selected_rows"]["csv_file"][0])

            # 時系列ファイルがあればグラフを画面表示する
            if "csv_data_path" in st.session_state:
                if os.path.isfile(st.session_state.csv_data_path):
                    df = pd.read_csv(st.session_state.csv_data_path)

                    # 画像データを作成して表示する
                    fig = display_image_data(df)
                    
                    # 画像ファイル保存ボタンを表示する
                    st.button("画像ファイルを保存", on_click=make_image_file, args=[fig])   

                    # 画像ファイルの存在確認
                    ret = check_image_file()
                    if len(ret) > 0:
                        st.write("画像ファイルは作成されています（" + ret + "）")
                    else:
                        st.error("画像ファイルは作成されていません")                
                else:
                    st.write("画像ファイルは作成されていません")
        with cols[3]:
            pass

def prep_vision_data():

    st.markdown("#### ローカル画像AI データ準備")
    
    with st.expander("条件設定", expanded=True):

        # データフォルダの選択
        dirlist = []
        for f in os.listdir(DATA_FOLDER):
            if os.path.isdir(os.path.join(DATA_FOLDER, f)):
                dirlist.append(f)
        cols = st.columns([30,3,67])
        with cols[0]:
            if "selected_data_folder" not in st.session_state:
                st.session_state.selected_data_folder = None
            st.session_state.selected_data_folder = st.selectbox('データフォルダの選択', dirlist)
        with cols[1]:
            pass
        with cols[2]:
            st.write("　")
            st.write("　")            
            st.write('選択：`%s`' % st.session_state.selected_data_folder)

        cols = st.columns([25,40,35])
        with cols[0]:
            # データソース選択
            st.radio("データ読込", ("CSV", "MySQL"), key="read_data_source", horizontal=True, args=[1, 0])

            # データソース選択
            st.radio("データ保存", ("CSV", "MySQL"), key="save_data_source", horizontal=True, args=[1, 0])

        with cols[1]:
            status = False
            if st.session_state.read_data_source == "MySQL":
                read_data_from_mysql()
                status = True
            elif st.session_state.read_data_source == "CSV":
                status = disp_read_csv_data_list()

            if st.session_state.save_data_source == "CSV":
                if status:
                    disp_save_data_to_csv()  # 保存するCSVファイル選択メニューを表示する
            elif st.session_state.save_data_source == "MySQL":
                pass

        with cols[2]:
            # 保存する画像ファイルの品質設定
            st.write("保存する画像ファイルの品質設定")
            ret = st.text_input("横サイズ", key="image_fig_size_x", value="6")
            try:
                float(ret)
            except ValueError:
                st.error("数値を入力してください")
            ret = st.text_input("縦サイズ", key="image_fig_size_y", value="4.5")
            try:
                float(ret)
            except ValueError:
                st.error("数値を入力してください")        
            ret = st.text_input("DPI", key="image_dpi", value="35")
            try:
                int(ret)
            except ValueError:
                st.error("正の整数を入力してください")

    # 選択されたデータリストのデータ準備
    if status:
        prap_data()

#-------------------------------------
# 画像データのメイン処理
if "prep_vision_data" in st.session_state:
    prep_vision_data()
else:
    st.error("アプリケーション立上げ時は「[ツールの設定](" + TOOL_SETTING_URL + ")」へアクセスしてください。")
