# Dockerfile

# ベースイメージの指定
FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04

# 基本パッケージのインストール
RUN apt-get update && apt-get install -y python3-pip vim 

# gitとgit-lfsをインストール
RUN apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys E1DD270288B4E6030699E45FA1715D88E1DF1F24
RUN su -c "echo 'deb http://ppa.launchpad.net/git-core/ppa/ubuntu trusty main' > /etc/apt/sources.list.d/git.list"
RUN apt-get install -y git
RUN apt-get install -y git-lfs

# MySQL関連のインストール
RUN apt-get install -y default-libmysqlclient-dev build-essential pkg-config && rm -rf /var/lib/apt/lists/*

# DeepSpeedで必要なパッケージのインストール
RUN apt-get update
RUN apt-get install -y libopenmpi-dev libaio-dev

# CUDA_HOME設定
ENV PATH=$PATH:/usr/local/cuda/bin
RUN echo $PATH
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
ENV CUDA_HOME=/usr/local/cuda

# PyTorchのインストール(for RTX40XX)
RUN pip install --upgrade pip
RUN pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128
# for RTX50XX (nightly版が必要)
# RUN pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128

# ワークスペースのディレクトリを作成
WORKDIR /workspaces/llm_lab

# プロジェクトのディレクトリをコンテナにコピー（モデルファイルなど巨大なファイルは含めないこと）
COPY . .

# Pythonパッケージのインストール
RUN pip install -r requirements.txt

# Flash attentionをインストール
RUN pip install flash-attn --no-build-isolation

# OpenCVのインストール（Time-Zoneで途中停止するため設定を追加している）
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y libopencv-dev

# Streamlitアプリ関連のインストール
ENV HOST=0.0.0.0
ENV LISTEN_PORT=8501
EXPOSE 8501

# コンテナの起動時にbashを実行
CMD ["/bin/bash"]
