FROM ubuntu:latest

WORKDIR /root

# Install system tools, Python 3.12, and OpenBLAS
RUN apt-get update && \
    apt-get install -y software-properties-common curl build-essential git libopenblas-dev && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y python3.12 python3.12-venv python3.12-dev && \
    ln -sf /usr/bin/python3.12 /usr/bin/python3 && \
    curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python3 get-pip.py --break-system-packages && \
    pip config set global.break-system-packages true && \
    rm get-pip.py && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install Python packages system-wide
RUN pip install --no-cache-dir imputegap==1.1.2 jupyter ipywidgets

# Copy current project files into container
COPY ./ /root/

RUN mkdir /root/imputegap_assets

EXPOSE 8888

# Print version and launch Jupyter
CMD ["sh", "-c", "imputegap; jupyter trust /root/*.ipynb; jupyter notebook --notebook-dir=/root --ip=0.0.0.0 --port=8888 --no-browser --allow-root --NotebookApp.token='' --NotebookApp.password=''"]
