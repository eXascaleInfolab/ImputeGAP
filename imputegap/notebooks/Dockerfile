
FROM python:3.12-slim

WORKDIR /root

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        curl \
        build-essential \
        git \
        libopenblas-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir imputegap==1.1.3 jupyter ipywidgets

# Copy current project files into container
COPY ./ /root/

# Create assets directory
RUN mkdir -p /root/imputegap_assets

EXPOSE 8888

# Print version and launch Jupyter
CMD ["sh", "-c", "imputegap; jupyter trust /root/*.ipynb; jupyter notebook --notebook-dir=/root --ip=0.0.0.0 --port=8888 --no-browser --allow-root --IdentityProvider.token='' --ServerApp.password=''"]
