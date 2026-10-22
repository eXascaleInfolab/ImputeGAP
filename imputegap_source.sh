#!/usr/bin/env bash

set -e

echo "======================================"
echo " ImputeGAP installation"
echo "======================================"

OS="$(uname -s)"

# ==================================================
# Linux
# ==================================================

if [ "$OS" = "Linux" ]; then

    echo ""
    echo "Linux detected."

    echo ""
    echo "[1/5] Installing system dependencies..."

    sudo apt-get update

    sudo apt install -y \
        build-essential \
        libssl-dev \
        zlib1g-dev \
        libncurses5-dev \
        libncursesw5-dev \
        libreadline-dev \
        libsqlite3-dev \
        libgdbm-dev \
        libdb5.3-dev \
        libbz2-dev \
        libexpat1-dev \
        liblzma-dev \
        tk-dev \
        python3-tk \
        libopenblas0 \
        libarmadillo-dev \
        software-properties-common \
        python3-pip

    echo ""
    echo "[2/5] Adding deadsnakes PPA..."

    sudo add-apt-repository -y ppa:deadsnakes/ppa
    sudo apt-get update

    echo ""
    echo "[3/5] Installing Python 3.12..."

    sudo apt-get install -y \
        python3.12 \
        python3.12-venv \
        python3.12-dev

    PYTHON="python3.12"

# ==================================================
# macOS
# ==================================================

elif [ "$OS" = "Darwin" ]; then

    echo ""
    echo "macOS detected."

    echo ""
    echo "[1/5] Checking Xcode Command Line Tools..."

    if ! xcode-select -p >/dev/null 2>&1; then
        echo "Installing Xcode Command Line Tools..."
        xcode-select --install

        echo ""
        echo "Complete the Xcode Command Line Tools installation"
        echo "and then run this script again."
        exit 0
    fi

    echo ""
    echo "[2/5] Checking Homebrew..."

    if ! command -v brew >/dev/null 2>&1; then
        echo "Installing Homebrew..."

        /bin/bash -c \
            "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

        # Apple Silicon
        if [ -x "/opt/homebrew/bin/brew" ]; then
            eval "$(/opt/homebrew/bin/brew shellenv)"
        # Intel Mac
        elif [ -x "/usr/local/bin/brew" ]; then
            eval "$(/usr/local/bin/brew shellenv)"
        fi
    fi

    echo ""
    echo "[3/5] Installing system dependencies and Python 3.12..."

    brew update

    brew install \
        openssl \
        zlib \
        ncurses \
        readline \
        sqlite \
        gdbm \
        berkeley-db@5 \
        bzip2 \
        expat \
        xz \
        tcl-tk \
        openblas \
        armadillo \
        python@3.12 \
        python-tk@3.12

    PYTHON="$(brew --prefix python@3.12)/bin/python3.12"

# ==================================================
# Unsupported OS
# ==================================================

else

    echo ""
    echo "Unsupported operating system: $OS"
    echo "ImputeGAP currently supports Linux and macOS."
    exit 1

fi

# ==================================================
# Verify Python
# ==================================================

echo ""
echo "Python installation:"
"$PYTHON" --version

# ==================================================
# Virtual environment
# ==================================================

echo ""
echo "[4/5] Creating ImputeGAP virtual environment..."

if [ ! -d "imputegap_env" ]; then
    "$PYTHON" -m venv imputegap_env
else
    echo "imputegap_env already exists."
fi

source imputegap_env/bin/activate

echo ""
echo "Virtual environment:"
echo "  Python: $(python --version)"
echo "  Path:   $(which python)"

# ==================================================
# ImputeGAP (GitHub source)
# ==================================================

echo ""
echo "[5/5] Installing ImputeGAP from GitHub source..."

python -m pip install --upgrade pip
python -m pip install -e .

# ==================================================
# Done
# ==================================================

echo ""
echo "======================================"
echo " ImputeGAP installation completed!"
echo "======================================"
echo ""
echo "Starting ImputeGAP environment..."
echo ""

exec bash --rcfile <(echo "
    source ~/.bashrc 2>/dev/null || true
    source imputegap_env/bin/activate
")

imputegap