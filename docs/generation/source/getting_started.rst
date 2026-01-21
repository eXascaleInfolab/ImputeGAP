###############
Getting Started
###############

.. _system-requirements:

System Requirements
~~~~~~~~~~~~~~~~~~~

ImputeGAP is compatible with Python>=3.10 and Unix-compatible environment.

.. _installation:

Installation
~~~~~~~~~~~~


.. tabs::

    .. tab:: pip

        To install/update the latest version of ImputeGAP, run the following command:

        .. code-block:: bash

            pip install imputegap

    .. tab:: venv

        .. raw:: html

           <br>

        .. tabs::

            .. tab:: Windows


                **Install WSL**

                To run ImputeGAP in a Unix-compatible environment on Windows, install the ``Windows Subsystem for Linux (WSL)``.

                1. Search for ``WSL`` in the Start menu to check if it's already installed.

                2. If not, open ``PowerShell`` as ``Administrator``.

                3. Run the following command:

                .. code-block:: bash

                    wsl --install

                4. Restart your computer once the installation completes.


                .. raw:: html

                   <br>


                **Prepare Python 3.12 Environment**

                To ensure a proper Python setup, we recommend creating a dedicated Python environment for the project. Python 3.12 is a suitable and supported choice.

                .. raw:: html

                   <br>

                *Step 1: Check Existing Python Version*

                Open your terminal and check the currently installed version of Python:

                .. code-block:: bash

                    python3 --version


                .. raw:: html

                   <br>

                *Step 2: Install Python 3.12*

                If needed, install Python 3.12 on your WSL system, follow these steps:

                1. Update your package list and install prerequisites:

                .. code-block:: bash

                    sudo apt-get update
                    sudo apt install -y build-essential libssl-dev zlib1g-dev libncurses5-dev libncursesw5-dev \
                    libreadline-dev libsqlite3-dev libgdbm-dev libdb5.3-dev libbz2-dev libexpat1-dev liblzma-dev \
                    tk-dev python3-tk libopenblas0 libarmadillo-dev software-properties-common python3-pip


                3. Add the deadsnakes PPA and update:

                .. code-block:: bash

                    sudo add-apt-repository ppa:deadsnakes/ppa
                    sudo apt-get update


                4. Install Python 3.12:

                .. code-block:: bash

                    sudo apt-get install python3.12 python3.12-venv python3.12-dev


                5. Verify the installation:

                .. code-block:: bash

                    python3.12 --version


                .. raw:: html

                   <br>


                **Install Python 3.12 Environment**

                1. Create a virtual environment:

                .. code-block:: bash

                    python3.12 -m venv imputegap_env

                2. Activate the virtual environment:

                .. code-block:: bash

                    source imputegap_env/bin/activate


                3. Install ImputeGAP

                .. code-block:: bash

                    pip install imputegap


                .. raw:: html

                   <br><br>


            .. tab:: Linux



                **Prepare Python 3.12 Environment**

                To ensure a proper Python setup, we recommend creating a dedicated Python environment for the project. Python 3.12 is a suitable and supported choice.

                .. raw:: html

                   <br>

                *Step 1: Check Existing Python Version**

                Open your terminal and check the currently installed version of Python:

                .. code-block:: bash

                    python3 --version


                .. raw:: html

                   <br>


                *Step 2: Install Python 3.12

                If needed, install Python 3.12 on your system, follow these steps:

                1. Update your package list and install prerequisites:

                .. code-block:: bash

                    sudo apt-get update

                    sudo apt install -y build-essential libssl-dev zlib1g-dev libncurses5-dev libncursesw5-dev \
                    libreadline-dev libsqlite3-dev libgdbm-dev libdb5.3-dev libbz2-dev libexpat1-dev liblzma-dev \
                    tk-dev python3-tk libopenblas0 libarmadillo-dev software-properties-common python3-pip


                3. Add the deadsnakes PPA and update:

                .. code-block:: bash

                    sudo add-apt-repository ppa:deadsnakes/ppa
                    sudo apt-get update


                4. Install Python 3.12:

                .. code-block:: bash

                    sudo apt-get install python3.12 python3.12-venv python3.12-dev


                5. Verify the installation:

                .. code-block:: bash

                    python3.12 --version


                .. raw:: html

                   <br>


                **Install Python 3.12 Environment**

                1. Create a virtual environment:

                .. code-block:: bash

                    python3.12 -m venv imputegap_env


                2. Activate the virtual environment:

                .. code-block:: bash

                    source imputegap_env/bin/activate


                3. Install ImputeGAP

                .. code-block:: bash

                    pip install imputegap



                .. raw:: html

                   <br><br>




            .. tab:: MacOS


                **Prepare Python 3.12 Environment**

                To ensure a proper Python setup, we recommend creating a dedicated Python environment for the project. Python 3.12 is a suitable and supported choice.

                1. Install Xcode Command Line Tools (required by Homebrew):

                .. code-block:: bash

                    xcode-select --install


                2. Install Homebrew (if not already installed):

                .. code-block:: bash

                    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"


                3. Update Homebrew, install prerequisites, and install Python 3.12:

                .. code-block:: bash

                    brew update
                    brew install openssl zlib ncurses readline sqlite gdbm berkeley-db@5 bzip2 expat xz tcl-tk openblas armadillo python@3.12


                4. Verify the installation:

                .. code-block:: bash

                    python3.12 --version


                .. raw:: html

                   <br>


                **Install Python 3.12 Environment**

                1. Create a virtual environment:

                .. code-block:: bash

                    python3.12 -m venv imputegap_env


                2. Activate the virtual environment:

                .. code-block:: bash

                    source imputegap_env/bin/activate


                3. Install ImputeGAP

                .. code-block:: bash

                    pip install imputegap


                4. ⚠️ Warning

                Depending on your macOS security settings, you may need to manually approve the C++ wrapper library (`.dylib` file) in **System Settings → Privacy & Security** before it can be loaded.


    .. tab:: source

            If you would like to extend the library, you can install from source:

            .. code-block:: bash

                git init
                git clone https://github.com/eXascaleInfolab/ImputeGAP
                cd ./ImputeGAP
                pip install -e .


    .. tab:: docker

        .. tabs::

            .. tab:: Windows

                Launch Docker from desktop of terminal. To make sure it is running:

                .. code-block:: powershell

                     docker version

                Pull the ImputeGAP Docker image:

                .. code-block:: powershell

                     docker pull qnater/imputegap:1.1.2

                Run the Docker container:

                .. code-block:: bash

                    docker run -p 8888:8888 qnater/imputegap:1.1.2

                Open the following link:

                .. code-block:: powershell

                    http://127.0.0.1:8888


            .. tab:: Linux

                Launch Docker from desktop of terminal. To make sure it is running:

                .. code-block:: powershell

                     docker version

                Pull the ImputeGAP Docker image:

                .. code-block:: bash

                    docker pull qnater/imputegap:1.1.2

                Run the Docker container:

                .. code-block:: bash

                    docker run -p 8888:8888 qnater/imputegap:1.1.2

                Open the following link:

                .. code-block:: powershell

                    http://127.0.0.1:8888


            .. tab:: MacOS

                Launch Docker from desktop of terminal. To make sure it is running:

                .. code-block:: powershell

                     docker version

                Pull the ImputeGAP Docker image:

                .. code-block:: bash

                    docker pull --platform linux/x86_64 qnater/imputegap:1.1.2

                Run the Docker container:

                .. code-block:: bash

                    docker run -p 8888:8888 qnater/imputegap:1.1.2

                Open the following link:

                .. code-block:: powershell

                    http://127.0.0.1:8888



Troubleshooting
~~~~~~~~~~~~~~~

If you face any problems, please open an issue here: https://github.com/eXascaleInfolab/ImputeGAP/issues


