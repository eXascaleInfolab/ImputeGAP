############
Installation
############

.. _system-requirements:

System Requirements
~~~~~~~~~~~~~~~~~~~

ImputeGAP is compatible with Python>=3.11 and Unix/WSL environment.

.. _installation:

.. tabs::


    .. tab:: pip

        If you are using ImputeGAP for the first time, the following script installs the required dependencies through a virtual environment.

            .. code-block:: bash

                git clone https://github.com/eXascaleInfolab/ImputeGAP.git
                cd ImputeGAP/
                source imputegap_lib.sh


        Run the following command to install the latest version of ImputeGAP:

        .. code-block:: bash

            pip install imputegap



    .. tab:: source

            To install ImputeGAP directly from the GitHub source in editable mode, clone the repository and run the source script:

            .. code-block:: bash

                git clone https://github.com/eXascaleInfolab/ImputeGAP.git
                cd ImputeGAP/
                source imputegap_source.sh

                pip install -e . # local installation




    .. .. tab:: source

      ..      If you would like to extend the library, you can install from source:

        ..    .. code-block:: bash

          ..      git init
          ..      git clone https://github.com/eXascaleInfolab/ImputeGAP
          ..      cd ./ImputeGAP
          ..      pip install -e .


    .. .. tab:: docker

        .. .. tabs::

        ..    .. tab:: Windows

        ..        Launch Docker from desktop of terminal. To make sure it is running:

        ..        .. code-block:: powershell

        ..             docker version

        ..       Pull the ImputeGAP Docker image:

        ..        .. code-block:: powershell

        ..             docker pull qnater/imputegap:1.1.21

        ..        Run the Docker container:

        ..        .. code-block:: powershell

        ..            docker run -p 8888:8888 qnater/imputegap:1.1.21

        ..        Open the following link:

        ..        .. code-block:: powershell

        ..            http://127.0.0.1:8888


        ..    .. tab:: Linux

        ..        Launch Docker from desktop of terminal. To make sure it is running:

        ..        .. code-block:: powershell

        ..             docker version

        ..        Pull the ImputeGAP Docker image:

        ..        .. code-block:: powershell

        ..            docker pull qnater/imputegap:1.1.21

        ..        Run the Docker container:

        ..        .. code-block:: powershell

        ..             docker run -p 8888:8888 qnater/imputegap:1.1.21

        ..        Open the following link:

        ..        .. code-block:: powershell

        ..           http://127.0.0.1:8888


        ..    .. tab:: MacOS

        ..        Launch Docker from desktop of terminal. To make sure it is running:

        ..        .. code-block:: powershell

        ..             docker version

       ..         Pull the ImputeGAP Docker image:

       ..         .. code-block:: powershell

       ..             docker pull --platform linux/x86_64 qnater/imputegap:1.1.21

       ..         Run the Docker container:

      ..          .. code-block:: powershell

       ..             docker run -p 8888:8888 qnater/imputegap:1.1.21

       ..         Open the following link:

      ..          .. code-block:: powershell

      ..             http://127.0.0.1:8888



Troubleshooting
~~~~~~~~~~~~~~~

If you face any problems, please open an issue here:

https://github.com/eXascaleInfolab/ImputeGAP/issues


