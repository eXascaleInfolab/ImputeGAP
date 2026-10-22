<img align="right" width="140" height="140" src="https://raw.githubusercontent.com/eXascaleInfolab/ImputeGAP/main/assets/logo_imputegab.png" >
<br /> <br />


# Installation for ImputeGAP

## System Requirements

The following prerequisites are required to use ImputeGAP:

- Python version **3.12**
- A **Unix-compatible environment** for execution

<br />

Please follow your guide:

[**Windows**](#windows)  | [**Linux**](#linux) | [**MAC OS**](#mac-os)

<br>

---


## Windows

### Install WSL
To run your implementation in a Unix-compatible environment on Windows, we recommend installing **WSL (Windows Subsystem for Linux)**.

1. Check if **WSL** is already installed by typing `WSL` in the search menu.


2. If it is not installed, open **PowerShell** as Administrator (right-click the Start menu and select **Windows PowerShell (Admin)**).


3. Run the following command to install WSL:
   ```powershell
   wsl --install
   ```

4. This will install the latest version of WSL and a default Linux distribution (usually Ubuntu). After the installation is complete, you'll need to restart your computer.
<br><br>
*WSL can be selected in the IDE under the interpreter parameters.*

<br>

### Install Python 3.12

To use **ImputeGAP** effectively, ensure that your environment has **Python** version **3.12.0** to **3.12.7** installed. Follow these steps to install or update Python in your Unix-compatible environment:

<br>

##### Step 1: Check Existing Python Version

Open your terminal and check the currently installed version of Python by running:

```bash
python3 --version
```
<br>

##### Step 2: Install Python 3.12


To install Python 3.12 on your wsl system, you can follow these steps before creating a virtual environment:

1) Open Your WSL from the search engin of Windows


2) Update your package list and install prerequisites:

```
sudo apt-get update
sudo apt install -y build-essential libssl-dev zlib1g-dev libncurses5-dev libncursesw5-dev libreadline-dev libsqlite3-dev libgdbm-dev libdb5.3-dev libbz2-dev libexpat1-dev liblzma-dev tk-dev python3-tk libopenblas0 software-properties-common python3-pip
```

3) Add the deadsnakes PPA (for Ubuntu): This PPA provides newer Python versions for Ubuntu.

```
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt-get update
```

4) Install Python 3.12:

```
sudo apt-get install python3.12 python3.12-venv python3.12-dev
```

5) Verify the installation:
```
python3.12 --version
```

6) Create a virtual environment using Python 3.12:
```
python3.12 -m venv myenv
```

7) Activate the virtual environment:
```
source myenv/bin/activate
```

Now, you are ready to install your project or any dependencies within the Python 3.12 virtual environment.

<br />



<br /><hr /><br />

## LINUX
##### Install Python 3.12

To install Python 3.12 on your system, you can follow these steps before creating a virtual environment:

1) Update your package list and install prerequisites:

```
sudo apt-get update
sudo apt install -y build-essential libssl-dev zlib1g-dev libncurses5-dev libncursesw5-dev libreadline-dev libsqlite3-dev libgdbm-dev libdb5.3-dev libbz2-dev libexpat1-dev liblzma-dev tk-dev python3-tk libopenblas0 software-properties-common python3-pip
```

2) Add the deadsnakes PPA (for Ubuntu): This PPA provides newer Python versions for Ubuntu.

```
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt-get update
```

3) Install Python 3.12:

```
sudo apt-get install python3.12 python3.12-venv python3.12-dev
```

4) Verify the installation:
```
python3.12 --version
```

5) Create a virtual environment using Python 3.12:
```
python3.12 -m venv myenv
```

6) Activate the virtual environment:
```
source myenv/bin/activate
```

Now, you are ready to install your project or any dependencies within the Python 3.12 virtual environment.

<br />

---


## MAC OS
##### Install Python 3.12

To install Python 3.12 on your system, you can follow these steps before creating a virtual environment:

1) Install Homebrew (if not already installed):

```
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

2) Update Homebrew and install Python 3.12:

```
brew update
brew install python@3.12
```

3) Verify the installation:

```
python3.12 --version
```

4) Create a virtual environment using Python 3.12:
```
python3.12 -m venv myenv
```

5) Activate the virtual environment:
```
source myenv/bin/activate
```

Now, you are ready to install your project or any dependencies within the Python 3.12 virtual environment.

<br />