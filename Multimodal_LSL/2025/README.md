
#TODO: NEEDS TO BE UPDATED

## 🚀 Getting Started (Everything in `Multimodal_LSL/2025`)

All setup and execution happens inside the `Multimodal_LSL/2025/` folder.

---

### 🐧 Raspberry Pi / Linux

#### 1. Install Python 3.10

Run these commands in the terminal (not Python shell):

```bash
cd /tmp
wget https://www.python.org/ftp/python/3.10.13/Python-3.10.13.tgz
tar -xf Python-3.10.13.tgz
cd Python-3.10.13
./configure --enable-optimizations
make -j$(nproc)
sudo make altinstall
```

Check the version:

```bash
python3.10 --version
```

---

#### 2. Set up the environment

```bash
cd Multimodal_LSL
cd 2025
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 main.py
```

---

### 🪟 Windows

#### 1. Install Python 3.10

- Download the installer:  
  https://www.python.org/downloads/release/python-31013/
- ✅ Be sure to check **“Add Python to PATH”** during installation.
- Install normally.

---

#### 2. Set up the environment

```cmd
cd Multimodal_LSL
cd 2025
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

---

### 🍎 macOS

#### 1. Install Python 3.10 via Homebrew

```bash
brew install python@3.10
brew link --overwrite python@3.10
```

---

#### 2. Set up the environment

```bash
cd Multimodal_LSL/2025
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 main.py
```

