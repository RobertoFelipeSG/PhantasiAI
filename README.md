# 🧠 PhantasiAI

**PhantasiAI** is a multimodal AI research prototype integrating live streaming data, RAG-based information retrieval, and cognitive architecture components.

> ⚠️ *TODO: Add a short description*

---

## 🗂️ Project Structure

```
PhantasiAI/
├── .venv/                  # Python virtual environment (DO NOT COMMIT)
├── Docs/                   # Project documentation
├── Multimodal_LSL/         # Multimodal Lab Streaming Layer components
├── RAG/                    # Retrieval-Augmented Generation module
├── requirements.txt        # List of Python dependencies
└── README.md               # This file
```

---

## 🚀 Getting Started (Python 3.10 + venv)

### 🐧 For Raspberry Pi (or Linux)

1. **Install Python 3.10** (if not already installed):

   ```bash
   sudo apt update && sudo apt upgrade -y
   sudo apt install -y make build-essential libssl-dev zlib1g-dev \
     libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm \
     libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev \
     libffi-dev liblzma-dev

   cd /tmp
   wget https://www.python.org/ftp/python/3.10.13/Python-3.10.13.tgz
   tar -xf Python-3.10.13.tgz
   cd Python-3.10.13
   ./configure --enable-optimizations
   make -j$(nproc)
   sudo make altinstall
   ```

2. **Clone the repository**:

   ```bash
   git clone https://github.com/your-username/PhantasiAI.git
   cd PhantasiAI
   ```

3. **Create and activate the virtual environment**:

   ```bash
   python3.10 -m venv .venv
   source .venv/bin/activate
   ```

4. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

---

## 🧪 Running the Prototype

1. **Activate the virtual environment**:

   ```bash
   source .venv/bin/activate
   ```

2. **Navigate to the application directory**:

   ```bash
   cd ../..  
   cd Multimodal_LSL/2025
   ```

3. **Run the application**:

   ```bash
   python3 main.py
   ```

---

## 📚 Documentation

The documentation is available in the `Docs/` folder. It currently includes (and will in the future):

- **Class diagram**: A visual representation of the classes and their relationships.
- **Use cases**: Detailed descriptions of the use cases for the application.
- **SRS**: Software Requirements Specification document outlining the functional and non-functional requirements.

---


