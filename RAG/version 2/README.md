to start:
1. open venv: python -m venv .venv
2. activate venv: .venv/Scripts/activate
3. start Ollama server (if not running): ollama serve
4. start Ollama model: ollama run llama3:8b
5. run app: python run.py
6. kill Ollama server: find PID running on Ollama port 11434 (netstat -aon | findstr 11434), then (taskkill //PID "PID" //F)
7. deactivate venv: deactivate

