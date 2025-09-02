import os, ollama
model = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
print("model:", model)
resp = ollama.chat(model=model, messages=[{"role":"user","content":"hello"}])
print(resp["message"]["content"][:200])
