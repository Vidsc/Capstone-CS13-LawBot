# ui_client.py
"""
Gradio UI (frontend) that calls the FastAPI backend via HTTP.
Set API_URL to your backend (default http://localhost:8000).

Run:
  python ui_client.py
"""

import os
import json
import requests
import gradio as gr

API_URL = os.environ.get("API_URL", "http://localhost:8000")

def send_message(session_id: str, message: str, top_k: int = 4):
    url = f"{API_URL}/api/chat"
    payload = {"session_id": session_id, "message": message, "top_k": top_k}
    try:
        resp = requests.post(url, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        # Format sources (if any)
        sources = data.get("sources", [])
        src_text = ""
        if sources:
            src_lines = [f"- {s.get('title')} ({s.get('chunk_id','')})" for s in sources]
            src_text = "Sources:\n" + "\n".join(src_lines)
        return data.get("answer", ""), src_text
    except requests.RequestException as e:
        return f"Backend error: {e}", ""

def upload_file(file_obj):
    if file_obj is None:
        return "Please choose a file."
    url = f"{API_URL}/api/ingest"
    try:
        with open(file_obj.name, "rb") as f:
            files = {"file": (os.path.basename(file_obj.name), f, "application/octet-stream")}
            resp = requests.post(url, files=files, timeout=120)
            resp.raise_for_status()
        return f"Ingested: {os.path.basename(file_obj.name)}"
    except requests.RequestException as e:
        return f"Ingest error: {e}"

with gr.Blocks(title="LawBot UI (Separated)") as demo:
    gr.Markdown("## LawBot — Separated UI/Backend\nThis UI calls the FastAPI backend at `API_URL`.")

    with gr.Row():
        session_id = gr.Textbox(value="demo", label="Session ID")
        top_k = gr.Slider(1, 10, value=4, step=1, label="Top-K")

    with gr.Row():
        msg = gr.Textbox(placeholder="Ask a question…", label="Your question")

    with gr.Row():
        send_btn = gr.Button("Send")
        clear_btn = gr.Button("Clear")

    answer = gr.Textbox(label="Answer", lines=6)
    sources = gr.Textbox(label="Sources", lines=6)

    with gr.Accordion("Ingest a document", open=False):
        file_upl = gr.File(label="Upload a PDF/DOCX/TXT")
        ingest_btn = gr.Button("Ingest")
        ingest_status = gr.Markdown()

    def on_send(m, sid, k):
        a, s = send_message(sid, m, int(k))
        return a, s

    send_btn.click(on_send, inputs=[msg, session_id, top_k], outputs=[answer, sources])
    clear_btn.click(lambda: ("", "", ""), outputs=[answer, sources, msg])

    ingest_btn.click(lambda f: upload_file(f), inputs=[file_upl], outputs=[ingest_status])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
