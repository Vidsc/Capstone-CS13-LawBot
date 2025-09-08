import os
from django.shortcuts import render, redirect
from django.views.decorators.http import require_http_methods
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse, HttpResponse
from dotenv import load_dotenv

# 向量检索 & 嵌入
from db import search_by_text_vector
from emb_backend import get_embeddings

import ollama
import uuid

load_dotenv()
EMB = get_embeddings()
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:7b")
# 距离阈值（越小越相似）；超过该阈值或无命中 → 走通用回答
SIM_THRESHOLD = float(os.environ.get("SIM_THRESHOLD", "0.85"))

# —— 提示词 —— #
RAG_SYSTEM_PROMPT = (
    "You are LawBot, a cautious legal assistant focusing on Queensland mining safety Recognised Standards. "
    "Use ONLY the provided CONTEXT to answer. If information is missing, say you cannot find it in the standards. "
    "Be concise and cite sources as [1],[2] based on the numbering of the provided context. Do not fabricate citations."
)

GEN_SYSTEM_PROMPT = (
    "You are LawBot, a careful legal assistant. "
    "When no relevant documents are provided, answer based on your general legal knowledge. "
    "If the question is outside your knowledge, say you are unsure. Be concise and helpful."
)


# --------- 会话历史管理 ---------
def _ensure_sessions(request):
    if "sessions" not in request.session:
        request.session["sessions"] = {}   # {chat_id: {"title": str, "history": list}}
    return request.session["sessions"]


def _current_chat_id(request):
    return request.session.get("current_chat_id")


def _get_current_chat(request):
    sessions = _ensure_sessions(request)
    cid = _current_chat_id(request)
    if cid and cid in sessions:
        return sessions[cid]
    return None


# --------- 页面 ---------
@require_http_methods(["GET"])
def chat_page(request):
    sessions = _ensure_sessions(request)
    cid = _current_chat_id(request)
    if not cid:
        # 如果没有当前会话，自动新建一个
        cid = str(uuid.uuid4())
        sessions[cid] = {"title": "新会话", "history": []}
        request.session["current_chat_id"] = cid
        request.session.modified = True

    current = sessions[cid]
    return render(request, "chat.html", {
        "history": current["history"],
        "sessions": sessions,
        "current_id": cid,
    })


@require_http_methods(["GET"])
def load_chat(request, chat_id):
    sessions = _ensure_sessions(request)
    if chat_id in sessions:
        request.session["current_chat_id"] = chat_id
        request.session.modified = True
        return redirect("chat_page")
    return HttpResponse("会话不存在", status=404)


@csrf_exempt
@require_http_methods(["POST"])
def new_chat(request):
    sessions = _ensure_sessions(request)
    cid = str(uuid.uuid4())
    sessions[cid] = {"title": "新会话", "history": []}
    request.session["current_chat_id"] = cid
    request.session.modified = True
    return redirect("chat_page")


# --------- 消息逻辑 ---------
@csrf_exempt
@require_http_methods(["POST"])
def send_message(request):
    msg = (request.POST.get("message") or "").strip()
    top_k = int(request.POST.get("top_k") or "5")

    sessions = _ensure_sessions(request)
    cid = _current_chat_id(request)
    if not cid:
        return HttpResponse("没有选择会话", status=400)
    chat = sessions[cid]

    if not msg:
        return render(request, "_message.html", {
            "role": "assistant",
            "content": "请先输入问题。",
            "sources": None
        })

    # ---------- 1) 向量检索 ----------
    try:
        qvec = EMB.embed([msg])[0]
        raw_hits = search_by_text_vector(qvec, top_k=top_k * 5)
    except Exception as e:
        return render(request, "_message.html", {
            "role": "assistant",
            "content": f"检索出错：{e}",
            "sources": None
        })

    # 去重
    hits, seen = [], set()
    for r in raw_hits:
        did = r.get("doc_id")
        if did in seen:
            continue
        seen.add(did)
        hits.append(r)
        if len(hits) >= top_k:
            break

    # ---------- 2) 判断是否走 RAG ----------
    use_rag = False
    if hits:
        best_score = hits[0].get("score")
        use_rag = (best_score is None) or (best_score <= SIM_THRESHOLD)

    recent = chat["history"][-8:]

    # ---------- 3A) RAG ----------
    if use_rag:
        ctx_lines = []
        for i, r in enumerate(hits, start=1):
            snippet = (r.get("text") or "").replace("\n", " ").strip()
            ctx_lines.append(f"[{i}] {snippet}")
        context_text = "\n".join(ctx_lines) if ctx_lines else "None."

        messages = [{"role": "system", "content": RAG_SYSTEM_PROMPT}]
        for m in recent:
            messages.append({"role": m["role"], "content": m["content"]})
        messages.append({
            "role": "user",
            "content": (
                f"Question:\n{msg}\n\n"
                f"CONTEXT:\n{context_text}\n\n"
                f"Answer ONLY using the context. Cite as [n]."
            )
        })

        try:
            resp = ollama.chat(model=OLLAMA_MODEL, messages=messages)
            answer = resp["message"]["content"].strip()
        except Exception as e:
            return render(request, "_message.html", {
                "role": "assistant",
                "content": f"Ollama 出错：{e}",
                "sources": None
            })

        chat["history"].append({"role": "user", "content": msg})
        chat["history"].append({"role": "assistant", "content": answer})
        if chat["title"] == "新会话":
            chat["title"] = msg[:15]  # 用第一句话作为标题
        request.session.modified = True

        return render(request, "_message.html", {
            "role": "assistant",
            "content": answer,
            "sources": hits
        })

    # ---------- 3B) 通用回答 ----------
    messages = [{"role": "system", "content": GEN_SYSTEM_PROMPT}]
    for m in recent:
        messages.append({"role": m["role"], "content": m["content"]})
    messages.append({"role": "user", "content": msg})

    try:
        resp = ollama.chat(model=OLLAMA_MODEL, messages=messages)
        answer = resp["message"]["content"].strip()
    except Exception as e:
        return render(request, "_message.html", {
            "role": "assistant",
            "content": f"Ollama 出错：{e}",
            "sources": None
        })

    answer = "（未检索到足够相关的标准，以下为模型通用回答）\n\n" + answer

    chat["history"].append({"role": "user", "content": msg})
    chat["history"].append({"role": "assistant", "content": answer})
    if chat["title"] == "新会话":
        chat["title"] = msg[:15]
    request.session.modified = True

    return render(request, "_message.html", {
        "role": "assistant",
        "content": answer,
        "sources": None
    })


@csrf_exempt
@require_http_methods(["POST"])
def reset_chat(request):
    cid = _current_chat_id(request)
    sessions = _ensure_sessions(request)
    if cid and cid in sessions:
        sessions[cid]["history"] = []
        request.session.modified = True
    return HttpResponse(status=204)
