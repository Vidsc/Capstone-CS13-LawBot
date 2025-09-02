import os
from django.shortcuts import render
from django.views.decorators.http import require_http_methods
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse, HttpResponse
from dotenv import load_dotenv

# 向量检索 & 嵌入
from db import search_by_text_vector
from emb_backend import get_embeddings

import ollama

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

def _ensure_history(request):
    if "history" not in request.session:
        request.session["history"] = []
    return request.session["history"]

@require_http_methods(["GET"])
def chat_page(request):
    history = _ensure_history(request)
    return render(request, "chat.html", {"history": history})

@csrf_exempt
@require_http_methods(["POST"])
def send_message(request):
    """
    先检索 → 命中(且相关)则 RAG；否则退回模型通识回答。
    总是返回一段 HTML（机器人气泡）。
    """
    msg = (request.POST.get("message") or "").strip()
    top_k = int(request.POST.get("top_k") or "5")
    if not msg:
        return render(request, "_message.html", {
            "role": "assistant",
            "content": "请先输入问题。",
            "sources": None
        })

    # ---------- 1) 向量检索（先多取，后去重） ----------
    try:
        qvec = EMB.embed([msg])[0]
        raw_hits = search_by_text_vector(qvec, top_k=top_k * 5)  # 先取更大候选
    except Exception as e:
        return render(request, "_message.html", {
            "role": "assistant",
            "content": f"检索出错：{e}",
            "sources": None
        })

    # 去重：每个 doc_id 保留一条（全局相似度顺序不变）
    hits, seen = [], set()
    for r in raw_hits:
        did = r.get("doc_id")
        if did in seen:
            continue
        seen.add(did)
        hits.append(r)
        if len(hits) >= top_k:
            break

    # ---------- 2) 是否走 RAG ----------
    use_rag = False
    if hits:
        best_score = hits[0].get("score")  # 注意：db.search_by_text_vector 需 SELECT (embedding <-> %s::vector) AS score
        use_rag = (best_score is None) or (best_score <= SIM_THRESHOLD)
    else:
        use_rag = False

    # 历史
    history = request.session.get("history", [])
    recent = history[-8:]

    # ---------- 3A) RAG 路径 ----------
    if use_rag:
        # 组装 CONTEXT（带编号）
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

        # 写历史 & 返回（含来源）
        history.append({"role": "user", "content": msg})
        history.append({"role": "assistant", "content": answer})
        request.session["history"] = history[-20:]
        request.session.modified = True

        return render(request, "_message.html", {
            "role": "assistant",
            "content": answer,
            "sources": hits  # 前端会显示“参考来源”
        })

    # ---------- 3B) 通用回答路径（未命中或相似度低） ----------
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

    # 可选：提示此次为通用回答
    answer = "（未检索到足够相关的标准，以下为模型通用回答）\n\n" + answer

    history.append({"role": "user", "content": msg})
    history.append({"role": "assistant", "content": answer})
    request.session["history"] = history[-20:]
    request.session.modified = True

    return render(request, "_message.html", {
        "role": "assistant",
        "content": answer,
        "sources": None  # 通用回答不显示来源
    })

@csrf_exempt
@require_http_methods(["POST"])
def reset_chat(request):
    """清空会话并返回 204，让前端不再等待内容"""
    request.session["history"] = []
    request.session.modified = True
    return HttpResponse(status=204)
