# hallucination_eval.py
import argparse, json, re, os
from typing import List, Dict, Any
import numpy as np
import pandas as pd
from tqdm import tqdm

# -----------------------------
# CONFIG: similarity thresholds
# -----------------------------
SIM_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"  # stronger than MiniLM for paraphrases
SIM_THRESHOLD_BINARY = 0.40   # binary 'supported' cut-off (was 0.60; made more tolerant)
TOPK_AVG = 3                  # average top-3 premise similarities per claim
MAX_PREMISES_PER_ITEM = 16    # cap to keep eval fast
MIN_WORDS_PER_CLAIM = 3       # ignore ultra-short sentences

_ST_MODEL = None  # lazy-loaded SentenceTransformer


def _load_st_model():
    global _ST_MODEL
    if _ST_MODEL is None:
        from sentence_transformers import SentenceTransformer
        _ST_MODEL = SentenceTransformer(SIM_MODEL_NAME)
    return _ST_MODEL


def _clean_answer(text: str) -> str:
    """Strip boilerplate that hurts evaluation (e.g., trailing References)."""
    if not text:
        return ""
    # drop everything after a 'References' header (case-insensitive)
    text = re.split(r"\n\s*References\s*\n", text, flags=re.I)[0]
    # normalize whitespace
    text = re.sub(r"[ \t]+", " ", text)
    # collapse multi-bullets into sentences
    text = re.sub(r"[;•\-]\s+", ". ", text)
    return text.strip()


def split_into_claims(text: str) -> List[str]:
    text = _clean_answer(text)
    if not text:
        return []
    # ensure terminal punctuation
    if text and text[-1] not in ".!?。！？；;":
        text += "."
    # split on EN/ZN sentence punctuation
    parts = re.split(r"(?<=[.!?。！？；;])\s+", text)
    claims = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        # too-short sentences are noisy; skip
        if len(p.split()) < MIN_WORDS_PER_CLAIM:
            continue
        claims.append(p)
    return claims


def claim_similarity_score(claim: str, premises: List[str]) -> float:
    """
    Compute a graded 'support' score for one claim:
      - embed claim and premises (mpnet)
      - cosine sim with each premise
      - take mean of top-K similarities as the score
    """
    if not premises:
        return 0.0
    model = _load_st_model()
    from sentence_transformers import util

    # encode
    emb_claim = model.encode(claim, convert_to_tensor=True, normalize_embeddings=True)
    emb_prems = model.encode(premises[:MAX_PREMISES_PER_ITEM],
                             convert_to_tensor=True, normalize_embeddings=True)

    # cosine similarities vector
    sims = util.cos_sim(emb_claim, emb_prems)[0].cpu().numpy().tolist()
    sims.sort(reverse=True)
    topk = sims[:max(1, min(TOPK_AVG, len(sims)))]
    return float(np.mean(topk))  # graded score in [0, 1] (typically ~0.3–0.8)


def get_answer_from_model(question: str) -> Dict[str, Any]:
    """
    Call LawBot and return:
      - the generated answer text
      - REAL text premises from the vector store (not just file/page metadata)
    """
    try:
        # 1) Ask your LawBot (RAG pipeline)
        from app import rag
        result = rag.answer(question, session_id="eval")
        answer_text = (result.get("text") or "").strip()

        # Debug info: confirm retrieval connection
        print(f"[Eval-debug] used_retrieval={result.get('used_retrieval')}, "
              f"score={result.get('score')}, "
              f"citations={len(result.get('citations', []))}")

        # 2) Collect premises: re-run retriever to pull actual chunk text
        from app.vectorstore import search
        from app.config import settings
        k = getattr(settings, "RETRIEVAL_K", 4)
        hits = search(question, k=max(k, 8))  # give the judge more context

        premises = []
        for h in hits:
            txt = (h.get("page_content") or "").strip()
            if txt:
                premises.append(txt)

        return {"answer": answer_text, "contexts": premises}

    except Exception as e:
        print(f"[Error calling LawBot for question: {question[:50]}] -> {e}")
        return {"answer": "", "contexts": []}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problems", required=True, help="JSONL with id, question, support_spans[text]")
    ap.add_argument("--out", default="report.csv")
    ap.add_argument("--use_gold_evidence", action="store_true",
                    help="Use provided support_spans[text] as premises (instead of retriever)")
    ap.add_argument("--details", default=None,
                    help="Optional JSONL path to save per-claim similarity details")
    args = ap.parse_args()

    # load problems
    with open(args.problems, "r", encoding="utf-8") as f:
        items = [json.loads(line) for line in f if line.strip()]

    rows = []
    detail_lines = []

    for item in tqdm(items, total=len(items), desc="Evaluating"):
        qid = item.get("id", "")
        q = item.get("question", "")
        etype = item.get("evaluation_type", "fact")
        gold_spans = [s.get("text", "") for s in item.get("support_spans", []) if s.get("text")]

        resp = get_answer_from_model(q)
        ans = (resp.get("answer") or "").strip()
        retrieved = resp.get("contexts", []) or []

        # choose premises
        if args.use_gold_evidence and gold_spans:
            premises = gold_spans
        else:
            premises = retrieved if retrieved else gold_spans  # fallback to gold if retriever empty

        claims = split_into_claims(ans)

        # If no answer sentences:
        if not claims:
            # for normal fact questions, treat as fully unsupported
            if etype != "refusal":
                rows.append({
                    "id": qid, "evaluation_type": etype, "question": q, "answer": ans,
                    "n_premises": len(premises), "n_claims": 0,
                    "SupportedClaimRate_binary": 0.0,
                    "HallucinationRate_binary": 1.0,
                    "SupportedClaimScore_avg": 0.0,
                    "HallucinationScore": 1.0,
                    "Notes": "No answer sentences."
                })
                continue
            else:
                rows.append({
                    "id": qid, "evaluation_type": etype, "question": q, "answer": ans,
                    "n_premises": len(premises), "n_claims": 0,
                    "SupportedClaimRate_binary": 1.0,
                    "HallucinationRate_binary": 0.0,
                    "SupportedClaimScore_avg": 1.0,
                    "HallucinationScore": 0.0,
                    "Notes": "Refusal with no answer is acceptable."
                })
                continue

        # score per-claim
        claim_scores = []
        claim_supported_flags = []
        for c in claims:
            s = claim_similarity_score(c, premises) if premises else 0.0
            claim_scores.append(s)
            claim_supported_flags.append(1.0 if s >= SIM_THRESHOLD_BINARY else 0.0)

        # aggregate metrics (binary + graded)
        n = len(claims)
        scr_binary = float(np.mean(claim_supported_flags)) if n else 0.0
        hr_binary = 1.0 - scr_binary
        scr_avg = float(np.mean(claim_scores)) if n else 0.0
        hr_avg = 1.0 - scr_avg

        rows.append({
            "id": qid,
            "evaluation_type": etype,
            "question": q,
            "answer": ans,
            "n_premises": len(premises),
            "n_claims": n,
            "SupportedClaimRate_binary": round(scr_binary, 3),
            "HallucinationRate_binary": round(hr_binary, 3),
            "SupportedClaimScore_avg": round(scr_avg, 3),
            "HallucinationScore": round(hr_avg, 3),
        })

        if args.details:
            detail_lines.append({
                "id": qid,
                "claims": claims,
                "premises_preview": [p[:300] for p in premises],
                "claim_scores": [round(s, 4) for s in claim_scores],
                "binary_supported_flags": claim_supported_flags,
            })

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)

    # quick summary
    if len(df) > 0:
        print(df.head())
        print("\n==== Summary ====")
        print("Average HallucinationRate_binary:", round(float(df["HallucinationRate_binary"].mean()), 3))
        print("Average HallucinationScore (graded):", round(float(df["HallucinationScore"].mean()), 3))

    if args.details and detail_lines:
        with open(args.details, "w", encoding="utf-8") as f:
            for d in detail_lines:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
