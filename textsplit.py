import re

HEADING_RE = re.compile(r"^(Recognised\s+Standard\s+\d+|RS\s*\d+|Clause\s+\d+(\.\d+)*|Section\s+\d+|[A-Z][A-Z \-]{5,})")

def smart_chunks(text: str, target=700, tolerance=200):
    paras = [p.strip() for p in text.splitlines() if p.strip()]
    blocks, cur = [], []
    for p in paras:
        if HEADING_RE.match(p) and cur:
            blocks.append("\n".join(cur)); cur = [p]
        else:
            cur.append(p)
    if cur: blocks.append("\n".join(cur))
    out = []
    for b in blocks:
        if len(b) <= target+tolerance:
            out.append(b); continue
        words = b.split()
        temp = []
        for w in words:
            temp.append(w)
            if len(" ".join(temp)) >= target:
                out.append(" ".join(temp)); temp = []
        if temp: out.append(" ".join(temp))
    return out
