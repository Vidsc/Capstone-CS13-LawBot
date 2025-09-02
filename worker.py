import os, io, re, asyncio, aiohttp
from pypdf import PdfReader
from pdfminer.high_level import extract_text
from textsplit import smart_chunks
from emb_backend import get_embeddings
from db import sha256_bytes, upsert_doc, insert_chunks, delete_chunks_of_doc

STORE="data/pdfs"; os.makedirs(STORE, exist_ok=True)

def try_read_pdf_text(bin_bytes:bytes):
    try:
        reader=PdfReader(io.BytesIO(bin_bytes))
        pages=len(reader.pages)
        txt="\n".join(page.extract_text() or "" for page in reader.pages)
        if txt.strip(): return txt, pages
    except: pass
    try:
        with open("/tmp/_tmp.pdf","wb") as f: f.write(bin_bytes)
        txt=extract_text("/tmp/_tmp.pdf") or ""
        return txt, max(1, txt.count("\x0c")+1)
    except:
        return "", 0

def guess_title_rs(text:str):
    head = "\n".join(text.splitlines()[:80])
    m = re.search(r"(Recognised\s+Standard\s*(\d+))", head, re.I)
    title_line = None
    for line in head.splitlines():
        if "Recognised Standard" in line.upper():
            title_line=line.strip(); break
    rs_no = m.group(2) if m else None
    return title_line or "Recognised Standard", (f"RS{rs_no}" if rs_no else None)

async def download_with_headers(session, url):
    async with session.get(url, timeout=180) as r:
        r.raise_for_status()
        etag=r.headers.get("ETag"); lm=r.headers.get("Last-Modified")
        data=await r.read()
        return data, etag, lm

async def consume(queue:asyncio.Queue, emb):
    async with aiohttp.ClientSession() as session:
        while True:
            url = await queue.get()
            try:
                bin_bytes, etag, lm = await download_with_headers(session, url)
                sha = sha256_bytes(bin_bytes)
                path = os.path.join(STORE, sha + ".pdf")
                if not os.path.exists(path):
                    with open(path,"wb") as f: f.write(bin_bytes)
                text, pages = try_read_pdf_text(bin_bytes)
                if not text.strip():
                    print("Empty text:", url); queue.task_done(); continue
                title, rs_no = guess_title_rs(text)
                chunks_txt = smart_chunks(text, target=800, tolerance=300)
                chunks = [(t, 1, max(1,pages)) for t in chunks_txt]
                vecs = emb.embed([t for t,_,_ in chunks])
                doc_meta = {
                    "doc_id": sha[:16],
                    "source_url": url,
                    "title": title,
                    "rs_no": rs_no,
                    "sha256": sha,
                    "etag": etag,
                    "last_modified": lm
                }
                upsert_doc(doc_meta)
                delete_chunks_of_doc(doc_meta["doc_id"])
                insert_chunks(doc_meta["doc_id"], chunks, vecs, url, title, rs_no)
                print("Indexed:", url, f"({len(chunks)} chunks)")
            except Exception as e:
                print("ERR:", url, e)
            finally:
                queue.task_done()
