import os, io, re, asyncio, aiohttp
from pypdf import PdfReader
from pdfminer.high_level import extract_text
from textsplit import smart_chunks
from emb_backend import get_embeddings
from db import sha256_bytes, upsert_doc, insert_chunks, delete_chunks_of_doc

STORE = "data/pdfs"
os.makedirs(STORE, exist_ok=True)


def try_read_pdf_text(bin_bytes: bytes):
    """尝试提取 PDF 文本"""
    try:
        reader = PdfReader(io.BytesIO(bin_bytes))
        pages = len(reader.pages)
        txt = "\n".join(page.extract_text() or "" for page in reader.pages)
        if txt.strip():
            return txt, pages
    except:
        pass
    try:
        with open("/tmp/_tmp.pdf", "wb") as f:
            f.write(bin_bytes)
        txt = extract_text("/tmp/_tmp.pdf") or ""
        return txt, max(1, txt.count("\x0c") + 1)
    except:
        return "", 0


def guess_title_rs(text: str):
    """粗略猜测标题和 RS 编号"""
    head = "\n".join(text.splitlines()[:80])
    m = re.search(r"(Recognised\s+Standard\s*(\d+))", head, re.I)
    title_line = None
    for line in head.splitlines():
        if "Recognised Standard" in line.upper():
            title_line = line.strip()
            break
    rs_no = m.group(2) if m else None
    return title_line or "Recognised Standard", (f"RS{rs_no}" if rs_no else None)


async def download_with_headers(session, url):
    async with session.get(url, timeout=180) as r:
        r.raise_for_status()
        etag = r.headers.get("ETag")
        lm = r.headers.get("Last-Modified")
        data = await r.read()
        return data, etag, lm


async def consume(queue: asyncio.Queue, emb):
    """消费队列中的 PDF URL，解析并写入数据库"""
    async with aiohttp.ClientSession() as session:
        while True:
            url = await queue.get()
            try:
                # 下载 PDF
                bin_bytes, etag, lm = await download_with_headers(session, url)
                sha = sha256_bytes(bin_bytes)
                path = os.path.join(STORE, sha + ".pdf")
                if not os.path.exists(path):
                    with open(path, "wb") as f:
                        f.write(bin_bytes)

                # 提取文本
                text, pages = try_read_pdf_text(bin_bytes)
                if not text.strip():
                    print("Empty text:", url)
                    queue.task_done()
                    continue

                # 文档元数据
                title, rs_no = guess_title_rs(text)
                filename = os.path.basename(url.split("?")[0])
                page_from, page_to = 1, max(1, pages)

                # 切块 + embedding
                chunks_txt = smart_chunks(text, target=800, tolerance=300)
                vecs = emb.embed(chunks_txt)

                # upsert 文档，返回 doc_id（整型）
                doc_id = upsert_doc(
                    source_url=url,
                    sha256_hex=sha,
                    filename=filename,
                    title=title,
                    rs_no=rs_no,
                )

                # 清理旧 chunks
                delete_chunks_of_doc(doc_id)

                # 组织 chunks 为字典列表
                chunks = []
                for i, txt in enumerate(chunks_txt):
                    chunks.append({
                        "text": txt,
                        "page_from": page_from,
                        "page_to": page_to,
                        "title": title,
                        "rs_no": rs_no,
                        "source_url": url,
                        "embedding": vecs[i],
                    })

                # 批量写入
                insert_chunks(doc_id, chunks)
                print(f"Indexed: {url} ({len(chunks)} chunks)")

            except Exception as e:
                print("ERR:", url, e)
            finally:
                queue.task_done()
