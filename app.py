import os, asyncio
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
from emb_backend import get_embeddings
from db import search_by_text_vector
from worker import consume
from crawler import list_pdfs

load_dotenv()
app = FastAPI(title="QLD RS Vector")
emb = get_embeddings()
queue: asyncio.Queue = asyncio.Queue()
SEEDS = [s.strip() for s in os.environ.get("SEED_URLS","").split(",") if s.strip()]

@app.on_event("startup")
async def startup_event():
    for _ in range(3):
        asyncio.create_task(consume(queue, emb))
    asyncio.create_task(produce(SEEDS, queue))

async def produce(seed_urls, queue):
    import aiohttp
    async with aiohttp.ClientSession() as session:
        for u in seed_urls:
            try:
                pdfs = await list_pdfs(session, u)
                for p in pdfs:
                    await queue.put(p)
                print(f"Discovered {len(pdfs)} PDFs from {u}")
            except Exception as e:
                print("discover ERR:", u, e)

class SearchReq(BaseModel):
    q: str
    top_k: int = 8

@app.post("/search")
def search(body: SearchReq):
    qvec = emb.embed([body.q])[0]
    res = search_by_text_vector(qvec, top_k=body.top_k)
    return {"query": body.q, "results": res}

@app.get("/stats")
def stats():
    return {"queue_size": queue.qsize()}
