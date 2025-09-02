import re, aiohttp, asyncio
from bs4 import BeautifulSoup

PDF_RE = re.compile(r"\.pdf($|\?)", re.I)

async def list_pdfs(session, url: str):
    async with session.get(url, timeout=60) as r:
        r.raise_for_status()
        html = await r.text()
    soup = BeautifulSoup(html, "html.parser")
    hrefs = set()
    for a in soup.find_all("a", href=True):
        h = a["href"]
        if PDF_RE.search(h):
            hrefs.add(h if h.startswith("http") else (url.rstrip("/") + "/" + h.lstrip("/")))
    return sorted(hrefs)
