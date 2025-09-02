# db.py —— psycopg2 连接池 + 数据导入/检索工具
import os
import hashlib
from typing import List, Dict, Any

from dotenv import load_dotenv
import psycopg2
import psycopg2.extras
from psycopg2 import pool
from psycopg2.extras import execute_values

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")

# ---------- 连接池 ----------
_CONN_POOL = None

def get_conn():
    global _CONN_POOL
    if _CONN_POOL is None:
        _CONN_POOL = psycopg2.pool.SimpleConnectionPool(
            minconn=1,
            maxconn=5,
            dsn=DATABASE_URL
        )
    return _CONN_POOL.getconn()

def put_conn(conn):
    if _CONN_POOL:
        _CONN_POOL.putconn(conn)

# ---------- 给 worker 用的工具函数 ----------

def sha256_bytes(b: bytes) -> str:
    """返回文件内容的 sha256 十六进制摘要"""
    return hashlib.sha256(b).hexdigest()

def upsert_doc(source_url: str,
               sha256_hex: str,
               filename: str | None = None,
               title: str | None = None,
               rs_no: str | None = None) -> int:
    """
    插入/更新一条文档记录，并返回 doc_id
    需要保证 docs(source_url) 有唯一约束
    docs 表（建议字段）：id PK, source_url UNIQUE, sha256, filename, title, rs_no, created_at, updated_at
    """
    sql = """
    INSERT INTO docs (source_url, sha256, filename, title, rs_no)
    VALUES (%s, %s, %s, %s, %s)
    ON CONFLICT (source_url)
    DO UPDATE SET
        sha256  = EXCLUDED.sha256,
        filename= COALESCE(EXCLUDED.filename, docs.filename),
        title   = COALESCE(EXCLUDED.title,    docs.title),
        rs_no   = COALESCE(EXCLUDED.rs_no,    docs.rs_no)
    RETURNING id;
    """
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(sql, (source_url, sha256_hex, filename, title, rs_no))
            doc_id = cur.fetchone()[0]
            conn.commit()
            return doc_id
    finally:
        put_conn(conn)

def delete_chunks_of_doc(doc_id: int) -> int:
    """删除某文档的所有切块，返回删除条数"""
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM chunks WHERE doc_id = %s;", (doc_id,))
            n = cur.rowcount
            conn.commit()
            return n
    finally:
        put_conn(conn)

def insert_chunks(doc_id: int, chunks: List[Dict[str, Any]]) -> int:
    """
    批量插入切块
    期望 chunks 的每个元素包含：text/txt, page_from, page_to, embedding(list/ndarray), 以及可选 title/rs_no/source_url
    如果没传 source_url，则用 docs 表里的
    表结构（示例）：chunks(id, doc_id, source_url, title, rs_no, page_from, page_to, txt, embedding vector(384))
    """
    if not chunks:
        return 0

    conn = get_conn()
    try:
        with conn.cursor() as cur:
            # 获取 source_url 作为兜底
            cur.execute("SELECT source_url FROM docs WHERE id=%s;", (doc_id,))
            row = cur.fetchone()
            default_source_url = row[0] if row else None

            rows = []
            for c in chunks:
                txt = c.get("text") or c.get("txt") or ""
                page_from = c.get("page_from")
                page_to = c.get("page_to")
                title = c.get("title")
                rs_no = c.get("rs_no")
                src = c.get("source_url") or default_source_url
                emb = c.get("embedding")
                if hasattr(emb, "tolist"):
                    emb = emb.tolist()  # ndarray -> list

                rows.append((
                    doc_id, src, title, rs_no, page_from, page_to, txt, emb
                ))

            # 用 execute_values 批量插入，并把向量参数 cast 为 ::vector
            sql = """
            INSERT INTO chunks
                (doc_id, source_url, title, rs_no, page_from, page_to, txt, embedding)
            VALUES %s
            """
            template = "(%s,%s,%s,%s,%s,%s,%s,%s::vector)"
            execute_values(cur, sql, rows, template=template)
            n = cur.rowcount
            conn.commit()
            return n
    finally:
        put_conn(conn)

# ---------- 聊天检索用（RAG） ----------
def search_by_text_vector(qvec, top_k=8):
    """
    用 pgvector 做相似度检索。这里不去重，方便调试；如需去重可改为 DISTINCT ON(doc_id)。
    另外：从 source_url 直接提取原始 PDF 文件名作为 filename。
    """
    q = qvec.tolist()

    sql = """
    SELECT
        id,
        doc_id,
        source_url,
        replace(
          split_part(regexp_replace(source_url, '.*/', ''), '?', 1),
          '%%20', ' '
        ) AS filename,
        rs_no,
        page_from,
        page_to,
        LEFT(txt, 800) AS text,
        (embedding <-> %s::vector) AS score
    FROM chunks
    ORDER BY embedding <-> %s::vector
    LIMIT %s;
    """

    conn = get_conn()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql, (q, q, top_k))
            return cur.fetchall()
    finally:
        put_conn(conn)
