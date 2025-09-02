-- 扩展
CREATE EXTENSION IF NOT EXISTS postgis;
CREATE EXTENSION IF NOT EXISTS vector;

-- 文档表
CREATE TABLE IF NOT EXISTS docs (
  doc_id         TEXT PRIMARY KEY,
  source_url     TEXT NOT NULL,
  title          TEXT,
  rs_no          TEXT,
  sha256         TEXT NOT NULL UNIQUE,
  etag           TEXT,
  last_modified  TEXT,
  created_at     TIMESTAMPTZ DEFAULT now(),
  updated_at     TIMESTAMPTZ DEFAULT now(),
  origin_geom    geometry(Geometry, 4326)
);

-- 分块表（文本块 + 向量）
CREATE TABLE IF NOT EXISTS chunks (
  id            BIGSERIAL PRIMARY KEY,
  doc_id        TEXT REFERENCES docs(doc_id) ON DELETE CASCADE,
  source_url    TEXT NOT NULL,
  title         TEXT,
  rs_no         TEXT,
  page_from     INT,
  page_to       INT,
  txt           TEXT NOT NULL,
  embedding     vector(384), -- 若后续换嵌入模型维度，要同步改
  created_at    TIMESTAMPTZ DEFAULT now(),
  updated_at    TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_chunks_doc_id ON chunks(doc_id);
CREATE INDEX IF NOT EXISTS idx_chunks_rs_no  ON chunks(rs_no);
CREATE INDEX IF NOT EXISTS idx_docs_sha256   ON docs(sha256);
CREATE INDEX IF NOT EXISTS idx_docs_rs_no    ON docs(rs_no);

-- 向量索引（IVFFLAT）
CREATE INDEX IF NOT EXISTS idx_chunks_emb_ivfflat
  ON chunks USING ivfflat (embedding vector_l2_ops)
  WITH (lists = 100);
