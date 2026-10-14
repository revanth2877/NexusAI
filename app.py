import os
import io
import uuid
import hashlib
import re
import sqlite3
from pathlib import Path
from datetime import datetime

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader
from google import genai
from google.genai import types

try:
    from document_store import retrieve_chunks
except Exception:
    retrieve_chunks = None

# ============================================================
# NEXUSAI — SINGLE CHAT WORKSPACE
# Replaces the old multi-page navigation while preserving DBs.
# ============================================================

load_dotenv()
BASE_DIR = Path(__file__).resolve().parent
DOCUMENT_DB = BASE_DIR / "nexusai_documents.db"
MEMORY_DB = BASE_DIR / "nexusai_memory.db"

API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-3.8-flash").strip()

st.set_page_config(
    page_title="NexusAI — Intelligent Workspace",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -------------------------- STYLE ----------------------------

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

:root {
  --bg: #080b14;
  --panel: #101525;
  --panel2: #151b2e;
  --line: rgba(174, 184, 218, .14);
  --muted: #929db8;
  --text: #f4f6ff;
  --purple: #8b5cf6;
  --blue: #4f8cff;
}
html, body, [class*="css"] {
  font-family: 'DM Sans', sans-serif;
}
.stApp {
  color: var(--text);
  background:
    radial-gradient(ellipse at 70% -10%, rgba(93, 63, 180, .25), transparent 38%),
    radial-gradient(ellipse at 0% 35%, rgba(38, 96, 177, .10), transparent 35%),
    var(--bg);
}
[data-testid="stHeader"] { background: rgba(8,11,20,.72); }
.block-container { max-width: 1120px; padding-top: 1.5rem; padding-bottom: 7rem; }
section[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #0d1120 0%, #0a0d17 100%);
  border-right: 1px solid var(--line);
}
section[data-testid="stSidebar"] > div { padding-top: 1.1rem; }
.nx-brand {
  font-family: 'Space Grotesk', sans-serif;
  font-weight: 700; font-size: 1.48rem; letter-spacing: -.7px;
  display:flex; align-items:center; gap:10px; color:#fff;
}
.nx-mark {
  width:35px;height:35px;border-radius:12px;display:inline-flex;
  align-items:center;justify-content:center;color:white;font-size:19px;
  background:linear-gradient(135deg,#9b6bff,#4e83ff);
  box-shadow:0 5px 22px rgba(119,90,255,.35);
}
.nx-muted { color: var(--muted); font-size: .84rem; }
.nx-section {
  color:#aab4ce; text-transform:uppercase; letter-spacing:1.5px;
  font-size:.68rem; font-weight:700; margin:1.25rem 0 .55rem;
}
.nx-hero { text-align:center; padding: 5.5rem 1rem 2rem; }
.nx-orb {
  width:72px;height:72px;margin:0 auto 1.2rem;border-radius:24px;
  display:flex;align-items:center;justify-content:center;font-size:34px;
  color:white;background:linear-gradient(135deg,rgba(155,107,255,.96),rgba(70,126,255,.96));
  box-shadow:0 12px 50px rgba(107,84,255,.3);
}
.nx-title {
  font-family:'Space Grotesk',sans-serif;font-size:clamp(2rem,4vw,3.15rem);
  line-height:1.08;letter-spacing:-1.8px;font-weight:700;color:#fff;
}
.nx-gradient {
  background:linear-gradient(90deg,#b69bff,#74b3ff);
  -webkit-background-clip:text;background-clip:text;color:transparent;
}
.nx-subtitle { color:#9da8c2;font-size:1rem;max-width:600px;margin:.8rem auto 0;line-height:1.7; }
.nx-chip {
  display:inline-block;padding:6px 11px;border:1px solid rgba(151,125,255,.25);
  border-radius:999px;background:rgba(119,87,226,.09);color:#c5b8ff;
  font-size:.75rem;margin:0 3px 6px;
}
.nx-card {
  background:linear-gradient(145deg,rgba(23,30,52,.92),rgba(14,19,34,.92));
  border:1px solid var(--line);border-radius:18px;padding:17px 18px;
  min-height:112px;transition:all .2s ease;
}
.nx-card-title { color:#f1f3ff;font-weight:650;margin-bottom:5px; }
.nx-card-copy { color:#96a2bd;font-size:.85rem;line-height:1.5; }
.nx-small {
  color:#a7b2cc;background:rgba(255,255,255,.035);border:1px solid var(--line);
  border-radius:12px;padding:10px 12px;font-size:.82rem;
}
.stButton > button {
  border-radius:12px;border:1px solid var(--line);font-weight:600;
  min-height:42px;transition:all .18s ease;
}
.stButton > button[kind="primary"] {
  background:linear-gradient(100deg,#8b5cf6,#4f83ff);color:#fff;border:0;
  box-shadow:0 5px 20px rgba(91,83,236,.18);
}
.stButton > button:hover { border-color:rgba(164,143,255,.65); transform:translateY(-1px); }
.stTextInput input, .stTextArea textarea {
  background:#111729 !important;color:#f5f7ff !important;border-radius:12px !important;
  border:1px solid var(--line) !important;
}
[data-testid="stChatMessage"] {
  border:1px solid rgba(157,169,207,.11);border-radius:17px;
  background:rgba(17,23,40,.72);padding:15px 17px;margin:.8rem 0;
}
[data-testid="stChatInput"] {
  border:1px solid rgba(143,124,255,.34) !important;
  background:#101627 !important;border-radius:18px !important;
  box-shadow:0 10px 40px rgba(0,0,0,.22);
}
[data-testid="stFileUploader"] {
  background:rgba(28,35,59,.55);border:1px dashed rgba(143,124,255,.42);
  border-radius:14px;padding:10px;
}
[data-testid="stExpander"] {
  border:1px solid var(--line);border-radius:12px;background:rgba(18,24,42,.45);
}
hr { border-color:var(--line); }
footer {visibility:hidden;}
#MainMenu {visibility:hidden;}
.nx-history-label {color:#f1f3ff;font-size:.87rem;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}
</style>
""", unsafe_allow_html=True)

# -------------------------- STORAGE --------------------------

def db_connect(path):
    return sqlite3.connect(str(path), timeout=15)

def initialize_databases():
    with db_connect(DOCUMENT_DB) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                file_hash TEXT UNIQUE NOT NULL,
                content TEXT NOT NULL,
                indexed_at TEXT NOT NULL
            )
        """)
    with db_connect(MEMORY_DB) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS chat_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                section TEXT NOT NULL,
                role TEXT NOT NULL,
                message TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS nexusai_conversations_v3 (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)
        # Backfill conversations from any older saved history rows.
        rows = conn.execute("""
            SELECT session_id, MIN(created_at), MAX(created_at)
            FROM chat_history GROUP BY session_id
        """).fetchall()
        for cid, created, updated in rows:
            title_row = conn.execute("""
                SELECT message FROM chat_history
                WHERE session_id=? AND role='user' ORDER BY id LIMIT 1
            """, (cid,)).fetchone()
            title = (title_row[0] if title_row else "Previous conversation")[:55]
            conn.execute("""
                INSERT OR IGNORE INTO nexusai_conversations_v3(id,title,created_at,updated_at)
                VALUES(?,?,?,?)
            """, (cid, title, created or datetime.now().isoformat(), updated or datetime.now().isoformat()))

def list_conversations():
    with db_connect(MEMORY_DB) as conn:
        rows = conn.execute("""
            SELECT c.id, c.title, c.updated_at,
                   (SELECT message FROM chat_history h
                    WHERE h.session_id=c.id AND h.role='user'
                    ORDER BY h.id DESC LIMIT 1)
            FROM nexusai_conversations_v3 c
            ORDER BY c.updated_at DESC
            LIMIT 80
        """).fetchall()
    return [{"id": r[0], "title": r[1], "updated": r[2], "last": r[3] or ""} for r in rows]

def create_conversation():
    cid = str(uuid.uuid4())
    now = datetime.now().isoformat(timespec="seconds")
    with db_connect(MEMORY_DB) as conn:
        conn.execute("INSERT INTO nexusai_conversations_v3(id,title,created_at,updated_at) VALUES(?,?,?,?)",
                     (cid, "New conversation", now, now))
    return cid

def save_message(cid, role, message):
    now = datetime.now().isoformat(timespec="seconds")
    with db_connect(MEMORY_DB) as conn:
        conn.execute("""
            INSERT INTO chat_history(session_id,section,role,message,created_at)
            VALUES(?,?,?,?,?)
        """, (cid, "Chat", role, message, now))
        if role == "user":
            row = conn.execute("SELECT title FROM nexusai_conversations_v3 WHERE id=?", (cid,)).fetchone()
            if row and row[0] == "New conversation":
                title = " ".join(message.split())[:48] or "New conversation"
                conn.execute("UPDATE nexusai_conversations_v3 SET title=? WHERE id=?", (title, cid))
        conn.execute("UPDATE nexusai_conversations_v3 SET updated_at=? WHERE id=?", (now, cid))

def get_history(cid, limit=80):
    with db_connect(MEMORY_DB) as conn:
        rows = conn.execute("""
            SELECT role,message FROM chat_history
            WHERE session_id=? ORDER BY id ASC LIMIT ?
        """, (cid, limit)).fetchall()
    return [{"role": r, "content": m} for r, m in rows]

initialize_databases()
if "active_chat_id" not in st.session_state:
    st.session_state.active_chat_id = create_conversation()
if "uploaded_file_signatures" not in st.session_state:
    st.session_state.uploaded_file_signatures = {}
if "pending_attachments" not in st.session_state:
    st.session_state.pending_attachments = []
if "last_uploaded_names" not in st.session_state:
    st.session_state.last_uploaded_names = []

# -------------------------- FILES ----------------------------

def extract_pdf_bytes(data):
    reader = PdfReader(io.BytesIO(data))
    parts = []
    for i, page in enumerate(reader.pages, 1):
        txt = page.extract_text() or ""
        if txt.strip():
            parts.append(f"[Page {i}]\n{txt.strip()}")
    return "\n\n".join(parts)

def extract_file(uploaded):
    name = uploaded.name
    suffix = Path(name).suffix.lower()
    raw = uploaded.getvalue()
    digest = hashlib.sha256(raw).hexdigest()
    text = ""
    image_part = None

    if suffix == ".pdf":
        text = extract_pdf_bytes(raw)
    elif suffix in {".txt", ".md", ".py", ".json", ".xml", ".html", ".log"}:
        text = raw.decode("utf-8", errors="replace")
    elif suffix == ".csv":
        df = pd.read_csv(io.BytesIO(raw))
        text = (
            f"CSV FILE: {name}\nRows: {len(df)}\nColumns: {list(df.columns)}\n\n"
            f"Sample rows:\n{df.head(100).to_string(index=False)}\n\n"
            f"Statistics:\n{df.describe(include='all').to_string()}"
        )
    elif suffix in {".xlsx", ".xls"}:
        df = pd.read_excel(io.BytesIO(raw))
        text = (
            f"SPREADSHEET: {name}\nRows: {len(df)}\nColumns: {list(df.columns)}\n\n"
            f"Sample rows:\n{df.head(100).to_string(index=False)}\n\n"
            f"Statistics:\n{df.describe(include='all').to_string()}"
        )
    elif suffix in {".png", ".jpg", ".jpeg", ".webp"}:
        mime = "image/png" if suffix == ".png" else "image/webp" if suffix == ".webp" else "image/jpeg"
        image_part = types.Part.from_bytes(data=raw, mime_type=mime)
        text = f"Image attachment: {name}. Analyze the image visually when answering the user's question."
    else:
        raise ValueError(f"Unsupported file type: {suffix or 'unknown'}")

    if text.strip():
        with db_connect(DOCUMENT_DB) as conn:
            conn.execute("""
                INSERT INTO documents(filename,file_hash,content,indexed_at)
                VALUES(?,?,?,?)
                ON CONFLICT(file_hash) DO UPDATE SET
                  filename=excluded.filename,content=excluded.content,indexed_at=excluded.indexed_at
            """, (name, digest, text, datetime.now().isoformat(timespec="seconds")))
    return {"name": name, "hash": digest, "text": text, "image_part": image_part, "raw": raw, "suffix": suffix}

def get_documents():
    with db_connect(DOCUMENT_DB) as conn:
        rows = conn.execute("SELECT filename,file_hash,content,indexed_at FROM documents ORDER BY indexed_at DESC").fetchall()
    return [{"filename": r[0], "file_hash": r[1], "content": r[2], "indexed_at": r[3]} for r in rows]

def retrieve_context(question):
    """Retrieve relevant passages with FAISS semantic search, with a safe keyword fallback."""
    docs = get_documents()
    if not docs:
        return "", []

    # Preferred path: semantic retrieval through document_store.py.
    # The index is cached by that module, so repeated questions over the same
    # document collection do not need to rebuild all passage embeddings.
    if retrieve_chunks is not None:
        try:
            semantic_results = retrieve_chunks(question, docs, top_k=5)
            if semantic_results:
                context_parts = []
                sources = []
                seen = set()
                for item in semantic_results:
                    filename = item.get("filename", "Unknown document")
                    passage = (item.get("text") or "").strip()
                    if not passage:
                        continue
                    key = (filename, passage)
                    if key in seen:
                        continue
                    seen.add(key)
                    score = item.get("score")
                    score_label = f"\nSIMILARITY SCORE: {float(score):.3f}" if isinstance(score, (int, float)) else ""
                    context_parts.append(
                        f"SOURCE FILE: {filename}{score_label}\nEXCERPT:\n{passage}"
                    )
                    sources.append({"filename": filename, "text": passage[:1000]})
                if context_parts:
                    return "\n\n---\n\n".join(context_parts), sources
        except Exception as exc:
            # A missing model dependency or local embedding error should not
            # take down the working chat; fall back to keyword retrieval.
            st.warning(
                "Semantic search is temporarily unavailable, so NexusAI is using "
                f"keyword search for this question. Details: {type(exc).__name__}: {str(exc)[:180]}"
            )

    # Fallback path: keyword ranking, used if semantic search is unavailable.
    tokens = [t.lower() for t in re.findall(r"[A-Za-z0-9_'-]+", question) if len(t) > 2]
    stop = {
        "the", "and", "for", "that", "this", "with", "from", "what", "when", "where",
        "which", "who", "how", "does", "are", "was", "were", "have", "has", "can",
        "you", "your", "about", "into", "tell", "give", "please", "file", "document", "uploaded"
    }
    qwords = [token for token in tokens if token not in stop]
    document_intent = any(term in question.lower() for term in (
        "uploaded", "document", "file", "pdf", "spreadsheet", "excel", "csv", "summarize",
        "summary", "according to", "in the", "from the", "this data", "these data",
        "my data", "my file", "my document"
    ))

    chunks = []
    chunk_size = 1500
    overlap = 220
    for doc in docs:
        content = doc.get("content") or ""
        if not content.strip():
            continue
        start_at = 0
        while start_at < len(content):
            end_at = min(len(content), start_at + chunk_size)
            if end_at < len(content):
                boundary = content.rfind("\n", start_at + 900, end_at)
                if boundary > start_at:
                    end_at = boundary
            chunk = content[start_at:end_at].strip()
            if chunk:
                chunks.append({
                    "filename": doc.get("filename", "Unknown document"),
                    "text": chunk,
                    "indexed_at": doc.get("indexed_at", ""),
                })
            if end_at >= len(content):
                break
            start_at = max(start_at + 1, end_at - overlap)

    scored = []
    for chunk in chunks:
        low = chunk["text"].lower()
        name_low = chunk["filename"].lower()
        score = sum(min(low.count(word), 4) * (2 if len(word) >= 6 else 1) for word in set(qwords))
        score += sum(3 for word in set(qwords) if word in name_low)
        normalized_question = " ".join(qwords)
        if len(normalized_question) > 5 and normalized_question in low:
            score += 8
        if score:
            scored.append((score, chunk))

    scored.sort(key=lambda pair: (pair[0], pair[1].get("indexed_at", "")), reverse=True)
    selected = [item for _, item in scored[:5]]
    if not selected and document_intent:
        selected = chunks[:4]

    context = "\n\n---\n\n".join(
        f"SOURCE FILE: {item['filename']}\nEXCERPT:\n{item['text']}"
        for item in selected
    )
    sources = [{"filename": item["filename"], "text": item["text"][:1000]} for item in selected]
    return context, sources

# -------------------------- GEMINI ---------------------------

def generate_answer(prompt, attachments=None):
    """Generate a response quickly, with one fast request per fallback model."""
    if not API_KEY:
        return "Gemini API key was not found. Check your existing .env file; do not paste your key into chat."

    # Disable long automatic retry cycles: try the configured model, then a short fallback list.
    try:
        client = genai.Client(
            api_key=API_KEY,
            http_options=types.HttpOptions(
                timeout=30000,
                retry_options=types.HttpRetryOptions(attempts=1),
            ),
        )
    except Exception:
        client = genai.Client(api_key=API_KEY)

    contents = [prompt]
    for item in attachments or []:
        if item.get("image_part") is not None:
            contents.append(item["image_part"])
        # Text attachments are already included once in make_prompt(). Do not duplicate them.

    candidates = []
    for model in [MODEL_NAME, "gemini-3.8-flash", "gemini-3.6-flash", "gemini-3.5-flash-lite", "gemini-3.1-flash-lite"]:
        model = (model or "").strip()
        if model and model not in candidates:
            candidates.append(model)

    errors = []
    for model in candidates:
        try:
            response = client.models.generate_content(
                model=model,
                contents=contents,
                config=types.GenerateContentConfig(temperature=0.3, max_output_tokens=1400),
            )
            text = (response.text or "").strip()
            if text:
                return text
            errors.append(f"{model}: empty response")
        except Exception as exc:
            message = str(exc)
            errors.append(f"{model}: {type(exc).__name__}: {message[:220]}")
            lower = message.lower()
            code = getattr(exc, "code", None)
            is_transient = code in (408, 429, 500, 502, 503, 504) or any(
                marker in lower for marker in ("503", "unavailable", "high demand", "429", "resource_exhausted", "500", "internal server error", "timeout")
            )
            # Try a fallback model for transient capacity errors or a model-not-found response.
            is_model_issue = any(marker in lower for marker in ("404", "not found", "unknown model", "unsupported model"))
            if not (is_transient or is_model_issue):
                break

    details = " | ".join(errors[-3:])
    if any(x in details.lower() for x in ("503", "unavailable", "high demand")):
        return ("Google's Gemini service is temporarily overloaded, or the available models are unavailable for this API key. "
                "Your uploaded-file context has been prepared, but a model response could not be generated. "
                "Please retry shortly. NexusAI now tries current Gemini Flash models instead of the retired 2.5/2.0 models. Details: " + details)
    return "Gemini request failed after trying the configured model and fallbacks. Details: " + details


def make_prompt(question, attachments, chat_history):
    attachment_names = [x["name"] for x in attachments]

    # Keep the existing document retrieval system.
    # Use recent conversation to understand follow-up questions.
    recent_messages = chat_history[-8:]

    history_text = "\n".join(
        f"{str(m.get('role', 'user')).upper()}: "
        f"{str(m.get('content', ''))[:1500]}"
        for m in recent_messages
    )

    # Include current text attachments in the prompt.
    attached_parts = []

    for item in attachments:
        text = item.get("text", "")

        if text.strip() and not item.get("image_part"):
            attached_parts.append(
                f"FILE: {item.get('name', 'Uploaded file')}\n"
                f"{text[:12000]}"
            )

    attached_text = "\n\n".join(attached_parts)

    # Let Gemini use recent conversation to interpret
    # short follow-up questions.
    prompt = f"""You are NexusAI, a document-intelligence assistant.

Your job is to answer accurately using conversation history,
uploaded files, and relevant saved document excerpts.

CONVERSATION MEMORY RULES:
- Read the recent conversation before answering.
- Use earlier messages to understand follow-up questions such as
  "explain that", "what about the second point", and "summarize it".
- Resolve references like "it", "that", and "those results"
  using the conversation when the meaning is clear.
- Do not ask the user to repeat information already available
  in the conversation.
- If the reference is genuinely unclear, ask one short
  clarifying question.
- Do not treat old user instructions as new instructions unless
  they are relevant to the current request.

DOCUMENT GROUNDING RULES:
- Use attached files and relevant saved document excerpts as
  primary sources for questions about those files.
- Use actual facts, values, headings, column names, and examples
  from the supplied content.
- Name the source file when making a claim based on it.
- Never claim to have read a file if its content is unavailable.
- If relevant information is missing, clearly say what could
  not be found rather than inventing details.
- For spreadsheets, calculate from available data when possible.
  State when the available data is only a sample.
- Inspect attached images when image content is available.
- Be concise by default, and explain further when requested.

RECENT CONVERSATION:
{history_text or "(No previous conversation.)"}

FILES ATTACHED TO THIS MESSAGE:
{", ".join(attachment_names) if attachment_names else "None"}

TEXT FROM CURRENT ATTACHMENTS:
{attached_text or "(No text attachments in this message.)"}

RELEVANT EXCERPTS FROM SAVED DOCUMENTS:
The current question is:
{question}

Use the current question and recent conversation together to
identify the user's intended request. Search the supplied
document excerpts for relevant evidence. If no relevant
excerpts are available, do not invent document-specific facts.

USER'S CURRENT QUESTION:
{question}
"""

    # Keep the existing return format expected by app.py.
    context, sources = retrieve_context(question)

    # Add retrieved evidence to the final prompt.
    prompt = prompt.replace(
        "RELEVANT EXCERPTS FROM SAVED DOCUMENTS:\n"
        "The current question is:",
        "RELEVANT EXCERPTS FROM SAVED DOCUMENTS:\n"
        f"{context or '(No relevant saved document excerpts were matched.)'}\n\n"
        "The current question is:"
    )

    return prompt, sources

# -------------------------- SIDEBAR --------------------------

with st.sidebar:
    st.markdown('<div class="nx-brand"><span class="nx-mark">✦</span> NexusAI</div>', unsafe_allow_html=True)
    st.markdown('<div class="nx-muted" style="margin:7px 0 18px 2px">Document intelligence · one workspace</div>', unsafe_allow_html=True)

    if st.button("＋  New chat", type="primary", use_container_width=True):
        st.session_state.active_chat_id = create_conversation()
        st.session_state.pending_attachments = []
        st.session_state.last_uploaded_names = []
        st.rerun()

    st.markdown('<div class="nx-section">Your conversations</div>', unsafe_allow_html=True)
    conversations = list_conversations()
    if not conversations:
        st.caption("Your conversations will appear here.")
    for conv in conversations:
        label = ("●  " if conv["id"] == st.session_state.active_chat_id else "   ") + conv["title"]
        if st.button(label[:58], key=f"open_{conv['id']}", use_container_width=True,
                     type="secondary" if conv["id"] == st.session_state.active_chat_id else "tertiary"):
            st.session_state.active_chat_id = conv["id"]
            st.session_state.pending_attachments = []
            st.session_state.last_uploaded_names = []
            st.rerun()

    st.divider()
    docs = get_documents()
    st.markdown(f'<div class="nx-small">📚 <b>{len(docs)}</b> saved file(s) in your knowledge base</div>', unsafe_allow_html=True)
    st.markdown('<div class="nx-muted" style="margin-top:14px;font-size:.75rem">Powered by Gemini · Local SQLite memory</div>', unsafe_allow_html=True)

# -------------------------- MAIN CHAT ------------------------

active_id = st.session_state.active_chat_id
history = get_history(active_id)
conversation_title = next((c["title"] for c in list_conversations() if c["id"] == active_id), "New conversation")

top1, top2 = st.columns([7, 1])
with top1:
    st.markdown('<div style="color:#9da8c2;font-size:.78rem;letter-spacing:1.4px;text-transform:uppercase;margin:3px 0 6px">NEXUSAI WORKSPACE</div>', unsafe_allow_html=True)
    st.markdown(f'<div style="font-family:Space Grotesk,sans-serif;font-size:1.35rem;font-weight:650;color:#f6f7ff">{conversation_title.replace("<","&lt;").replace(">","&gt;")}</div>', unsafe_allow_html=True)
with top2:
    st.markdown('<div style="text-align:right;padding-top:8px;color:#9da8c2;font-size:.8rem">● Online</div>', unsafe_allow_html=True)

if not history:
    st.markdown("""
    <div class="nx-hero">
      <div class="nx-orb">✦</div>
      <div class="nx-title">Your ideas, files & data.<br><span class="nx-gradient">One intelligent workspace.</span></div>
      <div class="nx-subtitle">Ask anything, explore documents, understand spreadsheets, or share an image. NexusAI brings it all into one conversation.</div>
      <div style="margin-top:22px">
        <span class="nx-chip">✧ Gemini AI</span>
        <span class="nx-chip">⌕ Document search</span>
        <span class="nx-chip">▦ Data insights</span>
        <span class="nx-chip">◈ Image understanding</span>
      </div>
    </div>
    """, unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown('<div class="nx-card"><div class="nx-card-title">✧ Think with me</div><div class="nx-card-copy">Break down complex ideas, plan projects, or solve a problem step by step.</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="nx-card"><div class="nx-card-title">⌕ Ask your files</div><div class="nx-card-copy">Drop in documents and ask specific questions about their contents.</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="nx-card"><div class="nx-card-title">▦ Explore your data</div><div class="nx-card-copy">Understand CSV and Excel files, find patterns, and generate insights.</div></div>', unsafe_allow_html=True)
else:
    for item in history:
        with st.chat_message(item["role"], avatar="✨" if item["role"] == "assistant" else "👤"):
            st.markdown(item["content"])

# One compact attachment area for every supported file type.
with st.expander("＋  Attach files  ·  PDF, images, CSV, Excel, text", expanded=False):
    uploads = st.file_uploader(
        "Add files to this message",
        type=["pdf", "png", "jpg", "jpeg", "webp", "csv", "xlsx", "xls", "txt", "md", "py", "json", "xml", "html", "log"],
        accept_multiple_files=True,
        key=f"files_{active_id}",
        label_visibility="collapsed",
        help="Files are processed locally where possible; images are sent to Gemini for visual understanding."
    )
    if uploads:
        st.caption("Selected: " + ", ".join(x.name for x in uploads))
        if st.button("Use these files in my next message", type="primary", key=f"attach_{active_id}"):
            processed = []
            for upload in uploads:
                try:
                    processed.append(extract_file(upload))
                except Exception as exc:
                    st.error(f"Could not read {upload.name}: {exc}")
            st.session_state.pending_attachments = processed
            st.session_state.last_uploaded_names = [x["name"] for x in processed]
            if processed:
                st.success(f"{len(processed)} file(s) ready for your next message.")
            st.rerun()
    pending = st.session_state.pending_attachments
    if pending:
        st.markdown("**Ready for next message**")
        for item in pending:
            st.write(f"📎 {item['name']}")
        if st.button("Clear attachments", key=f"clear_{active_id}"):
            st.session_state.pending_attachments = []
            st.session_state.last_uploaded_names = []
            st.rerun()

prompt = st.chat_input("Message NexusAI… ask a question or tell me what to do")
if prompt:
    attachments = list(st.session_state.pending_attachments)
    save_message(active_id, "user", prompt + (("\n\n📎 Attached: " + ", ".join(a["name"] for a in attachments)) if attachments else ""))
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)
        if attachments:
            st.caption("📎 " + " · ".join(a["name"] for a in attachments))
    prior_history = get_history(active_id)
    # Avoid duplicating the current user turn in the history passed to Gemini.
    prior_history = prior_history[:-1]
    with st.chat_message("assistant", avatar="✨"):
        with st.spinner("NexusAI is thinking…"):
            prompt_text, sources = make_prompt(prompt, attachments, prior_history)
            answer = generate_answer(prompt_text, attachments)
            st.markdown(answer)
            if sources:
                with st.expander(f"Sources used · {len(sources)} excerpt(s)"):
                    for source in sources:
                        st.markdown(f"**{source.get('filename', 'Document')}**")
                        st.write(source.get("text", ""))
    save_message(active_id, "assistant", answer)
    st.session_state.pending_attachments = []
    st.session_state.last_uploaded_names = []
    st.rerun()

st.markdown('<div style="text-align:center;color:#69748e;font-size:.73rem;margin-top:26px">NexusAI can make mistakes. Verify important information. · Your local chat history is stored in SQLite.</div>', unsafe_allow_html=True)
