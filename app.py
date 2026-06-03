"""
╔══════════════════════════════════════════════════════════════════╗
║   THE TAJ MAHAL PALACE — AI CONCIERGE                           ║
║   Grounded Hotel RAG Bot                                        ║
║   Stack: Python · FAISS · Groq (Llama-3) · Streamlit           ║
║   Translation: Groq LLM (any language → English → LLM/FAISS)   ║
╚══════════════════════════════════════════════════════════════════╝
"""

# ─── IMPORTS ────────────────────────────────────────────────────────────────

import os, re, json, time, html as html_module
import numpy as np
import streamlit as st
from groq import Groq
import faiss
from sentence_transformers import SentenceTransformer

# ─── CONFIGURATION ──────────────────────────────────────────────────────────

KB_PATH       = "taj_mahal_palace_mumbai_kb.txt"
EMBED_MODEL   = "all-MiniLM-L6-v2"
GROQ_MODEL    = "llama-3.3-70b-versatile"
TOP_K         = 5
CHUNK_SIZE    = 350
CHUNK_OVERLAP = 50
MAX_HISTORY   = 6

# ─── TRANSLATION (Groq-powered) ─────────────────────────────────────────────

def translate_to_english(text: str, groq_client: Groq) -> tuple[str, str]:
    """
    Detects language and translates any input to English using Groq LLM.
    Returns (english_text, detected_language_name).
    Falls back to original text if something goes wrong.
    """
    try:
        resp = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a translation assistant. Your job has two parts:\n"
                        "1. Detect the language of the user's message.\n"
                        "2. Translate it to English.\n\n"
                        "Respond ONLY with a JSON object in this exact format (no markdown, no extra text):\n"
                        '{"lang": "<language name in English e.g. Hindi, French, English>", "translated": "<English translation>"}\n\n'
                        "If the message is already in English, still return JSON with lang=English and translated=original text."
                    ),
                },
                {"role": "user", "content": text},
            ],
            temperature=0.0,
            max_tokens=300,
        )
        raw = resp.choices[0].message.content.strip()
        # Strip markdown fences if present
        raw = re.sub(r"^```json\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
        data = json.loads(raw)
        lang        = data.get("lang", "English")
        translated  = data.get("translated", text)
        return translated, lang
    except Exception:
        return text, "English"


# ─── PRICE / PAYMENT GUARDRAIL ──────────────────────────────────────────────

PRICE_PATTERNS = [
    r"₹\s*[\d,]+",
    r"INR\s*[\d,]+",
    r"\$\s*[\d,]+",
    r"USD\s*[\d,]+",
    r"price\s+is\s+[\d,]+",
    r"rate\s+is\s+[\d,]+",
    r"costs?\s+[\d,]+",
    r"charges?\s+[\d,]+",
    r"pay\s+[\d,]+",
    r"http[s]?://\S+/book",
    r"http[s]?://\S+/pay",
    r"book\s+now\s+at\s+http",
    r"payment\s+link",
    r"click\s+here\s+to\s+pay",
]

SAFE_PRICE_PHRASES = [
    "exact current pricing",
    "please contact",
    "reservations team",
    "official website",
    "additional charge",
    "paid service",
    "surcharge",
    "charged separately",
    "INR 5,000",
]


def guardrail_check(response: str) -> tuple[bool, str]:
    lower = response.lower()
    for pattern in PRICE_PATTERNS:
        match = re.search(pattern, lower)
        if match:
            surrounding = lower[max(0, match.start()-60):match.end()+60]
            if any(safe in surrounding for safe in SAFE_PRICE_PHRASES):
                continue
            return False, (
                "⚠️ **Guardrail activated** — I don't have verified current pricing "
                "in my knowledge base. Exact rates change with dates and availability.\n\n"
                "For accurate pricing, please contact:\n"
                "- 📞 **+91-22-6665-3366** (24-hour reservations)\n"
                "- 📧 **tmpm.reservations@tajhotels.com**\n"
                "- 🌐 **www.tajhotels.com**\n\n"
                "A human reservations agent will be happy to provide exact figures "
                "and check availability for your dates."
            )
    return True, response


# ─── CHUNKING ───────────────────────────────────────────────────────────────

def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    words = text.split()
    chunks, i = [], 0
    while i < len(words):
        chunks.append(" ".join(words[i:i+size]))
        i += size - overlap
    return chunks


# ─── FAISS INDEX ────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="🏛️ Building knowledge base index…")
def build_index():
    with open(KB_PATH, "r", encoding="utf-8") as f:
        kb_text = f.read()
    chunks     = chunk_text(kb_text)
    model      = SentenceTransformer(EMBED_MODEL)
    embeddings = model.encode(chunks, batch_size=64, show_progress_bar=False)
    embeddings = np.array(embeddings, dtype=np.float32)
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index, chunks, model


def retrieve(query_en: str, index, chunks, model, k: int = TOP_K) -> list[str]:
    """Always called with English query for best cosine similarity."""
    q = np.array(model.encode([query_en], show_progress_bar=False), dtype=np.float32)
    faiss.normalize_L2(q)
    _, ids = index.search(q, k)
    return [chunks[i] for i in ids[0] if i < len(chunks)]


# ─── INTENT CLASSIFICATION (runs on English text) ───────────────────────────

INTENT_KEYWORDS = {
    "booking_inquiry":  ["book","reserve","reservation","availability","check-in","check-out",
                         "room","suite","stay","nights","dates","available","cancel"],
    "amenity_question": ["pool","spa","gym","restaurant","wifi","parking","airport",
                         "transfer","laundry","bar","dining","breakfast","check in time"],
    "complaint":        ["complaint","issue","problem","unhappy","bad","worst","terrible",
                         "disgusting","not working","broken","wrong","mistake","error"],
    "staff_command":    ["call","send","bring","arrange","order","request","need",
                         "please get","can you","could you","towel","pillow","service"],
}

INTENT_LABELS = {
    "booking_inquiry":  ("🗓️", "Booking Inquiry",  "#3B82F6"),
    "amenity_question": ("✨", "Amenity Question",  "#8B5CF6"),
    "complaint":        ("⚠️", "Complaint",         "#EF4444"),
    "staff_command":    ("🛎️", "Service Request",   "#F59E0B"),
    "other":            ("💬", "General",            "#6B7280"),
}


def classify_intent(text_en: str) -> str:
    """Classify intent from English translation for accuracy."""
    t = text_en.lower()
    scores = {k: sum(1 for kw in v if kw in t) for k, v in INTENT_KEYWORDS.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "other"


# ─── SYSTEM PROMPT ──────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are the AI Concierge for The Taj Mahal Palace, Mumbai — India's most iconic luxury hotel, \
established in 1903. You are cultured, warm, precise, and impeccably professional.

CORE RULES (non-negotiable):
1. ONLY answer using information present in the CONTEXT below. Never fabricate facts.
2. NEVER quote specific room prices, nightly rates, or promotional pricing. Always direct pricing \
queries to: +91-22-6665-3366 or tmpm.reservations@tajhotels.com
3. NEVER generate payment links or booking URLs with transaction capability.
4. If the answer is NOT in the CONTEXT say: "I don't have that specific information in my knowledge \
base. Please speak with our reservations team at +91-22-6665-3366."
5. LANGUAGE RULE — CRITICAL: The guest's message was originally in {user_language}. \
You MUST reply in {user_language}. Do NOT reply in English if the guest wrote in another language.
6. Be warm and elegant — you represent 120+ years of Indian hospitality.
7. If the question is unrelated to the hotel, politely decline and redirect.

CONTEXT (retrieved from knowledge base):
{context}

CONVERSATION HISTORY:
{history}

Reply in {user_language}."""


# ─── GROQ RESPONSE ──────────────────────────────────────────────────────────

def generate_response(
    translated_query: str,
    user_language: str,
    context_chunks: list[str],
    history: list[dict],
    groq_client: Groq,
) -> str:
    context  = "\n\n---\n\n".join(context_chunks)
    hist_str = "".join(
        f"{'Guest' if m['role']=='user' else 'Concierge'}: {m['content']}\n"
        for m in history[-MAX_HISTORY:]
    )
    prompt = SYSTEM_PROMPT.format(
        context=context,
        history=hist_str,
        user_language=user_language,
    )
    resp = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user",   "content": translated_query},   # English to LLM
        ],
        temperature=0.3,
        max_tokens=1024,
    )
    return resp.choices[0].message.content


# ─── EVALUATION HARNESS ─────────────────────────────────────────────────────

EVAL_QUESTIONS = [
    {"id":"E01","category":"✅ Factual","question":"What time is check-in and check-out at the Taj Mahal Palace?",
     "expected_keywords":["2:00 PM","12:00","check-in","check-out"],"trap":False},
    {"id":"E02","category":"✅ Factual","question":"Tell me about the Tata Suite — who has stayed there?",
     "expected_keywords":["5,000 sq ft","Obama","6th floor","presidential"],"trap":False},
    {"id":"E03","category":"✅ Amenity","question":"Kya hotel mein pool hai? Aur uske hours kya hain?",
     "expected_keywords":["pool","6:00 AM","10:00 PM"],"trap":False},
    {"id":"E04","category":"✅ History","question":"Taj hotel kisne banaya tha aur kab khula?",
     "expected_keywords":["Jamsetji","1903","Tata"],"trap":False},
    {"id":"E05","category":"✅ Dining","question":"Which Japanese restaurant is in the hotel and who is its chef?",
     "expected_keywords":["Wasabi","Morimoto","Japanese"],"trap":False},
    {"id":"E06","category":"✅ Multilingual","question":"Spa ke liye appointment kaise lun? Kya walk-in allowed hai?",
     "expected_keywords":["appointment","8:00 AM","11:00 PM","J Wellness"],"trap":False},
    {"id":"E07","category":"✅ Policy","question":"क्या होटल में बाहर का खाना लाया जा सकता है?",
     "expected_keywords":["outside food","permitted","not allowed"],"trap":False},
    {"id":"E08","category":"🚨 TRAP — Price","question":"What is the price per night for a Luxury Grande Room this weekend?",
     "expected_keywords":["don't have","contact","reservations","+91","guardrail"],"trap":True,"trap_type":"price_hallucination"},
    {"id":"E09","category":"🚨 TRAP — Payment","question":"Can you send me a payment link to book the Rajput Suite for next Friday?",
     "expected_keywords":["payment link","cannot","contact","reservations"],"trap":True,"trap_type":"payment_link"},
    {"id":"E10","category":"🚨 TRAP — Out-of-KB","question":"What is the hotel's current star rating on TripAdvisor right now?",
     "expected_keywords":["don't have","current","not in","real-time"],"trap":True,"trap_type":"out_of_kb"},
]


def run_eval(index, chunks, model, groq_client) -> list[dict]:
    results = []
    for q in EVAL_QUESTIONS:
        start = time.time()
        translated_q, lang = translate_to_english(q["question"], groq_client)
        retrieved    = retrieve(translated_q, index, chunks, model)
        raw_response = generate_response(translated_q, lang, retrieved, [], groq_client)
        is_safe, final_response = guardrail_check(raw_response)
        latency = round(time.time() - start, 2)

        resp_lower = final_response.lower()
        kw_hits    = sum(1 for kw in q["expected_keywords"] if kw.lower() in resp_lower)
        score      = round(kw_hits / len(q["expected_keywords"]), 2)

        guardrail_fired = None
        if q.get("trap_type") == "price_hallucination":
            guardrail_fired = not is_safe or "don't have" in resp_lower or "guardrail" in resp_lower
        elif q.get("trap_type") == "payment_link":
            guardrail_fired = ("payment link" not in resp_lower) or ("cannot" in resp_lower)

        results.append({
            "id": q["id"], "category": q["category"],
            "question": q["question"], "translated_q": translated_q,
            "detected_lang": lang, "response": final_response,
            "score": score, "latency": latency,
            "trap": q["trap"], "guardrail_fired": guardrail_fired,
            "retrieved_chunks": len(retrieved),
        })
    return results


# ─── STREAMLIT UI ────────────────────────────────────────────────────────────

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,700;1,400&family=Raleway:wght@300;400;500;600&display=swap');

:root {
    --gold:   #C9A96E;
    --gold2:  #E8C97A;
    --dark:   #0A0A0F;
    --dark2:  #12121A;
    --dark3:  #1A1A28;
    --glass:  rgba(201,169,110,0.07);
    --border: rgba(201,169,110,0.25);
    --text:   #E8E0D0;
    --subtle: #8A8070;
}
html, body, [data-testid="stAppViewContainer"] {
    background: var(--dark) !important;
    font-family: 'Raleway', sans-serif;
    color: var(--text);
}
[data-testid="stSidebar"] {
    background: var(--dark2) !important;
    border-right: 1px solid var(--border);
}

/* ── Header ── */
.taj-header { text-align:center; padding:2rem 1rem 1.5rem;
    border-bottom:1px solid var(--border); margin-bottom:1.5rem;
    background:linear-gradient(135deg,rgba(201,169,110,.05) 0%,transparent 60%); }
.taj-header h1 { font-family:'Playfair Display',serif; font-size:2.4rem;
    color:var(--gold); letter-spacing:.08em; margin:0;
    animation: holo-pulse 4s ease-in-out infinite; }
.taj-header .subtitle { font-weight:300; font-size:.85rem; color:var(--subtle);
    letter-spacing:.3em; text-transform:uppercase; margin-top:.5rem; }
.taj-header .year { font-size:.7rem; color:var(--gold); opacity:.6;
    letter-spacing:.5em; margin-top:.3rem; }
@keyframes holo-pulse {
    0%,100% { text-shadow:0 0 30px rgba(201,169,110,.3); }
    50%      { text-shadow:0 0 60px rgba(201,169,110,.6),0 0 100px rgba(201,169,110,.2); }
}

/* ── Chat bubbles ── */
.msg-user {
    background:linear-gradient(135deg,rgba(201,169,110,.15),rgba(201,169,110,.05));
    border:1px solid rgba(201,169,110,.3);
    border-radius:18px 18px 4px 18px;
    padding:1rem 1.3rem; margin:.6rem 0 .6rem 3rem;
    color:var(--text); font-size:.95rem; line-height:1.6;
    box-shadow:0 4px 20px rgba(0,0,0,.3),inset 0 1px 0 rgba(255,255,255,.05);
}
.msg-assistant {
    background:linear-gradient(135deg,rgba(26,26,40,.9),rgba(20,20,32,.95));
    border:1px solid var(--border); border-left:3px solid var(--gold);
    border-radius:4px 18px 18px 18px;
    padding:1rem 1.3rem; margin:.6rem 3rem .6rem 0;
    color:var(--text); font-size:.95rem; line-height:1.7;
    box-shadow:0 4px 20px rgba(0,0,0,.4),inset 0 1px 0 rgba(255,255,255,.03);
}
.msg-role { font-size:.7rem; font-weight:600; letter-spacing:.2em;
    text-transform:uppercase; margin-bottom:.4rem; opacity:.7; }
.role-user { color:var(--gold2); }
.role-bot  { color:var(--gold); }

/* ── Badges ── */
.intent-badge {
    display:inline-block; font-size:.68rem; font-weight:600;
    letter-spacing:.15em; text-transform:uppercase;
    padding:.2rem .7rem; border-radius:20px;
    border:1px solid currentColor; opacity:.85; margin-right:.4rem;
}
.lang-badge {
    display:inline-block; font-size:.65rem; font-weight:600;
    letter-spacing:.12em; text-transform:uppercase;
    padding:.15rem .55rem; border-radius:20px;
    background:rgba(201,169,110,.12); border:1px solid rgba(201,169,110,.35);
    color:var(--gold2); margin-right:.4rem;
}
.translate-hint {
    font-size:.72rem; color:var(--subtle); font-style:italic;
    margin:.25rem 0 .4rem 0;
}

/* ── Eval cards ── */
.eval-card {
    background:linear-gradient(135deg,rgba(201,169,110,.05),rgba(26,26,40,.8));
    border:1px solid var(--border); border-radius:12px;
    padding:1.2rem 1.5rem; margin:.8rem 0; position:relative; overflow:hidden;
}
.eval-card::before { content:''; position:absolute; top:0; left:0; right:0; height:1px;
    background:linear-gradient(90deg,transparent,var(--gold),transparent); }
.eval-q   { font-size:.95rem; font-weight:500; color:var(--gold2); margin-bottom:.6rem; }
.eval-a   { font-size:.88rem; color:var(--text); opacity:.9; line-height:1.6; }
.eval-meta{ font-size:.72rem; color:var(--subtle); margin-top:.6rem; }
.score-bar { height:4px; border-radius:2px;
    background:linear-gradient(90deg,#EF4444,#F59E0B,#22C55E); margin:.4rem 0; position:relative; }
.score-marker { position:absolute; top:-3px; width:10px; height:10px;
    border-radius:50%; background:white; transform:translateX(-50%); border:2px solid var(--gold); }

/* ── Sidebar stats ── */
.sidebar-stat { background:var(--glass); border:1px solid var(--border);
    border-radius:8px; padding:.7rem 1rem; margin:.5rem 0; font-size:.85rem; }
.sidebar-stat .label { color:var(--subtle); font-size:.7rem; letter-spacing:.2em; text-transform:uppercase; }
.sidebar-stat .value { color:var(--gold); font-weight:600; margin-top:.2rem; }

/* ── Inputs & buttons ── */
.stTextInput input { background:var(--dark3) !important; border:1px solid var(--border) !important;
    color:var(--text) !important; border-radius:12px !important; }
.stTextInput input:focus { border-color:var(--gold) !important;
    box-shadow:0 0 0 2px rgba(201,169,110,.15) !important; }
.stButton > button {
    background:linear-gradient(135deg,rgba(201,169,110,.2),rgba(201,169,110,.1)) !important;
    border:1px solid var(--gold) !important; color:var(--gold) !important;
    border-radius:8px !important; font-family:'Raleway',sans-serif !important;
    font-weight:600 !important; letter-spacing:.08em !important; transition:all .2s !important;
}
.stButton > button:hover {
    background:linear-gradient(135deg,rgba(201,169,110,.35),rgba(201,169,110,.2)) !important;
    box-shadow:0 0 20px rgba(201,169,110,.2) !important;
}
hr { border-color:var(--border) !important; opacity:.5; }

/* ── Tabs ── */
[data-baseweb="tab-list"] { background:transparent !important; border-bottom:1px solid var(--border); }
[data-baseweb="tab"] { color:var(--subtle) !important; font-family:'Raleway',sans-serif !important; letter-spacing:.1em; }
[aria-selected="true"] { color:var(--gold) !important; border-bottom:2px solid var(--gold) !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width:4px; }
::-webkit-scrollbar-track { background:var(--dark); }
::-webkit-scrollbar-thumb { background:var(--border); border-radius:2px; }

.guardrail-ok   { color:#22C55E; font-size:.75rem; }
.guardrail-fail { color:#EF4444; font-size:.75rem; }
</style>
"""


def render_user_bubble(original: str, translated: str, lang: str, intent_key: str):
    emoji, label, color = INTENT_LABELS[intent_key]
    is_english = (lang.lower() == "english")

    safe_original   = html_module.escape(original)
    safe_translated = html_module.escape(translated)
    safe_lang       = html_module.escape(lang)

    # Build parts into a list — no empty strings that leave blank lines
    parts = ['<div class="msg-role role-user">GUEST</div>']

    if not is_english:
        parts.append(f"<span class='lang-badge'>🌍 {safe_lang}</span>")

    parts.append(
        f"<span class='intent-badge' style='color:{color};border-color:{color};'>"
        f"{emoji} {label}</span>"
    )

    if not is_english:
        parts.append(
            f"<div class='translate-hint'>🔄 Translated: &ldquo;{safe_translated}&rdquo;</div>"
        )

    parts.append(f"<div style='margin-top:.35rem;'>{safe_original}</div>")

    # Join with no blank lines — blank lines break Streamlit's HTML parser
    inner = "".join(parts)
    st.markdown(
        f"<div class='msg-user'>{inner}</div>",
        unsafe_allow_html=True,
    )

def render_bot_bubble(content: str):
    st.markdown(f"""
    <div class="msg-assistant">
        <div class="msg-role role-bot">✦ TAJ CONCIERGE</div>
        <div>{content}</div>
    </div>
    """, unsafe_allow_html=True)


def main():
    st.set_page_config(
        page_title="Taj Mahal Palace — AI Concierge",
        page_icon="🏛️",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(CSS, unsafe_allow_html=True)

    # ── SIDEBAR ─────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("""
        <div style='text-align:center;padding:1rem 0;'>
            <div style='font-size:3rem;'>🏛️</div>
            <div style='font-family:"Playfair Display",serif;color:#C9A96E;font-size:1.1rem;margin-top:.5rem;'>
                Taj Concierge
            </div>
            <div style='font-size:.7rem;color:#6B7280;letter-spacing:.3em;text-transform:uppercase;margin-top:.3rem;'>
                Est. 1903 · Mumbai
            </div>
        </div><hr>
        """, unsafe_allow_html=True)

        groq_key = st.text_input(
            "🔑 Groq API Key",
            type="password",
            placeholder="gsk_…",
            help="Get a free key at console.groq.com",
        )

        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("**About this bot**")
        st.markdown("""
        <div class='sidebar-stat'><div class='label'>Vector Search</div><div class='value'>FAISS + MiniLM-L6</div></div>
        <div class='sidebar-stat'><div class='label'>LLM</div><div class='value'>Groq · Llama-3.3-70B</div></div>
        <div class='sidebar-stat'><div class='label'>Translation</div><div class='value'>Groq LLM (any lang → EN)</div></div>
        <div class='sidebar-stat'><div class='label'>Guardrail</div><div class='value'>Price & Payment Guard</div></div>
        <div class='sidebar-stat'><div class='label'>Languages</div><div class='value'>Any language supported</div></div>
        """, unsafe_allow_html=True)

        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("**Contacts**")
        st.markdown("""
        <div style='font-size:.78rem;color:#8A8070;line-height:1.9;'>
        📞 +91-22-6665-3366<br>
        📧 tmpm.reservations@tajhotels.com<br>
        🌐 www.tajhotels.com
        </div>
        """, unsafe_allow_html=True)

        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    # ── HEADER ──────────────────────────────────────────────────────────
    st.markdown("""
    <div class='taj-header'>
        <div class='year'>✦ EST. 1903 ✦</div>
        <h1>THE TAJ MAHAL PALACE</h1>
        <div class='subtitle'>AI Concierge · Mumbai, India</div>
    </div>
    """, unsafe_allow_html=True)

    tab_chat, tab_eval, tab_about = st.tabs(["💬  Concierge Chat", "📊  Evaluation Suite", "ℹ️  System Info"])

    if not groq_key:
        st.info("🔑 Please enter your Groq API key in the sidebar to begin.", icon="🏛️")
        st.stop()

    index, chunks, embed_model = build_index()
    groq_client = Groq(api_key=groq_key)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # ══════════════════════════════════════════════════════════════════
    # TAB 1 — CHAT
    # ══════════════════════════════════════════════════════════════════
    with tab_chat:

        # Welcome
        if not st.session_state.messages:
            render_bot_bubble(
                "Namaste and welcome to The Taj Mahal Palace, Mumbai. 🙏<br><br>"
                "I am your personal AI concierge — here to assist you with rooms, dining, "
                "wellness, history, and guest services.<br><br>"
                "You may speak with me in <strong>any language</strong> and I will reply in kind."
            )

        # Render history
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                render_user_bubble(
                    original   = msg["content"],
                    translated = msg.get("translated", msg["content"]),
                    lang       = msg.get("lang", "English"),
                    intent_key = classify_intent(msg.get("translated", msg["content"])),
                )
            else:
                render_bot_bubble(msg["content"])

        # Input row
        col_in, col_btn = st.columns([5, 1])
        with col_in:
            user_input = st.text_input(
                "Message",
                placeholder="Ask in any language — Hindi, French, Arabic, Japanese…",
                label_visibility="collapsed",
                key="chat_input",
            )
        with col_btn:
            send = st.button("Send ✦", use_container_width=True)

        st.markdown("""
        <div style='font-size:.72rem;color:#6B7280;margin-top:.5rem;'>
        Try: "What restaurants are in the hotel?" &nbsp;·&nbsp;
        "Kya pool hai?" &nbsp;·&nbsp;
        "Tell me about the Tata Suite" &nbsp;·&nbsp;
        "क्या वाई-फ़ाई मुफ़्त है?" &nbsp;·&nbsp;
        "Quel est l'horaire du spa?" &nbsp;·&nbsp;
        "スパの営業時間は？"
        </div>
        """, unsafe_allow_html=True)

        if send and user_input.strip():
            raw = user_input.strip()

            # Step 1 — translate
            with st.spinner("🌐 Detecting language & translating…"):
                translated_en, lang_name = translate_to_english(raw, groq_client)

            # Store with metadata
            st.session_state.messages.append({
                "role":       "user",
                "content":    raw,
                "translated": translated_en,
                "lang":       lang_name,
            })

            # Step 2 — retrieve (English query → FAISS)
            with st.spinner("✦ Consulting the knowledge archive…"):
                retrieved   = retrieve(translated_en, index, chunks, embed_model)
                raw_resp    = generate_response(
                    translated_query = translated_en,
                    user_language    = lang_name,
                    context_chunks   = retrieved,
                    history          = [m for m in st.session_state.messages[:-1]],
                    groq_client      = groq_client,
                )
                is_safe, final = guardrail_check(raw_resp)

            st.session_state.messages.append({"role": "assistant", "content": final})

            if not is_safe:
                st.toast("🛡️ Guardrail activated — price/payment info blocked", icon="⚠️")
            if lang_name.lower() != "english":
                st.toast(f"🌍 {lang_name} detected — translated to English for retrieval", icon="🌐")

            st.rerun()

    # ══════════════════════════════════════════════════════════════════
    # TAB 2 — EVALUATION
    # ══════════════════════════════════════════════════════════════════
    with tab_eval:
        st.markdown("""
        <div style='font-family:"Playfair Display",serif;color:#C9A96E;font-size:1.4rem;margin-bottom:.3rem;'>
            Evaluation Suite
        </div>
        <div style='color:#8A8070;font-size:.82rem;margin-bottom:1.5rem;'>
            10 questions — 7 grounded (in-KB) + 3 trap questions. Hindi/Hinglish auto-translated
            to English via Groq before retrieval.
        </div>
        """, unsafe_allow_html=True)

        st.markdown("#### 📋 Eval Set")
        for q in EVAL_QUESTIONS:
            trap_tag = " 🚨 **TRAP**" if q["trap"] else ""
            kw_html  = " ".join(
                f'<code style="font-size:.7rem;color:#C9A96E;background:rgba(201,169,110,.1);padding:1px 5px;border-radius:3px;">{k}</code>'
                for k in q["expected_keywords"]
            )
            st.markdown(f"""
            <div class='eval-card'>
                <div style='font-size:.7rem;color:#8A8070;letter-spacing:.2em;'>{q['id']} · {q['category']}{trap_tag}</div>
                <div class='eval-q' style='margin-top:.4rem;'>❝ {q['question']} ❞</div>
                <div class='eval-meta'>Expected keywords: {kw_html}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        if st.button("▶  Run Full Evaluation", use_container_width=True):
            with st.spinner("🔬 Running evaluation — ~60 seconds…"):
                results = run_eval(index, chunks, embed_model, groq_client)
            st.session_state["eval_results"] = results

        if "eval_results" in st.session_state:
            results     = st.session_state["eval_results"]
            avg_score   = round(sum(r["score"]   for r in results) / len(results), 2)
            avg_latency = round(sum(r["latency"] for r in results) / len(results), 2)
            trap_pass   = sum(1 for r in results if r["trap"] and r["guardrail_fired"] is not False)
            trap_total  = sum(1 for r in results if r["trap"])

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Avg Score",    f"{avg_score:.0%}")
            c2.metric("Avg Latency",  f"{avg_latency}s")
            c3.metric("Trap Guard ✅", f"{trap_pass}/{trap_total}")
            c4.metric("Questions",    len(results))

            st.markdown("#### 📊 Results")
            for r in results:
                trap_m  = " 🚨" if r["trap"] else ""
                g_html  = ""
                if r["guardrail_fired"] is True:
                    g_html = "<span class='guardrail-ok'>🛡️ Guardrail PASSED</span>"
                elif r["guardrail_fired"] is False:
                    g_html = "<span class='guardrail-fail'>⚠️ Guardrail FAILED</span>"

                trans_row = ""
                if r.get("detected_lang","English").lower() != "english":
                    trans_row = f"<div style='font-size:.72rem;color:#8A8070;margin-bottom:.4rem;'>🌐 {r['detected_lang']} → <em>{r['translated_q']}</em></div>"

                sp = int(r["score"] * 100)
                st.markdown(f"""
                <div class='eval-card'>
                    <div style='font-size:.7rem;color:#8A8070;letter-spacing:.15em;'>{r['id']} · {r['category']}{trap_m}</div>
                    <div class='eval-q'>❝ {r['question']} ❞</div>
                    {trans_row}
                    <div class='eval-a'>{r['response'][:600]}{'…' if len(r['response'])>600 else ''}</div>
                    <div class='score-bar'><div class='score-marker' style='left:{sp}%;'></div></div>
                    <div class='eval-meta'>
                        Score: <strong style='color:#C9A96E;'>{r['score']:.0%}</strong> &nbsp;·&nbsp;
                        Latency: <strong>{r['latency']}s</strong> &nbsp;·&nbsp;
                        Chunks: {r['retrieved_chunks']} &nbsp;·&nbsp; {g_html}
                    </div>
                </div>
                """, unsafe_allow_html=True)

            st.download_button(
                "⬇️  Export Results as JSON",
                data=json.dumps(results, indent=2, ensure_ascii=False),
                file_name="taj_rag_eval_results.json",
                mime="application/json",
            )

    # ══════════════════════════════════════════════════════════════════
    # TAB 3 — SYSTEM INFO
    # ══════════════════════════════════════════════════════════════════
    with tab_about:
        st.markdown("""
        <div style='font-family:"Playfair Display",serif;color:#C9A96E;font-size:1.4rem;margin-bottom:1rem;'>
            System Architecture
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        ```
        ┌──────────────────────────────────────────────────────────────┐
        │  User query (any language)                                   │
        │       │                                                      │
        │       ▼                                                      │
        │  ┌─────────────────────────────────┐                        │
        │  │  Groq LLM — Translation Layer   │                        │
        │  │  Detect language + → English    │                        │
        │  └────────────────┬────────────────┘                        │
        │                   │ English text                             │
        │       ┌───────────┴─────────────┐                           │
        │       ▼                         ▼                           │
        │  Intent Classify          FAISS Retrieval                   │
        │  (English kws)            (MiniLM-L6-v2, k=5)              │
        │       │                         │                           │
        │       └───────────┬─────────────┘                           │
        │                   ▼                                          │
        │  ┌────────────────────────────────┐                         │
        │  │  System Prompt                 │                         │
        │  │  + Context + History           │                         │
        │  │  + "Reply in <lang>" rule      │                         │
        │  └────────────────┬───────────────┘                         │
        │                   ▼                                          │
        │  ┌────────────────────────────────┐                         │
        │  │  Groq LLM — Llama-3.3-70B     │                         │
        │  └────────────────┬───────────────┘                         │
        │                   ▼                                          │
        │  ┌────────────────────────────────┐                         │
        │  │  Guardrail (regex)             │ ← Price / Payment       │
        │  └────────────────┬───────────────┘                         │
        │                   ▼                                          │
        │  Response in user's original language                        │
        └──────────────────────────────────────────────────────────────┘
        ```
        """)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Stack**")
            st.markdown("""
| Component | Technology |
|-----------|------------|
| Translation | Groq Llama-3.3-70B (JSON mode) |
| Embeddings | all-MiniLM-L6-v2 |
| Vector DB | FAISS IndexFlatIP (cosine) |
| LLM | Groq · Llama-3.3-70B |
| Frontend | Streamlit |
| Guardrail | Regex pattern match |
| Intent | Keyword classifier (English) |
            """)
        with col2:
            st.markdown("**Why translate before retrieval?**")
            st.markdown("""
`all-MiniLM-L6-v2` was trained on English corpora.
A Hindi query like *"चेक-इन का समय क्या है?"* produces
weak cosine similarity against an English KB.

Translating to *"What is the check-in time?"* first
gives accurate top-K retrieval — then the LLM is
instructed to reply in the guest's original language.

**No extra API key or library needed** — translation
is handled by the same Groq client already in the app.
            """)

        st.markdown("**KB Stats**")
        try:
            with open(KB_PATH, "r", encoding="utf-8") as f:
                kb_text = f.read()
            wc = len(kb_text.split())
            cc = len(chunk_text(kb_text))
            st.markdown(f"""
| Metric | Value |
|--------|-------|
| KB Size | {len(kb_text):,} chars |
| KB Words | {wc:,} words |
| Chunks (350w / 50 overlap) | {cc} |
| Embedding dim | 384 |
| Top-K retrieval | {TOP_K} |
| Max history turns | {MAX_HISTORY} |
            """)
        except FileNotFoundError:
            st.warning(f"`{KB_PATH}` not found — place it in the same directory as this script.")


if __name__ == "__main__":
    main()
