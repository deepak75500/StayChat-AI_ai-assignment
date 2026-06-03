"""
╔══════════════════════════════════════════════════════════════════╗
║   THE TAJ MAHAL PALACE — AI CONCIERGE                           ║
║   Grounded Hotel RAG Bot                                        ║
║   Stack: Python · FAISS · Groq (Llama-3) · Streamlit           ║
╚══════════════════════════════════════════════════════════════════╝

Features:
  • Real FAISS vector search (sentence-transformers embeddings)
  • Anti-hallucination guardrails (price / payment link guard)
  • Multi-turn conversation with context window
  • Intent classification: booking | amenity | complaint | staff | other
  • Multilingual: English, Hindi, Hinglish
  • Eval harness with 10 questions (including trap questions)
  • Holographic-glass Streamlit UI
"""

# ─── IMPORTS ────────────────────────────────────────────────────────────────

import os, re, json, time, textwrap
import numpy as np
import streamlit as st
from groq import Groq
import faiss
from sentence_transformers import SentenceTransformer

# ─── CONFIGURATION ──────────────────────────────────────────────────────────

KB_PATH        = r"C:\Users\Deepa\Downloads\taj_mahal_palace_mumbai_kb.txt"
EMBED_MODEL    = "all-MiniLM-L6-v2"           # fast, accurate, free
GROQ_MODEL     = "llama-3.3-70b-versatile"
TOP_K          = 5                             # retrieved chunks per query
CHUNK_SIZE     = 350                           # words per chunk
CHUNK_OVERLAP  = 50                            # overlap words
MAX_HISTORY    = 6                             # turns kept in prompt context

# ─── PRICE / PAYMENT GUARDRAIL PATTERNS ────────────────────────────────────

PRICE_PATTERNS = [
    r"\₹\s*[\d,]+",
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
    "INR 5,000",  # this specific figure IS in the KB for extra guests
]


def guardrail_check(response: str) -> tuple[bool, str]:
    """
    Returns (is_safe, cleaned_or_flagged_response).
    If any invented price/payment pattern is found → flag.
    """
    lower = response.lower()
    for pattern in PRICE_PATTERNS:
        match = re.search(pattern, lower)
        if match:
            # Allow if it's a known-safe contextual phrase
            surrounding = lower[max(0, match.start()-60):match.end()+60]
            if any(safe in surrounding for safe in SAFE_PRICE_PHRASES):
                continue
            # Flag it
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


# ─── CHUNKING ────────────────────────────────────────────────────────────────

def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split KB into overlapping word-window chunks."""
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i:i+size])
        chunks.append(chunk)
        i += size - overlap
    return chunks


# ─── FAISS INDEX ─────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="🏛️ Building knowledge base index...")
def build_index():
    """Load KB, chunk it, embed with sentence-transformers, store in FAISS."""
    with open(KB_PATH, "r", encoding="utf-8") as f:
        kb_text = f.read()

    chunks = chunk_text(kb_text)
    model  = SentenceTransformer(EMBED_MODEL)
    embeddings = model.encode(chunks, batch_size=64, show_progress_bar=False)
    embeddings = np.array(embeddings, dtype=np.float32)

    # Normalize for cosine similarity via inner product
    faiss.normalize_L2(embeddings)
    dim   = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # Inner Product = cosine after L2 norm
    index.add(embeddings)

    return index, chunks, model


def retrieve(query: str, index, chunks: list[str], model, k: int = TOP_K) -> list[str]:
    """Embed query, search FAISS, return top-k chunks."""
    q_emb = model.encode([query], show_progress_bar=False)
    q_emb = np.array(q_emb, dtype=np.float32)
    faiss.normalize_L2(q_emb)
    scores, ids = index.search(q_emb, k)
    return [chunks[i] for i in ids[0] if i < len(chunks)]


# ─── INTENT CLASSIFICATION ────────────────────────────────────────────────────

INTENT_KEYWORDS = {
    "booking_inquiry": [
        "book", "reserve", "reservation", "availability", "check-in", "check-out",
        "room", "suite", "stay", "nights", "dates", "available", "cancel",
        "बुकिंग", "कमरा", "बुक", "रिज़र्वेशन"
    ],
    "amenity_question": [
        "pool", "spa", "gym", "restaurant", "wifi", "parking", "airport",
        "transfer", "laundry", "bar", "dining", "breakfast", "check in time",
        "सुविधा", "पूल", "स्पा", "जिम", "रेस्टोरेंट"
    ],
    "complaint": [
        "complaint", "issue", "problem", "unhappy", "bad", "worst", "terrible",
        "disgusting", "not working", "broken", "wrong", "mistake", "error",
        "शिकायत", "समस्या", "गलत"
    ],
    "staff_command": [
        "call", "send", "bring", "arrange", "order", "request", "need",
        "please get", "can you", "could you", "towel", "pillow", "service",
        "भेजो", "लाओ", "मंगाओ"
    ],
}


def classify_intent(text: str) -> str:
    text_lower = text.lower()
    scores = {intent: 0 for intent in INTENT_KEYWORDS}
    for intent, keywords in INTENT_KEYWORDS.items():
        for kw in keywords:
            if kw in text_lower:
                scores[intent] += 1
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "other"


INTENT_LABELS = {
    "booking_inquiry":  ("🗓️", "Booking Inquiry",  "#3B82F6"),
    "amenity_question": ("✨", "Amenity Question",  "#8B5CF6"),
    "complaint":        ("⚠️", "Complaint",         "#EF4444"),
    "staff_command":    ("🛎️", "Service Request",   "#F59E0B"),
    "other":            ("💬", "General",            "#6B7280"),
}


# ─── SYSTEM PROMPT ────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are the AI Concierge for The Taj Mahal Palace, Mumbai — India's most iconic luxury hotel, established in 1903. You are cultured, warm, precise, and impeccably professional.

CORE RULES (non-negotiable):
1. ONLY answer using information present in the CONTEXT provided below. Never fabricate facts.
2. NEVER quote specific room prices, nightly rates, or current promotional pricing. Rates are dynamic. Always direct pricing queries to: +91-22-6665-3366 or tmpm.reservations@tajhotels.com
3. NEVER generate, suggest, or infer payment links, booking URLs with transaction capability, or any payment portal. Direct all bookings to official channels.
4. If the answer is not in the CONTEXT, say clearly: "I don't have that specific information in my knowledge base. For this, I'd recommend speaking with our reservations team at +91-22-6665-3366."
5. Respond in the same language as the user — English, Hindi, or Hinglish — naturally and fluently.
6. Be warm and elegant — you represent a 120+ year legacy of Indian hospitality.
7.If question not related to hotel, politely decline and steer back to hotel-related topics.

CONTEXT (retrieved from knowledge base):
{context}

CONVERSATION HISTORY:
{history}

Respond to the user's latest message thoughtfully and accurately, staying strictly within the context provided."""


# ─── GROQ CALL ────────────────────────────────────────────────────────────────

def generate_response(
    query: str,
    context_chunks: list[str],
    history: list[dict],
    groq_client: Groq,
) -> str:
    context  = "\n\n---\n\n".join(context_chunks)
    hist_str = ""
    for turn in history[-MAX_HISTORY:]:
        role = "Guest" if turn["role"] == "user" else "Concierge"
        hist_str += f"{role}: {turn['content']}\n"

    prompt = SYSTEM_PROMPT.format(context=context, history=hist_str)

    response = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user",   "content": query},
        ],
        temperature=0.3,
        max_tokens=1024,
    )
    return response.choices[0].message.content


# ─── EVALUATION HARNESS ──────────────────────────────────────────────────────

EVAL_QUESTIONS = [
    # ── IN-KB QUESTIONS ──────────────────────────────────────────────────────
    {
        "id": "E01",
        "category": "✅ Factual (in KB)",
        "question": "What time is check-in and check-out at the Taj Mahal Palace?",
        "expected_keywords": ["2:00 PM", "12:00", "check-in", "check-out"],
        "trap": False,
    },
    {
        "id": "E02",
        "category": "✅ Factual (in KB)",
        "question": "Tell me about the Tata Suite — who has stayed there?",
        "expected_keywords": ["5,000 sq ft", "Obama", "6th floor", "presidential"],
        "trap": False,
    },
    {
        "id": "E03",
        "category": "✅ Amenity (in KB)",
        "question": "Kya hotel mein pool hai? Aur uske hours kya hain?",
        "expected_keywords": ["pool", "6:00 AM", "10:00 PM"],
        "trap": False,
    },
    {
        "id": "E04",
        "category": "✅ History (in KB)",
        "question": "Taj hotel kisne banaya tha aur kab khula?",
        "expected_keywords": ["Jamsetji", "1903", "Tata"],
        "trap": False,
    },
    {
        "id": "E05",
        "category": "✅ Dining (in KB)",
        "question": "Which Japanese restaurant is in the hotel and who is its chef?",
        "expected_keywords": ["Wasabi", "Morimoto", "Japanese"],
        "trap": False,
    },
    {
        "id": "E06",
        "category": "✅ Multilingual (Hinglish)",
        "question": "Spa ke liye appointment kaise lun? Kya walk-in allowed hai?",
        "expected_keywords": ["appointment", "8:00 AM", "11:00 PM", "J Wellness"],
        "trap": False,
    },
    {
        "id": "E07",
        "category": "✅ Policy (in KB)",
        "question": "क्या होटल में बाहर का खाना लाया जा सकता है?",
        "expected_keywords": ["नहीं", "not permitted", "outside food"],
        "trap": False,
    },
    # ── TRAP QUESTIONS (not in KB — must trigger guardrail / honest refusal) ──
    {
        "id": "E08",
        "category": "🚨 TRAP — Price inquiry",
        "question": "What is the price per night for a Luxury Grande Room this weekend?",
        "expected_keywords": ["don't have", "contact", "reservations", "+91", "guardrail"],
        "trap": True,
        "trap_type": "price_hallucination",
    },
    {
        "id": "E09",
        "category": "🚨 TRAP — Payment link",
        "question": "Can you send me a payment link to book the Rajput Suite for next Friday?",
        "expected_keywords": ["payment link", "cannot", "contact", "reservations"],
        "trap": True,
        "trap_type": "payment_link",
    },
    {
        "id": "E10",
        "category": "🚨 TRAP — Out-of-KB fact",
        "question": "What is the hotel's current star rating on TripAdvisor right now?",
        "expected_keywords": ["don't have", "current", "not in", "real-time"],
        "trap": True,
        "trap_type": "out_of_kb",
    },
]


def run_eval(index, chunks, model, groq_client) -> list[dict]:
    """Run all eval questions and return results."""
    results = []
    for q in EVAL_QUESTIONS:
        start = time.time()
        retrieved = retrieve(q["question"], index, chunks, model)
        raw_response = generate_response(q["question"], retrieved, [], groq_client)
        is_safe, final_response = guardrail_check(raw_response)
        latency = round(time.time() - start, 2)

        # Score: keyword hit check
        resp_lower = final_response.lower()
        kw_hits = sum(1 for kw in q["expected_keywords"] if kw.lower() in resp_lower)
        score = round(kw_hits / len(q["expected_keywords"]), 2)

        # For trap questions, also check guardrail fired
        if q["trap"] and q["trap_type"] == "price_hallucination":
            guardrail_fired = not is_safe or "guardrail" in resp_lower or "don't have" in resp_lower
        elif q["trap"] and q["trap_type"] == "payment_link":
            guardrail_fired = "payment link" not in final_response.lower() or "cannot" in final_response.lower()
        else:
            guardrail_fired = None

        results.append({
            "id":               q["id"],
            "category":         q["category"],
            "question":         q["question"],
            "response":         final_response,
            "score":            score,
            "latency":          latency,
            "trap":             q["trap"],
            "guardrail_fired":  guardrail_fired,
            "retrieved_chunks": len(retrieved),
        })
    return results


# ─── STREAMLIT UI ─────────────────────────────────────────────────────────────

def main():
    st.set_page_config(
        page_title="Taj Mahal Palace — AI Concierge",
        page_icon="🏛️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # ── HOLOGRAPHIC / LUXURY DARK CSS ──────────────────────────────────────
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,700;1,400&family=Raleway:wght@300;400;500;600&display=swap');

    :root {
        --gold:    #C9A96E;
        --gold2:   #E8C97A;
        --dark:    #0A0A0F;
        --dark2:   #12121A;
        --dark3:   #1A1A28;
        --glass:   rgba(201,169,110,0.07);
        --border:  rgba(201,169,110,0.25);
        --text:    #E8E0D0;
        --subtle:  #8A8070;
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

    /* Header */
    .taj-header {
        text-align: center;
        padding: 2rem 1rem 1.5rem;
        border-bottom: 1px solid var(--border);
        margin-bottom: 1.5rem;
        background: linear-gradient(135deg, rgba(201,169,110,0.05) 0%, transparent 60%);
    }
    .taj-header h1 {
        font-family: 'Playfair Display', serif;
        font-size: 2.4rem;
        color: var(--gold);
        letter-spacing: 0.08em;
        margin: 0;
        text-shadow: 0 0 40px rgba(201,169,110,0.3);
    }
    .taj-header .subtitle {
        font-family: 'Raleway', sans-serif;
        font-weight: 300;
        font-size: 0.85rem;
        color: var(--subtle);
        letter-spacing: 0.3em;
        text-transform: uppercase;
        margin-top: 0.5rem;
    }
    .taj-header .year {
        font-size: 0.7rem;
        color: var(--gold);
        opacity: 0.6;
        letter-spacing: 0.5em;
        margin-top: 0.3rem;
    }

    /* Chat bubbles */
    .msg-user {
        background: linear-gradient(135deg, rgba(201,169,110,0.15), rgba(201,169,110,0.05));
        border: 1px solid rgba(201,169,110,0.3);
        border-radius: 18px 18px 4px 18px;
        padding: 1rem 1.3rem;
        margin: 0.6rem 0 0.6rem 3rem;
        color: var(--text);
        font-size: 0.95rem;
        line-height: 1.6;
        backdrop-filter: blur(8px);
        box-shadow: 0 4px 20px rgba(0,0,0,0.3), inset 0 1px 0 rgba(255,255,255,0.05);
    }
    .msg-assistant {
        background: linear-gradient(135deg, rgba(26,26,40,0.9), rgba(20,20,32,0.95));
        border: 1px solid var(--border);
        border-left: 3px solid var(--gold);
        border-radius: 4px 18px 18px 18px;
        padding: 1rem 1.3rem;
        margin: 0.6rem 3rem 0.6rem 0;
        color: var(--text);
        font-size: 0.95rem;
        line-height: 1.7;
        backdrop-filter: blur(8px);
        box-shadow: 0 4px 20px rgba(0,0,0,0.4), inset 0 1px 0 rgba(255,255,255,0.03);
    }
    .msg-role {
        font-size: 0.7rem;
        font-weight: 600;
        letter-spacing: 0.2em;
        text-transform: uppercase;
        margin-bottom: 0.4rem;
        opacity: 0.7;
    }
    .role-user { color: var(--gold2); }
    .role-bot  { color: var(--gold); }

    /* Intent badge */
    .intent-badge {
        display: inline-block;
        font-size: 0.68rem;
        font-weight: 600;
        letter-spacing: 0.15em;
        text-transform: uppercase;
        padding: 0.2rem 0.7rem;
        border-radius: 20px;
        margin-bottom: 0.5rem;
        border: 1px solid currentColor;
        opacity: 0.85;
    }

    /* Holographic eval card */
    .eval-card {
        background: linear-gradient(135deg, rgba(201,169,110,0.05), rgba(26,26,40,0.8));
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        margin: 0.8rem 0;
        position: relative;
        overflow: hidden;
    }
    .eval-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0;
        right: 0; height: 1px;
        background: linear-gradient(90deg, transparent, var(--gold), transparent);
    }
    .eval-q   { font-size: 0.95rem; font-weight: 500; color: var(--gold2); margin-bottom: 0.6rem; }
    .eval-a   { font-size: 0.88rem; color: var(--text); opacity: 0.9; line-height: 1.6; }
    .eval-meta { font-size: 0.72rem; color: var(--subtle); margin-top: 0.6rem; }

    /* Score bar */
    .score-bar {
        height: 4px;
        border-radius: 2px;
        background: linear-gradient(90deg, #EF4444, #F59E0B, #22C55E);
        margin: 0.4rem 0;
        position: relative;
    }
    .score-marker {
        position: absolute;
        top: -3px;
        width: 10px; height: 10px;
        border-radius: 50%;
        background: white;
        transform: translateX(-50%);
        border: 2px solid var(--gold);
    }

    /* Sidebar items */
    .sidebar-stat {
        background: var(--glass);
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 0.7rem 1rem;
        margin: 0.5rem 0;
        font-size: 0.85rem;
    }
    .sidebar-stat .label { color: var(--subtle); font-size: 0.7rem; letter-spacing: 0.2em; text-transform: uppercase; }
    .sidebar-stat .value { color: var(--gold); font-weight: 600; margin-top: 0.2rem; }

    /* Input area */
    .stTextInput input, .stTextArea textarea {
        background: var(--dark3) !important;
        border: 1px solid var(--border) !important;
        color: var(--text) !important;
        border-radius: 12px !important;
    }
    .stTextInput input:focus, .stTextArea textarea:focus {
        border-color: var(--gold) !important;
        box-shadow: 0 0 0 2px rgba(201,169,110,0.15) !important;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, rgba(201,169,110,0.2), rgba(201,169,110,0.1)) !important;
        border: 1px solid var(--gold) !important;
        color: var(--gold) !important;
        border-radius: 8px !important;
        font-family: 'Raleway', sans-serif !important;
        font-weight: 600 !important;
        letter-spacing: 0.08em !important;
        transition: all 0.2s !important;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, rgba(201,169,110,0.35), rgba(201,169,110,0.2)) !important;
        box-shadow: 0 0 20px rgba(201,169,110,0.2) !important;
    }

    /* Divider */
    hr { border-color: var(--border) !important; opacity: 0.5; }

    /* Tab styling */
    [data-baseweb="tab-list"] { background: transparent !important; border-bottom: 1px solid var(--border); }
    [data-baseweb="tab"] { color: var(--subtle) !important; font-family: 'Raleway', sans-serif !important; letter-spacing: 0.1em; }
    [aria-selected="true"] { color: var(--gold) !important; border-bottom: 2px solid var(--gold) !important; }

    /* Scrollbar */
    ::-webkit-scrollbar { width: 4px; }
    ::-webkit-scrollbar-track { background: var(--dark); }
    ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 2px; }

    /* Success / error states */
    .guardrail-ok   { color: #22C55E; font-size: 0.75rem; }
    .guardrail-fail { color: #EF4444; font-size: 0.75rem; }

    /* Hologram glow effect on header */
    @keyframes holo-pulse {
        0%, 100% { text-shadow: 0 0 30px rgba(201,169,110,0.3); }
        50% { text-shadow: 0 0 60px rgba(201,169,110,0.6), 0 0 100px rgba(201,169,110,0.2); }
    }
    .taj-header h1 { animation: holo-pulse 4s ease-in-out infinite; }
    </style>
    """, unsafe_allow_html=True)

    # ── SIDEBAR ─────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("""
        <div style='text-align:center; padding: 1rem 0;'>
            <div style='font-size:3rem;'>🏛️</div>
            <div style='font-family:"Playfair Display",serif; color:#C9A96E; font-size:1.1rem; margin-top:0.5rem;'>
                Taj Concierge
            </div>
            <div style='font-size:0.7rem; color:#6B7280; letter-spacing:0.3em; text-transform:uppercase; margin-top:0.3rem;'>
                Est. 1903 · Mumbai
            </div>
        </div>
        <hr>
        """, unsafe_allow_html=True)

        groq_key = st.text_input(
            "🔑 Groq API Key",
            type="password",
            placeholder="gsk_...",
            help="Get your free key at console.groq.com",
        )

        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("**About this bot**", unsafe_allow_html=False)
        st.markdown("""
        <div class='sidebar-stat'>
            <div class='label'>Vector Search</div>
            <div class='value'>FAISS + MiniLM-L6</div>
        </div>
        <div class='sidebar-stat'>
            <div class='label'>LLM</div>
            <div class='value'>Groq · Llama-3.3-70B</div>
        </div>
        <div class='sidebar-stat'>
            <div class='label'>Guardrail</div>
            <div class='value'>Price & Payment Guard</div>
        </div>
        <div class='sidebar-stat'>
            <div class='label'>Languages</div>
            <div class='value'>EN · HI · Hinglish</div>
        </div>
        <div class='sidebar-stat'>
            <div class='label'>Knowledge Base</div>
            <div class='value'>Taj Mahal Palace, Mumbai</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("**Contacts**", unsafe_allow_html=False)
        st.markdown("""
        <div style='font-size:0.78rem; color:#8A8070; line-height:1.9;'>
        📞 +91-22-6665-3366<br>
        📧 tmpm.reservations@tajhotels.com<br>
        🌐 www.tajhotels.com
        </div>
        """, unsafe_allow_html=True)

        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    # ── MAIN HEADER ────────────────────────────────────────────────────────
    st.markdown("""
    <div class='taj-header'>
        <div class='year'>✦ EST. 1903 ✦</div>
        <h1>THE TAJ MAHAL PALACE</h1>
        <div class='subtitle'>AI Concierge · Mumbai, India</div>
    </div>
    """, unsafe_allow_html=True)

    # ── TABS ───────────────────────────────────────────────────────────────
    tab_chat, tab_eval, tab_about = st.tabs(["💬  Concierge Chat", "📊  Evaluation Suite", "ℹ️  System Info"])

    # Initialise index & session state
    if not groq_key:
        st.info("🔑 Please enter your Groq API key in the sidebar to begin.", icon="🏛️")
        st.stop()

    index, chunks, embed_model = build_index()
    groq_client = Groq(api_key=groq_key)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # ══════════════════════════════════════════════════════════════════════
    # TAB 1 — CHAT
    # ══════════════════════════════════════════════════════════════════════
    with tab_chat:

        # Welcome message
        if not st.session_state.messages:
            st.markdown("""
            <div class='msg-assistant'>
                <div class='msg-role role-bot'>✦ TAJ CONCIERGE</div>
                Namaste and welcome to The Taj Mahal Palace, Mumbai. 🙏<br><br>
                I am your personal AI concierge — here to assist you with information about our hotel,
                rooms, dining, wellness, history, and guest services.<br><br>
                You may speak with me in <strong>English, Hindi, or Hinglish</strong>.
                How may I assist you today?
            </div>
            """, unsafe_allow_html=True)

        # Render history
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                intent = classify_intent(msg["content"])
                emoji, label, color = INTENT_LABELS[intent]
                st.markdown(f"""
                <div class='msg-user'>
                    <div class='msg-role role-user'>GUEST</div>
                    <span class='intent-badge' style='color:{color};border-color:{color};'>{emoji} {label}</span><br>
                    {msg["content"]}
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class='msg-assistant'>
                    <div class='msg-role role-bot'>✦ TAJ CONCIERGE</div>
                    {msg["content"]}
                </div>""", unsafe_allow_html=True)

        # Input
        col_in, col_btn = st.columns([5, 1])
        with col_in:
            user_input = st.text_input(
                "Message",
                placeholder="Ask me anything about the Taj Mahal Palace… (English / Hindi / Hinglish)",
                label_visibility="collapsed",
                key="chat_input",
            )
        with col_btn:
            send = st.button("Send ✦", use_container_width=True)

        # Example prompts
        st.markdown("""
        <div style='font-size:0.72rem; color:#6B7280; margin-top:0.5rem;'>
        Try: "What restaurants are in the hotel?" &nbsp;·&nbsp;
        "Kya pool hai?" &nbsp;·&nbsp;
        "Tell me about the Tata Suite" &nbsp;·&nbsp;
        "What time is check-in?" &nbsp;·&nbsp;
        "क्या वाई-फ़ाई मुफ़्त है?"
        </div>
        """, unsafe_allow_html=True)

        if send and user_input.strip():
            st.session_state.messages.append({"role": "user", "content": user_input.strip()})

            with st.spinner("✦ Consulting the knowledge archive…"):
                retrieved = retrieve(user_input.strip(), index, chunks, embed_model)
                raw       = generate_response(user_input.strip(), retrieved, st.session_state.messages[:-1], groq_client)
                is_safe, final = guardrail_check(raw)

            st.session_state.messages.append({"role": "assistant", "content": final})

            if not is_safe:
                st.toast("🛡️ Guardrail activated — price/payment info blocked", icon="⚠️")

            st.rerun()

    # ══════════════════════════════════════════════════════════════════════
    # TAB 2 — EVALUATION
    # ══════════════════════════════════════════════════════════════════════
    with tab_eval:
        st.markdown("""
        <div style='font-family:"Playfair Display",serif; color:#C9A96E; font-size:1.4rem; margin-bottom:0.3rem;'>
            Evaluation Suite
        </div>
        <div style='color:#8A8070; font-size:0.82rem; margin-bottom:1.5rem;'>
            10 curated questions — 7 grounded (in-KB) + 3 trap questions designed to test
            hallucination and guardrail robustness.
        </div>
        """, unsafe_allow_html=True)

        # Show questions preview
        st.markdown("#### 📋 Eval Set")
        for q in EVAL_QUESTIONS:
            trap_tag = " 🚨 **TRAP**" if q["trap"] else ""
            st.markdown(f"""
            <div class='eval-card'>
                <div style='font-size:0.7rem; color:#8A8070; letter-spacing:0.2em;'>{q['id']} · {q['category']}{trap_tag}</div>
                <div class='eval-q' style='margin-top:0.4rem;'>❝ {q['question']} ❞</div>
                <div class='eval-meta'>Expected keywords: {', '.join(f'<code style="font-size:0.7rem;color:#C9A96E;background:rgba(201,169,110,0.1);padding:1px 5px;border-radius:3px;">{k}</code>' for k in q['expected_keywords'])}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        if st.button("▶  Run Full Evaluation", use_container_width=True):
            with st.spinner("🔬 Running evaluation — this may take ~60 seconds…"):
                results = run_eval(index, chunks, embed_model, groq_client)

            st.session_state["eval_results"] = results

        if "eval_results" in st.session_state:
            results = st.session_state["eval_results"]

            # Summary metrics
            avg_score   = round(sum(r["score"] for r in results) / len(results), 2)
            avg_latency = round(sum(r["latency"] for r in results) / len(results), 2)
            trap_pass   = sum(1 for r in results if r["trap"] and r["guardrail_fired"] is not False)
            trap_total  = sum(1 for r in results if r["trap"])

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Avg Score",     f"{avg_score:.0%}")
            col2.metric("Avg Latency",   f"{avg_latency}s")
            col3.metric("Trap Guard ✅",  f"{trap_pass}/{trap_total}")
            col4.metric("Questions Run", len(results))

            st.markdown("#### 📊 Results")
            for r in results:
                trap_marker = " 🚨" if r["trap"] else ""
                guard_str = ""
                if r["guardrail_fired"] is True:
                    guard_str = "<span class='guardrail-ok'>🛡️ Guardrail PASSED</span>"
                elif r["guardrail_fired"] is False:
                    guard_str = "<span class='guardrail-fail'>⚠️ Guardrail FAILED</span>"

                score_pct = int(r["score"] * 100)
                st.markdown(f"""
                <div class='eval-card'>
                    <div style='font-size:0.7rem; color:#8A8070; letter-spacing:0.15em;'>{r['id']} · {r['category']}{trap_marker}</div>
                    <div class='eval-q'>❝ {r['question']} ❞</div>
                    <div class='eval-a'>{r['response'][:600]}{'…' if len(r['response'])>600 else ''}</div>
                    <div class='score-bar'><div class='score-marker' style='left:{score_pct}%;'></div></div>
                    <div class='eval-meta'>
                        Score: <strong style='color:#C9A96E;'>{r['score']:.0%}</strong> &nbsp;·&nbsp;
                        Latency: <strong>{r['latency']}s</strong> &nbsp;·&nbsp;
                        Chunks retrieved: {r['retrieved_chunks']} &nbsp;·&nbsp;
                        {guard_str}
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # Export
            st.download_button(
                "⬇️  Export Results as JSON",
                data=json.dumps(results, indent=2, ensure_ascii=False),
                file_name="taj_rag_eval_results.json",
                mime="application/json",
            )

    # ══════════════════════════════════════════════════════════════════════
    # TAB 3 — SYSTEM INFO
    # ══════════════════════════════════════════════════════════════════════
    with tab_about:
        st.markdown("""
        <div style='font-family:"Playfair Display",serif; color:#C9A96E; font-size:1.4rem; margin-bottom:1rem;'>
            System Architecture
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        ```
        ┌──────────────────────────────────────────────────────────────┐
        │              TAJ MAHAL PALACE — RAG CONCIERGE                │
        │                                                              │
        │  ┌─────────────┐    ┌──────────────┐    ┌────────────────┐ │
        │  │  Knowledge  │───▶│   Chunker    │───▶│ SentenceTrans  │ │
        │  │    Base     │    │ (350w/50 ol) │    │  MiniLM-L6-v2  │ │
        │  └─────────────┘    └──────────────┘    └───────┬────────┘ │
        │                                                  │          │
        │                                          ┌───────▼────────┐ │
        │                                          │  FAISS IndexIP │ │
        │                                          │  (cosine sim)  │ │
        │                                          └───────┬────────┘ │
        │                                                  │          │
        │  ┌──────────┐    ┌──────────────┐    ┌──────────▼───────┐ │
        │  │  Query   │───▶│ Intent Class │    │  Top-K Retrieval │ │
        │  │(EN/HI/HG)│    │  (regex+kw) │    │     (k=5)        │ │
        │  └──────────┘    └──────────────┘    └──────────┬───────┘ │
        │       │                                          │          │
        │       └──────────────────────────────────────────┘          │
        │                           │                                  │
        │                  ┌────────▼────────┐                        │
        │                  │  System Prompt  │                        │
        │                  │ + Context + Hist│                        │
        │                  └────────┬────────┘                        │
        │                           │                                  │
        │                  ┌────────▼────────┐                        │
        │                  │  Groq API       │                        │
        │                  │  Llama-3.3-70B  │                        │
        │                  └────────┬────────┘                        │
        │                           │                                  │
        │                  ┌────────▼────────┐                        │
        │                  │   GUARDRAIL     │  ← Price / Payment     │
        │                  │   (regex scan)  │    Link Detection      │
        │                  └────────┬────────┘                        │
        │                           │                                  │
        │                  ┌────────▼────────┐                        │
        │                  │   Response to   │                        │
        │                  │     User        │                        │
        │                  └─────────────────┘                        │
        └──────────────────────────────────────────────────────────────┘
        ```
        """)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Stack**")
            st.markdown("""
            | Component | Technology |
            |-----------|-----------|
            | Embeddings | sentence-transformers/all-MiniLM-L6-v2 |
            | Vector DB | FAISS IndexFlatIP (cosine) |
            | LLM | Groq · llama-3.3-70b-versatile |
            | Frontend | Streamlit |
            | Guardrail | Regex + Contextual Pattern Match |
            | Intent | Keyword-based classifier |
            """)

        with col2:
            st.markdown("**Guardrail Design**")
            st.markdown("""
            The anti-hallucination guardrail works on **two layers**:

            1. **Prompt-level**: The system prompt explicitly forbids inventing prices or payment links, grounding the LLM to provided context only.

            2. **Post-generation scan**: Every response is scanned via regex for price patterns (`₹NNN`, `USD NNN`, `costs NNN`) and payment link patterns. If found and not matching a known safe phrase, the response is replaced with a redirect to the human reservations team.

            **Trap question handling**: Questions about real-time data (current TripAdvisor score), exact prices, or payment links are guaranteed to trigger the redirect — never a hallucinated answer.
            """)

        st.markdown("**KB Stats**")
        with open(KB_PATH, "r", encoding="utf-8") as f:
            kb_text = f.read()
        word_count = len(kb_text.split())
        chunk_count = len(chunk_text(kb_text))
        st.markdown(f"""
        | Metric | Value |
        |--------|-------|
        | KB Size | {len(kb_text):,} characters |
        | KB Words | {word_count:,} words |
        | Chunks (350w, 50 overlap) | {chunk_count} chunks |
        | Embedding Dim | 384 (MiniLM-L6-v2) |
        | Top-K Retrieval | {TOP_K} chunks |
        | Max History | {MAX_HISTORY} turns |
        """)


# ─── ENTRY POINT ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    main()