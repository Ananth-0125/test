"""
Customer Query Analyzer — Streamlit App  (Blue / Yellow Theme)
Run: streamlit run app.py
"""

import re
import json
import time
import requests
import torch
import torch.nn as nn
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from transformers import BertTokenizer, BertModel
from datetime import datetime
import os
from huggingface_hub import snapshot_download

st.set_page_config(
    page_title="Customer Query Analyzer",
    page_icon="▶",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Oswald:wght@400;500;600;700&family=Roboto+Mono:wght@400;500&display=swap');

:root {
    --primary:       #0058A3;
    --primary-dark:  #004080;
    --primary-dim:   rgba(0,88,163,0.10);
    --accent:        #FFDA1A;
    --accent-dark:   #E5C500;
    --accent-dim:    rgba(255,218,26,0.15);
    --bg:            #F5F5F5;
    --surface:       #FFFFFF;
    --surface2:      #EEF3FA;
    --surface3:      #E4EDF7;
    --border:        #C0CDD8;
    --border-dim:    #D8E3EC;
    --text:          #333333;
    --text-dim:      #666677;
    --text-mute:     #99A0AA;
    --white:         #FFFFFF;
    --danger:        #CC2200;
    --danger-dim:    rgba(204,34,0,0.10);
    --success:       #1A7A2A;
    --success-dim:   rgba(26,122,42,0.10);
    --warn:          #A06000;
    --warn-dim:      rgba(160,96,0,0.10);
}

html, body, [class*="css"] {
    font-family: 'Oswald', 'Roboto', sans-serif !important;
    color: var(--text) !important;
}

.stApp {
    background: var(--bg) !important;
}

.block-container {
    padding: 1rem 1rem 2rem 1rem !important;
    max-width: 100% !important;
}
@media (min-width: 768px) {
    .block-container { padding: 1.4rem 2rem 2rem 2rem !important; }
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] .block-container { padding: 1.2rem 1rem !important; }
[data-testid="stSidebarCollapseButton"] {
    background: var(--primary) !important;
    border-radius: 0 6px 6px 0 !important;
}
[data-testid="stSidebarCollapseButton"]:hover { background: var(--primary-dark) !important; }
[data-testid="stSidebarCollapseButton"] svg { fill: #ffffff !important; }
[data-testid="collapsedControl"] {
    background: var(--primary) !important;
    border-radius: 0 6px 6px 0 !important;
}
[data-testid="collapsedControl"]:hover { background: var(--primary-dark) !important; }
[data-testid="collapsedControl"] svg { fill: #ffffff !important; }

/* ── Page Header ── */
.page-header {
    background: linear-gradient(135deg, #E8F0FA 0%, #FFFFFF 60%, #FFFBF0 100%);
    border: 1px solid var(--border);
    border-left: 4px solid var(--primary);
    border-radius: 10px;
    padding: 20px 24px;
    margin-bottom: 18px;
    position: relative;
    overflow: hidden;
}
.page-header::before {
    content: '▶';
    position: absolute;
    right: 20px;
    top: 50%;
    transform: translateY(-50%);
    font-size: 6rem;
    color: rgba(0,88,163,0.06);
    pointer-events: none;
    line-height: 1;
}
.page-header h1 {
    font-size: 1.6rem;
    font-weight: 700;
    margin: 0 0 5px 0;
    color: var(--text);
    letter-spacing: -0.3px;
    font-family: 'Oswald', sans-serif !important;
}
.page-header p {
    margin: 0;
    font-size: 0.78rem;
    color: var(--text-dim);
    font-family: 'Roboto Mono', monospace !important;
    font-weight: 400;
}
.header-tags { margin-bottom: 10px; }
.htag {
    display: inline-block;
    background: var(--accent-dim);
    border: 1px solid rgba(255,218,26,0.5);
    color: #7A5800;
    padding: 2px 10px;
    border-radius: 2px;
    font-size: 0.62rem;
    font-family: 'Roboto Mono', monospace !important;
    margin-right: 5px;
    text-transform: uppercase;
    letter-spacing: 1px;
}

/* ── Section Labels ── */
.section-label {
    font-size: 0.65rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 2px;
    color: var(--text-mute);
    margin-bottom: 10px;
    padding-bottom: 6px;
    border-bottom: 1px solid var(--border-dim);
    font-family: 'Roboto Mono', monospace !important;
}
.sb-sec {
    font-size: 0.63rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    color: var(--text-mute);
    margin: 14px 0 6px 0;
    font-family: 'Roboto Mono', monospace !important;
}

/* ── Chat Window ── */
.chat-window {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 14px 16px;
    height: 400px;
    overflow-y: auto;
    margin-bottom: 8px;
}
.bubble-user, .bubble-bot, .bubble-security {
    max-width: 85%;
    padding: 9px 14px;
    font-size: 0.86rem;
    line-height: 1.5;
    margin-bottom: 8px;
    clear: both;
    font-family: 'Roboto Mono', monospace !important;
    font-weight: 400;
}
.bubble-user {
    background: var(--primary);
    color: #ffffff;
    border-radius: 14px 14px 3px 14px;
    float: right;
}
.bubble-bot {
    background: var(--surface2);
    border: 1px solid var(--border);
    color: var(--text);
    border-radius: 14px 14px 14px 3px;
    float: left;
}
.bubble-security {
    background: #FFF0EE;
    border: 1px solid #FFBBAA;
    color: #8B1500;
    border-radius: 14px 14px 14px 3px;
    float: left;
}
.msg-meta {
    font-size: 0.63rem;
    color: var(--text-mute);
    margin-bottom: 6px;
    font-family: 'Roboto Mono', monospace !important;
    display: flex;
    gap: 5px;
    flex-wrap: wrap;
    align-items: center;
    clear: both;
}
@media (min-width: 768px) {
    .bubble-user, .bubble-bot, .bubble-security { max-width: 70%; }
}

/* ── Tags ── */
.tag {
    display: inline-block;
    padding: 1px 7px;
    border-radius: 2px;
    font-size: 0.62rem;
    font-weight: 500;
    font-family: 'Roboto Mono', monospace !important;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
.t-intent  { background: var(--primary-dim); color: var(--primary); border: 1px solid rgba(0,88,163,0.3); }
.t-neg     { background: #FEE8E8; color: #CC2200; border: 1px solid #FFBBAA; }
.t-neu     { background: var(--surface3); color: var(--text-dim); border: 1px solid var(--border); }
.t-pos     { background: #E8FEEE; color: #1A7A2A; border: 1px solid #AAFFBB; }
.t-sec     { background: #CC2200; color: #ffffff; border: 1px solid #AA1800; font-weight: 700; }
.t-low     { background: #FFF8E0; color: #A06000; border: 1px solid #FFD080; }
.t-good    { background: #E8FEEE; color: #1A7A2A; border: 1px solid #AAFFBB; }
.t-bad     { background: #FEE8E8; color: #CC2200; border: 1px solid #FFBBAA; }

/* ── Progress Bars ── */
.bar-track { background: var(--surface3); border-radius: 2px; height: 5px; margin: 3px 0 9px 0; overflow: hidden; }
.bar-blue  { background: var(--primary); height: 5px; border-radius: 2px; }
.bar-red   { background: #CC2200; height: 5px; border-radius: 2px; }

/* ── Metric Tiles ── */
.metric-tile {
    background: var(--surface);
    border: 1px solid var(--border);
    border-top: 2px solid var(--primary);
    border-radius: 6px;
    padding: 12px 10px;
    text-align: center;
}
.metric-tile .val {
    font-size: 1.3rem;
    font-weight: 700;
    color: var(--text);
    font-family: 'Oswald', sans-serif !important;
    line-height: 1.1;
}
.metric-tile .lbl {
    font-size: 0.6rem;
    color: var(--text-mute);
    margin-top: 4px;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    font-family: 'Roboto Mono', monospace !important;
}

/* ── Empty State ── */
.empty-state { text-align: center; color: var(--text-mute); padding: 80px 20px; }
.empty-state .text { font-size: 0.87rem; color: var(--text-dim); }
.empty-state .hint { font-size: 0.75rem; color: var(--text-mute); margin-top: 5px; }

/* ── Buttons ── */
.stButton > button, .stDownloadButton > button {
    background: var(--surface2) !important;
    color: var(--text) !important;
    border: 1px solid var(--border) !important;
    border-radius: 4px !important;
    font-weight: 500 !important;
    font-family: 'Roboto Mono', monospace !important;
    font-size: 0.8rem !important;
    padding: 6px 16px !important;
    transition: all 0.15s ease !important;
    box-shadow: none !important;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
.stButton > button:hover, .stDownloadButton > button:hover {
    background: var(--primary) !important;
    color: #ffffff !important;
    border-color: var(--primary) !important;
}
.stFormSubmitButton > button {
    background: var(--accent) !important;
    color: #333333 !important;
    border: none !important;
    border-radius: 4px !important;
    font-size: 1.2rem !important;
    font-weight: 700 !important;
    padding: 0 !important;
    width: 100% !important;
    min-height: 38px !important;
    line-height: 38px !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
}
.stFormSubmitButton > button:hover { background: var(--accent-dark) !important; }

section[data-testid="stSidebar"] .stButton > button {
    background: var(--primary) !important;
    color: #ffffff !important;
    border: none !important;
    width: 100% !important;
}
section[data-testid="stSidebar"] .stButton > button:hover {
    background: var(--primary-dark) !important;
}
section[data-testid="stSidebar"] .stDownloadButton > button {
    background: var(--surface2) !important;
    color: var(--text) !important;
    border: 1px solid var(--border) !important;
    width: 100% !important;
}
section[data-testid="stSidebar"] .stDownloadButton > button:hover {
    background: var(--primary) !important;
    color: #ffffff !important;
}

/* ── Inputs ── */
div[data-baseweb="input"] input, .stTextInput input {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 4px !important;
    color: var(--text) !important;
    font-family: 'Roboto Mono', monospace !important;
    font-size: 0.88rem !important;
}
div[data-baseweb="input"] input::placeholder { color: var(--text-mute) !important; }
div[data-baseweb="input"] input:focus {
    border-color: var(--primary) !important;
    box-shadow: 0 0 0 2px rgba(0,88,163,0.15) !important;
}

/* ── Dataframe ── */
[data-testid="stDataFrameResizable"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 6px !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 2px; }
::-webkit-scrollbar-thumb:hover { background: var(--primary); }

/* ── Misc ── */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
.main .block-container { max-width: 100% !important; }
section.main { max-width: 100% !important; }

/* ── Spinner / info / success ── */
.stSpinner > div { border-top-color: var(--primary) !important; }
.stSuccess { background: #E8FEEE !important; color: #1A7A2A !important; border: 1px solid #AAFFBB !important; }
.stWarning { background: #FFF8E0 !important; color: #A06000 !important; border: 1px solid #FFD080 !important; }
.stError   { background: #FEE8E8 !important; color: #CC2200 !important; border: 1px solid #FFBBAA !important; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# SESSION STATE
# ============================================================
_defaults = {
    "messages"        : [],
    "conv_history"    : [],
    "history_log"     : [],
    "total_queries"   : 0,
    "sentiment_counts": {"negative": 0, "neutral": 0, "positive": 0},
    "security_count"  : 0,
    "lowconf_count"   : 0,
    "bert_loaded"     : False,
    "last_result"     : None,
    "intent_freq"     : {},
    "latencies"       : [],
    "feedback"        : {},
    "api_key"         : "",
}
for k, v in _defaults.items():
    if k not in st.session_state:
        if isinstance(v, dict):   st.session_state[k] = v.copy()
        elif isinstance(v, list): st.session_state[k] = []
        else:                     st.session_state[k] = v

# ============================================================
# CONSTANTS
# ============================================================
SENTIMENT_NAMES = ["negative", "neutral", "positive"]
SENTIMENT_LABEL = {"negative": "Negative", "neutral": "Neutral", "positive": "Positive"}
LOW_CONF        = 0.20
HF_REPO_ID      = "YamiChowdary/customer-query-analyzer-bert"

GROQ_MODELS = [
    "llama3-8b-8192",
    "llama-3.1-8b-instant",
    "llama-3.3-70b-versatile",
    "mixtral-8x7b-32768",
    "gemma2-9b-it",
]

# ============================================================
# SAFETY NET
# ============================================================
SAFETY_PATTERNS = {
    "unauthorized_access": [
        "someone else","someone is using","unauthori","hacked","hack",
        "not me","wasn't me","i didn't do","suspicious login","unknown login",
        "someone logged","someone accessed","strange activity","unusual activity",
        "unknown transaction","i didn't make this","i did not make","fraudulent login",
    ],
    "report_fraud": [
        "fraud","scam","scammed","cheated","stolen","stole","theft",
        "fake transaction","unauthorized transaction","didn't authorize",
        "did not authorize","money missing","money gone","money disappeared",
        "deducted without","charged without","debited without my",
    ],
    "emergency_block": [
        "block immediately","block my card now","freeze immediately",
        "lost my card","card stolen","stolen card","i lost my",
        "cant find my card","missing card","card is missing",
    ],
    "account_compromised": [
        "account compromised","account breached","password changed",
        "someone changed my password","locked out","cant access my account",
        "cant log in","cant login","login not working","otp not received",
        "not receiving otp","verification not working",
    ],
}

def pre_classify(query: str):
    q = query.lower()
    for intent, kws in SAFETY_PATTERNS.items():
        for kw in kws:
            if kw in q:
                return intent, 0.95
    return None, None

# ============================================================
# BERT MODEL
# ============================================================
class MultiTaskBERT(nn.Module):
    def __init__(self, bert_name, num_intents, num_sentiments, dropout=0.3):
        super().__init__()
        self.bert    = BertModel.from_pretrained(bert_name)
        h            = self.bert.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.intent_classifier = nn.Sequential(
            nn.Linear(h, 512), nn.GELU(), nn.Dropout(dropout), nn.Linear(512, num_intents)
        )
        self.sentiment_classifier = nn.Sequential(
            nn.Linear(h, 256), nn.GELU(), nn.Dropout(dropout), nn.Linear(256, num_sentiments)
        )

    def forward(self, input_ids, attention_mask, token_type_ids):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask,
                        token_type_ids=token_type_ids)
        cls = self.dropout(out.pooler_output)
        return self.intent_classifier(cls), self.sentiment_classifier(cls)

@st.cache_resource(show_spinner=False)
def get_model_path():
    cache_dir = os.path.join(os.getcwd(), ".cache", "hf_models", HF_REPO_ID.replace("/", "_"))
    os.makedirs(cache_dir, exist_ok=True)
    if not os.path.exists(os.path.join(cache_dir, "bert_best.pt")) or \
       not os.path.exists(os.path.join(cache_dir, "intent_label_map.json")):
        with st.spinner("Downloading model from Hugging Face... (~1-2 minutes)"):
            snapshot_download(repo_id=HF_REPO_ID, local_dir=cache_dir,
                              local_dir_use_symlinks=False, resume_download=True)
    return cache_dir

@st.cache_resource(show_spinner=False)
def load_model(model_dir, data_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open(os.path.join(data_dir, "intent_label_map.json")) as f:
        id2intent = json.load(f)
    n      = len(id2intent)
    oos_id = next((int(k) for k, v in id2intent.items() if v == "oos"), -1)
    tok    = BertTokenizer.from_pretrained(model_dir)
    mdl    = MultiTaskBERT("bert-base-uncased", n, 3)
    mdl.load_state_dict(torch.load(os.path.join(model_dir, "bert_best.pt"),
                                   map_location=device, weights_only=True))
    mdl.to(device).eval()
    return mdl, tok, id2intent, oos_id, device

# ============================================================
# CLASSIFICATION
# ============================================================
def clean_text(t):
    t = t.strip().lower()
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"[^\w\s\'\-\?\!\.,]", "", t)
    t = re.sub(r"(\w)\1{3,}", r"\1\1", t)
    return t

@torch.no_grad()
def classify(query, mdl, tok, id2intent, oos_id, device):
    cq = clean_text(query)
    oi, oc = pre_classify(cq)
    enc = tok(cq, max_length=64, padding="max_length", truncation=True, return_tensors="pt")
    il, sl = mdl(enc["input_ids"].to(device), enc["attention_mask"].to(device),
                 enc["token_type_ids"].to(device))
    ip   = torch.softmax(il, dim=-1)[0]
    sp   = torch.softmax(sl, dim=-1)[0]
    iid  = ip.argmax().item()
    sid  = sp.argmax().item()
    conf = ip[iid].item()
    if oi:
        intent_name = oi; conf = oc; low = False; pre = True
    elif conf < LOW_CONF and oos_id >= 0:
        intent_name = "out_of_scope"; low = True; pre = False
    else:
        intent_name = id2intent[str(iid)]; low = False; pre = False
    t3i = ip.topk(3).indices.cpu().numpy()
    t3s = ip.topk(3).values.cpu().numpy()
    return {
        "intent"              : intent_name,
        "intent_confidence"   : round(conf, 4),
        "top3_intents"        : [(id2intent[str(i)], round(float(s)*100, 1)) for i, s in zip(t3i, t3s)],
        "sentiment"           : SENTIMENT_NAMES[sid],
        "sentiment_confidence": round(sp[sid].item(), 4),
        "sentiment_scores"    : {
            "negative": round(sp[0].item()*100, 1),
            "neutral" : round(sp[1].item()*100, 1),
            "positive": round(sp[2].item()*100, 1),
        },
        "low_confidence": low,
        "pre_classified": pre,
    }

# ============================================================
# PROMPT BUILDER
# ============================================================
MAX_HISTORY_TURNS = 6

def build_system_prompt(intent, sentiment):
    ir   = intent.replace("_", " ")
    tone = {
        "negative": "The user seems frustrated. Be empathetic, calm, and solution-focused.",
        "neutral" : "The user is making a calm request. Be clear and concise.",
        "positive": "The user is in a positive mood. Be warm and match their energy.",
    }.get(sentiment, "Be helpful and polite.")
    return (
        f"You are a friendly and professional customer support AI assistant. "
        f"The user's current topic is: {ir}. "
        f"{tone} "
        f"Keep replies concise (2–4 sentences). "
        f"Never mention intent names, confidence scores, or any system labels."
    )

def build_messages(query, intent, sentiment, history=None):
    system_msg = {"role": "system", "content": build_system_prompt(intent, sentiment)}
    trimmed    = (history or [])[-MAX_HISTORY_TURNS:]
    api_history = []
    for turn in trimmed:
        role    = "assistant" if turn["role"] == "model" else "user"
        content = str(turn.get("content", "")).strip()
        if content:
            api_history.append({"role": role, "content": content})
    api_history.append({"role": "user", "content": query.strip()})
    return [system_msg] + api_history

# ============================================================
# AI RESPONSE — Groq  (auto model fallback)
# ============================================================
def _groq_post(api_key: str, model: str, messages: list, max_tokens: int = 300):
    try:
        r = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key.strip()}",
                "Content-Type": "application/json",
            },
            json={"model": model, "messages": messages,
                  "max_tokens": max_tokens, "temperature": 0.7},
            timeout=30,
        )
        if r.status_code == 200:
            return r.json()["choices"][0]["message"]["content"].strip(), None
        try:
            err = r.json().get("error", {}).get("message", r.text[:300])
        except Exception:
            err = r.text[:300]
        return None, f"HTTP {r.status_code}: {err}"
    except requests.exceptions.Timeout:
        return None, "Request timed out"
    except Exception as e:
        return None, str(e)[:200]


def get_ai_response(query, intent, sentiment, confidence, api_key, history=None):
    messages   = build_messages(query, intent, sentiment, history)
    last_error = "No models available"
    for model in GROQ_MODELS:
        text, err = _groq_post(api_key, model, messages)
        if text is not None:
            st.session_state["groq_model_ok"] = model
            return text
        last_error = f"{model} → {err}"
        if "401" in last_error or "invalid_api_key" in last_error.lower():
            return f"⚠ Invalid API key. Check the key pasted in the sidebar.\n\nDetail: {err}"
    return f"⚠ All Groq models failed. Last error:\n{last_error}"


def test_groq_connection(api_key: str):
    msgs = [{"role": "user", "content": "Reply with exactly: OK"}]
    for model in GROQ_MODELS:
        text, err = _groq_post(api_key, model, msgs, max_tokens=10)
        if text is not None:
            return True, model, text
    return False, None, err

def latency_stats():
    lats = st.session_state.latencies
    if not lats: return None
    return {"avg": round(sum(lats)/len(lats)), "min": min(lats), "max": max(lats)}

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("""
    <div style='padding:6px 0 14px 0; border-bottom:1px solid #C0CDD8; margin-bottom:2px;'>
        <div style='font-size:0.95rem;font-weight:700;color:#333333;font-family:Oswald,sans-serif;letter-spacing:0.5px;'>QUERY ANALYZER</div>
        <div style='font-size:0.62rem;color:#99A0AA;margin-top:2px;font-family:Roboto Mono,monospace;'>BERT + GROQ ENGINE</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='sb-sec'>▸ API KEY (GROQ)</div>", unsafe_allow_html=True)

    import os as _os
    api_key = ""
    _on_cloud = _os.environ.get("STREAMLIT_SHARING_MODE") or _os.path.exists("/mount/src")
    if _on_cloud:
        try:
            api_key = st.secrets["GROQ_API_KEY"]
            st.markdown("<div style='font-size:0.69rem;color:#1A7A2A;margin-bottom:8px;font-family:Roboto Mono,monospace;'>✓ KEY LOADED FROM SECRETS</div>", unsafe_allow_html=True)
        except Exception:
            api_key = ""
    if not api_key:
        api_key = st.text_input(
            "api_key_input", label_visibility="collapsed",
            type="password", placeholder="Paste your Groq API key...",
        )
        if api_key:
            masked = api_key[:4] + "x" * min(len(api_key)-8, 10) + api_key[-4:] if len(api_key) > 8 else "x" * len(api_key)
            st.markdown(f"<div style='font-size:0.69rem;color:#1A7A2A;margin:-2px 0 6px 0;font-family:Roboto Mono,monospace;'>✓ KEY SET: {masked}</div>", unsafe_allow_html=True)

    st.markdown(
        "<div style='font-size:0.68rem;color:#99A0AA;margin:4px 0 8px 0;font-family:Roboto Mono,monospace;'>"
        "FREE · <a href='https://console.groq.com' style='color:#0058A3;text-decoration:none;'>CONSOLE.GROQ.COM</a></div>",
        unsafe_allow_html=True
    )
    st.session_state.api_key = api_key

    if api_key:
        if st.button("▶ Test Connection", use_container_width=True, key="test_conn"):
            with st.spinner("Testing Groq API..."):
                ok, model_used, detail = test_groq_connection(api_key)
            if ok:
                st.markdown(
                    f"<div style='background:#E8FEEE;border:1px solid #AAFFBB;border-radius:6px;"
                    f"padding:8px 10px;font-family:Roboto Mono,monospace;font-size:0.68rem;color:#1A7A2A;"
                    f"margin-bottom:6px;'>✓ CONNECTED<br>"
                    f"<span style='color:#666677;'>Model: {model_used}</span></div>",
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f"<div style='background:#FEE8E8;border:1px solid #FFBBAA;border-radius:6px;"
                    f"padding:8px 10px;font-family:Roboto Mono,monospace;font-size:0.67rem;color:#CC2200;"
                    f"margin-bottom:6px;'>✕ FAILED<br>"
                    f"<span style='color:#AA4444;word-break:break-all;'>{detail}</span></div>",
                    unsafe_allow_html=True
                )

    st.markdown("<div style='height:1px;background:#D8E3EC;margin:10px 0;'></div>", unsafe_allow_html=True)
    st.markdown("<div class='sb-sec'>▸ SESSION STATS</div>", unsafe_allow_html=True)

    total = st.session_state.total_queries
    neg   = st.session_state.sentiment_counts["negative"]
    neu   = st.session_state.sentiment_counts["neutral"]
    pos   = st.session_state.sentiment_counts["positive"]
    sec   = st.session_state.security_count
    low   = st.session_state.lowconf_count
    ls    = latency_stats()

    rows = [
        ("TOTAL QUERIES",   str(total), "#0058A3"),
        ("NEGATIVE",        str(neg),   "#CC2200"),
        ("NEUTRAL",         str(neu),   "#666677"),
        ("POSITIVE",        str(pos),   "#1A7A2A"),
        ("SECURITY ALERTS", str(sec),   "#CC2200"),
        ("LOW CONFIDENCE",  str(low),   "#A06000"),
    ]
    if ls:
        rows += [
            ("AVG LATENCY", f"{ls['avg']} ms", "#0058A3"),
            ("MIN LATENCY", f"{ls['min']} ms", "#1A7A2A"),
            ("MAX LATENCY", f"{ls['max']} ms", "#CC2200"),
        ]
    for label, val, color in rows:
        st.markdown(
            f"<div style='display:flex;justify-content:space-between;"
            f"font-size:0.76rem;padding:4px 0;border-bottom:1px solid #EEF3FA;font-family:Roboto Mono,monospace;'>"
            f"<span style='color:#99A0AA;'>{label}</span>"
            f"<span style='font-weight:700;color:{color};'>{val}</span>"
            f"</div>", unsafe_allow_html=True
        )

    st.markdown("<div style='height:1px;background:#D8E3EC;margin:12px 0;'></div>", unsafe_allow_html=True)

    if st.session_state.history_log:
        df_exp   = pd.DataFrame(st.session_state.history_log)
        csv_data = df_exp.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇ Download History (CSV)", data=csv_data,
            file_name=f"queries_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv", use_container_width=True,
        )
        st.markdown("<div style='height:6px;'></div>", unsafe_allow_html=True)

    if st.button("✕ Clear Conversation", use_container_width=True):
        for k, v in _defaults.items():
            if isinstance(v, dict):   st.session_state[k] = v.copy()
            elif isinstance(v, list): st.session_state[k] = []
            else:                     st.session_state[k] = v
        st.rerun()

# ============================================================
# AUTO MODEL LOADING
# ============================================================
if not st.session_state.bert_loaded:
    with st.spinner("▶ Loading BERT model from Hugging Face... (first time ~1-2 minutes)"):
        try:
            model_path = get_model_path()
            mdl, tok, i2i, oid, dev = load_model(model_path, model_path)
            st.session_state.update({
                "bert_loaded": True,
                "model": mdl, "tokenizer": tok,
                "id2intent": i2i, "oos_id": oid, "device": dev,
            })
            st.success("▶ BERT model loaded successfully!")
        except Exception as e:
            st.error(f"✕ Failed to load model: {e}")
            st.info("Please check internet connection and that the Hugging Face repo is public.")
            st.stop()

# ============================================================
# PAGE HEADER
# ============================================================
st.markdown("""
<div class="page-header">
    <div class="header-tags">
        <span class="htag">BERT MULTI-TASK</span>
        <span class="htag">151 INTENTS</span>
        <span class="htag">SAFETY NET</span>
        <span class="htag">GROQ LLM</span>
    </div>
    <h1>▶ Customer Query Analyzer</h1>
    <p>INTENT CLASSIFICATION · SENTIMENT ANALYSIS · AUTOMATED RESPONSE GENERATION</p>
</div>
""", unsafe_allow_html=True)

# ============================================================
# MAIN LAYOUT
# ============================================================
col_chat, col_right = st.columns([1.05, 0.95], gap="large")

with col_chat:
    st.markdown('<div class="section-label">▸ Chat Interface</div>', unsafe_allow_html=True)

    if not st.session_state.messages:
        chat_html = """
        <div class="chat-window">
            <div class="empty-state">
                <div style="font-size:2.5rem;margin-bottom:12px;opacity:0.25;color:#0058A3;">▶</div>
                <div class="text">Model loaded — start a conversation</div>
                <div class="hint">
                    Try: "What is my account balance?" &nbsp;·&nbsp;
                    "Someone hacked my account" &nbsp;·&nbsp;
                    "I lost my card"
                </div>
            </div>
        </div>"""
    else:
        chat_html = '<div class="chat-window"><div style="overflow:auto">'
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                chat_html += (
                    f'<div class="bubble-user">{msg["content"]}</div>'
                    f'<div class="msg-meta" style="justify-content:flex-end;">{msg["time"]}</div>'
                )
            else:
                is_sec  = msg.get("pre_classified", False)
                is_low  = msg.get("low_confidence", False)
                bubble  = "bubble-security" if is_sec else "bubble-bot"
                i_label = msg.get("intent", "").replace("_", " ").upper()
                s       = msg.get("sentiment", "neutral")
                s_cls   = {"negative":"t-neg","neutral":"t-neu","positive":"t-pos"}.get(s,"t-neu")
                fb      = msg.get("feedback", "")
                fb_tag  = ""
                if fb == "up":     fb_tag = ' <span class="tag t-good">▲ HELPFUL</span>'
                elif fb == "down": fb_tag = ' <span class="tag t-bad">▼ NOT HELPFUL</span>'
                tags = f'<span class="tag t-intent">{i_label}</span> <span class="tag {s_cls}">{SENTIMENT_LABEL.get(s, s).upper()}</span>'
                if is_sec: tags += ' <span class="tag t-sec">⚠ SECURITY</span>'
                if is_low: tags += ' <span class="tag t-low">~ LOW CONF</span>'
                tags += f'{fb_tag} <span style="color:#C0CDD8;font-size:0.6rem;">{msg.get("time","")} · {msg.get("latency","")}</span>'
                chat_html += (
                    f'<div class="{bubble}">{msg["content"]}</div>'
                    f'<div class="msg-meta">{tags}</div>'
                )
        chat_html += '</div></div>'

    st.markdown(chat_html, unsafe_allow_html=True)

    with st.form("chat_form", clear_on_submit=True):
        input_col, arrow_col = st.columns([11, 1])
        with input_col:
            user_input = st.text_input(
                "input_field", label_visibility="collapsed",
                placeholder="Type your query here..."
            )
        with arrow_col:
            submitted = st.form_submit_button("▶", use_container_width=True)

    # Feedback
    bot_msgs = [m for m in st.session_state.messages if m["role"] == "bot"]
    if bot_msgs:
        last_idx = len(st.session_state.messages) - 1
        while last_idx >= 0 and st.session_state.messages[last_idx]["role"] != "bot":
            last_idx -= 1
        if last_idx >= 0 and st.session_state.messages[last_idx].get("feedback", "") == "":
            st.markdown("<div style='font-size:0.67rem;color:#99A0AA;margin:2px 0 4px 2px;font-family:Roboto Mono,monospace;'>WAS THIS RESPONSE HELPFUL?</div>", unsafe_allow_html=True)
            fb1, fb2, _ = st.columns([1, 1, 6])
            with fb1:
                if st.button("▲ Yes", key="fb_up", use_container_width=True):
                    st.session_state.messages[last_idx]["feedback"] = "up"
                    if st.session_state.history_log: st.session_state.history_log[-1]["Feedback"] = "Yes"
                    st.rerun()
            with fb2:
                if st.button("▼ No", key="fb_down", use_container_width=True):
                    st.session_state.messages[last_idx]["feedback"] = "down"
                    if st.session_state.history_log: st.session_state.history_log[-1]["Feedback"] = "No"
                    st.rerun()

    # Quick examples
    st.markdown("<div style='font-size:0.62rem;color:#99A0AA;margin:8px 0 5px 0;font-family:Roboto Mono,monospace;letter-spacing:1px;'>▸ QUICK EXAMPLES</div>", unsafe_allow_html=True)
    examples = [
        "What is my account balance?",
        "I lost my card, block it now",
        "Someone hacked my account",
        "Book a flight to Chennai",
        "Translate hello to French",
        "Late delivery, I am frustrated",
    ]
    eq1, eq2, eq3 = st.columns(3)
    for col, ex in zip([eq1, eq2, eq3, eq1, eq2, eq3], examples):
        with col:
            if st.button(ex, key=f"ex_{ex[:10].replace(' ','_')}", use_container_width=True):
                st.session_state["_prefill"] = ex
                st.rerun()

    if "_prefill" in st.session_state:
        user_input = st.session_state.pop("_prefill")
        submitted = True

    # Process query
    if submitted and user_input and user_input.strip():
        if not st.session_state.bert_loaded:
            st.warning("Model not loaded. Please wait.")
        elif not st.session_state.api_key:
            st.warning("Please paste your Groq API key in the sidebar.")
        else:
            with st.spinner("▶ Analyzing..."):
                t0 = time.time()
                result = classify(
                    user_input,
                    st.session_state.model, st.session_state.tokenizer,
                    st.session_state.id2intent, st.session_state.oos_id, st.session_state.device,
                )
                response = get_ai_response(
                    user_input, result["intent"], result["sentiment"],
                    result["intent_confidence"],
                    st.session_state.api_key,
                    st.session_state.conv_history,
                )
                latency = round((time.time() - t0) * 1000)
                now = datetime.now().strftime("%H:%M")

            st.session_state.conv_history.append({"role": "user",  "content": user_input})
            st.session_state.conv_history.append({"role": "model", "content": response})
            st.session_state.messages.append({"role": "user", "content": user_input, "time": now})
            st.session_state.messages.append({
                "role": "bot", "content": response,
                "intent": result["intent"], "sentiment": result["sentiment"],
                "pre_classified": result["pre_classified"],
                "low_confidence": result["low_confidence"],
                "time": now, "latency": f"{latency}ms", "feedback": "",
            })

            st.session_state.total_queries += 1
            st.session_state.sentiment_counts[result["sentiment"]] += 1
            st.session_state.latencies.append(latency)
            if result["pre_classified"]: st.session_state.security_count += 1
            if result["low_confidence"]: st.session_state.lowconf_count  += 1

            ik = result["intent"].replace("_", " ")
            st.session_state.intent_freq[ik] = st.session_state.intent_freq.get(ik, 0) + 1
            st.session_state.last_result = {**result, "response": response, "latency": latency, "query": user_input}

            flag = "Security" if result["pre_classified"] else ("Low conf" if result["low_confidence"] else "OK")
            st.session_state.history_log.append({
                "Time": now,
                "Query": user_input[:44]+"..." if len(user_input)>44 else user_input,
                "Intent": ik,
                "Confidence": f"{result['intent_confidence']*100:.1f}%",
                "Sentiment": SENTIMENT_LABEL.get(result["sentiment"], result["sentiment"]),
                "Status": flag,
                "Latency": f"{latency}ms",
                "Feedback": "",
            })
            st.rerun()

# ── ANALYTICS COLUMN ──
with col_right:
    st.markdown('<div class="section-label">▸ Analysis Panel</div>', unsafe_allow_html=True)

    if st.session_state.last_result:
        r = st.session_state.last_result
        m1, m2, m3 = st.columns(3)
        with m1:
            st.markdown(f'<div class="metric-tile"><div class="val" style="font-size:0.72rem;line-height:1.4;word-break:break-word;">{r["intent"].replace("_"," ").upper()}</div><div class="lbl">Intent</div></div>', unsafe_allow_html=True)
        with m2:
            st.markdown(f'<div class="metric-tile"><div class="val" style="font-size:0.85rem;">{SENTIMENT_LABEL.get(r["sentiment"], r["sentiment"]).upper()}</div><div class="lbl">Sentiment</div></div>', unsafe_allow_html=True)
        with m3:
            fl = "SECURITY" if r["pre_classified"] else ("LOW CONF" if r["low_confidence"] else "NORMAL")
            fv = "#CC2200" if r["pre_classified"] else ("#A06000" if r["low_confidence"] else "#1A7A2A")
            st.markdown(f'<div class="metric-tile"><div class="val" style="font-size:0.78rem;color:{fv};">{fl}</div><div class="lbl">Status</div></div>', unsafe_allow_html=True)

        st.markdown("<div style='height:14px;'></div>", unsafe_allow_html=True)
        st.markdown('<div class="section-label">▸ Intent Confidence</div>', unsafe_allow_html=True)
        conf_pct    = round(r["intent_confidence"] * 100, 1)
        # Security = danger red; low conf = amber; normal = primary blue
        gauge_color = "#CC2200" if r["pre_classified"] else ("#F59E0B" if conf_pct < 50 else "#0058A3")
        fig_g = go.Figure(go.Indicator(
            mode="gauge+number", value=conf_pct,
            number={"suffix":"%","font":{"size":20,"color":"#333333","family":"Roboto Mono"}},
            gauge={
                "axis":{"range":[0,100],"tickwidth":1,"tickcolor":"#C0CDD8","tickfont":{"size":9,"color":"#99A0AA"}},
                "bar":{"color":gauge_color,"thickness":0.24},
                "bgcolor":"#F5F5F5","bordercolor":"#C0CDD8","borderwidth":1,
                "steps":[
                    {"range":[0,40],  "color":"#FEE8E8"},
                    {"range":[40,70], "color":"#FFF8E0"},
                    {"range":[70,100],"color":"#E8FEEE"},
                ],
            },
        ))
        fig_g.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", height=155,
            margin=dict(l=16,r=16,t=8,b=8),
            font=dict(family="Roboto Mono", color="#333333")
        )
        st.plotly_chart(fig_g, use_container_width=True, config={"displayModeBar":False})

        st.markdown('<div class="section-label">▸ Top 3 Predictions</div>', unsafe_allow_html=True)
        bar_cls = "bar-red" if r["pre_classified"] else "bar-blue"
        for name, score in r["top3_intents"]:
            st.markdown(
                f'<div style="margin-bottom:9px;">'
                f'<div style="display:flex;justify-content:space-between;font-size:0.74rem;margin-bottom:3px;font-family:Roboto Mono,monospace;">'
                f'<span style="color:#666677;">{name.replace("_"," ").upper()}</span>'
                f'<span style="color:#0058A3;font-weight:700;">{score}%</span>'
                f'</div>'
                f'<div class="bar-track"><div class="{bar_cls}" style="width:{min(score,100)}%;"></div></div>'
                f'</div>', unsafe_allow_html=True
            )

        st.markdown('<div class="section-label">▸ Sentiment Breakdown</div>', unsafe_allow_html=True)
        ss = r["sentiment_scores"]
        fig = go.Figure(go.Bar(
            x=list(ss.values()), y=["Negative","Neutral","Positive"],
            orientation="h",
            marker_color=["#CC2200","#99A0AA","#1A7A2A"],
            text=[f"{v}%" for v in ss.values()],
            textposition="auto",
            textfont=dict(color="#333333", size=11, family="Roboto Mono"),
        ))
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#666677", family="Roboto Mono"), height=125,
            margin=dict(l=0,r=0,t=0,b=0),
            xaxis=dict(showgrid=False, showticklabels=False, range=[0,115]),
            yaxis=dict(showgrid=False, tickfont=dict(size=10, color="#666677")),
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})

    else:
        st.markdown(
            '<div style="text-align:center;padding:60px 20px;background:#FFFFFF;border:1px solid #C0CDD8;border-radius:8px;">'
            '<div style="font-size:2rem;margin-bottom:10px;opacity:0.2;color:#0058A3;">▶</div>'
            '<div style="font-size:0.84rem;color:#99A0AA;font-family:Roboto Mono,monospace;">ANALYSIS RESULTS WILL APPEAR AFTER YOUR FIRST QUERY.</div>'
            '<div style="font-size:0.73rem;color:#C0CDD8;margin-top:5px;font-family:Roboto Mono,monospace;">TYPE A QUERY ABOVE AND CLICK SEND</div>'
            '</div>', unsafe_allow_html=True
        )

    if st.session_state.total_queries > 0:
        st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)
        st.markdown('<div class="section-label">▸ Session Sentiment</div>', unsafe_allow_html=True)
        counts = st.session_state.sentiment_counts
        fig2 = go.Figure(go.Pie(
            labels=["Negative","Neutral","Positive"],
            values=[counts["negative"],counts["neutral"],counts["positive"]],
            hole=0.55,
            marker_colors=["#CC2200","#99A0AA","#1A7A2A"],
            textfont=dict(size=10, family="Roboto Mono"),
        ))
        fig2.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#666677", family="Roboto Mono"),
            height=185, margin=dict(l=0,r=0,t=0,b=0), showlegend=True,
            legend=dict(
                orientation="h", yanchor="bottom", y=-0.22,
                xanchor="center", x=0.5,
                font=dict(size=10, color="#666677")
            ),
            annotations=[dict(
                text=f"<b>{st.session_state.total_queries}</b>",
                x=0.5, y=0.5,
                font=dict(size=16, color="#0058A3", family="Oswald"),
                showarrow=False
            )]
        )
        st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar":False})

    if st.session_state.intent_freq:
        st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)
        st.markdown('<div class="section-label">▸ Intent Frequency</div>', unsafe_allow_html=True)
        sorted_i = sorted(st.session_state.intent_freq.items(), key=lambda x: x[1], reverse=True)[:6]
        fig3 = go.Figure(go.Bar(
            x=[x[1] for x in sorted_i],
            y=[x[0].upper() for x in sorted_i],
            orientation="h",
            marker_color="#0058A3", opacity=0.85,
            text=[x[1] for x in sorted_i],
            textposition="auto",
            textfont=dict(color="#ffffff", size=10, family="Roboto Mono"),
        ))
        fig3.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#666677", family="Roboto Mono"),
            height=max(110, len(sorted_i)*30),
            margin=dict(l=0,r=0,t=0,b=0),
            xaxis=dict(showgrid=False, showticklabels=False),
            yaxis=dict(showgrid=False, tickfont=dict(size=9, color="#666677")),
            showlegend=False
        )
        st.plotly_chart(fig3, use_container_width=True, config={"displayModeBar":False})

# ============================================================
# HISTORY TABLE
# ============================================================
if st.session_state.history_log:
    st.markdown("<div style='height:6px;'></div>", unsafe_allow_html=True)
    st.markdown('<div class="section-label">▸ Query History</div>', unsafe_allow_html=True)
    df = pd.DataFrame(st.session_state.history_log)
    st.dataframe(df, use_container_width=True, hide_index=True,
        column_config={
            "Query"     : st.column_config.TextColumn("Query",     width="large"),
            "Intent"    : st.column_config.TextColumn("Intent",    width="medium"),
            "Confidence": st.column_config.TextColumn("Conf",      width="small"),
            "Sentiment" : st.column_config.TextColumn("Sentiment", width="small"),
            "Status"    : st.column_config.TextColumn("Status",    width="small"),
            "Latency"   : st.column_config.TextColumn("Latency",   width="small"),
            "Feedback"  : st.column_config.TextColumn("Feedback",  width="small"),
        }
    )
