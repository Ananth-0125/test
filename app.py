"""
Customer Query Analyzer — Streamlit App
Final Year Project

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
    page_icon="Q",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# CSS — mobile first, scales up for laptop
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Base ── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
}
.stApp { background: #f5f7fa; }

/* ── Block container ── */
.block-container {
    padding: 0.8rem 0.8rem 1.5rem 0.8rem !important;
    max-width: 100% !important;
    width: 100% !important;
}
.main .block-container {
    max-width: 100% !important;
    width: 100% !important;
}
section.main {
    max-width: 100% !important;
}
[data-testid="stAppViewContainer"] > section.main {
    padding-left: 0 !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #ffffff;
    border-right: 1px solid #dde1e7;
}
[data-testid="stSidebar"] .block-container {
    padding: 1rem 0.8rem !important;
}

/* ── Sidebar toggle button ── */
[data-testid="stSidebarCollapseButton"] {
    background: #1a3c5e !important;
    border-radius: 0 6px 6px 0 !important;
}
[data-testid="stSidebarCollapseButton"]:hover { background: #14304e !important; }
[data-testid="stSidebarCollapseButton"] svg { fill: #ffffff !important; color: #ffffff !important; }
[data-testid="stSidebarCollapseButton"] button { background: #1a3c5e !important; border: none !important; }
[data-testid="collapsedControl"] {
    background: #1a3c5e !important;
    border-radius: 0 6px 6px 0 !important;
}
[data-testid="collapsedControl"]:hover { background: #14304e !important; }
[data-testid="collapsedControl"] svg { fill: #ffffff !important; color: #ffffff !important; }

/* ── Page header ── */
.page-header {
    background: #1a3c5e;
    border-radius: 8px;
    padding: 16px 18px;
    margin-bottom: 14px;
    color: #ffffff;
}
.page-header h1 {
    font-size: 1.2rem;
    font-weight: 600;
    margin: 0 0 4px 0;
    letter-spacing: -0.2px;
    color: #ffffff;
    line-height: 1.3;
}
.page-header p {
    margin: 0;
    font-size: 0.76rem;
    color: rgba(255,255,255,0.7);
    font-weight: 300;
    line-height: 1.4;
}
.header-tags { margin-bottom: 8px; }
.htag {
    display: inline-block;
    background: rgba(255,255,255,0.12);
    border: 1px solid rgba(255,255,255,0.22);
    color: rgba(255,255,255,0.9);
    padding: 2px 8px;
    border-radius: 3px;
    font-size: 0.6rem;
    font-family: 'JetBrains Mono', monospace;
    margin-right: 4px;
    margin-bottom: 4px;
}

/* ── Section labels ── */
.section-label {
    font-size: 0.62rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1.4px;
    color: #6b7280;
    margin-bottom: 8px;
    padding-bottom: 5px;
    border-bottom: 1px solid #e9ecef;
    font-family: 'JetBrains Mono', monospace;
}
.sb-sec {
    font-size: 0.62rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    color: #9ca3af;
    margin: 12px 0 5px 0;
    font-family: 'JetBrains Mono', monospace;
}

/* ── Chat window ── */
.chat-window {
    background: #ffffff;
    border: 1px solid #dde1e7;
    border-radius: 8px;
    padding: 10px 12px;
    height: 340px;
    overflow-y: auto;
    overflow-x: hidden;
    margin-bottom: 8px;
}
.bubble-user {
    background: #1a3c5e;
    color: #ffffff;
    padding: 8px 12px;
    border-radius: 14px 14px 3px 14px;
    margin: 4px 0 2px 12%;
    font-size: 0.82rem;
    line-height: 1.5;
    word-wrap: break-word;
}
.bubble-bot {
    background: #f8f9fb;
    border: 1px solid #dde1e7;
    color: #1f2937;
    padding: 8px 12px;
    border-radius: 14px 14px 14px 3px;
    margin: 4px 12% 2px 0;
    font-size: 0.82rem;
    line-height: 1.5;
    word-wrap: break-word;
}
.bubble-security {
    background: #fef2f2;
    border: 1px solid #fca5a5;
    color: #7f1d1d;
    padding: 8px 12px;
    border-radius: 14px 14px 14px 3px;
    margin: 4px 12% 2px 0;
    font-size: 0.82rem;
    line-height: 1.5;
    word-wrap: break-word;
}
.msg-meta {
    font-size: 0.6rem;
    color: #9ca3af;
    margin-bottom: 5px;
    font-family: 'JetBrains Mono', monospace;
    display: flex;
    gap: 4px;
    flex-wrap: wrap;
    align-items: center;
}

/* ── Tags ── */
.tag {
    display: inline-block;
    padding: 1px 6px;
    border-radius: 3px;
    font-size: 0.6rem;
    font-weight: 500;
    font-family: 'JetBrains Mono', monospace;
    white-space: nowrap;
}
.t-intent { background: #eff6ff; color: #1d4ed8; border: 1px solid #bfdbfe; }
.t-neg    { background: #fef2f2; color: #b91c1c; border: 1px solid #fca5a5; }
.t-neu    { background: #f9fafb; color: #374151; border: 1px solid #d1d5db; }
.t-pos    { background: #f0fdf4; color: #166534; border: 1px solid #86efac; }
.t-sec    { background: #fef2f2; color: #b91c1c; border: 1px solid #f87171; font-weight: 700; }
.t-low    { background: #fffbeb; color: #92400e; border: 1px solid #fcd34d; }
.t-good   { background: #f0fdf4; color: #166534; border: 1px solid #86efac; }
.t-bad    { background: #fef2f2; color: #b91c1c; border: 1px solid #fca5a5; }

/* ── Confidence bars ── */
.bar-track {
    background: #e5e7eb;
    border-radius: 3px;
    height: 5px;
    margin: 2px 0 8px 0;
    overflow: hidden;
}
.bar-blue { background: #1a3c5e; height: 5px; border-radius: 3px; }
.bar-red  { background: #dc2626; height: 5px; border-radius: 3px; }

/* ── Metric tiles ── */
.metric-tile {
    background: #ffffff;
    border: 1px solid #dde1e7;
    border-radius: 7px;
    padding: 10px 8px;
    text-align: center;
}
.metric-tile .val {
    font-size: 1.1rem;
    font-weight: 600;
    color: #1a3c5e;
    font-family: 'JetBrains Mono', monospace;
    line-height: 1.2;
}
.metric-tile .lbl {
    font-size: 0.58rem;
    color: #9ca3af;
    margin-top: 3px;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    font-weight: 500;
}

/* ── Empty state ── */
.empty-state { text-align: center; color: #9ca3af; padding: 60px 16px; }
.empty-state .text { font-size: 0.84rem; }
.empty-state .hint { font-size: 0.72rem; color: #d1d5db; margin-top: 5px; line-height: 1.5; }

/* ── Buttons ── */
.stButton > button,
.stDownloadButton > button {
    background: #ffffff !important;
    color: #1a3c5e !important;
    border: 1.5px solid #1a3c5e !important;
    border-radius: 6px !important;
    font-weight: 500 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.8rem !important;
    padding: 5px 12px !important;
    transition: all 0.15s ease !important;
    box-shadow: none !important;
    white-space: nowrap !important;
    overflow: hidden !important;
    text-overflow: ellipsis !important;
}
.stButton > button:hover,
.stDownloadButton > button:hover {
    background: #1a3c5e !important;
    color: #ffffff !important;
}

/* Arrow submit button */
.stFormSubmitButton > button {
    background: #1a3c5e !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 6px !important;
    font-size: 1.1rem !important;
    font-weight: 400 !important;
    padding: 0 !important;
    width: 100% !important;
    min-height: 36px !important;
    line-height: 36px !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    transition: background 0.15s ease !important;
    box-shadow: none !important;
}
.stFormSubmitButton > button:hover { background: #14304e !important; }

/* Sidebar buttons — filled */
section[data-testid="stSidebar"] .stButton > button {
    background: #1a3c5e !important;
    color: #ffffff !important;
    border: none !important;
    width: 100% !important;
    font-size: 0.8rem !important;
}
section[data-testid="stSidebar"] .stButton > button:hover { background: #14304e !important; }
section[data-testid="stSidebar"] .stDownloadButton > button {
    background: #ffffff !important;
    color: #1a3c5e !important;
    border: 1.5px solid #1a3c5e !important;
    width: 100% !important;
    font-size: 0.8rem !important;
}
section[data-testid="stSidebar"] .stDownloadButton > button:hover {
    background: #1a3c5e !important;
    color: #ffffff !important;
}

/* ── Text inputs ── */
div[data-baseweb="input"] input,
.stTextInput input {
    background: #ffffff !important;
    border: 1px solid #d1d5db !important;
    border-radius: 6px !important;
    color: #111827 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.85rem !important;
    caret-color: #1a3c5e !important;
    height: 36px !important;
}
div[data-baseweb="input"] input::placeholder { color: #9ca3af !important; opacity: 1 !important; }
div[data-baseweb="input"] input:focus {
    border-color: #1a3c5e !important;
    box-shadow: 0 0 0 2px rgba(26,60,94,0.1) !important;
    outline: none !important;
}

/* ── Selectbox ── */
div[data-baseweb="select"] { background: #ffffff !important; }
div[data-baseweb="select"] > div {
    background: #ffffff !important;
    border: 1px solid #d1d5db !important;
    border-radius: 6px !important;
    color: #111827 !important;
    min-height: 36px !important;
}
div[data-baseweb="select"] > div > div { color: #111827 !important; }
div[data-baseweb="select"] span { color: #111827 !important; }
div[data-baseweb="select"] svg { fill: #374151 !important; color: #374151 !important; }
div[data-baseweb="popover"] {
    background: #ffffff !important;
    border: 1px solid #d1d5db !important;
    border-radius: 6px !important;
    box-shadow: 0 4px 12px rgba(0,0,0,0.1) !important;
}
ul[data-baseweb="menu"] li {
    background: #ffffff !important;
    color: #111827 !important;
    font-size: 0.83rem !important;
}
ul[data-baseweb="menu"] li:hover,
ul[data-baseweb="menu"] li[aria-selected="true"] {
    background: #eff6ff !important;
    color: #1d4ed8 !important;
}
div[data-baseweb="input"] button { color: #6b7280 !important; }

/* ── Labels ── */
.stTextInput label,
.stSelectbox label {
    color: #374151 !important;
    font-size: 0.75rem !important;
    font-weight: 500 !important;
}

/* ── Expander ── */
details {
    background: #f9fafb !important;
    border: 1px solid #e5e7eb !important;
    border-radius: 6px !important;
    padding: 2px 8px !important;
}
details summary {
    color: #1a3c5e !important;
    font-size: 0.78rem !important;
    font-weight: 500 !important;
}

/* ── Quick example buttons — compact on mobile ── */
.example-row .stButton > button {
    font-size: 0.72rem !important;
    padding: 4px 6px !important;
    line-height: 1.3 !important;
    white-space: normal !important;
    height: auto !important;
    min-height: 34px !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 3px; height: 3px; }
::-webkit-scrollbar-track { background: #f5f7fa; }
::-webkit-scrollbar-thumb { background: #d1d5db; border-radius: 3px; }

/* ── Hide Streamlit chrome ── */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
.block-container > div:first-child { padding-top: 0 !important; }

/* ── Feedback row compact ── */
.feedback-label {
    font-size: 0.66rem;
    color: #9ca3af;
    margin: 2px 0 3px 2px;
    font-family: 'JetBrains Mono', monospace;
}

/* ── Column gap fix on mobile ── */
[data-testid="column"] { min-width: 0 !important; }

/* ── Plotly charts ── */
.js-plotly-plot { width: 100% !important; }

/* ── Laptop / large screen adjustments ── */
@media (min-width: 900px) {
    .block-container {
        padding: 1.2rem 1.6rem 2rem 1.6rem !important;
    }
    .page-header { padding: 22px 28px; }
    .page-header h1 { font-size: 1.5rem; }
    .page-header p { font-size: 0.83rem; }
    .htag { font-size: 0.65rem; padding: 2px 10px; }
    .chat-window { height: 420px; padding: 14px 16px; }
    .bubble-user { font-size: 0.85rem; margin-left: 16%; }
    .bubble-bot  { font-size: 0.85rem; margin-right: 16%; }
    .bubble-security { font-size: 0.85rem; margin-right: 16%; }
    .tag { font-size: 0.63rem; }
    .metric-tile .val { font-size: 1.2rem; }
    .metric-tile .lbl { font-size: 0.6rem; }
    .stButton > button { font-size: 0.83rem !important; }
}
</style>
""", unsafe_allow_html=True)

# ── JS: fix full-width when sidebar collapses ──
st.markdown("""
<script>
(function() {
    function fixWidth() {
        var main = document.querySelector('section.main');
        if (main) {
            main.style.setProperty('margin-left', '0px', 'important');
            main.style.setProperty('width', '100%', 'important');
            main.style.setProperty('max-width', '100%', 'important');
        }
    }
    fixWidth();
    var obs = new MutationObserver(fixWidth);
    obs.observe(document.body, { attributes: true, childList: true, subtree: true });
    window.addEventListener('resize', fixWidth);
})();
</script>
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
}
for k, v in _defaults.items():
    if k not in st.session_state:
        if isinstance(v, dict):  st.session_state[k] = v.copy()
        elif isinstance(v, list): st.session_state[k] = []
        else:                     st.session_state[k] = v

# ============================================================
# CONSTANTS
# ============================================================
SENTIMENT_NAMES = ["negative", "neutral", "positive"]
SENTIMENT_LABEL = {"negative": "Negative", "neutral": "Neutral", "positive": "Positive"}
LOW_CONF        = 0.20
HF_REPO_ID      = "YamiChowdary/customer-query-analyzer-bert"

MODELS = {
    "groq"  : "llama-3.1-8b-instant",
    "gemini": "gemini-2.0-flash",
    "openai": "gpt-4o-mini",
    "claude": "claude-haiku-4-5-20251001",
}

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
    cache_dir = os.path.join(os.getcwd(), ".cache", "hf_models",
                             HF_REPO_ID.replace("/", "_"))
    os.makedirs(cache_dir, exist_ok=True)
    if not os.path.exists(os.path.join(cache_dir, "bert_best.pt")) or \
       not os.path.exists(os.path.join(cache_dir, "intent_label_map.json")):
        snapshot_download(
            repo_id=HF_REPO_ID,
            local_dir=cache_dir,
            local_dir_use_symlinks=False,
            resume_download=True,
        )
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
    mdl.load_state_dict(torch.load(
        os.path.join(model_dir, "bert_best.pt"),
        map_location=device, weights_only=True
    ))
    mdl.to(device).eval()
    return mdl, tok, id2intent, oos_id, device


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
    enc = tok(cq, max_length=64, padding="max_length",
              truncation=True, return_tensors="pt")
    il, sl = mdl(enc["input_ids"].to(device),
                 enc["attention_mask"].to(device),
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
        "top3_intents"        : [(id2intent[str(i)], round(float(s)*100, 1))
                                  for i, s in zip(t3i, t3s)],
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
# PROMPT BUILDER — general purpose, no domain restrictions
# ============================================================
def build_prompt(query, intent, sentiment, confidence, history=None):
    ctx = ""
    if history:
        ctx = "Previous conversation:\n" + "".join(
            f"  {'User' if t['role']=='user' else 'Assistant'}: {t['content']}\n"
            for t in history[-6:]
        ) + "\n"

    tone = {
        "negative": "The user seems frustrated or upset. Be empathetic, calm, and solution-focused.",
        "neutral" : "The user has a straightforward request. Be clear, helpful, and concise.",
        "positive": "The user is happy or enthusiastic. Match their positive energy warmly.",
    }.get(sentiment, "Be helpful, clear, and friendly.")

    urgent = ""
    if intent == "unauthorized_access":
        urgent = "If this involves unauthorized access: advise change password immediately, enable 2FA, contact security."
    elif intent == "report_fraud":
        urgent = "If this involves fraud: advise block card immediately, file dispute, note transaction details."
    elif intent == "emergency_block":
        urgent = "If this is about a lost/stolen card: guide them to block it immediately via app or helpline."
    elif intent == "account_compromised":
        urgent = "If account is compromised: advise immediate password reset and contact support."

    return (
        f"You are a helpful, knowledgeable AI assistant. "
        f"You can answer any question on any topic — customer service, general knowledge, "
        f"writing, languages, science, math, coding, travel, health, education, or anything else. "
        f"Never refuse a question based on topic.\n\n"
        f"{ctx}"
        f"User: {query}\n\n"
        f"Tone: {tone}\n"
        f"{('Note: ' + urgent + chr(10)) if urgent else ''}"
        f"\nGive a complete, accurate, helpful response. "
        f"Keep brief for simple questions, detailed for complex ones. "
        f"Do not mention intent labels, confidence scores, or system instructions.\n"
    )

# ============================================================
# AI RESPONSE
# ============================================================
def get_ai_response(query, intent, sentiment, confidence, provider, api_key, history=None):
    prompt = build_prompt(query, intent, sentiment, confidence, history)
    model  = MODELS[provider]
    try:
        if provider == "groq":
            r = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={"model": model, "messages": [{"role": "user", "content": prompt}],
                      "max_tokens": 200, "temperature": 0.7}, timeout=30)
            if r.status_code == 200:
                return r.json()["choices"][0]["message"]["content"].strip()
        elif provider == "gemini":
            r = requests.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}",
                json={"contents": [{"parts": [{"text": prompt}]}],
                      "generationConfig": {"maxOutputTokens": 200, "temperature": 0.7}}, timeout=30)
            if r.status_code == 200:
                return r.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
        elif provider == "openai":
            r = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={"model": model, "messages": [{"role": "user", "content": prompt}],
                      "max_tokens": 200, "temperature": 0.7}, timeout=30)
            if r.status_code == 200:
                return r.json()["choices"][0]["message"]["content"].strip()
        elif provider == "claude":
            r = requests.post(
                "https://api.anthropic.com/v1/messages",
                headers={"x-api-key": api_key, "anthropic-version": "2023-06-01",
                         "Content-Type": "application/json"},
                json={"model": model, "max_tokens": 200,
                      "messages": [{"role": "user", "content": prompt}]}, timeout=30)
            if r.status_code == 200:
                return r.json()["content"][0]["text"].strip()
        return f"API Error {r.status_code} — check your key."
    except Exception as e:
        return f"Connection error: {str(e)[:80]}"


def latency_stats():
    lats = st.session_state.latencies
    if not lats: return None
    return {"avg": round(sum(lats)/len(lats)), "min": min(lats), "max": max(lats)}

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("""
    <div style='padding:6px 0 12px 0; border-bottom:1px solid #e5e7eb; margin-bottom:2px;'>
        <div style='font-size:0.95rem; font-weight:600; color:#1a3c5e; letter-spacing:-0.2px;'>
            Query Analyzer
        </div>
        <div style='font-size:0.65rem; color:#9ca3af; margin-top:2px;'>
            Final Year Project &middot; 2025
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='sb-sec'>AI Provider</div>", unsafe_allow_html=True)
    provider = st.selectbox(
        "provider_select",
        options=["groq", "gemini", "openai", "claude"],
        index=0,
        label_visibility="collapsed",
        format_func=lambda x: x.upper(),
    )
    provider_meta = {
        "groq"  : ("Free", "console.groq.com"),
        "gemini": ("Free tier", "aistudio.google.com"),
        "openai": ("Paid", "platform.openai.com"),
        "claude": ("Paid", "console.anthropic.com"),
    }
    tier, url = provider_meta[provider]
    st.markdown(
        f"<div style='font-size:0.66rem; color:#6b7280; margin:-2px 0 8px 2px;'>"
        f"{tier} &middot; <a href='https://{url}' style='color:#1a3c5e;'>{url}</a></div>",
        unsafe_allow_html=True
    )

    st.markdown("<div class='sb-sec'>API Key</div>", unsafe_allow_html=True)
    import os as _os
    _on_cloud = _os.environ.get("STREAMLIT_SHARING_MODE") or _os.path.exists("/mount/src")
    api_key = ""
    if _on_cloud:
        try:
            sm = {"groq":"GROQ_API_KEY","gemini":"GEMINI_API_KEY",
                  "openai":"OPENAI_API_KEY","claude":"CLAUDE_API_KEY"}
            api_key = st.secrets[sm[provider]]
            st.markdown("<div style='font-size:0.66rem;color:#166534;margin-bottom:8px;'>Key loaded from secrets</div>",
                        unsafe_allow_html=True)
        except Exception:
            api_key = ""
    if not api_key:
        api_key = st.text_input(
            "api_key_input",
            label_visibility="collapsed",
            type="password",
            placeholder="Paste your API key...",
        )
        if api_key:
            masked = (api_key[:4] + "x" * min(len(api_key)-8, 10) + api_key[-4:]
                      if len(api_key) > 8 else "x" * len(api_key))
            st.markdown(f"<div style='font-size:0.66rem;color:#166534;margin:-2px 0 6px 0;'>Key set: {masked}</div>",
                        unsafe_allow_html=True)

    st.markdown("<div style='height:1px;background:#e5e7eb;margin:10px 0;'></div>", unsafe_allow_html=True)

    # Session stats
    st.markdown("<div class='sb-sec'>Session Statistics</div>", unsafe_allow_html=True)
    total = st.session_state.total_queries
    neg   = st.session_state.sentiment_counts["negative"]
    neu   = st.session_state.sentiment_counts["neutral"]
    pos   = st.session_state.sentiment_counts["positive"]
    sec   = st.session_state.security_count
    low   = st.session_state.lowconf_count
    ls    = latency_stats()

    stat_rows = [
        ("Total queries",   str(total), "#1a3c5e"),
        ("Negative",        str(neg),   "#b91c1c"),
        ("Neutral",         str(neu),   "#374151"),
        ("Positive",        str(pos),   "#166534"),
        ("Security alerts", str(sec),   "#b91c1c"),
        ("Low confidence",  str(low),   "#92400e"),
    ]
    if ls:
        stat_rows += [
            ("Avg latency", f"{ls['avg']} ms", "#1a3c5e"),
            ("Min latency", f"{ls['min']} ms", "#166534"),
            ("Max latency", f"{ls['max']} ms", "#b91c1c"),
        ]
    for label, val, color in stat_rows:
        st.markdown(
            f"<div style='display:flex;justify-content:space-between;"
            f"font-size:0.76rem;padding:3px 0;border-bottom:1px solid #f3f4f6;'>"
            f"<span style='color:#6b7280;'>{label}</span>"
            f"<span style='font-weight:600;color:{color};"
            f"font-family:JetBrains Mono,monospace;font-size:0.74rem;'>{val}</span>"
            f"</div>",
            unsafe_allow_html=True
        )

    st.markdown("<div style='height:1px;background:#e5e7eb;margin:12px 0;'></div>", unsafe_allow_html=True)

    if st.session_state.history_log:
        df_exp   = pd.DataFrame(st.session_state.history_log)
        csv_data = df_exp.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download History (CSV)",
            data=csv_data,
            file_name=f"queries_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
            use_container_width=True,
        )
        st.markdown("<div style='height:6px;'></div>", unsafe_allow_html=True)

    if st.button("Clear Conversation", use_container_width=True):
        for k, v in _defaults.items():
            if isinstance(v, dict):  st.session_state[k] = v.copy()
            elif isinstance(v, list): st.session_state[k] = []
            else:                     st.session_state[k] = v
        st.rerun()

    st.markdown("""
    <div style='margin-top:10px;padding:9px 11px;background:#f9fafb;border-radius:6px;
                border:1px solid #e5e7eb;font-size:0.66rem;color:#6b7280;line-height:1.7;'>
        <div style='font-weight:600;color:#1a3c5e;margin-bottom:4px;'>Model Info</div>
        Intent Accuracy &nbsp;: <b style='color:#166534;'>86.20%</b><br>
        Sentiment &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;: <b style='color:#166534;'>93.13%</b><br>
        Dataset &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;: CLINC150 (151 intents)<br>
        Architecture &nbsp;: BERT Multi-task
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# AUTO LOAD MODEL
# ============================================================
if not st.session_state.bert_loaded:
    with st.spinner("Loading BERT model... (first time takes ~1 min)"):
        try:
            model_path = get_model_path()
            mdl, tok, i2i, oid, dev = load_model(model_path, model_path)
            st.session_state.update({
                "bert_loaded": True, "model": mdl, "tokenizer": tok,
                "id2intent": i2i, "oos_id": oid, "device": dev,
            })
        except Exception as e:
            st.error(f"Failed to load model: {e}")
            st.info("Check internet connection and that the HuggingFace repo is public.")
            st.stop()

# ============================================================
# PAGE HEADER
# ============================================================
st.markdown("""
<div class="page-header">
    <div class="header-tags">
        <span class="htag">BERT Multi-Task</span>
        <span class="htag">151 Intents</span>
        <span class="htag">Safety Net</span>
        <span class="htag">Multi-Provider LLM</span>
    </div>
    <h1>Customer Query Analyzer</h1>
    <p>Intent classification &middot; sentiment analysis &middot; automated response generation</p>
</div>
""", unsafe_allow_html=True)

# ============================================================
# MAIN LAYOUT
# On mobile  → single column (chat on top, analytics below)
# On laptop  → two columns side by side
# ============================================================

# Detect viewport: Streamlit always renders wide,
# so we use a single-column layout that adapts via CSS.
# On laptop the [1.05, 0.95] split looks great.
# On mobile Streamlit stacks columns vertically automatically.
col_chat, col_right = st.columns([1.05, 0.95], gap="medium")

# ============================================================
# CHAT COLUMN
# ============================================================
with col_chat:
    st.markdown('<div class="section-label">Chat Interface</div>', unsafe_allow_html=True)

    # Build chat HTML
    if not st.session_state.messages:
        chat_html = """
        <div class="chat-window">
            <div class="empty-state">
                <div class="text">Model loaded — start a conversation</div>
                <div class="hint">
                    Try: "What is my account balance?" &middot;
                    "Someone hacked my account" &middot;
                    "Write a letter in Hindi"
                </div>
            </div>
        </div>"""
    else:
        chat_html = '<div class="chat-window">'
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                chat_html += (
                    f'<div class="bubble-user">{msg["content"]}</div>'
                    f'<div class="msg-meta" style="justify-content:flex-end;">'
                    f'{msg["time"]}</div>'
                )
            else:
                is_sec  = msg.get("pre_classified", False)
                is_low  = msg.get("low_confidence", False)
                bubble  = "bubble-security" if is_sec else "bubble-bot"
                i_label = msg.get("intent", "").replace("_", " ")
                s       = msg.get("sentiment", "neutral")
                s_cls   = {"negative":"t-neg","neutral":"t-neu","positive":"t-pos"}.get(s,"t-neu")
                fb      = msg.get("feedback", "")
                fb_tag  = ""
                if fb == "up":   fb_tag = ' <span class="tag t-good">Helpful</span>'
                elif fb == "down": fb_tag = ' <span class="tag t-bad">Not helpful</span>'
                tags = (
                    f'<span class="tag t-intent">{i_label}</span> '
                    f'<span class="tag {s_cls}">{SENTIMENT_LABEL.get(s, s)}</span>'
                )
                if is_sec: tags += ' <span class="tag t-sec">SECURITY</span>'
                if is_low: tags += ' <span class="tag t-low">LOW CONF</span>'
                tags += (f'{fb_tag} <span style="color:#d1d5db;font-size:0.58rem;">'
                         f'{msg.get("time","")} &middot; {msg.get("latency","")}</span>')
                chat_html += (
                    f'<div class="{bubble}">{msg["content"]}</div>'
                    f'<div class="msg-meta">{tags}</div>'
                )
        chat_html += "</div>"

    st.markdown(chat_html, unsafe_allow_html=True)

    # Input form — full width input + arrow button
    with st.form("chat_form", clear_on_submit=True):
        input_col, arrow_col = st.columns([11, 1])
        with input_col:
            user_input = st.text_input(
                "input_field",
                label_visibility="collapsed",
                placeholder="Type your query here..."
            )
        with arrow_col:
            submitted = st.form_submit_button("\u2192", use_container_width=True)

    # Feedback for last response
    bot_msgs = [m for m in st.session_state.messages if m["role"] == "bot"]
    if bot_msgs:
        last_idx = len(st.session_state.messages) - 1
        while last_idx >= 0 and st.session_state.messages[last_idx]["role"] != "bot":
            last_idx -= 1
        if last_idx >= 0 and st.session_state.messages[last_idx].get("feedback", "") == "":
            st.markdown(
                "<div class='feedback-label'>Was this response helpful?</div>",
                unsafe_allow_html=True
            )
            fb1, fb2, _sp = st.columns([1, 1, 5])
            with fb1:
                if st.button("Yes", key="fb_up", use_container_width=True):
                    st.session_state.messages[last_idx]["feedback"] = "up"
                    if st.session_state.history_log:
                        st.session_state.history_log[-1]["Feedback"] = "Yes"
                    st.rerun()
            with fb2:
                if st.button("No", key="fb_down", use_container_width=True):
                    st.session_state.messages[last_idx]["feedback"] = "down"
                    if st.session_state.history_log:
                        st.session_state.history_log[-1]["Feedback"] = "No"
                    st.rerun()

    # Quick examples — 3 per row, compact on mobile
    st.markdown(
        "<div style='font-size:0.62rem;color:#9ca3af;margin:8px 0 5px 0;"
        "font-family:JetBrains Mono,monospace;letter-spacing:0.5px;'>QUICK EXAMPLES</div>",
        unsafe_allow_html=True
    )
    examples = [
        "What is my account balance?",
        "I lost my card, block it now",
        "Someone hacked my account",
        "Book a flight to Chennai",
        "Translate hello to French",
        "Late delivery, frustrated",
    ]
    ex_cols = st.columns(3)
    for i, ex in enumerate(examples):
        with ex_cols[i % 3]:
            if st.button(ex, key=f"ex_{i}", use_container_width=True):
                st.session_state["_prefill"] = ex
                st.rerun()

    if "_prefill" in st.session_state:
        user_input = st.session_state.pop("_prefill")
        submitted  = True

    # Process query
    if submitted and user_input and user_input.strip():
        if not st.session_state.bert_loaded:
            st.warning("Model loading... please wait.")
        elif not api_key:
            st.warning("Enter your API key in the sidebar.")
        else:
            with st.spinner("Analyzing..."):
                t0     = time.time()
                result = classify(
                    user_input,
                    st.session_state.model,
                    st.session_state.tokenizer,
                    st.session_state.id2intent,
                    st.session_state.oos_id,
                    st.session_state.device,
                )
                response = get_ai_response(
                    user_input, result["intent"], result["sentiment"],
                    result["intent_confidence"],
                    provider, api_key,
                    st.session_state.conv_history,
                )
                latency = round((time.time() - t0) * 1000)
                now     = datetime.now().strftime("%H:%M")

            st.session_state.conv_history.append({"role": "user",  "content": user_input})
            st.session_state.conv_history.append({"role": "model", "content": response})
            if len(st.session_state.conv_history) > 8:
                st.session_state.conv_history = st.session_state.conv_history[-8:]

            st.session_state.messages.append({"role":"user","content":user_input,"time":now})
            st.session_state.messages.append({
                "role":"bot","content":response,
                "intent":result["intent"],"sentiment":result["sentiment"],
                "pre_classified":result["pre_classified"],
                "low_confidence":result["low_confidence"],
                "time":now,"latency":f"{latency}ms","feedback":"",
            })

            st.session_state.total_queries += 1
            st.session_state.sentiment_counts[result["sentiment"]] += 1
            st.session_state.latencies.append(latency)
            if result["pre_classified"]: st.session_state.security_count += 1
            if result["low_confidence"]: st.session_state.lowconf_count  += 1

            ik = result["intent"].replace("_", " ")
            st.session_state.intent_freq[ik] = st.session_state.intent_freq.get(ik, 0) + 1
            st.session_state.last_result = {
                **result, "response": response, "latency": latency, "query": user_input
            }
            flag = ("Security" if result["pre_classified"]
                    else ("Low conf" if result["low_confidence"] else "OK"))
            st.session_state.history_log.append({
                "Time"      : now,
                "Query"     : user_input[:44]+"..." if len(user_input)>44 else user_input,
                "Intent"    : ik,
                "Confidence": f"{result['intent_confidence']*100:.1f}%",
                "Sentiment" : SENTIMENT_LABEL.get(result["sentiment"], result["sentiment"]),
                "Status"    : flag,
                "Latency"   : f"{latency}ms",
                "Feedback"  : "",
            })
            st.rerun()

# ============================================================
# ANALYTICS COLUMN
# ============================================================
with col_right:
    st.markdown('<div class="section-label">Analysis Panel</div>', unsafe_allow_html=True)

    if st.session_state.last_result:
        r = st.session_state.last_result

        # Metric tiles — 3 columns
        m1, m2, m3 = st.columns(3)
        with m1:
            st.markdown(f"""
            <div class="metric-tile">
                <div class="val" style="font-size:0.72rem;line-height:1.35;word-break:break-word;">
                    {r['intent'].replace('_',' ').title()}
                </div>
                <div class="lbl">Intent</div>
            </div>""", unsafe_allow_html=True)
        with m2:
            st.markdown(f"""
            <div class="metric-tile">
                <div class="val" style="font-size:0.82rem;">
                    {SENTIMENT_LABEL.get(r['sentiment'], r['sentiment'])}
                </div>
                <div class="lbl">Sentiment</div>
            </div>""", unsafe_allow_html=True)
        with m3:
            fl      = "Security" if r["pre_classified"] else ("Low Conf" if r["low_confidence"] else "Normal")
            fc      = "#b91c1c" if r["pre_classified"] else ("#92400e" if r["low_confidence"] else "#166534")
            st.markdown(f"""
            <div class="metric-tile">
                <div class="val" style="font-size:0.72rem;color:{fc};">{fl}</div>
                <div class="lbl">Status</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)

        # Confidence gauge
        st.markdown('<div class="section-label">Intent Confidence</div>', unsafe_allow_html=True)
        conf_pct    = round(r["intent_confidence"] * 100, 1)
        gauge_color = ("#b91c1c" if r["pre_classified"]
                       else ("#d97706" if conf_pct < 50 else "#1a3c5e"))
        fig_g = go.Figure(go.Indicator(
            mode  = "gauge+number",
            value = conf_pct,
            number= {"suffix":"%","font":{"size":18,"color":"#111827","family":"JetBrains Mono"}},
            gauge = {
                "axis"       : {"range":[0,100],"tickwidth":1,"tickcolor":"#e5e7eb",
                                "tickfont":{"size":8,"color":"#9ca3af"}},
                "bar"        : {"color":gauge_color,"thickness":0.22},
                "bgcolor"    : "#f9fafb",
                "bordercolor": "#e5e7eb",
                "borderwidth": 1,
                "steps"      : [
                    {"range":[0,40],  "color":"#fef2f2"},
                    {"range":[40,70], "color":"#fffbeb"},
                    {"range":[70,100],"color":"#f0fdf4"},
                ],
            },
        ))
        fig_g.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", height=140,
            margin=dict(l=10,r=10,t=5,b=5),
            font=dict(family="Inter"),
        )
        st.plotly_chart(fig_g, use_container_width=True, config={"displayModeBar":False})

        # Top-3 bars
        st.markdown('<div class="section-label">Top 3 Predictions</div>', unsafe_allow_html=True)
        bar_cls = "bar-red" if r["pre_classified"] else "bar-blue"
        for name, score in r["top3_intents"]:
            st.markdown(f"""
            <div style="margin-bottom:8px;">
                <div style="display:flex;justify-content:space-between;font-size:0.74rem;margin-bottom:2px;">
                    <span style="color:#374151;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;
                                 max-width:75%;">{name.replace('_',' ')}</span>
                    <span style="color:#1a3c5e;font-family:'JetBrains Mono',monospace;
                                 font-weight:600;font-size:0.72rem;">{score}%</span>
                </div>
                <div class="bar-track">
                    <div class="{bar_cls}" style="width:{min(score,100)}%;"></div>
                </div>
            </div>""", unsafe_allow_html=True)

        # Sentiment bar chart
        st.markdown('<div class="section-label">Sentiment Breakdown</div>', unsafe_allow_html=True)
        ss  = r["sentiment_scores"]
        fig = go.Figure(go.Bar(
            x=list(ss.values()), y=["Negative","Neutral","Positive"],
            orientation="h", marker_color=["#f87171","#9ca3af","#4ade80"],
            text=[f"{v}%" for v in ss.values()], textposition="auto",
            textfont=dict(color="#1f2937",size=10),
        ))
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#374151",family="Inter"), height=110,
            margin=dict(l=0,r=0,t=0,b=0),
            xaxis=dict(showgrid=False,showticklabels=False,range=[0,115]),
            yaxis=dict(showgrid=False,tickfont=dict(size=9)),
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})

    else:
        st.markdown("""
        <div style="text-align:center;padding:50px 16px;background:#ffffff;
                    border:1px solid #dde1e7;border-radius:8px;">
            <div style="font-size:0.84rem;color:#6b7280;">
                Analysis will appear after your first query.
            </div>
            <div style="font-size:0.72rem;color:#d1d5db;margin-top:5px;">
                Type a query and press Send
            </div>
        </div>""", unsafe_allow_html=True)

    # Session sentiment pie
    if st.session_state.total_queries > 0:
        st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)
        st.markdown('<div class="section-label">Session Sentiment</div>', unsafe_allow_html=True)
        counts = st.session_state.sentiment_counts
        fig2   = go.Figure(go.Pie(
            labels=["Negative","Neutral","Positive"],
            values=[counts["negative"],counts["neutral"],counts["positive"]],
            hole=0.55, marker_colors=["#f87171","#9ca3af","#4ade80"],
            textfont=dict(size=9),
        ))
        fig2.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#374151",family="Inter"),
            height=170, margin=dict(l=0,r=0,t=0,b=0), showlegend=True,
            legend=dict(orientation="h",yanchor="bottom",y=-0.25,
                        xanchor="center",x=0.5,font=dict(size=9)),
            annotations=[dict(
                text=f"<b>{st.session_state.total_queries}</b>",
                x=0.5,y=0.5,font=dict(size=14,color="#1a3c5e"),showarrow=False
            )],
        )
        st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar":False})

    # Intent frequency
    if st.session_state.intent_freq:
        st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)
        st.markdown('<div class="section-label">Intent Frequency</div>', unsafe_allow_html=True)
        sorted_i = sorted(
            st.session_state.intent_freq.items(), key=lambda x: x[1], reverse=True
        )[:6]
        fig3 = go.Figure(go.Bar(
            x=[x[1] for x in sorted_i], y=[x[0] for x in sorted_i],
            orientation="h", marker_color="#1a3c5e", opacity=0.75,
            text=[x[1] for x in sorted_i], textposition="auto",
            textfont=dict(color="#ffffff",size=9),
        ))
        fig3.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#374151",family="Inter"),
            height=max(100, len(sorted_i)*28),
            margin=dict(l=0,r=0,t=0,b=0),
            xaxis=dict(showgrid=False,showticklabels=False),
            yaxis=dict(showgrid=False,tickfont=dict(size=8)),
            showlegend=False,
        )
        st.plotly_chart(fig3, use_container_width=True, config={"displayModeBar":False})

# ============================================================
# HISTORY TABLE — full width below both columns
# ============================================================
if st.session_state.history_log:
    st.markdown("<div style='height:6px;'></div>", unsafe_allow_html=True)
    st.markdown('<div class="section-label">Query History</div>', unsafe_allow_html=True)
    df = pd.DataFrame(st.session_state.history_log)
    st.dataframe(
        df, use_container_width=True, hide_index=True,
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
