# app.py
# -*- coding: utf-8 -*-

import re
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st


# =========================
# Page config
# =========================
st.set_page_config(
    page_title="Investment Mix App",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================
# Global CSS (RTL + nice UI)
# =========================
st.markdown(
    """
<style>
/* App base */
.block-container { padding-top: 2rem; padding-bottom: 2.5rem; max-width: 1200px; }
html, body, [class*="css"] { font-family: system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, "Noto Sans Hebrew", "Heebo", sans-serif; }

/* RTL helpers */
.rtl { direction: rtl; text-align: right; }
.rtl * { direction: rtl; text-align: right; }

/* Header */
.hero {
  border: 1px solid rgba(255,255,255,0.12);
  background: linear-gradient(135deg, rgba(60,120,255,0.18), rgba(0,0,0,0));
  padding: 18px 18px;
  border-radius: 16px;
  margin-bottom: 14px;
}
.hero h1 { margin: 0; font-size: 28px; }
.hero p  { margin: 6px 0 0 0; opacity: 0.85; }

/* Cards */
.card {
  border: 1px solid rgba(255,255,255,0.12);
  background: rgba(255,255,255,0.04);
  border-radius: 16px;
  padding: 14px 14px;
  box-shadow: 0 6px 18px rgba(0,0,0,0.14);
}
.card h3 { margin: 0 0 6px 0; font-size: 18px; }
.muted { opacity: 0.8; font-size: 13px; }
.kgrid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-top: 10px; }
.k {
  border: 1px solid rgba(255,255,255,0.10);
  background: rgba(255,255,255,0.03);
  border-radius: 12px;
  padding: 10px 10px;
}
.k .lab { font-size: 12px; opacity: 0.75; }
.k .val { font-size: 16px; font-weight: 700; margin-top: 2px; }

/* Table */
.dataframe { direction: rtl; }

/* Sidebar */
section[data-testid="stSidebar"] .block-container { padding-top: 1.25rem; }
</style>
""",
    unsafe_allow_html=True,
)


# =========================
# Utilities
# =========================
def _norm_col(s: str) -> str:
    """
    Normalize Hebrew column names to match even if there are different quote marks:
    - removes spaces and punctuation
    - normalizes geresh / gershayim variants
    """
    if s is None:
        return ""
    s = str(s)

    # Normalize quote marks / apostrophes commonly found in Hebrew text
    s = s.replace("׳", "'").replace("״", '"').replace("`", "'").replace("´", "'")
    s = s.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")

    # Remove whitespace
    s = re.sub(r"\s+", "", s)

    # Remove punctuation except letters/numbers (keeps Hebrew)
    s = re.sub(r"[^\w\u0590-\u05FF]", "", s)

    return s.lower()


def find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """
    Find a column in df that matches any of candidates after normalization.
    """
    norm_map = {_norm_col(c): c for c in df.columns}
    for cand in candidates:
        key = _norm_col(cand)
        if key in norm_map:
            return norm_map[key]
    return None


@st.cache_data(show_spinner=False)
def load_data_from_excel(path: str) -> pd.DataFrame:
    df = pd.read_excel(path)
    # drop fully empty rows
    df = df.dropna(how="all").copy()
    return df


def pct(x) -> str:
    try:
        return f"{float(x):.0f}%"
    except Exception:
        return str(x)


def safe_num(x, default=np.nan) -> float:
    try:
        if pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


def build_card_html(row: pd.Series, cols: dict[str, str]) -> str:
    # cols keys: rank, kupa_a, mishkal_a, kupa_b, mishkal_b, yatro, moniot, lo_sachir, matach
    rank = row.get(cols["rank"], "")
    kupa_a = row.get(cols["kupa_a"], "")
    mishkal_a = row.get(cols["mishkal_a"], "")
    kupa_b = row.get(cols["kupa_b"], "")
    mishkal_b = row.get(cols["mishkal_b"], "")

    yatro = row.get(cols["yatro"], "")
    moniot = row.get(cols["moniot"], "")
    lo_sachir = row.get(cols["lo_sachir"], "")
    matach = row.get(cols["matach"], "")

    title = f"חלופה {rank}"
    line1 = f"{kupa_a} · {mishkal_a}"
    line2 = f"{kupa_b} · {mishkal_b}"

    return f"""
<div class="card rtl">
  <h3>{title}</h3>
  <div class="muted">{line1}<br/>{line2}</div>
  <div class="kgrid">
    <div class="k"><div class="lab">יתרה</div><div class="val">{yatro}</div></div>
    <div class="k"><div class="lab">מניות</div><div class="val">{moniot}</div></div>
    <div class="k"><div class="lab">לא סחיר</div><div class="val">{lo_sachir}</div></div>
    <div class="k"><div class="lab">מט״ח</div><div class="val">{matach}</div></div>
  </div>
</div>
"""


# =========================
# Header
# =========================
st.markdown(
    """
<div class="hero rtl">
  <h1>כלי התאמת מסלולי השקעה בקרנות השתלמות</h1>
  <p>השווה חלופות לפי קירבה ליעד חשיפה (מניות / לא סחיר / מט״ח), ובחר את 3 הקרובות ביותר.</p>
</div>
""",
    unsafe_allow_html=True,
)

# =========================
# Data source
# =========================
DEFAULT_FILE = "data.xlsx"
data_path = None

# If data.xlsx exists in repo – use it. Otherwise allow upload.
if Path(DEFAULT_FILE).exists():
    data_path = DEFAULT_FILE
else:
    st.info("לא נמצא data.xlsx בריפו. אפשר להעלות כאן קובץ אקסל במקום.")
    up = st.file_uploader("העלה קובץ Excel", type=["xlsx"])
    if up is not None:
        data_path = up

if data_path is None:
    st.stop()

try:
    df = load_data_from_excel(data_path)
except Exception as e:
    st.error(f"שגיאה בקריאת הקובץ: {e}")
    st.stop()

if df.empty:
    st.warning("הקובץ נטען אבל לא נמצאו נתונים.")
    st.stop()

# =========================
# Column mapping (robust to Hebrew quotes)
# =========================
# Try to locate columns even if they appear with geresh variants etc.
cols = {}

cols["rank"] = find_col(df, ["דירוג", "דירוג חלופה", "rank", "rating"])  # must exist
cols["kupa_a"] = find_col(df, ["א קופה", "קופה א", "א׳ קופה", "קופה א׳"])
cols["mishkal_a"] = find_col(df, ["א משקל", "משקל א", "א׳ משקל", "משקל א׳"])
cols["kupa_b"] = find_col(df, ["ב קופה", "קופה ב", "ב׳ קופה", "קופה ב׳"])
cols["mishkal_b"] = find_col(df, ["ב משקל", "משקל ב", "ב׳ משקל", "משקל ב׳"])

# Exposures / metrics
cols["yatro"] = find_col(df, ["יתרה", "יתרה₪", "יתרה ש״ח", "balance"])
cols["moniot"] = find_col(df, ["מניות", "% מניות", "אחוז מניות"])
cols["lo_sachir"] = find_col(df, ["לא סחיר", "% לא סחיר", "אלטרנטיבי", "illiquid"])
cols["matach"] = find_col(df, ["מט\"ח", "מט״ח", "% מט\"ח", "% מט״ח", "fx"])

missing = [k for k, v in cols.items() if v is None]
if missing:
    st.error(
        "חסרות עמודות בקובץ. חסר לי: "
        + ", ".join(missing)
        + "\n\nטיפ: בדוק את שמות העמודות בקובץ (שורה ראשונה) או הוסף זמנית `st.write(df.columns)`."
    )
    st.stop()

# =========================
# Sidebar controls
# =========================
with st.sidebar:
    st.markdown('<div class="rtl">', unsafe_allow_html=True)
    st.subheader("הגדרות יעד", anchor=False)

    target_equity = st.slider("יעד מניות (%)", 0, 100, 60, 1)
    target_illiquid = st.slider("יעד לא סחיר (%)", 0, 100, 10, 1)
    target_fx = st.slider('יעד מט״ח (%)', 0, 100, 30, 1)

    st.divider()

    st.subheader("משקולות", anchor=False)
    w_equity = st.slider("משקל מניות", 0.0, 5.0, 2.0, 0.1)
    w_illiquid = st.slider("משקל לא סחיר", 0.0, 5.0, 1.5, 0.1)
    w_fx = st.slider('משקל מט״ח', 0.0, 5.0, 1.0, 0.1)

    st.caption("הציון הוא מרחק משוקלל מהיעד: נמוך יותר = קרוב יותר.")
    st.markdown("</div>", unsafe_allow_html=True)

# =========================
# Compute distance score
# =========================
work = df.copy()

# Ensure numeric where needed
work["_equity"] = pd.to_numeric(work[cols["moniot"]], errors="coerce")
work["_illiquid"] = pd.to_numeric(work[cols["lo_sachir"]], errors="coerce")
work["_fx"] = pd.to_numeric(work[cols["matach"]], errors="coerce")

# If values are like "35%" – try strip
for c in ["_equity", "_illiquid", "_fx"]:
    if work[c].dtype == object:
        work[c] = work[c].astype(str).str.replace("%", "", regex=False)
        work[c] = pd.to_numeric(work[c], errors="coerce")

# Distance
work["_dist"] = (
    w_equity * (work["_equity"] - target_equity).abs()
    + w_illiquid * (work["_illiquid"] - target_illiquid).abs()
    + w_fx * (work["_fx"] - target_fx).abs()
)

# Sort best
best = work.sort_values("_dist", ascending=True).head(3).copy()

# Pretty values in cards (if numeric)
def fmt_val(x):
    if pd.isna(x):
        return "—"
    # if looks like percent
    try:
        return f"{float(x):.0f}%"
    except Exception:
        return str(x)

best_card = best.copy()
best_card[cols["moniot"]] = best_card["_equity"].apply(fmt_val)
best_card[cols["lo_sachir"]] = best_card["_illiquid"].apply(fmt_val)
best_card[cols["matach"]] = best_card["_fx"].apply(fmt_val)

# =========================
# Main layout
# =========================
left, right = st.columns([1.15, 0.85], vertical_alignment="top")

with left:
    st.markdown('<div class="rtl">', unsafe_allow_html=True)
    st.subheader("3 החלופות המובילות", anchor=False)

    c1, c2, c3 = st.columns(3, vertical_alignment="top")
    cards = [c1, c2, c3]

    for i, (_, r) in enumerate(best_card.iterrows()):
        html = build_card_html(r, cols)
        with cards[i]:
            st.markdown(html, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown('<div class="rtl">', unsafe_allow_html=True)
    st.subheader("טבלת השוואה", anchor=False)

    # Build a compact comparison table
    show_cols = [
        cols["rank"],
        cols["kupa_a"],
        cols["mishkal_a"],
        cols["kupa_b"],
        cols["mishkal_b"],
        cols["yatro"],
        cols["moniot"],
        cols["lo_sachir"],
        cols["matach"],
    ]

    out = best_card[show_cols].copy()

    # Try format weights if numeric
    for wcol in [cols["mishkal_a"], cols["mishkal_b"]]:
        out[wcol] = out[wcol].apply(lambda x: pct(x) if str(x).strip().replace(".", "", 1).isdigit() else x)

    st.dataframe(out, use_container_width=True, hide_index=True)

    with st.expander("הצג את כל הנתונים (לבדיקה)"):
        st.dataframe(work, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)
