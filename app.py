import streamlit as st
import pandas as pd
import numpy as np
import re
from pathlib import Path

# =========================
# Page
# =========================
st.set_page_config(
    page_title="אופטימיזציית שילוב קרנות השתלמות",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================
# RTL + UI styling
# =========================
st.markdown(
    """
    <style>
      :root{
        --bg: #0b1220;
        --card: rgba(255,255,255,0.06);
        --card2: rgba(255,255,255,0.08);
        --stroke: rgba(255,255,255,0.12);
        --text: rgba(255,255,255,0.92);
        --muted: rgba(255,255,255,0.72);
        --muted2: rgba(255,255,255,0.60);
        --accent: #7dd3fc;
        --accent2:#a78bfa;
        --good:#34d399;
        --warn:#fbbf24;
        --bad:#fb7185;
      }
      html, body, [class*="css"]  {direction: rtl; text-align: right;}
      .rtl {direction: rtl; text-align: right;}
      table {direction: rtl;}
      th, td {text-align: right !important;}

      /* Background */
      .stApp {
        background: radial-gradient(1200px 600px at 80% -10%, rgba(167,139,250,0.35), transparent 60%),
                    radial-gradient(1200px 600px at 15% 0%, rgba(125,211,252,0.25), transparent 55%),
                    linear-gradient(180deg, #070b14 0%, #0b1220 40%, #0a1224 100%);
        color: var(--text);
      }

      /* Sidebar */
      section[data-testid="stSidebar"]{
        background: rgba(0,0,0,0.22);
        border-left: 1px solid var(--stroke);
      }

      /* Typography */
      h1,h2,h3,h4 {letter-spacing: -0.2px;}
      .muted {color: var(--muted);}

      /* Header card */
      .hero {
        padding: 18px 18px 16px 18px;
        border: 1px solid var(--stroke);
        border-radius: 18px;
        background: linear-gradient(135deg, rgba(125,211,252,0.12), rgba(167,139,250,0.10));
        box-shadow: 0 10px 40px rgba(0,0,0,0.25);
        margin-bottom: 14px;
      }
      .hero-title{
        font-size: 1.45rem;
        font-weight: 750;
        margin: 0 0 6px 0;
      }
      .hero-sub{
        margin: 0;
        color: var(--muted);
        font-size: 0.98rem;
        line-height: 1.35rem;
      }
      .pill{
        display:inline-block;
        padding: 3px 10px;
        border-radius: 999px;
        border: 1px solid var(--stroke);
        background: rgba(255,255,255,0.05);
        color: var(--muted);
        font-size: 0.82rem;
        margin-left: 6px;
      }

      /* Cards */
      .card{
        border: 1px solid var(--stroke);
        border-radius: 18px;
        padding: 14px 14px 12px 14px;
        background: var(--card);
        box-shadow: 0 10px 40px rgba(0,0,0,0.22);
        height: 100%;
      }
      .card h3{
        font-size: 1.05rem;
        margin: 0 0 8px 0;
      }
      .kv{
        display:flex;
        gap: 10px;
        flex-wrap: wrap;
        margin: 6px 0 8px 0;
      }
      .k{
        border: 1px solid var(--stroke);
        background: rgba(255,255,255,0.05);
        border-radius: 12px;
        padding: 8px 10px;
        min-width: 120px;
      }
      .k .lab{color: var(--muted2); font-size: 0.78rem; margin-bottom: 2px;}
      .k .val{font-weight: 750; font-size: 1.05rem; color: var(--text);}
      .adv{
        color: var(--muted);
        font-size: 0.92rem;
        line-height: 1.25rem;
        margin-top: 8px;
      }

      /* Tables */
      .tbl-wrap{
        border: 1px solid var(--stroke);
        border-radius: 18px;
        overflow: hidden;
        background: rgba(0,0,0,0.18);
      }
      .tbl-wrap table{
        width: 100%;
        border-collapse: collapse;
      }
      .tbl-wrap th{
        background: rgba(255,255,255,0.06);
        color: var(--muted);
        font-weight: 650;
        padding: 10px 10px;
        border-bottom: 1px solid var(--stroke);
        font-size: 0.88rem;
      }
      .tbl-wrap td{
        padding: 10px 10px;
        border-bottom: 1px solid rgba(255,255,255,0.07);
        font-size: 0.92rem;
        color: var(--text);
        vertical-align: top;
      }
      .tbl-wrap tr:hover td{
        background: rgba(255,255,255,0.04);
      }

      /* Hide Streamlit default footer */
      footer {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================
# Helpers
# =========================
PARAM_MAP = [
    ("stocks",  ["סך חשיפה למניות", "חשיפה למניות"]),
    ("foreign", ["סך חשיפה לנכסים המושקעים בחו\"ל", "מושקעים בחו\"ל", "חשיפה לחו\"ל", "חו\"ל"]),
    ("sharpe",  ["מדד שארפ", "שארפ"]),
    ("illiquid",["נכסים לא סחירים", "לא סחירים"]),
    ("fx",      ["חשיפה למט\"ח", "מט\"ח"]),
    ("israel",  ["נכסים בארץ", "בארץ", "ישראל"]),  # נשמר רק לתצוגה, לא לחישוב יעד "בארץ"
]

def _to_float(v):
    """Parse numbers like '33.18%' / '1.24' / None."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return np.nan
    if isinstance(v, (int, float, np.number)):
        return float(v)
    s = str(v).strip()
    if s == "":
        return np.nan
    s = s.replace("\u200f", "").replace("\u200e", "")  # bidi marks
    s = s.replace(",", "")
    m = re.match(r"^(-?\d+(\.\d+)?)\s*%?$", s)
    if not m:
        return np.nan
    return float(m.group(1))

def _find_param_key(param_text: str):
    if not isinstance(param_text, str):
        return None
    p = param_text.strip()
    for key, needles in PARAM_MAP:
        for n in needles:
            if n in p:
                return key
    return None

@st.cache_data(show_spinner=False)
def load_excel(file_bytes: bytes) -> pd.DataFrame:
    """Return long-form table with columns:
       track, fund, stocks, foreign, sharpe, illiquid, fx, israel, israel_calc, manager
    """
    xls = pd.ExcelFile(file_bytes)
    rows = []
    for sheet in xls.sheet_names:
        df = xls.parse(sheet)
        if df.shape[1] < 2:
            continue

        param_col = df.columns[0]
        tmp = df.copy()
        tmp["_param_key"] = tmp[param_col].apply(_find_param_key)
        tmp = tmp[tmp["_param_key"].notna()].copy()
        if tmp.empty:
            continue

        fund_cols = [c for c in df.columns[1:] if str(c).strip() != ""]
        for fund in fund_cols:
            rec = {"track": sheet, "fund": str(fund).strip()}
            for _, r in tmp.iterrows():
                rec[r["_param_key"]] = _to_float(r[fund])
            rows.append(rec)

    out = pd.DataFrame(rows)
    for c in ["stocks","foreign","sharpe","illiquid","fx","israel"]:
        if c not in out.columns:
            out[c] = np.nan

    # כלל ישראל: לא להשתמש בעמודת "נכסים בארץ", אלא לחשב כהשלמה לחו"ל
    out["israel_calc"] = 100.0 - out["foreign"]

    out["manager"] = out["fund"].str.split().str[0]
    out = out.drop_duplicates(subset=["track","fund"]).reset_index(drop=True)
    return out

def render_table(df: pd.DataFrame):
    html = df.to_html(index=False, escape=False)
    st.markdown(f'<div class="tbl-wrap rtl">{html}</div>', unsafe_allow_html=True)

def fmt_pct(x, digits=2):
    if pd.isna(x):
        return ""
    return f"{x:.{digits}f}%"

def fmt_num(x, digits=3):
    if pd.isna(x):
        return ""
    return f"{x:.{digits}f}"

# =========================
# Header
# =========================
st.markdown(
    """
    <div class="hero rtl">
      <div class="hero-title">אופטימיזציית שילוב בין שני גופי השקעות</div>
      <p class="hero-sub">
        חישוב דטרמיניסטי על בסיס קובץ האקסל בלבד — בלי "ניחושים".
        <span class="pill">3 חלופות מובילות</span>
        <span class="pill">כלל ישראל: בארץ = 100% − חו"ל</span>
        <span class="pill">חלופה 3 לפי שירות</span>
      </p>
    </div>
    """,
    unsafe_allow_html=True
)

# =========================
# Sidebar: data
# =========================
st.sidebar.header("נתונים")
default_path = Path("data.xlsx")

use_upload = st.sidebar.toggle("להעלות קובץ אקסל במקום קובץ שמגיע עם הריפו", value=False)

file_bytes = None
if use_upload:
    up = st.sidebar.file_uploader("העלה קובץ .xlsx", type=["xlsx"])
    if up is not None:
        file_bytes = up.getvalue()
else:
    if default_path.exists():
        file_bytes = default_path.read_bytes()

if not file_bytes:
    st.info("כדי להתחיל: העלה קובץ אקסל (.xlsx) בסרגל הצד, או שים קובץ בשם data.xlsx ליד app.py בריפו.")
    st.stop()

data = load_excel(file_bytes)

tracks = sorted(data["track"].unique().tolist())
sel_tracks = st.sidebar.multiselect("אילו מסלולים לכלול?", tracks, default=tracks)

universe = data[data["track"].isin(sel_tracks)].copy()

exclude_ira = st.sidebar.toggle("להחריג מסלולי 'ניהול אישי' / IRA (אם קיימים)", value=True)
if exclude_ira:
    universe = universe[~universe["fund"].str.contains(r"\bIRA\b|ניהול אישי", regex=True, na=False)].copy()

st.sidebar.divider()
st.sidebar.header("יעדים ומגבלות")

# Presets
preset = st.sidebar.selectbox(
    "פריסט מהיר (לא חובה)",
    ["מותאם אישית", "30% חו\"ל · 40% מניות · <=20% לא סחיר", "40% חו\"ל · 25% מט\"ח · 20% לא סחיר", "60% חו\"ל · מט\"ח מקסימלי · 30% לא סחיר"],
    index=0
)

# Targets
target_mode = st.sidebar.radio(
    "איך מגדירים יעד 'בארץ'?",
    ["לא מגדירים", "מגדיר יעד לנכסים בארץ (מתורגם ליעד חו\"ל)", "מגדיר יעד לחו\"ל"],
    index=2
)

def _preset_defaults():
    if preset == "30% חו\"ל · 40% מניות · <=20% לא סחיר":
        return dict(stocks=40.0, foreign=30.0, fx=25.0, illiquid=20.0, max_illiquid=20.0)
    if preset == "40% חו\"ל · 25% מט\"ח · 20% לא סחיר":
        return dict(stocks=40.0, foreign=40.0, fx=25.0, illiquid=20.0, max_illiquid=20.0)
    if preset == "60% חו\"ל · מט\"ח מקסימלי · 30% לא סחיר":
        return dict(stocks=40.0, foreign=60.0, fx=60.0, illiquid=30.0, max_illiquid=30.0)
    return dict(stocks=40.0, foreign=30.0, fx=25.0, illiquid=20.0, max_illiquid=20.0)

d = _preset_defaults()

target_stocks = st.sidebar.slider("יעד חשיפה למניות (%)", 0.0, 150.0, d["stocks"], 0.5)

if target_mode == "מגדיר יעד לנכסים בארץ (מתורגם ליעד חו\"ל)":
    target_israel = st.sidebar.slider("יעד נכסים בארץ (%)", 0.0, 100.0, 70.0, 0.5)
    target_foreign = 100.0 - target_israel
else:
    target_foreign = st.sidebar.slider("יעד חשיפה לחו\"ל (%)", 0.0, 150.0, d["foreign"], 0.5)

target_fx = st.sidebar.slider("יעד חשיפה למט\"ח (%)", 0.0, 150.0, d["fx"], 0.5)
target_illiquid = st.sidebar.slider("יעד נכסים לא סחירים (%)", -10.0, 60.0, d["illiquid"], 0.5)

max_illiquid = st.sidebar.slider("מגבלת מקסימום נכסים לא סחירים (%)", -10.0, 60.0, d["max_illiquid"], 0.5)

tolerance = st.sidebar.slider("טולרנס ליעדי חו\"ל/מניות (לחלופות 2–3) (%)", 0.0, 10.0, 2.0, 0.5)
weight_step = st.sidebar.select_slider("רזולוציית חלוקה בין שני גופים", options=[0.5, 1.0, 2.0, 5.0], value=1.0)

objective = st.sidebar.selectbox(
    "קריטריון דירוג לחלופה 1",
    [
        "דיוק ליעדים (מינימום סטייה כוללת)",
        "מקסימום מדד שארפ (בתוך הטולרנס)",
        "מקסימום מט\"ח (בתוך הטולרנס)",
        "קרוב ככל האפשר למגבלת הלא-סחיר (מלמטה, בתוך הטולרנס)"
    ],
)

st.sidebar.divider()
st.sidebar.header("שירות (חלופה 3)")
enable_service = st.sidebar.toggle("להפעיל חלופה 3 לפי דירוג שירות", value=True)
st.sidebar.caption("אם כבוי — חלופה 3 תוצג כחזרה לגיבוי (דיוק) במקום שירות.")

default_scores = {
    "כלל": 6.5, "מנורה": 7.0, "הפניקס": 6.8, "מיטב": 6.6, "אנליסט": 6.7,
    "מגדל": 6.2, "מור": 6.4, "הראל": 6.6, "ילין": 6.9, "אלטשולר": 5.8,
    "אינפיניטי": 6.0, "סלייס": 5.5, "גלובל": 5.5
}
mgrs = sorted(universe["manager"].unique().tolist())
svc_scores = {}
if enable_service:
    with st.sidebar.expander("לעריכת ציוני שירות (0–10)", expanded=False):
        for m in mgrs:
            svc_scores[m] = st.number_input(
                f"{m}",
                min_value=0.0,
                max_value=10.0,
                value=float(default_scores.get(m, 6.0)),
                step=0.1
            )
else:
    # still populate with defaults so the code can run
    for m in mgrs:
        svc_scores[m] = float(default_scores.get(m, 6.0))

# =========================
# Data status
# =========================
c1, c2, c3, c4 = st.columns([1,1,1,1.4])
c1.metric("מסלולים שנבחרו", len(sel_tracks))
c2.metric("גופים/קופות ביקום", len(universe))
c3.metric("מנהלים ייחודיים", universe["manager"].nunique())
c4.markdown(f'<div class="muted rtl" style="padding-top:6px">כלל ישראל פעיל: <b>ישראל = 100% − חו״ל</b> (גם אם קיימת עמודה “בארץ”).</div>', unsafe_allow_html=True)

# Prepare universe for computation
needed_cols = ["stocks","foreign","fx","illiquid","sharpe","israel_calc","fund","track","manager"]
u = universe[needed_cols].copy()

core = u.dropna(subset=["stocks","foreign","illiquid","fx"]).reset_index(drop=True)
if core.empty:
    st.error("לא נמצאו מספיק נתונים לחישוב (חסרים ערכים בעמודות ליבה במסלולים שנבחרו).")
    st.stop()

# =========================
# Computation (vectorized)
# =========================
w_vals = np.arange(0.0, 100.0 + 1e-9, weight_step) / 100.0
n = len(core)

X = core[["stocks","foreign","fx","illiquid","sharpe","israel_calc"]].to_numpy(dtype=float)
funds = core["fund"].to_numpy()
tracks_arr = core["track"].to_numpy()
mgr_arr = core["manager"].to_numpy()

idx_i, idx_j = np.triu_indices(n, k=1)
Ai = X[idx_i]
Bj = X[idx_j]

pairs = idx_i.shape[0]
W = w_vals.reshape(1, -1)
WA = W
WB = 1.0 - W

mix = Ai[:, None, :] * WA[:, :, None] + Bj[:, None, :] * WB[:, :, None]

mix_stocks   = mix[:,:,0]
mix_foreign  = mix[:,:,1]
mix_fx       = mix[:,:,2]
mix_illiquid = mix[:,:,3]
mix_sharpe   = mix[:,:,4]
mix_israel   = mix[:,:,5]

ok = mix_illiquid <= max_illiquid + 1e-9

# Distance (L1) to all targets (good default for "דיוק")
dist = (
    np.abs(mix_foreign - target_foreign) +
    np.abs(mix_stocks - target_stocks) +
    np.abs(mix_fx - target_fx) +
    np.abs(mix_illiquid - target_illiquid)
)

def best_by_masked(value, maximize=False, extra_mask=None):
    m = ok.copy()
    if extra_mask is not None:
        m = m & extra_mask
    v = np.where(m, value, -np.inf if maximize else np.inf)
    k = np.argmax(v) if maximize else np.argmin(v)
    if not np.isfinite(v.flat[k]):
        return None
    return np.unravel_index(k, v.shape)

within_tol = (np.abs(mix_foreign - target_foreign) <= tolerance) & (np.abs(mix_stocks - target_stocks) <= tolerance)

# Option 1 pick
if objective == "דיוק ליעדים (מינימום סטייה כוללת)":
    pick1 = best_by_masked(dist, maximize=False)
    why1 = "מקסימום דיוק מספרי ביחס ליעדים שהוגדרו."
elif objective == "מקסימום מדד שארפ (בתוך הטולרנס)":
    pick1 = best_by_masked(mix_sharpe, maximize=True, extra_mask=within_tol)
    why1 = "שואף לשארפ משוקלל גבוה — תוך שמירה על עמידה בטולרנס."
elif objective == "מקסימום מט\"ח (בתוך הטולרנס)":
    pick1 = best_by_masked(mix_fx, maximize=True, extra_mask=within_tol)
    why1 = "שואף לחשיפה גבוהה למט\"ח — תוך שמירה על עמידה בטולרנס."
else:
    gap = (max_illiquid - mix_illiquid)
    gap = np.where(gap >= -1e-9, gap, np.inf)  # only from below
    pick1 = best_by_masked(gap, maximize=False, extra_mask=within_tol)
    why1 = "מנסה להתקרב למקסימום לא-סחיר (מלמטה) בלי לחרוג."

if pick1 is None:
    st.error("לא נמצא אף שילוב שעומד במגבלת הלא-סחיר/טולרנס. נסה להגדיל מגבלה/טולרנס או לכלול מסלולים נוספים.")
    st.stop()

def build_row(pick, rank, advantage, service_note=""):
    pk, wk = pick
    i = idx_i[pk]; j = idx_j[pk]
    wA = float(w_vals[wk]); wB = 1.0 - wA
    svc = wA*svc_scores.get(mgr_arr[i], 6.0) + wB*svc_scores.get(mgr_arr[j], 6.0)

    row = {
        "דירוג": rank,
        "קופה א'": funds[i],
        "מסלול א'": tracks_arr[i],
        "משקל א'": f"{wA*100:.1f}%",
        "קופה ב'": funds[j],
        "מסלול ב'": tracks_arr[j],
        "משקל ב'": f"{wB*100:.1f}%",
        "חו\"ל": fmt_pct(float(mix_foreign[pk, wk])),
        "ישראל (מחושב)": fmt_pct(float(mix_israel[pk, wk])),
        "מניות": fmt_pct(float(mix_stocks[pk, wk])),
        "מט\"ח": fmt_pct(float(mix_fx[pk, wk])),
        "לא סחיר": fmt_pct(float(mix_illiquid[pk, wk])),
        "שארפ (משוקלל)": fmt_num(float(mix_sharpe[pk, wk])),
        "שירות (משוקלל)": f"{svc:.2f}" if enable_service else "—",
        "יתרון מרכזי": advantage + (f" {service_note}" if service_note else "")
    }
    return row, {
        "pair": tuple(sorted([funds[i], funds[j]])),
        "svc": svc
    }

row1, meta1 = build_row(pick1, "1", why1)

# Build candidates dataframe (best weight per pair by dist) for options 2/3
dist_masked = np.where(ok, dist, np.inf)
best_w = np.argmin(dist_masked, axis=1)
best_score = dist_masked[np.arange(pairs), best_w]
mask_pairs = np.isfinite(best_score)

cand = []
for pk in np.where(mask_pairs)[0]:
    wk = int(best_w[pk])
    i = idx_i[pk]; j = idx_j[pk]
    wA = float(w_vals[wk]); wB = 1.0 - wA
    svc = wA*svc_scores.get(mgr_arr[i], 6.0) + wB*svc_scores.get(mgr_arr[j], 6.0)
    cand.append({
        "pair_k": pk, "w_k": wk,
        "dist": float(best_score[pk]),
        "sharpe": float(mix_sharpe[pk, wk]),
        "fx": float(mix_fx[pk, wk]),
        "illiquid": float(mix_illiquid[pk, wk]),
        "foreign": float(mix_foreign[pk, wk]),
        "stocks": float(mix_stocks[pk, wk]),
        "service": float(svc),
        "pair": tuple(sorted([funds[i], funds[j]])),
    })

df_cand = pd.DataFrame(cand)
if df_cand.empty:
    st.error("לא נותרו מועמדים אחרי סינון. נסה להרחיב טולרנס או מגבלה.")
    st.stop()

# Option 2: best Sharpe within tolerance; fallback to lowest dist
df2 = df_cand.copy()
df2["within"] = (np.abs(df2["foreign"] - target_foreign) <= tolerance) & (np.abs(df2["stocks"] - target_stocks) <= tolerance)
df2 = df2[df2["pair"] != meta1["pair"]].copy()

pick2 = None
if not df2.empty:
    df2_sort = df2.sort_values(["within","sharpe","dist"], ascending=[False, False, True]).reset_index(drop=True)
    r2 = df2_sort.iloc[0]
    pick2 = (int(r2["pair_k"]), int(r2["w_k"]))

rows = [row1]
meta2 = None

if pick2 is not None:
    row2, meta2 = build_row(pick2, "2", "חלופה שמנסה לשפר שארפ (או להישאר מאוד קרובה ליעדים) יחסית לחלופה 1.")
    rows.append(row2)

# Option 3: maximize service within tolerance (if enabled), else best distance alternative
df3 = df_cand.copy()
df3["within"] = (np.abs(df3["foreign"] - target_foreign) <= tolerance) & (np.abs(df3["stocks"] - target_stocks) <= tolerance)
exclude_pairs = {meta1["pair"]}
if meta2 is not None:
    exclude_pairs.add(meta2["pair"])
df3 = df3[~df3["pair"].isin(exclude_pairs)].copy()

pick3 = None
if not df3.empty:
    if enable_service:
        df3 = df3[df3["within"]].copy()
        if not df3.empty:
            r3 = df3.sort_values(["service","dist"], ascending=[False, True]).iloc[0]
            pick3 = (int(r3["pair_k"]), int(r3["w_k"]))
            row3, _ = build_row(pick3, "3", "חלופה שמעדיפה איכות שירות (משוקלל) גבוהה יותר במסגרת הטולרנס.", service_note="(ציונים ניתנים לעריכה בסרגל הצד).")
            rows.append(row3)
    else:
        # fallback: next-best by distance
        r3 = df3.sort_values(["dist"], ascending=[True]).iloc[0]
        pick3 = (int(r3["pair_k"]), int(r3["w_k"]))
        row3, _ = build_row(pick3, "3", "חלופת גיבוי: דיוק גבוה (אחרי שהחרגנו את שתי הראשונות).")
        rows.append(row3)

res = pd.DataFrame(rows)

# =========================
# Main: Cards + table
# =========================
st.subheader("3 חלופות מובילות (מדורג מהגבוה לנמוך)")

cards = st.columns(3)

def card_html(r: pd.Series):
    title = f"חלופה {r['דירוג']}"
    a = f"{r['קופה א\']} · {r['משקל א\']}"
    b = f"{r['קופה ב\']} · {r['משקל ב\']}"
    return f"""
    <div class="card rtl">
      <h3>{title}</h3>
      <div class="muted" style="margin-bottom:6px">{a}<br/>{b}</div>
      <div class="kv">
        <div class="k"><div class="lab">חו״ל</div><div class="val">{r['חו\"ל']}</div></div>
        <div class="k"><div class="lab">מניות</div><div class="val">{r['מניות']}</div></div>
        <div class="k"><div class="lab">לא סחיר</div><div class="val">{r['לא סחיר']}</div></div>
        <div class="k"><div class="lab">מט״ח</div><div class="val">{r['מט\"ח']}</div></div>
        <div class="k"><div class="lab">שארפ</div><div class="val">{r['שארפ (משוקלל)']}</div></div>
        <div class="k"><div class="lab">שירות</div><div class="val">{r['שירות (משוקלל)']}</div></div>
      </div>
      <div class="adv"><b>יתרון:</b> {r['יתרון מרכזי']}</div>
    </div>
    """

for i in range(min(3, len(res))):
    with cards[i]:
        st.markdown(card_html(res.iloc[i]), unsafe_allow_html=True)

st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

# Table
show_cols = [
    "דירוג",
    "קופה א'", "מסלול א'", "משקל א'",
    "קופה ב'", "מסלול ב'", "משקל ב'",
    "חו\"ל", "ישראל (מחושב)", "מניות", "מט\"ח", "לא סחיר", "שארפ (משוקלל)", "שירות (משוקלל)",
    "יתרון מרכזי",
]
render_table(res[show_cols])

# =========================
# Debug / transparency
# =========================
with st.expander("פירוט יעדים ומגבלות (לבדיקה)", expanded=False):
    st.markdown(
        f"""
        <div class="rtl muted">
          <ul>
            <li><b>יעדים:</b> חו״ל={target_foreign:.2f}% · מניות={target_stocks:.2f}% · מט״ח={target_fx:.2f}% · לא-סחיר={target_illiquid:.2f}%</li>
            <li><b>מגבלה:</b> לא-סחיר ≤ {max_illiquid:.2f}%</li>
            <li><b>טולרנס:</b> ±{tolerance:.2f}% ליעדי חו״ל/מניות עבור חלופות 2–3</li>
            <li><b>רזולוציית חלוקה:</b> {weight_step}%</li>
          </ul>
        </div>
        """,
        unsafe_allow_html=True
    )

with st.expander("להציג את כל הגופים והפרמטרים שנקלטו (לבדיקה)", expanded=False):
    show = universe[["track","fund","stocks","foreign","israel","israel_calc","fx","illiquid","sharpe"]].copy()
    show = show.sort_values(["track","fund"])
    # nicer formatting
    show_fmt = show.copy()
    for c in ["stocks","foreign","israel","israel_calc","fx","illiquid"]:
        show_fmt[c] = show_fmt[c].map(lambda x: "" if pd.isna(x) else f"{x:.2f}")
    show_fmt["sharpe"] = show_fmt["sharpe"].map(lambda x: "" if pd.isna(x) else f"{x:.3f}")
    render_table(show_fmt)
