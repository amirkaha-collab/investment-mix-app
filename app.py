import streamlit as st
import pandas as pd
import numpy as np
import re
from pathlib import Path

st.set_page_config(page_title="אופטימיזציית שילוב קרנות השתלמות", layout="wide")

# --- RTL styling ---
st.markdown(
    """
    <style>
      html, body, [class*="css"]  {direction: rtl; text-align: right;}
      .rtl {direction: rtl; text-align: right;}
      table {direction: rtl;}
      th, td {text-align: right !important;}
      .small {font-size: 0.9rem; opacity: 0.9;}
      .mono {font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;}
    </style>
    """,
    unsafe_allow_html=True
)

st.title("אופטימיזציית שילוב בין שני גופי השקעות")
st.caption("הכלי מחשב שילובים באופן דטרמיניסטי על סמך נתוני האקסל בלבד (ללא “ניחושים”).")

# ---------- Helpers ----------
PARAM_MAP = [
    ("stocks",  ["סך חשיפה למניות", "חשיפה למניות"]),
    ("foreign", ["סך חשיפה לנכסים המושקעים בחו\"ל", "מושקעים בחו\"ל", "חשיפה לחו\"ל", "חו\"ל"]),
    ("sharpe",  ["מדד שארפ", "שארפ"]),
    ("illiquid",["נכסים לא סחירים", "לא סחירים"]),
    ("fx",      ["חשיפה למט\"ח", "מט\"ח"]),
    ("israel",  ["נכסים בארץ", "בארץ", "ישראל"]),
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
       track, fund, stocks, foreign, sharpe, illiquid, fx, israel
    """
    xls = pd.ExcelFile(file_bytes)
    rows = []
    for sheet in xls.sheet_names:
        df = xls.parse(sheet)
        if df.shape[1] < 2:
            continue
        # First column is parameter names
        param_col = df.columns[0]
        # Map each row to param key
        tmp = df.copy()
        tmp["_param_key"] = tmp[param_col].apply(_find_param_key)
        tmp = tmp[tmp["_param_key"].notna()].copy()
        if tmp.empty:
            continue

        # Build fund dicts
        fund_cols = [c for c in df.columns[1:] if str(c).strip() != ""]
        for fund in fund_cols:
            rec = {"track": sheet, "fund": str(fund).strip()}
            for _, r in tmp.iterrows():
                k = r["_param_key"]
                rec[k] = _to_float(r[fund])
            rows.append(rec)

    out = pd.DataFrame(rows)
    # Ensure numeric columns exist
    for c in ["stocks","foreign","sharpe","illiquid","fx","israel"]:
        if c not in out.columns:
            out[c] = np.nan

    # Add computed Israel as 100 - foreign (requested rule)
    out["israel_calc"] = 100.0 - out["foreign"]

    # A simple manager name (first token) for grouping / service scoring
    out["manager"] = out["fund"].str.split().str[0]

    # Drop duplicates
    out = out.drop_duplicates(subset=["track","fund"]).reset_index(drop=True)
    return out

def rtl_table(df: pd.DataFrame):
    html = df.to_html(index=False, escape=False)
    st.markdown(f'<div class="rtl">{html}</div>', unsafe_allow_html=True)

def pick_top_k(df_scores: pd.DataFrame, k: int = 3, diversify: bool = True) -> pd.DataFrame:
    """Pick top k rows by 'score_rank' (ascending)."""
    if df_scores.empty:
        return df_scores
    df_scores = df_scores.sort_values("score_rank", ascending=True).reset_index(drop=True)
    if not diversify:
        return df_scores.head(k)
    chosen = []
    seen_pairs = set()
    seen_managers = set()
    for _, r in df_scores.iterrows():
        pair = tuple(sorted([r["fund_a"], r["fund_b"]]))
        # diversify: avoid same pair & try to diversify managers where possible
        mgr_pair = tuple(sorted([r["manager_a"], r["manager_b"]]))
        if pair in seen_pairs:
            continue
        if len(chosen) < 2 and mgr_pair in seen_managers:
            # allow repeats later, but diversify early
            continue
        chosen.append(r)
        seen_pairs.add(pair)
        seen_managers.add(mgr_pair)
        if len(chosen) >= k:
            break
    return pd.DataFrame(chosen)

def explain_row(r):
    parts = []
    # Primary advantage depends on objective label already stored
    parts.append(r["why_primary"])
    # Secondary: highlight best among metrics vs option 1 is handled later
    return " · ".join([p for p in parts if p])

# ---------- Sidebar: data ----------
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

# Universe selection
tracks = sorted(data["track"].unique().tolist())
sel_tracks = st.sidebar.multiselect("אילו מסלולים לכלול?", tracks, default=tracks)

universe = data[data["track"].isin(sel_tracks)].copy()

# Filter out IRA / personal management if requested
exclude_ira = st.sidebar.toggle("להחריג מסלולי 'ניהול אישי' / IRA (אם קיימים)", value=True)
if exclude_ira:
    universe = universe[~universe["fund"].str.contains(r"\bIRA\b|ניהול אישי", regex=True, na=False)].copy()

st.sidebar.divider()
st.sidebar.header("יעדים ומגבלות")

# Targets
target_mode = st.sidebar.radio("איך מגדירים 'נכסים בארץ'?", ["לא מגדירים", "מגדיר יעד לנכסים בארץ (מתורגם ליעד חו\"ל)", "מגדיר יעד לחו\"ל"], index=2)

target_stocks = st.sidebar.slider("יעד חשיפה למניות (%)", 0.0, 150.0, 40.0, 0.5)
if target_mode == "מגדיר יעד לנכסים בארץ (מתורגם ליעד חו\"ל)":
    target_israel = st.sidebar.slider("יעד נכסים בארץ (%)", 0.0, 100.0, 70.0, 0.5)
    target_foreign = 100.0 - target_israel
else:
    target_foreign = st.sidebar.slider("יעד חשיפה לחו\"ל (%)", 0.0, 150.0, 30.0, 0.5)

target_fx = st.sidebar.slider("יעד חשיפה למט\"ח (%)", 0.0, 150.0, 25.0, 0.5)
target_illiquid = st.sidebar.slider("יעד נכסים לא סחירים (%)", -10.0, 60.0, 20.0, 0.5)

# Constraints
max_illiquid = st.sidebar.slider("מגבלת מקסימום נכסים לא סחירים (%)", -10.0, 60.0, 20.0, 0.5)
tolerance = st.sidebar.slider("טולרנס ליעדים (לחלופות 2-3) (%)", 0.0, 10.0, 2.0, 0.5)

weight_step = st.sidebar.select_slider("רזולוציית חלוקה בין שני גופים", options=[0.5, 1.0, 2.0, 5.0], value=1.0)

objective = st.sidebar.selectbox(
    "מה הקריטריון הראשי לדירוג (חלופה 1)?",
    ["דיוק ליעדים (מינימום סטייה)", "מקסימום מדד שארפ (בתוך הטולרנס)", "מקסימום מט\"ח (בתוך הטולרנס)", "להיות קרוב ככל האפשר למגבלת הלא-סחיר (מלמטה)"],
)

st.sidebar.divider()
st.sidebar.header("שירות (חלופה 3)")

st.sidebar.caption("חלופה 3 תעדיף שילוב עם ציון שירות משוקלל גבוה. אפשר לעדכן כאן ציונים (0-10).")

# Default service scores (editable)
default_scores = {
    "כלל": 6.5, "מנורה": 7.0, "הפניקס": 6.8, "מיטב": 6.6, "אנליסט": 6.7,
    "מגדל": 6.2, "מור": 6.4, "הראל": 6.6, "ילין": 6.9, "אלטשולר": 5.8,
    "אינפיניטי": 6.0, "סלייס": 5.5, "גלובל": 5.5
}
mgrs = sorted(universe["manager"].unique().tolist())
svc_scores = {}
for m in mgrs:
    svc_scores[m] = st.sidebar.number_input(f"{m}", min_value=0.0, max_value=10.0, value=float(default_scores.get(m, 6.0)), step=0.1)

# ---------- Computation ----------
st.subheader("מצב הנתונים")
c1, c2, c3 = st.columns(3)
c1.metric("מסלולים שנבחרו", len(sel_tracks))
c2.metric("גופים / קופות ביקום", len(universe))
c3.metric("מנהלים ייחודיים", universe["manager"].nunique())

st.markdown('<div class="small rtl">הערה: "נכסים בארץ" מחושב גם כ־100% − חו"ל (כלל ישראל). אם המשתמש מבקש יעד "בארץ", אנחנו מתרגמים אותו ליעד חו"ל בהתאם.</div>', unsafe_allow_html=True)

# Prepare arrays
needed_cols = ["stocks","foreign","fx","illiquid","sharpe","israel_calc","fund","track","manager"]
u = universe[needed_cols].copy()

# Remove funds missing required core inputs (stocks/foreign/illiquid/fx)
core = u.dropna(subset=["stocks","foreign","illiquid","fx"]).reset_index(drop=True)
if core.empty:
    st.error("לא נמצאו מספיק נתונים לחישוב (חסרות עמודות/ערכים במסלולים שנבחרו).")
    st.stop()

# Build pairwise combinations
w_vals = np.arange(0.0, 100.0 + 1e-9, weight_step) / 100.0  # weight for A
n = len(core)

# Pre-extract numeric matrices
X = core[["stocks","foreign","fx","illiquid","sharpe","israel_calc"]].to_numpy(dtype=float)
funds = core["fund"].to_numpy()
tracks_arr = core["track"].to_numpy()
mgr_arr = core["manager"].to_numpy()

# Pair indices i<j
idx_i, idx_j = np.triu_indices(n, k=1)

Ai = X[idx_i]  # shape (pairs, 6)
Bj = X[idx_j]

# Expand weights: for each pair, for each weight
pairs = idx_i.shape[0]
W = w_vals.reshape(1, -1)  # (1, m)
WA = W
WB = 1.0 - W

# Weighted mix for each metric (broadcast)
mix = Ai[:, None, :] * WA[:, :, None] + Bj[:, None, :] * WB[:, :, None]  # (pairs, m, 6)

mix_stocks   = mix[:,:,0]
mix_foreign  = mix[:,:,1]
mix_fx       = mix[:,:,2]
mix_illiquid = mix[:,:,3]
mix_sharpe   = mix[:,:,4]
mix_israel   = mix[:,:,5]

# Constraints
ok = mix_illiquid <= max_illiquid + 1e-9

# Define distance to targets (L1)
dist = np.abs(mix_foreign - target_foreign) + np.abs(mix_stocks - target_stocks) + np.abs(mix_fx - target_fx) + np.abs(mix_illiquid - target_illiquid)

# Mask infeasible
dist_masked = np.where(ok, dist, np.inf)

# Helper to get best index for different objectives
def best_by_masked(value, maximize=False, extra_mask=None):
    m = ok.copy()
    if extra_mask is not None:
        m = m & extra_mask
    v = np.where(m, value, -np.inf if maximize else np.inf)
    if maximize:
        k = np.argmax(v)
    else:
        k = np.argmin(v)
    if not np.isfinite(v.flat[k]):
        return None
    return np.unravel_index(k, v.shape)  # (pair_idx, w_idx)

# Objective 1 (primary)
if objective == "דיוק ליעדים (מינימום סטייה)":
    pick1 = best_by_masked(dist, maximize=False)
elif objective == "מקסימום מדד שארפ (בתוך הטולרנס)":
    within = (np.abs(mix_foreign - target_foreign) <= tolerance) & (np.abs(mix_stocks - target_stocks) <= tolerance)
    pick1 = best_by_masked(mix_sharpe, maximize=True, extra_mask=within)
elif objective == "מקסימום מט\"ח (בתוך הטולרנס)":
    within = (np.abs(mix_foreign - target_foreign) <= tolerance) & (np.abs(mix_stocks - target_stocks) <= tolerance)
    pick1 = best_by_masked(mix_fx, maximize=True, extra_mask=within)
else:
    # closest to illiquid cap from below, with core within tolerance on stocks/foreign
    within = (np.abs(mix_foreign - target_foreign) <= tolerance) & (np.abs(mix_stocks - target_stocks) <= tolerance)
    gap = (max_illiquid - mix_illiquid)
    gap = np.where(gap >= -1e-9, gap, np.inf)  # only from below
    pick1 = best_by_masked(gap, maximize=False, extra_mask=within)

def row_from_pick(pick, label, why_primary):
    pair_k, w_k = pick
    i = idx_i[pair_k]; j = idx_j[pair_k]
    wA = float(w_vals[w_k]); wB = 1.0 - wA
    out = {
        "דירוג": label,
        "קופה א'": funds[i],
        "מסלול א'": tracks_arr[i],
        "משקל א' (%)": round(wA*100, 1),
        "קופה ב'": funds[j],
        "מסלול ב'": tracks_arr[j],
        "משקל ב' (%)": round(wB*100, 1),
        'חו"ל (%)': round(float(mix_foreign[pair_k, w_k]), 2),
        'ישראל (%) (מחושב)': round(float(mix_israel[pair_k, w_k]), 2),
        'מניות (%)': round(float(mix_stocks[pair_k, w_k]), 2),
        'מט"ח (%)': round(float(mix_fx[pair_k, w_k]), 2),
        'לא סחיר (%)': round(float(mix_illiquid[pair_k, w_k]), 2),
        'שארפ (משוקלל)': round(float(mix_sharpe[pair_k, w_k]), 3),
        "why_primary": why_primary,
        "fund_a": funds[i], "fund_b": funds[j],
        "manager_a": mgr_arr[i], "manager_b": mgr_arr[j],
    }
    return out

if pick1 is None:
    st.error("לא נמצא אף שילוב שעומד במגבלה של הלא-סחיר. נסה להגדיל את מגבלת הלא-סחיר או לבחור מסלולים אחרים.")
    st.stop()

opt1 = row_from_pick(pick1, "1", f"החלופה המדורגת ראשונה לפי הקריטריון הראשי: {objective}.")

# Build a candidate list for picking option 2 (secondary: best alternative by a different criterion)
# We'll score all feasible combos with a composite ranking: prefer low dist, but also keep within tolerance.
within2 = (np.abs(mix_foreign - target_foreign) <= tolerance) & (np.abs(mix_stocks - target_stocks) <= tolerance)
score2 = dist.copy()
score2 = np.where(ok, score2, np.inf)
# create dataframe of best weight per pair
best_w_per_pair = np.argmin(score2, axis=1)
best_score = score2[np.arange(pairs), best_w_per_pair]
mask_pairs = np.isfinite(best_score)
cand = []
for pk in np.where(mask_pairs)[0]:
    wk = int(best_w_per_pair[pk])
    i = idx_i[pk]; j = idx_j[pk]
    wA = float(w_vals[wk])
    wB = 1.0 - wA
    # service weighted
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
        "fund_a": funds[i], "fund_b": funds[j],
        "manager_a": mgr_arr[i], "manager_b": mgr_arr[j],
    })

df_cand = pd.DataFrame(cand)
if df_cand.empty:
    st.error("לא נותרו מועמדים אחרי סינון. נסה להרחיב טולרנס או מגבלה.")
    st.stop()

# Option 2: pick best by Sharpe within tolerance (else best by distance but different from opt1)
df2 = df_cand.copy()
df2["within"] = (np.abs(df2["foreign"] - target_foreign) <= tolerance) & (np.abs(df2["stocks"] - target_stocks) <= tolerance)
df2 = df2.sort_values(["within","sharpe","dist"], ascending=[False, False, True]).reset_index(drop=True)

# Exclude the exact same pair as option 1
pair1 = tuple(sorted([opt1["fund_a"], opt1["fund_b"]]))
df2 = df2[df2.apply(lambda r: tuple(sorted([r["fund_a"], r["fund_b"]])) != pair1, axis=1)]
pick2 = None
if not df2.empty:
    r2 = df2.iloc[0]
    pick2 = (int(r2["pair_k"]), int(r2["w_k"]))
opt2 = row_from_pick(pick2, "2", f"חלופה שנייה: מעדיפה שארפ גבוה יותר (בתוך הטולרנס) תוך שמירה על עמידה במגבלות.") if pick2 else None

# Option 3: maximize service (within tolerance), excluding option 1 pair (and option 2 pair if exists)
df3 = df_cand.copy()
df3["within"] = (np.abs(df3["foreign"] - target_foreign) <= tolerance) & (np.abs(df3["stocks"] - target_stocks) <= tolerance)
df3 = df3[df3["within"]].copy()
if opt2 is not None:
    pair2 = tuple(sorted([opt2["fund_a"], opt2["fund_b"]]))
else:
    pair2 = None
df3 = df3[df3.apply(lambda r: tuple(sorted([r["fund_a"], r["fund_b"]])) not in {pair1, pair2}, axis=1)]
df3 = df3.sort_values(["service","dist"], ascending=[False, True]).reset_index(drop=True)
pick3 = None
if not df3.empty:
    r3 = df3.iloc[0]
    pick3 = (int(r3["pair_k"]), int(r3["w_k"]))
opt3 = row_from_pick(pick3, "3", "חלופה שלישית: מעדיפה ציון שירות משוקלל גבוה יותר (כפי שהוגדר בסרגל הצד).") if pick3 else None

# Build results table
opts = [opt1] + ([opt2] if opt2 is not None else []) + ([opt3] if opt3 is not None else [])
res = pd.DataFrame(opts)

# Add explanation column
res["יתרון מרכזי"] = res.apply(explain_row, axis=1)

# Present
st.subheader("3 חלופות מובילות (מדורג מהגבוה לנמוך)")
rtl_table(res.drop(columns=["why_primary","fund_a","fund_b","manager_a","manager_b"]))

st.markdown('<div class="small rtl">טיפ: אם אתה מחפש “לגרד” את תקרת הלא-סחיר, בחר בקריטריון הראשי המתאים והגדל טולרנס מעט.</div>', unsafe_allow_html=True)

# Optional: show universe table
with st.expander("להציג את כל הגופים והפרמטרים שנקלטו (לבדיקה)"):
    show = universe[["track","fund","stocks","foreign","israel","israel_calc","fx","illiquid","sharpe"]].copy()
    show = show.sort_values(["track","fund"])
    rtl_table(show)
