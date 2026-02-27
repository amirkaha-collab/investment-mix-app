import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from datetime import datetime

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="התאמת מסלולי השקעה – קרנות השתלמות",
    layout="wide",
)

# -----------------------------
# RTL + UI CSS
# -----------------------------
st.markdown(
    """
    <style>
      html, body, [class*="css"]  { direction: rtl; }
      .block-container { padding-top: 2rem; max-width: 1250px; }
      .title-wrap{
        background: linear-gradient(135deg, rgba(70,120,255,.10), rgba(120,70,255,.08));
        border: 1px solid rgba(0,0,0,.06);
        padding: 22px 22px;
        border-radius: 18px;
        margin-bottom: 18px;
      }
      .subtitle{ color: rgba(0,0,0,.65); margin-top: 6px; }
      .card{
        border: 1px solid rgba(0,0,0,.08);
        background: rgba(255,255,255,.78);
        border-radius: 16px;
        padding: 14px 14px;
        box-shadow: 0 8px 20px rgba(0,0,0,.04);
      }
      .card h3{ margin: 0 0 10px 0; font-size: 1.05rem; }
      .pill{
        display:inline-block;
        padding: 4px 10px;
        border-radius: 999px;
        border: 1px solid rgba(0,0,0,.08);
        background: rgba(0,0,0,.03);
        font-size: .85rem;
        margin-left: 6px;
        margin-top: 4px;
      }
      .muted{ color: rgba(0,0,0,.60); font-size: .9rem; }
      .kpi{
        display:flex;
        justify-content: space-between;
        border-top: 1px dashed rgba(0,0,0,.12);
        padding-top: 8px;
        margin-top: 8px;
      }
      .kpi .lab{ color: rgba(0,0,0,.65); }
      .kpi .val{ font-weight: 700; }
      .small{ font-size: .85rem; color: rgba(0,0,0,.60); }
      .section{
        border: 1px solid rgba(0,0,0,.06);
        background: rgba(255,255,255,.65);
        border-radius: 16px;
        padding: 14px 14px;
        margin-top: 12px;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------
# Helpers
# -----------------------------
def pct(x):
    if pd.isna(x):
        return "—"
    return f"{float(x):.2f}%"

def make_radar(title, target_dict, row_dict, labels_order):
    t_vals = [target_dict[k] for k in labels_order] + [target_dict[labels_order[0]]]
    r_vals = [row_dict[k] for k in labels_order] + [row_dict[labels_order[0]]]
    theta = labels_order + [labels_order[0]]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=t_vals, theta=theta, fill='toself', name='יעד'))
    fig.add_trace(go.Scatterpolar(r=r_vals, theta=theta, fill='toself', name='מסלול'))
    fig.update_layout(
        title=title,
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        showlegend=True,
        margin=dict(l=30, r=30, t=50, b=20),
    )
    return fig

def ensure_state():
    if "scenarios" not in st.session_state:
        st.session_state["scenarios"] = {}  # name -> dict
    if "active_scenario" not in st.session_state:
        st.session_state["active_scenario"] = None

def scenario_payload(target_equity, target_alt, target_fx, w_equity, w_alt, w_fx, w_sharpe):
    return {
        "created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "target": {"מניות": target_equity, "אלטרנטיבי": target_alt, "מט\"ח": target_fx},
        "weights": {
            "w_equity": w_equity,
            "w_alt": w_alt,
            "w_fx": w_fx,
            "w_sharpe": w_sharpe,
        },
    }

# -----------------------------
# Header
# -----------------------------
st.markdown(
    """
    <div class="title-wrap">
      <h1 style="margin:0;">כלי התאמת מסלולי השקעה בקרנות השתלמות</h1>
      <div class="subtitle">השוואת מסלולים לפי חשיפות, בחירת הקרובים ביותר ליעד, פילטרים ושמירת תרחישים.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# -----------------------------
# Load data
# -----------------------------
@st.cache_data
def load_data(path="data.xlsx"):
    df = pd.read_excel(path)
    return df

df = load_data("data.xlsx")

# Expect structure:
# Col A = "פרמטר"
# Col B.. = bodies
if df.shape[1] < 2 or df.shape[0] < 2:
    st.error("הקובץ data.xlsx לא נראה תקין (צריך עמודת 'פרמטר' ועוד עמודות גופים).")
    st.stop()

param_col = df.columns[0]
data = df.iloc[:, 1:].copy()
data.columns = df.columns[1:]

# Based on your screenshot order (rows):
needed = {
    "מניות": 0,
    "אלטרנטיבי": 1,
    "שארפ": 2,
    "נכסים סחירים": 3,
    "נכסים בארץ": 4,
    'מט"ח': 5,
}
missing_rows = [k for k, idx in needed.items() if idx >= len(df)]
if missing_rows:
    st.error(f"חסרות שורות בקובץ עבור: {', '.join(missing_rows)}")
    st.stop()

rows = []
for body in data.columns:
    rows.append({
        "גוף": str(body),
        "מניות": float(data[body].iloc[needed["מניות"]]),
        "אלטרנטיבי": float(data[body].iloc[needed["אלטרנטיבי"]]),
        "שארפ": float(data[body].iloc[needed["שארפ"]]),
        "נכסים סחירים": float(data[body].iloc[needed["נכסים סחירים"]]),
        "נכסים בארץ": float(data[body].iloc[needed["נכסים בארץ"]]),
        'מט"ח': float(data[body].iloc[needed['מט"ח']]),
    })

df2_base = pd.DataFrame(rows)

# -----------------------------
# Sidebar: targets, weights, filters, scenarios
# -----------------------------
ensure_state()
st.sidebar.header("הגדרות")

with st.sidebar.expander("יעדי חשיפה (%)", expanded=True):
    target_equity = st.slider("מניות", 0, 100, 50, 1)
    target_alt = st.slider("אלטרנטיבי", 0, 100, 30, 1)
    target_fx = st.slider('מט"ח', 0, 100, 20, 1)

with st.sidebar.expander("משקולות (חשיבות)", expanded=True):
    w_equity = st.slider("משקל מניות", 0.0, 5.0, 1.0, 0.1)
    w_alt = st.slider("משקל אלטרנטיבי", 0.0, 5.0, 1.0, 0.1)
    w_fx = st.slider('משקל מט"ח', 0.0, 5.0, 1.0, 0.1)
    w_sharpe = st.slider("משקל שארפ (עדיף גבוה)", 0.0, 5.0, 0.5, 0.1)
    st.caption("שארפ עובד כ״עדיף גבוה״: ככל שהוא גבוה יותר, הציון משתפר.")

with st.sidebar.expander("פילטרים", expanded=False):
    search = st.text_input("חיפוש גוף (חלק מהשם)", value="")
    min_sharpe = st.slider("שארפ מינימלי", 0.0, 3.0, 0.0, 0.01)
    min_liquid = st.slider("נכסים סחירים מינימום (%)", 0.0, 100.0, 0.0, 0.5)
    min_israel = st.slider("נכסים בארץ מינימום (%)", 0.0, 100.0, 0.0, 0.5)

with st.sidebar.expander("תרחישים: שמירה והשוואה", expanded=False):
    # Save
    name = st.text_input("שם תרחיש לשמירה", value="")
    colA, colB = st.columns(2)
    with colA:
        if st.button("שמור תרחיש", use_container_width=True):
            if not name.strip():
                st.warning("תן שם לתרחיש.")
            else:
                st.session_state["scenarios"][name.strip()] = scenario_payload(
                    target_equity, target_alt, target_fx, w_equity, w_alt, w_fx, w_sharpe
                )
                st.session_state["active_scenario"] = name.strip()
                st.success("נשמר.")

    # Load/Delete
    scenarios = list(st.session_state["scenarios"].keys())
    if scenarios:
        chosen = st.selectbox("תרחיש פעיל", options=scenarios, index=scenarios.index(st.session_state["active_scenario"]) if st.session_state["active_scenario"] in scenarios else 0)
        st.session_state["active_scenario"] = chosen

        payload = st.session_state["scenarios"].get(chosen)
        if payload:
            st.caption(f"נוצר: {payload.get('created','')}")
            c1, c2 = st.columns(2)
            with c1:
                if st.button("טען תרחיש לערכים", use_container_width=True):
                    t = payload["target"]
                    w = payload["weights"]
                    # Set sliders by rerun with session state keys? Streamlit sliders don't accept direct set unless key provided.
                    # So we show a message with the values to copy quickly:
                    st.info(
                        f"ערכי תרחיש:\n"
                        f"מניות={t['מניות']}, אלטרנטיבי={t['אלטרנטיבי']}, מט\"ח={t['מט\"ח']}\n"
                        f"משקולות: מניות={w['w_equity']}, אלטרנטיבי={w['w_alt']}, מט\"ח={w['w_fx']}, שארפ={w['w_sharpe']}"
                    )
            with c2:
                if st.button("מחק תרחיש", use_container_width=True):
                    st.session_state["scenarios"].pop(chosen, None)
                    st.session_state["active_scenario"] = None
                    st.success("נמחק.")
    else:
        st.caption("עדיין אין תרחישים שמורים.")

target = {"מניות": target_equity, "אלטרנטיבי": target_alt, 'מט"ח': target_fx}

# -----------------------------
# Apply filters
# -----------------------------
df2 = df2_base.copy()
if search.strip():
    df2 = df2[df2["גוף"].str.contains(search.strip(), case=False, na=False)]
df2 = df2[df2["שארפ"] >= min_sharpe]
df2 = df2[df2["נכסים סחירים"] >= min_liquid]
df2 = df2[df2["נכסים בארץ"] >= min_israel]

if df2.empty:
    st.warning("אין תוצאות אחרי הפילטרים. נסה להקל על תנאי הסינון.")
    st.stop()

# -----------------------------
# Scoring
# -----------------------------
df2["מרחק מניות"] = (df2["מניות"] - target_equity).abs()
df2["מרחק אלטרנטיבי"] = (df2["אלטרנטיבי"] - target_alt).abs()
df2['מרחק מט"ח'] = (df2['מט"ח'] - target_fx).abs()

sh_ref = float(df2["שארפ"].max()) if len(df2) else 1.0
df2["קנס שארפ"] = (sh_ref - df2["שארפ"]).clip(lower=0)

df2["ציון"] = (
    w_equity * df2["מרחק מניות"]
    + w_alt * df2["מרחק אלטרנטיבי"]
    + w_fx * df2['מרחק מט"ח']
    + w_sharpe * df2["קנס שארפ"] * 10.0
)

best = df2.sort_values("ציון").head(3).reset_index(drop=True)

# -----------------------------
# Top 3 cards
# -----------------------------
st.subheader("3 המסלולים הקרובים ביותר")

c1, c2, c3 = st.columns(3)

def render_card(col, r, idx):
    with col:
        st.markdown(
            f"""
            <div class="card">
              <h3>#{idx+1} — {r['גוף']}</h3>
              <div class="muted">
                <span class="pill">ציון: {r['ציון']:.2f}</span>
                <span class="pill">שארפ: {r['שארפ']:.2f}</span>
              </div>

              <div class="kpi"><div class="lab">מניות</div><div class="val">{pct(r['מניות'])}</div></div>
              <div class="kpi"><div class="lab">אלטרנטיבי</div><div class="val">{pct(r['אלטרנטיבי'])}</div></div>
              <div class="kpi"><div class="lab">מט"ח</div><div class="val">{pct(r['מט"ח'])}</div></div>

              <div class="small" style="margin-top:10px;">
                סטיות מהיעד: מניות {r['מרחק מניות']:.1f} | אלטרנטיבי {r['מרחק אלטרנטיבי']:.1f} | מט"ח {r['מרחק מט\"ח']:.1f}
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

if len(best) > 0: render_card(c1, best.iloc[0], 0)
if len(best) > 1: render_card(c2, best.iloc[1], 1)
if len(best) > 2: render_card(c3, best.iloc[2], 2)

st.divider()

# -----------------------------
# Full table
# -----------------------------
st.subheader("טבלת כל המסלולים (ממוינת לפי התאמה)")

show_cols = ["גוף", "ציון", "מניות", "אלטרנטיבי", 'מט"ח', "שארפ", "נכסים סחירים", "נכסים בארץ"]
tbl = df2.sort_values("ציון")[show_cols].copy()

tbl_disp = tbl.copy()
for col in ["מניות", "אלטרנטיבי", 'מט"ח', "נכסים סחירים", "נכסים בארץ"]:
    tbl_disp[col] = tbl_disp[col].map(lambda x: f"{x:.2f}%")
tbl_disp["ציון"] = tbl_disp["ציון"].map(lambda x: f"{x:.2f}")
tbl_disp["שארפ"] = tbl_disp["שארפ"].map(lambda x: f"{x:.2f}")

st.dataframe(tbl_disp, use_container_width=True)

st.divider()

# -----------------------------
# Radar charts
# -----------------------------
st.subheader("גרף השוואה: יעד מול המסלולים שנבחרו")
labels_order = ["מניות", "אלטרנטיבי", 'מט"ח']
rad_cols = st.columns(3)
for i in range(min(3, len(best))):
    r = best.iloc[i].to_dict()
    row_dict = {k: float(r[k]) for k in labels_order}
    fig = make_radar(f"{r['גוף']} — יעד מול מסלול", target, row_dict, labels_order)
    rad_cols[i].plotly_chart(fig, use_container_width=True)

st.divider()

# -----------------------------
# Scenario comparison table (Top3 per scenario)
# -----------------------------
st.subheader("השוואת תרחישים (Top 3 לכל תרחיש)")
scenarios = st.session_state["scenarios"]
if not scenarios:
    st.caption("אין עדיין תרחישים שמורים. שמור תרחיש בצד כדי להשוות.")
else:
    comp_rows = []
    for scen_name, payload in scenarios.items():
        t = payload["target"]
        w = payload["weights"]

        tmp = df2_base.copy()
        # same filters for fairness? we'll apply current filters too (search/min thresholds)
        if search.strip():
            tmp = tmp[tmp["גוף"].str.contains(search.strip(), case=False, na=False)]
        tmp = tmp[tmp["שארפ"] >= min_sharpe]
        tmp = tmp[tmp["נכסים סחירים"] >= min_liquid]
        tmp = tmp[tmp["נכסים בארץ"] >= min_israel]
        if tmp.empty:
            comp_rows.append({"תרחיש": scen_name, "תוצאה": "אין תוצאות (פילטרים)", "מקום 1": "", "מקום 2": "", "מקום 3": ""})
            continue

        tmp["מרחק מניות"] = (tmp["מניות"] - t["מניות"]).abs()
        tmp["מרחק אלטרנטיבי"] = (tmp["אלטרנטיבי"] - t["אלטרנטיבי"]).abs()
        tmp['מרחק מט"ח'] = (tmp['מט"ח'] - t['מט"ח']).abs()
        sh_ref2 = float(tmp["שארפ"].max()) if len(tmp) else 1.0
        tmp["קנס שארפ"] = (sh_ref2 - tmp["שארפ"]).clip(lower=0)

        tmp["ציון"] = (
            w["w_equity"] * tmp["מרחק מניות"]
            + w["w_alt"] * tmp["מרחק אלטרנטיבי"]
            + w["w_fx"] * tmp['מרחק מט"ח']
            + w["w_sharpe"] * tmp["קנס שארפ"] * 10.0
        )

        top3 = tmp.sort_values("ציון").head(3)["גוף"].tolist()
        while len(top3) < 3:
            top3.append("")

        comp_rows.append({
            "תרחיש": scen_name,
            "יעד מניות": t["מניות"],
            "יעד אלטרנטיבי": t["אלטרנטיבי"],
            'יעד מט"ח': t['מט"ח'],
            "משקל מניות": w["w_equity"],
            "משקל אלטרנטיבי": w["w_alt"],
            'משקל מט"ח': w["w_fx"],
            "משקל שארפ": w["w_sharpe"],
            "מקום 1": top3[0],
            "מקום 2": top3[1],
            "מקום 3": top3[2],
        })

    comp = pd.DataFrame(comp_rows)
    st.dataframe(comp, use_container_width=True)

st.caption("אם תרצה: אני יכול להפוך את 'טען תרחיש' לכפתור שממש מעדכן את הסליידרים אוטומטית (עם keys) — תגיד לי וזהו.")
