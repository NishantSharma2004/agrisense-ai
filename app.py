"""
AgriSense AI — Industry-Level Smart Agriculture Platform
Modules: Crop Recommender | Disease Risk Predictor + Real FAISS RAG
Features: Fully Responsive | PDF/CSV Download | Groq/Gemini Assistant | Clean UX
"""
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import io
import os
from datetime import datetime
import warnings
try:
    import faiss as _faiss_lib
    FAISS_OK = True
except ImportError:
    FAISS_OK = False

try:
    from google import genai
    GEMINI_OK = True
except ImportError:
    genai = None
    GEMINI_OK = False

try:
    from groq import Groq
    GROQ_OK = True
except ImportError:
    Groq = None
    GROQ_OK = False

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="AgriSense AI",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

for k, v in {"page":"home","cr":None,"dr":None,"assistant_msgs":[]}.items():
    if k not in st.session_state:
        st.session_state[k] = v

@st.cache_resource(show_spinner="🌾 Loading AgriSense AI models...")
def load_all():
    with open("models/agrisense_model.pkl","rb") as f:    DM = pickle.load(f)
    with open("models/crop_recommendation_model.pkl","rb") as f: CM = pickle.load(f)
    with open("models/rag_index.pkl","rb") as f:          RAG= pickle.load(f)
    return DM, CM, RAG

DM, CM, RAG = load_all()

# ── CSS ───────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
*,html,body,[class*="css"]{font-family:'Inter',sans-serif;box-sizing:border-box;}
#MainMenu,footer,header{visibility:hidden;}

/* Background */
.stApp{background:#0b1120;}

/* Sidebar */
section[data-testid="stSidebar"]{
  background:linear-gradient(180deg,#0d1f0f,#0a1a10) !important;
  border-right:1px solid rgba(46,204,113,.15);}
section[data-testid="stSidebar"] *{color:rgba(255,255,255,.88) !important;}

/* Inputs */
.stSelectbox>div>div{background:rgba(255,255,255,.06) !important;
  border:1px solid rgba(255,255,255,.13) !important;color:#fff !important;border-radius:8px !important;}
.stSlider [data-testid="stSliderThumb"]{background:#2ecc71 !important;}

/* Button */
.stButton>button{
  background:linear-gradient(135deg,#1a6e38,#2ecc71) !important;
  color:#fff !important;border:none !important;border-radius:10px !important;
  font-weight:600 !important;font-size:.93rem !important;
  padding:.55rem 1.2rem !important;
  box-shadow:0 4px 16px rgba(46,204,113,.28) !important;
  transition:all .18s ease !important;width:100%;}
.stButton>button:hover{transform:translateY(-2px) !important;
  box-shadow:0 6px 22px rgba(46,204,113,.42) !important;}

/* Download button — distinct style */
.stDownloadButton>button{
  background:rgba(46,204,113,.12) !important;
  border:1.5px solid rgba(46,204,113,.4) !important;
  color:#2ecc71 !important;border-radius:10px !important;
  font-weight:600 !important;font-size:.88rem !important;
  padding:.45rem 1rem !important;width:100%;
  transition:all .18s ease !important;}
.stDownloadButton>button:hover{
  background:rgba(46,204,113,.22) !important;
  border-color:rgba(46,204,113,.7) !important;}

/* Expander */
.streamlit-expanderHeader{background:rgba(255,255,255,.04) !important;
  border:1px solid rgba(255,255,255,.09) !important;border-radius:10px !important;
  color:#fff !important;font-weight:500 !important;}
.streamlit-expanderContent{background:rgba(255,255,255,.025) !important;
  border:1px solid rgba(255,255,255,.07) !important;border-radius:0 0 10px 10px !important;}

hr{border-color:rgba(255,255,255,.07) !important;margin:1.2rem 0 !important;}

/* ── HERO ── */
.hero{
  background:linear-gradient(135deg,#0d4f2b,#155e34,#0a3d22);
  border:1px solid rgba(46,204,113,.25);border-radius:18px;
  padding:2rem 2rem;text-align:center;margin-bottom:1.6rem;
  box-shadow:0 8px 40px rgba(0,0,0,.5);}
.hero h1{color:#fff;font-weight:800;margin:0;letter-spacing:-.02em;
  font-size:clamp(1.5rem,4vw,2.4rem);}
.hero p{color:rgba(255,255,255,.75);margin:.4rem 0 0;
  font-size:clamp(.82rem,2vw,1rem);}

/* ── MODULE CARDS (home) ── */
.mod-card{
  background:rgba(255,255,255,.04);border:1.5px solid rgba(255,255,255,.09);
  border-radius:16px;padding:1.8rem 1.6rem;transition:all .22s ease;
  height:100%;cursor:pointer;}
.mod-card:hover{background:rgba(46,204,113,.07);border-color:rgba(46,204,113,.4);
  transform:translateY(-3px);box-shadow:0 12px 36px rgba(46,204,113,.15);}
.mod-card .icon{font-size:2.6rem;margin-bottom:.7rem;display:block;}
.mod-card h2{color:#fff;font-size:1.25rem;font-weight:700;margin:0 0 .5rem;}
.mod-card p{color:rgba(255,255,255,.6);font-size:.88rem;margin:0;line-height:1.6;}
.mod-card .tag{display:inline-block;margin-top:.9rem;
  background:rgba(46,204,113,.15);border:1px solid rgba(46,204,113,.3);
  color:#2ecc71;border-radius:20px;padding:.18rem .7rem;font-size:.78rem;font-weight:600;}

/* ── SECTION HEADER ── */
.sh{color:#2ecc71;font-size:1.02rem;font-weight:700;margin:1.1rem 0 .65rem;}

/* ── GLASS CARD ── */
.gc{background:rgba(255,255,255,.04);border:1px solid rgba(255,255,255,.09);
  border-radius:14px;padding:1.2rem 1.3rem;margin-bottom:.7rem;}
.gc h4{color:#fff;font-size:.95rem;font-weight:600;margin:0 0 .4rem;}
.gc p{color:rgba(255,255,255,.62);font-size:.86rem;margin:0;line-height:1.55;}

/* ── RISK BADGES ── */
.risk-H{background:linear-gradient(135deg,#5c1010,#c0392b);
  border:1px solid rgba(231,76,60,.45);color:#fff;padding:1rem 1.2rem;
  border-radius:14px;text-align:center;font-weight:800;letter-spacing:.02em;
  box-shadow:0 6px 24px rgba(192,57,43,.35);
  font-size:clamp(1.1rem,2.5vw,1.5rem);}
.risk-M{background:linear-gradient(135deg,#5c3a00,#d68910);
  border:1px solid rgba(243,156,18,.45);color:#fff;padding:1rem 1.2rem;
  border-radius:14px;text-align:center;font-weight:800;letter-spacing:.02em;
  box-shadow:0 6px 24px rgba(214,137,16,.35);
  font-size:clamp(1.1rem,2.5vw,1.5rem);}
.risk-L{background:linear-gradient(135deg,#0a3520,#1e8449);
  border:1px solid rgba(46,204,113,.4);color:#fff;padding:1rem 1.2rem;
  border-radius:14px;text-align:center;font-weight:800;letter-spacing:.02em;
  box-shadow:0 6px 24px rgba(30,132,73,.3);
  font-size:clamp(1.1rem,2.5vw,1.5rem);}

/* ── CROP BOX ── */
.crop-box{background:linear-gradient(135deg,#0a3520,#0d5c2e,#155e34);
  border:1px solid rgba(46,204,113,.38);border-radius:16px;padding:1.6rem;
  text-align:center;box-shadow:0 8px 32px rgba(46,204,113,.18);margin-bottom:1rem;}
.crop-box .ce{font-size:3.2rem;display:block;margin-bottom:.3rem;}
.crop-box .cn{color:#2ecc71;font-weight:800;font-size:clamp(1.4rem,3vw,2rem);}
.crop-box .cs{color:rgba(255,255,255,.65);font-size:.88rem;margin-top:.25rem;}

/* ── ADVISORY ── */
.adv{background:rgba(46,204,113,.06);border-left:3px solid #2ecc71;
  border-radius:0 10px 10px 0;padding:.6rem .95rem;
  color:rgba(255,255,255,.85);font-size:.87rem;margin-bottom:.42rem;line-height:1.5;}

/* ── RAG source ── */
.rag-src{background:rgba(155,89,182,.09);border-left:3px solid #9b59b6;
  border-radius:0 8px 8px 0;padding:.42rem .82rem;margin-top:.45rem;
  color:rgba(200,180,220,.85);font-size:.78rem;}
.rag-score{display:inline-block;background:rgba(46,204,113,.12);
  border:1px solid rgba(46,204,113,.25);color:#2ecc71;border-radius:6px;
  padding:.08rem .42rem;font-size:.74rem;font-weight:600;margin-left:.35rem;}

/* ── PROB BAR ── */
.pb{display:flex;align-items:center;gap:.55rem;margin:.28rem 0;}
.pb-l{color:rgba(255,255,255,.68);font-size:.85rem;width:68px;flex-shrink:0;}
.pb-bg{flex:1;height:8px;background:rgba(255,255,255,.08);border-radius:4px;}
.pb-f{height:8px;border-radius:4px;}
.pb-v{color:#fff;font-size:.83rem;font-weight:700;width:42px;text-align:right;}

/* ── CROP ROW (top-5) ── */
.cr-row{background:rgba(255,255,255,.03);border:1px solid rgba(255,255,255,.08);
  border-radius:11px;padding:.62rem .95rem;margin-bottom:.4rem;
  display:flex;align-items:center;gap:.75rem;transition:border-color .15s;}
.cr-row:hover{border-color:rgba(46,204,113,.3);}
.cr-row.top{background:rgba(46,204,113,.1);border-color:rgba(46,204,113,.35);}

/* ── INFO CARD ── */
.info-card{background:rgba(46,204,113,.05);border:1px solid rgba(46,204,113,.2);
  border-radius:12px;padding:.95rem 1.15rem;margin-top:.9rem;
  color:rgba(255,255,255,.7);font-size:.87rem;line-height:1.6;}
.info-card b{color:#2ecc71;}

/* ── DOWNLOAD SECTION ── */
.dl-section{background:rgba(255,255,255,.03);border:1px solid rgba(255,255,255,.08);
  border-radius:14px;padding:1.1rem 1.3rem;margin-top:1rem;}
.dl-section .dl-title{color:#fff;font-size:.95rem;font-weight:600;margin-bottom:.7rem;}

/* ── STAT STRIP ── */
.stat-strip{display:flex;gap:.8rem;margin-bottom:1.3rem;flex-wrap:wrap;}
.stat{flex:1;min-width:100px;background:rgba(255,255,255,.04);
  border:1px solid rgba(255,255,255,.09);border-radius:12px;
  padding:.8rem .9rem;text-align:center;}
.stat .sv{color:#2ecc71;font-size:1.3rem;font-weight:800;}
.stat .sl{color:rgba(255,255,255,.5);font-size:.75rem;margin-top:.12rem;}

/* ── FEATURE GRID ── */
.feat-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:.9rem;margin-bottom:1.1rem;}

/* ── CROP BADGE ── */
.cbadge{display:inline-block;background:rgba(46,204,113,.09);
  border:1px solid rgba(46,204,113,.22);border-radius:20px;
  padding:.18rem .68rem;font-size:.79rem;color:#2ecc71;margin:.16rem;}

/* ── STEP CIRCLE ── */
.step{display:inline-flex;align-items:center;justify-content:center;
  width:24px;height:24px;border-radius:50%;
  background:rgba(46,204,113,.18);border:1.5px solid rgba(46,204,113,.4);
  color:#2ecc71;font-size:.8rem;font-weight:700;flex-shrink:0;margin-right:.45rem;}

/* Table */
table{color:rgba(255,255,255,.85) !important;}
th,td{border-color:rgba(255,255,255,.08) !important;}
th{background:rgba(46,204,113,.1) !important;}

/* ── RESPONSIVE ── */
/* Mobile: stack columns, shrink padding, smaller fonts */
@media(max-width:768px){
  .hero{padding:1.3rem 1rem;}
  .feat-grid{grid-template-columns:1fr !important;}
  .stat-strip{gap:.5rem;}
  .stat{min-width:80px;padding:.6rem .5rem;}
  .stat .sv{font-size:1.1rem;}
  .mod-card{padding:1.3rem 1.1rem;}
  .crop-box{padding:1.1rem;}
  .gc{padding:1rem 1.1rem;}
  .adv{font-size:.82rem;}
  .pb-l{width:58px;font-size:.8rem;}
  /* Force columns to stack on mobile */
  [data-testid="stHorizontalBlock"]{flex-direction:column !important;}
  [data-testid="stHorizontalBlock"]>[data-testid="stVerticalBlock"]{width:100% !important;}
}
@media(max-width:480px){
  .hero h1{font-size:1.3rem;}
  .hero p{font-size:.8rem;}
  .crop-box .cn{font-size:1.3rem;}
  .sh{font-size:.95rem;}
  .stat .sv{font-size:1rem;}
}
/* Tablet */
@media(min-width:769px) and (max-width:1024px){
  .feat-grid{grid-template-columns:repeat(2,1fr) !important;}
  .hero h1{font-size:1.9rem;}
}
</style>
""", unsafe_allow_html=True)

# ── DATA ─────────────────────────────────────────────────────
CROP_INFO = {
    'rice':        {'e':'🌾','season':'Kharif (Jun–Nov)',      'water':'High 1200–1800mm',     'soil':'Clay / Loamy',         'temp':'20–27°C','tip':'Needs standing water during early growth phase.'},
    'maize':       {'e':'🌽','season':'Kharif / Rabi',          'water':'Moderate 500–800mm',   'soil':'Well-drained Loam',    'temp':'18–27°C','tip':'Sensitive to waterlogging — ensure proper drainage.'},
    'chickpea':    {'e':'🫘','season':'Rabi (Oct–Mar)',         'water':'Low 400–500mm',        'soil':'Sandy Loam',           'temp':'15–25°C','tip':'Deep-rooted — excellent for dryland farming.'},
    'kidneybeans': {'e':'🫘','season':'Kharif',                 'water':'Moderate 600–1500mm',  'soil':'Sandy Loam',           'temp':'15–25°C','tip':'Fixes nitrogen — great for crop rotation.'},
    'pigeonpeas':  {'e':'🫘','season':'Kharif (Jun–Nov)',       'water':'Low 400–700mm',        'soil':'Sandy Loam',           'temp':'25–35°C','tip':'Drought-tolerant once established.'},
    'mothbeans':   {'e':'🫘','season':'Kharif',                 'water':'Very Low 200–400mm',   'soil':'Sandy',                'temp':'25–35°C','tip':'Excellent for arid and semi-arid regions.'},
    'mungbean':    {'e':'🫘','season':'Zaid / Kharif',          'water':'Low–Moderate',         'soil':'Sandy Loam',           'temp':'25–35°C','tip':'Short duration — 60–75 days to maturity.'},
    'blackgram':   {'e':'🫘','season':'Kharif / Rabi',          'water':'Moderate',             'soil':'Loam',                 'temp':'25–35°C','tip':'Better moisture stress tolerance than most pulses.'},
    'lentil':      {'e':'🫘','season':'Rabi (Oct–Mar)',         'water':'Low 250–400mm',        'soil':'Loam',                 'temp':'15–25°C','tip':'Good protein source for cool dry climates.'},
    'pomegranate': {'e':'🍎','season':'Perennial',              'water':'Low 500–800mm',        'soil':'Deep Sandy Loam',      'temp':'20–30°C','tip':'Drought-resistant — very high commercial value.'},
    'banana':      {'e':'🍌','season':'Year-round',             'water':'High 1200–2200mm',     'soil':'Rich Loam',            'temp':'25–32°C','tip':'Needs windbreaks and consistent soil moisture.'},
    'mango':       {'e':'🥭','season':'Perennial',              'water':'Low–Moderate',         'soil':'Alluvial / Sandy Loam','temp':'25–35°C','tip':'Dry period before flowering improves yield.'},
    'grapes':      {'e':'🍇','season':'Perennial',              'water':'Moderate 600–800mm',   'soil':'Sandy Loam',           'temp':'8–22°C', 'tip':'High commercial value — careful trellis management.'},
    'watermelon':  {'e':'🍉','season':'Zaid (Mar–Jun)',         'water':'Moderate',             'soil':'Sandy Loam',           'temp':'24–32°C','tip':'Needs warm dry climate during fruit development.'},
    'muskmelon':   {'e':'🍈','season':'Zaid (Mar–Jun)',         'water':'Moderate',             'soil':'Sandy Loam',           'temp':'26–32°C','tip':'Reduce irrigation during ripening.'},
    'apple':       {'e':'🍎','season':'Perennial (Temperate)',  'water':'Moderate 1000–1250mm', 'soil':'Well-drained Loam',    'temp':'0–15°C', 'tip':'Requires winter chilling hours for fruit set.'},
    'orange':      {'e':'🍊','season':'Perennial',              'water':'Moderate',             'soil':'Sandy Loam',           'temp':'8–20°C', 'tip':'Subtropical climate preferred — protect from frost.'},
    'papaya':      {'e':'🍈','season':'Year-round',             'water':'Moderate–High',        'soil':'Sandy Loam',           'temp':'30–40°C','tip':'Very sensitive to waterlogging — use raised beds.'},
    'coconut':     {'e':'🥥','season':'Perennial',              'water':'High 1500–2500mm',     'soil':'Sandy Loam',           'temp':'25–32°C','tip':'Coastal areas ideal — salt-tolerant crop.'},
    'cotton':      {'e':'🌿','season':'Kharif (Jun–Nov)',       'water':'Moderate 500–1000mm',  'soil':'Black / Clay Loam',    'temp':'23–32°C','tip':'Long warm growing season — high commercial value.'},
    'jute':        {'e':'🌿','season':'Kharif (Mar–Jun)',       'water':'High 1000–2000mm',     'soil':'Sandy Loam',           'temp':'22–30°C','tip':'Needs alluvial fertile soil and high humidity.'},
    'coffee':      {'e':'☕','season':'Perennial',              'water':'Moderate 1500–2500mm', 'soil':'Rich Loam',            'temp':'20–28°C','tip':'Shade-grown coffee produces premium quality beans.'},
}

ALL_DISEASE_CROPS = [
    "Wheat","Mustard","Bajra","Soybean","Chickpea","Maize","Rice","Cotton",
    "Kidneybeans","Pigeonpeas","Mothbeans","Mungbean","Blackgram","Lentil",
    "Pomegranate","Banana","Mango","Grapes","Watermelon","Muskmelon",
    "Apple","Orange","Papaya","Coconut","Jute","Coffee",
]

ADVISORIES = {
    "High":[
        "🚨 Inspect your field immediately for visible disease symptoms",
        "🌧️ Switch to drip/furrow irrigation — stop overhead watering now",
        "💊 Apply recommended fungicide within 48 hours",
        "📞 Contact your nearest agricultural extension office or KVK",
        "🚫 Do not harvest for at least 14 days after fungicide application",
        "📋 Maintain spray records — essential for crop insurance claims",
    ],
    "Medium":[
        "⚠️ Monitor your crop every 3 days for disease progression",
        "💧 Reduce irrigation frequency to lower field humidity",
        "🌿 Apply a preventive fungicide spray as precaution",
        "🔍 Scout lower leaves first — diseases begin from ground level",
        "📊 Ensure proper plant spacing for good air circulation",
    ],
    "Low":[
        "✅ Current conditions are safe — maintain routine monitoring",
        "📅 Continue your regular crop care and nutrition schedule",
        "🌱 Use balanced fertilization — avoid excess nitrogen",
        "💧 Maintain your optimal irrigation schedule",
        "📖 Keep field observation records for future reference",
    ],
}

# ── RAG ───────────────────────────────────────────────────────
def rag_search(crop, humidity, temp, rain, top_k=2):
    vectorizer = RAG["vectorizer"]
    index      = RAG["index"]
    docs       = RAG["docs"]
    conds = []
    if humidity > 80:    conds.append("very high humidity wet moist")
    elif humidity > 65:  conds.append("high humidity moist conditions")
    else:                conds.append("moderate humidity dry conditions")
    if temp < 15:        conds.append("cool cold low temperature")
    elif temp < 25:      conds.append("moderate temperature")
    else:                conds.append("warm hot high temperature")
    if rain > 20:        conds.append("heavy rain wet waterlogged")
    elif rain > 8:       conds.append("moderate rain wet conditions")
    else:                conds.append("low rainfall dry")
    query = f"{crop.lower()} {crop.lower()} crop disease leaves {' '.join(conds)}"
    q = vectorizer.transform([query]).toarray().astype(np.float32)
    if FAISS_OK:
        _faiss_lib.normalize_L2(q)
        scores, idxs = index.search(q, len(docs))
    else:
        from sklearn.metrics.pairwise import cosine_similarity
        vecs  = vectorizer.transform([d["text"] for d in docs]).toarray().astype(np.float32)
        sims  = cosine_similarity(q, vecs)[0]
        idxs  = [np.argsort(sims)[::-1]]
        scores= [sims[idxs[0]]]
    same  = [(docs[i], float(scores[0][j])) for j,i in enumerate(idxs[0]) if docs[i]["crop"]==crop]
    other = [(docs[i], float(scores[0][j])) for j,i in enumerate(idxs[0]) if docs[i]["crop"]!=crop]
    results = same[:top_k] if len(same)>=top_k else (same+other)[:top_k]
    return [{"disease":d["disease"],"desc":d["desc"],"tx":d["tx"],
             "prev":d["prev"],"src":d["src"],"score":round(s,3)} for d,s in results]

# ── ML predict ────────────────────────────────────────────────
def predict_disease(crop,temp,humid,rain,soil,wind,age,prev):
    X = pd.DataFrame([[temp,humid,rain,soil,wind,age,prev]], columns=DM["features"])
    risk = DM["encoder"].inverse_transform([DM["model"].predict(X)[0]])[0]
    prob = DM["model"].predict_proba(X)[0]
    labs = list(DM["encoder"].classes_)
    docs = rag_search(crop,humid,temp,rain)
    return dict(crop=crop,risk=risk,prob=prob,labs=labs,docs=docs,
                temp=temp,humid=humid,rain=rain,soil=soil,wind=wind,age=age,prev=prev)

def predict_crop(N,P,K,temp,humid,ph,rain):
    X = pd.DataFrame([[N,P,K,temp,humid,ph,rain]], columns=CM["features"])
    name = CM["encoder"].inverse_transform([CM["model"].predict(X)[0]])[0]
    prob = CM["model"].predict_proba(X)[0]
    cls  = list(CM["encoder"].classes_)
    return dict(name=name,prob=prob,cls=cls,N=N,P=P,K=K,temp=temp,humid=humid,ph=ph,rain=rain)

# ── Download helpers ──────────────────────────────────────────
def crop_report_csv(res):
    info  = CROP_INFO.get(res["name"],{})
    prob  = res["prob"]; cls = res["cls"]
    top5  = np.argsort(prob)[::-1][:5]
    rows  = []
    rows.append(["=== AgriSense AI — Crop Recommendation Report ===",""])
    rows.append(["Generated",datetime.now().strftime("%Y-%m-%d %H:%M")])
    rows.append(["",""])
    rows.append(["--- YOUR INPUTS ---",""])
    rows.append(["Nitrogen (N mg/kg)",   res["N"]])
    rows.append(["Phosphorus (P mg/kg)", res["P"]])
    rows.append(["Potassium (K mg/kg)",  res["K"]])
    rows.append(["Temperature (°C)",     res["temp"]])
    rows.append(["Humidity (%)",         res["humid"]])
    rows.append(["Soil pH",              res["ph"]])
    rows.append(["Rainfall (mm/yr)",     res["rain"]])
    rows.append(["",""])
    rows.append(["--- RECOMMENDATION ---",""])
    rows.append(["Recommended Crop",     res["name"].title()])
    rows.append(["Confidence",           f"{round(max(prob)*100,1)}%"])
    rows.append(["Season",               info.get("season","—")])
    rows.append(["Water Requirement",    info.get("water","—")])
    rows.append(["Soil Type",            info.get("soil","—")])
    rows.append(["Optimal Temperature",  info.get("temp","—")])
    rows.append(["Expert Tip",           info.get("tip","—")])
    rows.append(["",""])
    rows.append(["--- TOP 5 MATCHES ---",""])
    rows.append(["Rank","Crop","Confidence %"])
    for rank,idx in enumerate(top5):
        rows.append([f"#{rank+1}", cls[idx].title(), f"{round(prob[idx]*100,1)}%"])
    df = pd.DataFrame(rows)
    buf = io.StringIO()
    df.to_csv(buf, index=False, header=False)
    return buf.getvalue().encode()

def disease_report_csv(res):
    rows = []
    rows.append(["=== AgriSense AI — Disease Risk Report ===",""])
    rows.append(["Generated", datetime.now().strftime("%Y-%m-%d %H:%M")])
    rows.append(["",""])
    rows.append(["--- YOUR INPUTS ---",""])
    rows.append(["Crop",            res["crop"]])
    age_name = {15:"Seedling",35:"Vegetative",65:"Flowering",100:"Maturity"}.get(res["age"],"Flowering")
    rows.append(["Growth Stage",    age_name])
    rows.append(["Temperature (°C)",res["temp"]])
    rows.append(["Humidity (%)",    res["humid"]])
    rows.append(["Rainfall (mm)",   res["rain"]])
    rows.append(["Soil Moisture (%)",res["soil"]])
    rows.append(["Wind Speed km/h", res["wind"]])
    rows.append(["Prev. Disease",   "Yes" if res["prev"] else "No"])
    rows.append(["",""])
    rows.append(["--- PREDICTION ---",""])
    rows.append(["Disease Risk Level", res["risk"].upper()])
    for lab,p in zip(res["labs"],res["prob"]):
        rows.append([f"{lab} Risk Probability", f"{round(p*100,1)}%"])
    rows.append(["",""])
    rows.append(["--- RAG ADVISORY (FAISS Vector Search) ---",""])
    for i,doc in enumerate(res["docs"],1):
        rows.append([f"Disease #{i}", doc["disease"]])
        rows.append(["Description",  doc["desc"]])
        rows.append(["Treatment",    doc["tx"]])
        rows.append(["Prevention",   doc["prev"]])
        rows.append(["Source",       doc["src"]])
        rows.append(["Similarity Score", doc["score"]])
        rows.append(["",""])
    rows.append(["--- FIELD ADVISORY ---",""])
    for adv in ADVISORIES[res["risk"]]:
        rows.append([adv,""])
    df = pd.DataFrame(rows)
    buf = io.StringIO()
    df.to_csv(buf, index=False, header=False)
    return buf.getvalue().encode()


# ── AgriSense Assistant (Groq/Gemini + Context-Aware Fallback) ──────
def get_secret_or_env(name):
    """Read API key/model name from Streamlit Secrets or environment. Never hardcode keys."""
    try:
        value = st.secrets.get(name, None)
        if value:
            return str(value).strip()
    except Exception:
        pass
    value = os.getenv(name)
    return value.strip() if value else None

def get_groq_key():
    return get_secret_or_env("GROQ_API_KEY")

def get_gemini_key():
    return get_secret_or_env("GEMINI_API_KEY")

def get_groq_model():
    # Llama 3.1 8B is lightweight and suitable for fast project-demo explanations.
    return get_secret_or_env("GROQ_MODEL") or "llama-3.1-8b-instant"

def assistant_context_text():
    """Build context from current Crop/Disease results for grounded answers."""
    parts = []
    cr = st.session_state.get("cr")
    dr = st.session_state.get("dr")

    if cr:
        name = cr["name"]
        info = CROP_INFO.get(name, {})
        prob, cls = cr["prob"], cr["cls"]
        top5 = np.argsort(prob)[::-1][:5]
        top5_text = ", ".join([f"{cls[i].title()} ({round(prob[i]*100,1)}%)" for i in top5])
        parts.append(f"""
CURRENT CROP RECOMMENDATION RESULT:
Recommended Crop: {name.title()}
Confidence: {round(max(prob)*100,1)}%
Top 5 Matches: {top5_text}
Inputs: N={cr['N']}, P={cr['P']}, K={cr['K']}, pH={cr['ph']}, Temperature={cr['temp']}°C, Humidity={cr['humid']}%, Rainfall={cr['rain']}mm
Crop Profile: Season={info.get('season','N/A')}, Water Requirement={info.get('water','N/A')}, Soil={info.get('soil','N/A')}, Optimal Temp={info.get('temp','N/A')}
Expert Tip: {info.get('tip','N/A')}
""")
    else:
        parts.append("CURRENT CROP RECOMMENDATION RESULT: Not generated yet.")

    if dr:
        docs_text = []
        for doc in dr.get("docs", []):
            docs_text.append(
                f"Disease: {doc['disease']} | Description: {doc['desc']} | Treatment: {doc['tx']} | Prevention: {doc['prev']} | Source: {doc['src']} | Similarity: {doc['score']}"
            )
        prob_text = ", ".join([f"{lab}: {round(p*100,1)}%" for lab, p in zip(dr["labs"], dr["prob"])])
        age_name = {15:"Seedling",35:"Vegetative",65:"Flowering",100:"Maturity"}.get(dr["age"],"Flowering")
        field_adv = " | ".join(ADVISORIES.get(dr["risk"], []))
        parts.append(f"""
CURRENT DISEASE RISK RESULT:
Crop: {dr['crop']}
Growth Stage: {age_name}
Risk Level: {dr['risk']}
Risk Probabilities: {prob_text}
Inputs: Temperature={dr['temp']}°C, Humidity={dr['humid']}%, Rainfall={dr['rain']}mm, Soil Moisture={dr['soil']}%, Wind Speed={dr['wind']} km/h, Previous Disease={'Yes' if dr['prev'] else 'No'}
Retrieved FAISS/RAG Documents:
{chr(10).join(docs_text) if docs_text else 'No retrieved documents.'}
Field Advisory: {field_adv}
""")
    else:
        parts.append("CURRENT DISEASE RISK RESULT: Not generated yet.")

    return "\n".join(parts)

def local_assistant_answer(question):
    """Safe context-aware fallback when Gemini key/package is unavailable."""
    q = question.lower().strip()
    cr = st.session_state.get("cr")
    dr = st.session_state.get("dr")

    if not cr and not dr:
        return (
            "Please run **Crop Recommender** or **Disease Risk Predictor** first. "
            "After that, I can explain the recommendation, risk level, confidence score, RAG advisory, treatment, prevention, and next steps."
        )

    hindi = any(x in q for x in [
        "hindi", "hinglish", "samjhao", "bata", "batao", "kya", "kyu", "kaise",
        "karna", "kare", "chahiye", "ab", "farmer", "kisaan", "kisan"
    ])

    wants_crop = any(x in q for x in [
        "confidence", "score", "top", "recommend", "recommand", "recommended", "recommendation",
        "crop", "maize", "fasal", "why this crop", "kyu recommend"
    ])

    wants_disease = any(x in q for x in [
        "risk", "disease", "bimari", "बीमारी", "what should", "karu", "karna", "kare",
        "action", "plan", "next", "treatment", "prevention", "spray", "medicine", "advisory"
    ])

    wants_rag = any(x in q for x in ["source", "rag", "faiss", "similarity", "document", "docs"])
    wants_overall = any(x in q for x in ["overall", "summary", "explain overall", "project", "result explain"])

    def crop_answer():
        if not cr:
            return "Please run the Crop Recommender first so I can explain the crop result."
        name = cr["name"].title()
        conf = round(max(cr["prob"])*100, 1)
        info = CROP_INFO.get(cr["name"], {})
        prob, cls = cr["prob"], cr["cls"]
        top5 = np.argsort(prob)[::-1][:5]
        top5_text = ", ".join([f"{cls[i].title()} ({round(prob[i]*100,1)}%)" for i in top5])
        if hindi:
            return (
                f"**{name}** recommend hua kyunki tumhare soil inputs **N={cr['N']}, P={cr['P']}, K={cr['K']}, pH={cr['ph']}** "
                f"aur climate values **Temp={cr['temp']}°C, Humidity={cr['humid']}%, Rainfall={cr['rain']}mm** is crop ke liye best match hain.\n\n"
                f"- Confidence: **{conf}%**\n"
                f"- Top matches: {top5_text}\n"
                f"- Season: {info.get('season','N/A')}\n"
                f"- Water requirement: {info.get('water','N/A')}\n"
                f"- Expert tip: {info.get('tip','Regular monitoring rakho.')}"
            )
        return (
            f"**{name}** was recommended because your soil profile and climate values matched it best among 22 crop classes.\n\n"
            f"- Confidence: **{conf}%**\n"
            f"- Top matches: {top5_text}\n"
            f"- Season: {info.get('season','N/A')}\n"
            f"- Water requirement: {info.get('water','N/A')}\n"
            f"- Expert tip: {info.get('tip','Maintain regular monitoring and balanced inputs.')}"
        )

    def disease_answer():
        if not dr:
            return "Please run the Disease Risk Predictor first so I can explain disease risk and RAG advisory."
        docs = dr.get("docs", [])
        prob_text = ", ".join([f"{lab}: {round(p*100,1)}%" for lab, p in zip(dr["labs"], dr["prob"])])
        doc_lines = "\n".join([
            f"- **{d['disease']}**: {d['prev']} Source: {d['src']} (similarity {d['score']})" for d in docs
        ]) or "- No source documents retrieved."
        adv = "\n".join([f"- {a}" for a in ADVISORIES.get(dr["risk"], [])])
        if hindi:
            return (
                f"Aapki **{dr['crop']}** crop ka disease risk **{dr['risk']}** hai. Probability breakdown: **{prob_text}**.\n\n"
                f"**Ab farmer ko kya karna chahiye:**\n{adv}\n\n"
                f"**Source-backed advisory:**\n{doc_lines}\n\n"
                "Chemical treatment sirf symptoms visible hone par aur local KVK/agriculture expert ki guidance ke baad follow karein."
            )
        return (
            f"Your **{dr['crop']}** disease risk is **{dr['risk']}**. Probability breakdown: **{prob_text}**.\n\n"
            f"**Recommended action plan:**\n{adv}\n\n"
            f"**Source-backed advisory retrieved:**\n{doc_lines}\n\n"
            "Use chemical treatment only when symptoms are visible and preferably after consulting a local KVK/agriculture expert."
        )

    def rag_answer():
        if not dr:
            return "RAG/FAISS advisory is available after running the Disease Risk Predictor."
        docs = dr.get("docs", [])
        if not docs:
            return "No RAG documents were retrieved for the current result."
        return (
            "FAISS vector search retrieved these source-backed disease documents:\n"
            + "\n".join([f"- **{d['disease']}** from {d['src']} with similarity score {d['score']}" for d in docs])
            + "\n\nSimilarity score shows how closely the retrieved advisory matched your crop and condition-based query."
        )

    def overall_answer():
        out = []
        if cr:
            out.append(crop_answer())
        if dr:
            out.append(disease_answer())
        return "\n\n---\n\n".join(out)

    if wants_rag:
        return rag_answer()
    if wants_disease:
        return disease_answer()
    if wants_crop:
        return crop_answer()
    if wants_overall:
        return overall_answer()

    # If the user asks a vague follow-up, answer using available context instead of saying “run prediction first”.
    if dr:
        return disease_answer()
    if cr:
        return crop_answer()

    return "Run Crop Recommender or Disease Risk Predictor first for personalized guidance."

def llm_assistant_answer(question):
    """Generate grounded answer using Groq first, then Gemini, then clean local fallback."""
    context = assistant_context_text()
    system_prompt = """You are AgriSense Assistant, a context-aware agriculture decision-support chatbot.

Rules:
1. Use only the provided project context.
2. Do not invent pesticide names, chemical dosage, disease facts, or sources.
3. If exact information is not available in the context, say that the user should consult a local agriculture officer or KVK.
4. Explain in simple farmer-friendly language.
5. If the user asks in Hindi/Hinglish, answer in Hindi/Hinglish. Otherwise answer in clear English.
6. Make clear that this is a decision-support tool, not a replacement for agricultural experts.
7. Use short bullet points for action steps.
8. For treatment guidance, rely only on retrieved source-backed advisory.
"""
    user_prompt = f"""
Project context:
{context}

User question:
{question}
"""

    # 1) Try Groq first
    groq_key = get_groq_key()
    if GROQ_OK and groq_key:
        try:
            client = Groq(api_key=groq_key)
            completion = client.chat.completions.create(
                model=get_groq_model(),
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.2,
                max_tokens=600,
            )
            answer = completion.choices[0].message.content
            if answer:
                return answer.strip()
        except Exception:
            # Do not expose API errors/quota details to farmers or evaluators.
            pass

    # 2) Try Gemini as secondary fallback
    gemini_key = get_gemini_key()
    if GEMINI_OK and gemini_key:
        try:
            client = genai.Client(api_key=gemini_key)
            response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=system_prompt + "\n\n" + user_prompt,
            )
            if getattr(response, "text", None):
                return response.text.strip()
        except Exception:
            # Do not show RESOURCE_EXHAUSTED / 429 or other raw API errors in UI.
            pass

    # 3) Clean offline fallback from current project context
    return local_assistant_answer(question)

def context_badge_html():
    cr = st.session_state.get("cr")
    dr = st.session_state.get("dr")
    crop_text = "No crop result yet" if not cr else f"{cr['name'].title()} · {round(max(cr['prob'])*100,1)}% confidence"
    disease_text = "No disease result yet" if not dr else f"{dr['crop']} · {dr['risk']} risk"
    docs_text = "No RAG docs yet" if not dr else f"{len(dr.get('docs', []))} RAG docs retrieved"
    return f"""
    <div class="gc">
      <h4>🧠 Current Assistant Context</h4>
      <p>🌱 <b>Crop Result:</b> {crop_text}<br>
      🦠 <b>Disease Result:</b> {disease_text}<br>
      📚 <b>Knowledge Context:</b> {docs_text}</p>
    </div>
    """

# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:1rem 0 .7rem;'>
      <div style='font-size:2.2rem;'>🌾</div>
      <div style='color:#2ecc71;font-size:1.05rem;font-weight:700;'>AgriSense AI</div>
      <div style='color:rgba(255,255,255,.38);font-size:.72rem;margin-top:.1rem;'>Smart Agriculture Platform</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<hr/>", unsafe_allow_html=True)

    nav = st.radio("", ["🏠 Home","🌱 Crop Recommender","🦠 Disease Risk Predictor","🤖 AgriSense Assistant"],
                   label_visibility="collapsed",
                   index=["home","crop","disease","assistant"].index(st.session_state.page))
    new_page = {"🏠 Home":"home","🌱 Crop Recommender":"crop","🦠 Disease Risk Predictor":"disease","🤖 AgriSense Assistant":"assistant"}[nav]
    if new_page != st.session_state.page:
        st.session_state.page = new_page
        st.rerun()

    st.markdown("<hr/>", unsafe_allow_html=True)

    if st.session_state.page == "crop":
        st.markdown("**🔬 Soil Nutrients**")
        N  = st.slider("Nitrogen — N (mg/kg)",   0, 150,  60)
        P  = st.slider("Phosphorus — P (mg/kg)", 0, 150,  50)
        K  = st.slider("Potassium — K (mg/kg)",  0, 210,  40)
        st.markdown("**🌦️ Climate & Soil**")
        ct = st.slider("Temperature (°C)",   0,  48, 25)
        ch = st.slider("Humidity (%)",       10, 100, 70)
        ph = st.slider("Soil pH",           3.0, 10.0, 6.5, step=0.1)
        cr = st.slider("Rainfall (mm/yr)",  20,  300, 100)
        st.markdown("")
        if st.button("🌱 Recommend Best Crop", use_container_width=True):
            st.session_state.cr = predict_crop(N,P,K,ct,ch,ph,cr)
            st.rerun()
        if st.session_state.cr:
            if st.button("🔄 Reset", use_container_width=True, key="cr_rst"):
                st.session_state.cr = None
                st.rerun()

    elif st.session_state.page == "disease":
        st.markdown("**🌿 Crop**")
        sel_crop = st.selectbox("Select Crop", ALL_DISEASE_CROPS, label_visibility="collapsed")
        stage    = st.selectbox("Growth Stage",[
            "Seedling (0–20 days)","Vegetative (21–50 days)",
            "Flowering (51–80 days)","Maturity (81–120 days)"])
        age_map  = {"Seedling (0–20 days)":15,"Vegetative (21–50 days)":35,
                    "Flowering (51–80 days)":65,"Maturity (81–120 days)":100}
        prev = st.radio("Previous Disease?",["No","Yes"], horizontal=True)
        st.markdown("**🌡️ Weather & Soil**")
        dt = st.slider("Temperature (°C)",  0,  48, 24)
        dh = st.slider("Humidity (%)",     20, 100, 68)
        dr = st.slider("Rainfall (mm)",     0,  60, 12)
        ds = st.slider("Soil Moisture (%)", 10,  95, 52)
        dw = st.slider("Wind Speed (km/h)", 0,  35,  9)
        st.markdown("")
        if st.button("🔍 Predict Disease Risk", use_container_width=True):
            st.session_state.dr = predict_disease(
                sel_crop,dt,dh,dr,ds,dw,
                age_map[stage], 1 if prev=="Yes" else 0)
            st.rerun()
        if st.session_state.dr:
            if st.button("🔄 Reset", use_container_width=True, key="dr_rst"):
                st.session_state.dr = None
                st.rerun()

    st.markdown("<hr/>", unsafe_allow_html=True)
    st.markdown("""
    <div style='color:rgba(255,255,255,.28);font-size:.7rem;text-align:center;line-height:1.8;'>
      🎯 SDG 2: Zero Hunger<br>🌿 SDG 15: Life on Land
    </div>""", unsafe_allow_html=True)

# ── HERO ──────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>🌾 AgriSense AI</h1>
  <p>Smart Crop Recommendation · Disease Risk Prediction · RAG Advisory</p>
</div>
""", unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════
# HOME
# ═════════════════════════════════════════════════════════════
if st.session_state.page == "home":
    st.markdown("""
    <div class="stat-strip">
      <div class="stat"><div class="sv">22</div><div class="sl">Crops Supported</div></div>
      <div class="stat"><div class="sv">52</div><div class="sl">RAG Documents</div></div>
      <div class="stat"><div class="sv">89.6%</div><div class="sl">Disease Accuracy</div></div>
      <div class="stat"><div class="sv">100%</div><div class="sl">Crop Accuracy</div></div>
      <div class="stat"><div class="sv">FAISS</div><div class="sl">Vector RAG</div></div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown("""<div class="mod-card">
          <span class="icon">🌱</span>
          <h2>Crop Recommender</h2>
          <p>Enter soil test values (N, P, K, pH) and local climate data to find
          the single best crop to plant. Powered by Random Forest trained on 2,200 records.</p>
          <span class="tag">Random Forest · 100% Test Accuracy</span>
        </div>""", unsafe_allow_html=True)
        if st.button("Open Crop Recommender →", key="h_crop"):
            st.session_state.page = "crop"; st.rerun()
    with c2:
        st.markdown("""<div class="mod-card">
          <span class="icon">🦠</span>
          <h2>Disease Risk Predictor</h2>
          <p>Select your crop and enter weather conditions to predict disease risk.
          Real FAISS vector search retrieves precise treatment & prevention from curated docs.</p>
          <span class="tag">Random Forest + FAISS RAG · 89.6% Test Accuracy</span>
        </div>""", unsafe_allow_html=True)
        if st.button("Open Disease Predictor →", key="h_dis"):
            st.session_state.page = "disease"; st.rerun()

    st.markdown("<hr/>", unsafe_allow_html=True)
    st.markdown('<div class="sh">⚙️ How It Works</div>', unsafe_allow_html=True)
    st.markdown("""<div class="feat-grid">
      <div class="gc"><h4><span class="step">1</span>Enter Your Data</h4>
        <p>Soil test values (N, P, K, pH) for crop recommendation, or
        current weather & soil conditions for disease risk analysis.</p></div>
      <div class="gc"><h4><span class="step">2</span>ML Model Predicts</h4>
        <p>Random Forest analyses your inputs and returns the recommended
        crop or disease risk level with probability confidence scores.</p></div>
      <div class="gc"><h4><span class="step">3</span>RAG Retrieves Advisory</h4>
        <p>FAISS vector search retrieves the most relevant disease documents
        from a curated ICAR / FAO knowledge base of 52 documents.</p></div>
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="sh">🌾 Supported Crops</div>', unsafe_allow_html=True)
    st.markdown("".join(f'<span class="cbadge">{v["e"]} {k.title()}</span>' for k,v in CROP_INFO.items()),
                unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════
# CROP RECOMMENDER
# ═════════════════════════════════════════════════════════════
elif st.session_state.page == "crop":
    if not st.session_state.cr:
        st.markdown('<div class="sh">🌱 Crop Recommender</div>', unsafe_allow_html=True)
        st.info("👈 Enter soil test values and climate data in the sidebar → click **Recommend Best Crop**")
        st.markdown("""<div class="feat-grid">
          <div class="gc"><h4>🔬 Soil Nutrients</h4>
            <p>Get N, P, K values from your soil test report via Krishi Vigyan Kendra or Soil Health Card portal.</p></div>
          <div class="gc"><h4>🌡️ Climate Matching</h4>
            <p>Temperature, humidity, pH and rainfall matched against ideal growing conditions for 22 crop types.</p></div>
          <div class="gc"><h4>📊 Ranked Results</h4>
            <p>Full Top-5 list with confidence % for each crop so you can make an informed decision.</p></div>
        </div>""", unsafe_allow_html=True)
        st.markdown('<div class="sh">🌾 22 Supported Crops</div>', unsafe_allow_html=True)
        st.markdown("".join(f'<span class="cbadge">{v["e"]} {k.title()}</span>' for k,v in CROP_INFO.items()),
                    unsafe_allow_html=True)
    else:
        res   = st.session_state.cr
        name  = res["name"]
        info  = CROP_INFO.get(name,{"e":"🌿","season":"—","water":"—","soil":"—","temp":"—","tip":"—"})
        prob  = res["prob"]; cls = res["cls"]
        top5  = np.argsort(prob)[::-1][:5]

        col1, col2 = st.columns([1,1], gap="large")
        with col1:
            st.markdown('<div class="sh">🌱 Recommended Crop</div>', unsafe_allow_html=True)
            st.markdown(f"""<div class="crop-box">
              <span class="ce">{info['e']}</span>
              <div class="cn">{name.title()}</div>
              <div class="cs">Best match for your soil &amp; climate</div>
            </div>""", unsafe_allow_html=True)

            st.markdown('<div class="sh">📋 Crop Profile</div>', unsafe_allow_html=True)
            st.markdown(f"""
| Property | Details |
|---|---|
| **Season** | {info['season']} |
| **Water Requirement** | {info['water']} |
| **Soil Type** | {info['soil']} |
| **Optimal Temperature** | {info['temp']} |
| **Soil Inputs** | N:{res['N']} · P:{res['P']} · K:{res['K']} · pH:{res['ph']} |
| **Climate Inputs** | Temp:{res['temp']}°C · Humidity:{res['humid']}% · Rain:{res['rain']}mm |
""")
            st.markdown(f'<div class="adv">💡 <b>Expert Tip:</b> {info["tip"]}</div>', unsafe_allow_html=True)

            st.markdown(f"""<div class="info-card">
              <b>🦠 Want to check disease risk for {name.title()}?</b><br>
              Go to <b>Disease Risk Predictor</b> from the sidebar,
              select <b>{name.title()}</b> as your crop and enter current
              weather conditions to get a full disease analysis with RAG advisory.
            </div>""", unsafe_allow_html=True)

            # ── DOWNLOAD ────────────────────────────────────
            st.markdown('<div class="sh">⬇️ Download Your Report</div>', unsafe_allow_html=True)
            st.markdown('<div class="dl-section"><div class="dl-title">Save your crop recommendation as a CSV report</div>', unsafe_allow_html=True)
            fname = f"crop_recommendation_{name}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
            st.download_button(
                label="⬇️ Download Crop Report (CSV)",
                data=crop_report_csv(res),
                file_name=fname,
                mime="text/csv",
                key="dl_crop",
            )
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="sh">📊 Top 5 Crop Matches</div>', unsafe_allow_html=True)
            for rank, idx in enumerate(top5):
                cname = cls[idx]; ci = CROP_INFO.get(cname,{"e":"🌿"})
                pct   = round(prob[idx]*100,1)
                is_top = rank==0
                cls_n  = "cr-row top" if is_top else "cr-row"
                rc2    = "#2ecc71" if is_top else "rgba(255,255,255,.45)"
                bc     = "#2ecc71" if is_top else "#3498db"
                st.markdown(f"""
                <div class="{cls_n}">
                  <span style="color:{rc2};font-weight:700;font-size:.92rem;width:24px;">#{rank+1}</span>
                  <span style="font-size:1.25rem;">{ci.get('e','🌿')}</span>
                  <div style="flex:1;">
                    <div style="color:#fff;font-weight:600;font-size:.88rem;">{cname.title()}</div>
                    <div style="background:rgba(255,255,255,.08);border-radius:3px;height:5px;margin-top:4px;">
                      <div style="width:{pct}%;height:5px;border-radius:3px;background:{bc};"></div></div>
                  </div>
                  <span style="color:{rc2};font-weight:700;font-size:.88rem;min-width:42px;text-align:right;">{pct}%</span>
                </div>""", unsafe_allow_html=True)

            st.markdown('<div class="sh">🔍 Why This Crop?</div>', unsafe_allow_html=True)
            st.markdown(f"""<div class="gc"><p>
              Soil profile — <span style='color:#2ecc71;font-weight:600;'>
              N={res['N']}, P={res['P']}, K={res['K']}, pH={res['ph']}</span> —
              combined with climate —
              <span style='color:#3498db;font-weight:600;'>
              Temp={res['temp']}°C, Humidity={res['humid']}%, Rainfall={res['rain']}mm</span>
              — was matched against all 22 crop profiles.<br><br>
              <b style='color:#fff;'>{name.title()}</b> returned the highest similarity score
              ({round(max(prob)*100,1)}% confidence) across all crops.
            </p></div>""", unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════
# DISEASE PREDICTOR
# ═════════════════════════════════════════════════════════════
elif st.session_state.page == "disease":
    if not st.session_state.dr:
        st.markdown('<div class="sh">🦠 Disease Risk Predictor</div>', unsafe_allow_html=True)
        st.info("👈 Select your crop and enter weather & soil values in the sidebar → click **Predict Disease Risk**")
        st.markdown("""<div class="feat-grid">
          <div class="gc"><h4>🧠 ML Risk Prediction</h4>
            <p>Random Forest trained on 1,200 field records predicts Low / Medium / High
            disease risk based on 7 weather and soil features (89.6% test accuracy).</p></div>
          <div class="gc"><h4>📚 FAISS RAG Advisory</h4>
            <p>Real vector similarity search over 52 curated disease documents from
            ICAR, FAO and Agriculture Dept. Top-2 most relevant docs retrieved per query.</p></div>
          <div class="gc"><h4>🌍 26 Crops Covered</h4>
            <p>Grains, pulses, fruits and cash crops — each with two disease profiles,
            treatment protocols, prevention guidelines and cited sources.</p></div>
        </div>""", unsafe_allow_html=True)
        st.markdown('<div class="sh">🌾 Supported Crops (26)</div>', unsafe_allow_html=True)
        st.markdown("".join(
            f'<span class="cbadge">{CROP_INFO.get(c.lower(),{}).get("e","🌿")} {c}</span>'
            for c in ALL_DISEASE_CROPS), unsafe_allow_html=True)
    else:
        res  = st.session_state.dr
        risk = res["risk"]
        rc   = {"High":"risk-H","Medium":"risk-M","Low":"risk-L"}[risk]
        re   = {"High":"🚨","Medium":"⚠️","Low":"✅"}[risk]
        fc   = {"High":"#c0392b","Medium":"#d68910","Low":"#1e8449"}

        col1, col2 = st.columns([1,1], gap="large")
        with col1:
            st.markdown('<div class="sh">🎯 Prediction Result</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="{rc}">{re}&nbsp; {risk.upper()} DISEASE RISK</div>', unsafe_allow_html=True)

            st.markdown('<div class="sh">📊 Risk Probability</div>', unsafe_allow_html=True)
            for lab, p in zip(res["labs"], res["prob"]):
                pct = round(p*100,1); col = fc.get(lab,"#2ecc71")
                st.markdown(f"""<div class="pb">
                  <span class="pb-l">{lab}</span>
                  <div class="pb-bg"><div class="pb-f" style="width:{pct}%;background:{col};"></div></div>
                  <span class="pb-v">{pct}%</span>
                </div>""", unsafe_allow_html=True)

            age_name = {15:"Seedling",35:"Vegetative",65:"Flowering",100:"Maturity"}.get(res["age"],"Flowering")
            st.markdown('<div class="sh">📋 Input Summary</div>', unsafe_allow_html=True)
            st.markdown(f"""
| Field | Value |
|---|---|
| **Crop** | {res['crop']} |
| **Growth Stage** | {age_name} |
| **Temperature** | {res['temp']}°C |
| **Humidity** | {res['humid']}% |
| **Rainfall** | {res['rain']} mm |
| **Soil Moisture** | {res['soil']}% |
| **Wind Speed** | {res['wind']} km/h |
| **Prev. Disease** | {"Yes" if res['prev'] else "No"} |
""")
            # ── DOWNLOAD ────────────────────────────────────
            st.markdown('<div class="sh">⬇️ Download Your Report</div>', unsafe_allow_html=True)
            st.markdown('<div class="dl-section"><div class="dl-title">Save disease risk analysis as a CSV report</div>', unsafe_allow_html=True)
            fname = f"disease_risk_{res['crop']}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
            st.download_button(
                label="⬇️ Download Disease Report (CSV)",
                data=disease_report_csv(res),
                file_name=fname,
                mime="text/csv",
                key="dl_disease",
            )
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="sh">📚 RAG Disease Advisory</div>', unsafe_allow_html=True)
            st.caption(f"FAISS vector search · {len(res['docs'])} documents retrieved for **{res['crop']}**")
            if res["docs"]:
                for doc in res["docs"]:
                    with st.expander(f"🦠 {doc['disease']}", expanded=True):
                        st.markdown(f"<span style='color:rgba(255,255,255,.88);font-size:.88rem;'>📌 <b>Description:</b> {doc['desc']}</span>", unsafe_allow_html=True)
                        st.markdown(f"<span style='color:rgba(255,255,255,.88);font-size:.88rem;'>💊 <b>Treatment:</b> {doc['tx']}</span>", unsafe_allow_html=True)
                        st.markdown(f"<span style='color:rgba(255,255,255,.88);font-size:.88rem;'>🛡️ <b>Prevention:</b> {doc['prev']}</span>", unsafe_allow_html=True)
                        st.markdown(f'<div class="rag-src">📖 {doc["src"]}<span class="rag-score">similarity {doc["score"]:.3f}</span></div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="gc"><p>No disease documents matched — risk appears minimal for current conditions.</p></div>', unsafe_allow_html=True)

        st.markdown("<hr/>", unsafe_allow_html=True)
        st.markdown('<div class="sh">🌾 Field Advisory</div>', unsafe_allow_html=True)
        a1, a2 = st.columns(2)
        for i, adv in enumerate(ADVISORIES[risk]):
            with (a1 if i%2==0 else a2):
                st.markdown(f'<div class="adv">{adv}</div>', unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════
# AGRISENSE ASSISTANT
# ═════════════════════════════════════════════════════════════
elif st.session_state.page == "assistant":
    st.markdown('<div class="sh">🤖 AgriSense Assistant</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="gc">
      <h4>Ask follow-up questions about your crop recommendation, disease risk, RAG advisory, sources, similarity scores, treatment, prevention, or next steps.</h4>
      <p>This assistant uses your latest Crop Recommender and Disease Risk Predictor results as context. For best answers, run a prediction first.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(context_badge_html(), unsafe_allow_html=True)

    groq_key = get_groq_key()
    gemini_key = get_gemini_key()
    if groq_key and GROQ_OK:
        st.success("Groq AI provider is configured. Assistant will use Groq first, Gemini second, and offline fallback if needed.")
    elif groq_key and not GROQ_OK:
        st.warning("GROQ_API_KEY is configured, but groq package is missing. Add groq to requirements.txt and redeploy the app.")
    elif gemini_key and GEMINI_OK:
        st.info("Gemini key is configured. Assistant will use Gemini and offline fallback if quota is unavailable.")
    elif gemini_key and not GEMINI_OK:
        st.warning("GEMINI_API_KEY is configured, but google-genai package is missing. Add google-genai to requirements.txt and redeploy the app.")
    else:
        st.info("No AI API key configured yet. Assistant is running in offline project-context mode. Add GROQ_API_KEY or GEMINI_API_KEY in Streamlit Secrets to enable AI answers.")

    st.markdown('<div class="sh">✨ Suggested Questions</div>', unsafe_allow_html=True)
    q1, q2, q3, q4 = st.columns(4)
    suggested_question = None
    with q1:
        if st.button("🌱 Why this crop?", use_container_width=True):
            suggested_question = "Why was this crop recommended? Explain the confidence score."
    with q2:
        if st.button("🦠 Explain risk", use_container_width=True):
            suggested_question = "Explain my disease risk result and probability breakdown."
    with q3:
        if st.button("📚 Explain RAG", use_container_width=True):
            suggested_question = "Explain the RAG advisory, sources, and similarity score."
    with q4:
        if st.button("🇮🇳 Hindi action plan", use_container_width=True):
            suggested_question = "Hindi me simple action plan batao. Ab farmer ko kya karna chahiye?"

    st.markdown('<div class="sh">💬 Chat</div>', unsafe_allow_html=True)

    # Display previous messages
    for msg in st.session_state.assistant_msgs:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_question = st.chat_input("Ask AgriSense Assistant...")
    final_question = suggested_question or user_question

    if final_question:
        st.session_state.assistant_msgs.append({"role":"user", "content":final_question})
        with st.chat_message("user"):
            st.markdown(final_question)

        with st.chat_message("assistant"):
            with st.spinner("AgriSense Assistant is thinking..."):
                answer = llm_assistant_answer(final_question)
                st.markdown(answer)
        st.session_state.assistant_msgs.append({"role":"assistant", "content":answer})

    c1, c2 = st.columns([1,3])
    with c1:
        if st.button("🧹 Clear Chat", use_container_width=True):
            st.session_state.assistant_msgs = []
            st.rerun()
    with c2:
        st.caption("Safety: advisory is generated from project context and retrieved sources. Always verify chemical treatment with local KVK/agriculture experts.")


# ── FOOTER ────────────────────────────────────────────────────
st.markdown("""
<hr/>
<div style='text-align:center;padding:.5rem 0 .9rem;'>
  <span style='color:rgba(46,204,113,.38);font-size:.74rem;'>
    🎯 SDG 2: Zero Hunger &nbsp;·&nbsp; 🌿 SDG 15: Life on Land
  </span>
</div>
""", unsafe_allow_html=True)
