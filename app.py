import pathlib
import os
import platform
import streamlit as st
from fastai.vision.all import *
from PIL import Image as PIL_Img
import gdown

# --- 1. תיקון קריטי לשגיאת ה-Resolver (פותר את הבעיה בתמונה) ---
class Resolver:
    def __init__(self, *args, **kwargs): pass
    def dict(self, *args, **kwargs): return {}
    def __call__(self, *args, **kwargs): return self

# --- 2. תיקון תאימות לנתיבים (Windows/Linux) ---
if platform.system() == 'Linux':
    pathlib.WindowsPath = pathlib.PosixPath
else:
    pathlib.PosixPath = pathlib.WindowsPath

# --- 3. אתחול הזיכרון (סל הפירות) ---
if 'basket' not in st.session_state:
    st.session_state.basket = []
if 'camera_key' not in st.session_state:
    st.session_state.camera_key = 0

# הגדרות דף
st.set_page_config(page_title="Fruit Guard AI", page_icon="🍎")

# --- 4. עיצוב המסגרת הוירטואלית (מתאימה גם לאבטיחים) ---
st.markdown("""
    <style>
    .instruction-text { text-align: center; font-size: 20px; font-weight: bold; direction: rtl; }
    
    .focus-box {
        position: absolute; 
        top: 50px; 
        left: 50%; 
        transform: translateX(-50%);
        width: 85%; 
        max-width: 450px;
        height: 300px; 
        border: 5px dashed #FFEB3B; 
        border-radius: 25px;
        pointer-events: none; 
        z-index: 99;
    }
    
    div.stButton > button { border-radius: 10px; font-weight: bold; height: 3em; width: 100%; }
    </style>
    """, unsafe_allow_html=True)

st.title("🍎 Fruit Guard AI")
st.subheader("פרויקט גמר: שי עטר, שוהם גדליה, מיכאל פילוסוף")

# --- 5. טעינת המודל מ-Google Drive ---
@st.cache_resource
def load_my_model():
    file_id = '1YSaA9C6evr7I5yGpCXN8QY7fuGLETDL-'
    model_path = 'fruit_model.pkl'
    if not os.path.exists(model_path):
        with st.spinner('מוריד מודל מ-Google Drive...'):
            url = f'https://drive.google.com/uc?id={file_id}'
            gdown.download(url, model_path, quiet=False)
    # שימוש ב-cpu=True כדי למנוע קריסות בענן
    return load_learner(model_path, cpu=True)

try:
    learn = load_my_model()
except Exception as e:
    st.error(f"שגיאה בטעינת המודל: {e}")
    st.info("נסה ללחוץ על שלוש הנקודות בפינה ולבחור 'Clear Cache'")

# --- 6. ממשק בחירה ---
option = st.sidebar.radio("בחר שיטה:", ("העלאת קבצים", "מצלמה חיה"))
should_analyze = False

if option == "העלאת קבצים":
    uploaded_files = st.file_uploader("בחר תמונות להוספה לסל:", type=["jpg", "png", "jpeg"], accept_multiple_files=True)
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("➕ הוסף תמונות לסל"):
            if uploaded_files:
                st.session_state.basket.extend(uploaded_files)
                st.toast(f"נוספו {len(uploaded_files)} תמונות!")
    with col2:
        if st.button("🚀 נתח את כל הסל"):
            should_analyze = True

else:
    st.markdown('<p class="instruction-text">מקם את הפרי בתוך המסגרת הצהובה</p>', unsafe_allow_html=True)
    st.markdown('<div class="focus-box"></div>', unsafe_allow_html=True)
    
    cam_file = st.camera_input("", key=f"cam_{st.session_state.camera_key}")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("➕ הוסף תמונה לסל"):
            if cam_file:
                st.session_state.basket.append(cam_file)
                st.toast("התמונה נוספה!")
    with col2:
        if st.button("🚀 נתח את כל הסל"):
            should_analyze = True

# --- 7. הצגת הסל וביצוע הניתוח ---
if st.session_state.basket:
    st.write(f"### 🧺 סל הפירות שלך ({len(st.session_state.basket)} תמונות)")
    if st.button("🗑️ רוקן סל"):
        st.session_state.basket = []
        st.session_state.camera_key += 1
        st.rerun()

if st.session_state.basket and should_analyze:
    st.write("---")
    results_list = []
    
    for i, file in enumerate(st.session_state.basket):
        try:
            img_pil = PIL_Img.open(file).convert('RGB')
            pred, pred_idx, probs = learn.predict(img_pil)
            res_label = str(pred).lower()
            results_list.append(res_label)
            
            with st.container():
                c1, c2 = st.columns([1, 2])
                c1.image(img_pil, width=150)
                trans = {"ripe": "בשל", "unripe": "בוסר", "rotten": "רקוב"}
                heb = trans.get(res_label, res_label)
                c2.write(f"**תמונה {i+1}:** {heb} ({probs[pred_idx].item()*100:.1f}%)")
        except: continue

    st.write("---")
    if "rotten" in results_list:
        st.error("## סיכום סופי: נמצא פרי רקוב בסל ❌")
    elif "unripe" in results_list:
        st.warning("## סיכום סופי: הפירות בוסר ⚠️")
    else:
        st.success("## סיכום סופי: הפירות בשלים וטובים ✅")
        st.balloons()