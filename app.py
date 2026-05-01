import pathlib
import os

# --- תיקון דחוף לשגיאת ה-IPython והתצוגה (מטפל ב-image_a7995e.png) ---
import matplotlib
matplotlib.use('Agg') # מונע מהמערכת לנסות לפתוח חלונות תצוגה שלא קיימים בשרת
os.environ['FASTAI_TB_CLEAR_OUTPUT'] = 'never'

from fastai.vision.all import *
import streamlit as st
from PIL import Image as PIL_Img
import numpy as np
import gdown

# --- תיקון קריטי לשגיאת ה-Resolver ---
class Resolver:
    def dict(self): return {}

# תיקון תאימות לנתיבים (Windows/Linux)
import platform
if platform.system() == 'Linux':
    pathlib.WindowsPath = pathlib.PosixPath
else:
    pathlib.PosixPath = pathlib.WindowsPath

# אתחול הזיכרון
if 'basket' not in st.session_state:
    st.session_state.basket = []
if 'camera_key' not in st.session_state:
    st.session_state.camera_key = 0

st.set_page_config(page_title="Fruit Guard AI", page_icon="🍎")

# עיצוב
st.markdown("""
    <style>
    .instruction-text { text-align: center; font-size: 20px; font-weight: bold; direction: rtl; }
    div.stButton > button { border-radius: 10px; font-weight: bold; height: 3em; width: 100%; }
    </style>
    """, unsafe_allow_html=True)

st.title("🍎 Fruit Guard AI")
st.subheader("פרויקט גמר: שי עטר, שוהם גדליה, מיכאל פילוסוף")

@st.cache_resource
def load_my_model():
    file_id = '1YSaA9C6evr7I5yGpCXN8QY7fuGLETDL-'
    model_path = 'fruit_model.pkl'
    
    if not os.path.exists(model_path):
        with st.spinner('מוריד מודל מ-Google Drive...'):
            url = f'https://drive.google.com/uc?id={file_id}'
            gdown.download(url, model_path, quiet=False)
    
    return load_learner(model_path, cpu=True)

try:
    learn = load_my_model()
except Exception as e:
    st.error(f"שגיאה בטעינת המודל: {e}")

option = st.radio("בחר שיטה:", ("העלאת קבצים", "מצלמה חיה"))

should_analyze = False

if option == "העלאת קובצים":
    uploaded = st.file_uploader("בחר תמונות...", type=["jpg", "png", "jpeg"], accept_multiple_files=True)
    if uploaded:
        st.session_state.basket = list(uploaded)
        should_analyze = True
else:
    st.markdown('<p class="instruction-text">צלם את הפרי ממרחק כף יד (12 ס"מ)</p>', unsafe_allow_html=True)
    cam_file = st.camera_input("", key=f"cam_{st.session_state.camera_key}")
    
    if cam_file:
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("➕ הוסף לסל"):
                st.session_state.basket.append(cam_file)
                st.toast("נוסף לסל!")
        with col2:
            if st.button("🔄 צילום חדש"):
                st.session_state.camera_key += 1
                st.rerun()
        with col3:
            if st.button("🚀 נתח פרי"):
                if cam_file not in st.session_state.basket:
                    st.session_state.basket.append(cam_file)
                should_analyze = True 

if st.session_state.basket and should_analyze:
    st.write("---")
    st.write(f"### 🧺 תוצאות ניתוח ({len(st.session_state.basket)} תמונות):")
    
    results_list = []
    
    for i, file in enumerate(st.session_state.basket):
        try:
            img_pil = PIL_Img.open(file).convert('RGB')
            # שימוש בשיטה בטוחה יותר לסיווג בשרת
            pred, pred_idx, probs = learn.predict(img_pil)
            res_label = str(pred).lower()
            conf = probs[pred_idx].item()
            
            results_list.append(res_label)
            
            with st.container():
                c1, c2 = st.columns([1, 2])
                c1.image(img_pil, width=150)
                trans = {"ripe": "בשל", "unripe": "בוסר", "rotten": "רקוב"}
                heb = trans.get(res_label, res_label)
                c2.write(f"**תמונה {i+1}:** {heb} ({conf*100:.1f}%)")
        except Exception as e:
            st.error(f"שגיאה בעיבוד תמונה {i+1}: {e}")

    st.write("---")
    if "rotten" in results_list:
        st.error("## 🏁 סיכום: הפרי רקוב ❌")
    elif "unripe" in results_list:
        st.warning("## 🏁 סיכום: הפרי בוסר ⚠️")
    elif "ripe" in results_list:
        st.success("## 🏁 סיכום: הפרי בשל וטוב למאכל ✅")
        st.balloons()

    if st.button("🗑️ נקה הכל"):
        st.session_state.basket = []
        st.session_state.camera_key += 1
        st.rerun()