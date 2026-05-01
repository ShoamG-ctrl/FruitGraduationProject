import pathlib
from fastai.vision.all import *
import streamlit as st
from PIL import Image as PIL_Img
import numpy as np
import os
import gdown

# --- תיקון קריטי לשגיאת ה-Resolver (המערכת מחפשת את המחלקה הזו) ---
class Resolver:
    def dict(self): return {}

# תיקון לנתיבי Windows/Linux
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

# אתחול הזיכרון של האפליקציה
if 'basket' not in st.session_state:
    st.session_state.basket = []
if 'camera_key' not in st.session_state:
    st.session_state.camera_key = 0

# הגדרות דף הממשק
st.set_page_config(page_title="Fruit Guard AI", page_icon="🍎")

# עיצוב ויזואלי לממשק
st.markdown("""
    <style>
    .instruction-text { text-align: center; font-size: 20px; font-weight: bold; direction: rtl; }
    .focus-box {
        position: absolute; bottom: 150px; left: 50%; transform: translateX(-50%);
        width: 350px; height: 250px; border: 4px dashed black; border-radius: 15px;
        pointer-events: none; z-index: 99;
    }
    div.stButton > button { border-radius: 10px; font-weight: bold; height: 3em; width: 100%; }
    .add-btn > div > button { background-color: #2196F3; color: white; }
    .retake-btn > div > button { background-color: #607D8B; color: white; }
    .analyze-btn > div > button { background-color: #4CAF50; color: white; }
    </style>
    """, unsafe_allow_html=True)

st.title("🍎 Fruit Guard AI")
st.subheader("פרויקט גמר: שי עטר, שוהם גדליה, מיכאל פילוסוף")

# פונקציה להורדת המודל מ-Google Drive
@st.cache_resource
def load_my_model():
    file_id = '1YSaA9C6evr7I5yGpCXN8QY7fuGLETDL-'
    model_path = 'fruit_model.pkl'
    
    if not os.path.exists(model_path):
        with st.spinner('מוריד מודל מ-Google Drive...'):
            url = f'https://drive.google.com/uc?id={file_id}'
            gdown.download(url, model_path, quiet=False)
    
    # טעינת המודל עם הגדרות תאימות
    return load_learner(model_path, cpu=True)

try:
    learn = load_my_model()
except Exception as e:
    st.error(f"שגיאה בטעינת המודל: {e}")

option = st.radio("בחר שיטה:", ("העלאת קובצים", "מצלמה חיה"))

should_analyze = False

if option == "העלאת קובצים":
    uploaded = st.file_uploader("בחר תמונות...", type=["jpg", "png", "jpeg"], accept_multiple_files=True)
    if uploaded:
        st.session_state.basket = list(uploaded)
        should_analyze = True
else:
    # הנחיית מרחק כף היד (12 ס"מ) שפיתחתם
    st.markdown('<p class="instruction-text">צלם את הפרי ממרחק כף יד (12 ס"מ)</p>', unsafe_allow_html=True)
    cam_file = st.camera_input("", key=f"cam_{st.session_state.camera_key}")
    st.markdown('<div class="focus-box"></div>', unsafe_allow_html=True)
    
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

# ביצוע הסיווג (Ripeness Classification)
if st.session_state.basket and should_analyze:
    st.write("---")
    st.write(f"### 🧺 תוצאות ניתוח ({len(st.session_state.basket)} תמונות):")
    
    results_list = []
    
    for i, file in enumerate(st.session_state.basket):
        try:
            img_pil = PIL_Img.open(file).convert('RGB')
            dl = learn.dls.test_dl([img_pil])
            batch_preds = learn.get_preds(dl=dl, with_decoded=True)
            res_idx = batch_preds[2][0].item()
            res_label = str(learn.dls.vocab[res_idx]).lower()
            conf = batch_preds[0][0][res_idx].item()
            
            results_list.append(res_label)
            
            with st.container():
                c1, c2 = st.columns([1, 2])
                c1.image(img_pil, width=150)
                trans = {"ripe": "בשל", "unripe": "בוסר", "rotten": "רקוב"}
                heb = trans.get(res_label, res_label)
                c2.write(f"**תמונה {i+1}:** {heb} ({conf*100:.1f}%)")
        except Exception as e:
            st.error(f"שגיאה בעיבוד תמונה {i+1}: {e}")

    # לוגיקה לסיכום מצב הפרי
    st.write("---")
    if "rotten" in results_list:
        st.error("## 🏁 סיכום: הפרי רקוב ❌")
    elif "unripe" in results_list:
        st.warning("## 🏁 סיכום: הפרי בוסר ⚠️")
    else:
        st.success("## 🏁 סיכום: הפרי בשל וטוב למאכל ✅")
        st.balloons()

    if st.button("🗑️ נקה הכל"):
        st.session_state.basket = []
        st.session_state.camera_key += 1
        st.rerun()