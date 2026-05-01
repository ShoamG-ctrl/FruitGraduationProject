import pathlib
from fastai.vision.all import *
import streamlit as st
from PIL import Image as PIL_Img
import numpy as np
import os

# תיקון לנתיבי Windows
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

# אתחול הזיכרון (Session State)
if 'basket' not in st.session_state:
    st.session_state.basket = []
if 'camera_key' not in st.session_state:
    st.session_state.camera_key = 0

# עיצוב הממשק (CSS)
st.markdown("""
    <style>
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

st.set_page_config(page_title="Fruit Guard AI", page_icon="🍎")
st.title("🍎 Fruit Guard AI")
st.subheader("פרויקט גמר: שוהם גדליה, שי עטר, מיכאל פילוסוף")

@st.cache_resource
def load_my_model():
    # שם המודל המעודכן
    model_path = 'fastai_classification_model_fruits_shoam_v3.1.pkl'
    if os.path.exists(model_path):
        return load_learner(model_path)
    return None

learn = load_my_model()

if learn is None:
    st.error("קובץ מודל חסר במערכת")
    st.stop()

# בחירת שיטה ללא טקסט מיותר
option = st.radio("בחר שיטה:", ("מצלמה חיה", "העלאת קובצים"), label_visibility="collapsed")

should_analyze = False

if option == "העלאת קובצים":
    uploaded = st.file_uploader("בחר קבצים", type=["jpg", "png", "jpeg"], accept_multiple_files=True, label_visibility="collapsed")
    if uploaded:
        st.session_state.basket = list(uploaded)
        should_analyze = True
else:
    # מצלמה עם Label מינימלי למניעת שגיאות
    cam_file = st.camera_input("צילום פרי", key=f"cam_{st.session_state.camera_key}", label_visibility="collapsed")
    st.markdown('<div class="focus-box"></div>', unsafe_allow_html=True)
    
    if cam_file:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown('<div class="add-btn">', unsafe_allow_html=True)
            if st.button("➕ הוסף"):
                st.session_state.basket.append(cam_file)
                st.toast("נוסף!")
            st.markdown('</div>', unsafe_allow_html=True)
        with col2:
            st.markdown('<div class="retake-btn">', unsafe_allow_html=True)
            if st.button("🔄 צילום"):
                st.session_state.camera_key += 1
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
        with col3:
            st.markdown('<div class="analyze-btn">', unsafe_allow_html=True)
            if st.button("🚀 נתח"):
                if cam_file not in st.session_state.basket:
                    st.session_state.basket.append(cam_file)
                should_analyze = True 
            st.markdown('</div>', unsafe_allow_html=True)

# הרצת הניתוח
if st.session_state.basket and should_analyze:
    st.write("---")
    results_list = []
    
    for i, file in enumerate(st.session_state.basket):
        try:
            img_pil = PIL_Img.open(file).convert('RGB')
            w, h = img_pil.size
            # שימוש בלוגיקת ה-Crop המקורית שלכם
            if w > 350 and h > 250:
                left, top = (w-350)/2, (h-250)/2
                img_pil = img_pil.crop((left, top, left+350, top+250))
            
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
                c2.write(f"**תוצאה:** {heb} ({conf*100:.1f}%)")
        except:
            continue

    st.write("---")
    if results_list:
        if "rotten" in results_list:
            st.error("## סיכום: הפרי רקוב ❌")
        elif "unripe" in results_list:
            st.warning("## סיכום: הפרי בוסר ⚠️")
        else:
            st.success("## סיכום: הפרי בשל ✅")
            st.balloons()

    if st.button("🗑️ נקה הכל"):
        st.session_state.basket = []
        st.session_state.camera_key += 1
        st.rerun()