import pathlib
from fastai.vision.all import *
import streamlit as st
from PIL import Image as PIL_Img
import numpy as np

# תיקון לנתיבי Windows
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

# אתחול הזיכרון
if 'basket' not in st.session_state:
    st.session_state.basket = []
if 'camera_key' not in st.session_state:
    st.session_state.camera_key = 0

# עיצוב הממשק
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

st.set_page_config(page_title="Fruit Guard AI", page_icon="🍎")
st.title("🍎 Fruit Guard AI")
st.subheader("פרויקט גמר: שי עטר, שוהם גדליה, מיכאל פילוסוף")

@st.cache_resource
def load_my_model():
    return load_learner('fastai_classification_model_fruits_shoam_v2.1.pkl1.1.pkl')

try:
    learn = load_my_model()
except Exception as e:
    st.error(f"שגיאה בטעינת המודל: {e}")

option = st.radio("בחר שיטה:", ("העלאת קובצים", "מצלמה חיה"))

# משתנה שיעזור לנו לדעת אם להריץ ניתוח עכשיו
should_analyze = False

if option == "העלאת קובצים":
    uploaded = st.file_uploader("בחר תמונות...", type=["jpg", "png", "jpeg"], accept_multiple_files=True)
    if uploaded:
        st.session_state.basket = list(uploaded)
        should_analyze = True
else:
    st.markdown('<p class="instruction-text">צלם את הפרי מכל הצדדים</p>', unsafe_allow_html=True)
    cam_file = st.camera_input("", key=f"cam_{st.session_state.camera_key}")
    st.markdown('<div class="focus-box"></div>', unsafe_allow_html=True)
    
    if cam_file:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown('<div class="add-btn">', unsafe_allow_html=True)
            if st.button("➕ הוסף לסל"):
                st.session_state.basket.append(cam_file)
                st.toast("נוסף לסל!")
            st.markdown('</div>', unsafe_allow_html=True)
        with col2:
            st.markdown('<div class="retake-btn">', unsafe_allow_html=True)
            if st.button("🔄 צילום חדש"):
                st.session_state.camera_key += 1
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
        with col3:
            st.markdown('<div class="analyze-btn">', unsafe_allow_html=True)
            if st.button("🚀 נתח פרי"):
                if cam_file not in st.session_state.basket:
                    st.session_state.basket.append(cam_file)
                should_analyze = True # הפעלה ישירה
            st.markdown('</div>', unsafe_allow_html=True)

# ביצוע הסיווג
if st.session_state.basket and should_analyze:
    st.write("---")
    st.write(f"### 🧺 תוצאות ניתוח ({len(st.session_state.basket)} תמונות):")
    
    results_list = []
    
    for i, file in enumerate(st.session_state.basket):
        try:
            img_pil = PIL_Img.open(file).convert('RGB')
            w, h = img_pil.size
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
                c2.write(f"**תמונה {i+1}:** {heb} ({conf*100:.1f}%)")
        except:
            st.error(f"שגיאה בעיבוד תמונה {i+1}")

    # שורת סיכום סופית (הלוגיקה המחמירה)
    st.write("---")
    if "rotten" in results_list:
        st.error("## 🏁 סיכום: הפרי רקוב ❌")
        st.write("הסבר: נמצאו סימני ריקבון לפחות באחת מהתמונות.")
    elif "unripe" in results_list:
        st.warning("## 🏁 סיכום: הפרי בוסר ⚠️")
        st.write("הסבר: הפרי לא רקוב, אך עדיין לא הבשיל לגמרי.")
    else:
        st.success("## 🏁 סיכום: הפרי בשל וטוב למאכל ✅")
        st.balloons()

    if st.button("🗑️ נקה הכל והתחל מחדש"):
        st.session_state.basket = []
        st.session_state.camera_key += 1
        st.rerun()
