from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from tensorflow.keras.models import load_model

from src.data_preprocessing import IMG_SIZE, preprocess_fingerprint


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "final_model.keras"
LABELS_PATH = ROOT / "models" / "class_names.json"

st.set_page_config(
    page_title="Blood Group Vision",
    page_icon="+",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
    :root { --ink: #18232b; --muted: #617078; --red: #c8463d; --rose: #f8e9e4; --line: #e8dfd9; --cream: #fbf8f3; --mint: #dcece5; }
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; color: var(--ink); }
    .stApp { background: var(--cream); }
    [data-testid="stSidebar"] { background: #18232b; border-right: 1px solid #30414a; }
    [data-testid="stSidebar"] * { color: #f8f3ed !important; }
    [data-testid="stSidebar"] hr { border-color: #40515a; }
    [data-testid="stSidebar"] .note { color: #c4d0d1 !important; }
    .sidebar-brand { border-bottom: 1px solid #40515a; padding-bottom: 1.2rem; margin-bottom: 1.3rem; }
    .sidebar-kicker { color: #f5b4a7 !important; font-size: .68rem; font-weight: 700; letter-spacing: .16em; text-transform: uppercase; }
    .sidebar-title { color: #ffffff !important; font-family: 'Space Grotesk', sans-serif; font-size: 1.45rem; font-weight: 700; margin-top: .35rem; }
    .sidebar-copy { color: #c4d0d1 !important; font-size: .86rem; line-height: 1.45; margin-top: .45rem; }
    .sidebar-section { color: #f5b4a7 !important; font-size: .68rem; font-weight: 700; letter-spacing: .14em; text-transform: uppercase; margin: 1.4rem 0 .65rem; }
    .sidebar-row { border-bottom: 1px solid #30414a; padding: .62rem 0; }
    .sidebar-label { color: #9fb0b3 !important; display: block; font-size: .73rem; margin-bottom: .12rem; }
    .sidebar-value { color: #ffffff !important; display: block; font-size: .9rem; font-weight: 600; }
    h1, h2, h3 { font-family: 'Space Grotesk', sans-serif; letter-spacing: 0; }
    .hero { padding: 2.2rem 0 1.4rem; border-bottom: 1px solid var(--line); }
    .eyebrow { color: var(--red); font-size: .76rem; font-weight: 700; letter-spacing: .13em; text-transform: uppercase; }
    .hero-headline { color: #18232b !important; display: block; font-family: 'Space Grotesk', sans-serif; font-size: clamp(2.4rem, 5vw, 4.6rem); font-weight: 700; line-height: .98; margin: .35rem 0 .8rem; max-width: 760px; opacity: 1 !important; }
    .hero p { color: var(--muted); font-size: 1.02rem; max-width: 620px; margin: 0; }
    .section-title { margin: 1.5rem 0 .7rem; font-family: 'Space Grotesk', sans-serif; font-size: 1.08rem; font-weight: 700; }
    .upload-panel { background: white; border: 1px solid var(--line); border-radius: 8px; padding: 1.1rem 1.2rem .8rem; }
    .result-card { background: var(--ink); color: white; padding: 1.35rem 1.5rem; border-radius: 8px; min-height: 142px; box-shadow: 0 10px 24px rgba(24,35,43,.12); }
    .result-label { color: #f5b4a7; font-size: .75rem; font-weight: 700; letter-spacing: .12em; text-transform: uppercase; }
    .result-value { font-family: 'Space Grotesk', sans-serif; font-size: 3.8rem; font-weight: 700; line-height: 1; margin: .35rem 0; }
    .result-confidence { color: #d5e4e3; font-size: .95rem; }
    .status-strip { display: flex; gap: .55rem; flex-wrap: wrap; margin-top: 1rem; }
    .status-pill { display: inline-block; background: var(--mint); color: #245246; border-radius: 999px; padding: .35rem .7rem; font-size: .78rem; font-weight: 700; }
    .subtle-pill { background: var(--rose); color: #803b34; }
    .note { color: var(--muted); font-size: .86rem; line-height: 1.5; }
    .stButton > button { background: var(--red); color: white; border: 0; border-radius: 5px; font-weight: 700; min-height: 2.8rem; }
    .stButton > button:hover { background: #a93630; color: white; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_model():
    return load_model(MODEL_PATH)


@st.cache_data
def get_class_names():
    import json

    with LABELS_PATH.open(encoding="utf-8") as labels_file:
        return json.load(labels_file)


def predict_image(image_bytes):
    image_array = np.frombuffer(image_bytes, dtype=np.uint8)
    model = get_model()
    channels = model.input_shape[-1]
    read_mode = cv2.IMREAD_COLOR if channels == 3 else cv2.IMREAD_GRAYSCALE
    image = cv2.imdecode(image_array, read_mode)
    if image is None:
        raise ValueError("The uploaded file is not a readable image.")

    processed = preprocess_fingerprint(image, IMG_SIZE, channels=channels)
    probabilities = model.predict(np.expand_dims(processed, axis=0), verbose=0)[0]
    class_names = get_class_names()
    ranked = sorted(
        zip(class_names, probabilities), key=lambda item: item[1], reverse=True
    )
    return image, ranked


with st.sidebar:
    st.markdown(
        '<div class="sidebar-brand"><div class="sidebar-kicker">Visual classifier</div><div class="sidebar-title">Blood Group<br>Vision</div><div class="sidebar-copy">A focused workspace for image-based blood-group research.</div></div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="sidebar-section">System</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-row"><span class="sidebar-label">Model</span><span class="sidebar-value">MobileNetV2 transfer</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-row"><span class="sidebar-label">Input format</span><span class="sidebar-value">RGB · 128 × 128 px</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-row"><span class="sidebar-label">Holdout accuracy</span><span class="sidebar-value">80.95%</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-section">Blood groups</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-copy">A+ · A− · AB+ · AB−<br>B+ · B− · O+ · O−</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="note" style="margin-top: 1.8rem; border-top: 1px solid #40515a; padding-top: 1rem;">For research and educational use only. This prediction is not a clinical blood-typing result.</div>',
        unsafe_allow_html=True,
    )

st.markdown(
    '<div class="hero"><div class="eyebrow">Blood group vision / eight classes</div><div class="hero-headline">One image.<br>Eight possibilities.</div><p>Turn a captured sample into a ranked blood-group prediction with a clear view of model confidence.</p><div class="status-strip"><span class="status-pill">MobileNetV2</span><span class="status-pill">RGB input</span><span class="status-pill subtle-pill">Research use</span></div></div>',
    unsafe_allow_html=True,
)

st.markdown('<div class="section-title">Analyze a sample</div>', unsafe_allow_html=True)
with st.container(border=True):
    uploaded_file = st.file_uploader(
        "Upload image",
        type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"],
        label_visibility="visible",
        help="Supported formats: JPG, PNG, BMP, and TIFF.",
    )

if uploaded_file is None:
    st.info("Upload a JPG, PNG, BMP, or TIFF image to begin.")
else:
    left, right = st.columns([.9, 1.35], gap="large")
    with left:
        st.image(uploaded_file, caption=uploaded_file.name, use_container_width=True)
        analyze = st.button("Analyze image", use_container_width=True)

    if analyze:
        if not MODEL_PATH.is_file() or not LABELS_PATH.is_file():
            st.error("Model files are missing. Run training first to create the files in models/.")
        else:
            with st.spinner("Analyzing image..."):
                try:
                    _, ranked = predict_image(uploaded_file.getvalue())
                except Exception as error:
                    st.error(f"Prediction failed: {error}")
                else:
                    predicted_class, confidence = ranked[0]
                    with right:
                        st.markdown(
                            f'<div class="result-card"><div class="result-label">Top prediction</div><div class="result-value">{predicted_class}</div><div class="result-confidence">{confidence * 100:.1f}% model confidence</div></div>',
                            unsafe_allow_html=True,
                        )
                        st.markdown('<div class="section-title">Confidence distribution</div>', unsafe_allow_html=True)
                        chart_data = pd.DataFrame(
                            {"Blood group": [name for name, _ in ranked], "Confidence": [float(value) for _, value in ranked]}
                        ).set_index("Blood group")
                        st.bar_chart(chart_data, horizontal=True, height=330)
