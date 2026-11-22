from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
from streamlit_drawable_canvas import st_canvas


# ---------- Page & design ----------
st.set_page_config(
    page_title="Handwriting AI | Character Recognition",
    page_icon="✍️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
    :root { --bg:#071510; --card:#10251d; --line:#27513f; --muted:#a7c8b6; --accent:#34d399; }
    .stApp { background: radial-gradient(ellipse at 70% -10%, #123d30 0%, #071510 52%); color:#f0fdf4; font-family:'DM Sans',sans-serif; }
    header[data-testid="stHeader"] { background:transparent; }
    .block-container { max-width:1140px; padding-top:2.2rem; padding-bottom:4rem; }
    h1, h2, h3 { font-family:'Space Grotesk',sans-serif; color:#f0fdf4!important; }
    p, label { color:#d1e9db; }
    .brand { color:#6ee7b7; font-size:.85rem; font-weight:700; letter-spacing:.15em; text-transform:uppercase; }
    .hero-title { font-family:'Space Grotesk',sans-serif; font-size:clamp(2.4rem,5vw,4.3rem); font-weight:700; line-height:1.09; letter-spacing:-.05em; margin:.6rem 0 1rem; color:#f0fdf4; }
    .gradient-text { background:linear-gradient(90deg,#34d399,#a3e635); color:transparent; background-clip:text; -webkit-background-clip:text; }
    .lead { color:#b7d7c4; font-size:1.06rem; max-width:700px; line-height:1.7; }
    .pill { display:inline-block; background:#133527; border:1px solid #2f7654; color:#b3f2d0; border-radius:99px; padding:.35rem .8rem; font-size:.83rem; margin-right:.4rem; margin-bottom:.5rem; }
    .section-title { font-family:'Space Grotesk',sans-serif; font-size:1.5rem; font-weight:700; margin:1.8rem 0 1rem; color:#f0fdf4; }
    .stat { background:linear-gradient(145deg,#153327,#0b2119); border:1px solid #28513e; border-radius:18px; padding:1rem 1.25rem; min-height:100px; }
    .stat-number { font-family:'Space Grotesk',sans-serif; font-size:1.75rem; font-weight:700; color:#f2fff5; }
    .stat-label { color:#9ac4ac; font-size:.85rem; margin-top:.3rem; }
    .pane-title { font-family:'Space Grotesk',sans-serif; font-size:1.15rem; font-weight:700; margin:.25rem 0 .5rem; color:#f0fdf4; }
    .pane-sub { font-size:.9rem; color:#9ac4ac; margin-bottom:1rem; }
    .result-card { background:linear-gradient(135deg,#163a2c,#0f291f); border:1px solid #33845b; border-radius:18px; padding:1.4rem; text-align:center; }
    .result-char { font-family:'Space Grotesk',sans-serif; font-size:5rem; font-weight:700; line-height:1.15; color:#6ee7b7; overflow-wrap:anywhere; }
    .result-label { color:#9ccbb1; font-size:.85rem; text-transform:uppercase; letter-spacing:.1em; }
    .foot { text-align:center; color:#82ab93; font-size:.82rem; padding-top:2rem; }
    div[data-testid="stVerticalBlockBorderWrapper"] { background:#0e251c; border-color:#29533e!important; border-radius:20px; }
    .stButton>button[kind="primary"] { background:linear-gradient(90deg,#059669,#16a34a); color:white; border:0; border-radius:11px; font-weight:700; min-height:3.05rem; }
    .stButton>button { border-radius:11px; border-color:#32654d; }
    .stButton>button:hover { border-color:#34d399; color:#a7f3d0; }
    .stButton>button[kind="primary"]:hover { background:linear-gradient(90deg,#047857,#15803d); color:#ffffff; }
    div[data-testid="stProgress"] > div > div { background-color:#10b981; }
    div[data-testid="stSlider"] [role="slider"] { background-color:#10b981; }
    a { color:#6ee7b7; }
    a:hover { color:#a3e635; }
    div[data-testid="stFileUploader"] { border-radius:14px; }
    div[data-testid="stMetric"] { background:#102b20; border:1px solid #2d5f46; padding:.8rem 1rem; border-radius:12px; }
    div[data-testid="stRadio"] label { color:#d9fbe5; }
    @media(max-width:640px) { .block-container { padding-left:1rem; padding-right:1rem; padding-top:1.4rem; } .hero-title {font-size:2.4rem;} }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="brand">✦ DANIA YASIR · MACHINE LEARNING PROJECT</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-title">Make handwriting<br><span class="gradient-text">machine readable.</span></div>', unsafe_allow_html=True)
st.markdown('<div class="lead">Draw a character or upload an image and see how a convolutional neural network reads it — instantly.</div>', unsafe_allow_html=True)
st.markdown('<span class="pill">✍️ Draw or upload</span><span class="pill">🧠 CNN-powered</span><span class="pill">🔎 47 classes</span>', unsafe_allow_html=True)

s1, s2, s3 = st.columns(3)
for col, number, label in [
    (s1, "87.35%", "EMNIST test accuracy"),
    (s2, "47", "Letters & digits"),
    (s3, "28 × 28", "Model input resolution"),
]:
    with col:
        st.markdown(f'<div class="stat"><div class="stat-number">{number}</div><div class="stat-label">{label}</div></div>', unsafe_allow_html=True)


# ---------- Model ----------
LABELS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabdefghnqrt"


@st.cache_resource(show_spinner="Loading handwriting model…")
def load_model():
    path = Path(__file__).parent / "models" / "emnist_final_model.keras"
    return tf.keras.models.load_model(path, compile=False)


@st.cache_data(show_spinner=False)
def preprocess(image_bytes: bytes, mode: str):
    """Prepare a 28x28 MNIST-style image from a drawing or an upload.

    Upload mode restores the original adaptive-threshold + largest-contour
    pipeline used before the frontend redesign. Draw mode keeps the simple
    Otsu-based extraction for the white canvas.
    """
    from io import BytesIO

    pil = Image.open(BytesIO(image_bytes)).convert("RGB")
    gray = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2GRAY)

    if mode == "upload":
        # Original, previously working upload preprocessing.
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        binary = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 31, 15
        )
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            raise ValueError("No handwritten character detected.")
        contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(contour)
        char = binary[y:y + h, x:x + w]
    else:
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        _, ink = cv2.threshold(
            blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
        n, labels, stats, _ = cv2.connectedComponentsWithStats(ink, 8)
        min_area = max(2, int(ink.size * 0.00008))
        cleaned = np.zeros_like(ink)
        for idx in range(1, n):
            if stats[idx, cv2.CC_STAT_AREA] >= min_area:
                cleaned[labels == idx] = 255
        ys, xs = np.where(cleaned > 0)
        if len(xs) == 0:
            raise ValueError("No character detected. Draw a darker character.")
        char = cleaned[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        h, w = char.shape

    scale = 20 / max(w, h)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    resized = cv2.resize(
        char, (new_w, new_h),
        interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    )
    canvas = np.zeros((28, 28), dtype=np.uint8)
    ox, oy = (28 - new_w) // 2, (28 - new_h) // 2
    canvas[oy:oy + new_h, ox:ox + new_w] = resized
    tensor = canvas.astype(np.float32).reshape(1, 28, 28, 1) / 255.0
    return tensor, Image.fromarray(canvas)


st.markdown('<div class="section-title">Try the model</div>', unsafe_allow_html=True)
left, right = st.columns([1.14, 0.86], gap="large")

with left:
    with st.container(border=True):
        st.markdown('<div class="pane-title">Your handwriting</div><div class="pane-sub">Choose one input method. Use a single English letter or digit.</div>', unsafe_allow_html=True)
        method = st.radio("Input method", ["✍️ Draw", "📤 Upload"], horizontal=True, label_visibility="collapsed")
        image_bytes = None
        input_mode = "draw" if method == "✍️ Draw" else "upload"
        if method == "✍️ Draw":
            if "canvas_version" not in st.session_state:
                st.session_state.canvas_version = 0
            bar1, bar2 = st.columns([2, 1])
            with bar1:
                stroke = st.slider("Pen thickness", 6, 28, 17)
            with bar2:
                st.write("")
                st.write("")
                if st.button("↺ Clear drawing", use_container_width=True):
                    st.session_state.canvas_version += 1
                    st.rerun()
            drawing = st_canvas(
                stroke_width=stroke,
                stroke_color="#111111",
                background_color="#FFFFFF",
                drawing_mode="freedraw",
                width=300,
                height=300,
                key=f"canvas_{st.session_state.canvas_version}",
                update_streamlit=True,
                return_image_data=True,
            )
            if drawing.image_data is not None:
                arr = drawing.image_data.astype(np.uint8)
                rgb = Image.fromarray(arr, "RGBA").convert("RGB")
                if np.any(np.asarray(rgb.convert("L")) < 220):
                    from io import BytesIO
                    buffer = BytesIO()
                    rgb.save(buffer, format="PNG")
                    image_bytes = buffer.getvalue()
        else:
            upload = st.file_uploader("Upload a JPG or PNG with one handwritten character", type=["jpg", "jpeg", "png"])
            if upload:
                image_bytes = upload.getvalue()
                st.image(image_bytes, caption="Original uploaded image", use_container_width=True)
                st.caption("Tip: crop out paper borders or patterned backgrounds before uploading.")

        recognize = st.button("✦ Recognize character", type="primary", use_container_width=True, disabled=image_bytes is None)

with right:
    with st.container(border=True):
        st.markdown('<div class="pane-title">Recognition result</div><div class="pane-sub">Prediction and image sent to the CNN</div>', unsafe_allow_html=True)
        if recognize and image_bytes:
            try:
                tensor, preview = preprocess(image_bytes, input_mode)
                cnn = load_model()
                scores = cnn.predict(tensor, verbose=0)[0]
                idx = int(np.argmax(scores))
                st.markdown(f'<div class="result-card"><div class="result-label">Predicted character</div><div class="result-char">{LABELS[idx]}</div></div>', unsafe_allow_html=True)
                st.metric("Model score", f"{float(scores[idx])*100:.2f}%")
                st.caption("Score is a softmax output, not a calibrated likelihood of correctness.")
                st.image(preview.resize((224, 224), Image.Resampling.NEAREST), caption="Processed 28 × 28 input", width=224)
                st.markdown("**Top 5 alternatives**")
                top = np.argsort(scores)[-5:][::-1]
                for k in top:
                    st.write(f"**{LABELS[k]}** · {scores[k]*100:.1f}%")
                    st.progress(float(scores[k]))
            except Exception as exc:
                st.error(f"Could not recognize this image: {exc}")
        else:
            st.info("Your prediction will appear here after you draw or upload a character and click **Recognize character**.")

st.markdown('<div class="section-title">Built with deep learning</div>', unsafe_allow_html=True)
st.write("A convolutional neural network trained on the **EMNIST Balanced** dataset. The test-set accuracy is **87.35%**; individual handwriting and photos may be harder to classify, especially similar-looking characters such as **O/0** and **I/1**.")
st.markdown('[View source code on GitHub](https://github.com/Dania-Yasir/CodeAlpha_Handwritten_Character_Recognition)')
st.markdown('<div class="foot">Built by Dania Yasir · CodeAlpha Machine Learning Internship</div>', unsafe_allow_html=True)
