import streamlit as st
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image, ImageOps, ImageEnhance

# =====================================
# 1. PAGE CONFIGURATION
# =====================================

st.set_page_config(
    page_title="Handwritten Character Recognition",
    page_icon="✍️",
    layout="centered"
)

st.title("✍️ Handwritten Character Recognition")

st.write(
    "Upload an image containing a handwritten "
    "English letter or digit."
)

st.info(
    "CNN Model | EMNIST Balanced | "
    "47 Classes | Test Accuracy: 87.35%"
)


# =====================================
# 2. LOAD TRAINED MODEL
# =====================================

@st.cache_resource
def load_cnn_model():
    return tf.keras.models.load_model(
        "models/emnist_final_model.keras"
    )


model = load_cnn_model()


# =====================================
# 3. CLASS LABELS
# =====================================

balanced_characters = (
    "0123456789"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "abdefghnqrt"
)


# =====================================
# 4. IMAGE PREPROCESSING
# =====================================

def preprocess_image(image):

    # Convert image to RGB
    image = image.convert("RGB")

    # Convert to grayscale
    gray = ImageOps.grayscale(image)

    # Convert to NumPy
    arr = np.array(gray)

    # Estimate background from image borders
    border = np.concatenate([
        arr[0, :],
        arr[-1, :],
        arr[:, 0],
        arr[:, -1]
    ])

    background = np.median(border)

    # Convert to white character on black background
    if background > 127:
        image = ImageOps.invert(gray)
    else:
        image = gray.copy()

    # Enhance contrast
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(2.0)

    # Remove weak background pixels
    arr = np.array(image)
    arr[arr < 70] = 0

    image = Image.fromarray(arr)

    # Detect character bounding box
    bbox = image.getbbox()

    if bbox is None:
        raise ValueError(
            "No handwritten character detected."
        )

    # Crop character
    image = image.crop(bbox)

    # Resize while preserving aspect ratio
    image.thumbnail(
        (20, 20),
        Image.Resampling.LANCZOS
    )

    # Create black canvas
    canvas = Image.new(
        "L",
        (28, 28),
        color=0
    )

    # Center character
    x = (28 - image.width) // 2
    y = (28 - image.height) // 2

    canvas.paste(image, (x, y))

    # Normalize pixels
    img_array = (
        np.array(canvas, dtype=np.float32)
        / 255.0
    )

    # Add batch and channel dimensions
    img_array = img_array.reshape(
        1, 28, 28, 1
    )

    return img_array, canvas


# =====================================
# 5. IMAGE UPLOAD
# =====================================

uploaded_file = st.file_uploader(
    "📤 Upload Handwritten Character",
    type=["png", "jpg", "jpeg"]
)


if uploaded_file is not None:

    original_image = Image.open(uploaded_file)

    st.subheader("🖼️ Original Image")

    st.image(
        original_image,
        width=250
    )

    st.warning(
        "For best results, upload an image "
        "containing one handwritten character "
        "on a plain background."
    )

    # =====================================
    # 6. PREDICTION
    # =====================================

    if st.button("🔍 Recognize Character"):

        try:

            processed_array, processed_image = (
                preprocess_image(original_image)
            )

            st.subheader("⚙️ Processed Image")

            st.image(
                processed_image.resize(
                    (280, 280),
                    Image.Resampling.NEAREST
                ),
                width=280
            )

            # Predict
            prediction = model.predict(
                processed_array,
                verbose=0
            )[0]

            predicted_class = int(
                np.argmax(prediction)
            )

            predicted_character = (
                balanced_characters[predicted_class]
            )

            confidence = float(
                np.max(prediction) * 100
            )

            # =====================================
            # 7. DISPLAY RESULT
            # =====================================

            st.success(
                f"Predicted Character: "
                f"{predicted_character}"
            )

            st.metric(
                "Model Confidence",
                f"{confidence:.2f}%"
            )

            # =====================================
            # 8. TOP 5 PREDICTIONS
            # =====================================

            st.subheader("📊 Top 5 Predictions")

            top_indices = np.argsort(
                prediction
            )[-5:][::-1]

            top_characters = [
                balanced_characters[i]
                for i in top_indices
            ]

            top_probabilities = [
                float(prediction[i] * 100)
                for i in top_indices
            ]

            fig, ax = plt.subplots(
                figsize=(8, 4)
            )

            ax.barh(
                top_characters[::-1],
                top_probabilities[::-1]
            )

            ax.set_xlabel(
                "Model Score (%)"
            )

            ax.set_title(
                "Top 5 Character Predictions"
            )

            st.pyplot(fig)
            plt.close(fig)

            st.caption(
                "Confidence is based on the model's "
                "softmax output and is not a calibrated "
                "probability of correctness."
            )

        except Exception as e:

            st.error(
                f"Unable to process image: {e}"
            )


# =====================================
# 9. ABOUT MODEL
# =====================================

st.divider()

st.subheader("🧠 About This Model")

st.write(
    "This application uses a Convolutional "
    "Neural Network trained on the "
    "EMNIST Balanced dataset."
)

st.write(
    "The model recognizes 47 classes "
    "of handwritten English letters "
    "and digits."
)

st.write(
    "Test Accuracy: 87.35%"
)

st.caption(
    "Developed by Dania Yasir | "
    "CodeAlpha Machine Learning Internship"
)
