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

    import cv2
    import numpy as np
    from PIL import Image

    # 1. Convert image to grayscale
    image = image.convert("RGB")
    gray = cv2.cvtColor(
        np.array(image),
        cv2.COLOR_RGB2GRAY
    )

    # 2. Reduce noise
    blurred = cv2.GaussianBlur(
        gray, (5, 5), 0
    )

    # 3. Detect foreground using adaptive thresholding
    binary = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        15
    )

    # 4. Find contours
    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        raise ValueError(
            "No handwritten character detected."
        )

    # 5. Select largest detected contour
    contour = max(
        contours,
        key=cv2.contourArea
    )

    # 6. Get character bounding box
    x, y, w, h = cv2.boundingRect(contour)

    # 7. Crop character
    cropped = binary[y:y+h, x:x+w]

    # 8. Resize while preserving aspect ratio
    target_size = 20

    scale = target_size / max(w, h)

    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))

    resized = cv2.resize(
        cropped,
        (new_w, new_h),
        interpolation=cv2.INTER_AREA
        if scale < 1
        else cv2.INTER_LINEAR
    )

    # 9. Create black 28x28 canvas
    canvas = np.zeros(
        (28, 28),
        dtype=np.uint8
    )

    # 10. Center the character
    x_offset = (28 - new_w) // 2
    y_offset = (28 - new_h) // 2

    canvas[
        y_offset:y_offset+new_h,
        x_offset:x_offset+new_w
    ] = resized

    # 11. Normalize
    img_array = (
        canvas.astype(np.float32) / 255.0
    )

    # 12. Prepare CNN input
    img_array = img_array.reshape(
        1, 28, 28, 1
    )

    return img_array, Image.fromarray(canvas)

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
