import streamlit as st

# Page configuration
st.set_page_config(
    page_title="Handwritten Character Recognition",
    page_icon="✍️",
    layout="centered"
)

# Website title
st.title("✍️ Handwritten Character Recognition")

# Description
st.write(
    "Welcome to my AI-powered Handwritten Character "
    "Recognition application."
)

st.write(
    "This application uses a Convolutional Neural "
    "Network (CNN) trained on the EMNIST Balanced dataset."
)

# Model information
st.info("Model: CNN | Dataset: EMNIST Balanced | Accuracy: 87.35%")

# Upload image
uploaded_file = st.file_uploader(
    "Upload a handwritten character image",
    type=["png", "jpg", "jpeg"]
)

# Display uploaded image
if uploaded_file is not None:

    st.image(
        uploaded_file,
        caption="Uploaded Image",
        width=250
    )

    st.success("Image uploaded successfully!")
