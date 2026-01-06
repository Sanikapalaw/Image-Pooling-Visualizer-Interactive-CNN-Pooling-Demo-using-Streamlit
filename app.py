import streamlit as st
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="Image Pooling Playground", layout="wide")
st.title("🧠 Interactive Image Pooling Playground")

st.write(
    "Upload an image and experiment with grayscale conversion and CNN pooling "
    "using fully customizable parameters."
)

# -----------------------------
# SIDEBAR CONTROLS
# -----------------------------
st.sidebar.header("⚙️ Controls")

# Grayscale
gray_method = st.sidebar.selectbox(
    "Grayscale Conversion",
    ["Average", "Luminosity"]
)

gray_size = st.sidebar.selectbox(
    "Grayscale Matrix Size",
    [8, 16, 32]
)

# Pooling
pool_type = st.sidebar.selectbox(
    "Pooling Type",
    ["max", "min", "avg"]
)

pool_h = st.sidebar.selectbox("Pooling Height", [2, 3, 4])
pool_w = st.sidebar.selectbox("Pooling Width", [2, 3, 4])

overlap = st.sidebar.checkbox("Overlapping Pooling", value=False)
stride = 1 if overlap else st.sidebar.selectbox("Stride", [2, 3, 4])

round_avg = st.sidebar.checkbox("Round Average Pooling", value=True)

# Visualization
st.sidebar.header("🎨 Visualization")
show_numbers = st.sidebar.checkbox("Show Numbers", value=True)
font_size = st.sidebar.slider("Font Size", 5, 14, 7)
text_color = st.sidebar.selectbox("Text Color", ["red", "black", "blue"])
colormap = st.sidebar.selectbox(
    "Color Map",
    ["gray", "viridis", "plasma", "inferno"]
)

# -----------------------------
# IMAGE UPLOAD
# -----------------------------
uploaded_file = st.file_uploader(
    "📤 Upload an image",
    type=["jpg", "jpeg", "png"]
)

# -----------------------------
# POOLING FUNCTION
# -----------------------------
def pooling(matrix, ph, pw, stride, mode):
    h, w = matrix.shape
    out_h = (h - ph) // stride + 1
    out_w = (w - pw) // stride + 1
    pooled = np.zeros((out_h, out_w))

    for i in range(out_h):
        for j in range(out_w):
            window = matrix[
                i*stride:i*stride+ph,
                j*stride:j*stride+pw
            ]
            if mode == "max":
                pooled[i, j] = np.max(window)
            elif mode == "min":
                pooled[i, j] = np.min(window)
            elif mode == "avg":
                pooled[i, j] = np.mean(window)

    return pooled

# -----------------------------
# DISPLAY MATRIX
# -----------------------------
def show_matrix(matrix, title):
    fig, ax = plt.subplots()
    ax.imshow(matrix, cmap=colormap)
    ax.set_title(title)
    ax.axis("off")

    if show_numbers:
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                ax.text(
                    j, i, int(matrix[i, j]),
                    ha="center", va="center",
                    color=text_color, fontsize=font_size
                )
    return fig

# -----------------------------
# MAIN LOGIC
# -----------------------------
if uploaded_file:

    img = Image.open(uploaded_file).convert("RGB")
    st.subheader("📸 Original Image")
    st.image(img, use_column_width=True)

    img_array = np.array(img)

    # Grayscale conversion
    if gray_method == "Average":
        gray = img_array.mean(axis=2)
    else:
        gray = (
            0.299 * img_array[:, :, 0] +
            0.587 * img_array[:, :, 1] +
            0.114 * img_array[:, :, 2]
        )

    gray = gray.astype(np.uint8)

    gray_matrix = np.array(
        Image.fromarray(gray).resize((gray_size, gray_size))
    )

    pooled_matrix = pooling(
        gray_matrix, pool_h, pool_w, stride, pool_type
    )

    if pool_type == "avg" and round_avg:
        pooled_matrix = np.round(pooled_matrix)

    # Layout
    col1, col2 = st.columns(2)

    with col1:
        st.subheader(f"Grayscale Matrix ({gray_size}×{gray_size})")
        st.pyplot(show_matrix(gray_matrix, "Grayscale Matrix"))
        st.dataframe(pd.DataFrame(gray_matrix))

    with col2:
        st.subheader("Pooling Output")
        st.pyplot(show_matrix(pooled_matrix, "Pooled Output"))
        st.dataframe(pd.DataFrame(pooled_matrix))

    # Download
    csv = pd.DataFrame(pooled_matrix).to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download Pooled Output (CSV)",
        csv,
        "pooled_output.csv",
        "text/csv"
    )

else:
    st.info("👆 Upload an image to begin.")
