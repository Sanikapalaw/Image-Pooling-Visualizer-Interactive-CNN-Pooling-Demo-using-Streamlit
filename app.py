import streamlit as st
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd

# =============================
# PAGE CONFIG
# =============================
st.set_page_config(
    page_title="VisionPool Studio",
    layout="wide"
)

# =============================
# HEADER
# =============================
st.markdown(
    """
    <h1 style='text-align:center;'>🎨 VisionPool Studio</h1>
    <h4 style='text-align:center; color:gray;'>
    Interactive CNN Pooling Visualizer
    </h4>
    <hr>
    """,
    unsafe_allow_html=True
)

st.write(
    "Upload an image, choose grayscale size, pooling window, pooling type, and stride "
    "to visually understand CNN pooling."
)

# =============================
# SIDEBAR CONTROLS
# =============================
st.sidebar.title("🧭 Control Panel")

# -------- Matrix settings --------
st.sidebar.subheader("1️⃣ Grayscale Settings")
gray_size = st.sidebar.radio(
    "Grayscale Matrix Size",
    [8, 16, 32],
    horizontal=True
)

gray_method = st.sidebar.selectbox(
    "Grayscale Conversion",
    ["Average", "Luminosity"]
)

# -------- Pooling settings --------
st.sidebar.subheader("2️⃣ Pooling Settings")

pool_type = st.sidebar.radio(
    "Pooling Type",
    ["max", "min", "avg"],
    horizontal=True
)

pool_window = st.sidebar.selectbox(
    "Pooling Window Size",
    ["2x2", "2x3", "3x2", "3x3", "3x4", "4x3", "4x4"]
)

stride = st.sidebar.slider(
    "Stride",
    min_value=1,
    max_value=10,
    value=2
)

# -------- Display settings --------
st.sidebar.subheader("3️⃣ Display Settings")
show_numbers = st.sidebar.checkbox("Show Numbers", True)
font_size = st.sidebar.slider("Font Size", 5, 14, 7)
colormap = st.sidebar.selectbox(
    "Color Map",
    ["gray", "viridis", "plasma", "inferno"]
)

# =============================
# IMAGE UPLOAD
# =============================
st.markdown("## 📤 Step 1: Upload Image")
uploaded_file = st.file_uploader(
    "Upload JPG / PNG image",
    type=["jpg", "jpeg", "png"]
)

# =============================
# FUNCTIONS
# =============================
def pooling(matrix, ph, pw, stride, mode):
    h, w = matrix.shape
    out_h = (h - ph) // stride + 1
    out_w = (w - pw) // stride + 1

    if out_h <= 0 or out_w <= 0:
        return None

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
                    fontsize=font_size,
                    color="red"
                )
    return fig

# =============================
# MAIN LOGIC
# =============================
if uploaded_file:

    img = Image.open(uploaded_file).convert("RGB")

    st.markdown("## 🖼️ Step 2: Original Image")
    st.image(img, use_column_width=True)

    img_array = np.array(img)

    # Grayscale
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

    # Parse pooling window
    ph, pw = map(int, pool_window.split("x"))

    pooled_matrix = pooling(
        gray_matrix, ph, pw, stride, pool_type
    )

    st.markdown("## 🧩 Step 3: Results")

    if pooled_matrix is None:
        st.error("❌ Pooling window and stride are too large for the selected matrix.")
    else:
        tab1, tab2 = st.tabs(["Grayscale Matrix", "Pooling Output"])

        with tab1:
            st.pyplot(show_matrix(
                gray_matrix,
                f"Grayscale Matrix ({gray_size}×{gray_size})"
            ))
            st.dataframe(pd.DataFrame(gray_matrix))

        with tab2:
            st.pyplot(show_matrix(
                pooled_matrix,
                f"{pool_type.upper()} Pooling ({pool_window}, stride={stride})"
            ))
            st.dataframe(pd.DataFrame(pooled_matrix))

        csv = pd.DataFrame(pooled_matrix).to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download Pooled Matrix (CSV)",
            csv,
            "pooled_output.csv",
            "text/csv"
        )

else:
    st.info("👆 Upload an image to start.")
