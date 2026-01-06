import streamlit as st
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="Image Pooling Playground", layout="wide")
st.title("🧠 Interactive Image Pooling Playground")

st.write(
    "Upload an image, choose grayscale size, pooling type, pooling window, and stride "
    "to visualize CNN pooling operations step by step."
)

# -----------------------------
# SIDEBAR CONTROLS
# -----------------------------
st.sidebar.header("⚙️ Controls")

gray_size = st.sidebar.selectbox(
    "Grayscale Matrix Size",
    [8, 16, 32]
)

pool_type = st.sidebar.selectbox(
    "Pooling Type",
    ["max", "min", "avg"]
)

pool_h = st.sidebar.selectbox(
    "Pooling Height",
    [2, 3, 4]
)

pool_w = st.sidebar.selectbox(
    "Pooling Width",
    [2, 3, 4]
)

stride = st.sidebar.selectbox(
    "Stride",
    [1, 2, 3]
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
# DISPLAY MATRIX WITH NUMBERS
# -----------------------------
def show_matrix(matrix, title, font_size):
    fig, ax = plt.subplots()
    ax.imshow(matrix, cmap="gray")
    ax.set_title(title)
    ax.axis("off")

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(
                j, i, int(matrix[i, j]),
                ha="center", va="center",
                color="red", fontsize=font_size
            )
    return fig

# -----------------------------
# MAIN LOGIC
# -----------------------------
if uploaded_file is not None:

    # Read and show original image
    img = Image.open(uploaded_file).convert("RGB")
    st.subheader("📸 Original Image")
    st.image(img, use_column_width=True)

    # Convert to grayscale
    img_array = np.array(img)
    gray = img_array.mean(axis=2).astype(np.uint8)

    # Resize grayscale matrix
    gray_matrix = np.array(
        Image.fromarray(gray).resize((gray_size, gray_size))
    )

    # Apply pooling
    pooled_matrix = pooling(
        gray_matrix,
        pool_h,
        pool_w,
        stride,
        pool_type
    )

    # Layout
    col1, col2 = st.columns(2)

    with col1:
        st.subheader(f"🧩 Grayscale Matrix ({gray_size}×{gray_size})")
        fig1 = show_matrix(gray_matrix, "Grayscale Matrix", 6)
        st.pyplot(fig1)

    with col2:
        st.subheader(
            f"🔽 {pool_type.upper()} Pooling Output ({pool_h}×{pool_w})"
        )
        fig2 = show_matrix(pooled_matrix, "Pooled Output", 8)
        st.pyplot(fig2)

    # Info
    st.markdown("---")
    st.subheader("ℹ️ Details")
    st.write(f"""
    • **Grayscale Size:** {gray_size} × {gray_size}  
    • **Pooling Type:** {pool_type.upper()}  
    • **Pooling Window:** {pool_h} × {pool_w}  
    • **Stride:** {stride}  
    • **Output Size:** {pooled_matrix.shape[0]} × {pooled_matrix.shape[1]}
    """)

else:
    st.info("👆 Please upload an image to start.")
