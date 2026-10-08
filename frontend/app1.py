import streamlit as st
import requests
from PIL import Image

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="MDC-Net | Steel Defect Intelligence",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

API_URL = "http://127.0.0.1:8000/predict"


# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------

st.markdown("""
<style>

    /* Main background */
    .stApp {
        background-color: #0e1117;
    }

    /* Header */
    .main-header {
        padding: 25px 0 10px 0;
    }

    .main-title {
        font-size: 38px;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 16px;
        color: #9ca3af;
        margin-bottom: 25px;
    }

    /* Cards */
    .card {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 14px;
        padding: 22px;
        margin-bottom: 15px;
    }

    .card-title {
        color: #9ca3af;
        font-size: 14px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        margin-bottom: 8px;
    }

    .card-value {
        color: #ffffff;
        font-size: 25px;
        font-weight: 700;
    }

    /* Prediction card */
    .prediction-card {
        background: linear-gradient(
            135deg,
            #172554,
            #111827
        );
        border: 1px solid #2563eb;
        border-radius: 16px;
        padding: 25px;
        margin-top: 10px;
    }

    .prediction-label {
        color: #93c5fd;
        font-size: 14px;
        font-weight: 600;
        text-transform: uppercase;
    }

    .prediction-value {
        color: #ffffff;
        font-size: 32px;
        font-weight: 700;
        margin-top: 5px;
    }

    /* Caption */
    .caption-box {
        background-color: #111827;
        border-left: 4px solid #3b82f6;
        padding: 18px;
        border-radius: 8px;
        color: #d1d5db;
        font-size: 16px;
        line-height: 1.6;
    }

    /* Section headings */
    .section-title {
        color: #ffffff;
        font-size: 22px;
        font-weight: 650;
        margin-top: 20px;
        margin-bottom: 12px;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #0b0f14;
        border-right: 1px solid #30363d;
    }

    /* Buttons */
    .stButton > button {
        width: 100%;
        border-radius: 10px;
        height: 48px;
        font-size: 16px;
        font-weight: 600;
    }

    /* Upload box */
    [data-testid="stFileUploader"] {
        background-color: #161b22;
        border: 1px dashed #4b5563;
        border-radius: 14px;
        padding: 10px;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #6b7280;
        font-size: 13px;
        padding: 30px 0 10px 0;
    }

</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:

    st.markdown("## 🔬 MDC-Net")

    st.markdown(
        """
        **Multimodal Defect Captioning Network**

        AI-powered steel surface defect analysis using computer vision and deep learning.
        """
    )

    st.divider()

    st.markdown("### Supported Defects")

    defects = [
        "Crazing",
        "Inclusion",
        "Patches",
        "Pitted Surface",
        "Rolled-in Scale",
        "Scratches"
    ]

    for defect in defects:
        st.markdown(f"• {defect}")

    st.divider()

    st.markdown("### System")

    st.caption("Backend: FastAPI")
    st.caption("Frontend: Streamlit")
    st.caption("Model: MDC-Net")
    st.caption("Dataset: NEU-DET")


# --------------------------------------------------
# HEADER
# --------------------------------------------------

    st.title("🔬 Steel Surface Intelligence")

    st.caption(
        "MDC-Net based multimodal steel surface defect detection "
        "and caption generation"
    )


# --------------------------------------------------
# UPLOAD SECTION
# --------------------------------------------------

st.markdown(
    '<div class="section-title">Upload Steel Surface Image</div>',
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed"
)


# --------------------------------------------------
# IMAGE + PREDICTION
# --------------------------------------------------

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    left, right = st.columns([1.1, 1], gap="large")

    # ----------------------------------------------
    # IMAGE PREVIEW
    # ----------------------------------------------

    with left:

        st.markdown(
            '<div class="section-title">Image Preview</div>',
            unsafe_allow_html=True
        )

        st.image(
            image,
            caption=uploaded_file.name,
            use_container_width=True
        )


    # ----------------------------------------------
    # PREDICTION PANEL
    # ----------------------------------------------

    with right:

        st.markdown(
            '<div class="section-title">Defect Analysis</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            """
            <div class="card">
                <div class="card-title">Analysis Status</div>
                <div class="card-value">
                    Ready for Prediction
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        predict_button = st.button(
            "🔍 Analyze Steel Surface",
            type="primary"
        )

        if predict_button:

            with st.spinner("MDC-Net is analyzing the image..."):

                try:

                    files = {
                        "file": (
                            uploaded_file.name,
                            uploaded_file.getvalue(),
                            uploaded_file.type
                        )
                    }

                    response = requests.post(
                        API_URL,
                        files=files,
                        timeout=120
                    )

                    if response.status_code == 200:

                        result = response.json()

                        # Store result
                        st.session_state["prediction"] = result

                    else:

                        st.error(
                            f"Prediction failed: HTTP {response.status_code}"
                        )

                        st.code(response.text)

                except requests.exceptions.ConnectionError:

                    st.error(
                        "❌ Could not connect to the MDC-Net API. "
                        "Make sure FastAPI is running on port 8000."
                    )

                except requests.exceptions.Timeout:

                    st.error(
                        "⏱️ Prediction timed out. "
                        "The model may still be loading."
                    )

                except Exception as e:

                    st.error(f"Unexpected error: {str(e)}")


# --------------------------------------------------
# RESULTS
# --------------------------------------------------

if "prediction" in st.session_state:

    result = st.session_state["prediction"]

    st.divider()

    st.markdown(
        '<div class="section-title">Prediction Results</div>',
        unsafe_allow_html=True
    )

    # Get values safely
    defect = result.get("defect", "Unknown")
    location = result.get("location", "Not available")
    caption = result.get(
        "caption",
        "No caption generated."
    )

    confidence = result.get("confidence", None)

    # ----------------------------------------------
    # MAIN RESULT
    # ----------------------------------------------

    st.markdown("### 🔍 Detection Result")

    st.success(
        f"Detected Defect: {defect.replace('_', ' ').title()}"
    )

    st.write("")

    # ----------------------------------------------
    # DETAILS
    # ----------------------------------------------
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            label="📍 Defect Location",
            value=location.replace("_", " ").title()
        )
    with col2:
        if confidence is not None:
            try:
                confidence_display = f"{float(confidence) * 100:.2f}%"
            except:
                confidence_display = str(confidence)
        else:
            confidence_display = "N/A"
        st.metric(
            label="🎯 Confidence",
            value=confidence_display
        )
    with col3:
        st.metric(
            label="🧠 Model",
            value="MDC-Net"
        )

    # ----------------------------------------------
    # GENERATED CAPTION
    # ----------------------------------------------

    st.markdown(
        '<div class="section-title">Generated Explanation</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        f"""
        <div class="caption-box">
            💬 {caption}
        </div>
        """,
        unsafe_allow_html=True
    )


    # ----------------------------------------------
    # RAW RESPONSE - OPTIONAL
    # ----------------------------------------------

    with st.expander("View API Response"):

        st.json(result)


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.markdown(
    """
    <div class="footer">
        MDC-Net • Multimodal Steel Surface Defect Analysis
        <br>
        AI & Data Science Project
    </div>
    """,
    unsafe_allow_html=True
)
