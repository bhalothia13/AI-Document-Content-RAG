import streamlit as st
import requests


# ============================================================
# CONFIG
# ============================================================

API_URL = (
    "https://bhalothia13-ai-document-content-git-main-"
    "bhalothia13s-projects.vercel.app/api"
)

st.set_page_config(
    page_title="AI Document Intelligence RAG",
    page_icon="📄",
    layout="centered"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .title {
        text-align: center;
        font-size: 40px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        font-size: 17px;
        opacity: 0.75;
        margin-bottom: 30px;
    }

    .answer-box {
        padding: 20px;
        border-radius: 14px;
        border: 1px solid rgba(128,128,128,0.35);
        margin-top: 15px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# TITLE
# ============================================================

st.markdown(
    '<div class="title">📄 AI Document Intelligence RAG</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Upload a document and ask questions using AI'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# SESSION STATE
# ============================================================

if "uploaded" not in st.session_state:
    st.session_state.uploaded = False

if "filename" not in st.session_state:
    st.session_state.filename = ""


# ============================================================
# HEALTH CHECK
# ============================================================

with st.expander("Backend Status"):

    if st.button("Check Backend"):

        try:

            response = requests.get(
                f"{API_URL}/health",
                timeout=20
            )

            if response.status_code == 200:

                data = response.json()

                st.success(
                    f"Backend online: {data.get('status', 'OK')}"
                )

            else:

                st.error(
                    f"Backend returned {response.status_code}"
                )

        except Exception as e:

            st.error(
                f"Backend connection failed: {e}"
            )


# ============================================================
# DOCUMENT UPLOAD
# ============================================================

st.header("1. 📤 Document Upload")

uploaded_file = st.file_uploader(
    "Upload PDF or TXT",
    type=["pdf", "txt"]
)


if uploaded_file:

    if st.button(
        "📤 Upload Document",
        use_container_width=True
    ):

        with st.spinner(
            "Processing and indexing document..."
        ):

            try:

                files = {
                    "file": (
                        uploaded_file.name,
                        uploaded_file.getvalue(),
                        uploaded_file.type
                    )
                }

                response = requests.post(
                    f"{API_URL}/upload",
                    files=files,
                    timeout=120
                )

                # --------------------------------------------
                # HTML RESPONSE PROTECTION
                # --------------------------------------------

                content_type = response.headers.get(
                    "content-type",
                    ""
                ).lower()

                if "text/html" in content_type:

                    st.error(
                        "❌ Backend returned HTML instead of JSON."
                    )

                    st.code(
                        response.text[:1000],
                        language="html"
                    )

                elif response.status_code == 200:

                    data = response.json()

                    st.session_state.uploaded = True
                    st.session_state.filename = (
                        uploaded_file.name
                    )

                    st.success(
                        data.get(
                            "message",
                            "Document uploaded successfully!"
                        )
                    )

                    if data.get("chunks") is not None:

                        st.info(
                            f"📚 Created "
                            f"{data['chunks']} text chunks."
                        )

                else:

                    try:
                        error_data = response.json()
                        error_message = error_data.get(
                            "detail",
                            str(error_data)
                        )
                    except Exception:
                        error_message = response.text[:500]

                    st.error(
                        f"❌ Upload failed "
                        f"({response.status_code}): "
                        f"{error_message}"
                    )

            except requests.exceptions.Timeout:

                st.error(
                    "⏱️ Upload timed out. "
                    "The document may be too large."
                )

            except requests.exceptions.RequestException as e:

                st.error(
                    f"❌ Connection error: {e}"
                )

            except Exception as e:

                st.error(
                    f"❌ Unexpected error: {e}"
                )


# ============================================================
# ACTIVE DOCUMENT
# ============================================================

if st.session_state.uploaded:

    st.success(
        f"📄 Active Document: "
        f"**{st.session_state.filename}**"
    )


# ============================================================
# QUESTION
# ============================================================

st.header("2. 🤖 Ask Question")

question = st.text_area(
    "Enter your question:",
    placeholder=(
        "Example: What is the main topic of this document?"
    ),
    height=100
)


if st.button(
    "🔍 Submit Question",
    use_container_width=True
):

    if not st.session_state.uploaded:

        st.warning(
            "⚠️ Please upload a document first."
        )

    elif not question.strip():

        st.warning(
            "⚠️ Please enter a question."
        )

    else:

        with st.spinner(
            "🔎 Searching document and generating answer..."
        ):

            try:

                response = requests.post(
                    f"{API_URL}/chat",
                    json={
                        "question": question.strip()
                    },
                    timeout=120
                )

                # --------------------------------------------
                # HTML RESPONSE PROTECTION
                # --------------------------------------------

                content_type = response.headers.get(
                    "content-type",
                    ""
                ).lower()

                if "text/html" in content_type:

                    st.error(
                        "❌ Vercel returned an HTML page "
                        "instead of the FastAPI response."
                    )

                    st.code(
                        response.text[:1000],
                        language="html"
                    )

                elif response.status_code == 200:

                    data = response.json()

                    answer = data.get(
                        "answer",
                        "No answer returned."
                    )

                    st.subheader("💡 Answer")

                    st.markdown(
                        f"""
                        <div class="answer-box">
                        {answer}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    # ----------------------------------------
                    # SOURCES
                    # ----------------------------------------

                    sources = data.get(
                        "sources",
                        []
                    )

                    if sources:

                        st.subheader("📚 Sources")

                        for source in sources:

                            page = source.get(
                                "page",
                                "Unknown"
                            )

                            preview = source.get(
                                "preview",
                                ""
                            )

                            st.markdown(
                                f"""
                                **Page:** {page}

                                > {preview}
                                """
                            )

                else:

                    try:

                        error_data = response.json()

                        error_message = error_data.get(
                            "detail",
                            str(error_data)
                        )

                    except Exception:

                        error_message = response.text[:500]

                    st.error(
                        f"❌ Backend Error "
                        f"({response.status_code})"
                    )

                    st.code(
                        error_message
                    )

            except requests.exceptions.Timeout:

                st.error(
                    "⏱️ Request timed out. "
                    "Please try again."
                )

            except requests.exceptions.RequestException as e:

                st.error(
                    f"❌ Connection error: {e}"
                )

            except Exception as e:

                st.error(
                    f"❌ Unexpected error: {e}"
                )


# ============================================================
# EXAMPLES
# ============================================================

st.divider()

st.subheader("💡 Example Questions")

examples = [
    "What is the main topic of this document?",
    "Explain the important concepts mentioned in the document.",
    "What are the key points?",
    "Summarize this document.",
    "What does the document say about deep learning?"
]

for example in examples:

    st.markdown(
        f"- `{example}`"
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Built with Python • FastAPI • RAG • "
    "Hugging Face • Streamlit"
)
