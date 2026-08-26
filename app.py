import streamlit as st
import requests


# --------------------------------------------------
# BACKEND URL
# --------------------------------------------------

BASE_URL = st.secrets.get(
    BACKEND_URL = "https://ai-document-content-ragss.vercel.app"
).rstrip("/")


# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="AI Document Intelligence RAG",
    page_icon="📄",
    layout="centered"
)


# --------------------------------------------------
# CSS
# --------------------------------------------------

st.markdown(
    """
    <style>

    .main-title {
        text-align: center;
        font-size: 42px;
        font-weight: 700;
    }

    .subtitle {
        text-align: center;
        font-size: 18px;
        margin-bottom: 30px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# --------------------------------------------------
# TITLE
# --------------------------------------------------

st.markdown(
    '<div class="main-title">📄 AI Document Intelligence RAG</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Upload a document and ask questions using AI'
    '</div>',
    unsafe_allow_html=True
)


# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------

if "uploaded" not in st.session_state:
    st.session_state.uploaded = False

if "filename" not in st.session_state:
    st.session_state.filename = ""


# --------------------------------------------------
# BACKEND STATUS
# --------------------------------------------------

with st.expander(
    "🔌 Backend Status",
    expanded=True
):

    if st.button("Check Backend"):

        try:

            response = requests.get(
                f"{BASE_URL}/health",
                timeout=20
            )

            if response.status_code == 200:

                data = response.json()

                st.success(
                    "Backend connected successfully."
                )

                st.json(data)

            else:

                st.error(
                    f"Backend returned HTTP "
                    f"{response.status_code}"
                )

                st.code(
                    response.text[:1000]
                )

        except Exception as e:

            st.error(
                f"Backend connection failed: {e}"
            )


# --------------------------------------------------
# DOCUMENT UPLOAD
# --------------------------------------------------

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
                    f"{BASE_URL}/upload",
                    files=files,
                    timeout=120
                )

                content_type = response.headers.get(
                    "content-type",
                    ""
                )

                if "application/json" not in content_type:

                    st.error(
                        "Backend returned HTML instead of JSON."
                    )

                    st.code(
                        response.text[:1000]
                    )

                elif response.status_code == 200:

                    data = response.json()

                    st.session_state.uploaded = True

                    st.session_state.filename = (
                        data.get(
                            "filename",
                            uploaded_file.name
                        )
                    )

                    st.success(
                        "Document uploaded and indexed successfully!"
                    )

                    st.write(
                        f"Chunks created: "
                        f"{data.get('chunks', 'N/A')}"
                    )

                else:

                    try:
                        data = response.json()

                        st.error(
                            data.get(
                                "error",
                                "Upload failed."
                            )
                        )

                    except Exception:

                        st.error(
                            f"Upload failed: "
                            f"HTTP {response.status_code}"
                        )

            except requests.exceptions.Timeout:

                st.error(
                    "Backend timeout. "
                    "Please try again."
                )

            except Exception as e:

                st.error(
                    f"Connection error: {e}"
                )


# --------------------------------------------------
# ACTIVE DOCUMENT
# --------------------------------------------------

if st.session_state.uploaded:

    st.info(
        f"📄 Active Document: "
        f"**{st.session_state.filename}**"
    )


# --------------------------------------------------
# ASK QUESTION
# --------------------------------------------------

st.header("2. 🤖 Ask Question")

question = st.text_area(
    "Enter your question:",
    placeholder="Example: What is the main topic of this document?"
)


if st.button(
    "🔎 Submit Question",
    use_container_width=True
):

    if not st.session_state.uploaded:

        st.warning(
            "Please upload a document first."
        )

    elif not question.strip():

        st.warning(
            "Please enter a question."
        )

    else:

        with st.spinner(
            "Searching document and generating answer..."
        ):

            try:

                response = requests.post(
                    f"{BASE_URL}/chat",
                    data={
                        "question": question
                    },
                    timeout=120
                )

                content_type = response.headers.get(
                    "content-type",
                    ""
                )

                if "application/json" not in content_type:

                    st.error(
                        "Backend returned HTML instead of JSON."
                    )

                    st.code(
                        response.text[:1000]
                    )

                elif response.status_code == 200:

                    data = response.json()

                    st.subheader("Answer")

                    st.write(
                        data.get(
                            "answer",
                            "No answer returned."
                        )
                    )

                    sources = data.get(
                        "sources",
                        []
                    )

                    if sources:

                        st.subheader("Sources")

                        pages = sorted(
                            set(
                                str(
                                    source.get("page")
                                )
                                for source in sources
                                if source.get("page") is not None
                            )
                        )

                        if pages:

                            st.write(
                                "Pages: "
                                + ", ".join(pages)
                            )

                else:

                    try:

                        data = response.json()

                        st.error(
                            data.get(
                                "error",
                                "Backend error."
                            )
                        )

                    except Exception:

                        st.error(
                            f"Backend Error: "
                            f"HTTP {response.status_code}"
                        )

            except requests.exceptions.Timeout:

                st.error(
                    "Request timed out. "
                    "Please try again."
                )

            except Exception as e:

                st.error(
                    f"Connection error: {e}"
                )


# --------------------------------------------------
# EXAMPLES
# --------------------------------------------------

st.divider()

st.subheader("💡 Example Questions")

st.markdown(
    """
- What is the main topic of this document?
- What are the important concepts?
- Explain the first chapter.
- What does the document say about machine learning?
- Give me a summary of this document.
"""
)


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "Built with Python • FastAPI • RAG • "
    "FAISS • Hugging Face • Streamlit"
)
