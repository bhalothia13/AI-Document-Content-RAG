import os
import tempfile

import streamlit as st

from document_processor import process_pdf

from rag_pipeline import (
    create_vectorstore,
    load_llm,
    ask_question
)


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="AI Document Intelligence",
    page_icon="📄",
    layout="wide"
)


# =========================================================
# TITLE
# =========================================================

st.title("📄 AI-Powered Document Intelligence System")

st.markdown(
    """
    Upload a PDF document and ask questions about its content.

    **Powered by:** Hugging Face • LangChain • FAISS • RAG
    """
)


# =========================================================
# SESSION STATE
# =========================================================

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "llm" not in st.session_state:
    st.session_state.llm = None

if "messages" not in st.session_state:
    st.session_state.messages = []

if "document_processed" not in st.session_state:
    st.session_state.document_processed = False

if "file_name" not in st.session_state:
    st.session_state.file_name = None


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.header("📚 Document Upload")

    uploaded_file = st.file_uploader(
        "Upload your PDF",
        type=["pdf"]
    )

    if uploaded_file is not None:

        st.info(
            f"Selected: {uploaded_file.name}"
        )

        if st.button(
            "🚀 Process Document",
            use_container_width=True
        ):

            try:

                # =========================================
                # CREATE TEMPORARY PDF FILE
                # =========================================

                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=".pdf"
                ) as temp_file:

                    temp_file.write(
                        uploaded_file.getbuffer()
                    )

                    temp_pdf_path = temp_file.name


                # =========================================
                # PROCESS PDF
                # =========================================

                with st.spinner(
                    "📄 Loading and splitting document..."
                ):

                    chunks = process_pdf(
                        temp_pdf_path
                    )


                st.write(
                    f"📑 Total chunks: {len(chunks)}"
                )


                # =========================================
                # CREATE VECTOR DATABASE
                # =========================================

                with st.spinner(
                    "🔎 Creating embeddings and FAISS database..."
                ):

                    vectorstore = create_vectorstore(
                        chunks
                    )


                st.success(
                    "✅ Vector database created!"
                )


                # =========================================
                # LOAD LLM
                # =========================================

                if st.session_state.llm is None:

                    with st.spinner(
                        "🤖 Loading Hugging Face model..."
                    ):

                        llm = load_llm()

                    st.session_state.llm = llm


                # =========================================
                # SAVE SESSION DATA
                # =========================================

                st.session_state.vectorstore = vectorstore

                st.session_state.document_processed = True

                st.session_state.file_name = uploaded_file.name

                st.session_state.messages = []


                st.success(
                    "🎉 Document processed successfully!"
                )


            except Exception as e:

                st.error(
                    f"❌ Error: {e}"
                )


            finally:

                # =========================================
                # DELETE TEMP FILE
                # =========================================

                if (
                    "temp_pdf_path" in locals()
                    and os.path.exists(temp_pdf_path)
                ):

                    os.remove(
                        temp_pdf_path
                    )


    # =====================================================
    # DOCUMENT STATUS
    # =====================================================

    st.divider()

    if st.session_state.document_processed:

        st.success(
            f"📄 Ready: {st.session_state.file_name}"
        )

    else:

        st.info(
            "No document processed yet."
        )


    # =====================================================
    # CLEAR CHAT
    # =====================================================

    if st.button(
        "🗑️ Clear Chat",
        use_container_width=True
    ):

        st.session_state.messages = []

        st.rerun()


# =========================================================
# MAIN AREA
# =========================================================

if not st.session_state.document_processed:

    st.info(
        "👈 Upload a PDF from the sidebar and click "
        "**Process Document**."
    )

else:

    st.success(
        f"✅ **{st.session_state.file_name}** is ready. "
        "Ask questions below."
    )


# =========================================================
# CHAT HISTORY
# =========================================================

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )


# =========================================================
# CHAT INPUT
# =========================================================

question = st.chat_input(
    "Ask a question about your document..."
)


if question:

    # =============================================
    # CHECK VECTORSTORE
    # =============================================

    if st.session_state.vectorstore is None:

        st.warning(
            "Please process a PDF first."
        )

        st.stop()


    # =============================================
    # USER MESSAGE
    # =============================================

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )


    with st.chat_message("user"):

        st.markdown(
            question
        )


    # =============================================
    # AI RESPONSE
    # =============================================

    with st.chat_message("assistant"):

        with st.spinner(
            "🔎 Searching your document..."
        ):

            try:

                response, docs = ask_question(
                    st.session_state.vectorstore,
                    st.session_state.llm,
                    question
                )


                # =====================================
                # ANSWER
                # =====================================

                st.markdown(
                    response
                )


                # =====================================
                # SOURCES
                # =====================================

                st.markdown(
                    "### 📚 Sources"
                )


                seen_sources = set()


                for doc in docs:

                    source = doc.metadata.get(
                        "source",
                        "Unknown"
                    )

                    page = doc.metadata.get(
                        "page",
                        None
                    )


                    if isinstance(page, int):

                        page_number = page + 1

                    else:

                        page_number = page


                    source_name = os.path.basename(
                        source
                    )


                    source_key = (
                        source_name,
                        page_number
                    )


                    if source_key not in seen_sources:

                        st.write(
                            f"📄 **{source_name}** "
                            f"— Page **{page_number}**"
                        )


                        seen_sources.add(
                            source_key
                        )


                # =====================================
                # SAVE AI RESPONSE
                # =====================================

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": response
                    }
                )


            except Exception as e:

                st.error(
                    f"❌ Error generating answer: {e}"
                )