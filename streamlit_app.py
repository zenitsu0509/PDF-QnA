import streamlit as st
import os
from ai_helper import PDFQuestionAnswerer

def main():
    st.title("📄 PDF Question Answering App")

    if "pdf_qa" not in st.session_state:
        st.session_state.pdf_qa = PDFQuestionAnswerer()

    with st.sidebar:
        st.header("Upload PDF")
        uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

        if uploaded_file is not None:
            with open(os.path.join("uploads", uploaded_file.name), "wb") as f:
                f.write(uploaded_file.getbuffer())

            if st.button("Process PDF"):
                with st.spinner("Processing PDF..."):
                    try:
                        num_chunks = st.session_state.pdf_qa.load_and_process_pdf(
                            os.path.join("uploads", uploaded_file.name)
                        )
                        st.success(f"PDF processed successfully with {num_chunks} chunks!")
                    except Exception as e:
                        st.error(f"Error processing PDF: {e}")

    question = st.text_input("Ask a question about the PDF")

    if st.button("Get Answer"):
        if not question:
            st.warning("Please enter a question")
        else:
            try:
                with st.spinner("Generating answer..."):
                    answer = st.session_state.pdf_qa.ask_question(question)
                    st.markdown("### Answer")
                    st.write(answer)
            except Exception as e:
                st.error(f"Error generating answer: {e}")

if __name__ == "__main__":
    os.makedirs("uploads", exist_ok=True)
    main()
