import streamlit as st
import os
from ai_helper import PDFQuestionAnswerer

def main():
    st.title("📄 PDF Question Answering App")

    # Initialize session state variables
    if "pdf_qa" not in st.session_state:
        st.session_state.pdf_qa = PDFQuestionAnswerer()
    
    # Initialize search history if not exists
    if "search_history" not in st.session_state:
        st.session_state.search_history = []

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
        
        # Search History Section
        st.header("Search History")
        if st.session_state.search_history:
            # Display search history with option to clear
            for idx, (question, answer) in enumerate(st.session_state.search_history, 1):
                with st.expander(f"Search {idx}"):
                    st.write(f"**Question:** {question}")
                    st.write(f"**Answer:** {answer}")
            
            # Clear history button
            if st.button("Clear Search History"):
                st.session_state.search_history = []
                st.experimental_rerun()
        else:
            st.write("No search history yet.")

    question = st.text_input("Ask a question about the PDF")

    if st.button("Get Answer"):
        if not question:
            st.warning("Please enter a question")
        else:
            try:
                with st.spinner("Generating answer..."):
                    answer = st.session_state.pdf_qa.ask_question(question)
                    
                    # Add to search history
                    st.session_state.search_history.append((question, answer))
                    
                    # Display current answer
                    st.markdown("### Answer")
                    st.write(answer)
            except Exception as e:
                st.error(f"Error generating answer: {e}")

if __name__ == "__main__":
    os.makedirs("uploads", exist_ok=True)
    main()