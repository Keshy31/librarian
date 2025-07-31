import streamlit as st
from src.qa import QAChain

st.set_page_config(page_title="Librarian", page_icon="📚", layout="wide")

st.title("📚 Librarian: Your Ebook Q&A Assistant")

# Cache the QA chain to avoid re-initializing on every interaction
@st.cache_resource
def load_qa_chain():
    return QAChain()

qa_chain = load_qa_chain()

# Get available books
book_list = ["All Books"] + qa_chain.get_available_books()

# UI Elements
selected_book = st.selectbox("Filter by book:", options=book_list)

# User input
query = st.text_input("Ask a question about your books:", placeholder="e.g., What are the main themes in 'The Art of War'?")

if st.button("Get Answer"):
    if query:
        with st.spinner("Thinking..."):
            try:
                # Pass the selected book to the ask method
                book_filter = None if selected_book == "All Books" else selected_book
                result = qa_chain.ask(query, book_title=book_filter)
                answer = result.get('answer')
                sources = result.get('context', [])

                st.subheader("Answer")
                st.write(answer)

                if sources:
                    st.subheader("Sources")
                    for doc in sources:
                        source_info = f"- **{doc.metadata.get('book_title', 'Unknown Title')}** (Chapter: {doc.metadata.get('chapter', 'N/A')}, Page: {doc.metadata.get('page', 'N/A')})"
                        st.markdown(source_info, unsafe_allow_html=True)

            except Exception as e:
                st.error(f"An error occurred: {e}")
    else:
        st.warning("Please enter a question.")
