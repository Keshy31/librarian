from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

# Configuration
VECTOR_DB_PATH = "db"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "gemma3:4b"

# This prompt formats each individual document.
DOCUMENT_PROMPT_TEMPLATE = """
---
Source: {book_title}, Chapter: {chapter}, Page: {page}

Content:
{page_content}
---
"""

# This prompt combines the documents and the question.
PROMPT_TEMPLATE = """
Use the following context to answer the question. Your answer must be grounded in the text provided.

Include a citation for each piece of information you use, with the book title, chapter, and page number.
For example: The author argues for a focus on fundamentals (The Art of Doing Science and Engineering, Chapter 5, Page 23).

If the context does not contain the answer, state that you cannot answer the question based on the provided text.

Context:
{context}

Question: {input}

Answer:
"""

class QAChain:
    def __init__(self):
        print("Initializing QA Chain...")
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
        
        print(f"Loading vector store from '{VECTOR_DB_PATH}'...")
        self.vector_store = Chroma(
            persist_directory=VECTOR_DB_PATH, 
            embedding_function=embeddings
        )
        
        llm = OllamaLLM(model=LLM_MODEL)
        
        # Create a prompt for formatting each document
        document_prompt = PromptTemplate.from_template(DOCUMENT_PROMPT_TEMPLATE)

        # Create a prompt for the final question-answering
        qa_prompt = PromptTemplate.from_template(PROMPT_TEMPLATE)
        
        # Create a chain that stuffs the formatted documents into the final prompt
        stuff_chain = create_stuff_documents_chain(llm, qa_prompt, document_prompt=document_prompt)
        
        # Create the retrieval chain
        self.retrieval_chain = create_retrieval_chain(
            retriever=self.vector_store.as_retriever(search_kwargs={"k": 5}),
            combine_docs_chain=stuff_chain
        )

        print("QA Chain initialized successfully.")

    def get_available_books(self) -> list[str]:
        """
        Gets a list of available book titles from the vector store.

        Returns:
            A sorted list of unique book titles.
        """
        # The .get() method without IDs or where filter fetches all documents
        all_docs = self.vector_store.get()
        all_metadata = all_docs.get('metadatas', [])
        
        book_titles = {meta['book_title'] for meta in all_metadata if 'book_title' in meta}
        
        return sorted(list(book_titles))

    def ask(self, query: str, book_title: str | None = None) -> dict:
        """
        Asks a question to the QA chain.

        Args:
            query: The question to ask.
            book_title: If provided, filters the search to a specific book.

        Returns:
            A dictionary containing the query, answer, and source documents.
        """
        print(f"Received query: {query}")
        
        search_kwargs = {"k": 5}
        if book_title:
            print(f"Filtering by book: {book_title}")
            search_kwargs['filter'] = {'book_title': book_title}
            
        retriever = self.vector_store.as_retriever(
            search_kwargs=search_kwargs
        )
        
        # We need to recreate the retrieval chain with the potentially updated retriever
        # This is a bit inefficient, but necessary with the current structure.
        # A more advanced implementation might cache chains per book.
        llm = OllamaLLM(model=LLM_MODEL)
        document_prompt = PromptTemplate.from_template(DOCUMENT_PROMPT_TEMPLATE)
        qa_prompt = PromptTemplate.from_template(PROMPT_TEMPLATE)
        stuff_chain = create_stuff_documents_chain(llm, qa_prompt, document_prompt=document_prompt)
        
        retrieval_chain = create_retrieval_chain(
            retriever=retriever,
            combine_docs_chain=stuff_chain
        )
        
        result = retrieval_chain.invoke({"input": query})
        return result

if __name__ == '__main__':
    # Example of how to use the QA chain
    qa_system = QAChain()
    
    print("\n--- Ready to answer questions! ---")
    # Example query
    example_query = "What are the author's views on addiction?"
    print(f"Sending example query: {example_query}")
    
    response = qa_system.ask(example_query)
    
    print("\n--- Response ---")
    print(response.get('answer'))
    print("\n--- Source Documents (Context) ---")
    for doc in response.get('context', []):
        print(f"- Source: {doc.metadata.get('source')}, Page: {doc.metadata.get('page')}")
