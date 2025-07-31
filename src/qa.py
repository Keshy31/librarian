from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Configuration
VECTOR_DB_PATH = "db"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "gemma3:4b"

PROMPT_TEMPLATE = """
Use the following context to answer the question. Your answer should be grounded in the text provided.

Include citations with the book title, chapter, and page number at the end of each relevant sentence or paragraph.
For example: The author argues for a focus on fundamentals (The Art of Doing Science and Engineering, Chapter 5, Page 23).

If you are asked to retrieve a direct quote, provide it and its citation.
If the context does not contain the answer, state that you cannot answer the question based on the provided text.

Context: {context}

Question: {question}

Answer:
"""

class QAChain:
    def __init__(self):
        print("Initializing QA Chain...")
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
        
        print(f"Loading vector store from '{VECTOR_DB_PATH}'...")
        vector_store = Chroma(
            persist_directory=VECTOR_DB_PATH, 
            embedding_function=embeddings
        )
        
        llm = Ollama(model=LLM_MODEL)
        
        prompt = PromptTemplate(
            template=PROMPT_TEMPLATE, 
            input_variables=["context", "question"]
        )
        
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vector_store.as_retriever(search_kwargs={"k": 5}),
            chain_type_kwargs={"prompt": prompt},
            return_source_documents=True
        )
        print("QA Chain initialized successfully.")

    def ask(self, query: str) -> dict:
        """
        Asks a question to the QA chain.

        Args:
            query: The question to ask.

        Returns:
            A dictionary containing the query, result, and source documents.
        """
        print(f"Received query: {query}")
        result = self.qa_chain.invoke({"query": query})
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
    print(response.get('result'))
    print("\n--- Source Documents ---")
    for doc in response.get('source_documents', []):
        print(f"- Source: {doc.metadata.get('source')}, Page: {doc.metadata.get('page')}")
        # print(f"  Content: {doc.page_content[:100]}...") # Uncomment to see content
