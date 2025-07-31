from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

def create_and_persist_store(
    chunks: list[Document], 
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    persist_directory: str = "db"
):
    """
    Embeds document chunks and stores them in a Chroma vector store.

    Args:
        chunks: A list of Document objects to embed and store.
        embedding_model_name: The name of the Hugging Face model to use for embeddings.
        persist_directory: The directory to persist the vector store to.
    """
    print("Initializing embeddings model... (This may take a moment for the first run as the model is downloaded)")
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)

    print(f"Creating and persisting vector store at '{persist_directory}'...")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    print("Vector store created successfully.")
