import json
import re
import os
from pathlib import Path
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings


def _load_documents_from_book(book_dir_path: Path) -> list[Document]:
    """
    Loads a book's markdown and metadata, splits it into chunks,
    and enriches each chunk with metadata.

    Args:
        book_dir_path: The path to the directory containing book.md and metadata.json.

    Returns:
        A list of Document objects (chunks) with metadata.
    """
    metadata_path = book_dir_path / 'metadata.json'
    book_path = book_dir_path / 'book.md'

    if not metadata_path.exists() or not book_path.exists():
        print(f"Warning: Missing files in {book_dir_path}. Skipping.")
        return []

    # Load metadata
    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)

    # Create a simplified page-to-chapter map from the TOC
    toc = metadata.get('toc', [])
    page_to_chapter = {entry['page']: entry['title'] for entry in toc if 'page' in entry}

    # Load markdown content
    loader = UnstructuredMarkdownLoader(str(book_path))
    docs = loader.load()

    # Split document into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    # Process chunks to add metadata
    current_page = 1
    current_chapter = "Introduction"  # Default chapter
    for chunk in chunks:
        # Check for page markers like [Page X] in the text
        page_match = re.search(r'\[Page (\d+)\]', chunk.page_content)
        if page_match:
            current_page = int(page_match.group(1))
            # Update chapter if a new page is found in our map
            current_chapter = page_to_chapter.get(current_page, current_chapter)
        
        meta_info = metadata.get('meta', {})
        chunk.metadata = {
            "book_title": meta_info.get('title'),
            "author": meta_info.get('authorList', [None])[0],
            "publisher": meta_info.get('publisher'),
            "release_date": meta_info.get('releaseDate'),
            "asin": meta_info.get('asin'),
            "chapter": current_chapter,
            "page": current_page,
            "source": str(book_path)
        }

    return chunks

def load_all_books(books_base_dir: str = "books") -> list[Document]:
    """
    Loads all books from the specified base directory.

    Args:
        books_base_dir: The directory containing all book subdirectories.

    Returns:
        A list of all document chunks from all books.
    """
    all_chunks = []
    base_path = Path(books_base_dir)

    for book_dir in base_path.iterdir():
        if book_dir.is_dir():
            print(f"Processing book: {book_dir.name}")
            book_chunks = _load_documents_from_book(book_dir)
            all_chunks.extend(book_chunks)
    
    return all_chunks

def embed_and_store(
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
    print("Initializing embeddings model...")
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)

    print(f"Creating and persisting vector store at '{persist_directory}'...")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    vectorstore.persist()
    print("Vector store created and persisted successfully.")


if __name__ == '__main__':
    # Example of how to run the ingestion
    print("Starting book ingestion...")
    documents = load_all_books()
    
    if documents:
        print(f"Loaded a total of {len(documents)} document chunks.")
        embed_and_store(documents)
        print("\n--- Sample Chunk ---")
        print(documents[0].page_content)
        print("\n--- Metadata ---")
        print(documents[0].metadata)
        print("--------------------")
        print("\nIngestion and embedding complete.")
    else:
        print("No documents were loaded. Aborting.")
