# Librarian: AI-Powered Ebook Query System

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green)](https://github.com/your-repo/librarian/blob/main/LICENSE)
[![LangChain](https://img.shields.io/badge/powered%20by-LangChain-orange)](https://langchain.com/)

Librarian is a Retrieval-Augmented Generation (RAG) system built with LangChain that allows you to interactively query and chat with your ebook library. It ingests ebooks exported as markdown files (e.g., from tools like [Kindle AI Export](https://github.com/transitive-bullshit/kindle-ai-export)), embeds them into a vector database, and provides a Streamlit-based web interface for asking questions, retrieving specific paragraphs, or summarizing concepts. Responses are grounded in the book content with citations (book title, chapter, page).

This project integrates with ebook extraction tools to handle your Kindle library, enabling features like semantic search across books or filtered queries by title.

## Features
- **Ebook Ingestion**: Load markdown files and metadata JSON from exported ebooks, split into chunks, and enrich with details like book title, chapter, and page.
- **Vector Embedding**: Use Hugging Face embeddings and Chroma for efficient semantic search.
- **Querying**: Ask natural language questions (e.g., "Summarize Hamming's views on style in chapter 1") with grounded responses via a local LLM (Ollama).
- **Filtering**: Query across all books or filter by specific title.
- **Citations**: Responses include precise references (e.g., "The Art of Doing Science and Engineering, Chapter 1, Page 2").
- **UI**: Streamlit web app for an interactive chat experience.
- **Local & Offline**: Runs entirely locally with no external API dependencies beyond initial setup.

## Installation
1. **Prerequisites**:
   - Python 3.10+
   - Ollama installed and running (download from [ollama.com](https://ollama.com); run `ollama run gemma2:2b` or your preferred model like `mistral`).
   - FFmpeg (for potential future audio features; install via `brew install ffmpeg` on macOS or equivalent).

2. **Clone the Repository**:
   ```
   git clone https://github.com/your-repo/librarian.git
   cd librarian
   ```

3. **Install Dependencies**:
   ```
   pip install -r requirements.txt
   ```
   (Create `requirements.txt` with: `langchain langchain-community langchain-chroma langchain-huggingface langchain-ollama streamlit` if not present.)

## Usage
### 1. Ingest Ebooks
Export your Kindle books using [Kindle AI Export](https://github.com/transitive-bullshit/kindle-ai-export) or similar tools. Place exported books in a `books/` directory, each in a subfolder with `book.md` and `metadata.json`.

Run the ingestion script:
```
python ingestion.py
```
- This loads all books, splits into chunks, adds metadata, and persists to a Chroma vector store in `db/`.
- Output: Confirms loaded chunks and sample metadata.

### 2. Run the QA Chain (Console Mode)
For testing:
```
python qa.py
```
- Initializes the chain and runs an example query.
- Outputs the answer and source documents.

### 3. Launch the Streamlit UI
```
streamlit run app.py
```
- Open in your browser (usually http://localhost:8501).
- Select a book (or "All Books"), enter a query, and get responses with sources.
- Example Query: "What are the author's views on addiction?"

## Project Structure
- `ingestion.py`: Loads markdown and metadata, splits documents, and creates the vector store.
- `vector_store.py`: Handles embedding and persistence with Chroma.
- `qa.py`: Defines the QA chain for retrieval and generation, with book filtering.
- `app.py`: Streamlit UI for interactive querying.
- `books/`: Directory for exported ebook folders (e.g., `books/ASIN/` with `book.md` and `metadata.json`).
- `db/`: Persisted Chroma vector store.

## Tech Stack
- **LangChain**: Core framework for document loading, splitting, embedding, retrieval, and chaining.
- **Hugging Face Embeddings**: `sentence-transformers/all-MiniLM-L6-v2` for local vectorization.
- **Chroma**: Local vector database.
- **Ollama**: Local LLM (e.g., Gemma2 or Mistral) for generation.
- **Streamlit**: Web UI for chat interface.

## Integration with Kindle AI Export
This project builds on [Kindle AI Export](https://github.com/transitive-bullshit/kindle-ai-export), which extracts Kindle books into markdown and JSON metadata. Use it to prepare your ebooks:
- Follow their README to export books.
- Place outputs in `books/` subfolders.
- Run `ingestion.py` to index them.

For details on extraction, see their [README](./readme.md) (included in this repo for reference).

## Contributing
Contributions welcome! Fork the repo, make changes, and submit a PR. Ensure code follows PEP8 and tests pass (add pytest if needed).

## License
Apache License 2.0 © 2025 Keshav. See [LICENSE](LICENSE) for details.