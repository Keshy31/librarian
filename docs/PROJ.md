# Project Overview: Ebook RAG System with LangChain

## Introduction
This project implements a Retrieval-Augmented Generation (RAG) system using LangChain to enable interactive querying of ebooks stored as markdown files. The system ingests ebook content, embeds it for semantic search, stores it in a vector database, and provides a chat interface for users to ask questions, retrieve specific paragraphs, or summarize concepts. It leverages metadata from JSON files to enrich queries with chapter and page references.

The initial prototype focuses on simplicity and local execution, supporting a small library (starting with 2 test ebooks). It's designed for scalability, with potential extensions for larger libraries or image integration. The core goal is to "chat with books," e.g., "Summarize Hamming's views on style in chapter 1" or "Retrieve paragraphs on education vs. training from page 3."

## Key Architecture Components
The system follows a standard RAG pipeline:
1. **Ingestion**: Load markdown files, split into chunks, and add metadata (e.g., book title, chapter, page).
2. **Embedding & Storage**: Convert chunks to vectors and store in a local vector database.
3. **Querying**: Use semantic search to retrieve relevant chunks, then generate responses via a local LLM.
4. **User Interface**: A web-based chat UI for natural language queries.

High-level flow:
- User inputs a query via the UI.
- System retrieves top relevant chunks (e.g., 5) based on similarity.
- LLM generates a grounded response, including citations (e.g., chapter/page).

## Tech Stack
- **Framework**: LangChain (for loaders, splitters, embeddings, vector stores, and chains).
- **Document Loading**: `UnstructuredMarkdownLoader` for markdown files.
- **Text Splitting**: `RecursiveCharacterTextSplitter` (chunk size ~1000 chars, overlap 200).
- **Embeddings**: Hugging Face `sentence-transformers/all-MiniLM-L6-v2` (local, lightweight).
- **Vector Database**: Chroma (local, persistent on disk).
- **LLM**: Ollama with Mistral model (local inference; alternatives like Llama3 possible).
- **UI**: Streamlit (simple web app for chat interface).
- **Python Version**: 3.10+.
- **Installation**: `pip install langchain langchain-community sentence-transformers chromadb langchain_ollama streamlit`.

All components are local and offline to ensure privacy and zero cost. No external APIs are required.

## Implementation Details

### 1. Data Ingestion
- **Input**: Markdown files (one per ebook) and corresponding metadata JSON (e.g., TOC, pages).
- **Process**:
  - Parse JSON to map pages to chapters.
  - Load markdown content.
  - Split into chunks using `RecursiveCharacterTextSplitter` to preserve structure (e.g., paragraphs, headers).
  - Enrich each chunk with metadata: book title, author, chapter, page (extracted via regex for [Page X] markers).
- **Example Code Snippet**:
  ```python
  import json
  import re
  from langchain.document_loaders import UnstructuredMarkdownLoader
  from langchain.text_splitter import RecursiveCharacterTextSplitter
  from langchain.schema import Document

  # Load metadata
  with open('metadata.json', 'r') as f:
      metadata = json.load(f)

  # TOC to page-chapter map (simplified)
  toc = metadata['toc']
  page_to_chapter = {entry['page']: entry['title'] for entry in toc if 'page' in entry}

  # Load markdown
  loader = UnstructuredMarkdownLoader('hamming_book.md')
  docs = loader.load()

  # Split
  splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
  chunks = splitter.split_documents(docs)

  # Add metadata
  current_page = 1
  current_chapter = "Introduction"
  for chunk in chunks:
      page_match = re.search(r'\[Page (\d+)\]', chunk.page_content)
      if page_match:
          current_page = int(page_match.group(1))
          current_chapter = page_to_chapter.get(current_page, current_chapter)
      chunk.metadata = {
          "book": metadata['meta']['title'],
          "author": metadata['meta']['authorList'][0],
          "chapter": current_chapter,
          "page": current_page
      }
  ```

### 2. Embedding and Storage
- **Process**: Embed chunks using the Hugging Face model and store in Chroma.
- **Benefits**: Supports metadata filtering (e.g., query only specific chapters).
- **Example Code Snippet**:
  ```python
  from langchain.embeddings import HuggingFaceEmbeddings
  from langchain.vectorstores import Chroma

  embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
  vectorstore = Chroma.from_documents(
      documents=chunks,
      embedding=embeddings,
      persist_directory="./ebook_db"
  )
  vectorstore.persist()
  ```

### 3. Querying and Response Generation
- **Process**: Use a RetrievalQA chain to fetch chunks and generate responses.
- **Custom Prompt**: Ensures responses include citations and handle summarization/retrieval.
- **Example Code Snippet**:
  ```python
  from langchain_ollama import OllamaLLM
  from langchain.chains import RetrievalQA
  from langchain.prompts import PromptTemplate

  llm = OllamaLLM(model="mistral")

  prompt_template = """Use the following context to answer the question. Include page/chapter citations.
  If summarizing a concept, be concise. If retrieving paragraphs, quote them directly.

  Context: {context}

  Question: {question}
  Answer:"""
  PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

  qa_chain = RetrievalQA.from_chain_type(
      llm=llm,
      chain_type="stuff",
      retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
      chain_type_kwargs={"prompt": PROMPT}
  )
  ```

### 4. Streamlit User Interface
- **Description**: A simple web-based chat interface for querying the system. Users type questions, and responses appear in a conversation history. Includes a sidebar for instructions or future filters (e.g., book selection).
- **Features**:
  - Real-time chat simulation.
  - Displays responses with citations.
  - Session state for conversation history.
- **Running**: `streamlit run app.py` (separate file).
- **Example Code Snippet** (in `app.py`):
  ```python
  import streamlit as st
  from langchain_ollama import OllamaLLM
  from langchain.vectorstores import Chroma
  from langchain.embeddings import HuggingFaceEmbeddings
  from langchain.chains import RetrievalQA
  from langchain.prompts import PromptTemplate

  # Load vectorstore and chain (assume ingested already)
  embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
  vectorstore = Chroma(persist_directory="./ebook_db", embedding_function=embeddings)
  llm = OllamaLLM(model="mistral")

  # Prompt (same as above)
  PROMPT = PromptTemplate(...)  # Copy from querying section

  qa_chain = RetrievalQA.from_chain_type(
      llm=llm,
      chain_type="stuff",
      retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
      chain_type_kwargs={"prompt": PROMPT}
  )

  # Streamlit UI
  st.title("Ebook Chat RAG System")
  st.sidebar.header("Instructions")
  st.sidebar.write("Ask questions about your ebooks, e.g., 'Summarize chapter 1' or 'Retrieve info on style'.")

  # Chat history
  if "messages" not in st.session_state:
      st.session_state.messages = []

  for message in st.session_state.messages:
      with st.chat_message(message["role"]):
          st.markdown(message["content"])

  # User input
  if prompt := st.chat_input("Your query:"):
      st.session_state.messages.append({"role": "user", "content": prompt})
      with st.chat_message("user"):
          st.markdown(prompt)

      with st.chat_message("assistant"):
          with st.spinner("Thinking..."):
              response = qa_chain.run(prompt)
          st.markdown(response)
      st.session_state.messages.append({"role": "assistant", "content": response})
  ```

## Deployment and Usage
- **Setup**: Install dependencies, run Ollama (`ollama run mistral`), ingest data once (via script), then launch Streamlit.
- **Testing**: Use sample queries on test ebooks. Expected latency: <5s per query on standard hardware.
- **Multi-Book Support**: Ingest multiple markdown/JSON pairs; use Chroma collections for separation.
- **Limitations**: Local-only; no real-time updates. For production, consider containerization (e.g., Docker).

## Next Steps
- Test with full ebooks and refine chunking/metadata.
- Add filters in UI (e.g., dropdown for books/chapters).
- Monitor performance; optimize embeddings if needed.
- Gather team feedback for enhancements like image viewing from JSON screenshots.

This overview captures the build's essentials—feel free to expand or demo the prototype!