# Copyright 2025 Keshav
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
