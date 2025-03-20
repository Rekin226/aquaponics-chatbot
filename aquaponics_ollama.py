#!/usr/bin/env python3
"""
Aquaponics RAG Chatbot - A Retrieval-Augmented Generation pipeline 
that answers questions about aquaponics using information from web documents.
This version uses Llama-3-8B through the Ollama API with GPU acceleration.
"""

import os
import sys
import requests
import re
import json
import torch
from typing import List, Dict, Any, Optional, Union
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# Import the correct Ollama class
try:
    # Try the new package first
    from langchain_ollama import OllamaLLM as Ollama
    print("Using langchain_ollama.OllamaLLM")
except ImportError:
    # Fall back to the deprecated one if necessary
    from langchain_community.llms import Ollama
    print("Using langchain_community.llms.Ollama (deprecated)")

# Define paths
URLS_PATH = "aquaponics_urls.txt"
FAISS_INDEX_PATH = "aquaponics_faiss_index"

# Ollama API settings
OLLAMA_HOST = "http://localhost:11434"  # Default Ollama API host
OLLAMA_MODEL = "llama3"                 # The model name in Ollama (llama3 is the 8B version)

class DataIngestion:
    """
    Handles reading URLs and fetching content from web documents
    """
    
    def read_urls_from_file(self, file_path: str) -> List[str]:
        """Reads URLs from a text file."""
        try:
            with open(file_path, 'r') as f:
                urls = [line.strip() for line in f if line.strip()]
            print(f"Read {len(urls)} URLs from {file_path}")
            return urls
        except FileNotFoundError:
            print(f"Error: File '{file_path}' not found.")
            sys.exit(1)
    
    def fetch_document(self, url: str) -> str:
        """Fetches document content from a URL."""
        try:
            print(f"Fetching content from {url}")
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            # Check if it's PDF by content type or URL extension
            if 'application/pdf' in response.headers.get('Content-Type', '') or url.endswith('.pdf'):
                print(f"Skipping PDF content from {url} - PDF parsing requires additional libraries")
                return ""
            
            return response.text
        except requests.exceptions.RequestException as e:
            print(f"Error fetching {url}: {e}")
            return ""
            
    def preprocess_text(self, text: str) -> str:
        """Cleans and preprocesses the fetched text."""
        # Remove HTML tags
        soup = BeautifulSoup(text, 'html.parser')
        clean_text = soup.get_text(separator=' ', strip=True)
        
        # Remove extra whitespaces
        clean_text = re.sub(r'\s+', ' ', clean_text)
        
        # Remove special characters and normalize text
        clean_text = clean_text.replace('\n', ' ').replace('\r', ' ')
        
        # Additional preprocessing steps
        clean_text = re.sub(r'[^\w\s]', '', clean_text)  # Remove non-alphanumeric characters
        clean_text = clean_text.lower()  # Convert to lowercase
        
        return clean_text


class DocumentProcessor:
    """
    Handles document splitting and embedding computation with configurable GPU/CPU support
    """
    
    def __init__(self, force_cpu=False):
        # Initialize text splitter for chunking
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        
        # Check for GPU availability unless CPU is forced
        self.device = "cpu" if force_cpu else ("cuda" if torch.cuda.is_available() else "cpu")
        if self.device == "cuda":
            print("GPU acceleration enabled")
        else:
            print("Running on CPU" + (" (forced)" if force_cpu else " - GPU not detected"))
        
        # Initialize embeddings with device support
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': self.device}
        )
    
    def split_documents(self, documents: List[Dict[str, str]]) -> List[Document]:
        """Splits documents into smaller chunks."""
        docs = [Document(page_content=doc["text"], metadata={"source": doc["url"]}) 
                for doc in documents if doc["text"]]
        
        print(f"Splitting {len(docs)} documents into chunks...")
        splits = self.text_splitter.split_documents(docs)
        print(f"Created {len(splits)} document chunks")
        return splits
    
    def create_faiss_index(self, documents: List[Document]) -> FAISS:
        """Creates a FAISS index from document chunks with optional GPU acceleration."""
        print("Computing embeddings and creating FAISS index...")
        use_gpu = self.device == "cuda"
        if use_gpu:
            print("Using GPU acceleration for FAISS index creation")
        vectorstore = FAISS.from_documents(
            documents, 
            self.embeddings,
            gpu=use_gpu
        )
        print("FAISS index created successfully")
        return vectorstore
    
    def save_faiss_index(self, vectorstore: FAISS, path: str) -> None:
        """Saves FAISS index to disk."""
        print(f"Saving FAISS index to {path}...")
        vectorstore.save_local(path)
        print("FAISS index saved")
    
    def load_faiss_index(self, path: str) -> FAISS:
        """Loads FAISS index from disk."""
        print(f"Loading FAISS index from {path}...")
        vectorstore = FAISS.load_local(path, self.embeddings, allow_dangerous_deserialization=True)
        print("FAISS index loaded")
        return vectorstore


class RAGChain:
    """
    Implements the retrieval-augmented generation chain
    """
    
    def __init__(self, vectorstore: FAISS):
        self.vectorstore = vectorstore
        
        # Check if Ollama is running by making a simple API call
        try:
            response = requests.get(f"{OLLAMA_HOST}/api/tags")
            if response.status_code != 200:
                raise Exception(f"Ollama API returned status code {response.status_code}")
            
            # Check if the model is available
            models_data = response.json()
            available_models = []
            
            # Handle different response formats
            if "models" in models_data:
                available_models = [model["name"] for model in models_data["models"]]
            elif "models" not in models_data:
                # Newer Ollama versions list models at the top level
                available_models = [model["name"] for model in models_data.get("models", [])]
            
            print(f"Available models: {available_models}")
            
            if OLLAMA_MODEL not in " ".join(available_models):
                print(f"Model {OLLAMA_MODEL} not found. You may need to pull it first with:")
                print(f"  ollama pull {OLLAMA_MODEL}")
                print("Attempting to continue anyway...")
            
        except Exception as e:
            print(f"Error connecting to Ollama API: {e}")
            print("Please make sure Ollama is installed and running.")
            print("You can download it from https://ollama.com/download")
            sys.exit(1)
        
        # Initialize the Ollama LLM
        print(f"Connecting to Ollama API for model: {OLLAMA_MODEL}...")
        try:
            self.llm = Ollama(
                base_url=OLLAMA_HOST,
                model=OLLAMA_MODEL,
                temperature=0.7,
                num_ctx=4096,  # Larger context window
                verbose=False,  # Set to True for debugging
            )
            print("Ollama connection established successfully")
        except Exception as e:
            print(f"Error initializing Ollama: {e}")
            sys.exit(1)
        
        # Create the prompt template optimized for Llama-3
        template = """
            <|system|>
            You are a subject matter expert and aquaponics consultant with extensive experience advising both individuals and businesses on designing, implementing, and managing aquaponics systems. When a user asks a question:
            1. Check if the provided context is sufficient.
            2. If the context is insufficient, ask one round of clarifying questions to better understand their needs.
            3. Once the user responds with additional details or confirms that no further clarification is needed, provide your final answer based solely on the updated context.
            4. Do not ask for clarifications more than once per query.
            Maintain a friendly, professional, and consultative tone throughout.
            </|system|>

            <|user|>
            Context:
            {context}

            Question:
            {question}
            </|user|>

            <|assistant|>
        """
        
        self.prompt = PromptTemplate(
            input_variables=["context", "question"],
            template=template,
        )
        
        # Create the retrieval chain
        self.qa_chain = self._create_qa_chain()
    
    def _create_qa_chain(self) -> RetrievalQA:
        """Creates a question-answering chain with retrieval capabilities."""
        retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}  # Retrieve top 3 most similar chunks
        )
        
        return RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",  # "stuff" method: Stuff all retrieved docs into prompt
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": self.prompt}
        )
    
    def answer_query(self, query: str) -> Dict[str, Any]:
        """Processes a user query and returns an answer with source documents."""
        print(f"Processing query: {query}")
        try:
            result = self.qa_chain.invoke({"query": query})
            return result
        except Exception as e:
            print(f"Error in answer generation: {e}")
            return {
                "result": f"I'm sorry, I encountered an error while processing your question: {str(e)}",
                "source_documents": []
            }


def main():
    """Main function that orchestrates the RAG pipeline."""
    
    # Add command line argument parsing for CPU/GPU choice
    force_cpu = "--cpu" in sys.argv
    if force_cpu:
        print("Forcing CPU mode due to --cpu flag")

    # Check if the FAISS index already exists
    if os.path.exists(FAISS_INDEX_PATH) and os.path.isdir(FAISS_INDEX_PATH):
        print("Found existing FAISS index. Loading...")
        processor = DocumentProcessor(force_cpu=force_cpu)
        vectorstore = processor.load_faiss_index(FAISS_INDEX_PATH)
    else:
        print("Building new FAISS index...")
        
        # Step 1: Load URLs
        ingestion = DataIngestion()
        urls = ingestion.read_urls_from_file(URLS_PATH)
        
        # Step 2: Fetch and preprocess documents
        documents = []
        for url in urls:
            text = ingestion.fetch_document(url)
            if text:  # Only process non-empty texts
                clean_text = ingestion.preprocess_text(text)
                documents.append({"url": url, "text": clean_text})
        
        # Step 3: Process documents and create embeddings
        processor = DocumentProcessor(force_cpu=force_cpu)
        doc_chunks = processor.split_documents(documents)
        
        # Step 4: Create and save FAISS index
        vectorstore = processor.create_faiss_index(doc_chunks)
        processor.save_faiss_index(vectorstore, FAISS_INDEX_PATH)

    # Step 5: Initialize RAG Chain
    rag_chain = RAGChain(vectorstore)
    
    # Step 6: Interactive query loop
    print("\n===== Aquaponics Chatbot (Ollama Llama-3) =====")
    print("Type 'exit' to quit")
    
    while True:
        query = input("\nYour question: ").strip()
        if query.lower() in ["exit", "quit"]:
            break
        
        if not query:
            continue
        
        # Process query
        result = rag_chain.answer_query(query)
        
        # Display answer
        print("\nAnswer:")
        print(result["result"])
        
        # Display sources
        if result.get("source_documents"):
            print("\nSources:")
            seen_sources = set()
            for doc in result["source_documents"]:
                source = doc.metadata.get("source", "Unknown")
                if source not in seen_sources:
                    print(f"- {source}")
                    seen_sources.add(source)
        
        print("\n" + "-" * 40)


if __name__ == "__main__":
    main()