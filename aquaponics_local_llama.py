#!/usr/bin/env python3
"""
Aquaponics RAG Chatbot - A Retrieval-Augmented Generation pipeline 
that answers questions about aquaponics using information from web documents.
This version uses a local LLaMA model for the language model.
"""

import os
import sys
import requests
import re
import json
from typing import List, Dict, Any, Optional, Union
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.llms import LlamaCpp

# Define paths
URLS_PATH = "aquaponics_urls.txt"
FAISS_INDEX_PATH = "aquaponics_faiss_index"
MODEL_PATH = "models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"

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
        
        return clean_text


class DocumentProcessor:
    """
    Handles document splitting and embedding computation
    """
    
    def __init__(self):
        # Initialize text splitter for chunking
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        
        # Initialize embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
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
        """Creates a FAISS index from document chunks."""
        print("Computing embeddings and creating FAISS index...")
        vectorstore = FAISS.from_documents(documents, self.embeddings)
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
        
        # Check if the model file exists
        if not os.path.exists(MODEL_PATH):
            print(f"Error: Model file '{MODEL_PATH}' not found.")
            print("Please make sure the TinyLlama model is downloaded correctly.")
            sys.exit(1)
        
        # Initialize the local LLaMA model
        print(f"Loading local LLaMA model from {MODEL_PATH}...")
        print("This might take a moment depending on your hardware...")
        
        # Initialize with parameters suitable for most computers
        # Adjust n_gpu_layers and n_threads based on your hardware
        self.llm = LlamaCpp(
            model_path=MODEL_PATH,
            temperature=0.7,
            max_tokens=512,
            n_ctx=2048,  # Context window size
            n_gpu_layers=1,  # Increase if you have a good GPU
            n_threads=4,  # Adjust based on your CPU
            verbose=False,  # Set to True for debugging
        )
        
        print("Local LLaMA model loaded successfully")
        
        # Create the prompt template with a format tailored for TinyLlama
        template = """
        <|system|>
        You are an expert in aquaponics, a system that combines aquaculture 
        (raising aquatic animals) with hydroponics (cultivating plants in water) 
        in a symbiotic environment. Answer questions based on the context provided.
        If the answer is not contained within the context, say "I don't have enough 
        information to answer this question" and suggest resources they could look into.
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
                "result": "I'm sorry, I encountered an error while processing your question.",
                "source_documents": []
            }


def main():
    """Main function that orchestrates the RAG pipeline."""
    
    # Check if the FAISS index already exists
    if os.path.exists(FAISS_INDEX_PATH) and os.path.isdir(FAISS_INDEX_PATH):
        print("Found existing FAISS index. Loading...")
        processor = DocumentProcessor()
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
        processor = DocumentProcessor()
        doc_chunks = processor.split_documents(documents)
        
        # Step 4: Create and save FAISS index
        vectorstore = processor.create_faiss_index(doc_chunks)
        processor.save_faiss_index(vectorstore, FAISS_INDEX_PATH)

    # Step 5: Initialize RAG Chain
    rag_chain = RAGChain(vectorstore)
    
    # Step 6: Interactive query loop
    print("\n===== Aquaponics Chatbot (Local LLaMA) =====")
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