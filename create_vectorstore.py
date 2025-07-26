# --- File: create_vectorstore.py ---

import os
import re
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings

print("--- Starting Vector Store Creation ---")

# --- 1. Load Environment Variables ---
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY not found. Please set it in your .env file.")
print("API Key loaded.")

# --- 2. Load the Cleaned Knowledge Base ---
try:
    with open("hsc_bangla_clean.txt", "r", encoding="utf-8") as f:
        full_text = f.read()
    print("Successfully loaded the clean knowledge base.")
except Exception as e:
    raise FileNotFoundError(f"Error: 'hsc_bangla_clean.txt' not found. Details: {e}")

# --- 3. Structure-Aware Chunking (Your excellent function) ---
def create_structured_chunks(text):
    print("Applying structure-aware chunking...")
    sections = re.split(r'PAGE \d+', text)
    all_chunks = []
    story_pages_indices = list(range(6, 18))
    author_intro_index = 18
    prose_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    metadata_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)

    for i, section_text in enumerate(sections):
        page_num = i
        if not section_text.strip():
            continue
        if page_num in story_pages_indices:
            chunks = prose_splitter.split_text(section_text)
            for chunk in chunks:
                all_chunks.append(Document(page_content=chunk, metadata={"source": f"Main Story (Page {page_num})"}))
        elif page_num == author_intro_index:
            content = "লেখক পরিচিতি: রবীন্দ্রনাথ ঠাকুর\n" + section_text.strip()
            chunks = metadata_splitter.split_text(content)
            for chunk in chunks:
                all_chunks.append(Document(page_content=chunk, metadata={"source": f"Author Introduction (Page {page_num})"}))
        else:
            lines = section_text.strip().split('\n')
            title = lines[0] if lines else ""
            chunks = prose_splitter.split_text(section_text)
            for chunk in chunks:
                chunk_content = f"বিষয়: {title}\n{chunk}"
                all_chunks.append(Document(page_content=chunk_content, metadata={"source": f"Section (Page {page_num})"}))
    print(f"Successfully created {len(all_chunks)} structured document chunks.")
    return all_chunks

split_docs = create_structured_chunks(full_text)

# --- 4. Create and Save the Vector Store ---
print("Creating embeddings and building FAISS index. This may take a few minutes...")
embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=api_key)
vectorstore = FAISS.from_documents(split_docs, embeddings)

# **THE MOST IMPORTANT PART: SAVE THE INDEX TO DISK**
FAISS_INDEX_PATH = "faiss_index_hsc_bangla"
vectorstore.save_local(FAISS_INDEX_PATH)

print("\n--- ✅ Success! ---")
print(f"FAISS index has been created and saved to the folder: '{FAISS_INDEX_PATH}'")