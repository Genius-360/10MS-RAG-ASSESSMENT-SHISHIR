# 10MS-RAG-ASSESSMENT-SHISHIR

# Multilingual RAG System for 10 Minute School AI Engineer Assessment

This repository contains the source code for a multilingual Retrieval-Augmented Generation (RAG) system, developed as a technical assessment for the AI Engineer (Level 1) position at 10 Minute School.

The application functions as an AI-powered tutor for the HSC Bangla text 'অপরিচিতা' (Oporichita), capable of answering user queries in both **Bengali** and **English**.

[![Hugging Face Spaces](https://img.shields.io/badge/🤗%20Hugging%20Face-Live%20Demo-yellow.svg)](https://huggingface.co/spaces/GeniusGuy/10ms-hsc-bangla-tutor)  
---

## 🚀 Features

-   **Multilingual Support:** Accepts and responds to queries in both Bengali and English.
-   **Knowledge Base:** Utilizes the 'HSC26 Bangla 1st paper' PDF as its core knowledge source.
-   **Intelligent Chunking:** Employs a structure-aware chunking strategy to maintain semantic context.
-   **Conversational Memory:** Maintains short-term memory of the chat history to answer follow-up questions.
-   **Interactive UI:** Built with Gradio for an easy-to-use, web-based chat interface.
-   **Optimized for Deployment:** Features a pre-processing script to build the vector store, allowing the main application to load quickly.

---

## 🛠️ Tech Stack & Architecture

-   **LLM:** Google Gemini 2.5 Flash
-   **Framework:** LangChain
-   **Embedding Model:** `models/text-embedding-004` (Google's Multilingual Embedding Model)
-   **Vector Store:** FAISS (Facebook AI Similarity Search)
-   **UI:** Gradio
-   **Deployment:** Hugging Face Spaces

The system follows a standard RAG pipeline:
1.  **Offline Indexing:** The Bengali PDF is parsed, cleaned, and chunked. These chunks are converted into vector embeddings and stored in a FAISS index.
2.  **Live Inference:** When a user asks a question, it is first reformulated based on chat history. The reformulated query is then used to retrieve the most relevant document chunks from the FAISS index.
3.  **Generation:** The original query, chat history, and the retrieved chunks are passed to the Gemini LLM in a structured prompt to generate a final, context-grounded answer.

---

## 📋 Setup and Run Locally

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Genius-360/10MS-RAG-ASSESSMENT-SHISHIR.git
    cd your-repo-name
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up your API Key:**
    -   Create a file named `.env` in the root directory.
    -   Add your Google API key to the file:
        ```
        GOOGLE_API_KEY="actual_api_key_here"
        ```

5.  **Build the Vector Store (One-time setup):**
    -   If the `faiss_index_hsc_bangla` folder does not exist, run the pre-processing script:
    ```bash
    python create_vectorstore.py
    ```

6.  **Run the application:**
    ```bash
    python app.py
    ```
    The Gradio app will be available at `http://127.0.0.1:7860`.

---

## 📝 Assessment Questions & Answers

#### Q1: What method or library did you use to extract the text, and why? Did you face any formatting challenges?
**A:** I used a combination of **PyMuPDF (`fitz`)** for initial PDF parsing and **EasyOCR** for Optical Character Recognition. While PyMuPDF is great, many educational PDFs in Bengali are scanned images, making direct text extraction unreliable. EasyOCR, with its strong support for Bengali (`bn`) and English (`en`), was used to accurately extract text from each page image. The primary challenge was inconsistent spacing and line breaks, which I normalized by processing the OCR output into a clean `.txt` file before chunking.

#### Q2: What chunking strategy did you choose? Why do you think it works well for semantic retrieval?
**A:** I implemented a **structure-aware chunking strategy** using LangChain's `RecursiveCharacterTextSplitter`. I identified that the PDF had distinct sections (story, author bio, Q&A). I applied a smaller `chunk_size` (1000) for the dense prose of the main story and a larger `chunk_size` (1500) for the author's biography to keep it more intact. This hybrid approach is superior to a single fixed-size strategy because it respects the document's natural structure, leading to more semantically complete and contextually relevant chunks for retrieval.

#### Q3: What embedding model did you use? Why did you choose it?
**A:** I used Google's `models/text-embedding-004`. I chose this model for two critical reasons: **1) State-of-the-Art Multilingual Performance:** It is designed to create a shared vector space for many languages, which is essential for this project's requirement to handle both Bengali and English queries against a Bengali corpus. **2) High Performance & Integration:** It is highly optimized and integrates seamlessly with the Gemini LLM within the Google/LangChain ecosystem.

#### Q4: How are you comparing the query with your stored chunks? Why did you choose this similarity method and storage setup?
**A:** The comparison is performed using **Cosine Similarity** within a **FAISS** vector store. When a query is received, it's converted to a vector using the same embedding model. FAISS then performs a highly efficient search to find the vectors (and their corresponding text chunks) that have the highest cosine similarity to the query vector. I chose FAISS because it is incredibly fast, memory-efficient, and the industry standard for local, medium-scale similarity search tasks, making it perfect for this application.

#### Q5: How do you ensure that the question and the document chunks are compared meaningfully? What would happen if the query is vague?
**A:** Meaningful comparison is ensured by using the **exact same embedding model** for both the document chunks (during indexing) and the user query (at inference time). This maps them into the same semantic space. If a query is vague (e.g., "তার সম্পর্কে বলুন" / "Tell me about him"), the retriever's performance will be limited. However, the system is designed to be robust. The `history_aware_retriever` first tries to add context from the chat history. If the query remains vague, the retrieved context will be generic, and the final LLM prompt instructs the model to state that it cannot find a specific answer if the context is insufficient, preventing hallucination.

#### Q6: Do the results seem relevant? If not, what might improve them?
**A:** For specific, fact-based queries provided in the test case, the results are highly relevant. The system successfully identifies the correct chunks containing the answers. For more complex or ambiguous questions, relevance could be further improved by:
1.  **Hybrid Search:** Combining the current semantic search with a keyword-based search (like BM25) to better match specific names and terms.
2.  **Re-ranking:** Using a cross-encoder model to re-rank the top retrieved chunks for finer relevance before passing them to the LLM.
3.  **Advanced Chunking:** Implementing more sophisticated chunking logic that explicitly understands document elements like titles, tables, and lists.
