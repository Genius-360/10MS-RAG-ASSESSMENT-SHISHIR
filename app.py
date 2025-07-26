# --- START OF MODIFIED FILE app.py ---

import os
import gradio as gr
from dotenv import load_dotenv

# Import LangChain components
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage

# --- 1. Load Environment Variables ---
load_dotenv()
# On Hugging Face, we'll use Space secrets instead of a .env file
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    # A fallback for local testing if you remove .env
    print("Warning: GOOGLE_API_KEY not found in environment. The app may fail.")
    # In a real scenario, you'd want to handle this more gracefully.
    # For Hugging Face, this is fine because the secret will be set.
    
print("Initializing application...")

# --- 2. ⭐ NEW: Load the Pre-built Vector Store ---
FAISS_INDEX_PATH = "faiss_index_hsc_bangla"

if not os.path.exists(FAISS_INDEX_PATH):
    raise FileNotFoundError(
        f"FAISS index not found at '{FAISS_INDEX_PATH}'. "
        "Please run the `create_vectorstore.py` script first to generate it."
    )

print("Loading pre-built FAISS index and embeddings model...")
embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=api_key)
vectorstore = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 10, "fetch_k": 20})
print("Vector store and retriever are ready.")

# --- 3. Setup LLM and RAG Chain (No changes from here) ---
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.25, google_api_key=api_key) 

contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", "Given a chat history and the latest user question, formulate a standalone question that can be understood without the chat history. Do NOT answer the question, just reformulate it."),
    ("placeholder", "{chat_history}"),
    ("human", "{input}"),
])
history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

qa_system_prompt = """
You are an expert AI Tutor from 10 Minute School specializing in Rabindranath Tagore's 'Oporichita' (অপরিচিতা). An expert on Rabindranath Tagore's short story 'Oporichita' (অপরিচিতা). Your knowledge encompasses every line of the story, the complete biography and relevant works of Rabindranath Tagore, and the Bengali meaning of every word used in 'Oporichita'. You will only discuss and answer questions related to 'Oporichita', its author, and its vocabulary...

Context:
{context}
"""
qa_prompt = ChatPromptTemplate.from_messages([
    ("system", qa_system_prompt), ("placeholder", "{chat_history}"), ("human", "{input}"),
])
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

# --- 4. Gradio UI and Logic (No changes from here) ---
# ... (The entire Gradio UI code block is identical to your original version) ...
def format_context_for_display(context_docs):
    if not context_docs:
        return "⚠️ **No context was retrieved.** This is why the AI could not answer. Check if the query is related to the document."
    formatted_context = ""
    for i, doc in enumerate(context_docs):
        page_content = doc.page_content.replace('\n', ' ').strip()
        source_info = doc.metadata.get('source', 'Unknown')
        formatted_context += f"**উৎস:** {source_info}\n> {page_content}\n\n---\n"
    return formatted_context

def chat_function(message, history):
    chat_history_for_chain = []
    for human, ai in history:
        chat_history_for_chain.append(HumanMessage(content=human))
        chat_history_for_chain.append(AIMessage(content=ai))
    response = rag_chain.invoke({"input": message, "chat_history": chat_history_for_chain})
    answer = response["answer"]
    context_display = format_context_for_display(response["context"])
    return answer, context_display

APP_CSS = """
:root { --accent-color: #FF6633; }
#app-title { color: var(--accent-color); font-size: 2.5em; font-weight: bold; text-align: center; }
#app-subtitle { text-align: center; color: grey; }
#logo-container { padding: 10px; display: flex; justify-content: center; }
.gradio-container { background: #0F0F0F; border-radius: 15px !important; }
footer { display: none !important; }
#custom-footer { text-align: center; padding: 10px; font-size: 0.8em; color: grey; }
"""

print("Building Gradio UI...")
with gr.Blocks(css=APP_CSS, theme=gr.themes.Soft(primary_hue="orange")) as demo:
    with gr.Column():
        with gr.Row(elem_id="logo-container"):
            gr.Image("logo.png", width=250, show_label=False, show_download_button=False, interactive=False)
        gr.Markdown("HSC Bangla AI Tutor", elem_id="app-title")
        gr.Markdown("Ask questions about 'Oporichita'. আমি আপনার যেকোনো প্রশ্নের উত্তর দিতে প্রস্তুত।", elem_id="app-subtitle")
        chatbot = gr.Chatbot(label="Chat History", height=500, avatar_images=("user_avatar.png", "bot_avatar.png"))
        with gr.Accordion("📚 View Retrieved Context", open=False):
            retrieved_context_display = gr.Markdown("Context will be shown here...")
        with gr.Row():
            msg = gr.Textbox(placeholder="এখানে আপনার প্রশ্ন লিখুন...", label="Your Question", scale=7)
            clear_btn = gr.Button("🗑️ Clear Chat", scale=1)
    chat_history_state = gr.State([])
    def handle_submit(message, history_list):
        history_tuples = [tuple(item) for item in history_list]
        answer, context_display = chat_function(message, history_tuples)
        updated_history_list = history_list + [[message, answer]]
        return updated_history_list, updated_history_list, context_display
    msg.submit(handle_submit, [msg, chat_history_state], [chatbot, chat_history_state, retrieved_context_display])
    clear_btn.click(lambda: ([], [], "Context will be shown here..."), None, [chatbot, chat_history_state, retrieved_context_display])
    gr.Examples(
        examples=[
            "অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে?",
            "বিয়ের সময় কল্যাণীর প্রকৃত বয়স কত ছিল?",
            "'জড়িমা' শব্দের অর্থ কী?",
            "রবীন্দ্রনাথ ঠাকুর কত সালে নোবেল পুরস্কার পান?",
            "অনুপমের আসল অভিভাবক কে?"
        ],
        inputs=msg,
        label="Sample Questions to Try"
    )
    gr.Markdown("Powered by Google Gemini & LangChain. A project by an aspiring AI Engineer (Shishir M.).", elem_id="custom-footer")

if __name__ == "__main__":
    print("Launching Gradio App...")
    demo.launch()

# --- END OF MODIFIED FILE app.py ---
