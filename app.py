import os
import streamlit as st
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

st.set_page_config(page_title="Domain-Specific RAG Chatbot", page_icon="🤖")
st.title("Domain-Specific RAG Chatbot")
st.markdown("Based on Lab Exercise PT-M1: Nephrotic Syndrome Patient Education Q&A")

@st.cache_resource(show_spinner=False)
def init_rag(api_key):
    os.environ["GROQ_API_KEY"] = api_key
    
    from langchain_community.document_loaders import PyPDFDirectoryLoader
    
    # Load all PDF documents from the directory
    loader = PyPDFDirectoryLoader("my_data/")
    raw_documents = loader.load()

    # Split documents into smaller semantic chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    documents = text_splitter.split_documents(raw_documents)

    # Initialize open-source embedding model
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # Store embeddings into Chroma vector database
    vectorstore = Chroma.from_documents(documents=documents, embedding=embeddings)

    # Set vectorstore as a retriever
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # Initialize the SLM (OpenAI model on Groq as per notebook)
    llm = ChatGroq(model_name="openai/gpt-oss-20b", temperature=0)

    # Custom domain system prompt
    system_prompt = (
        "You are a specialized AI assistant for the user's uploaded domain.\n"
        "Answer questions strictly using ONLY the provided context below.\n"
        "If the answer cannot be found in the context, reply: 'I cannot answer based on the provided domain data.'\n\n"
        "Context:\n{context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    # Assemble full retrieval-augmented generation chain
    combine_docs_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, combine_docs_chain)
    
    return rag_chain

# --- Sidebar ---
api_key = st.sidebar.text_input("Enter Groq API Key", type="password")

st.sidebar.markdown("### How to use:")
st.sidebar.markdown("1. Make sure you have your documents in the `my_data/` folder.")
st.sidebar.markdown("2. Enter your Groq API key above.")
st.sidebar.markdown("3. Ask questions about the documents.")

# --- Main App Logic ---
if not api_key:
    st.info("Please enter your Groq API Key in the sidebar to initialize the chatbot.")
    st.stop()

if not os.path.exists("my_data") or not os.listdir("my_data"):
    st.error("The `my_data` directory is missing or empty. Please create a folder named `my_data` in the same directory as this script and add your PDF documents there.")
    st.stop()

with st.spinner("Initializing RAG pipeline (this may take a moment)..."):
    try:
        rag_chain = init_rag(api_key)
    except Exception as e:
        st.error(f"Error initializing RAG pipeline: {e}")
        st.stop()

# --- Chat Interface ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("Ask a question about the documents..."):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = rag_chain.invoke({"input": prompt})
                answer = response["answer"]
                
                # Format sources
                sources = response.get("context", [])
                if sources and answer != "I cannot answer based on the provided domain data.":
                    source_names = list(set([doc.metadata.get("source", "Unknown") for doc in sources]))
                    answer += "\n\n**Sources:**\n" + "\n".join([f"- {src}" for src in source_names])

                st.markdown(answer)
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": answer})
            except Exception as e:
                st.error(f"Error generating response: {e}")
