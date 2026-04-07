import streamlit as st
from src.chains.rag_chain import RAGPipeline

# Page Config
st.set_page_config(page_title="PDF RAG Chatbot",layout="wide")

st.title("PDF RAG Chatbot")
st.markdown("Chat with your PDF (Aurelien Geron ML Book)")

# Load Pipeline
@st.cache_resource
def load_pipeline():
    return RAGPipeline(top_k=8)  # slightly higher for better answers

rag = load_pipeline()

# Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display Chat History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        # Show sources for assistant messages
        if msg["role"] == "assistant" and "sources" in msg:
            with st.expander("Sources"):
                for i, doc in enumerate(msg["sources"]):
                    st.markdown(
                        f"**Source {i+1} (Page {doc.metadata.get('page')}):**"
                    )
                    st.write(doc.page_content[:400])
                    st.markdown("---")

# Chat Input
query = st.chat_input("Ask something about the book...")

if query:
    # Add user message
    st.session_state.messages.append({
        "role": "user",
        "content": query
    })

    # Display user message
    with st.chat_message("user"):
        st.markdown(query)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = rag.ask(query)
            answer = result["answer"]

            st.markdown(answer)

            # Show sources
            with st.expander("Sources"):
                for i, doc in enumerate(result["sources"]):
                    st.markdown(
                        f"**Source {i+1} (Page {doc.metadata.get('page')}):**"
                    )
                    st.write(doc.page_content[:400])
                    st.markdown("---")

    # Save assistant response
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": result["sources"]
    })