import streamlit as st
import numpy as np
import faiss
from pathlib import Path
from pypdf import PdfReader
import requests
import json

# Set up API keys
MISTRAL_API_KEY = os.getenv('MISTRAL_API_KEY')

# Function to get embeddings from Mistral AI
def get_embeddings(text):
    response = requests.post(
        'https://api.mistral.ai/v1/embeddings',
        headers={'Authorization': f'Bearer {MISTRAL_API_KEY}'},
        json={'text': text}
    )
    return np.array(response.json()['embeddings'])

# Function to read PDF and return text chunks
def read_pdf(file_path, chunk_size=500):
    reader = PdfReader(file_path)
    text = ''.join(page.extract_text() for page in reader.pages)
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

# Function to build FAISS index
def build_index(chunks):
    dimension = len(get_embeddings(chunks[0]))
    index = faiss.IndexFlatL2(dimension)
    embeddings = [get_embeddings(chunk) for chunk in chunks]
    index.add(np.vstack(embeddings))
    return index, chunks

# Function to handle user query
def handle_query(query, index, chunks):
    query_embedding = get_embeddings(query)
    D, I = index.search(query_embedding, k=1)
    return chunks[I[0][0]]

# Function to generate response using Mistral AI
def generate_response(context, query):
    prompt = f"Context: {context}\nQuestion: {query}\nAnswer:"
    response = requests.post(
        'https://api.mistral.ai/v1/chat/completions',
        headers={'Authorization': f'Bearer {MISTRAL_API_KEY}'},
        json={'model': 'mistral-medium', 'prompt': prompt}
    )
    return response.json()['text']

# Main Streamlit app
def main():
    st.title("Chat with Your Documents")
    st.sidebar.title("Options")
    reset_button = st.sidebar.button("Reset Conversation")

    if reset_button:
        st.session_state['messages'] = []

    if 'messages' not in st.session_state:
        st.session_state['messages'] = []

    # Load document and build index
    doc_path = st.sidebar.file_uploader("Upload your document", type="pdf")
    if doc_path:
        chunks = read_pdf(doc_path)
        index, chunks = build_index(chunks)
        st.session_state['index'] = index
        st.session_state['chunks'] = chunks

    # Chat interface
    if 'index' in st.session_state and 'chunks' in st.session_state:
        user_query = st.text_input("Ask a question")
        if user_query:
            context = handle_query(user_query, st.session_state['index'], st.session_state['chunks'])
            response = generate_response(context, user_query)
            st.session_state['messages'].append((user_query, response))

        for query, response in st.session_state['messages']:
            st.write(f"**You:** {query}")
            st.write(f"**Bot:** {response}")

if __name__ == "__main__":
    main()
