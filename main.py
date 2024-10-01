import streamlit as st
import openai
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader

# Sidebar function
def sidebar():
    st.sidebar.title("Settings")

    # Input for OpenAI API key
    api_key = st.sidebar.text_input("Enter OpenAI API Key", type="password")

    # File upload
    uploaded_file = st.sidebar.file_uploader(
        "Upload a file (PDF, TXT, DOCX)", type=["pdf", "txt", "docx"])

    # Chunk size input
    chunk_size = st.sidebar.number_input(
        "Chunk Size", min_value=100, max_value=1000, value=256, step=50)

    # k value input (number of results to retrieve)
    k_value = st.sidebar.number_input(
        "Number of results to retrieve (k)", min_value=1, max_value=10, value=3)

    # Add Data button
    add_data_button = st.sidebar.button("Add Data")

    return api_key, uploaded_file, chunk_size, k_value, add_data_button

# Function to load the file and return as LangChain documents
def load_file(file):
    file_extension = file.name.split('.')[-1].lower()

    # Save the file to a temporary location
    with open(file.name, "wb") as f:
        f.write(file.getbuffer())

    if file_extension == 'pdf':
        loader = PyPDFLoader(file.name)
    elif file_extension == 'docx':
        loader = Docx2txtLoader(file.name)
    elif file_extension == 'txt':
        loader = TextLoader(file.name)
    else:
        st.error(f"Unsupported file format: {file_extension}")
        return None

    # Load the documents
    return loader.load()

# Function to split documents into chunks
def split_document_into_chunks(documents, chunk_size=256, overlap=20):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=overlap)
    chunks = []
    for doc in documents:
        chunks.extend(text_splitter.split_text(doc.page_content))
    return chunks

# Function to create embeddings and store them in ChromaDB
def create_embeddings_and_store(chunks, api_key, chroma_db_path="chroma_db"):
    openai.api_key = api_key  # Set the OpenAI API key for openai library

    # Initialize OpenAI embeddings, passing the API key explicitly
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=api_key)

    # Initialize Chroma vector store
    chroma_store = Chroma(persist_directory=chroma_db_path,
                          embedding_function=embeddings)

    # Add chunks to Chroma
    chroma_store.add_texts(texts=chunks)

    # Persist the store
    chroma_store.persist()

    return chroma_store

# Function to build the RetrievalQA chain
def build_retrieval_qa(chroma_store, k=3, model_name="gpt-3.5-turbo", api_key=None):
    # Pass the OpenAI API key explicitly to ChatOpenAI
    llm = ChatOpenAI(model_name=model_name, openai_api_key=api_key)
    retriever = chroma_store.as_retriever(
        search_type="similarity", search_kwargs={"k": k})
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm, retriever=retriever, return_source_documents=True)
    return qa_chain

# Function to process and answer question with custom instructions
def process_and_answer_question_with_instruction(qa_chain, question):
    custom_instruction = (
        "\nAnswer only based on the text input. Don't search external sources. "
        "If you can't answer the question, then just write 'I don't know'."
    )
    question_with_instruction = question + custom_instruction

    if question:
        # Use the __call__ method to get both answer and source documents
        result = qa_chain({"query": question_with_instruction})

        # Extract the answer from the 'result' key
        return result["result"]

    return "No question provided."

# Main function to run the Streamlit app
def main():
    st.title("LLM Question Answering APP 🤖")

    # Sidebar inputs
    api_key, uploaded_file, chunk_size, k_value, add_data_button = sidebar()

    if 'chroma_store' not in st.session_state:
        st.session_state['chroma_store'] = None

    if add_data_button:
        if not api_key:
            st.error("Please enter your OpenAI API Key.")
        elif not uploaded_file:
            st.error("Please upload a file.")
        else:
            # Load the file
            documents = load_file(uploaded_file)
            if documents:
                # Split the documents into chunks
                chunks = split_document_into_chunks(
                    documents, chunk_size=chunk_size)

                # Create embeddings and store in ChromaDB, store it in session state
                chroma_store = create_embeddings_and_store(chunks, api_key)
                st.session_state['chroma_store'] = chroma_store

                st.success("Data successfully added and embeddings created.")

    # Build the QA system only if we have the chroma store in session state
    if st.session_state['chroma_store']:
        # Pass the API key explicitly when building the QA chain
        qa_chain = build_retrieval_qa(
            st.session_state['chroma_store'], k=k_value, api_key=api_key)

        # Ask the user to enter a question
        question = st.text_input("Enter your question")

        # Add a button to submit the question
        if st.button("Submit Question"):
            # Process and answer the question
            answer = process_and_answer_question_with_instruction(
                qa_chain, question)

            # Display the answer in a text area
            st.text_area("Answer", value=answer, height=200)

# Run the app
if __name__ == "__main__":
    main()