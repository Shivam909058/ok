import streamlit as st
import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import MongodbLoader
from dotenv import load_dotenv
import time
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
import certifi

load_dotenv()

# Load OpenAI API key from environment or a direct variable
openai_api_key = os.getenv("OPENAI_API_KEY")

# MongoDB connection details
mongo_uri = os.getenv("mongo_uri")
db_name = "yoomdb"
collection_name = "yoomkam"
field_name = ""

st.title("YOOM's Document Q&A")
st.write("please continue to upload the documents and ask questions based on the meeting context")

def get_mongo_client():
    """Establish a connection to MongoDB."""
    try:
        client = MongoClient(
            mongo_uri,
            tlsCAFile=certifi.where(),
            serverSelectionTimeoutMS=5000,
            tls=True
        )
        # Validate the connection
        client.admin.command('ping')
        return client
    except (ConnectionFailure, ServerSelectionTimeoutError) as e:
        st.error(f"MongoDB Connection Error: {e}")
        return None

def initialize_mongodb():
    client = get_mongo_client()
    if client:
        try:
            db = client[db_name]
            collection = db[collection_name]
            
            # Check if the collection is empty
            if collection.count_documents({}) == 0:
                new_document = {
                    "name": "Example Document",
                    "description": "This is a new document because none existed.",
                    "created_at": "2024-09-13",
                    field_name: "This is an example text content."
                }
                result = collection.insert_one(new_document)
                st.success(f"Inserted new document with ID: {result.inserted_id}")
            else:
                st.info("MongoDB collection already contains documents.")
            
            return True
        except Exception as e:
            st.error(f"An error occurred while initializing MongoDB: {e}")
            return False
        finally:
            client.close()
    else:
        st.warning("Unable to connect to MongoDB. Please check your connection string and credentials.")
        return False

# Initialize MongoDB on app startup
if initialize_mongodb():
    st.success("MongoDB initialized successfully!")

uploaded_file = st.file_uploader("Choose a text file", type="txt")

if uploaded_file is not None:
    content = uploaded_file.read().decode("utf-8")
    
    client = get_mongo_client()
    if client:
        try:
            db = client[db_name]
            collection = db[collection_name]
            
            document = {field_name: content}
            collection.insert_one(document)
            st.success("File uploaded and stored in MongoDB successfully!")
        except Exception as e:
            st.error(f"An error occurred while storing the document: {e}")
        finally:
            client.close()
    else:
        st.warning("Unable to connect to MongoDB. Please check your connection string and credentials.")

# Initialize OpenAI LLM
llm = ChatOpenAI(api_key=openai_api_key, model_name="gpt-3.5-turbo")

# Define prompt template for question-answering
prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question.
    <context>
    {context}
    </context>
    Question: {input}
    """
)

def vector_embedding():
    """Create embeddings from documents in MongoDB."""
    if "vectors" not in st.session_state:
        st.session_state.embeddings = OpenAIEmbeddings(api_key=openai_api_key)
        
        try:
            loader = MongodbLoader(
                connection_string=mongo_uri,
                db_name=db_name,
                collection_name=collection_name
            )
            docs = loader.load()
            
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            final_documents = text_splitter.split_documents(docs[:20])  # Limiting to first 20 documents for now
            st.session_state.vectors = FAISS.from_documents(final_documents, st.session_state.embeddings)
            return True
        except Exception as e:
            st.error(f"An error occurred while loading documents: {e}")
            return False
    return True

# User input for questions
prompt1 = st.text_input("Enter Your Question About the Documents")

if st.button("Load Documents and Create Embeddings"):
    if vector_embedding():
        st.write("Vector Store DB is ready!")
    else:
        st.warning("Failed to create Vector Store DB. Please check your MongoDB connection.")

if prompt1:
    if "vectors" not in st.session_state:
        st.warning("Please load the documents first.")
    else:
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        
        start = time.process_time()
        response = retrieval_chain.invoke({'input': prompt1})
        elapsed_time = time.process_time() - start
        st.write(f"Response time: {elapsed_time} seconds")
        
        st.write(response['answer'])

        with st.expander("Document Similarity Search"):
            for i, doc in enumerate(response["context"]):
                st.write(doc.page_content)
                st.write("--------------------------------")
