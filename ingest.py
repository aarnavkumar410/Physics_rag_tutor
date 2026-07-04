import os
import glob
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

# Basic Setup
load_dotenv()
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.environ.get("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = "Physics_Tutor_RAG"


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
data_folder = os.path.join(BASE_DIR, "data")
db_path = os.path.join(BASE_DIR, "chroma_db")


pdf_files = glob.glob(os.path.join(data_folder, "*.pdf"))
all_chunks = []
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

print("\nPHASE 1 Loading up Ingestion Engine...")
print(f"Found {len(pdf_files)} PDF(s) in the data folder.")

for volume in pdf_files:
    
    volume_name = os.path.basename(volume)
    print(f"Extracting: {volume_name}")
    loader = PyPDFLoader(volume)
    docs = loader.load()
    chunks = text_splitter.split_documents(docs)

    for chunk in chunks:
        
        chunk.metadata["volume"] = volume_name

    all_chunks.extend(chunks)
print(f"Succesfully extracted {len(all_chunks)} total chunks.")


# Database Storage
print("\nPHASE 2 building a neural network for vector embeddings...")
print("This step could take a few minutes for massive datasets)")

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

print("\nPHASE 3 Writing vectors to Chroma database")
Chroma.from_documents(
    documents=all_chunks, 
    embedding=embeddings, 
    persist_directory=db_path
)

print("\nComplete! The database is ready for the RAG agent.\n")



