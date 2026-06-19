import os
import tiktoken
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.environ.get("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = "Physics_Tutor_RAG"

pdf_volumes = [
    {"path": r"C:\Users\Aarna\OneDrive\Physics_rag_tutor\data\university_physics_volume_1.pdf", "name": "University Physics Volume 1"},
    {"path": r"C:\Users\Aarna\OneDrive\Physics_rag_tutor\data\university_physics_volume_2.pdf", "name": "University Physics Volume 2"},
    {"path": r"C:\Users\Aarna\OneDrive\Physics_rag_tutor\data\university_physics_volume_3.pdf", "name": "University Physics Volume 3"}
]

all_chunks = []

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

for volume in pdf_volumes:
    if os.path.exists(volume["path"]):
        loader = PyPDFLoader(volume["path"])
        docs = loader.load()
        chunks = text_splitter.split_documents(docs)

        for chunk in chunks:
            chunk.metadata["volume"] = volume["name"]
        
        all_chunks.extend(chunks)
    else:
        print(f"File not found: {volume['path']}")


embeddings = HuggingFaceEmbeddings(model_name = "all-MiniLM-L6-v2")
embd = embeddings.embed_documents([chunk.page_content for chunk in all_chunks])

Chroma.from_documents(
    documents=all_chunks, 
    embedding=embeddings, 
    persist_directory=r"C:\Users\Aarna\OneDrive\Physics_rag_tutor\chroma_db"
    )


