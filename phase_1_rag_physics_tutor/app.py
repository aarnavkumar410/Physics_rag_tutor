import sys
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser 
from operator import itemgetter


#from langchain_ollama import OllamaLLM
print("Initializing retriever and model...")

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

vector_store = Chroma(
    persist_directory=r"C:\Users\Aarna\OneDrive\Physics_rag_tutor\chroma_db",
    embedding_function=embeddings
    )

retriever = vector_store.as_retriever(search_kwargs={"k": 3})


print("Retriever initialized and ready to use.")
# Initialize the model running via Ollama
print("Database connected! Waking up Llama 3...")
llm = OllamaLLM(model="llama3.1")

prompt_template = """
You are a physics tutor. Use the following pieces of retrieved textbook context to answer the questions.
If you don't know the answer, say that you don't know. Do not make up answers or use any other information.
Keep your answers concise and to the point. Do not make up formulas.

Chat History: {chat_history}

Context: {context}

Question: {question}

Answer:
"""
prompt = PromptTemplate(
    template = prompt_template,
    input_variables=["context", "chat_history","question"]
)

rag_chain = prompt | llm | StrOutputParser()

def format_docs(docs): 
    """Helper function to combine retrieved text chunks into one big string"""
    return "\n\n".join(doc.page_content for doc in docs)

chat_history = []

print("Physics Tutor is avaiable, ask your questions! Type 'quit' to exit.")

while True:
    user_input = input("You: ")

    if user_input.lower() in ["quit", "exit"]:
        print("Goodbye!")
        break
    
    retrieved_docs = retriever.invoke(user_input)
    context_text = format_docs(retrieved_docs)

    recent_history = "\n".join(chat_history[-4:])
    
    inputs = {
        "context": context_text,
        "chat_history": recent_history,
        "question": user_input
    }

    print("Tutor: ", end="", flush=True)
    ai_response = ""

    for chunk in rag_chain.stream(inputs):
        sys.stdout.write(chunk)
        sys.stdout.flush()
        ai_response += chunk
    print("\n")

    print(" [Sources used:]")
    for doc in retrieved_docs:
        volume = doc.metadata.get("volume", "Unknown Volume")
        page = doc.metadata.get("page", "Unknown Page")
        print(f" - {volume}, Page {page}")


    chat_history.append(f"User: {user_input}")
    chat_history.append(f"Tutor: {ai_response}")
