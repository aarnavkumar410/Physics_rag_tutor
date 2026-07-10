import os
import sys
import re
import warnings
print("\n[*] Booting up AI Physics Tutor...")
print("[*] Configuring environment and muting warnings...")
warnings.filterwarnings("ignore")
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import transformers
from sympy import sympify
from tqdm import tqdm
from functools import partialmethod
from huggingface_hub import logging
from langchain_ollama import OllamaLLM 
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate

tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)
logging.set_verbosity_error()

if not os.environ.get("HF_TOKEN"):
    print("\n[Optimization Note]: Running unauthenticated. Add HF_TOKEN to your .env file for faster downloads.")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
db_path = os.path.join(BASE_DIR, "chroma_db")
print(f"[*] Locating textbook database at: {db_path}")
print("[*] Loading embedding models and vector store (this may take a moment)...")
db = Chroma(persist_directory=db_path, embedding_function=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"))
print("[*] Connecting to local Llama 3.1 instance...")

def calculate(expression):
    try:
        result = sympify(expression)
        return f"Computation Result: {result}"
    except Exception as e:
        return f"Error parsing math expression: {str(e)}"

def search_textbook(query):
    try:
        docs = db.similarity_search(query, k=3)

        if not docs:
            return "Observation: No relevant information found."

        unique_docs = []
        seen_content = set()

        for doc in docs:
            content_hash = doc.page_content.strip()
            if content_hash not in seen_content:
                seen_content.add(content_hash)
                unique_docs.append(doc)
        
        result_text = "Retrieved Textbook Context:\n"
        for doc in unique_docs:
            volume = doc.metadata.get("volume", "Unknown Source")
            result_text += f"- (Source: {volume}): {doc.page_content}\n"

        return result_text
    except Exception as e:
        return f"Error accessing vector database: {str(e)}"


AVAILABLE_TOOLS = {
    "math_tool": calculate,
    "search_textbook": search_textbook
}

llm = OllamaLLM(model="llama3.1", temperature=0)

agent_template = """
You are an expert AI Physics Tutor equipped with tools. You solve problems step-by-step using a Thought, Action, and Observation loop.

Available Tools:
- math_tool[expression]: Solves mathematical equations. Input must be a valid python/sympy math string. Example: math_tool[10 * 9.81 * 5]
- search_textbook[query]: Searches local physics textbooks for concepts, formulas, or rules. Use this FIRST for finding formulas before calculating. Example: search_textbook[kinematic formula for final velocity]

RULES:
1. Never do math without the math_tool. You MUST use the math_tool for EVERY calculation, regardless of the simplicity.
2. IF the user asks a conceptual question without specific numbers, DO NOT invent numbers to calculate. Use the search_textbook tool to find the definition, and then give your Final Answer.
3. To use a tool, You MUST use this format:
Thought: Reason about what you need to do.
Action: tool_name[arguments]
4. You MUST BLINDLY trust the Observation. Your Final answer MUST return what the tool outputs. DO NOT alter the results (DO NOT ROUND THE ANSWER) or recalculate.
5. EXCLUSION RULE: Your Final Answer must ONLY contain the physics explanation or solution for the student. You are strictly FORBIDDEN from mentioning the inner workings of the agent, the system, or whether you did or did not use tools. NEVER include meta-notes or post-scripts about your process. 

To use a tool, you MUST use the exact format:
Thought: Reason about what you need to do.
Action: tool_name[arguments]

When you have the final answer after observing tool results, or if no tools are needed, use this format:

Thought: I have the final answer.
Final Answer: Your complete, helpful response to the student.

Start.

Question: {question}
{agent_scratchpad}"""

prompt = PromptTemplate(
    template=agent_template, 
    input_variables=["question", "agent_scratchpad"]
)

print("\n Phase 2 Autonomous Agent Loop Online!")
print("Type 'exit' to quit.\n")

while True:
    user_question = input("You: ")
    if user_question.lower() == 'exit':
        break
    
    DEBUG_MODE = False
    scratchpad = "" 
    max_iterations = 5 
    
    for i in range(max_iterations):
        formatted_prompt = prompt.format(question=user_question, agent_scratchpad=scratchpad)
        response = llm.invoke(formatted_prompt, stop=["Observation:"])
        
        scratchpad += response + "\n"
        
        if "Final Answer:" in response:
            final_output = response.split("Final Answer:")[-1].strip()
            print(f"\nTutor: {final_output}\n")
            break
            
        action_match = re.search(r"Action:\s*(\w+)\[(.*?)\]", response)
        
        if action_match:
            tool_name = action_match.group(1)
            tool_args = action_match.group(2)
            
            if tool_name == "search_textbook":
                print(f"Searching textbook for: '{tool_args}'...")
            elif tool_name == "math_tool":
                print(f"Calculating equation: {tool_args}...")
            
            if DEBUG_MODE:
                print(f"   [Agent Action] Triggering tool '{tool_name}' with inputs: {tool_args}")
            
            if tool_name in AVAILABLE_TOOLS:
                observation = AVAILABLE_TOOLS[tool_name](tool_args)
                if DEBUG_MODE:
                    print(f"   [Agent Observation] Tool returned: {observation}")
                
                scratchpad += f"Observation: {observation}\nThought: "
            else:
                scratchpad += f"Observation: Error - Tool '{tool_name}' does not exist.\nThought: "
        else:
            print("\nTutor: I encountered a formatting error in my reasoning loop.\n")
            break