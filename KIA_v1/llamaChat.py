from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import LlamaCpp
import os
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import MessagesPlaceholder
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from dotenv import load_dotenv
import os

load_dotenv()
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
PINECONE_NAME_SPACE = os.getenv("PINECONE_NAME_SPACE")

embeddings=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

def create_chain(vectorStore): 
    model_path = "./models/llama-2-7b-chat.Q4_0.gguf"
    model = LlamaCpp(model_path=model_path,streaming=False,temperature=0.4)

    prompt = ChatPromptTemplate.from_messages([
        ("system","Answer the following question based only on the provided context: {context} and chat history: {chat_history}. If you don't know the answer only say you do not know"),
        ("human","question: {input}")
    ])

    
    chain = create_stuff_documents_chain(
        llm=model,
        prompt=prompt
    )

    retriever = vectorStore.as_retriever(search_kwargs={"k":3})

    retrieval_chain = create_retrieval_chain(
        retriever,
        chain
    )

    return retrieval_chain

def process_chat(chain,question,chat_history):
    response = chain.invoke({
    "input":question,
    "chat_history":chat_history
    })
    return response["answer"]

if __name__ == '__main__':
    vectorstore = PineconeVectorStore.from_existing_index(PINECONE_INDEX_NAME,embeddings,"text","ea1")
    chain = create_chain(vectorstore)

    chat_history = []

    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            break
        response = process_chat(chain,user_input,chat_history)
        chat_history.append(HumanMessage(user_input))
        chat_history.append(AIMessage(content=response))
        print("Assistant:", response)





