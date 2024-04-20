from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import LlamaCpp
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

load_dotenv()

#define environment variables
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_API_ENV = os.getenv('PINECONE_API_ENV')
HF_TOKEN = os.getenv('HUGGINGFACEHUB_API_TOKEN')
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
PINECONE_NAME_SPACE = os.getenv("PINECONE_NAME_SPACE")

embeddings=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')


#initialize vectorstore
vectorstore = PineconeVectorStore.from_existing_index(PINECONE_INDEX_NAME,embeddings,"text",PINECONE_NAME_SPACE)

query="Tell me something you know"

model_path = "./models/llama-2-7b-chat.Q4_0.gguf"

model = LlamaCpp(
    model_path=model_path,
    streaming=True,
)

template = """Consider this context:
{context}
and answer this question:{question}
"""
prompt = ChatPromptTemplate.from_template(template)

chain = (
    {"context": vectorstore.as_retriever(), "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)

response = chain.invoke(query)
print(response)