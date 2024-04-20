from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Pinecone
from pinecone import Pinecone
import os
from dotenv import load_dotenv

load_dotenv()

#define environment variables
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_API_ENV = os.getenv('PINECONE_API_ENV')
HF_TOKEN = os.getenv('HUGGINGFACEHUB_API_TOKEN')
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
PINECONE_NAME_SPACE = os.getenv("PINECONE_NAME_SPACE")

def verbalize_pinecone_response(response,original_chunks):
    #This function assumes that the ids of the matches are numerical
    #and correspond to the indexes on the original_chunks list
    for match in response.matches:
        print(match.id,original_chunks[int(match.id)-1].page_content,"\n")

# load doc
loader = UnstructuredMarkdownLoader("") ### us file name as parameter 
data = loader.load()
#split text
text_splitter=RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=0)
docs=text_splitter.split_documents(data)

#create embeddings
embeddings=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
embedded_data_list = embeddings.embed_documents([t.page_content for t in docs])

#initialize index
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)

#embed query
query="Can I use mongosh?"
embedded_query = embeddings.embed_query(query)

#query pinecone
response = index.query(
  vector=embedded_query,
  top_k=10,
  namespace=PINECONE_NAME_SPACE,
  include_metadata=True
)

print(response)



