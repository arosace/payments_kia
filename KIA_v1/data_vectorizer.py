from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Pinecone
from sentence_transformers import SentenceTransformer
from langchain.chains.question_answering import load_qa_chain
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

def from_embedding_list_to_pinecone_format(l,docs,source,file_number):
    #pinecone relies on a specific structure for th embeddings
    #we originally get embeddings in the form of a list
    #we need to pass a list dictionaries with a unique ids
    #one dictionary per embedding
    vectors=[]
    id=1
    for ll in l:
        print(id,"/",len(l))
        dic={"id":str(id)+"."+str(file_number),"values":ll,"metadata":{"text":docs[id-1].page_content,"source":source}}
        id+=1
        vectors.append(dic)
    return vectors

def vectorize_folder_content(folder_path):
    #prepare embeddings
    embeddings=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    # prepare pinecone client
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(PINECONE_INDEX_NAME)
    # read file list from folder
    file_list = os.listdir(folder_path)
    n=1
    for file in file_list:
        print(n,"/",len(file_list),"\n")
        # load doc
        loader = UnstructuredMarkdownLoader(os.path.join(folder_path,file))
        data = loader.load()
        #split text
        text_splitter=RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
        docs=text_splitter.split_documents(data)

        #create embeddings
        embedded_data_list = embeddings.embed_documents([t.page_content for t in docs])
        #add data to pinecone
        index.upsert(from_embedding_list_to_pinecone_format(embedded_data_list,docs,file,n),namespace=PINECONE_NAME_SPACE)
        n+=1

    
vectorize_folder_content("data")