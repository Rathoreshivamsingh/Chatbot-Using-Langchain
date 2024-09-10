from dotenv import load_dotenv

load_dotenv()

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import ReadTheDocsLoader
from langchain_groq import ChatGroq
from langchain_cohere import CohereEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_community.vectorstores import FAISS
embeddings = CohereEmbeddings(model="embed-english-light-v3.0")


def ingest_docs(): 
    #!wget -r -A.html -P rtdocs https://python.langchain.com/en/latest/= we have scrabe the content in the doc using the command and HTML content store in the directorty
    loader = ReadTheDocsLoader(r"C:\Users\Shivam singh rathore\Desktop\ChatBot\langchain-docs",encoding='utf-8')
    raw_documents = loader.load()
    print(f"loaded {len(raw_documents)} documents")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=50)
    documents = text_splitter.split_documents(raw_documents)
    for doc in documents:
        new_url = doc.metadata["source"]
        new_url = new_url.replace("langchain-docs", "https:/")
        doc.metadata.update({"source": new_url})

    print(f"Going to add {len(documents)} to Vectore STORE")

    faiss_index = PineconeVectorStore.from_documents(documents, embeddings, index_name = "myindex")
    # faiss_index.save_local("Faiss_data")
    # FAISS.load_local('Faiss_data',embeddings,allow_dangerous_deserialization=True)
    print("****Loading to vectorstore done ***")


if __name__ == "__main__":
    ingest_docs()