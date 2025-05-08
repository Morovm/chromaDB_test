from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma


loader = PyPDFLoader("python-for-everybody.pdf")
pages = loader.load_and_split()


text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
texts = text_splitter.split_documents(pages)


embeddings = SentenceTransformerEmbeddings(model_name="paraphrase-multilingual-MiniLM-L12-v2")
db = Chroma.from_documents(texts, embeddings, persist_directory="db")


query = "What is Python?"
similar_docs = db.similarity_search(query, k=2)

print("پاسخ‌های احتمالی:")
for doc in similar_docs:
    print(doc.page_content)