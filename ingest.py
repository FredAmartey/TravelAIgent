import os
import time
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

# --------------------------------------------
# Set Environment Variables
# --------------------------------------------
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"
HF_TOKEN = "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"
pinecone_api_key = "pcsk_bqQk9_8JdPzQa3ZW55bAHJNAsZTJfMNWxBun8ktBSsyktP2KdgVgXsyPyFvPZWLXy9ReM"
os.environ["PINECONE_API_KEY"] = pinecone_api_key



# --------------------------------------------
# Initialize Pinecone Client and Server Specifications
# --------------------------------------------
pc = Pinecone(api_key=pinecone_api_key)
cloud = os.environ.get('PINECONE_CLOUD', 'aws')  # Default to AWS cloud.
region = os.environ.get('PINECONE_REGION', 'us-east-1')  # Default to us-east-1 region.
spec = ServerlessSpec(cloud=cloud, region=region)




# --------------------------------------------
# Define Paths to PDF Documents
# --------------------------------------------
# Dynamically fetch all PDF files in the './docs/' directory.
docs_dir = "./docs/"
pdf_paths = [os.path.join(docs_dir, file) for file in os.listdir(docs_dir) if file.endswith(".pdf")]




# --------------------------------------------
# Load Documents from PDF Files
# --------------------------------------------
# Use PyPDFLoader to load all PDF documents into LangChain-compatible format.
docs = []
namespaces = []
for pdf_path in pdf_paths:
    loader = PyPDFLoader(pdf_path)
    docs.append(loader.load())
    # Use the file name (without extension) as the namespace.
    namespaces.append(os.path.basename(pdf_path).replace(".pdf", "").lower().replace("_", "-"))




# --------------------------------------------
# Split Documents into Smaller Chunks
# --------------------------------------------
text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
chunks = [text_splitter.split_documents(doc) for doc in docs]



# --------------------------------------------
# Initialize Hugging Face Embeddings
# --------------------------------------------
embeddings = HuggingFaceInferenceAPIEmbeddings(
    api_key=HF_TOKEN, model_name="BAAI/bge-base-en-v1.5"
)


# --------------------------------------------
# Define Pinecone Index Name
# --------------------------------------------
index_name = "travel-assistant"

# Create the index if it doesn't already exist.
if index_name not in pc.list_indexes().names():
    pc.create_index(name=index_name, dimension=768, metric="cosine", spec=spec)

    # Wait until the index is ready to accept upserts.
    while not pc.describe_index(index_name).status['ready']:
        time.sleep(1)



# --------------------------------------------
# Display Index Statistics (Before Upsert)
# --------------------------------------------
print("Index before upsert:")
print(pc.Index(index_name).describe_index_stats())
print("\n")



# --------------------------------------------
# Insert Documents into Pinecone Index
# --------------------------------------------
for i, chunk in enumerate(chunks):
    namespace = namespaces[i]  # Map each document to its namespace.
    print(f"Upserting documents into namespace: {namespace}")
    
    # Create a PineconeVectorStore for the current namespace and upsert documents.
    docsearch = PineconeVectorStore.from_documents(
        documents=chunk,
        index_name=index_name,
        embedding=embeddings,
        namespace=namespace
    )
    
    # Add a delay to avoid overloading the Pinecone server during upserts.
    time.sleep(5)



# --------------------------------------------
# Display Index Statistics (After Upsert)
# --------------------------------------------
print("Index after upsert:")
print(pc.Index(index_name).describe_index_stats())
print("\n")
time.sleep(2)
