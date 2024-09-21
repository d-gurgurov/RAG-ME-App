import requests
from bs4 import BeautifulSoup
import os

# Set user agent for web requests
os.environ['USER_AGENT'] = "your-user-agent-string"

# Import necessary modules from Langchain and other libraries
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from urllib.parse import urljoin, urlparse

# Function to retrieve all subpages from the main URL, excluding PDFs
def get_all_subpages(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Parse the base domain from the main URL
    parsed_url = urlparse(url)
    base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
    
    links = set()  # Use a set to avoid duplicates
    for link in soup.find_all('a', href=True):
        href = link['href']
        
        # Convert relative URLs to absolute URLs
        full_url = urljoin(base_url, href)
        
        # Only keep internal links (no PDFs) belonging to the same domain
        if full_url.startswith(base_url) and not full_url.endswith('.pdf'):
            links.add(full_url)
    
    return list(links)

# Main URL and retrieve all subpages
main_url = "https://d-gurgurov.github.io/"
subpages = get_all_subpages(main_url)

# Load content from each subpage using WebBaseLoader
loaders = [WebBaseLoader(link) for link in subpages]
data = []
for loader in loaders:
    data.extend(loader.load())

# Split the loaded documents into chunks with a specific size and overlap
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
all_splits = text_splitter.split_documents(data)

# Import necessary modules for vector store and embeddings
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

# Initialize embeddings using the Ollama model "nomic-embed-text"
local_embeddings = OllamaEmbeddings(model="nomic-embed-text")

# Store the document chunks in a Chroma vectorstore with their embeddings
vectorstore = Chroma.from_documents(documents=all_splits, embedding=local_embeddings)

# Perform a similarity search to retrieve relevant documents based on a question
question = "Where does Daniil Gurgurov work?"
docs = vectorstore.similarity_search(question)

# Import the ChatOllama model for interacting with the chat model
from langchain_ollama import ChatOllama

# Initialize the chat model with "phi3" Ollama model
model = ChatOllama(model="phi3")

# Function to format retrieved documents by concatenating their content
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Define a prompt template to generate a short description of work positions as bullet points
prompt = ChatPromptTemplate.from_template(
    "Tell me where the person described in the website has worked so far, just shortly describe the work positions as bullet points: {docs}"
)

# Create a chain that links the formatted documents, prompt, model, and output parser
chain = {"docs": format_docs} | prompt | model | StrOutputParser()

# Print the result of invoking the chain
print(chain.invoke(docs))
