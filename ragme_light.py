import os

# Set user agent for web requests
os.environ['USER_AGENT'] = "your-user-agent-string"

# Import necessary modules from Langchain and other libraries
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# Load data from a website using WebBaseLoader
loader = WebBaseLoader("https://d-gurgurov.github.io/")
data = loader.load()

# Split the loaded documents into smaller chunks for processing
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
all_splits = text_splitter.split_documents(data)

# Initialize embeddings using the Ollama model "nomic-embed-text"
local_embeddings = OllamaEmbeddings(model="nomic-embed-text")

# Store the document chunks in a Chroma vectorstore with their embeddings
vectorstore = Chroma.from_documents(documents=all_splits, embedding=local_embeddings)

# Ask a question and retrieve the most relevant document chunks using similarity search
question = "Who is Daniil Gurgurov?"
docs = vectorstore.similarity_search(question)

# Initialize the chat model with "phi3" Ollama model
model = ChatOllama(model="phi3")

# Query the model directly with a simple question
response_message = model.invoke("Do you know who Daniil Gurgurov is?")
print(response_message.content)

# Define a prompt template to generate a humorous description of the person from the website
prompt = ChatPromptTemplate.from_template(
    "Tell me shortly about the guy described from the website in a funny way: {docs}"
)

# Convert loaded documents into strings by concatenating their content and ignoring metadata
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Chain the formatted docs into the prompt and model to generate a response
chain = {"docs": format_docs} | prompt | model | StrOutputParser()

# Print the result of invoking the chain
print("-----------------------------------")
print(chain.invoke(docs))
print("-----------------------------------")
