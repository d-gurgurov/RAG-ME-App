# RAG-Me: A Simple Retrieval-Augmented Generation App

This repository contains two lightweight Retrieval-Augmented Generation (RAG) applications that answer questions about me using website content. The system uses Langchain, Chroma for vector storage, and Ollama models for generating text.

## Setup Instructions

1. **Install dependencies**:
   ```
   pip3 install -r requirements.txt
   ```

2. **Install Ollama**: 
   Download and install from the official website: [Ollama](https://ollama.com/).

3. **Pull a model**:
   ```
   ollama pull phi2
   ```

## Usage

### Light Version
This version provides a funny response to the question "Who is [PERSON]?".

Run:
   python3 ragme_light.py

### Advanced Version
This version gives a detailed, normal response to the question "Where has [PERSON] worked?".

Run:
   python3 ragme_advanced.py


## My notes on RAG

- Important notions:
	- Query Translation: translate a question into a form that is more suitable for retrieval
		- Re-phrase, break-down, abstract
	- Routing: taking the question and routing it into the right place, let LLM choose the right DB 
	- Query construction: Relational DBs, Graph DBs, VectorDB
	- Retrieval: rank documents based on relevance 
	- Generation: active retrieval - question re-writing, re-retrieval, etc.

- Basics:
	- Indexing
		- Embedding methods:
			- Statistical 
			- Machine Learning based
		- Documents are loaded, split and embedded
		- Perform similarity search using the vector space model
	- Retrieval
		- Closest documents to the query
		- Cosine similarity
		- KNN answers
	- Generation

- Query Translation = FIRST STAGE:
	- Multi-query - translate the question to improve the query 
		- Re-writing - re-framing:
			- Re-writing the query in a few different way will increase the likelihood of retrieving the right documents
			- Use with parallalized retrieval 
			- Build sub-questions
			- Make the question more abstract - step-back prompting - higher level questions 
		- RAG Fusion:
			- A list of document lists coming from different questions
			- Generating multiple user queries and ranking the results using strategies like Reciprocal Rank Fusion.
			- RRF works by combining the ranks of different search queries and increases the chances for all the relevant documents to appear in the final results.
		- RAG Decomposition:
			- Breaking a question into a set of sub-questions
			- Using the answer to each sub-question as an input for the next sub-question
				- CoT style
			- It is also possible to answer all sub-questions separately and then combine them answers for the final answer
		- Step-back:
			- Abstract questions for the queries 
		- HyDE - hypothetical document expansion:
			- Using fake documents to improve the answers of LLMs
- Routing = SECOND STAGE:
	- GraphDB, RelationalDB, VectorDB
		 - Logical Routing
			- Routing to a specific topic based on the classification results
	- Query Construction:
		- Constructing a query for one of the types of possible data bases
		- i.e. create a query for SQL
	- Indexing: 
		- Multi-Representation
			- Using summaries for big documents to get out the full docs
			- Past the document that this chunk is a part of
		- RAPTOR
			- hierarchical indexing of documents
			- cluster documents, then summarize clusters 
			- create a tree of information and retrieve from there
		- ColBERT 
			- fine-tuning approach
			- for every token in a question compute a similarity between the tokens in a vector from the doc


- Dealing with Different Data Types:
	- PDFs
	- Tables
	- Images
		- Extract all elements
		- Create summaries for all elements
			- Language model for PDFs and Tables
			- Vision model for Images 
		- **Unstructured** - library used for extracting different types of data and info on it
			- PDF files - `pdfplumber`, `PyMuPDF`, or `PDFMiner`
			- Images - OCR tools like `Tesseract` to convert images of text into machine-readable text

- General Data processing pipeline for RAG:
	- Text pre-processing, 
	- Embedding creation - nomic, openai, bla bla
	- Storing embeddings - vector databases - FAISS
		- Databases optimized for storing high-dimensional vectors
		- These databases enable efficient similarity search
		- Core features:
			- Efficient Similarity Search, Scalability, Dimensionality Handling, Indexing Algorithms for faster retrieval 
		- Examples:
			- FAISS - Facebook AI Similarity Search
				- Multi-GPU support
				- Both exact and approximate nearest neighbor search options 
				- Supports product quantization - a technique to compress high-dimensional vectors into more memory-efficient representations
					- original vector is split into smaller sub-vectors
					- each sub-vector is quantized by mapping it to a predifined set of codewords - centroids
					- instead of storing the full vectors, only the index of the closest codeword is stored for each sub-vector
			- Pinecone - serverless vector database
				- all about integration, rather than infrastructure 
				- supports real-time updates and filtering
				- metadata filtering
			- Chroma - open-source
				- easy to store, index and search over vector bases
				- supports in-memory mode (very fast queries) and persistent mode (persistent storage, allows to obtain vectors even if the systems shuts down)
				- support for metadata along the results
				- real-time updataes
				- built-in visualizations



## License
This project is licensed under the [MIT] License.
