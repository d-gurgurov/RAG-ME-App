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

## License
This project is licensed under the [MIT] License.
