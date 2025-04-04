### Objective
To demonstrate how Retrieval-Augmented Generation (RAG) can enhance LLM performance by integrating external documents into a chatbot built with Python, ChromaDB, LangChain, OpenAI, and Gradio.

This exercise will guide you through the process of building, running, and interacting with your own RAG-enabled chatbot. The chatbot uses the GPT-4o model from OpenAI to generate responses based on both general knowledge and document-specific data.

### Introduction to the Tools

Before diving into the hands-on exercise, here’s a brief introduction to the key technologies used:

Python: A widely used programming language known for its simplicity and effectiveness in data science and machine learning applications.

ChromaDB: A vector database that stores and retrieves embeddings, making it efficient for handling document-based queries.

LangChain: A framework that helps manage interactions between language models and external data sources, such as document retrieval and processing.

OpenAI: The provider of the GPT-4o model, which powers the chatbot's ability to generate human-like responses.

Gradio: A Python-based UI library that allows users to interact with machine learning models through an easy-to-use web interface.

### Troubleshooting Tips
1. API Errors: Verify the API key in .env.

2. Dependency Issues: Run pip install --upgrade pip and retry installing.

3. Document Not Indexed: Ensure PDFs are in the docs/ folder before running the app.

### Step 1: Prerequisites

Before starting, ensure you have installed all the software in the RAG/requirements.txt () file:
#### Installation Steps:
```bash
# Clone the repository
git clone https://github.com/pngimbwa/rag_hands_on_training.git

# Navigate to the RAG directory
cd rag_hands_on_training/RAG

# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```
Note: Ensure you have your OpenAI API key ready. If you dont have one, visit [OpenAI](https://platform.openai.com/docs/overview) to create an account and get your API key.

### Step 2: Environment Configuration

Create a .env file in the root of the project with the following content:
```bash
OPENAI_API_KEY=your_openai_api_key_here
```
### Step 3: Prepare Documents for RAG
Place any PDF documents you want the chatbot to access in the RAG/data folder.

The chatbot uses ChromaDB to index these documents and retrieve relevant information during queries.

Example Document:

```bash
RAG/data/Georgia_Cotton_Production_Guide.pdf
```

### Step 4: Ingest data to the database
Run this code to indest data to the database
```bash
python ingest_database.py
```
### Step 5: Run the RAG Chatbot
To start the chatbot interface:
```bash
python app.py
```
This will launch a Gradio interface accessible locally at http://127.0.0.1:7860.

### Step 6: Interact with the Chatbot
Open your web browser and go to http://127.0.0.1:7860.

Ask the chatbot general questions (e.g., "How do I manage cotton pests?").

Ask document-specific questions (e.g., "According to the uploaded document, what are the recommended soil pH levels for cotton?").
Observation:
- Without the document: Answers will be more general.
- With the document: Responses will include details from the uploaded PDF

### Step 6: Understanding the Code
Key Components Explained:
- ChromaDB: Vector store for efficient document retrieval.
- LangChain: Manages interactions between the LLM and document embeddings.
- OpenAI GPT-4o: Generates responses based on retrieved data.
- Gradio: Provides a simple user interface.
