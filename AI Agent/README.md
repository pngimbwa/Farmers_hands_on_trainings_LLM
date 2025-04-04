### Objective
To demonstrate how an AI Agent can improve task automation and decision-making by combining reasoning capabilities with external tools. This hands on practice uses Python, OpenAI, and tool integrations (i.e., weather API) to build a AI agent recommendation tool.

### Introduction to the Tools

Before diving into the hands-on exercise, here’s a brief introduction to the key technologies used:

Python: A widely used programming language known for its simplicity and effectiveness in data science and machine learning applications.

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
cd HandsOnTraining/AI Agent

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

### Step 3: Run the RAG Chatbot
To start the AI Agent interface:
```bash
python farmer_ai_agent.py
```
This will launch a Gradio interface accessible locally at http://127.0.0.1:7860.

### Step 4: Interact with the AI Agent
Open your web browser and go to http://127.0.0.1:7860.

Enter latitude and longitude, crop of the area of your interest to get the recommendation.

### Step 5: Understanding the Code
Key Components Explained:
- OpenAI gpt-4o: Generates responses based on retrieved data.
- Gradio: Provides a simple user interface.