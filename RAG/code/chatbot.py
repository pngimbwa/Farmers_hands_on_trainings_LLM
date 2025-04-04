# ==================================================
# 1. Imports
# ==================================================
# Import the 'os' module to interact with the operating system.
# This is useful for tasks like accessing environment variables, file paths, and directory management.
import os

# Import the 'requests' module to make HTTP requests.
# It allows you to send GET, POST, and other types of requests to APIs, such as fetching weather data from an API.
import requests

# Chat model and embeddings from OpenAI
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Vector store for retrieving embeddings
from langchain_chroma import Chroma

# UI framework for building the chatbot interfaces
import gradio as gr

# Load environment variables
from dotenv import load_dotenv

# Fuzzy string matching for keyword detection
from fuzzywuzzy import process

# Load the .env file from the parent directory
env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")

# Clear any existing environment variable to isolate the .env file
if "OPENAI_API_KEY" in os.environ:
    del os.environ["OPENAI_API_KEY"]

# Load environment variables from .env file
load_dotenv(dotenv_path=env_path)

# ==================================================
# 2. Configuration
# ==================================================

CHROMA_PATH = r"../chromaDB"  # Directory (CHROMA_PATH) to persist (i.e., save data permanently) vector store data
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # OPENAI API key from .env file
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")  # Weather API key from .env file
# OpenWeatherMap API endpoint
WEATHER_API_URL = "http://api.openweathermap.org/data/2.5/weather"

# ==================================================
# 3. Initialize Embeddings Model
# ==================================================

# Use OpenAI's "text-embedding-3-large" model
embeddings_model = OpenAIEmbeddings(model="text-embedding-3-large")

# ==================================================
# 4. Initialize Chat Model
# ==================================================

# Initialize a ChatOpenAI instance using the GPT-4o model
llm = ChatOpenAI(
    temperature=0.5,  # Controls the randomness of the output; 0.5 gives a balance between creativity and determinism
    model="gpt-4o",  # Specifies the model to use; "gpt-4o" is a more advanced and optimized version of GPT-4,
    # max_tokens=150,  # The maximum number of tokens (words or parts of words) in the response (here is 150)
    api_key=OPENAI_API_KEY,
)

# ==================================================
# 5. Connect to Chroma Vector Store
# ==================================================


def get_vector_store(collection_name):
    """Connect to a Chroma collection.
    This function initializes and connects to a Chroma vector store collection,
    which is used to store and retrieve document embeddings for similarity searches.

    Args:
        collection_name (str): The name of the Chroma collection to connect to or create.

    Returns:
        Chroma: A Chroma vector store object configured with the specified collection.
    """
    # Return a Chroma vector store object configured with the specified collection name
    return Chroma(
        collection_name=collection_name,  # The name of the vector store collection to connect to or create
        embedding_function=embeddings_model,  # Function/model used to convert text data into vector embeddings
        # Directory where the vector data will be saved to disk for future reuse
        persist_directory=CHROMA_PATH,  # Persist (i.e., save data permanently) data to disk for reuse
    )


# ==================================================
# 6. Set Up Retriever
# ==================================================


def get_retriever(collection_name):
    """Create a retriever for a Chroma colection.

    This function connects to a chroma  vector store and converts it into a retriever object.
    The retriever can be used to fetch the most relevant document chunks based on similarity searches.

    Args:
        collection_name (str): The name of the Chroma collection to connect to.

    Returns:
        retriever: A retriever object that fetches relevant document chunks
    """
    # Initialize the vector store by connecting to the specified Chroma collection
    vector_store = get_vector_store(collection_name)
    num_results = 5  # Number of document chunks to retrieve during a search/query
    # Convert the vector store into a retriever object
    # The retriever will return the top 'k' most relevant document chunks based on similarity
    return vector_store.as_retriever(search_kwargs={"k": num_results})


# ==================================================
# 7. Weather API Function
# ==================================================


def get_weather(city_name):
    """
    Fetch weather data for a given city using the OpenWeatherMap API.

    This function makes an HTTP request to the OpenWeatherMap API to retrieve real-time weather data
    for the specified city. The response includes temperature, weather description, humidity, and wind speed.

    Args:
        city_name (str): The name of the city for which to fetch weather data.

    Returns:
        str: A formatted string with the current weather information for the city.
             If the request fails, an error message is returned.

    Example:
        Input: get_weather("New York")
        Output:
            Weather in New York:
            - Temperature: 22°C
            - Description: clear sky
            - Humidity: 50%
            - Wind Speed: 3.5 m/s
    """
    try:
        # Make API request to OpenWeatherMap
        params = {
            "q": city_name,  # The city name to fetch weather data for
            "appid": WEATHER_API_KEY,  # API key for authentication with OpenWeatherMap
            "units": "metric",  # Use metric units (temperature in Celsius)
        }
        response = requests.get(WEATHER_API_URL, params=params)
        response.raise_for_status()  # Raise an exception for HTTP erros (e.g., 404, 500)
        # Parse and return the raw JSON weather data
        return response.json()
    except requests.exceptions.HTTPError:
        return {
            "error": f"Sorry, I couldn't retrieve the weather for {city_name}. Please check the city name."
        }
    except requests.exceptions.RequestException as e:
        # Handle HTTP and connection-related errors
        return {"error": f"An error occured while fetching the weather data {e}"}


# ==================================================
# 8. Determine Topics and Crops
# ==================================================


def determine_topics_and_crops(question):
    """
    Identify relevant topics and crops using fuzzy matching.

    Args:
        question (str): The input question or query from the user.

    Returns:
        tuple: A list of relevant topics and a list of relevant crops.

    Example:
        Input: "What's the best irrigation method for cotton?"
        Output: topics=["irrigation"], crops=["cotton"]
    """
    # Convert the question to lowercase for case-insensitivity matching
    question_lower = question.lower()
    relevant_topics = set()  # Initialize an empty set to store identified topics
    relevant_crops = set()  # Initialize an empty set to store identified crops

    # Map keywords to topics and cops for matching
    topic_mapping = {
        "crop_management": [
            "crop",
            "management",
            "cultivation",
            "planting",
            "harvesting",
        ],
        "pest_control": ["pest", "insect", "disease", "fungus", "weed"],
        "weather": ["weather", "rain", "temperature", "humidity", "climate"],
        "irrigation": ["irrigation", "water", "drip", "sprinkler", "moisture"],
        "soil_health": ["soil", "fertilizer", "nutrient", "compost", "pH"],
        "market_prices": ["market", "price", "cost", "sell", "buy"],
        "government_schemes": ["government", "scheme", "subsidy", "loan", "grant"],
        "cotton": ["cotton"],  # Crop-specific keywords
        "soybean": ["soybean", "soya"],
        "corn": ["corn", "maize"],
    }

    # Extract all keywords from topic_mapping for fuzzy matching
    all_keywords = [
        keyword for keywords in topic_mapping.values() for keyword in keywords
    ]

    # Perform fuzzy matching to find the top 5 closest keyword matches in the question
    matches = process.extract(question_lower, all_keywords, limit=5)

    # Loop through matched keywords and their similarity scores
    for matched_keyword, score in matches:
        print(f"matched_keyword:{matched_keyword},score:{score}")
        if score >= 80:  # Only consider matches with a similarity score ≥ 80.
            # Identify and add the corresponding topics based on matched keywords
            for topic, keywords in topic_mapping.items():
                if matched_keyword in keywords:
                    relevant_topics.add(topic)

            # If the matched keyword is a crop, add it to the relevant_crops set
            if matched_keyword in ["cotton", "soybean", "corn"]:
                relevant_topics.add(matched_keyword)

        # If no specific topics are identified, default to "general"
        if not relevant_topics:
            relevant_topics.add("general")

    # return the identified topics and crops as lists
    return list(relevant_topics), list(relevant_crops)


# ==================================================
# 9. Stream weather response function
# ==================================================


def stream_weather_response(message):
    """
    Generate a steaming weather response based on the user's message

    This function:
    - Checks if the message contains a weather query
    - Extracts the city name from the message.
    - Calls the weather API to get weather data.
    - Uses the LLM to generate a natural language weather report.
    - Yields the response incrementally.

    Args:
        message (str): The user's input message that includes a weather query.

    Yields:
        str: Parts of the generated weather report.
    """
    # Check if the message is about the weather (this check is optional if you
    # ensure that only weather-related messages are passed here)
    if "weather" not in message.lower():
        return

    # Extract city name from the question (e.g. What's the weather in New York)
    city_name = message.split("in")[-1].strip()  # Simple extraction
    if city_name:
        weather_data = get_weather(city_name)

        # Handle API errors
        if "error" in weather_data:
            yield weather_data["error"]
            return
        # Initialize an empty string to accumulate the streamed response
        weather_partial_message = ""
        # Use LLM to generate a natural language response based on the weather data
        weather_prompt = f"""
        You are an assistant that answers friendly weather report for {city_name} for farmers by using both the 
        provided external weather data and your internal knowledge.
        
        Follow these rules:
        
        1. **External Weather Data Priority**:
            - Use the external weather data provided below as the primary source of information.
            - Do not contradict or ignore this data.

        2. **Farmer-Friendly Language**:
            - Use simple, clear language that is easily understood by farmers.
            - Avoid technical meteorological jargon.
            - Present the information in a way that is practical and relevant to farming activities.

        3. **Internal Knowledge Fallback**:
            - If the external weather data is incomplete, supplement it with your internal knowledge.
            - Do not fabricate or hallucinate details; if uncertain, state that the information is limited.

        4. **Report Format**:
            - Clearly list the temperature, weather description, humidity, and wind speed.
            - Provide any practical advice or remarks that might help farmers plan their day.

        ---

        Weather Data for {city_name}:
        - **Temperature**: {weather_data["main"]["temp"]}°C
        - **Description**: {weather_data["weather"][0]["description"]}
        - **Humidity**: {weather_data["main"]["humidity"]}%
        - **Wind Speed**: {weather_data["wind"]["speed"]} m/s

        Please generate a concise and friendly weather report based on the above data.
        """

        # Stream the LLM-generated weather report
        for response in llm.stream(weather_prompt):
            weather_partial_message += response.content
            yield weather_partial_message
        return
    else:
        yield "Please specify a city for the weather forecast (e.g., 'What's the weather in New York?')."
        return


# ==================================================
# 10. Chatbot Response Function
# ==================================================


def stream_response(message, history):
    """
    Generate a response RAG (Retrieval-Augmented Generation)

    This function enhances the LLM's (Large Language Model) responses by incorporating
    relevant external knowledge from a document retriever, following these stepsL

    Steps:
        1. Check if the question is about the weather
        2. Detect relevant topics and crop from the user's question.
        3. Retrieve related documents from the knowledge base.
        4. Generate and stream a response using the LLM with the retrieved information
    Args:
        message (str): The user's question or input message
        history (list): A list of prevuous interactions (conversation history)

    Yields:
        str: The generated response, streamed token-by-token for real-time feedback.

    Example:
        Input: "What's the best irrigation method for cotton?"
        Output (Streamed):
            "The best irrigation method for cotton is drip irrigation, as it ensures efficient water usage."
    """
    # Step 1: Check if the question is about the weather
    if "weather" in message.lower():
        # Delegate weather query processing to the new function
        yield from stream_weather_response(message)
        return

    # Step 2: Detect topics and crops (e.g., topics = ["Irrigation"], crops = ["cotton"])
    topics, crops = determine_topics_and_crops(message)

    # Step 3: Retrieve and combine relevant knowledge from documents
    knowledge = ""  # Initialize an empty string to store retrieved document content
    for topic in topics:
        # Connect to the farmer knowledge base
        retriever = get_retriever("farmer_knowledge_base")
        # Retrieve documents based on the user's message
        docs = retriever.invoke(message)

        for doc in docs:
            # Filter documents by matching topic and crop (if crops are specified)
            if doc.metadata.get("topic") == topic:
                if not crops or doc.metadata.get("crop") in crops:
                    # Append document content to knowledge
                    knowledge += doc.page_content + "\n\n"

    # Step 4: Generate response using the LLM (with external knowledge if available)
    if message is not None:
        # Initialize an empty string to accumulate the streamed response
        partial_message = ""
        # Construct the RAG (Retrieval-Augmented Generation) prompt
        rag_prompt = f"""
        You are an assistant that answers questions using a combination of external knowledge (provided below) and your internal knowledge. Follow these rules:

        1. **External Knowledge Priority**: 
           - If external knowledge is provided, always use it as the primary source of information.
           - Do not contradict or ignore the external knowledge.

        2. **Internal Knowledge Fallback**:
           - If no external knowledge is provided, use your internal knowledge to answer the question.
           - Do not hallucinate or make up information. If you don't know the answer, say so.

        3. **Combining Knowledge**:
           - If external knowledge is provided but incomplete, you can supplement it with your internal knowledge.
           - Clearly indicate when you are using internal knowledge to supplement the external knowledge.

        4. **Avoid Hallucination**:
           - Never make up information. If the external knowledge does not provide a clear answer, and your internal knowledge is insufficient, politely inform the user that you cannot provide a definitive answer.

        ---

        The question: {message}

        Conversation history: {history}

        External knowledge: {knowledge if knowledge else "No external knowledge provided."}
        """

        # Stream the response to the Gradio app in real-time
        for response in llm.stream(rag_prompt):
            partial_message += response.content
            yield partial_message


# ==================================================
# 11. Main Execution
# ==================================================
if __name__ == "__main__":
    # Initialize Gradio Chat Interface
    chatbot = gr.ChatInterface(
        stream_response,
        textbox=gr.Textbox(
            placeholder="Ask me anything about farming...",
            container=False,
            autoscroll=True,
            scale=7,
        ),
    )
    # Launch Gradio App
    # share=True will generate a public URL that you can share with anyone. No additional network configuration is required.
    chatbot.launch(share=True)
