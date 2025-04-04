# ==================================================
# 1. Imports
# ==================================================
# Document loader for PDF files in a directory
from langchain_community.document_loaders import PyPDFDirectoryLoader

# Text splitter to break documents into chunks
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Embeddings model from OpenAI
from langchain_openai.embeddings import OpenAIEmbeddings

# Vector store for storing embeddings
from langchain_chroma import Chroma

# Generate unique IDs for document chunks
# we use uuid4 (not uuid1, 2, or 3) because it generates a completely random 128-bit UUID, which is highly unpredictable; hence good for security-sensitive contexts.
from uuid import uuid4

# Load environment  variables (e.g., OpenAI API key)
from dotenv import load_dotenv
import os
import re  # For file parsing

# Clear any existing environment variable to isolate the .env file
if "OPENAI_API_KEY" in os.environ:
    del os.environ["OPENAI_API_KEY"]
    
# Load environment variables from .env file
load_dotenv()

# ==================================================
# 2. Configuration
# ==================================================
DATA_PATH = r"../data/"  # Directory containing PDF files
# r in r"../data" or r"../chromaDB" indicates a raw string in Python
# Directory (CHROMA_PATH) to persist (i.e., save data permanently) vector store data
CHROMA_PATH = r"../chromaDB"

# Metadata mapping for documents. Each PDF file is tagged with:
# - A general topic (e.g., "irrigation")
# - Optionally, a specific crop (e.g., cotton)
DOCUMENT_METADATA = {
    # General documents (no crop specified)
    "crop_management.pdf": {"topic": "crop_management"},
    "pest_control.pdf": {"topic": "pest_control"},
    "weather_forecasting.pdf": {"topic": "weather"},
    "market_prices.pdf": {"topic": "market_prices"},
    "government_schemes.pdf": {"topic": "government_schemes"},
    "irrigation.pdf": {"topic": "irrigation"},  # General irrigation principles
    "soil_health.pdf": {"topic": "soil_health"},  # General soil health
    # Crop-specific documents (both topic and crop specified)
    "cotton.pdf": {"topic": "crop_management", "crop": "cotton"},
    "soybean.pdf": {"topic": "crop_management", "crop": "soybean"},
    "corn.pdf": {"topic": "crop_management", "crop": "corn"},
    "cotton_irrigation.pdf": {
        "topic": "irrigation",
        "crop": "cotton",
    },  # Cotton-specific irrigation
    "soybean_irrigation.pdf": {
        "topic": "irrigation",
        "crop": "soybean",
    },  # Soybean-specific irrigation
    "corn_irrigation.pdf": {
        "topic": "irrigation",
        "crop": "corn",
    },  # Corn-specific irrigation
}

# ==================================================
# 3. Initialize Embeddings Model
# ==================================================

# Use OpenAI's "text-embedding-3-large" model for generating embeddings
embeddings_model = OpenAIEmbeddings(model="text-embedding-3-large")

# ==================================================
# 4. Initialize Vector Store
# ==================================================


def get_vector_store(collection_name):
    """
    Connect to a Chroma vector store collection.

    This function initializes a Chroma vector store collection for storing and retrieving
    vector embeddings. The collection is configured with an embedding function to convert
    text into vectors and a directory path to persist the data for future use.

    Args:
        collection_name (str): The name of the Chroma collection to connect to or create.

    Returns:
        Chroma: A Chroma vector store object that allows for adding, storing, and retrieving
        vector embeddings for similarity searches.

    Example:
        vector_store = get_vector_store("farmer_knowledge_base")
        vector_store.add_documents(docs)
    """
    # Return a Chroma vector store object configured with the specified collection name
    return Chroma(
        collection_name=collection_name,  # The name of the vector store collection to connect to or create
        embedding_function=embeddings_model,  # Function/model used to convert text data into vector embeddings
        # Directory where the vector data will be saved to disk for future reuse
        persist_directory=CHROMA_PATH,  # Persist (i.e., save data permanently) data to disk for reuse
    )


# ==================================================
# 5. Load PDF Documents
# ==================================================


def load_documents(data_path):
    """
    Load all PDF documents from the specified directory.

    This function initializes a PDF loader to scan the specified directory for PDF files
    and loads them into a list of Document objects for further processing, such as text extraction
    or embedding in a vector store.

    Args:
        data_path (str): The file path to the directory containing PDF documents.

    Returns:
        list: A list of Document objects, each representing a loaded PDF file.

    Example:
        Input: data_path = "./data/"
        Output: [Document(page_content='...', metadata={'source': 'file1.pdf'}),
                 Document(page_content='...', metadata={'source': 'file2.pdf'})]
    """
    loader = PyPDFDirectoryLoader(data_path)  # Initialize the loader
    return loader.load()  # Returns a list of Document objects


# ==================================================
# 6. Split Documents into Chunks
# ==================================================


def split_documents(raw_documents):
    """
    Split documents into smaller, overlapping chunks for improved retrieval.

    This function uses a recursive character-based text splitter to break large documents
    into smaller chunks, which enhances retrieval accuracy in vector stores.
    Overlapping chunks help maintain context across document sections.

    Args:
        raw_documents (list): A list of raw documents (as text or document objects) to be split.

    Returns:
        list: A list of smaller, overlapping text chunks ready for embedding and retrieval.

    Example:
        Input: ["This is a long document about irrigation techniques..."]
        Output: ["This is a long document about irrigation techniques...",
                 "techniques... (next part of the document)"]
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,  # Maximum characters per chunk
        chunk_overlap=100,  # Overlap between chunks to preserve context
        length_function=len,  # Function to calculate text length
        is_separator_regex=False,  # Treat separatos as literal strings
    )
    return text_splitter.split_documents(raw_documents)


# ==================================================
# 7. Add Metadata to Chunks
# ==================================================


def add_metadata_to_chunks(chunks, metadata_mapping):
    """
    Attach predefined metadata to document chunks based on their source filenames.

    This function uses metadata to automatically determine which collection(s) to search
    based on the content of the documents themselves. This approach makes the system more
    scalable and adaptable to new collections or topics without requiring manual updates.

    Args:
        chunks (list): A list of document chunks, where each chunk has metadata including its source.
        metadata_mapping (dict): A mapping of filenames to metadata.
                                 Example: {"cotton.pdf": {"topic": "crop_management", "crop": "cotton"}}

    Returns:
        list: The document chunks with updated metadata, including topic and crop information if available.

    Example:
        Input chunk metadata: {"source": "data/cotton1.pdf"}
        After processing: {"source": "data/cotton1.pdf", "topic": "crop_management", "crop": "cotton"}
    """
    # Attach metadata to document chunks based on their source file.
    for chunk in chunks:
        # Extract filename from the "source" metadata
        # Example: "data/cotton.pdf" → ["data", "cotton.pdf"] → "cotton.pdf".
        # Why?: The filename acts as a key to look up predefined metadata
        # [-1] extract the last element in a list i.e., ["data", "cotton.pdf"] -> 'cotton.pdf'
        file_name = chunk.metadata.get("source", "").split("/")[-1]
        # Use regex to remove digits from the filename (before .pdf)
        file_name = re.sub(r"\d+", "", file_name)
        # Check if the filename exists in the metadata mapping
        if file_name in metadata_mapping:
            # Merge the chunk's existing metadata with the predifined metadata
            # Example: If a chunk’s original metadata is {"source": "data/cotton.pdf"}, after merging,
            # it becomes: {"source": "data/cotton.pdf", "topic": "crop_management", "crop": "cotton"}
            chunk.metadata.update(metadata_mapping[file_name])
    return chunks


# ==================================================
# 8. Add Chunks to Vector Store
# ==================================================


def add_documents_to_collection(collection_name, documents):
    """Process and store documents in the vector store.

    This function takes a list of documents, processes them by splitting into smaller chunks,
    adds metadata, and stores them in a Chroma vector store for efficient retrieval.

    Args:
        collection_name (str): The name of the vector store collection where documents will be stored.
        documents (list): A list of documents (text data) to be processed and added to the vector store.

    Returns:
        None
    """
    # Connect to the specified collection
    vector_store = get_vector_store(collection_name)
    # Split documents into chunks
    chunks = split_documents(documents)
    # Add metadata to chunks (e.g., topic, crop)
    chunks_with_metadata = add_metadata_to_chunks(chunks, DOCUMENT_METADATA)
    # print(f"chunks_with_metadata:{chunks_with_metadata}")
    # Generate unique IDs for all chunks
    uuids = [str(uuid4()) for _ in range(len(chunks_with_metadata))]
    # Add chunks to the vector store
    vector_store.add_documents(chunks_with_metadata, ids=uuids)
    print(f"Documents added to collection: {collection_name}")


# ==================================================
# 9. Main Execution
# ==================================================
if __name__ == "__main__":
    # Load environment variables (e.g., API keys)
    load_dotenv()
    # Default collection name for farmer knowledge
    collection_name = "farmer_knowledge_base"
    # Load raw documents from the data directory
    raw_documents = load_documents(DATA_PATH)
    # Process and store documents in the vector store
    add_documents_to_collection(collection_name, raw_documents)
