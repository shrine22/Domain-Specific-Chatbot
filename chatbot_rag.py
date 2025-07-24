import os
import json
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_community.vectorstores import Pinecone as LangchainPinecone
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Load environment variables
load_dotenv()

# Pinecone configuration
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

# Ollama configuration
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral") # Ensure you have pulled this model with `ollama pull mistral`

class ChatbotRAG:
    def __init__(self):
        print("Initializing ChatbotRAG...")
        self.vector_store = self._initialize_vector_store()
        self.llm = self._initialize_llm()
        self.qa_chain = self._setup_qa_chain()
        print("ChatbotRAG initialized.")

    def _initialize_vector_store(self):
        """Initializes and returns the Pinecone vector store."""
        if not PINECONE_API_KEY or not PINECONE_ENVIRONMENT or not PINECONE_INDEX_NAME:
            raise ValueError("Pinecone API key, environment, or index name not set in .env")

        pc = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)
        
        # Ensure the index exists before trying to connect
        if PINECONE_INDEX_NAME not in pc.list_indexes().names():
            raise ValueError(f"Pinecone index '{PINECONE_INDEX_NAME}' does not exist. Please run db_manager.py first.")

        # Initialize the embedding model (same as used in embedder.py)
        embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

        # Connect to the existing Pinecone index via LangChain
        text_field = "text"  # This is the metadata field where your original text is stored
        vector_store = LangchainPinecone.from_existing_index(
            index_name=PINECONE_INDEX_NAME,
            embedding=embeddings,
            text_key=text_field
        )
        print("Pinecone vector store connected.")
        return vector_store

    def _initialize_llm(self):
        """Initializes and returns the Ollama LLM."""
        print(f"Connecting to Ollama at {OLLAMA_BASE_URL} with model {OLLAMA_MODEL}...")
        llm = Ollama(base_url=OLLAMA_BASE_URL, model=OLLAMA_MODEL)
        print("Ollama LLM initialized.")
        return llm

    def _setup_qa_chain(self):
        """Sets up the RetrievalQA chain."""
        # Define the custom prompt template
        # {context} will be replaced by the retrieved documents
        # {question} will be replaced by the user's query
        template = """You are a helpful assistant for Changi Airport and Jewel Changi Airport.
        Answer the question based only on the following context, do not make up answers:
        {context}

        Question: {question}
        """
        PROMPT = PromptTemplate(template=template, input_variables=["context", "question"])

        # Create the RetrievalQA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff", # 'stuff' means all retrieved docs are "stuffed" into the prompt
            retriever=self.vector_store.as_retriever(search_kwargs={"k": 3}), # Retrieve top 3 relevant documents
            return_source_documents=True, # Return the source documents along with the answer
            chain_type_kwargs={"prompt": PROMPT}
        )
        print("RetrievalQA chain setup complete.")
        return qa_chain

    def ask(self, query: str):
        """Processes a query and returns the answer with sources."""
        result = self.qa_chain({"query": query})
        answer = result["result"]
        source_documents = result["source_documents"]

        sources_info = []
        if source_documents:
            for i, doc in enumerate(source_documents):
                # Accessing text and metadata as per LangChain Document structure
                text = doc.page_content.replace('\n', ' ')[:100] + "..." # Snippet
                source_url = doc.metadata.get('source_url', 'N/A')
                sources_info.append(f"Source {i+1}: URL: {source_url}, Content Snippet: \"{text}\"")
        
        return {
            "answer": answer,
            "sources": sources_info
        }

if __name__ == "__main__":
    chatbot = ChatbotRAG()
    print("Chatbot ready! Type 'exit' to quit.")
    while True:
        user_query = input("You: ")
        if user_query.lower() == 'exit':
            break
        response = chatbot.ask(user_query)
        print(f"Chatbot: {response['answer']}")
        if response['sources']:
            print("Sources:")
            for src in response['sources']:
                print(f"  - {src}")
        print("-" * 50)