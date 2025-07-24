import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from chatbot_rag import ChatbotRAG

# Initialize FastAPI app
app = FastAPI(
    title="Changi Chatbot API",
    description="API for a RAG-based chatbot answering questions about Changi Airport and Jewel Changi Airport.",
    version="1.0.0"
)

# Initialize the chatbot globally to avoid re-loading model on each request
# This will take some time at startup
try:
    chatbot = ChatbotRAG()
    print("ChatbotRAG instance created successfully for API.")
except Exception as e:
    print(f"Failed to initialize ChatbotRAG for API: {e}")
    # You might want to handle this more gracefully in a production environment
    # e.g., by logging the error and not starting the API, or returning a 500 status.
    chatbot = None # Set to None if initialization fails

class QueryRequest(BaseModel):
    query: str

class ChatbotResponse(BaseModel):
    answer: str
    sources: list[str]

@app.get("/")
async def root():
    return {"message": "Welcome to the Changi Chatbot API! Visit /docs for API documentation."}

@app.post("/ask", response_model=ChatbotResponse)
async def ask_chatbot(request: QueryRequest):
    """
    Endpoint to ask the chatbot a question.
    """
    if chatbot is None:
        return ChatbotResponse(answer="Chatbot is not ready. Please check server logs for errors.", sources=[])
    
    try:
        response = chatbot.ask(request.query)
        return ChatbotResponse(answer=response["answer"], sources=response["sources"])
    except Exception as e:
        print(f"Error processing query: {e}")
        return ChatbotResponse(answer=f"An error occurred while processing your request: {e}", sources=[])

if __name__ == "__main__":
    # Run the FastAPI application using Uvicorn
    # The --reload option is great for development as it restarts the server on code changes
    # For production, remove --reload and set --host to a specific IP if needed
    uvicorn.run(app, host="0.0.0.0", port=8000)