# streamlit_app.py
import streamlit as st
import requests
import json # To pretty print JSON if needed for debugging

# --- Configuration ---
# The URL where your FastAPI backend is running
FASTAPI_BACKEND_URL = "http://localhost:8000/ask"

# --- Streamlit UI ---
st.set_page_config(page_title="Changi & Jewel Chatbot", page_icon="✈️")

st.title("✈️ Changi & Jewel Chatbot")
st.markdown("Ask me anything about Changi Airport or Jewel Changi Airport!")

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["sources"]:
            st.markdown("---")
            st.markdown("**Sources:**")
            for src in message["sources"]:
                st.markdown(f"- {src}")

# Accept user input
if prompt := st.chat_input("What would you like to know?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt, "sources": []})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Call FastAPI backend
    with st.spinner("Thinking..."):
        try:
            response = requests.post(
                FASTAPI_BACKEND_URL,
                json={"query": prompt},
                timeout=120 # Increased timeout for LLM response
            )
            response.raise_for_status() # Raise an exception for HTTP errors (4xx or 5xx)
            
            chatbot_response = response.json()
            bot_answer = chatbot_response.get("answer", "No answer received.")
            bot_sources = chatbot_response.get("sources", [])

            # Add assistant response to chat history
            st.session_state.messages.append({
                "role": "assistant",
                "content": bot_answer,
                "sources": bot_sources
            })

            # Display assistant response
            with st.chat_message("assistant"):
                st.markdown(bot_answer)
                if bot_sources:
                    st.markdown("---")
                    st.markdown("**Sources:**")
                    for src in bot_sources:
                        st.markdown(f"- {src}")

        except requests.exceptions.ConnectionError:
            st.error("Could not connect to the FastAPI backend. Please ensure your FastAPI server is running at http://localhost:8000.")
            st.session_state.messages.append({
                "role": "assistant",
                "content": "Error: Could not connect to the backend API.",
                "sources": []
            })
        except requests.exceptions.Timeout:
            st.error("The request timed out. The LLM might be taking too long to respond. Try again or check the backend server.")
            st.session_state.messages.append({
                "role": "assistant",
                "content": "Error: Request timed out.",
                "sources": []
            })
        except requests.exceptions.RequestException as e:
            st.error(f"An error occurred while communicating with the backend: {e}")
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"Error: {e}",
                "sources": []
            })
        except json.JSONDecodeError:
            st.error("Received an invalid JSON response from the backend. Check backend logs.")
            st.session_state.messages.append({
                "role": "assistant",
                "content": "Error: Invalid response from backend.",
                "sources": []
            })