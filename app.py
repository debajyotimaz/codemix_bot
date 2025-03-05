# app.py
import gradio as gr
from ollama import chat

# Store chat history
messages = []

# Function to handle chat interactions using Ollama
def chat_with_ollama(message, history):
    global messages

    # Append user message to history
    messages.append({'role': 'user', 'content': message})

    # Generate response from Ollama
    response = chat('llama3.2:1b', messages=messages)

    # Append assistant response to history
    messages.append({'role': 'assistant', 'content': response.message.content})

    return response.message.content.strip()

# Create Gradio interface
demo = gr.ChatInterface(
    fn=chat_with_ollama,
    title="Ollama LLaMA-3.2-1B Chatbot",
    description="Chat with LLaMA-3.2-1B using Ollama as inference.",
    theme="soft"
)

# Launch the app
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
