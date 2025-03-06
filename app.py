# app.py
import gradio as gr
import time
import requests
from ollama import chat

# Store chat history
# Function to check if Ollama is running
def wait_for_ollama():
    while True:
        try:
            response = requests.get("http://localhost:11434")
            if response.status_code == 200:
                print("✅ Ollama is running!")
                return
        except requests.exceptions.ConnectionError:
            print("⏳ Waiting for Ollama to start...")
            time.sleep(5)

# Wait for Ollama before running the app
wait_for_ollama()

# Chat function using Ollama
messages = []

# Function to handle chat interactions using Ollama
def chat_with_ollama(message, history):
def chat_with_ollama(user_input):
    global messages

    # Append user message to history
    messages.append({'role': 'user', 'content': message})

    # Generate response from Ollama
    response = chat('llama3.2:1b', messages=messages)

    # Append assistant response to history
    messages.append({'role': 'assistant', 'content': response.message.content})

    return response.message.content.strip()
    response = chat(
        'llama3.2:1b',
        messages=messages + [{"role": "user", "content": user_input}]
    )

# Create Gradio interface
demo = gr.ChatInterface(
    fn=chat_with_ollama,
    title="Ollama LLaMA-3.2-1B Chatbot",
    description="Chat with LLaMA-3.2-1B using Ollama as inference.",
    theme="soft"
)
    messages.append({"role": "user", "content": user_input})
    messages.append({"role": "assistant", "content": response["message"]["content"]})

# Launch the app
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
    return response["message"]["content"].strip()
