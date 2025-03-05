import time
import requests
from ollama import chat

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
def chat_with_ollama(user_input):
    global messages

    response = chat(
        'llama3.2:1b',
        messages=messages + [{"role": "user", "content": user_input}]
    )

    messages.append({"role": "user", "content": user_input})
    messages.append({"role": "assistant", "content": response["message"]["content"]})

    return response["message"]["content"].strip()
