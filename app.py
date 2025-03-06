import time
import requests
from ollama import chat



# Initialize message history
messages = []

def chat_with_ollama(user_input):
    global messages
    response = chat("llama3.2:1b", messages=messages + [{"role": "user", "content": user_input}])
    
    messages += [
        {"role": "user", "content": user_input},
        {"role": "assistant", "content": response.message.content},
    ]
    
    return response.message.content.strip()

if __name__ == "__main__":
    print("🚀 Ollama Chatbot is ready! Type your message:")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            print("👋 Exiting...")
            break
        response = chat_with_ollama(user_input)
        print(f"Ollama: {response}")
