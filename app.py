import time
import requests
from ollama import chat

# Wait for Ollama to start
def wait_for_ollama():
    for _ in range(30):  # Try for ~30 seconds
        try:
            response = requests.get("http://localhost:11434/api/status")
            if response.status_code == 200:
                print("✅ Ollama is running!")
                return
        except requests.exceptions.ConnectionError:
            pass
        print("⏳ Waiting for Ollama to start...")
        time.sleep(2)
    raise RuntimeError("❌ Ollama failed to start.")

# Call the waiting function
wait_for_ollama()

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
