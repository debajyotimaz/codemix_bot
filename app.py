import time
import requests
import gradio as gr
import ollama

# Wait for Ollama to start
def wait_for_ollama(max_attempts=30):
    for attempt in range(max_attempts):
        try:
            response = requests.get("http://localhost:11434/api/status", timeout=5)
            if response.status_code == 200:
                print("✅ Ollama is running!")
                return True
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
            print(f"⏳ Waiting for Ollama to start... (Attempt {attempt + 1}/{max_attempts})")
            time.sleep(2)
    
    print("❌ Ollama failed to start. Check logs.")
    return False

# Chat function
def chat_with_ollama(user_input):
    try:
        response = ollama.chat(
            model="llama3.2:1b", 
            messages=[{"role": "user", "content": user_input}]
        )
        return response['message']['content'].strip()
    except Exception as e:
        print(f"Error in chat: {e}")
        return "Sorry, I encountered an error processing your request."

# Create Gradio interface
def gradio_chat(message, history):
    # Convert Gradio history format to a list of messages
    ollama_history = [
        {"role": "user" if msg[0] else "assistant", "content": msg[1]} 
        for msg in history
    ]
    
    # Add current user message
    ollama_history.append({"role": "user", "content": message})
    
    try:
        response = ollama.chat(
            model="llama3.2:1b", 
            messages=ollama_history
        )
        return response['message']['content'].strip()
    except Exception as e:
        print(f"Error in Gradio chat: {e}")
        return "Sorry, I encountered an error processing your request."

def main():
    # Wait for Ollama to start
    if not wait_for_ollama():
        raise RuntimeError("Ollama did not start successfully")
    
    # Create Gradio interface
    iface = gr.ChatInterface(
        fn=gradio_chat,
        title="Ollama Chatbot",
        description="Chat with Ollama's Llama 3.2 1B model"
    )
    
    # Launch Gradio app
    iface.launch(server_name="0.0.0.0", server_port=7860)

if __name__ == "__main__":
    main()
