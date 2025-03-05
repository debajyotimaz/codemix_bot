import os
import subprocess
import time
import requests
import gradio as gr

# Function to check if Ollama is running
def is_ollama_running():
    try:
        response = requests.get("http://localhost:11434/api/version", timeout=5)
        return response.status_code == 200
    except:
        return False

# Function to start Ollama service
def start_ollama():
    # Check if Ollama is already running
    if is_ollama_running():
        print("✅ Ollama is already running")
        return True
    
    try:
        # Start Ollama service
        subprocess.Popen(["ollama", "serve"], start_new_session=True)
        
        # Wait for Ollama to start
        for _ in range(30):  # Try for 30 seconds
            if is_ollama_running():
                print("✅ Ollama service started successfully")
                return True
            time.sleep(1)
        
        print("❌ Failed to start Ollama service")
        return False
    except Exception as e:
        print(f"Error starting Ollama: {e}")
        return False

# Function to pull the model
def pull_model(model_name):
    try:
        # Pull the model using subprocess
        result = subprocess.run(
            ["ollama", "pull", model_name], 
            capture_output=True, 
            text=True,
            timeout=600  # 10-minute timeout
        )
        
        if result.returncode == 0:
            print(f"✅ Successfully pulled {model_name}")
            return True
        else:
            print(f"❌ Failed to pull {model_name}")
            print(result.stderr)
            return False
    except subprocess.TimeoutExpired:
        print(f"❌ Model pull timed out: {model_name}")
        return False
    except Exception as e:
        print(f"Error pulling model: {e}")
        return False

# Chat function using Ollama's REST API
def chat_with_ollama(message, history):
    try:
        # Prepare the API request
        payload = {
            "model": "llama3.2:1b",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."}
            ] + [
                {"role": "user" if idx % 2 == 0 else "assistant", "content": msg} 
                for sublist in history for idx, msg in enumerate(sublist)
            ] + [
                {"role": "user", "content": message}
            ]
        }
        
        # Send request to Ollama
        response = requests.post(
            "http://localhost:11434/api/chat", 
            json=payload,
            timeout=30
        )
        
        # Check response
        if response.status_code == 200:
            # Stream the response
            full_response = ""
            for line in response.iter_lines():
                if line:
                    json_response = line.decode('utf-8')
                    if '"done":true' in json_response:
                        break
                    if '"content"' in json_response:
                        chunk = json_response.split('"content":"')[1].split('","')[0]
                        full_response += chunk
        else:
            full_response = f"Error: {response.status_code} - {response.text}"
        
        return full_response
    except Exception as e:
        return f"Error in chat: {e}"

def main():
    # Start Ollama service
    if not start_ollama():
        raise RuntimeError("Ollama service failed to start")
    
    # Pull the model
    if not pull_model("llama3.2:1b"):
        raise RuntimeError("Failed to pull the model")
    
    # Create Gradio interface
    iface = gr.ChatInterface(
        fn=chat_with_ollama,
        title="Ollama Backend Chatbot",
        description="Chat with Ollama's Llama 3.2 1B model"
    )
    
    # Launch Gradio app
    iface.launch(server_name="0.0.0.0", server_port=7860)

if __name__ == "__main__":
    main()
