import os
import subprocess
import sys
import time
import requests
import gradio as gr
import platform

def install_ollama():
    """Install Ollama with architecture detection"""
    try:
        # Detect system architecture
        arch = platform.machine().lower()
        
        # Mapping of architectures to Ollama download URLs
        arch_urls = {
            'x86_64': 'https://ollama.com/download/ollama-linux-amd64',
            'amd64': 'https://ollama.com/download/ollama-linux-amd64',
            'arm64': 'https://ollama.com/download/ollama-linux-arm64',
            'aarch64': 'https://ollama.com/download/ollama-linux-arm64'
        }
        
        # Select appropriate URL
        download_url = arch_urls.get(arch)
        if not download_url:
            print(f"❌ Unsupported architecture: {arch}")
            return False
        
        # Create bin directory
        os.makedirs(os.path.expanduser("~/bin"), exist_ok=True)
        ollama_path = os.path.expanduser("~/bin/ollama")
        
        # Download Ollama binary
        print(f"Downloading Ollama for {arch}...")
        download_result = subprocess.run([
            "curl", "-L", download_url, "-o", ollama_path
        ], capture_output=True, text=True)
        
        if download_result.returncode != 0:
            print(f"Download failed: {download_result.stderr}")
            return False
        
        # Make the binary executable
        subprocess.run(["chmod", "+x", ollama_path], check=True)
        
        # Add to PATH
        os.environ['PATH'] = f"{os.path.expanduser('~/bin')}:{os.environ.get('PATH', '')}"
        
        print("✅ Ollama installed successfully in user directory")
        return True
    
    except Exception as e:
        print(f"❌ Ollama installation failed: {e}")
        return False

def is_ollama_running():
    """Check if Ollama service is running"""
    try:
        response = requests.get("http://localhost:11434/api/version", timeout=5)
        return response.status_code == 200
    except:
        return False

def start_ollama():
    """Start Ollama service"""
    try:
        ollama_path = os.path.expanduser("~/bin/ollama")
        if not os.path.exists(ollama_path):
            print("Ollama binary not found. Attempting installation...")
            return False
        
        # Start Ollama service
        subprocess.Popen([ollama_path, "serve"], start_new_session=True)
        
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

def pull_model(model_name):
    """Pull Ollama model"""
    try:
        ollama_path = os.path.expanduser("~/bin/ollama")
        
        # Pull the model using subprocess
        result = subprocess.run(
            [ollama_path, "pull", model_name], 
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

def chat_with_ollama(message, history):
    """Chat with Ollama model"""
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
    # Print system information for debugging
    print(f"System Architecture: {platform.machine()}")
    print(f"Python Version: {sys.version}")
    
    # Install Ollama if not present
    if not install_ollama():
        raise RuntimeError("Failed to install Ollama")
    
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
