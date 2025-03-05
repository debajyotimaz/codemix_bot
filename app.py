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

# Rest of the script remains the same as in the previous version...

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
