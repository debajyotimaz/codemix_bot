import gradio as gr
import subprocess

def chat_with_ollama(prompt):
    result = subprocess.run(["ollama", "run", "llama3:2:1b"], input=prompt, text=True, capture_output=True)
    return result.stdout

iface = gr.Interface(fn=chat_with_ollama, 
                     inputs="text", 
                     outputs="text",
                     title="Ollama Chatbot",
                     description="Chat with Llama3 on Hugging Face Spaces!")

iface.launch()
