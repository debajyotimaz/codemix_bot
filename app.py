import gradio as gr
from ollama import chat

# Maintain chat history
messages = []

def chat_with_ollama(message, history):
    global messages
    
    # Prepare conversation context
    conversation = "The following is a conversation between a helpful AI assistant and a human.\n"
    for human, assistant in history:
        conversation += f"Human: {human}\nAssistant: {assistant}\n"
    conversation += f"Human: {message}\nAssistant:"

    # Get response from Ollama
    response = chat(
        'llama3.2:1b',
        messages=messages + [{'role': 'user', 'content': conversation}]
    )
    
    # Extract and store response
    assistant_response = response.message.content.strip()
    messages.append({'role': 'user', 'content': message})
    messages.append({'role': 'assistant', 'content': assistant_response})

    return assistant_response

# Create Gradio interface
demo = gr.ChatInterface(
    fn=chat_with_ollama,
    title="Interactive Llama-3.2-1B Chatbot (Ollama)",
    description="Chat with Llama-3.2-1B using Ollama for faster inference.",
    theme="soft"
)

# Launch the app
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
