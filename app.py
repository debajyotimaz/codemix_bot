import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import gradio as gr
from ollama import chat

# Model and tokenizer loading
model_name = "meta-llama/Llama-3.2-1B"
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    token=os.environ.get('HF_TOKEN'),
    trust_remote_code=True
)
model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    torch_dtype=torch.float16,
    device_map="auto",
    token=os.environ.get('HF_TOKEN'),
    trust_remote_code=True
)

# Conversation management function
def chat_with_model(message, history):
    # Prepare conversation context
    conversation = "The following is a conversation between a helpful AI assistant and a human.\n"
    for human, assistant in history:
        conversation += f"Human: {human}\nAssistant: {assistant}\n"
    conversation += f"Human: {message}\nAssistant:"
# Store chat history
messages = []

    # Tokenize input
    inputs = tokenizer(conversation, return_tensors="pt").to(model.device)
# Function to handle chat interactions using Ollama
def chat_with_ollama(message, history):
    global messages

    # Generate response
    outputs = model.generate(
        inputs.input_ids, 
        max_new_tokens=50,  # Limit new tokens
        num_return_sequences=1,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.2,
        pad_token_id=tokenizer.eos_token_id
    )
    # Append user message to history
    messages.append({'role': 'user', 'content': message})

    # Decode response
    response = tokenizer.decode(
        outputs[0][inputs.input_ids.shape[1]:], 
        skip_special_tokens=True
    ).strip()
    # Generate response from Ollama
    response = chat('llama3.2:1b', messages=messages)

    # Clean up the response
    response = response.split('\n')[0].strip()
    # Append assistant response to history
    messages.append({'role': 'assistant', 'content': response.message.content})

    return response
    return response.message.content.strip()

# Create Gradio interface
demo = gr.ChatInterface(
    fn=chat_with_model,
    title="Interactive Llama-3.2-1B Chatbot",
    description="Chat with Llama-3.2-1B model - Send multiple queries and maintain conversation context",
    fn=chat_with_ollama,
    title="Ollama LLaMA-3.2-1B Chatbot",
    description="Chat with LLaMA-3.2-1B using Ollama as inference.",
    theme="soft"
)
