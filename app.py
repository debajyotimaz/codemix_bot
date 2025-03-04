# requirements.txt

# app.py
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import gradio as gr

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

    # Tokenize input
    inputs = tokenizer(conversation, return_tensors="pt").to(model.device)

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

    # Decode response
    response = tokenizer.decode(
        outputs[0][inputs.input_ids.shape[1]:], 
        skip_special_tokens=True
    ).strip()

    # Clean up the response
    response = response.split('\n')[0].strip()

    return response

# Create Gradio interface
demo = gr.ChatInterface(
    fn=chat_with_model,
    title="Interactive Llama-3.2-1B Chatbot",
    description="Chat with Llama-3.2-1B model - Send multiple queries and maintain conversation context",
    theme="soft"
)

# Launch the app
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
