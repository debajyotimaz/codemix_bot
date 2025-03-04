import gradio as gr
import os
import spaces
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from threading import Thread

HF_TOKEN = os.environ.get("HF_TOKEN", None)

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B", token=HF_TOKEN)
tokenizer.pad_token = tokenizer.eos_token  # Fix padding issue ✅

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B", device_map="auto", token=HF_TOKEN)

# ✅ **Fixed Chat History Formatting**
def format_chat_history(history, message):
    chat_str = "[INST] You are a helpful AI assistant. [/INST]\n"  # System role
    for user, assistant in history:
        chat_str += f"[INST] {user} [/INST] {assistant}\n"
    chat_str += f"[INST] {message} [/INST]"
    return chat_str

# ✅ **Fix EOS Handling**
eos_token = tokenizer.eos_token_id

@spaces.GPU(duration=120)
def chat_llama3_1b(message: str, history: list, temperature: float, max_new_tokens: int):
    conversation = format_chat_history(history, message)

    # ✅ **Fix Input Tokenization**
    input_ids = tokenizer(
        conversation, return_tensors="pt", padding=True, truncation=True
    ).input_ids.to(model.device)

    # ✅ **Ensure Streamer Works Properly**
    streamer = TextIteratorStreamer(tokenizer, timeout=10.0, skip_prompt=True, skip_special_tokens=True)

    generate_kwargs = {
        "input_ids": input_ids,
        "streamer": streamer,
        "max_new_tokens": max_new_tokens,
        "do_sample": temperature > 0,
        "temperature": temperature,
        "eos_token_id": eos_token,
        "pad_token_id": tokenizer.pad_token_id,  # Fix padding ✅
    }

    # ✅ **Fix Streaming**
    model.generate(**generate_kwargs)

    outputs = []
    for text in streamer:
        outputs.append(text)
        yield "".join(outputs)

# ✅ **Gradio Fix: Ensures Proper Inputs**
chatbot = gr.Chatbot(height=450, label="Gradio ChatInterface")

with gr.Blocks() as demo:
    gr.Markdown("### Meta Llama3 1B Chatbot")
    gr.ChatInterface(
        fn=chat_llama3_1b,
        chatbot=chatbot,
        additional_inputs=[
            gr.Slider(0, 1, step=0.1, value=0.7, label="Temperature"),
            gr.Slider(128, 4096, step=1, value=512, label="Max new tokens"),
        ],
        cache_examples=False,
    )

if __name__ == "__main__":
    demo.launch()
