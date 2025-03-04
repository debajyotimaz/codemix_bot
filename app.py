import gradio as gr
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from threading import Thread

# Set Hugging Face token
HF_TOKEN = os.environ.get("HF_TOKEN", None)

DESCRIPTION = '''
<div>
<h1 style="text-align: center;">Meta Llama3 1B</h1>
<p>This Space demonstrates the instruction-tuned model <a href="https://huggingface.co/meta-llama/Llama-3.2-1B"><b>Meta Llama3 1B Chat</b></a>. Meta Llama3 is an open LLM designed for efficient and accurate language modeling.</p>
</div>
'''

LICENSE = """
<p/>
---
Built with Meta Llama 3
"""

PLACEHOLDER = """
<div style="padding: 30px; text-align: center; display: flex; flex-direction: column; align-items: center;">
   <h1 style="font-size: 28px; margin-bottom: 2px; opacity: 0.55;">Meta Llama3 1B</h1>
   <p style="font-size: 18px; margin-bottom: 2px; opacity: 0.65;">Ask me anything...</p>
</div>
"""

css = """
h1 {
  text-align: center;
  display: block;
}
#duplicate-button {
  margin: auto;
  color: white;
  background: #1565c0;
  border-radius: 100vh;
}
"""

# Load the tokenizer and model with authentication
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B", token=HF_TOKEN)
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B", device_map="auto", token=HF_TOKEN)

# Fix chat formatting
def format_chat_history(history, message):
    chat_str = ""
    for user, assistant in history:
        chat_str += f"[INST] {user} [/INST] {assistant}\n"
    chat_str += f"[INST] {message} [/INST]"
    return chat_str.strip()

terminators = [tokenizer.eos_token_id]

def chat_llama3_1b(message: str, history: list, temperature: float, max_new_tokens: int):
    conversation = format_chat_history(history, message)

    input_ids = tokenizer(conversation, return_tensors="pt").input_ids.to(model.device)
    
    streamer = TextIteratorStreamer(tokenizer, timeout=10.0, skip_prompt=True, skip_special_tokens=True)

    generate_kwargs = dict(
        input_ids=input_ids,
        streamer=streamer,
        max_new_tokens=max_new_tokens,
        do_sample=(temperature > 0),
        temperature=temperature,
        eos_token_id=terminators,
    )
    
    t = Thread(target=model.generate, kwargs=generate_kwargs)
    t.start()

    outputs = []
    for text in streamer:
        outputs.append(text)
        yield "".join(outputs)

# Gradio UI
with gr.Blocks(css=css) as demo:
    gr.Markdown(DESCRIPTION)
    gr.DuplicateButton(value="Duplicate Space for private use", elem_id="duplicate-button")

    gr.ChatInterface(
        fn=chat_llama3_1b,
        chatbot=gr.Chatbot(height=450, label="Gradio ChatInterface"),
        additional_inputs=[
            gr.Slider(minimum=0, maximum=1, step=0.1, value=0.95, label="Temperature"),
            gr.Slider(minimum=128, maximum=4096, step=1, value=512, label="Max new tokens"),
        ],
        examples=[
            ['How to set up a human base on Mars?'],
            ['Explain relativity to a child.'],
            ['What is 9,000 * 9,000?'],
            ['Write a birthday message with puns.'],
            ['Why would a penguin make a good king?']
        ],
        cache_examples=False,
    )
    gr.Markdown(LICENSE)

if __name__ == "__main__":
    demo.launch()
