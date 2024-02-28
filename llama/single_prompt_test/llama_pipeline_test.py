"""
Conda environment to activate before running the code : llama
conda activate llama
"""

import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from transformers import Conversation, pipeline

DEVICE_MAP = "auto"
PIPE_TYPE = "conversational"
# PIPE_TYPE = "text-generation"

def main():
    
    if PIPE_TYPE == "text-generation":

        MODEL_PATH = "openlm-research/open_llama_3b_v2"
        prompt = "This is a spoken dialog scenario between a teacher and a 8 years old child student. \
        The teacher is teaching mathemathics to the child student. \
        As the student is a child, the teacher needs to stay gentle all the time. Please provide the next valid response for the followig conversation. You play the role of a teacher.\
        Here is the dialog: \
        Teacher : Hi! How are your today ? \
        Child : I am fine, and I can't wait to learn mathematics !"

    if PIPE_TYPE == "conversational":

        MODEL_PATH = "mediocredev/open-llama-3b-v2-chat"
        chat_history = [
            {"role": "user", "content": "Hello"}, 
            {"role": "assistant", "content": "Hello! I am your math teacher, you are a 8 years old student. This is a dialog during which I will teach you how to add two numbers together."},
            {"role": "user", "content": "Okay, I can't wait to learn about mathematics !"},
        ]
        prompt = Conversation(chat_history)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        # bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_compute_dtype=getattr(torch,"float16"),
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    m = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        device_map=DEVICE_MAP,
        quantization_config=bnb_config
    )

    text_generation_pipeline = pipeline(
        PIPE_TYPE,
        model=m,
        torch_dtype=torch.float16,
        device_map=DEVICE_MAP,
        tokenizer=tokenizer
    )

    time_0 = time.time()

    result = text_generation_pipeline(
        prompt,
        # prefix=prefix,
        do_sample=True,
        top_k=10,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=1000,
        # max_length=100,
        # min_length=20,
    )

    time_1 = time.time()

    if PIPE_TYPE == "text-generation":
        print("["+str(round(time_1 - time_0, 3)) + "s]")
        for r in result:
            print(r['generated_text'])
    if PIPE_TYPE == "conversational":
        print("["+str(round(time_1 - time_0, 3)) + "s] " + result.messages[-1]["content"])
        print()

if __name__ == '__main__':
    main()