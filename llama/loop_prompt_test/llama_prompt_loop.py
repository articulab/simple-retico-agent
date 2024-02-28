"""
Conda environment to activate before running the code : llama
conda activate llama
"""

import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

def generate_next_sentence(chat_history, tokenizer, model):
    input_ids = tokenizer.apply_chat_template(
        chat_history,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)
    output_tokens = model.generate(
        input_ids,
        repetition_penalty=1.05,
        max_new_tokens=1000,
    )
    output_text = tokenizer.decode(
        output_tokens[0][len(input_ids[0]) :], skip_special_tokens=True
    )
    chat_history.append({"role": "assistant", "content": output_text})
    return output_text

def ask_for_sentence(chat_history):
    sentence = input("your answer : ")
    chat_history.append({"role": "user", "content": sentence})
    return sentence

def generate_next_sentence_text_gen(prompt, tokenizer, model):
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs.input_ids
    input_ids = input_ids.to(model.device)

    output_tokens = model.generate(
        input_ids=input_ids,
        repetition_penalty=1.05,
        max_new_tokens=100
    )
    output_text = tokenizer.decode(output_tokens[0])
    return output_text

def ask_for_sentence_text_gen(prompt):
    sentence = input("your answer : ")
    return sentence

CONV_LENGTH = 10
DEVICE_MAP = "auto"
# PROMPT_TYPE = "conversational"
PROMPT_TYPE = "text-generation"

def main():

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=getattr(torch,"float16"),
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    if PROMPT_TYPE == "conversational":

        MODEL_PATH = "mediocredev/open-llama-3b-v2-chat"

        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            device_map=DEVICE_MAP,
            quantization_config=bnb_config
        )

        chat_history = [
            {"role": "user", "content": "Hello"}, 
            {"role": "assistant", "content": "Hello! I am your math teacher and I will teach you addition today."},
            {"role": "user", "content": "I am your 8 years old child student and I can't wait to learn about mathematics !"},
        ]

        for i in range(CONV_LENGTH):
            print("Loading...")
            time_0 = time.time()
            assistant_sentence = generate_next_sentence(chat_history, tokenizer, model)
            time_1 = time.time()
            print("["+str(round(time_1 - time_0, 3)) + "s] " + assistant_sentence)
            user_sentence = ask_for_sentence(chat_history)


    if PROMPT_TYPE == "text-generation":

        MODEL_PATH = "openlm-research/open_llama_3b_v2"

        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            device_map=DEVICE_MAP,
            quantization_config=bnb_config
        )

        prompt = "This is a spoken dialog scenario between a teacher and a 8 years old child student. \
        The teacher is teaching mathemathics to the child student. \
        As the student is a child, the teacher needs to stay gentle all the time. Please provide the next valid response for the followig conversation. You play the role of a teacher.\
        Here is the dialog: \
        Teacher : Hi! How are your today ? \
        Child : I am fine, and I can't wait to learn mathematics !"

        for i in range(CONV_LENGTH):
            print("Loading...")
            time_0 = time.time()
            assistant_sentence = generate_next_sentence_text_gen(prompt, tokenizer, model)
            time_1 = time.time()
            print("["+str(round(time_1 - time_0, 3)) + "s] " + assistant_sentence)
            user_sentence = ask_for_sentence_text_gen(prompt)
            prompt = assistant_sentence + "\nChild : " +user_sentence

if __name__ == '__main__':
    main()