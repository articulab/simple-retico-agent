import time
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer
import torch

# conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=11.8 -c pytorch -c nvidia
# nvidia cuda 11.6.134 driver

# prefix = "This is a spoken dialog scenario between a teacher and a 8 years old child student. \
#     The teacher is teaching mathemathics to the child student. \
#     As the student is a child, the teacher needs to stay gentle all the time. Please provide the next valid response for the followig conversation. You play the role of a teacher."
# prompt_without_prefix = "Teacher : Hi! How are your today ? \
# Child : I am fine, and I can't wait to learn mathematics !"

def generate_next_sentence(chat_history, tokenizer, model):
    # print("chat_history\n")
    # print(chat_history)
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
    # print("chat_history\n")
    # {"role": "user", "content": "Hello"},
    sentence = input("your answer : ")
    chat_history.append({"role": "user", "content": sentence})
    return sentence



CONV_LENGTH = 10

def main():

    model_id = 'mediocredev/open-llama-3b-v2-chat'
    tokenizer_id = 'mediocredev/open-llama-3b-v2-chat'

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=getattr(torch,"float16"),
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    # device_map = {"":0}
    # device_map = "auto"
    device_map = "cuda"

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map=device_map,
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
        # print(user_sentence)
    

if __name__ == '__main__':
    main()