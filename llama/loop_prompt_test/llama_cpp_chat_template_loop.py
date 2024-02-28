"""
Conda environment to activate before running the code : llama_cpp_cuda
conda activate llama_cpp_cuda
python llama/loop_prompt_test/llama_cpp_chat_template_loop.py
"""

import time
# from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer
# import torch
from llama_cpp import Llama

# conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=11.8 -c pytorch -c nvidia
# nvidia cuda 11.6.134 driver

# prefix = "This is a spoken dialog scenario between a teacher and a 8 years old child student. \
#     The teacher is teaching mathemathics to the child student. \
#     As the student is a child, the teacher needs to stay gentle all the time. Please provide the next valid response for the followig conversation. You play the role of a teacher."
# prompt_without_prefix = "Teacher : Hi! How are your today ? \
# Child : I am fine, and I can't wait to learn mathematics !"

def generate_next_sentence(
    my_prompt,
    my_model,
    max_tokens = 100,
    temperature = 0.3,
    top_p = 0.1,
    echo = True,
    stop = ["Q", "\n"],
    ):
    # Define the parameters
    model_output = my_model(
        my_prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        echo=echo,
        stop=stop,
    )
    return model_output

def ask_for_sentence():
    sentence = input("your answer : ")
    return sentence

def add_sentence(chat_history, total_prompt, role, sentence, end):
    if end == True:
        chat_history.append({"role": role, "content": sentence})
        # total_conv += str(role) + " : " + assistant_sentence + "\ "
        total_prompt += str(role) + " : " + sentence + "</s>"
    else :
        chat_history.append({"role": role, "content": sentence})
        # total_conv += str(role) + " : " + assistant_sentence + "\ "
        total_prompt += "[INST]" + str(role) + " : " + sentence + "[/INST]"
    return total_prompt

def add_both_sentences(total_prompt, sentence_user, sentence_model):
    total_prompt += "<s>[INST]" + sentence_user + "[/INST]"
    total_prompt += sentence_model + "</s>"
    return total_prompt

def get_total_prompt(base_instructions, total_conv, instruction):
    return base_instructions + total_conv + "</s>\ " + instruction



CONV_LENGTH = 10
CONTEXT_SIZE = 512
N_GPU_LAYERS = 100

def main():

    model_path = "./models/mistral-7b-instruct-v0.2.Q4_K_S.gguf"

    my_model = Llama(model_path=model_path, n_ctx=CONTEXT_SIZE, n_gpu_layers=N_GPU_LAYERS, chat_format="llama-2")

    # my_prompt = "<s>[INST] This is a spoken dialog scenario between a teacher and a 8 years old child student. \
    # The teacher is teaching mathemathics to the child student. \
    # As the student is a child, the teacher needs to stay gentle all the time. Please provide the next valid response for the followig conversation.\
    # You play the role of a teacher. Here is the beginning of the conversation : [/INST] \
    # Teacher : Hi! How are your today ? \
    # Child : I am fine, and I can't wait to learn mathematics !</s>\
    # [INST] Generate the next Teacher sentence [/INST]"

    # base_instructions = "This is a spoken dialog scenario between a teacher and a 8 years old child student. \
    # The teacher is teaching mathemathics to the child student. \
    # As the student is a child, the teacher needs to stay gentle all the time. Please provide the next valid response for the followig conversation.\
    # You play the role of the teacher. Here is the beginning of the conversation :"

    # # instruction = "[INST] Generate the next Teacher sentence [/INST]"

    # total_conv = "Teacher : Hi! How are your today ? \
    #     Child : I am fine, and I can't wait to learn mathematics !\ "
    
    # my_prompt = "[INST]This is a spoken dialog scenario between a teacher and a 8 years old child student. \
    # The teacher is teaching mathemathics to the child student. \
    # As the student is a child, the teacher needs to stay gentle all the time. Please provide the next valid response for the followig conversation.\
    # You play the role of a teacher. Here is the beginning of the conversation : \
    # Teacher : Hi! How are your today ? \
    # Child : I am fine, and I can't wait to learn mathematics ![/INST]"

    user_sentence = "This is a spoken dialog scenario between a teacher and a 8 years old child student. \
    The teacher is teaching mathemathics to the child student. \
    As the student is a child, the teacher needs to stay gentle all the time. Please provide the next valid response for the followig conversation.\
    You play the role of a teacher. Here is the beginning of the conversation : \
    Teacher : Hi! How are your today ? \
    Child : I am fine, and I can't wait to learn mathematics !"
    
    total_prompt = ""

    chat_history = [
        {"role": "Teacher", "content": "Hi! How are your today ?"}, 
        {"role": "Child", "content": "I am fine, and I can't wait to learn mathematics !"},
    ]

    for i in range(CONV_LENGTH):
        print("Loading...")
        time_0 = time.time()
        # assistant_sentence = generate_next_sentence(chat_history, tokenizer, model)
        # model_output = generate_next_sentence(get_total_prompt(base_instructions, total_conv, instruction), my_model)
        prompt = total_prompt + "[INST]" + user_sentence + "[/INST]"
        model_output = generate_next_sentence(prompt, my_model)
        time_1 = time.time()
        print(model_output)
        assistant_sentence = model_output["choices"][0]["text"].strip()
        s = assistant_sentence.split("[/INST]")
        text = s[-1]
        print("text = "+text)
        # total_prompt = add_sentence(chat_history, total_prompt, "Teacher", assistant_sentence)
        total_prompt = add_both_sentences(total_prompt, user_sentence, text)

        # print(text)
        print("Teacher answer : ")
        print("["+str(round(time_1 - time_0, 3)) + "s] " + text)
        user_sentence = ask_for_sentence()
        # total_conv = add_sentence(chat_history, total_conv, "Child", user_sentence)
        # print(user_sentence)
        # print("total conv :")
        # print(total_conv)
    

if __name__ == '__main__':
    main()

# Chat Completion API

llm = Llama(model_path="./mistral-7b-instruct-v0.2.Q4_K_M.gguf", chat_format="llama-2")  # Set chat_format according to the model you are using
llm.create_chat_completion(
    messages = [
        {"role": "system", "content": "You are a story writing assistant."},
        {
            "role": "user",
            "content": "Write a story about llamas."
        }
    ]
)