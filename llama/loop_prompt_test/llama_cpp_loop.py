"""
Conda environment to activate before running the code : llama_cpp_cuda
conda activate llama_cpp_cuda

python llama/loop_prompt_test/llama_cpp_inst_template_loop.py
"""

import time
from llama_cpp import Llama

def ask_for_sentence():
    sentence = input("your answer : ")
    return sentence

def print_chat_history(chat_history):
    for sentence in chat_history:
        print("[" + sentence["role"] + "] : " + sentence["content"] + "\n")

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

def generate_text_from_chat_history(
    chat_history,
    my_model,
    # max_tokens = 200,
    max_tokens = 100,
    temperature = 0.3,
    top_p = 0.1,
    # stop = ["Q", "\n"],
    ):

    # Define the parameters
    model_output = my_model.create_chat_completion(
        chat_history,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        # stop=stop,
    )
    return model_output

# Instruct with : [INST], <s>
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

# Instruct with : [INST]


# Instruct with : [INST], <<SYS>>




# Instanciate the model
# LOCAL MODELS
# MODEL_PATH = "./models/llama-2-7b-chat.Q4_K_S.gguf"
MODEL_PATH = "./models/mistral-7b-instruct-v0.2.Q4_K_S.gguf"
# MODEL_PATH = "./models/mistral-7b-v0.1.Q4_K_S.gguf"
# MODEL_PATH = "./models/zephyr-7b-beta.Q4_0.gguf"
# MODEL_PATH = "./models/llama-2-13b-chat.Q4_K_S.gguf"
# MODEL_PATH = "./models/llama-2-13b.Q4_K_S.gguf"

# DISTANT MODELS
# MODEL_PATH = "mediocredev/open-llama-3b-v2-chat"
# MODEL_PATH = "openlm-research/open_llama_3b_v2"

CONV_LENGTH = 10
CONTEXT_SIZE = 512
N_GPU_LAYERS = 100

# TEMPLATE = "mistral"
TEMPLATE = "instruct"
# TEMPLATE = "instruct_2"

def main():

    my_model = Llama(model_path=MODEL_PATH, n_ctx=CONTEXT_SIZE, n_gpu_layers=N_GPU_LAYERS)

    if TEMPLATE == "mistral":

        chat_history = [
            {"role": "system", "content": "This is a spoken dialog scenario between a teacher and a 8 years old child student. \
                The teacher is teaching mathemathics to the child student. \
                As the student is a child, the teacher needs to stay gentle all the time. Please provide the next valid response for the followig conversation.\
                You play the role of a teacher. Here is the beginning of the conversation :"},
            {"role": "user", "content": "Hello !"}, 
            {"role": "assistant", "content": "Hi! How are your today ?"},
            {"role": "user", "content": "I am fine, and I can't wait to learn mathematics !"},
        ]

        for i in range(CONV_LENGTH):

            time_0 = time.time()
            model_output = generate_text_from_chat_history(chat_history, my_model)
            time_1 = time.time()

            # print(model_output)
            role = model_output["choices"][0]["message"]['role']
            text = model_output["choices"][0]["message"]['content']
            # print("\n["+str(round(time_1 - time_0, 3)) + "s] " + role + " : " + text)
            assert role == "assistant"
            chat_history.append({"role":role, "content":text})

            print_chat_history(chat_history)
            print("\n["+str(round(time_1 - time_0, 3)) + "s]")
            user_sentence = ask_for_sentence()
            chat_history.append({"role":"user", "content":user_sentence})

            # print(chat_history)

    if TEMPLATE == "instruct_2":

        chat_history = [
            {"role": "system", "content": "This is a spoken dialog scenario between a teacher and a 8 years old child student. \
                The teacher is teaching mathemathics to the child student. \
                As the student is a child, the teacher needs to stay gentle all the time. Please provide the next valid response for the followig conversation.\
                You play the role of a teacher. Here is the beginning of the conversation :"},
            {"role": "user", "content": "Hello !"}, 
            {"role": "assistant", "content": "Hi! How are your today ?"},
            {"role": "user", "content": "I am fine, and I can't wait to learn mathematics !"},
        ]

        my_prompt = "<<SYS>>This is a spoken dialog scenario between a teacher and a 8 years old child student. \
        The teacher is teaching mathemathics to the child student. \
        As the student is a child, the teacher needs to stay gentle all the time. Please provide the next valid response for the followig conversation.\
        You play the role of a teacher. Here is the beginning of the conversation :<</SYS>>\
        [INST] Child : Hello ! [/INST]\
        Teacher : Hi! How are your today ?\
        [INST]Child : I am fine, and I can't wait to learn mathematics ![/INST]"

        for i in range(CONV_LENGTH):
            print("Loading...")
            time_0 = time.time()
            model_output = generate_next_sentence(my_prompt, my_model)
            time_1 = time.time()
            # print(model_output)
            assistant_sentence = model_output["choices"][0]["text"].strip()
            text = assistant_sentence.split("[/INST]")[-1].split(":")[-1]
            # print("text = "+text)
            my_prompt += text + "\n"
            chat_history.append({"role":"assistant", "content":text})
            
            print(my_prompt)
            print_chat_history(chat_history)
            print("["+str(round(time_1 - time_0, 3)) + "s]")
            user_sentence = ask_for_sentence()
            my_prompt += "[INST]Child : " + user_sentence + "[/INST]"
            chat_history.append({"role":"user", "content":user_sentence})
            

    if TEMPLATE == "instruct":

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
    

if __name__ == '__main__':
    main()