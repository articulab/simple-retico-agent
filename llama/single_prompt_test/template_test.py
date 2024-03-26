# from llama_cpp import Llama
# from transformers import AutoTokenizer

# MODEL_REPO = "TheBloke/Llama-2-7B-Chat-GGUF"
# MODEL_NAME = "llama-2-7b-chat.Q4_K_S.gguf"
# MODEL_PATH = "./models/mistral-7b-instruct-v0.2.Q4_K_S.gguf"

# LLAMA_REPO = "meta-llama/Llama-2-7b-hf"
# MISTRAL_REPO = "mistralai/Mistral-7B-Instruct-v0.2"

# mistral_cpp = Llama(MODEL_PATH, echo=True)
# # mistral_cpp_llama_template = Llama(MODEL_PATH, chat_format='llama-2', echo=True)
# mistral_tokenizer = AutoTokenizer.from_pretrained(MISTRAL_REPO)
# # llama_tokenizer = AutoTokenizer.from_pretrained(LLAMA_REPO)


# ## PROMPTS

# chat_history = [
#     # {"role": "system", "content": "This is a spoken dialog scenario between a teacher and a 8 years old child student. \
#     #     The teacher is teaching mathemathics to the child student. \
#     #     As the student is a child, the teacher needs to stay gentle all the time. Please provide the next valid response for the followig conversation.\
#     #     You play the role of a teacher. Here is the beginning of the conversation :"},
#     {"role": "user", "content": "Hello !"}, 
#     {"role": "assistant", "content": "Hi! How are your today ?"},
#     {"role": "user", "content": "I am fine, and I can't wait to learn mathematics !"},
# ]

# my_prompt_llama = "<<SYS>>This is a spoken dialog scenario between a teacher and a 8 years old child student. \
# The teacher is teaching mathemathics to the child student.\
# As the student is a child, the teacher needs to stay gentle all the time. Please provide the next valid response for the followig conversation.\
# You play the role of a teacher. Here is the beginning of the conversation :<</SYS>>\
# [INST]Child : Hello ! [/INST]\
# Teacher : Hi! How are your today ?\
# [INST]Child : I am fine, and I can't wait to learn mathematics ![/INST]"
# my_prompt_llama_2 = "<s>[INST] <<SYS>>\n\
# This is a spoken dialog scenario between a teacher and a 8 years old child student. \
# The teacher is teaching mathemathics to the child student.\
# As the student is a child, the teacher needs to stay gentle all the time. Please provide the next valid response for the followig conversation.\
# You play the role of a teacher. Here is the beginning of the conversation :\n\
# <</SYS>>\n\n\
# Child : Hello ! [/INST]\
# Teacher : Hi! How are your today ?</s><s>\
# [INST] Child : I am fine, and I can't wait to learn mathematics ![/INST]"
# my_prompt_mistral = "<s>[INST]This is a spoken dialog scenario between a teacher and a 8 years old child student.\
# The teacher is teaching mathemathics to the child student.\
# As the student is a child, the teacher needs to stay gentle all the time.\
# Please provide the next valid response for the followig conversation.\
# You play the role of a teacher. Here is the beginning of the conversation :[/INST]\n\
# Child : Hello !\
# Teacher : Hi! How are your today ?\
# Child : I am fine, and I can't wait to learn mathematics !</s>\n"
# my_prompt_mistral_2 = "<s>[INST] This is a spoken dialog scenario between a teacher and a 8 years old child student.\
# The teacher is teaching mathemathics to the child student.\
# As the student is a child, the teacher needs to stay gentle all the time.\
# Please provide the next valid response for the followig conversation.\
# You play the role of a teacher. Here is the beginning of the conversation :\
# Child : Hello !\
# Teacher : Hi! How are your today ?\
# Child : I am fine, and I can't wait to learn mathematics ![/INST]\n</s>\n"
# my_prompt_mistral_3 = "<s>[INST] This is a spoken dialog scenario between a teacher and a 8 years old child student.\
# The teacher is teaching mathemathics to the child student.\
# As the student is a child, the teacher needs to stay gentle all the time.\
# Please provide the next valid response for the followig conversation.\
# You play the role of a teacher. Here is the beginning of the conversation :\
# Child : Hello ![/INST]\n\
# Teacher : Hi! How are your today ?\n</s>\n\
# <s>[INST]Child : I am fine, and I can't wait to learn mathematics ![/INST]\n"

# tm = mistral_cpp.tokenize(bytes(my_prompt))
# tl = mistral_cpp_llama_template.tokenize(chat_history)
# print(tm)
# print(tl)

# mistral_chat = mistral_tokenizer.apply_chat_template(chat_history, tokenize=False)
# llama_chat = llama_tokenizer.apply_chat_template(chat_history,tokenize=False)
# mistral_chat_tokens = mistral_tokenizer.apply_chat_template(chat_history)
# llama_chat_tokens = llama_tokenizer.apply_chat_template(chat_history)
# print(mistral_chat)
# print(llama_chat)
# print(" = ", mistral_tokenizer.encode(" "))
# print(" Hi = ", mistral_tokenizer.encode(" Hi "))
# print("[INST] = ", mistral_tokenizer.encode("[INST]"))
# print("[/INST] = ", mistral_tokenizer.encode("[/INST]"))
# print("<s> = ", mistral_tokenizer.encode("<s>"))
# print("</s> = ", mistral_tokenizer.encode("</s>"))
# print(mistral_cpp.detokenize([733]))
# print(mistral_cpp.detokenize([28792]))
# print(bytes(mistral_tokenizer.decode([733]), "utf-8"))
# print(bytes(mistral_tokenizer.decode([28792]), "utf-8"))


# # Same result
# print(mistral_tokenizer.encode("<s>[INST] Hi [/INST] Hi</s>[INST] Hi [/INST]"))
# print(mistral_cpp.tokenize(bytes("<s> [INST] Hi [/INST] Hi</s> [INST] Hi [/INST]", "utf-8"), special=True))
# print(mistral_cpp.tokenize(bytes("<s>[INST] Hi [/INST] Hi</s>[INST] Hi [/INST]", "utf-8"), special=True))

# # Same result
# # print(mistral_tokenizer.encode("<s>[INST] Hi [/INST] Hi</s> [INST] Hi [/INST]"))
# print(mistral_cpp.tokenize(bytes("<s> [INST] Hi [/INST] Hi</s> [INST] Hi [/INST]", "utf-8")))
# print(mistral_cpp.tokenize(bytes("<s>[INST] Hi [/INST] Hi</s>[INST] Hi [/INST]", "utf-8"), special=False))


# # Same result
# print(mistral_tokenizer.encode("[INST] Hi [/INST] Hi [INST] Hi [/INST]"))
# print(mistral_cpp.tokenize(bytes("[INST] Hi [/INST] Hi [INST] Hi [/INST]", "utf-8"), special=False))
# print(mistral_cpp.tokenize(bytes("[INST] Hi [/INST] Hi [INST] Hi [/INST]", "utf-8"), special=True))



# print(llama_tokenizer.encode("<s>[INST] Hi [/INST] Hi </s> [INST] Hi [/INST]"))
# print(mistral_cpp.tokenize(b"<s>[INST] Hi [/INST] Hi </s> [INST] Hi [/INST]",special=True))
# print(mistral_cpp.tokenize(b"[INST] Hi [/INST] Hi [INST] Hi [/INST]",special=True))

# tokens_1 = mistral_tokenizer.encode("<s>[INST] Hi [/INST] Hi </s> [INST] Hi [/INST]")
# tokens_2 = mistral_cpp.tokenize(b"<s>[INST] Hi [/INST] Hi </s> [INST] Hi [/INST]", special=True)
# # print(mistral_cpp.generate(tokens_1, max_tokens=0)["choices"][0]["text"])
# s_1 = b""
# for token in mistral_cpp.generate(tokens_1):
#     s_1 += mistral_cpp.detokenize([token])
# s_2 = b""
# for token in mistral_cpp.generate(tokens_2):
#     s_2 += mistral_cpp.detokenize([token])
# print(mistral_cpp("<s>[INST] Hi [/INST] Hi </s> [INST] Hi [/INST]", max_tokens=0)["choices"][0]["text"])

# print(mistral_cpp_llama_template.tokenize(bytes(my_prompt, "utf-8")))
# print(mistral_cpp.tokenize(bytes(my_prompt, "utf-8")))
# print(mistral_tokenizer(my_prompt))
# print(llama_tokenizer(my_prompt))
# print(mistral_cpp_llama_template.tokenizer_.apply_chat_template(chat_history))

# def stop_true(a,b):
#     print(mistral_cpp_llama_template.detokenize(a))
#     return True

# comp = mistral_cpp_llama_template.create_chat_completion(chat_history, stop=" ")
# print(comp)

# print(mistral_cpp(my_prompt_llama, max_tokens=0)["choices"][0]["text"])
# print(mistral_cpp(my_prompt_llama_2,max_tokens=0)["choices"][0]["text"])
# print(mistral_cpp(my_prompt_mistral,max_tokens=0)["choices"][0]["text"])
# print(mistral_cpp(my_prompt_mistral_2,max_tokens=0)["choices"][0]["text"])
# print(mistral_cpp(my_prompt_mistral_3,max_tokens=0)["choices"][0]["text"])


"""
Conda environment to activate before running the code : llama_cpp_cuda
conda activate llama_cpp_cuda
python llama/single_prompt_test/template_test.py

llama_cpp_python from API :

To understand all the different ways to generate text using the model :
__call__    ->    create_completion()    ->    _create_completion()    ->    generate()
create_chat_completion()    ->    create_completion()
"""

import re
import time
from llama_cpp import Llama
import numpy as np

def generate_special(
    my_prompt,
    my_model,
    stop_criteria,
    ):
    tokens = my_model.tokenize(bytes(my_prompt, "utf-8"), special=True)
    # print(tokens)
    last_sentence = b""
    for token in my_model.generate(
        tokens,
        stopping_criteria=stop_criteria,
        # temp=TEMP,
        # top_p=TOP_P,
        # echo=ECHO,
        reset=True
        ):        
        last_sentence += my_model.detokenize([token])
    last_sentence = remove_stop_patterns(last_sentence)
    return last_sentence

def generate_text_with_model_call(
    my_prompt,
    my_model,
    ):

    # Define the parameters
    model_output = my_model(
        my_prompt,
        max_tokens=MAX_TOKENS,
        # temperature=TEMP,
        # top_p=TOP_P,
        # echo=ECHO,
        stop=STOP_TOKEN_TEXT,
    )
    return model_output

def generate_text_with_chat_completion(
    chat_history,
    my_model,
    ):

    # Define the parameters
    model_output = my_model.create_chat_completion(
        chat_history,
        max_tokens=MAX_TOKENS,
        # temperature=TEMP,
        # top_p=TOP_P,
        # echo=ECHO,
        stop=STOP_TOKEN_TEXT,
        stream=True
    )
    return model_output

def generate_text_with_chat_completion_stream(
    my_model,
    chat_history,
    subprocess,
    ponctuation_list,
    ):

    model_output = {}
    role = ""
    for m in my_model.create_chat_completion(
        chat_history,
        max_tokens=MAX_TOKENS,
        # temperature=TEMP,
        # top_p=TOP_P,
        # echo=ECHO,
        stop=STOP_TOKEN_TEXT,
        stream=True
    ):
        print(m)
        if "role" in m["choices"][0]["delta"]:
            role = m["choices"][0]["delta"]["role"]
            model_output = m.copy()
            model_output["choices"][0]["message"] = {"role":role,"content":""}
        elif "content" in m["choices"][0]["delta"]:
            model_output["choices"][0]["message"]["content"] += m["choices"][0]["delta"]["content"]
        # model_output += m
        else :
            pass
    return model_output

def generate_text_with_generate_function(
    my_prompt,
    my_model,
    stop_criteria,
    ):
    tokens = my_model.tokenize(my_prompt)
    last_sentence = b""
    for token in my_model.generate(
        tokens,
        stopping_criteria=stop_criteria,
        # temp=TEMP,
        # top_p=TOP_P,
        # echo=ECHO,
        ):        
        last_sentence += my_model.detokenize([token])
    return last_sentence

def generate_next_sentence(
        my_model,
        my_prompt,
        subprocess,
        ponctuation_list,
        ):

        def is_ponctuation(word):
            # is_ponctuation = word in ponctuation_list
            is_ponctuation = any([ponct in word for ponct in ponctuation_list])
            return is_ponctuation

        # Define the parameters
        last_sentence = ""
        last_sentence_nb_tokens = 0
        time_0 = time.time()
        for token in my_model(
            my_prompt,
            max_tokens=0,
            stream=True
            ):
            # print("word = ", token)
            word = token["choices"][0]["text"]
            # remove_name = payload_text.split("Teacher :")
            is_ponct = is_ponctuation(word)
            subprocess(last_sentence, round(time.time() - time_0), is_ponct)
            # Update model short term memory
            last_sentence += word
            last_sentence_nb_tokens += 1

        return last_sentence, last_sentence_nb_tokens


##############
# MODEL CONFIG
##############

# LOCAL FILE
# MODEL_PATH = "./models/llama-2-7b-chat.Q4_K_S.gguf"
MODEL_PATH = "./models/mistral-7b-instruct-v0.2.Q4_K_S.gguf"
# MODEL_PATH = "./models/mistral-7b-v0.1.Q4_K_S.gguf"
# MODEL_PATH = "./models/zephyr-7b-beta.Q4_0.gguf"
# MODEL_PATH = "./models/llama-2-13b-chat.Q4_K_S.gguf"
# MODEL_PATH = "./models/llama-2-13b.Q4_K_S.gguf"

# HUGGING FACE REPO
# MODEL_REPO = "TheBloke/Mistral-7B-Instruct-v0.2-GGUF"
# MODEL_NAME = "mistral-7b-instruct-v0.2.Q4_K_S.gguf"
MODEL_REPO = "TheBloke/Llama-2-7B-Chat-GGUF"
MODEL_NAME = "llama-2-7b-chat.Q4_K_S.gguf"


#######################
# LLM PARAMETERS CONFIG
#######################

N_GPU_LAYERS = 100
DEVICE_MAP = "cuda"
# DEVICE_MAP = "auto"

# CONTEXT_SIZE = 512
CONTEXT_SIZE = 2000
# CONTEXT_SIZE = 4096
# CONTEXT_SIZE = 8192
# CONTEXT_SIZE = 32768
# CONTEXT_SIZE = 0
# CONTEXT_SIZE = 100

# STOP_TOKEN_IDS = [2]
STOP_TOKEN_IDS = []
# STOP_TOKEN_TEXT = ["</s>", "Q:", "\n"]
STOP_TOKEN_TEXT = []
STOP_TOKEN_PATTERNS = []
STOP_TOKEN_TEXT_PATTERNS = []

# GENERATION PARAMETERS
MAX_TOKENS = 0 # max_token = n_ctx
# TEMP = 0.3
# TOP_P = 0.1
# TEMP = 0.8
# TOP_P = 0.95
# TEMP = 0.6
# TOP_P = 0.9
ECHO = True


################
# CONFIG METHODS
################

# MODEL_LOADING_METHOD = "from_pretrained"
MODEL_LOADING_METHOD = "local_file"

# MODEL_GENERATION_METHOD = "stream"
# MODEL_GENERATION_METHOD = "chat_completion"
MODEL_GENERATION_METHOD = "model_call"
# MODEL_GENERATION_METHOD = "generate_function"
# MODEL_GENERATION_METHOD = "special"

# PROMPT_TEMPLATE = "mistral"
# PROMPT_TEMPLATE = "llama"
PROMPT_TEMPLATE = "instruct"
# PROMPT_TEMPLATE = "chat_mistral"
# PROMPT_TEMPLATE = "chat_llama"
# PROMPT_TEMPLATE = "chat"

CHAT_FORMAT = "llama-2"
# CHAT_FORMAT = "mistral-instruct"
# CHAT_FORMAT = "chatml"
# CHAT_FORMAT = "zephyr"

from llama_cpp.llama_chat_format import format_llama2, format_zephyr, format_mistral_instruct

def set_prompts():

    prompts = []
    long_prompts = []

    if PROMPT_TEMPLATE == "llama":
        # template = "<s>[INST] <<SYS>>\
        # {your_system_message}\
        # <</SYS>>\
        # {user_message_1} [/INST] {model_reply_1}</s><s>[INST] {user_message_2} [/INST]"

        my_prompt = "<s>[INST] <<SYS>>\
        This is a spoken dialog scenario between a teacher and a 8 years old child student. \
        # The teacher is teaching mathemathics to the child student. \
        # As the student is a child, the teacher needs to stay gentle all the time. Please provide the next valid response for the followig conversation.\
        # You play the role of a teacher. Here is the beginning of the conversation : \
        <</SYS>>\
        Child : Hello ! [/INST] \
        Teacher : Hi! How are your today ?</s>\
        <s>[INST] Child : I am fine, and I can't wait to learn mathematics ! [/INST]"
        prompts.append(my_prompt)


        # # long
        # my_prompt = "<<SYS>>  This is a spoken dialog scenario between a teacher and a 8 years old child student. \
        # The teacher is teaching mathemathics to the child student. \
        # As the student is a child, the teacher needs to stay gentle all the time. Please provide the next valid response for the followig conversation.\
        # You play the role of a teacher. Here is the beginning of the conversation : <</SYS>> \
        # [INST] Child : Hello ! \
        # Teacher : Hi! How are your today ? \
        # Child : I am fine, and I can't wait to learn mathematics ! \
        # Teacher : Okay, we will start with the simpler operation in mathematics, addition. What do you know about addition ? \
        # Child : Humm, I know that one plus one equals two ! \
        # Teacher : That's right ! And this is a good start. So, addition is the operation of adding two numbers together to get a bigger number that is their sum. In our case, we added one to one to get the bigger number two, that is their sum. Is it clear ? \
        # Child : I... I am not sure...  \
        # Teacher : Okay I will take another example, when we add one number, let's say two, to another number, let's say one, we get as a result three, a number bigger than both our previous numbers. \
        # Child : okay, I think I get it now ! [/INST]"
        # long_prompts.append(my_prompt)

        my_prompt = "<s>[INST] <<SYS>>\
        This is a spoken dialog scenario between a teacher and a 8 years old child student. \
        # The teacher is teaching mathemathics to the child student. \
        # As the student is a child, the teacher needs to stay gentle all the time. Please provide the next valid response for the followig conversation.\
        # You play the role of a teacher. Here is the beginning of the conversation : \
        <</SYS>>\
        Child : Hello ! [/INST] \
        Teacher : Hi! How are your today ?</s>\
        <s>[INST] Child : I am fine, and I can't wait to learn mathematics ! [/INST] \
        Teacher : Okay, we will start with the simpler operation in mathematics, addition. What do you know about addition ?</s>\
        <s>[INST] Child : Humm, I know that one plus one equals two ! [/INST] \
        Teacher : That's right ! And this is a good start. So, addition is the operation of adding two numbers together to get a bigger number that is their sum. In our case, we added one to one to get the bigger number two, that is their sum. Is it clear ?</s>\
        <s>[INST] Child : I... I am not sure... [/INST] \
        Teacher : Okay I will take another example, when we add one number, let's say two, to another number, let's say one, we get as a result three, a number bigger than both our previous numbers.</s>\
        <s>[INST] Child : okay, I think I get it now ! [/INST]"
        long_prompts.append(my_prompt)
        
        my_prompt = "<s>[INST] <<SYS>>\
        This is a spoken dialog scenario between a teacher and a 8 years old child student. \
        # The teacher is teaching mathemathics to the child student. \
        # As the student is a child, the teacher needs to stay gentle all the time. Please provide the next valid response for the followig conversation.\
        # You play the role of a teacher. Here is the beginning of the conversation : \
        <</SYS>>\
        Child : Hello ! [/INST] \
        Teacher : Hi! How are your today ?</s>\
        <s>[INST] Child : I am fine, and I can't wait to learn mathematics ! [/INST] \
        Teacher : Okay, we will start with the simpler operation in mathematics, addition. What do you know about addition ?</s>\
        <s>[INST] Child : Humm, I know that one plus one equals two ! [/INST] \
        Teacher : That's right ! And this is a good start. So, addition is the operation of adding two numbers together to get a bigger number that is their sum. In our case, we added one to one to get the bigger number two, that is their sum. Is it clear ?</s>\
        <s>[INST] Child : I... I am not sure... [/INST] \
        Teacher : Okay I will take another example, when we add one number, let's say two, to another number, let's say one, we get as a result three, a number bigger than both our previous numbers.</s>\
        <s>[INST] Child : okay, I think I get it now ! [/INST] \
        Teacher: Great job understanding addition so far! Now let's try a few sums together. I will say the numbers, and you tell me what number comes after when we add them together. Ready?</s>\
        <s>[INST] Child: Yes! [/INST] \
        Teacher: Alright, let's start with three and one.</s>\
        <s>[INST] Child: Four! [/INST] \
        Teacher: That's correct! Keep up the good work. Next, let's try five and two.</s>\
        <s>[INST] Child: Seven! [/INST]"
        long_prompts.append(my_prompt)

        my_prompt = "<s>[INST] <<SYS>>\
        This is a spoken dialog scenario between a teacher and a 8 years old child student. \
        # The teacher is teaching mathemathics to the child student. \
        # As the student is a child, the teacher needs to stay gentle all the time. Please provide the next valid response for the followig conversation.\
        # You play the role of a teacher. Here is the beginning of the conversation : \
        <</SYS>>\
        Child : Hello ! [/INST] \
        Teacher : Hi! How are your today ?</s>\
        <s>[INST] Child : I am fine, and I can't wait to learn mathematics ! [/INST] \
        Teacher : Okay, we will start with the simpler operation in mathematics, addition. What do you know about addition ?</s>\
        <s>[INST] Child : Humm, I know that one plus one equals two ! [/INST] \
        Teacher : That's right ! And this is a good start. So, addition is the operation of adding two numbers together to get a bigger number that is their sum. In our case, we added one to one to get the bigger number two, that is their sum. Is it clear ?</s>\
        <s>[INST] Child : I... I am not sure... [/INST] \
        Teacher : Okay I will take another example, when we add one number, let's say two, to another number, let's say one, we get as a result three, a number bigger than both our previous numbers.</s>\
        <s>[INST] Child : okay, I think I get it now ! [/INST] \
        Teacher : Great attitude! I'm glad you're excited about math today. Let's start with something simple: can you count to ten with me?</s>\
        <s>[INST]  Child: Sure thing! One, two, three, four, five, six, seven, eight, nine, ten! [/INST] \
        Teacher: Excellent job! Now, let's try something a little more challenging. Can you identify these numbers: 4, 7, and 9?</s>\
        <s>[INST]  Child: Yes! Four, seven, and nine! [/INST] \
        Teacher: That's correct. Now let's practice adding some numbers together. What do you get when you add 2 and 3?</s>\
        <s>[INST]  Child: Five! [/INST] \
        Teacher: That's right! Good job. Let's try another one: what do you get when you add 5 and 7?</s>\
        <s>[INST]  Child: Twelve! [/INST] \
        Teacher: Not quite. Try again.</s>\
        <s>[INST]  Child: Uh... eleven? [/INST] \
        Teacher: No, we get twelve. Keep trying! Remember that five and seven make twelve when you put them together.</s>\
        <s>[INST]  Child: Oh, I see! I got it now! [/INST] \
        Teacher: Great job! Keep up the good work! Now let's practice subtraction: what's five take away two?</s>\
        <s>[INST]  Child: Three! [/INST]"
        long_prompts.append(my_prompt)


    if PROMPT_TEMPLATE == "instruct":

        # # nothing
        # # Do not Work :
        # # - generate mutiple turn for both child and teacher
        # my_prompt = "This is a spoken dialog scenario between a teacher and a 8 years old child student. \
        # The teacher is teaching mathemathics to the child student. \
        # As the student is a child, the teacher needs to stay gentle all the time. Please provide the next valid response for the followig conversation.\
        # You play the role of a teacher. Here is the beginning of the conversation : \
        # Child : Hello !\
        # Teacher : Hi! How are your today ? \
        # Child : I am fine, and I can't wait to learn mathematics !"
        # prompts.append(my_prompt)

        # # SYS
        # # Do not Work :
        # # - generate mutiple turn for both child and teacher
        # my_prompt = "<<SYS>> This is a spoken dialog scenario between a teacher and a 8 years old child student. \
        # The teacher is teaching mathemathics to the child student. \
        # As the student is a child, the teacher needs to stay gentle all the time. Please provide the next valid response for the followig conversation.\
        # You play the role of a teacher. Here is the beginning of the conversation :\
        # Child : Hello !\
        # Teacher : Hi! How are your today ? <</SYS>> \
        # Child : I am fine, and I can't wait to learn mathematics !"
        # prompts.append(my_prompt)

        # # 1 INST
        # # WORKS
        # my_prompt = "[INST] This is a spoken dialog scenario between a teacher and a 8 years old child student. \
        # The teacher is teaching mathemathics to the child student. \
        # As the student is a child, the teacher needs to stay gentle all the time. Please provide the next valid response for the followig conversation.\
        # You play the role of a teacher. Here is the beginning of the conversation : \
        # Child : Hello !\
        # Teacher : Hi! How are your today ? \
        # Child : I am fine, and I can't wait to learn mathematics ! [/INST]"
        # prompts.append(my_prompt)

        # # 1 INST + SYS
        # # WORKS
        # my_prompt = "<<SYS>> This is a spoken dialog scenario between a teacher and a 8 years old child student. \
        # The teacher is teaching mathemathics to the child student. \
        # As the student is a child, the teacher needs to stay gentle all the time. Please provide the next valid response for the followig conversation.\
        # You play the role of a teacher. Here is the beginning of the conversation :\
        # Child : Hello !\
        # Teacher : Hi! How are your today ? <</SYS>> \
        # [INST] Child : I am fine, and I can't wait to learn mathematics ! [/INST]"
        # prompts.append(my_prompt)

        # # 1 INST + BOS
        # # WORKS sometimes
        # my_prompt = "<s> [INST] This is a spoken dialog scenario between a teacher and a 8 years old child student. \
        # The teacher is teaching mathemathics to the child student. \
        # As the student is a child, the teacher needs to stay gentle all the time. Please provide the next valid response for the followig conversation. \
        # You play the role of a teacher. Here is the beginning of the conversation : \
        # Child : Hello ! \
        # Teacher : Hi! How are your today ? \
        # Child :  I am fine, and I can't wait to learn mathematics ! [\INST]"
        # prompts.append(my_prompt)

        # # multiple INST + SYS
        # # Do not Work : 
        # # - generate mutiple turn for teacher
        # # - doesn't understand that it is spoken dialogue (generates silent content)
        # my_prompt = "<<SYS>> This is a spoken dialog scenario between a teacher and a 8 years old child student. \
        # The teacher is teaching mathemathics to the child student.\
        # As the student is a child, the teacher needs to stay gentle all the time. Please provide the next valid response for the followig conversation.\
        # You play the role of a teacher. Here is the beginning of the conversation : <</SYS>> \
        # [INST] Child : Hello ! [/INST] \
        # Teacher : Hi! How are your today ?\
        # [INST] Child : I am fine, and I can't wait to learn mathematics ! [/INST]"
        # prompts.append(my_prompt)

        # # multiple INST + BOS + EOS
        # # Do not Work :
        # # - generate mutiple turn for both child and teacher
        # # - doesn't understand that it is spoken dialogue (generates silent content)
        # my_prompt = "<s> [INST] This is a spoken dialog scenario between a teacher and a 8 years old child student. \
        # The teacher is teaching mathemathics to the child student. \
        # As the student is a child, the teacher needs to stay gentle all the time. Please provide the next valid response for the followig conversation. \
        # You play the role of a teacher. Here is the beginning of the conversation : \
        # Child : Hello ! [/INST] \
        # Teacher : Hi! How are your today ?</s> \
        # [INST] Child :  I am fine, and I can't wait to learn mathematics ! [\INST]"
        # prompts.append(my_prompt)

        # # multiple INST + SYS + BOS + EOS
        # # Do not Work :
        # # - generate mutiple turn for both child and teacher
        # my_prompt = "<<SYS>> This is a spoken dialog scenario between a teacher and a 8 years old child student. \
        # The teacher is teaching mathemathics to the child student. \
        # As the student is a child, the teacher needs to stay gentle all the time. Please provide the next valid response for the followig conversation. \
        # You play the role of a teacher. Here is the beginning of the conversation : <</SYS>>\
        # <s> [INST] Child : Hello ! [/INST] \
        # Teacher : Hi! How are your today ?</s> \
        # [INST] Child :  I am fine, and I can't wait to learn mathematics ! [\INST]"
        # prompts.append(my_prompt)
        
        # # Adding "Teacher :" at the end of the prompt
        # my_prompt = "<<SYS>>This is a spoken dialog scenario between a teacher and a 8 years old child student. \
        # The teacher is teaching mathemathics to the child student.\
        # As the student is a child, the teacher needs to stay gentle all the time. Please provide the next valid response for the followig conversation.\
        # You play the role of a teacher. Here is the beginning of the conversation :<</SYS>>\
        # [INST]Child : Hello ! [/INST]\
        # Teacher : Hi! How are your today ?\
        # [INST]Child : I am fine, and I can't wait to learn mathematics ![/INST]Teacher : "
        # prompts.append(my_prompt)


        # Long prompt
        # template :
        # "<s> [INST] Hi [/INST] Hi</s> [INST] Hi [/INST] Hi</s> Hi [/INST]"

        # 1 INST
        # WORKS
        # But generates weird [] / \n  characters at the end of the utterance
        # my_prompt = "[INST] This is a spoken dialog scenario between a teacher and a 8 years old child student. \
        # The teacher is teaching mathemathics to the child student. \
        # As the student is a child, the teacher needs to stay gentle all the time. Please provide the next valid response for the followig conversation.\
        # You play the role of a teacher. Here is the beginning of the conversation : \
        # Child : Hello ! \
        # Teacher : Hi! How are your today ? \
        # Child : I am fine, and I can't wait to learn mathematics ! \
        # Teacher : Okay, we will start with the simpler operation in mathematics, addition. What do you know about addition ? \
        # Child : Humm, I know that one plus one equals two ! \
        # Teacher : That's right ! And this is a good start. So, addition is the operation of adding two numbers together to get a bigger number that is their sum. In our case, we added one to one to get the bigger number two, that is their sum. Is it clear ? \
        # Child : I... I am not sure...  \
        # Teacher : Okay I will take another example, when we add one number, let's say two, to another number, let's say one, we get as a result three, a number bigger than both our previous numbers. \
        # Child : okay, I think I get it now ! [/INST]"
        # long_prompts.append(my_prompt)

        # # 1 INST + BOS
        # # WORKS
        # my_prompt = "<s> [INST] This is a spoken dialog scenario between a teacher and a 8 years old child student. \
        # The teacher is teaching mathemathics to the child student. \
        # As the student is a child, the teacher needs to stay gentle all the time. Please provide the next valid response for the followig conversation. \
        # You play the role of a teacher. Here is the beginning of the conversation : \
        # Child : Hello ! \
        # Teacher : Hi! How are your today ? \
        # Child : I am fine, and I can't wait to learn mathematics ! \
        # Teacher : Okay, we will start with the simpler operation in mathematics, addition. What do you know about addition ? \
        # Child : Humm, I know that one plus one equals two ! \
        # Teacher : That's right ! And this is a good start. So, addition is the operation of adding two numbers together to get a bigger number that is their sum. In our case, we added one to one to get the bigger number two, that is their sum. Is it clear ? \
        # Child : I... I am not sure...  \
        # Teacher : Okay I will take another example, when we add one number, let's say two, to another number, let's say one, we get as a result three, a number bigger than both our previous numbers. \
        # Child : okay, I think I get it now ! [/INST]"
        # long_prompts.append(my_prompt)


        # # Multiple INST + BOS + EOS
        # # Do not Work :
        # # - generate mutiple turn for both child and teacher
        # my_prompt = "<s>[INST]This is a spoken dialog scenario between a teacher and a 8 years old child student. \
        # The teacher is teaching mathemathics to the child student. \
        # As the student is a child, the teacher needs to stay gentle all the time. Please provide the next valid response for the followig conversation. \
        # You play the role of a teacher. Here is the beginning of the conversation : \
        # Child : Hello ! [/INST] \
        # Teacher : Hi! How are your today ?</s> \
        # [INST] Child : I am fine, and I can't wait to learn mathematics ! [/INST] \
        # Teacher : Okay, we will start with the simpler operation in mathematics, addition. What do you know about addition ?</s>\
        # [INST] Child : Humm, I know that one plus one equals two ! [/INST] \
        # Teacher : That's right ! And this is a good start. So, addition is the operation of adding two numbers together to get a bigger number that is their sum. In our case, we added one to one to get the bigger number two, that is their sum. Is it clear ?</s>\
        # [INST] Child : I... I am not sure... [/INST] \
        # Teacher : Okay I will take another example, when we add one number, let's say two, to another number, let's say one, we get as a result three, a number bigger than both our previous numbers.</s>\
        # [INST] Child : okay, I think I get it now ! [/INST]"
        # long_prompts.append(my_prompt)

        # # 1 INST + SYS
        # # Do not Work :
        # # - generate mutiple turn for both child and teacher
        # my_prompt = "[INST] <<SYS>>\
        # This is a spoken dialog scenario between a teacher and a 8 years old child student. \
        # The teacher is teaching mathemathics to the child student. \
        # As the student is a child, the teacher needs to stay gentle all the time. Please provide the next valid response for the followig conversation.\
        # You play the role of a teacher. Here is the beginning of the conversation : \
        # <</SYS>> \
        # \n\n\
        # Child : Hello ! \n\n\
        # Teacher : Hi! How are your today ? \n\n\
        # Child : I am fine, and I can't wait to learn mathematics ! \n\n\
        # Teacher : Okay, we will start with the simpler operation in mathematics, addition. What do you know about addition ? \n\n\
        # Child : Humm, I know that one plus one equals two ! \n\n\
        # Teacher : That's right ! And this is a good start. So, addition is the operation of adding two numbers together to get a bigger number that is their sum. In our case, we added one to one to get the bigger number two, that is their sum. Is it clear ? \n\n\
        # Child : I... I am not sure...  \n\n\
        # Teacher : Okay I will take another example, when we add one number, let's say two, to another number, let's say one, we get as a result three, a number bigger than both our previous numbers. \n\n\
        # Child : okay, I think I get it now !\
        # [/INST]"
        # long_prompts.append(my_prompt)

        # # 1 INST + SYS
        # my_prompt = "[INST] <<SYS>>\
        # This is a spoken dialog scenario between a teacher and a 8 years old child student. \
        # The teacher is teaching mathemathics to the child student. \
        # As the student is a child, the teacher needs to stay gentle all the time. Please provide the next valid response for the followig conversation.\
        # You play the role of a teacher. Here is the beginning of the conversation :\
        # <</SYS>> \
        # \n\n\
        # Child : Hello ! \n\n\
        # Teacher : Hi! How are your today ? \n\n\
        # Child : I am fine, and I can't wait to learn mathematics ! \n\n\
        # Teacher : Okay, we will start with the simpler operation in mathematics, addition. What do you know about addition ? \n\n\
        # Child : Humm, I know that one plus one equals two ! \n\n\
        # Teacher : That's right ! And this is a good start. So, addition is the operation of adding two numbers together to get a bigger number that is their sum. In our case, we added one to one to get the bigger number two, that is their sum. Is it clear ? \n\n\
        # Child : I... I am not sure...  \n\n\
        # Teacher : Okay I will take another example, when we add one number, let's say two, to another number, let's say one, we get as a result three, a number bigger than both our previous numbers. \n\n\
        # Child : okay, I think I get it now ! \n\n\
        # Teacher: Great job understanding addition so far! Now let's try a few sums together. I will say the numbers, and you tell me what number comes after when we add them together. Ready? \n\n\
        # Child: Yes! \n\n\
        # Teacher: Alright, let's start with three and one. \n\n\
        # Child: Four! \n\n\
        # Teacher: That's correct! Keep up the good work. Next, let's try five and two. \n\n\
        # Child: Seven!\
        # [/INST]"
        # long_prompts.append(my_prompt)

        # # 1 INST + SYS
        # my_prompt = "[INST] <<SYS>>\
        # This is a spoken dialog scenario between a teacher and a 8 years old child student. \
        # The teacher is teaching mathemathics to the child student. \
        # As the student is a child, the teacher needs to stay gentle all the time. Please provide the next valid response for the followig conversation.\
        # You play the role of a teacher. Here is the beginning of the conversation : \
        # <</SYS>> \
        # \n\n\
        # Child : Hello ! \n\n\
        # Teacher : Hi! How are your today ? \n\n\
        # Child : I am fine, and I can't wait to learn mathematics ! \n\n\
        # Teacher : Okay, we will start with the simpler operation in mathematics, addition. What do you know about addition ? \n\n\
        # Child : Humm, I know that one plus one equals two ! \n\n\
        # Teacher : That's right ! And this is a good start. So, addition is the operation of adding two numbers together to get a bigger number that is their sum. In our case, we added one to one to get the bigger number two, that is their sum. Is it clear ? \n\n\
        # Child : I... I am not sure...  \n\n\
        # Teacher : Okay I will take another example, when we add one number, let's say two, to another number, let's say one, we get as a result three, a number bigger than both our previous numbers. \n\n\
        # Child : okay, I think I get it now ! \n\n\
        # Teacher : Great attitude! I'm glad you're excited about math today. Let's start with something simple: can you count to ten with me? \n\n\
        # Child: Sure thing! One, two, three, four, five, six, seven, eight, nine, ten! \n\n\
        # Teacher: Excellent job! Now, let's try something a little more challenging. Can you identify these numbers: 4, 7, and 9? \n\n\
        # Child: Yes! Four, seven, and nine! \n\n\
        # Teacher: That's correct. Now let's practice adding some numbers together. What do you get when you add 2 and 3? \n\n\
        # Child: Five! \n\n\
        # Teacher: That's right! Good job. Let's try another one: what do you get when you add 5 and 7? \n\n\
        # Child: Twelve! \n\n\
        # Teacher: Not quite. Try again. \n\n\
        # Child: Uh... eleven? \n\n\
        # Teacher: No, we get twelve. Keep trying! Remember that five and seven make twelve when you put them together. \n\n\
        # Child: Oh, I see! I got it now! \n\n\
        # Teacher: Great job! Keep up the good work! Now let's practice subtraction: what's five take away two? \n\n\
        # Child: Three! \
        # [/INST]"
        # long_prompts.append(my_prompt)

        # This is a spoken dialog scenario between a teacher and a 8 years old child student. \
        # The teacher is teaching mathemathics to the child student. \
        # As the student is a child, the teacher needs to stay gentle all the time.\
        # You play the role of a teacher. Here is the beginning of the conversation :

        my_prompt = "<<SYS>>\
        You are a mathematics Teacher, you must teach addition to a 8 year old child student.\
        You interact with the student through a spoken dialogue.\
        As your student is a child, you must stay gentle and supportive all the time.\
        <</SYS>>\
        \n\n\
        The following is the conversation between you and your child student:\
        Child : Hello ! \n\n\
        Teacher : Hi! How are your today ? \n\n\
        Child : I am fine, and I can't wait to learn mathematics ! \n\n\
        Teacher : Okay, we will start with the simpler operation in mathematics, addition. What do you know about addition ? \n\n\
        Child : Humm, I know that one plus one equals two ! \n\n\
        Teacher : That's right ! And this is a good start. So, addition is the operation of adding two numbers together to get a bigger number that is their sum. In our case, we added one to one to get the bigger number two, that is their sum. Is it clear ? \n\n\
        Child : I... I am not sure...  \n\n\
        Teacher : Okay I will take another example, when we add one number, let's say two, to another number, let's say one, we get as a result three, a number bigger than both our previous numbers. \n\n\
        Child : okay, I think I get it now !\n\n\
        [INST]\
        Please only provide the very next Teacher response.\
        [/INST]"
        long_prompts.append(my_prompt)

        # 1 INST + SYS
        my_prompt = "<<SYS>>\
        You are a mathematics Teacher, you must teach addition to a 8 year old child student.\
        You interact with the student through a spoken dialogue.\
        As your student is a child, you must stay gentle and supportive all the time.\
        <</SYS>>\
        \n\n\
        The following is the conversation between you and your child student:\
        Child : Hello ! \n\n\
        Teacher : Hi! How are your today ? \n\n\
        Child : I am fine, and I can't wait to learn mathematics ! \n\n\
        Teacher : Okay, we will start with the simpler operation in mathematics, addition. What do you know about addition ? \n\n\
        Child : Humm, I know that one plus one equals two ! \n\n\
        Teacher : That's right ! And this is a good start. So, addition is the operation of adding two numbers together to get a bigger number that is their sum. In our case, we added one to one to get the bigger number two, that is their sum. Is it clear ? \n\n\
        Child : I... I am not sure...  \n\n\
        Teacher : Okay I will take another example, when we add one number, let's say two, to another number, let's say one, we get as a result three, a number bigger than both our previous numbers. \n\n\
        Child : okay, I think I get it now ! \n\n\
        Teacher: Great job understanding addition so far! Now let's try a few sums together. I will say the numbers, and you tell me what number comes after when we add them together. Ready? \n\n\
        Child: Yes! \n\n\
        Teacher: Alright, let's start with three and one. \n\n\
        Child: Four! \n\n\
        Teacher: That's correct! Keep up the good work. Next, let's try five and two. \n\n\
        Child: Seven! \n\n\
        [INST]\
        Please only provide the very next Teacher response.\
        [/INST]"
        long_prompts.append(my_prompt)

        # 1 INST + SYS
        my_prompt = "<<SYS>>\
        You are a mathematics Teacher, you must teach addition to a 8 year old child student.\
        You interact with the student through a spoken dialogue.\
        As your student is a child, you must stay gentle and supportive all the time.\
        <</SYS>>\
        \n\n\
        The following is the conversation between you and your child student:\
        Child : Hello ! \n\n\
        Teacher : Hi! How are your today ? \n\n\
        Child : I am fine, and I can't wait to learn mathematics ! \n\n\
        Teacher : Okay, we will start with the simpler operation in mathematics, addition. What do you know about addition ? \n\n\
        Child : Humm, I know that one plus one equals two ! \n\n\
        Teacher : That's right ! And this is a good start. So, addition is the operation of adding two numbers together to get a bigger number that is their sum. In our case, we added one to one to get the bigger number two, that is their sum. Is it clear ? \n\n\
        Child : I... I am not sure...  \n\n\
        Teacher : Okay I will take another example, when we add one number, let's say two, to another number, let's say one, we get as a result three, a number bigger than both our previous numbers. \n\n\
        Child : okay, I think I get it now ! \n\n\
        Teacher : Great attitude! I'm glad you're excited about math today. Let's start with something simple: can you count to ten with me? \n\n\
        Child: Sure thing! One, two, three, four, five, six, seven, eight, nine, ten! \n\n\
        Teacher: Excellent job! Now, let's try something a little more challenging. Can you identify these numbers: 4, 7, and 9? \n\n\
        Child: Yes! Four, seven, and nine! \n\n\
        Teacher: That's correct. Now let's practice adding some numbers together. What do you get when you add 2 and 3? \n\n\
        Child: Five! \n\n\
        Teacher: That's right! Good job. Let's try another one: what do you get when you add 5 and 7? \n\n\
        Child: Twelve! \n\n\
        Teacher: Not quite. Try again. \n\n\
        Child: Uh... eleven? \n\n\
        Teacher: No, we get twelve. Keep trying! Remember that five and seven make twelve when you put them together. \n\n\
        Child: Oh, I see! I got it now! \n\n\
        Teacher: Great job! Keep up the good work! Now let's practice subtraction: what's five take away two? \n\n\
        Child: Three! \n\n\
        [INST]\
        Please only provide the very next Teacher response.\
        [/INST]"
        long_prompts.append(my_prompt)



        my_prompt = "<<SYS>>\
        You are a mathematics Teacher, you must teach addition to a 8 year old child student.\
        You interact with the student through a spoken dialogue.\
        As your student is a child, you must stay gentle and supportive all the time.\
        <</SYS>>\
        \n\n\
        The following is the conversation between you and your child student:\
        Child : Hello ! \n\n\
        Teacher : Hi! How are your today ? \n\n\
        Child : I am fine, and I can't wait to learn mathematics ! \n\n\
        Teacher : Okay, we will start with the simpler operation in mathematics, addition. What do you know about addition ? \n\n\
        Child : Humm, I know that one plus one equals two ! \n\n\
        Teacher : That's right ! And this is a good start. So, addition is the operation of adding two numbers together to get a bigger number that is their sum. In our case, we added one to one to get the bigger number two, that is their sum. Is it clear ? \n\n\
        Child : I... I am not sure...  \n\n\
        Teacher : Okay I will take another example, when we add one number, let's say two, to another number, let's say one, we get as a result three, a number bigger than both our previous numbers. \n\n\
        Child : okay, I think I get it now !\n\n\
        [INST]\
        Please complete the following Teacher response :\
        Teacher: \
        [/INST]"
        long_prompts.append(my_prompt)

        # 1 INST + SYS
        my_prompt = "<<SYS>>\
        You are a mathematics Teacher, you must teach addition to a 8 year old child student.\
        You interact with the student through a spoken dialogue.\
        As your student is a child, you must stay gentle and supportive all the time.\
        <</SYS>>\
        \n\n\
        The following is the conversation between you and your child student:\
        Child : Hello ! \n\n\
        Teacher : Hi! How are your today ? \n\n\
        Child : I am fine, and I can't wait to learn mathematics ! \n\n\
        Teacher : Okay, we will start with the simpler operation in mathematics, addition. What do you know about addition ? \n\n\
        Child : Humm, I know that one plus one equals two ! \n\n\
        Teacher : That's right ! And this is a good start. So, addition is the operation of adding two numbers together to get a bigger number that is their sum. In our case, we added one to one to get the bigger number two, that is their sum. Is it clear ? \n\n\
        Child : I... I am not sure...  \n\n\
        Teacher : Okay I will take another example, when we add one number, let's say two, to another number, let's say one, we get as a result three, a number bigger than both our previous numbers. \n\n\
        Child : okay, I think I get it now ! \n\n\
        Teacher: Great job understanding addition so far! Now let's try a few sums together. I will say the numbers, and you tell me what number comes after when we add them together. Ready? \n\n\
        Child: Yes! \n\n\
        Teacher: Alright, let's start with three and one. \n\n\
        Child: Four! \n\n\
        Teacher: That's correct! Keep up the good work. Next, let's try five and two. \n\n\
        Child: Seven! \n\n\
        [INST]\
        Please complete the following Teacher response :\
        Teacher: \
        [/INST]"
        long_prompts.append(my_prompt)

        # 1 INST + SYS
        my_prompt = "<<SYS>>\
        You are a mathematics Teacher, you must teach addition to a 8 year old child student.\
        You interact with the student through a spoken dialogue.\
        As your student is a child, you must stay gentle and supportive all the time.\
        <</SYS>>\
        \n\n\
        The following is the conversation between you and your child student:\
        Child : Hello ! \n\n\
        Teacher : Hi! How are your today ? \n\n\
        Child : I am fine, and I can't wait to learn mathematics ! \n\n\
        Teacher : Okay, we will start with the simpler operation in mathematics, addition. What do you know about addition ? \n\n\
        Child : Humm, I know that one plus one equals two ! \n\n\
        Teacher : That's right ! And this is a good start. So, addition is the operation of adding two numbers together to get a bigger number that is their sum. In our case, we added one to one to get the bigger number two, that is their sum. Is it clear ? \n\n\
        Child : I... I am not sure...  \n\n\
        Teacher : Okay I will take another example, when we add one number, let's say two, to another number, let's say one, we get as a result three, a number bigger than both our previous numbers. \n\n\
        Child : okay, I think I get it now ! \n\n\
        Teacher : Great attitude! I'm glad you're excited about math today. Let's start with something simple: can you count to ten with me? \n\n\
        Child: Sure thing! One, two, three, four, five, six, seven, eight, nine, ten! \n\n\
        Teacher: Excellent job! Now, let's try something a little more challenging. Can you identify these numbers: 4, 7, and 9? \n\n\
        Child: Yes! Four, seven, and nine! \n\n\
        Teacher: That's correct. Now let's practice adding some numbers together. What do you get when you add 2 and 3? \n\n\
        Child: Five! \n\n\
        Teacher: That's right! Good job. Let's try another one: what do you get when you add 5 and 7? \n\n\
        Child: Twelve! \n\n\
        Teacher: Not quite. Try again. \n\n\
        Child: Uh... eleven? \n\n\
        Teacher: No, we get twelve. Keep trying! Remember that five and seven make twelve when you put them together. \n\n\
        Child: Oh, I see! I got it now! \n\n\
        Teacher: Great job! Keep up the good work! Now let's practice subtraction: what's five take away two? \n\n\
        Child: Three! \n\n\
        [INST]\
        Please complete the following Teacher response :\
        Teacher: \
        [/INST]"
        long_prompts.append(my_prompt)


        # # 1 INST + SYS
        # # Do not Work :
        # # - generate mutiple turn for both child and teacher
        # my_prompt = "<<SYS>>  This is a spoken dialog scenario between a teacher and a 8 years old child student. \
        # The teacher is teaching mathemathics to the child student. \
        # As the student is a child, the teacher needs to stay gentle all the time. Please provide the next valid response for the followig conversation.\
        # You play the role of a teacher. Here is the beginning of the conversation : <</SYS>> \
        # [INST] Child : Hello ! \
        # Teacher : Hi! How are your today ? \
        # Child : I am fine, and I can't wait to learn mathematics ! \
        # Teacher : Okay, we will start with the simpler operation in mathematics, addition. What do you know about addition ? \
        # Child : Humm, I know that one plus one equals two ! \
        # Teacher : That's right ! And this is a good start. So, addition is the operation of adding two numbers together to get a bigger number that is their sum. In our case, we added one to one to get the bigger number two, that is their sum. Is it clear ? \
        # Child : I... I am not sure...  \
        # Teacher : Okay I will take another example, when we add one number, let's say two, to another number, let's say one, we get as a result three, a number bigger than both our previous numbers. \
        # Child : okay, I think I get it now ! [/INST]"
        # long_prompts.append(my_prompt)

        # # 1 INST + SYS
        # my_prompt = "<<SYS>>  This is a spoken dialog scenario between a teacher and a 8 years old child student. \
        # The teacher is teaching mathemathics to the child student. \
        # As the student is a child, the teacher needs to stay gentle all the time. Please provide the next valid response for the followig conversation.\
        # You play the role of a teacher. Here is the beginning of the conversation : <</SYS>> \
        # [INST] Child : Hello ! \
        # Teacher : Hi! How are your today ? \
        # Child : I am fine, and I can't wait to learn mathematics ! \
        # Teacher : Okay, we will start with the simpler operation in mathematics, addition. What do you know about addition ? \
        # Child : Humm, I know that one plus one equals two ! \
        # Teacher : That's right ! And this is a good start. So, addition is the operation of adding two numbers together to get a bigger number that is their sum. In our case, we added one to one to get the bigger number two, that is their sum. Is it clear ? \
        # Child : I... I am not sure...  \
        # Teacher : Okay I will take another example, when we add one number, let's say two, to another number, let's say one, we get as a result three, a number bigger than both our previous numbers. \
        # Child : okay, I think I get it now ! \
        # Teacher: Great job understanding addition so far! Now let's try a few sums together. I will say the numbers, and you tell me what number comes after when we add them together. Ready? \
        # Child: Yes! \
        # Teacher: Alright, let's start with three and one. \
        # Child: Four! \
        # Teacher: That's correct! Keep up the good work. Next, let's try five and two. \
        # Child: Seven! [/INST]"
        # long_prompts.append(my_prompt)

        # # 1 INST + SYS
        # my_prompt = "<<SYS>>  This is a spoken dialog scenario between a teacher and a 8 years old child student. \
        # The teacher is teaching mathemathics to the child student. \
        # As the student is a child, the teacher needs to stay gentle all the time. Please provide the next valid response for the followig conversation.\
        # You play the role of a teacher. Here is the beginning of the conversation : <</SYS>> \
        # [INST] Child : Hello ! \
        # Teacher : Hi! How are your today ? \
        # Child : I am fine, and I can't wait to learn mathematics ! \
        # Teacher : Okay, we will start with the simpler operation in mathematics, addition. What do you know about addition ? \
        # Child : Humm, I know that one plus one equals two ! \
        # Teacher : That's right ! And this is a good start. So, addition is the operation of adding two numbers together to get a bigger number that is their sum. In our case, we added one to one to get the bigger number two, that is their sum. Is it clear ? \
        # Child : I... I am not sure...  \
        # Teacher : Okay I will take another example, when we add one number, let's say two, to another number, let's say one, we get as a result three, a number bigger than both our previous numbers. \
        # Child : okay, I think I get it now ! \
        # Teacher : Great attitude! I'm glad you're excited about math today. Let's start with something simple: can you count to ten with me? \
        # Child: Sure thing! One, two, three, four, five, six, seven, eight, nine, ten! \
        # Teacher: Excellent job! Now, let's try something a little more challenging. Can you identify these numbers: 4, 7, and 9? \
        # Child: Yes! Four, seven, and nine! \
        # Teacher: That's correct. Now let's practice adding some numbers together. What do you get when you add 2 and 3? \
        # Child: Five! \
        # Teacher: That's right! Good job. Let's try another one: what do you get when you add 5 and 7? \
        # Child: Twelve! \
        # Teacher: Not quite. Try again. \
        # Child: Uh... eleven? \
        # Teacher: No, we get twelve. Keep trying! Remember that five and seven make twelve when you put them together. \
        # Child: Oh, I see! I got it now! \
        # Teacher: Great job! Keep up the good work! Now let's practice subtraction: what's five take away two? \
        # Child: Three! [/INST]"
        # long_prompts.append(my_prompt)

        # # Doesn't work : generates more than 1 sentence
        # my_prompt = "<s>[INST]This is a spoken dialog scenario between a teacher and a 8 years old child student. \
        # The teacher is teaching mathemathics to the child student. \
        # As the student is a child, the teacher needs to stay gentle all the time. Please provide the next valid response for the followig conversation. \
        # You play the role of a teacher. Here is the beginning of the conversation : \
        # Child : Hello ! [/INST] \
        # Teacher : Hi! How are your today ?</s> \
        # [INST] Child : I am fine, and I can't wait to learn mathematics ! [/INST] \
        # Teacher : Okay, we will start with the simpler operation in mathematics, addition. What do you know about addition ?</s>\
        # [INST] Child : Humm, I know that one plus one equals two ! [/INST] \
        # Teacher : That's right ! And this is a good start. So, addition is the operation of adding two numbers together to get a bigger number that is their sum. In our case, we added one to one to get the bigger number two, that is their sum. Is it clear ?</s>\
        # [INST] Child : I... I am not sure... [/INST] \
        # Teacher : Okay I will take another example, when we add one number, let's say two, to another number, let's say one, we get as a result three, a number bigger than both our previous numbers.</s>\
        # [INST] Child : okay, I think I get it now ! [/INST] \
        # Teacher : "
        # long_prompts.append(my_prompt)

        # my_prompt = "<s>[INST]This is a spoken dialog scenario between a teacher and a 8 years old child student. \
        # The teacher is teaching mathemathics to the child student. \
        # As the student is a child, the teacher needs to stay gentle all the time. Please provide the next valid response for the followig conversation. \
        # You play the role of a teacher. Here is the beginning of the conversation : \
        # Child : Hello !\
        # Teacher : Hi! How are your today ?\
        # Child : I am fine, and I can't wait to learn mathematics !\
        # Teacher : Okay, we will start with the simpler operation in mathematics, addition. What do you know about addition ?\
        # Child : Humm, I know that one plus one equals two !\
        # Teacher : That's right ! And this is a good start. So, addition is the operation of adding two numbers together to get a bigger number that is their sum. In our case, we added one to one to get the bigger number two, that is their sum. Is it clear ?\
        # Child : I... I am not sure...\
        # Teacher : Okay I will take another example, when we add one number, let's say two, to another number, let's say one, we get as a result three, a number bigger than both our previous numbers.\
        # Child : okay, I think I get it now ! [/INST]"
        # long_prompts.append(my_prompt)

        # my_prompt = "[INST]This is a spoken dialog scenario between a teacher and a 8 years old child student. \
        # The teacher is teaching mathemathics to the child student. \
        # As the student is a child, the teacher needs to stay gentle all the time. Please provide the next valid response for the followig conversation.\
        # You play the role of a teacher. Here is the beginning of the conversation : \
        # Child : Hello !\
        # Teacher : Hi! How are your today ? \
        # Child : I am fine, and I can't wait to learn mathematics !\
        # Teacher : Okay, we will start with the simpler operation in mathematics, addition. What do you know about addition ?\
        # Child : Humm, I know that one plus one equals two !\
        # Teacher : That's right ! And this is a good start. So, addition is the operation of adding two numbers together to get a bigger number that is their sum. In our case, we added one to one to get the bigger number two, that is their sum. Is it clear ?\
        # Child : I... I am not sure...\
        # Teacher : Okay I will take another example, when we add one number, let's say two, to another number, let's say one, we get as a result three, a number bigger than both our previous numbers.\
        # Child : okay, I think I get it now ![/INST]"
        # long_prompts.append(my_prompt)

        # my_prompt = "<<SYS>>This is a spoken dialog scenario between a teacher and a 8 years old child student. \
        # The teacher is teaching mathemathics to the child student. \
        # As the student is a child, the teacher needs to stay gentle all the time. Please provide the next valid response for the followig conversation.\
        # You play the role of a teacher. Here is the beginning of the conversation :<</SYS>>\
        # [INST]Child : Hello ![/INST]\
        # Teacher : Hi! How are your today ? \
        # [INST]Child : I am fine, and I can't wait to learn mathematics ![/INST]\
        # Teacher : Okay, we will start with the simpler operation in mathematics, addition. What do you know about addition ?\
        # [INST]Child : Humm, I know that one plus one equals two ![/INST]\
        # Teacher : That's right ! And this is a good start. So, addition is the operation of adding two numbers together to get a bigger number that is their sum. In our case, we added one to one to get the bigger number two, that is their sum. Is it clear ?\
        # [INST]Child : I... I am not sure...[/INST]\
        # Teacher : Okay I will take another example, when we add one number, let's say two, to another number, let's say one, we get as a result three, a number bigger than both our previous numbers.\
        # [INST]Child : okay, I think I get it now ![/INST]"
        # long_prompts.append(my_prompt)

        # my_prompt = "<<SYS>>This is a spoken dialog scenario between a teacher and a 8 years old child student. \
        # The teacher is teaching mathemathics to the child student. \
        # As the student is a child, the teacher needs to stay gentle all the time.\
        # You play the role of a teacher. Here is the beginning of the conversation :\
        # Child : Hello !\
        # Teacher : Hi! How are your today ? \
        # Child : I am fine, and I can't wait to learn mathematics !\
        # Teacher : Okay, we will start with the simpler operation in mathematics, addition. What do you know about addition ?\
        # Child : Humm, I know that one plus one equals two !\
        # Teacher : That's right ! And this is a good start. So, addition is the operation of adding two numbers together to get a bigger number that is their sum. In our case, we added one to one to get the bigger number two, that is their sum. Is it clear ?\
        # Child : I... I am not sure...\
        # Teacher : Okay I will take another example, when we add one number, let's say two, to another number, let's say one, we get as a result three, a number bigger than both our previous numbers.\
        # Child : okay, I think I get it now !<</SYS>>\
        # [INST]Please provide the next valid response for the previous conversation.[/INST]"
        # long_prompts.append(my_prompt)
    
    return prompts, long_prompts


def set_prompt():

    my_prompt, chat_history, chat_format = None, None, None

    # The prompt will be generated from a list of messages (chat_history)
    # we set the chat_format argument to initialize the model
    if PROMPT_TEMPLATE == "chat":

        chat_format = CHAT_FORMAT

        chat_history = [
            {"role": "system", "content": "This is a spoken dialog scenario between a teacher and a 8 years old child student. \
                The teacher is teaching mathemathics to the child student. \
                As the student is a child, the teacher needs to stay gentle all the time. Please provide the next valid response for the followig conversation.\
                You play the role of a teacher. Here is the beginning of the conversation :"},
            {"role": "user", "content": "Hello !"}, 
            {"role": "assistant", "content": "Hi! How are your today ?"},
            {"role": "user", "content": "I am fine, and I can't wait to learn mathematics !"},
        ]

    elif PROMPT_TEMPLATE == "instruct":

        # works
        my_prompt = "[INST]This is a spoken dialog scenario between a teacher and a 8 years old child student. \
        The teacher is teaching mathemathics to the child student. \
        As the student is a child, the teacher needs to stay gentle all the time. Please provide the next valid response for the followig conversation.\
        You play the role of a teacher. Here is the beginning of the conversation : \
        Child : Hello !\
        Teacher : Hi! How are your today ? \
        Child : I am fine, and I can't wait to learn mathematics ![/INST]"

        # my_prompt = "<<SYS>>[INST]This is a spoken dialog scenario between a teacher and a 8 years old child student. \
        # The teacher is teaching mathemathics to the child student. \
        # As the student is a child, the teacher needs to stay gentle all the time. Please provide the next valid response for the followig conversation.\
        # You play the role of a teacher. Here is the beginning of the conversation : \
        # Child : Hello !\
        # Teacher : Hi! How are your today ? \
        # Child : I am fine, and I can't wait to learn mathematics ![/INST]<</SYS>>"

        # my_prompt = "<<SYS>>This is a spoken dialog scenario between a teacher and a 8 years old child student. \
        # The teacher is teaching mathemathics to the child student.\
        # As the student is a child, the teacher needs to stay gentle all the time. Please provide the next valid response for the followig conversation.\
        # You play the role of a teacher. Here is the beginning of the conversation :<</SYS>>\
        # [INST]Child : Hello ! [/INST]\
        # Teacher : Hi! How are your today ?\
        # [INST]Child : I am fine, and I can't wait to learn mathematics ![/INST]"

        # Doesn't work
        # my_prompt = "<s>[INST]This is a spoken dialog scenario between a teacher and a 8 years old child student. \
        # The teacher is teaching mathemathics to the child student. \
        # As the student is a child, the teacher needs to stay gentle all the time. Please provide the next valid response for the followig conversation. \
        # You play the role of a teacher. Here is the beginning of the conversation : \
        # Child : Hello ! [/INST] \
        # Teacher : Hi! How are your today ?</s> \
        # [INST] Child :  I am fine, and I can't wait to learn mathematics ! [\INST]"
        
        # my_prompt = "<<SYS>>This is a spoken dialog scenario between a teacher and a 8 years old child student. \
        # The teacher is teaching mathemathics to the child student.\
        # As the student is a child, the teacher needs to stay gentle all the time. Please provide the next valid response for the followig conversation.\
        # You play the role of a teacher. Here is the beginning of the conversation :<</SYS>>\
        # [INST]Child : Hello ! [/INST]\
        # Teacher : Hi! How are your today ?\
        # [INST]Child : I am fine, and I can't wait to learn mathematics ![/INST]Teacher : "


        # Long prompt
        # template :
        # "<s> [INST] Hi [/INST] Hi</s> [INST] Hi [/INST] Hi</s> Hi [/INST]"

        # Works !
        # my_prompt = "<s>[INST]This is a spoken dialog scenario between a teacher and a 8 years old child student. \
        # The teacher is teaching mathemathics to the child student. \
        # As the student is a child, the teacher needs to stay gentle all the time. Please provide the next valid response for the followig conversation. \
        # You play the role of a teacher. Here is the beginning of the conversation : \
        # Child : Hello ! [/INST] \
        # Teacher : Hi! How are your today ?</s> \
        # [INST] Child : I am fine, and I can't wait to learn mathematics ! [/INST] \
        # Teacher : Okay, we will start with the simpler operation in mathematics, addition. What do you know about addition ?</s>\
        # [INST] Child : Humm, I know that one plus one equals two ! [/INST] \
        # Teacher : That's right ! And this is a good start. So, addition is the operation of adding two numbers together to get a bigger number that is their sum. In our case, we added one to one to get the bigger number two, that is their sum. Is it clear ?</s>\
        # [INST] Child : I... I am not sure... [/INST] \
        # Teacher : Okay I will take another example, when we add one number, let's say two, to another number, let's say one, we get as a result three, a number bigger than both our previous numbers.</s>\
        # [INST] Child : okay, I think I get it now ! [/INST]"


        # Doesn't work : generates more than 1 sentence
        # my_prompt = "<s>[INST]This is a spoken dialog scenario between a teacher and a 8 years old child student. \
        # The teacher is teaching mathemathics to the child student. \
        # As the student is a child, the teacher needs to stay gentle all the time. Please provide the next valid response for the followig conversation. \
        # You play the role of a teacher. Here is the beginning of the conversation : \
        # Child : Hello ! [/INST] \
        # Teacher : Hi! How are your today ?</s> \
        # [INST] Child : I am fine, and I can't wait to learn mathematics ! [/INST] \
        # Teacher : Okay, we will start with the simpler operation in mathematics, addition. What do you know about addition ?</s>\
        # [INST] Child : Humm, I know that one plus one equals two ! [/INST] \
        # Teacher : That's right ! And this is a good start. So, addition is the operation of adding two numbers together to get a bigger number that is their sum. In our case, we added one to one to get the bigger number two, that is their sum. Is it clear ?</s>\
        # [INST] Child : I... I am not sure... [/INST] \
        # Teacher : Okay I will take another example, when we add one number, let's say two, to another number, let's say one, we get as a result three, a number bigger than both our previous numbers.</s>\
        # [INST] Child : okay, I think I get it now ! [/INST] \
        # Teacher : "

        # my_prompt = "<s>[INST]This is a spoken dialog scenario between a teacher and a 8 years old child student. \
        # The teacher is teaching mathemathics to the child student. \
        # As the student is a child, the teacher needs to stay gentle all the time. Please provide the next valid response for the followig conversation. \
        # You play the role of a teacher. Here is the beginning of the conversation : \
        # Child : Hello !\
        # Teacher : Hi! How are your today ?\
        # Child : I am fine, and I can't wait to learn mathematics !\
        # Teacher : Okay, we will start with the simpler operation in mathematics, addition. What do you know about addition ?\
        # Child : Humm, I know that one plus one equals two !\
        # Teacher : That's right ! And this is a good start. So, addition is the operation of adding two numbers together to get a bigger number that is their sum. In our case, we added one to one to get the bigger number two, that is their sum. Is it clear ?\
        # Child : I... I am not sure...\
        # Teacher : Okay I will take another example, when we add one number, let's say two, to another number, let's say one, we get as a result three, a number bigger than both our previous numbers.\
        # Child : okay, I think I get it now ! [/INST]"

        # my_prompt = "[INST]This is a spoken dialog scenario between a teacher and a 8 years old child student. \
        # The teacher is teaching mathemathics to the child student. \
        # As the student is a child, the teacher needs to stay gentle all the time. Please provide the next valid response for the followig conversation.\
        # You play the role of a teacher. Here is the beginning of the conversation : \
        # Child : Hello !\
        # Teacher : Hi! How are your today ? \
        # Child : I am fine, and I can't wait to learn mathematics !\
        # Teacher : Okay, we will start with the simpler operation in mathematics, addition. What do you know about addition ?\
        # Child : Humm, I know that one plus one equals two !\
        # Teacher : That's right ! And this is a good start. So, addition is the operation of adding two numbers together to get a bigger number that is their sum. In our case, we added one to one to get the bigger number two, that is their sum. Is it clear ?\
        # Child : I... I am not sure...\
        # Teacher : Okay I will take another example, when we add one number, let's say two, to another number, let's say one, we get as a result three, a number bigger than both our previous numbers.\
        # Child : okay, I think I get it now ![/INST]"

        # my_prompt = "<<SYS>>This is a spoken dialog scenario between a teacher and a 8 years old child student. \
        # The teacher is teaching mathemathics to the child student. \
        # As the student is a child, the teacher needs to stay gentle all the time. Please provide the next valid response for the followig conversation.\
        # You play the role of a teacher. Here is the beginning of the conversation :<</SYS>>\
        # [INST]Child : Hello ![/INST]\
        # Teacher : Hi! How are your today ? \
        # [INST]Child : I am fine, and I can't wait to learn mathematics ![/INST]\
        # Teacher : Okay, we will start with the simpler operation in mathematics, addition. What do you know about addition ?\
        # [INST]Child : Humm, I know that one plus one equals two ![/INST]\
        # Teacher : That's right ! And this is a good start. So, addition is the operation of adding two numbers together to get a bigger number that is their sum. In our case, we added one to one to get the bigger number two, that is their sum. Is it clear ?\
        # [INST]Child : I... I am not sure...[/INST]\
        # Teacher : Okay I will take another example, when we add one number, let's say two, to another number, let's say one, we get as a result three, a number bigger than both our previous numbers.\
        # [INST]Child : okay, I think I get it now ![/INST]"

        # my_prompt = "<<SYS>>This is a spoken dialog scenario between a teacher and a 8 years old child student. \
        # The teacher is teaching mathemathics to the child student. \
        # As the student is a child, the teacher needs to stay gentle all the time.\
        # You play the role of a teacher. Here is the beginning of the conversation :\
        # Child : Hello !\
        # Teacher : Hi! How are your today ? \
        # Child : I am fine, and I can't wait to learn mathematics !\
        # Teacher : Okay, we will start with the simpler operation in mathematics, addition. What do you know about addition ?\
        # Child : Humm, I know that one plus one equals two !\
        # Teacher : That's right ! And this is a good start. So, addition is the operation of adding two numbers together to get a bigger number that is their sum. In our case, we added one to one to get the bigger number two, that is their sum. Is it clear ?\
        # Child : I... I am not sure...\
        # Teacher : Okay I will take another example, when we add one number, let's say two, to another number, let's say one, we get as a result three, a number bigger than both our previous numbers.\
        # Child : okay, I think I get it now !<</SYS>>\
        # [INST]Please provide the next valid response for the previous conversation.[/INST]"

    # The prompt has been written following the llama template.
    elif PROMPT_TEMPLATE == "llama":

        my_prompt = "This is a spoken dialog scenario between a teacher and a 8 years old child student. \
        The teacher is teaching mathemathics to the child student. \
        As the student is a child, the teacher needs to stay gentle all the time. Please provide the next valid response for the followig conversation.\
        You play the role of a teacher. Here is the beginning of the conversation : \
        Teacher : Hi! How are your today ? \
        Child : I am fine, and I can't wait to learn mathematics !"

        # my_prompt = "Here is a conversation between two Participants A and B who are in a virtual space that has lots of different rooms that are depicted with images. Each room has a type (such as kitchen, bathroom, bedroom, etc.).  The participants are initially located in different rooms. The goal of the game is for the two participants to locate themselves in the same room. In order to achieve this goal, the participants communicate with one another by text and describe the room they find themselves in. On the basis of those descriptions, they move to different rooms and describe their new room to the other participant. The game ends when the two participants find themselves in the same room. We translated the images that the participants saw into text. That description of the room is provided below as soon as a participant enters a given room. The current room description of User A starts with a token <Image A> and the current room description of User B starts with a token <Image B>. Every utterance from A or B is preceded with a timestamp closed under brackets. Some text is provided by GM, a non-participant in the game who provides essential information regarding the game to both participants.\
        # You are participant B who can see the following image - <Image B> The image showcases a parking area with three parked cars next to a building. The building is red in colour and has a glass door at it’s entrance. There is a green car on the left, a blue sedan in the middle and another blue car on the right. There is an empty parking slot between the two blue cars.Here is the dialog history - \
        # [00:21] B: It seems like I am in a parking lot.\
        # [00:27] A: What do you see?\
        # [00:32] B: Umm three cars parked next to the building. One green car and two blue ones.\
        # [00:36] A: Okay. Go north\
        # [00:40] B: You want me to go north? \
        # [00:45] A: Sorry, I meant go south to come inside. \
        # Please provide the next utterance keeping in mind that it's a spoken conversation. Make sure to ask for clarifications in case there is any ambiguity and also provide additional information in case there is a clarification question from user A - \
        # [00 50] B : "

    # The generation method is not implemented yet.
    else:
        raise NotImplementedError("Template "+PROMPT_TEMPLATE+" not implemented")

    return my_prompt, chat_history, chat_format


def set_model(chat_format):
    # Load the model from a local file (.gguf) or from a pre-trained model on a hugging face repo
    if MODEL_LOADING_METHOD == "from_pretrained":

        my_model = Llama.from_pretrained(
            repo_id=MODEL_REPO, 
            filename=MODEL_NAME,
            device_map=DEVICE_MAP,
            n_ctx=CONTEXT_SIZE, 
            n_gpu_layers=N_GPU_LAYERS,
            chat_format=chat_format,
        )
        
    elif MODEL_LOADING_METHOD == "local_file":

        my_model = Llama(
            model_path=MODEL_PATH, 
            n_ctx=CONTEXT_SIZE, 
            n_gpu_layers=N_GPU_LAYERS,
            chat_format=chat_format,
            device_map=DEVICE_MAP,
        )
    return my_model


def generate(my_model, my_prompt, chat_history, stop_criteria):

    if MODEL_GENERATION_METHOD == "stream":

        ponctuation_list = [".", ",", ";", ":", "!", "?", "..."]
        def sub_process(chunk, timing, is_ponct):
            if is_ponct:
                print("["+str(timing) + "s] " + chunk)

        time_0 = time.time()
        last_sentence, last_sentence_nb_token = generate_next_sentence(my_model, my_prompt, sub_process, ponctuation_list)
        time_1 = time.time()

        print(last_sentence)
        print("\n\n["+str(round(time_1 - time_0, 3)) + "s] " + last_sentence)
        

    # The generation method used is the model.generate(tokens) function.
    if MODEL_GENERATION_METHOD == "generate_function":

        my_bytes_prompt = bytes(my_prompt, "utf-8")

        time_0 = time.time()
        last_sentence = generate_text_with_generate_function(my_bytes_prompt, my_model, stop_criteria)
        time_1 = time.time()
        
        # print(model_output)
        # full_prompt = model_output["choices"][0]["text"].strip()
        # last_sentence = full_prompt.split("[/INST]")[-1]
        print("\n\n["+str(round(time_1 - time_0, 3)) + "s] " + last_sentence.decode("utf-8"))

    # The generation method used is the model(prompt) call.
    elif MODEL_GENERATION_METHOD == "model_call":

        # The prompt has been written following the Mistral instruct template.
        # if PROMPT_TEMPLATE == "instruct":

        #     # my_prompt = "[INST]This is a spoken dialog scenario between a teacher and a 8 years old child student. \
        #     # The teacher is teaching mathemathics to the child student. \
        #     # As the student is a child, the teacher needs to stay gentle all the time. Please provide the next valid response for the followig conversation.\
        #     # You play the role of a teacher. Here is the beginning of the conversation : \
        #     # Teacher : Hi! How are your today ? \
        #     # Child : I am fine, and I can't wait to learn mathematics ![/INST]"

        #     my_prompt = "<<SYS>>This is a spoken dialog scenario between a teacher and a 8 years old child student. \
        #     The teacher is teaching mathemathics to the child student. \
        #     As the student is a child, the teacher needs to stay gentle all the time. Please provide the next valid response for the followig conversation.\
        #     You play the role of a teacher. Here is the beginning of the conversation :<</SYS>>\
        #     [INST] Child : Hello ! [/INST]\
        #     Teacher : Hi! How are your today ?\
        #     [INST]Child : I am fine, and I can't wait to learn mathematics ![/INST]"

        # # The prompt has been written following the llama template.
        # if PROMPT_TEMPLATE == "llama":

        #     my_prompt = "This is a spoken dialog scenario between a teacher and a 8 years old child student. \
        #     The teacher is teaching mathemathics to the child student. \
        #     As the student is a child, the teacher needs to stay gentle all the time. Please provide the next valid response for the followig conversation.\
        #     You play the role of a teacher. Here is the beginning of the conversation : \
        #     Teacher : Hi! How are your today ? \
        #     Child : I am fine, and I can't wait to learn mathematics !"

        # # The generation method is not implemented yet.
        # else:
        #     raise NotImplementedError("Template "+PROMPT_TEMPLATE+" not implemented")
        
        time_0 = time.time()
        model_output = generate_text_with_model_call(my_prompt, my_model)
        time_1 = time.time()
        
        # for all templates
        last_sentence = model_output["choices"][0]["text"].strip()
        print("\n\n["+str(round(time_1 - time_0, 3)) + "s] " + last_sentence)

        # only for instruct template
        # last_sentence = full_prompt.split("[/INST]")[-1]
        # print("["+str(round(time_1 - time_0, 3)) + "s] " + last_sentence)

    # The generation method used is the model.create_chat_completion(chat_history) function.
    # With the llama_cpp_python library, and the create_chat_completion() function, the chat_history is 
    # automatically generated from the template corresponding to the model.
    elif MODEL_GENERATION_METHOD == "chat_completion":
        
        time_0 = time.time()
        # model_output = generate_text_with_chat_completion(chat_history, my_model)
        model_output = generate_text_with_chat_completion_stream(chat_history, my_model)
        time_1 = time.time()

        # print(model_output)
        role = model_output["choices"][0]["message"]['role']
        text = model_output["choices"][0]["message"]['content']
        last_sentence = role + " : " + text
        # if stream
        # role = model_output['role']
        # text = model_output['content']
        print("\n\n["+str(round(time_1 - time_0, 3)) + "s] " + last_sentence)
    
    elif MODEL_GENERATION_METHOD == "special" :
        time_0 = time.time()
        model_output = generate_special(my_prompt, my_model, stop_criteria)
        time_1 = time.time()

        # print(model_output)
        # last_sentence = model_output["choices"][0]["text"].strip()
        last_sentence = model_output.decode("utf-8")
        print("\n\n["+str(round(time_1 - time_0, 3)) + "s] " + last_sentence)
    
    return last_sentence

def add_stop_pattern_string(pattern_string, tokenizer):
    STOP_TOKEN_PATTERNS.append(tokenizer.tokenize(bytes(pattern_string, "utf-8"), add_bos=False))

def add_stop_pattern_string_list(pattern_string_list, tokenizer):
    for pat in pattern_string_list:
        add_stop_pattern_string(pat, tokenizer)

def remove_stop_patterns(sentence):
    last_chunck_string = sentence.decode("utf-8")
    for i, pat in enumerate(STOP_TOKEN_TEXT_PATTERNS):
        if pat == last_chunck_string[-len(pat):]:
            sentence = sentence[:-len(pat)]
            break
    while sentence[-1:] == b"\n":
        sentence = sentence[:-1]
    return sentence

def main():

    # my_prompt, chat_history, chat_format = set_prompt()
    # my_model = set_model(chat_format)
    short_prompts, long_prompts = set_prompts()
    print("nb prompts = ", len(short_prompts) + len(long_prompts))
    my_model = set_model(None)

    # patterns_list = ["Child:", "Child :", " Child:", " Child :", " Child: ", " Child : "]
    patterns_list = ["Child:", "Child :"]
    add_stop_pattern_string_list(patterns_list, my_model)
    STOP_TOKEN_TEXT_PATTERNS.extend(patterns_list)
    print("STOP_TOKEN_PATTERNS = ", STOP_TOKEN_PATTERNS)
    print("STOP_TOKEN_PATTERNS = ", STOP_TOKEN_TEXT_PATTERNS)

    def stop_criteria(tokens, logits):
        """
        Function used by the LLM to stop generate tokens when it meets certain criteria.

        Args:
            tokens (_type_): tokens generated by the LLM
            logits (_type_): _description_

        Returns:
            bool: returns True if it generated one of the tokens corresponding to STOP_TOKEN_IDS or STOP_TOKEN_TEXT.
        """

        is_stopping_id = tokens[-1] in STOP_TOKEN_IDS
        is_stopping_text = my_model.detokenize([tokens[-1]]) in STOP_TOKEN_TEXT
        # if is_stopping_id or is_stopping_text:
        #     print(tokens[-1])
        #     print(my_model.detokenize([tokens[-1]]))
        #     print(my_model.detokenize(tokens))
        return is_stopping_id or is_stopping_text

    def stop_regulate_multiple_utterances(tokens, logits):
        """
        Function used by the LLM to stop generate tokens when it meets certain criteria.

        Args:
            tokens (_type_): tokens generated by the LLM
            logits (_type_): _description_

        Returns:
            bool: returns True if it generated one of the tokens corresponding to STOP_TOKEN_IDS or STOP_TOKEN_TEXT.
        """

        is_stopping_id = tokens[-1] in STOP_TOKEN_IDS
        is_stopping_text = my_model.detokenize([tokens[-1]]) in STOP_TOKEN_TEXT

        # last_chunk = my_model.detokenize(tokens[-10:])
        # is_stopping_text_pattern = False
        # print(last_chunk)
        # print(STOP_TOKEN_TEXT_PATTERNS)
        # for pat in STOP_TOKEN_TEXT_PATTERNS:
        #     if bytes(pat,  "utf-8") == last_chunk[-len(pat):]:
        #         return True
                # is_stopping_text_pattern = True
                # break
        
        # is_stopping_pattern = False
        # for pat in STOP_TOKEN_PATTERNS:
        #     if pat == list(tokens[-len(pat):]):
        #         is_stopping_pattern = True
        #         break

        is_stopping_pattern = False
        max_pattern_size = max([len(p) for p in STOP_TOKEN_PATTERNS])
        last_chunck_string = my_model.detokenize(tokens[-max_pattern_size:]).decode("utf-8")
        for i, pat in enumerate(STOP_TOKEN_TEXT_PATTERNS):
            if pat == last_chunck_string[-len(pat):]:
                return True

        return is_stopping_id or is_stopping_text or is_stopping_pattern
        
    # Add the model's EOS token to the STOP_TOKEN_IDS if it's not already in it.
    STOP_TOKEN_IDS.append(my_model.token_eos())

    short_nb_u = []
    long_nb_u = []
    for i, my_prompt in enumerate(short_prompts):
        print("\n\nshort prompt "+str(i)+" :")
        generated_u = generate(my_model, my_prompt, None, stop_regulate_multiple_utterances)
        short_nb_u.append(count_nb_utterances_2(generated_u))
        print("\nnumber of Turns generated : ",short_nb_u[-1])
    for i, my_prompt in enumerate(long_prompts):
        print("\n\nlong prompt "+str(i)+" :")
        generated_u = generate(my_model, my_prompt, None, stop_regulate_multiple_utterances)
        long_nb_u.append(count_nb_utterances_2(generated_u))
        print("\nnumber of Turns generated : ",long_nb_u[-1])

    print("list nb u short prompts : ", short_nb_u)
    print("short prompts working : ", [s == 1 for s in short_nb_u])
    print("list nb u long prompts : ", long_nb_u)
    print("long prompts working : ", [l == 1 for l in long_nb_u])

def count_nb_utterances(dialogue):
    list_u = dialogue.split("Child")
    return len(list_u)

def count_nb_utterances_2(dialogue):
        delimiters = ["Child:", "Child :"]
        regex_pattern = '|'.join(map(re.escape, delimiters))
        list_u = re.split(regex_pattern, dialogue)
        return len(list_u)

if __name__ == '__main__':
    main()