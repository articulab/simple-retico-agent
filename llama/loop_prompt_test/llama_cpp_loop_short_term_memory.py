"""
Conda environment to activate before running the code : llama_cpp_cuda
conda activate llama_cpp_cuda

python llama/loop_prompt_test/llama_cpp_loop_short_term_memory.py
"""

import time
from llama_cpp import Llama, llama_token_eos

def ask_for_sentence():
    sentence = input("your answer : ")
    return sentence

def print_chat_history(chat_history):
    for sentence in chat_history:
        print("[" + sentence["role"] + "] : " + sentence["content"] + "\n")

def generate_next_sentence(
    my_prompt,
    my_model,
    max_tokens = 150,
    temperature = 0.3,
    top_p = 0.9,
    echo = True,
    stop = ["Q", "\n"], #stop=["</s>"]
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
    # max_tokens = 100,
    # temperature = 0.3,
    # top_p = 0.9,
    # stop = ["Q", "\n"],
    ):

    # stop = [str(my_model.token_eos()), "</s>"]
    # stop = ["</s>", "<|im_end|>"]
    stop = ["</s>"]
    # stop = ["<|im_end|>"]

    # Define the parameters
    model_output = my_model.create_chat_completion(
        chat_history,
        # max_tokens=max_tokens,
        # temperature=temperature,
        # top_p=top_p,
        stop=stop,
    )
    return model_output

def calculate_short_memory_chat(my_model, chat_history):
    formatted_chat = format_llama2(chat_history).prompt
    print("formatted_chat = ", formatted_chat)
    formatted_chat_nb_tokens = len(my_model.tokenize(bytes(formatted_chat, "utf-8")))
    print("formatted_chat_nb_tokens = ", formatted_chat_nb_tokens)
    while formatted_chat_nb_tokens > SHORT_TERM_MEMORY_CONTEXT_SIZE :
        chat_history.pop(1)
        formatted_chat = format_llama2(chat_history).prompt
        formatted_chat_nb_tokens = len(my_model.tokenize(bytes(formatted_chat, "utf-8")))
        print("formatted_chat_nb_tokens = ", formatted_chat_nb_tokens)
    return chat_history

def calculate_short_memory(chat_history):
    if len(chat_history) > SHORT_TERM_MEMORY_SIZE+1:
        short_term_memory = chat_history[-SHORT_TERM_MEMORY_SIZE:]
        short_term_memory.insert(0,chat_history[0])
    else :
        short_term_memory = chat_history.copy()
    return short_term_memory

def calculate_short_memory_2(chat_history, tokenizer):
    short_term_memory = chat_history
    input_ids = tokenizer.apply_chat_template(
            short_term_memory,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt"
        )
    nb_tokens = len(input_ids[0])
    print("nb tokens "+str(nb_tokens))
    while nb_tokens > SHORT_TERM_MEMORY_CONTEXT_SIZE:
        print("\nPOPING A PROMPT\n")
        assert len(short_term_memory) > 2 # assert that there is more than the system prompt and the last prompt
        short_term_memory.pop(0)
        # short_term_memory.pop(1) # poping the oldest prompt that is not the system prompt
        input_ids = tokenizer.apply_chat_template(
            short_term_memory,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt"
        )
        nb_tokens = len(input_ids[0])
        print("nb tokens "+str(nb_tokens))
    return short_term_memory



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
# CONTEXT_SIZE = 512
CONTEXT_SIZE = 1000
# CONTEXT_SIZE = 8192
# CONTEXT_SIZE = 32768
# CONTEXT_SIZE = 0 # from model
SHORT_TERM_MEMORY_SIZE = 5
# SHORT_TERM_MEMORY_CONTEXT_SIZE = 4096
SHORT_TERM_MEMORY_CONTEXT_SIZE = 300
N_GPU_LAYERS = 100

# CHAT_FORMAT = "llama-2"
# CHAT_FORMAT = "mistral-instruct"

# TEMPLATE = "llama-2"
# TEMPLATE = "zephyr"
# TEMPLATE = "mistral-instruct"
# TEMPLATE = "instruct"
# TEMPLATE = "instruct_2"
# TEMPLATE = "generate"
TEMPLATE = "chat"


MODEL_REPO = "TheBloke/Mistral-7B-Instruct-v0.2-GGUF"
MODEL_NAME = "mistral-7b-instruct-v0.2.Q4_K_S.gguf"

MODEL_LOADING_METHOD = "from_pretrained"
# MODEL_LOADING_METHOD = "local_file"

MODEL_GENERATION_METHOD = "chat_completion"
# MODEL_GENERATION_METHOD = "model_call"
# MODEL_GENERATION_METHOD = "generate_function"

STOP_TOKEN_IDS = [2,13]
STOP_TOKEN_TEXT = [b"\n", b"        "]

from transformers import AutoTokenizer
from llama_cpp.llama_chat_format import LlamaChatCompletionHandlerRegistry, hf_tokenizer_config_to_chat_formatter, ChatFormatter, chat_formatter_to_chat_completion_handler, register_chat_format, format_llama2, format_zephyr, format_mistral_instruct
# import tiktoken
# TOKENIZER_PATH = "TheBloke/Mistral-7B-Instruct-v0.2-GGUF"
TOKENIZER_PATH = "mistralai/Mistral-7B-Instruct-v0.2"

# def num_tokens_from_string(string, encoding_name):
#     """Returns the number of tokens in a text string."""
#     encoding = tiktoken.get_encoding(encoding_name)
#     num_tokens = len(encoding.encode(string))
#     return num_tokens

def main():

    # my_model = Llama(model_path=MODEL_PATH, n_ctx=CONTEXT_SIZE, n_gpu_layers=N_GPU_LAYERS)
    # my_model = Llama(model_path=MODEL_PATH, n_ctx=CONTEXT_SIZE, n_gpu_layers=N_GPU_LAYERS, chat_format=CHAT_FORMAT)
    
    # my_model = Llama(model_path=MODEL_PATH, n_gpu_layers=N_GPU_LAYERS)

    if TEMPLATE == "generate":

        my_model = Llama(model_path=MODEL_PATH, n_ctx=CONTEXT_SIZE, n_gpu_layers=N_GPU_LAYERS)

        STOP_TOKEN_IDS.append(my_model.token_eos())

        def stop_criteria(tokens, logits):
                # OLD
                # is_eos = tokens[-1] in my_model.token_eos()
                # is_next_line = my_model.detokenize([tokens[-1]]) == b"\n"
                # is_tab = my_model.detokenize([tokens[-1]]) == b"      "
                # if is_eos or is_next_line or is_tab:
                #     print(tokens[-1])
                #     print(my_model.detokenize([tokens[-1]]))
                #     print(my_model.detokenize(tokens))
                # return is_eos or is_next_line or is_tab

                is_stopping_id = tokens[-1] in STOP_TOKEN_IDS
                is_stopping_text = my_model.detokenize([tokens[-1]]) in STOP_TOKEN_TEXT
                # if is_stopping_id or is_stopping_text:
                #     print(tokens[-1])
                #     print(my_model.detokenize([tokens[-1]]))
                #     print(my_model.detokenize(tokens))
                return is_stopping_id or is_stopping_text

        # my_prompt = b"<<SYS>>This is a spoken dialog scenario between a teacher and a 8 years old child student. \
        # The teacher is teaching mathemathics to the child student. \
        # As the student is a child, the teacher needs to stay gentle all the time. Please provide the next valid response for the followig conversation.\
        # You play the role of a teacher. Here is the beginning of the conversation :<</SYS>>\
        # [INST]Child : Hello ![/INST]\
        # Teacher : Hi! How are your today ?\
        # [INST]Child : I am fine, and I can't wait to learn mathematics ![/INST]"

        my_prompt = b"<<SYS>>This is a spoken dialog scenario between a teacher and a 8 years old child student.\
        The teacher is teaching mathemathics to the child student.\
        As the student is a child, the teacher needs to stay gentle all the time. Please provide the next valid response for the followig conversation.\
            You play the role of a teacher. Here is the beginning of the conversation :<</SYS>>\
        [INST]Child : Hello ![/INST]\
        Teacher : Hi! How are your today ?\
        [INST]Child : I am fine, and I can't wait to learn mathematics ![/INST]\
        Teacher : Great news! I'm excited to help you learn as well. Let's start with something simple. Can you tell me what comes before the number five ?\
        [INST]Child : four ?[/INST]\
        Teacher : Excellent job! Keep it up. Let's move on to something a little more challenging. Can you tell me how many apples you have if I give you two baskets, each having three apples?\
        [INST]Child : six ?[/INST]\
        Teacher : Correct! Now let's see if you can find 3 + 3. Great work! Keep going like this and we will learn maths together step by step.\
        [INST]Child : okay.[/INST]\
        Teacher: That's the spirit! Let's continue practicing these simple addition problems together. If you ever feel confused or have any doubts, please don't hesitate to ask questions.\
        [INST]Child : yeah ! let's do another exercice.[/INST]\
        Teacher: "

        DELIMITER = b"[INST]"
        # Initialize short term memory from first prompt
        split_utterances = my_prompt.split(DELIMITER)
        # print("nb QA = "+str(len(split_utterances)))
        utterances = [split_utterances[0]]
        # size_per_utterance = []
        size_per_utterance = [len(my_model.tokenize(utterances[0]))]
        for u in split_utterances[1:]:
            utterances.append(DELIMITER+u)
            size_per_utterance.append(len(my_model.tokenize(utterances[-1])))

        print(size_per_utterance)

        for i in range(CONV_LENGTH):

            print("Loading...")
            # Ask for model sentence
            time_0 = time.time()
            tokens = my_model.tokenize(my_prompt)
            last_sentence = b""
            last_sentence_nb_tokens = 0
            for token in my_model.generate(
                tokens,
                stopping_criteria=stop_criteria,
                top_k=40,
                top_p=0.95,
                temp=1.0,
                repeat_penalty=1.1
                ):        
                last_sentence += my_model.detokenize([token])
                last_sentence_nb_tokens += 1
            time_1 = time.time()
            my_prompt += last_sentence + b"\n"            
            print(my_prompt.decode("utf-8"))
            print("["+str(round(time_1 - time_0, 3)) + "s]")

            # Add model sentence to short term memory
            # add the last sentence from model to the last utterance which contains only the sentence from user
            tmp_utt = last_sentence
            tmp_utt_size = last_sentence_nb_tokens
            utterances[-1] += tmp_utt
            size_per_utterance[-1] += tmp_utt_size

            # Ask for user sentence
            user_sentence = bytes(ask_for_sentence(), 'utf-8')
            user_sentence_complete =  b"[INST]Child : " + user_sentence + b"[/INST]     Teacher: "
            my_prompt += user_sentence_complete

            # Add user sentence to short term memory
            # append last sentence from user as the last utterance
            tmp_utt = user_sentence_complete
            tmp_utt_size = len(my_model.tokenize(user_sentence_complete))
            utterances.append(tmp_utt)
            size_per_utterance.append(tmp_utt_size)

            # Calculate short term memory
            nb_tokens = sum(size_per_utterance)
            print("num token : "+str(nb_tokens))
            if nb_tokens >= SHORT_TERM_MEMORY_CONTEXT_SIZE:
                print("nb QA = "+str(len(utterances)))
                print("size_per_utterance = "+str(size_per_utterance))
                while sum(size_per_utterance) >= SHORT_TERM_MEMORY_CONTEXT_SIZE:
                    utterances.pop(1) # do not pop the system prompt explaining the scenario
                    res = size_per_utterance.pop(1)
                    print("POP "+str(res))
                my_prompt = b"".join(utterances)
                print(my_prompt)

            # recalculate every turn (more calculation)
            # tokens = my_model.tokenize(my_prompt)
            # print("num token : "+str(len(tokens)))
            # if len(tokens) >= SHORT_TERM_MEMORY_CONTEXT_SIZE:
            #     split_utterances = my_prompt.split(DELIMITER)
            #     print("nb QA = "+str(len(split_utterances)))
            #     size_per_utterance = []
            #     utterances = [split_utterances[0]]
            #     for u in split_utterances[1:]:
            #         utterances.append(DELIMITER+u)
            #         size_per_utterance.append(len(my_model.tokenize(utterances[-1])))
            #     print("nb QA = "+str(len(utterances)))
            #     print("size_per_utterance = "+str(size_per_utterance))
            #     while sum(size_per_utterance) >= SHORT_TERM_MEMORY_CONTEXT_SIZE:
            #         utterances.pop(1) # do not pop the system prompt explaining the scenario
            #         res = size_per_utterance.pop(1)
            #         print("POP "+str(res))
            #     my_prompt = b"".join(utterances)
            #     print(my_prompt)

                # TODO: find a way to not calculate these split and tokenize operations every time.


    elif TEMPLATE == "chat":

        # my_model = Llama(
        #     model_path=MODEL_PATH,
        #     n_ctx=CONTEXT_SIZE,
        #     n_gpu_layers=N_GPU_LAYERS,
        #     # chat_format="llama-2"
        #     )
        
        my_model = Llama.from_pretrained(
            repo_id=MODEL_REPO, 
            filename=MODEL_NAME,
            # device_map=DEVICE_MAP,
            n_ctx=CONTEXT_SIZE, 
            n_gpu_layers=N_GPU_LAYERS,
            # chat_format=chat_format,
        )

        tokenizer = AutoTokenizer.from_pretrained(MODEL_REPO)

        chat_history = [
            {"role": "system", "content": "This is a spoken dialog scenario between a teacher and a 8 years old child student. \
                The teacher is teaching mathemathics to the child student. \
                As the student is a child, the teacher needs to stay gentle all the time. You play the role of the teacher. \
                Please provide the next valid response for the following conversation."},
            {"role": "user", "content": "Hello !"}, 
            {"role": "assistant", "content": "Hi! How are your today ?"},
            {"role": "user", "content": "I am fine, and I can't wait to learn mathematics !"},
        ]


        print(tokenizer.apply_chat_template(chat_history))
        print(my_model.apply_chat_template(chat_history))
        print("handler = ",my_model.chat_handler)
        print("handler = ",my_model.chat_format)
        print(register_chat_format)
        print(register_chat_format(my_model.chat_format))
        # print(register_chat_format(my_model.chat_format)(chat_history))
        # print(LlamaChatCompletionHandlerRegistry)
        a = LlamaChatCompletionHandlerRegistry().get_chat_completion_handler_by_name(name=my_model.chat_format)
        print(a)
        print(a(llama=my_model, messages=chat_history))
        # print(ChatFormatter)
        # print(ChatFormatter(my_model.chat_format))
        # a = chat_formatter_to_chat_completion_handler(my_model.chat_format)
        # a = chat_formatter_to_chat_completion_handler(ChatFormatter)
        # a = hf_autotokenizer_to_chat_formatter(pretrained_model_name_or_path)
        # bos = my_model.detokenize([my_model.token_bos()])
        # eos = my_model.detokenize([my_model.token_eos()])
        # print(my_model.token_bos())
        # print(llama_token_eos())
        # print(bos)
        # print(eos)
        # bos = "<s>"
        # eos = "</s>"
        # tokenizer_config = {'chat_template':my_model.chat_format, "bos_token":bos, "eos_token":eos}
        # a = hf_tokenizer_config_to_chat_formatter(
        #     tokenizer_config
        # )
        # print("handler = ",a)
        # # print("handler = ",a())
        # print("handler = ",a(llama=my_model, messages=chat_history))
        # b = a(chat_history)
        # print(b)

        for i in range(CONV_LENGTH):

            chat_history = calculate_short_memory_chat(my_model, chat_history)

            time_0 = time.time()
            model_output = generate_text_from_chat_history(chat_history, my_model)
            time_1 = time.time()

            role = model_output["choices"][0]["message"]['role']
            text = model_output["choices"][0]["message"]['content']
            assert role == "assistant"
            chat_history.append({"role":role, "content":text})
            print_chat_history(chat_history)
            print("\n["+str(round(time_1 - time_0, 3)) + "s]")

            user_sentence = ask_for_sentence()
            chat_history.append({"role":"user", "content":user_sentence})

    elif TEMPLATE == "zephyr": # doesn't work yet because we haven't found zephyr stop parameter yet : the stop parameter of the create_chat_completion function

        # my_model = Llama(model_path=MODEL_PATH, n_ctx=CONTEXT_SIZE, n_gpu_layers=N_GPU_LAYERS, chat_format=TEMPLATE)
        # my_model = Llama(model_path=MODEL_PATH, n_ctx=CONTEXT_SIZE, n_gpu_layers=N_GPU_LAYERS)

        chat_history = [
            {"role": "system", "content": "This is a spoken dialog scenario between a teacher and a 8 years old child student. \
                The teacher is teaching mathemathics to the child student. \
                As the student is a child, the teacher needs to stay gentle all the time. You play the role of the teacher. \
                Please provide the next valid response for the following conversation."},
            {"role": "user", "content": "Hello !"}, 
            {"role": "assistant", "content": "Hi! How are your today ?"},
            {"role": "user", "content": "I am fine, and I can't wait to learn mathematics !"},
        ] 

        for i in range(CONV_LENGTH):

            short_term_memory = calculate_short_memory(chat_history)

            time_0 = time.time()
            model_output = generate_text_from_chat_history(short_term_memory, my_model)
            time_1 = time.time()

            print(model_output)
            usage = model_output["usage"]
            prompt_tokens = usage["prompt_tokens"]
            completion_tokens = usage["completion_tokens"]
            total_tokens = usage["total_tokens"]
            print(usage)
            role = model_output["choices"][0]["message"]['role']
            text = model_output["choices"][0]["message"]['content']
            assert role == "assistant"
            chat_history.append({"role":role, "content":text})
            short_term_memory.append({"role":role, "content":text})
            print_chat_history(short_term_memory)
            print("\n["+str(round(time_1 - time_0, 3)) + "s]")

            user_sentence = ask_for_sentence()
            chat_history.append({"role":"user", "content":user_sentence})

    elif TEMPLATE == "mistral-instruct": # doesn't work because it doesn't consider the system prompt as it is not in mistral template.

        my_model = Llama(model_path=MODEL_PATH, n_ctx=CONTEXT_SIZE, n_gpu_layers=N_GPU_LAYERS, chat_format=TEMPLATE)

        chat_history = [
            {"role": "system", "content": "This is a spoken dialog scenario between a teacher and a 8 years old child student. \
                The teacher is teaching mathemathics to the child student. \
                As the student is a child, the teacher needs to stay gentle all the time. You play the role of the teacher. \
                Please provide the next valid response for the following conversation."},
            {"role": "user", "content": "Hello !"}, 
            {"role": "assistant", "content": "Hi! How are your today ?"},
            {"role": "user", "content": "I am fine, and I can't wait to learn mathematics !"},
        ] 
        # nb_tokens = 41
        # chat_history = [
        #     {"role": "user", "content": "Hello !"}, 
        #     {"role": "assistant", "content": "Hi! How are your today ?"},
        #     {"role": "user", "content": "I am fine, and I can't wait to learn mathematics !"},
        # ]
        # nb_tokens = 16, 28, 46
        chat_history_2 = [
            {"role": "user", "content": "Hello !"}, 
            {"role": "assistant", "content": "Hi! How are your today ?"},
            {"role": "user", "content": "I am fine, and I can't wait to learn mathematics !"},
        ]
        # nb_tokens = 10, 18, 40
        # nb tokens calculation through tokenizer is smaller by 6 if it user is last speakerm and 10 if assistant is last speaker


        # tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
        # short_term_memory = calculate_short_memory_2(chat_history_2, tokenizer)
        # encoding = tiktoken.get_encoding("cl100k_base")
        # nb_tokens = len(encoding.encode("tiktoken is great!"))
        # nb_total_tokens = 0
        # for msg in chat_history_2 :
        #     nb_total_tokens += len(tiktoken.get_encoding("cl100k_base").encode(msg["content"]))
        # print("nb_tokens = "+str(nb_tokens) + " , "+str(nb_total_tokens) + " , " + str(nb_tokens - nb_total_tokens ))

        for i in range(CONV_LENGTH):

            short_term_memory = calculate_short_memory(chat_history)
            # short_term_memory = calculate_short_memory_2(chat_history, tokenizer)

            time_0 = time.time()
            model_output = generate_text_from_chat_history(short_term_memory, my_model)
            time_1 = time.time()

            usage = model_output["usage"]
            prompt_tokens = usage["prompt_tokens"]
            completion_tokens = usage["completion_tokens"]
            total_tokens = usage["total_tokens"]
            print(usage)
            role = model_output["choices"][0]["message"]['role']
            text = model_output["choices"][0]["message"]['content']
            # print("\n["+str(round(time_1 - time_0, 3)) + "s] " + role + " : " + text)
            assert role == "assistant"
            chat_history.append({"role":role, "content":text})
            short_term_memory.append({"role":role, "content":text})

            print_chat_history(short_term_memory)
            print("\n["+str(round(time_1 - time_0, 3)) + "s]")
            user_sentence = ask_for_sentence()
            chat_history.append({"role":"user", "content":user_sentence})

            # print(chat_history)

    elif TEMPLATE == "instruct_2":

        chat_history = [
            {"role": "system", "content": "This is a spoken dialog scenario between a teacher and a 8 years old child student. \
                The teacher is teaching mathemathics to the child student. \
                As the student is a child, the teacher needs to stay gentle all the time. Please provide the next valid response for the followig conversation.\
                You play the role of a teacher. Here is the beginning of the conversation :"},
            {"role": "user", "content": "Hello !"}, 
            {"role": "assistant", "content": "Hi! How are you today ?"},
            {"role": "user", "content": "I am fine, and I can't wait to learn mathematics !"},
        ]

        my_prompt = "<<SYS>>This is a spoken dialog scenario between a teacher and a 8 years old child student. \
        The teacher is teaching mathemathics to the child student. \
        As the student is a child, the teacher needs to stay gentle all the time. Please provide the next valid response for the followig conversation.\
        You play the role of a teacher. Here is the beginning of the conversation :<</SYS>>\
        [INST] Child : Hello ! [/INST]\
        Teacher : Hi! How are you today ?\
        [INST]Child : I am fine, and I can't wait to learn mathematics ![/INST]"

        for i in range(CONV_LENGTH):
            # short_term_memory = calculate_short_memory_instruct_2(chat_history)

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
            

    elif TEMPLATE == "instruct":

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