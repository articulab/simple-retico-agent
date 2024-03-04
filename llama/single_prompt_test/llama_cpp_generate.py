"""
Conda environment to activate before running the code : llama_cpp_cuda
conda activate llama_cpp_cuda
python llama/single_prompt_test/llama_cpp_generate.py
"""

import time
from llama_cpp import Llama

def generate_text_from_prompt(
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
        # max_tokens=max_tokens,
        max_tokens=0,
        temperature=temperature,
        top_p=top_p,
        echo=echo,
        stop=stop,
    )
    return model_output

def generate_text_from_chat_history(
    chat_history,
    my_model,
    max_tokens = 100,
    temperature = 0.3,
    top_p = 0.1,
    stop = ["Q", "\n"],
    ):

    # Define the parameters
    model_output = my_model.create_chat_completion(
        chat_history,
        # max_tokens=max_tokens,
        max_tokens=0,
        temperature=temperature,
        top_p=top_p,
        stop=stop,
    )
    return model_output

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

MODEL_REPO = "TheBloke/Mistral-7B-Instruct-v0.2-GGUF"
MODEL_NAME = "mistral-7b-instruct-v0.2.Q4_K_S.gguf"

MODEL_LOADING_METHOD = "from_pretrained"
# MODEL_LOADING_METHOD = "local_file"

MODEL_GENERATION_METHOD = "chat_completion"
# MODEL_GENERATION_METHOD = "model_call"
# MODEL_GENERATION_METHOD = "generate_function"

# CONTEXT_SIZE = 512
CONTEXT_SIZE = 8192
# CONTEXT_SIZE = 32768
# CONTEXT_SIZE = 0
# CONTEXT_SIZE = 100
N_GPU_LAYERS = 100

# PROMPT_TEMPLATE = "mistral"
PROMPT_TEMPLATE = "llama"
# PROMPT_TEMPLATE = "instruct"


# Defining stoping tokens so that the generation is not infinite
# b"\n" == 13
# eos == 2
# b"      " == ?
STOP_TOKEN_IDS = [2,13]
STOP_TOKEN_TEXT = [b"\n", b"        "]

if __name__ == "__main__":

    if MODEL_LOADING_METHOD == "from_pretrained":

        my_model = Llama.from_pretrained(
            repo_id=MODEL_REPO, 
            filename=MODEL_NAME,
            device_map="cuda",
            n_ctx=CONTEXT_SIZE, 
            n_gpu_layers=N_GPU_LAYERS,
        )
    elif MODEL_LOADING_METHOD == "local_file":

        my_model = Llama(
            model_path=MODEL_PATH, 
            n_ctx=CONTEXT_SIZE, 
            n_gpu_layers=N_GPU_LAYERS
        )

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
    
    # MODEL_GENERATION_METHOD = "chat_completion"
    # MODEL_GENERATION_METHOD = "model_call"
    if MODEL_GENERATION_METHOD == "generate_function":

        my_prompt = b"<<SYS>>This is a spoken dialog scenario between a teacher and a 8 years old child student. \
        The teacher is teaching mathemathics to the child student. \
        As the student is a child, the teacher needs to stay gentle all the time. Please provide the next valid response for the followig conversation.\
        You play the role of a teacher. Here is the beginning of the conversation :<</SYS>>\
        [INST] Child : Hello ! [/INST]\
        Teacher : Hi! How are your today ?\
        [INST]Child : I am fine, and I can't wait to learn mathematics ![/INST]"

        time_0 = time.time()
        tokens = my_model.tokenize(my_prompt)
        last_sentence = b""
        for token in my_model.generate(
            tokens,
            stopping_criteria=stop_criteria,
            top_k=40,
            top_p=0.95,
            temp=1.0,
            repeat_penalty=1.1
            ):        
            last_sentence += my_model.detokenize([token])
        time_1 = time.time()
        
        # print(model_output)
        # full_prompt = model_output["choices"][0]["text"].strip()
        # last_sentence = full_prompt.split("[/INST]")[-1]
        print("["+str(round(time_1 - time_0, 3)) + "s] " + last_sentence.decode("utf-8"))
    
    elif MODEL_GENERATION_METHOD == "model_call":

        if PROMPT_TEMPLATE == "instruct":

            # my_prompt = "[INST]This is a spoken dialog scenario between a teacher and a 8 years old child student. \
            # The teacher is teaching mathemathics to the child student. \
            # As the student is a child, the teacher needs to stay gentle all the time. Please provide the next valid response for the followig conversation.\
            # You play the role of a teacher. Here is the beginning of the conversation : \
            # Teacher : Hi! How are your today ? \
            # Child : I am fine, and I can't wait to learn mathematics ![/INST]"

            my_prompt = "<<SYS>>This is a spoken dialog scenario between a teacher and a 8 years old child student. \
            The teacher is teaching mathemathics to the child student. \
            As the student is a child, the teacher needs to stay gentle all the time. Please provide the next valid response for the followig conversation.\
            You play the role of a teacher. Here is the beginning of the conversation :<</SYS>>\
            [INST] Child : Hello ! [/INST]\
            Teacher : Hi! How are your today ?\
            [INST]Child : I am fine, and I can't wait to learn mathematics ![/INST]"

        else:
            raise NotImplementedError("Template "+PROMPT_TEMPLATE+" not implemented")
        
        time_0 = time.time()
        model_output = generate_text_from_prompt(my_prompt, my_model)
        time_1 = time.time()
        # print(model_output)
        full_prompt = model_output["choices"][0]["text"].strip()
        last_sentence = full_prompt.split("[/INST]")[-1]
        print("["+str(round(time_1 - time_0, 3)) + "s] " + last_sentence)

    elif MODEL_GENERATION_METHOD == "chat_completion":

        if PROMPT_TEMPLATE == "llama":

            chat_history = [
                {"role": "system", "content": "This is a spoken dialog scenario between a teacher and a 8 years old child student. \
                    The teacher is teaching mathemathics to the child student. \
                    As the student is a child, the teacher needs to stay gentle all the time. Please provide the next valid response for the followig conversation.\
                    You play the role of a teacher. Here is the beginning of the conversation :"},
                {"role": "user", "content": "Hello !"}, 
                {"role": "assistant", "content": "Hi! How are your today ?"},
                {"role": "user", "content": "I am fine, and I can't wait to learn mathematics !"},
            ]
        else:
            raise NotImplementedError("Template "+PROMPT_TEMPLATE+" not implemented")
        
        time_0 = time.time()
        model_output = generate_text_from_chat_history(chat_history, my_model)
        time_1 = time.time()

        # print(model_output)
        role = model_output["choices"][0]["message"]['role']
        text = model_output["choices"][0]["message"]['content']
        print("["+str(round(time_1 - time_0, 3)) + "s] " + role + " : " + text)

# my_model = Llama.from_pretrained(
#     repo_id="TheBloke/TinyLlama-1.1B-Chat-v0.3-GGUF",
#     filename="*Q5_K_M.gguf",
#     verbose=False
# )