"""
Conda environment to activate before running the code : llama_cpp_cuda
conda activate llama_cpp_cuda
python llama/single_prompt_test/llama_cpp_speculative_test.py
"""

import time
from llama_cpp import Llama
from llama_cpp.llama_speculative import LlamaPromptLookupDecoding

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
    max_tokens = 100,
    temperature = 0.3,
    top_p = 0.1,
    stop = ["Q", "\n"],
    ):

    # Define the parameters
    model_output = my_model.create_chat_completion(
        chat_history,
        max_tokens=max_tokens,
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

CONTEXT_SIZE = 512
N_GPU_LAYERS = 100
TEMPLATE = "mistral"
# TEMPLATE = "instruct"

if __name__ == "__main__":

    # my_speculative_model = Llama(model_path=MODEL_PATH, n_ctx=CONTEXT_SIZE, n_gpu_layers=N_GPU_LAYERS)
    
    my_speculative_model = Llama(
        model_path=MODEL_PATH,
        n_ctx=CONTEXT_SIZE,
        n_gpu_layers=N_GPU_LAYERS,
        draft_model=LlamaPromptLookupDecoding(num_pred_tokens=10) # num_pred_tokens is the number of tokens to predict 10 is the default and generally good for gpu, 2 performs better for cpu-only machines.
    )

    # No difference using the draft_model argument.



    if TEMPLATE == "mistral":

        # chat_history = [
        #     {"role": "system", "content": "This is a spoken dialog scenario between a teacher and a 8 years old child student. \
        #         The teacher is teaching mathemathics to the child student. \
        #         As the student is a child, the teacher needs to stay gentle all the time. Please provide the next valid response for the followig conversation.\
        #         You play the role of a teacher. Here is the beginning of the conversation :"},
        #     {"role": "user", "content": "Hello !"}, 
        #     {"role": "assistant", "content": "Hi! How are your today ?"},
        #     {"role": "user", "content": "I am fine, and I can't wait to learn mathematics !"},
        # ]

        chat_history = [
            {"role": "system", "content": "This is a spoken dialog scenario between a teacher and a 8 years old child student. \
                The teacher is teaching mathemathics to the child student. \
                As the student is a child, the teacher needs to stay gentle all the time. Please provide the next valid response for the followig conversation.\
                You play the role of a teacher. Here is the beginning of the conversation :"},
            {"role": "user", "content": "Hello !"}, 
            {"role": "assistant", "content": "Hi! How are"},
        ]

        time_0 = time.time()
        model_output = generate_text_from_chat_history(chat_history, my_speculative_model)
        time_1 = time.time()

        # print(model_output)
        role = model_output["choices"][0]["message"]['role']
        text = model_output["choices"][0]["message"]['content']
        print("["+str(round(time_1 - time_0, 3)) + "s] " + role + " : " + text)

    if TEMPLATE == "instruct":

        # my_prompt = "[INST]This is a spoken dialog scenario between a teacher and a 8 years old child student. \
        # The teacher is teaching mathemathics to the child student. \
        # As the student is a child, the teacher needs to stay gentle all the time. Please provide the next valid response for the followig conversation.\
        # You play the role of a teacher. Here is the beginning of the conversation : \
        # Teacher : Hi! How are your today ? \
        # Child : I am fine, and I can't wait to learn mathematics ![/INST]"

        # my_prompt = "<<SYS>>This is a spoken dialog scenario between a teacher and a 8 years old child student. \
        # The teacher is teaching mathemathics to the child student. \
        # As the student is a child, the teacher needs to stay gentle all the time. Please provide the next valid response for the followig conversation.\
        # You play the role of a teacher. Here is the beginning of the conversation :<</SYS>>\
        # [INST] Child : Hello ! [/INST]\
        # Teacher : Hi! How are your today ?\
        # [INST]Child : I am fine, and I can't wait to learn mathematics ![/INST]"

        my_prompt = "<<SYS>>This is a spoken dialog scenario between a teacher and a 8 years old child student. \
        The teacher is teaching mathemathics to the child student. \
        As the student is a child, the teacher needs to stay gentle all the time. Please provide the next valid response for the followig conversation.\
        You play the role of a teacher. Here is the beginning of the conversation :<</SYS>>\
        [INST] Child : Hello ! [/INST]\
        Teacher : Hi! How are"

        # my_prompt = "[INST] Teacher : Hi! How are your today ? [/INST]\
        # Student : Hello, I am"
        
        time_0 = time.time()
        model_output = generate_text_from_prompt(my_prompt, my_speculative_model)
        time_1 = time.time()
        # print(model_output)
        full_prompt = model_output["choices"][0]["text"].strip()
        print(full_prompt)
        last_sentence = full_prompt.split("[/INST]")[-1]
        print("["+str(round(time_1 - time_0, 3)) + "s] " + last_sentence)


# my_model = Llama.from_pretrained(
#     repo_id="TheBloke/TinyLlama-1.1B-Chat-v0.3-GGUF",
#     filename="*Q5_K_M.gguf",
#     verbose=False
# )