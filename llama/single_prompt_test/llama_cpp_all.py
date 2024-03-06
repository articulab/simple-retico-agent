"""
Conda environment to activate before running the code : llama_cpp_cuda
conda activate llama_cpp_cuda
python llama/single_prompt_test/llama_cpp_clean.py

llama_cpp_python from API :

To understand all the different ways to generate text using the model :
__call__    ->    create_completion()    ->    _create_completion()    ->    generate()
create_chat_completion()    ->    create_completion()
"""

import time
from llama_cpp import Llama

def generate_text_with_model_call(
    my_prompt,
    my_model,
    ):

    # Define the parameters
    model_output = my_model(
        my_prompt,
        max_tokens=MAX_TOKENS,
        temperature=TEMP,
        top_p=TOP_P,
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
        temperature=TEMP,
        top_p=TOP_P,
        # echo=ECHO,
        stop=STOP_TOKEN_TEXT,
    )
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
        temp=TEMP,
        top_p=TOP_P,
        # echo=ECHO,
        ):        
        last_sentence += my_model.detokenize([token])
    return last_sentence


##############
# MODEL CONFIG
##############

# LOCAL FILE
# MODEL_PATH = "./models/llama-2-7b-chat.Q4_K_S.gguf"
# MODEL_PATH = "./models/mistral-7b-instruct-v0.2.Q4_K_S.gguf"
# MODEL_PATH = "./models/mistral-7b-v0.1.Q4_K_S.gguf"
# MODEL_PATH = "./models/zephyr-7b-beta.Q4_0.gguf"
MODEL_PATH = "./models/llama-2-13b-chat.Q4_K_S.gguf"
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

CONTEXT_SIZE = 512
# CONTEXT_SIZE = 2000
# CONTEXT_SIZE = 4096
# CONTEXT_SIZE = 8192
# CONTEXT_SIZE = 32768
# CONTEXT_SIZE = 0
# CONTEXT_SIZE = 100

# STOP_TOKEN_IDS = [2]
STOP_TOKEN_IDS = []
# STOP_TOKEN_TEXT = ["</s>", "Q:", "\n"]
STOP_TOKEN_TEXT = []

# GENERATION PARAMETERS
MAX_TOKENS = 0 # max_token = n_ctx
TEMP = 0.3
TOP_P = 0.1
ECHO = True


################
# CONFIG METHODS
################

MODEL_LOADING_METHOD = "from_pretrained"
# MODEL_LOADING_METHOD = "local_file"

MODEL_GENERATION_METHOD = "chat_completion"
# MODEL_GENERATION_METHOD = "model_call"
# MODEL_GENERATION_METHOD = "generate_function"

# PROMPT_TEMPLATE = "mistral"
# PROMPT_TEMPLATE = "llama"
# PROMPT_TEMPLATE = "instruct"
# PROMPT_TEMPLATE = "chat_mistral"
# PROMPT_TEMPLATE = "chat_llama"
PROMPT_TEMPLATE = "chat"

CHAT_FORMAT = "llama-2"
# CHAT_FORMAT = "mistral-instruct"
# CHAT_FORMAT = "chatml"
# CHAT_FORMAT = "zephyr"


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

    # The prompt has been written following the llama template.
    elif PROMPT_TEMPLATE == "llama":

        my_prompt = "This is a spoken dialog scenario between a teacher and a 8 years old child student. \
        The teacher is teaching mathemathics to the child student. \
        As the student is a child, the teacher needs to stay gentle all the time. Please provide the next valid response for the followig conversation.\
        You play the role of a teacher. Here is the beginning of the conversation : \
        Teacher : Hi! How are your today ? \
        Child : I am fine, and I can't wait to learn mathematics !"

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
        )
    return my_model


def generate(my_model, my_prompt, chat_history, stop_criteria):

    # The generation method used is the model.generate(tokens) function.
    if MODEL_GENERATION_METHOD == "generate_function":

        my_bytes_prompt = bytes(my_prompt, "utf-8")

        time_0 = time.time()
        last_sentence = generate_text_with_generate_function(my_bytes_prompt, my_model, stop_criteria)
        time_1 = time.time()
        
        # print(model_output)
        # full_prompt = model_output["choices"][0]["text"].strip()
        # last_sentence = full_prompt.split("[/INST]")[-1]
        print("["+str(round(time_1 - time_0, 3)) + "s] " + last_sentence.decode("utf-8"))

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
        full_prompt = model_output["choices"][0]["text"].strip()
        print("["+str(round(time_1 - time_0, 3)) + "s] " + full_prompt)

        # only for instruct template
        # last_sentence = full_prompt.split("[/INST]")[-1]
        # print("["+str(round(time_1 - time_0, 3)) + "s] " + last_sentence)

    # The generation method used is the model.create_chat_completion(chat_history) function.
    # With the llama_cpp_python library, and the create_chat_completion() function, the chat_history is 
    # automatically generated from the template corresponding to the model.
    elif MODEL_GENERATION_METHOD == "chat_completion":
        
        time_0 = time.time()
        model_output = generate_text_with_chat_completion(chat_history, my_model)
        time_1 = time.time()

        # print(model_output)
        role = model_output["choices"][0]["message"]['role']
        text = model_output["choices"][0]["message"]['content']
        print("["+str(round(time_1 - time_0, 3)) + "s] " + role + " : " + text)


if __name__ == "__main__":

    my_prompt, chat_history, chat_format = set_prompt()

    my_model = set_model(chat_format)

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
        
    # Add the model's EOS token to the STOP_TOKEN_IDS if it's not already in it.
    STOP_TOKEN_IDS.append(my_model.token_eos())

    generate(my_model, my_prompt, chat_history, stop_criteria)