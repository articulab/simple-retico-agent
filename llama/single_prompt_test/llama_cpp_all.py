"""
Conda environment to activate before running the code : llama_cpp_cuda
conda activate llama_cpp_cuda
python llama/single_prompt_test/llama_cpp_all.py

llama_cpp_python from API :

To understand all the different ways to generate text using the model :
__call__    ->    create_completion()    ->    _create_completion()    ->    generate()
create_chat_completion()    ->    create_completion()
"""

import time
from llama_cpp import Llama

def generate_special(
    my_prompt,
    my_model,
    # stop_criteria,
    ):
    tokens = my_model.tokenize(bytes(my_prompt, "utf-8"), special=False)
    print(tokens)
    last_sentence = b""
    for token in my_model.generate(
        tokens,
        # stopping_criteria=stop_criteria,
        # temp=TEMP,
        # top_p=TOP_P,
        # echo=ECHO,
        ):        
        last_sentence += my_model.detokenize([token])
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
        # my_prompt = "[INST]This is a spoken dialog scenario between a teacher and a 8 years old child student. \
        # The teacher is teaching mathemathics to the child student. \
        # As the student is a child, the teacher needs to stay gentle all the time. Please provide the next valid response for the followig conversation.\
        # You play the role of a teacher. Here is the beginning of the conversation : \
        # Child : Hello !\
        # Teacher : Hi! How are your today ? \
        # Child : I am fine, and I can't wait to learn mathematics ![/INST]"

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
        print("["+str(round(time_1 - time_0, 3)) + "s] " + last_sentence)
        

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
        # model_output = generate_text_with_chat_completion(chat_history, my_model)
        model_output = generate_text_with_chat_completion_stream(chat_history, my_model)
        time_1 = time.time()

        # print(model_output)
        role = model_output["choices"][0]["message"]['role']
        text = model_output["choices"][0]["message"]['content']
        # if stream
        # role = model_output['role']
        # text = model_output['content']
        print("["+str(round(time_1 - time_0, 3)) + "s] " + role + " : " + text)
    
    elif MODEL_GENERATION_METHOD == "special" :
        time_0 = time.time()
        model_output = generate_special(my_prompt, my_model)
        time_1 = time.time()

        full_prompt = model_output["choices"][0]["text"].strip()
        print("["+str(round(time_1 - time_0, 3)) + "s] " + full_prompt)


def main():

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


if __name__ == '__main__':
    main()